// Run CPU or one native ROCm backend directly, never a scheduler fallback.
// In a private native Linux gfx1103 d82 build, add -DLLAMA_INKLING_CONCAT4_FIXTURE=ON.
//   cmake --build "$B" --target ggml-base llama ggml-hip ggml-cpu test-inkling-concat4
// CPU checks outside the GPU window use the separate CPU-only GGML_BACKEND_DL build.
// On anchor run the HIP-linked binaries ONLY through the protected locked runner,
// with llama-server and apport inactive, no unified memory, and its pinned ROCm env.
// Set L to the real staged bin dir, LD_LIBRARY_PATH="$L:$ROCM_PATH/lib"; do not mix DSOs.
//   GGML_INKLING_CONCAT4=0 "$L/test-inkling-concat4" "$L" disabled
//   for k in 0 1; do GGML_CUDA_DISABLE_GRAPHS=1 GGML_INKLING_CONCAT4=1 rocprofv3 --preload "$ROCM_PATH/lib/libamdhip64.so" --kernel-trace -f csv -d "$TRACE/$k" -- "$L/test-inkling-concat4" "$L" ROCm0 "$k" || exit 1; done
//   python3 tests/inkling-hip-concat4-coverage.py --fixture "$TRACE/0" "$TRACE/1"
// Same staged build, same tokens and batching, separate synchronized 4k model captures;
// keep graph settings and GGML_INKLING_DEPTHMIX_RAW_V2=1 fixed; toggle only CONCAT4=0/1.
// Compare both binary PP logits against the d82 oracle, and pass the untraced full model gate.
//   python3 tests/inkling-hip-concat4-coverage.py "$OFF_TRACE" "$ON_TRACE"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-inkling-concat4.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

static void check_maps(const std::string & dir, const std::string & mode, bool loaded) {
    std::ifstream maps("/proc/self/maps");
    bool base = false, backend = false;
    for (std::string line; std::getline(maps, line);) {
        auto i = line.find('/');
        if (i == line.npos) { continue; }
        auto path = line.substr(i), name = path.substr(path.rfind('/') + 1);
        if (name.rfind("libggml", 0) != 0) { continue; }
        check(path == dir + "/" + name, "foreign ggml library");
        base |= name.rfind("libggml-base.so", 0) == 0;
        bool cpu = name.rfind("libggml-cpu.so", 0) == 0, hip = name.rfind("libggml-hip.so", 0) == 0;
        if (mode == "CPU") {
            check(cpu || name.rfind("libggml.so", 0) == 0 || name.rfind("libggml-base.so", 0) == 0,
                  "use the CPU-only build; refusing accelerator registry initialization");
        }
        backend |= mode == "CPU" ? cpu : hip;
    }
    check(base && (!loaded || backend), "missing staged ggml-base/backend mapping");
}

int main(int argc, char ** argv) {
    check(argc >= 3, "usage: test-inkling-concat4 ABS_LIBDIR CPU|disabled|ROCm0 [payload]");
    const std::string mode = argv[2];
    const bool hip = mode == "ROCm0";
    check(hip ? argc == 4 && (std::string(argv[3]) == "0" || std::string(argv[3]) == "1")
              : argc == 3 && (mode == "CPU" || mode == "disabled"), "bad backend/payload");
    const int first = hip ? argv[3][0] - '0' : 0, count = hip ? 1 : 2;
    check_maps(argv[1], mode, false);
    constexpr int D = 2304, T = 256, words = D*T;
    ggml_context * c = ggml_init({4u << 20, nullptr, true});
    ggml_tensor * x[4]; // blocks[0], blocks[1], blocks[2], cur
    for (auto & v : x) { v = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T); }
    auto * pack = ggml_inkling_try_concat4(c, x[0], x[1], x[2], x[3]);
    if (mode == "disabled") {
        check(!pack && !ggml_inkling_concat4_enabled(), "disabled created a node");
        ggml_free(c);
        std::puts("PASS pack disabled");
        return 0;
    }
    // Exercise exactly the scratch budget and external source lifetimes used by graph::mix.
    auto * probe = ggml_init({5*ggml_tensor_overhead(), nullptr, true});
    auto * probe_op = ggml_inkling_try_concat4(probe, x[0], x[1], x[2], x[3]);
    check(ggml_inkling_is_concat4(probe_op), "metadata-only placement probe");
    ggml_free(probe);
    check(ggml_inkling_is_concat4(pack), "factory/identity");
    constexpr int order[] = {2, 1, 0, 3};
    for (int s = 0; s < 4; ++s) {
        check(pack->src[s]->op == GGML_OP_RESHAPE && pack->src[s]->src[0] == x[order[s]], "live ordered reshape operands");
    }
    auto attempt = [&](ggml_tensor * bad) { return ggml_inkling_try_concat4(c, x[0], bad, x[2], x[3]); };
    check(!attempt(ggml_new_tensor_2d(c, GGML_TYPE_F16, D, T)), "type");
    check(!attempt(ggml_new_tensor_1d(c, GGML_TYPE_F32, D)), "size");
    check(!attempt(ggml_transpose(c, x[0])), "noncontiguous");
#ifdef GGML_BACKEND_DL
    std::string plugin = std::string(argv[1]) + (hip ? "/libggml-hip.so" : "/libggml-cpu.so");
    check(ggml_backend_load(plugin.c_str()) != nullptr, "load selected backend");
#endif
    auto be = ggml_backend_init_by_name(argv[2], nullptr);
    check_maps(argv[1], mode, true);
    check(be && ggml_backend_supports_op(be, pack), "backend must accept genuine pack");
    auto reject = [&](ggml_tensor * node) {
        check(!ggml_inkling_is_concat4(node), "recognizer accepted invalid pack");
        check(mode != "ROCm0" || !ggml_backend_supports_op(be, node), "HIP claimed invalid custom op");
    };
    auto * foreign = ggml_custom_4d(c, GGML_TYPE_F32, D, 4, T, 1, pack->src, 4,
            [](ggml_tensor *, int, int, void *) {}, GGML_N_TASKS_MAX, nullptr);
    reject(foreign); // same shape and four sources, different CPU semantics
    pack->src[4] = x[0]; reject(pack); pack->src[4] = nullptr;
    auto * last = pack->src[3]; pack->src[3] = nullptr; reject(pack); pack->src[3] = last;
    last->nb[2] += 4; reject(pack); last->nb[2] -= 4;

    auto * g = ggml_new_graph(c);
    ggml_set_output(pack);
    ggml_build_forward_expand(g, pack);
    int custom = 0, concats = 0;
    for (int j = 0; j < ggml_graph_n_nodes(g); ++j) {
        auto op = ggml_graph_node(g, j)->op;
        custom += op == GGML_OP_CUSTOM; concats += op == GGML_OP_CONCAT;
    }
    check(custom == 1 && !concats, "candidate graph contains intermediate concat(s)");
    // Observable output with another consumer. No backend fusion may drop either store.
    auto * copy = ggml_dup(c, pack);
    ggml_build_forward_expand(g, copy);
    auto * ref = ggml_reshape_3d(c, x[3], D, 1, T);
    for (int j = 0; j < 3; ++j) {
        ref = ggml_concat(c, ggml_reshape_3d(c, x[j], D, 1, T), ref, 1);
    }
    ggml_build_forward_expand(g, ref);
    auto buf = ggml_backend_alloc_ctx_tensors(c, be); check(buf, "allocation");
    std::vector<uint32_t> row(words), expected(4*words), readback(4*words);
    constexpr uint32_t edges[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u, 1, 0x80000001u,
        0x7fc01234u, 0x7f812341u, 0xff800321u, 0xffffffffu};
    for (int k = first; k < first + count; ++k) {
        uint32_t rand = k + 19;
        for (int s = 0; s < 4; ++s) {
            for (int i = 0; i < words; ++i) {
                rand = 1664525u*rand + 1013904223u;
                row[i] = i % 17 == 0 ? edges[(i + k + s) % 10] : rand;
                expected[(4*(i/D) + s)*D + i%D] = row[i];
            }
            ggml_backend_tensor_set(x[order[s]], row.data(), 0, 4*words);
        }
        std::transform(expected.begin(), expected.end(), readback.begin(), [](uint32_t v) { return ~v; });
        for (auto * out : {pack, copy, ref}) { ggml_backend_tensor_set(out, readback.data(), 0, 4*readback.size()); }
        check(ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS, "compute");
        for (auto * out : {pack, copy, ref}) {
            ggml_backend_tensor_get(out, readback.data(), 0, 4*readback.size());
            check(expected == readback, "pack/consumer/stock must match integer word oracle");
        }
    }
    ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(c);
    std::printf("PASS %s: payloads %d..%d, %d words each, pack + consumer + stock\n", argv[2], first, first + count - 1, 4*words);
}

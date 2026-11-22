// Prefill attention fixture with the Inkling shapes: 256 queries, 24 heads on 8 KV heads, head size 128, F16 K and V.
// Usage: inkling-hip-pfattn LIB_DIR OUTPUT_FILE REPS CASE...   CASE: dense:<keys> | global:<keys> | swa:<keys>
//   dense:  the attention op with a ready mask tensor
//   global: the mask nodes of a full-attention layer (bias table, gather by distance, mask add) and the attention op
//   swa:    the same nodes with a 1024 token window
// K/V storage follows the KV cache: [row][KV head][channel] in memory.
// The output of the first and of the last run of each case goes to OUTPUT_FILE. Compare the files of two builds or switch sets.
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void need(bool x, const char * s) { if (!x) { throw std::runtime_error(s); } }

static uint32_t seed = 276411;
static float gen() { seed = 1664525u*seed + 1013904223u; return (int(seed >> 20) - 2048)*.00048828125f; } // [-1, 1)

static void run(ggml_backend_t be, FILE * out, const std::string & mode, int n, int reps) {
    constexpr int D = 128, H = 24, G = 8, NQ = 256, E = 1025, R = 16;
    const bool dense = mode == "dense", swa = mode == "swa";
    need(dense || swa || mode == "global", "unknown case");
    need(n % 256 == 0 && n >= 1536, "keys must be a multiple of 256, at least 1536");

    ggml_context * c = ggml_init({16u << 20, nullptr, true});
    auto input = [&](ggml_type ty, int64_t a, int64_t b, int64_t cc, int64_t d) { auto * t = ggml_new_tensor_4d(c, ty, a, b, cc, d); ggml_set_input(t); return t; };

    auto * qr = input(GGML_TYPE_F32, D, H, NQ, 1);
    auto * kr = input(GGML_TYPE_F16, D, G, n, 1);
    auto * vr = input(GGML_TYPE_F16, D, G, n, 1);
    auto * q = ggml_permute(c, qr, 0, 2, 1, 3);
    auto * k = ggml_view_4d(c, kr, D, n, G, 1, kr->nb[2], kr->nb[1], kr->nb[3], 0);
    auto * v = ggml_view_4d(c, vr, D, n, G, 1, vr->nb[2], vr->nb[1], vr->nb[3], 0);

    // cache rows 0..n_used-1 hold consecutive positions, the last 256 of them are the queries; the other rows are empty
    const int n_used = n - 64, p0 = n_used - NQ;
    auto visible = [&](int t, int row) { return row < n_used && row <= p0 + t && (!swa || p0 + t - row < E - 1); };
    auto distance = [&](int t, int row) { const int d = row < n_used ? p0 + t - row : -1; return d >= 0 && d < E - 1 ? d : E - 1; };

    ggml_tensor * m = nullptr, * weight = nullptr, * r = nullptr, * sw = nullptr, * sx = nullptr, * ids = nullptr, * raw = nullptr;
    if (dense) {
        m = input(GGML_TYPE_F16, n, NQ, H, 1);
    } else {
        weight = input(GGML_TYPE_F32, R, E - 1, 1, 1);
        r      = input(GGML_TYPE_F32, R, H, NQ, 1);
        sw     = input(GGML_TYPE_F32, 8, H, 1, 1);
        sx     = input(GGML_TYPE_F32, 8, NQ, 1, 1);
        ids    = input(GGML_TYPE_I32, n, NQ, 1, 1);
        raw    = input(GGML_TYPE_F16, n, NQ, 1, 1);
        ggml_set_name(ids, swa ? "attn_inp_rel_pos_swa" : "attn_inp_rel_pos");
        // same nodes as in llama_model_inkling2::graph::build_layer
        auto * table = ggml_mul_mat(c, ggml_pad(c, weight, 0, 1, 0, 0), r);
        table = ggml_sub(c, table, ggml_reshape_3d(c, ggml_mul_mat(c, sw, sx), 1, H, NQ));
        table = ggml_cont(c, ggml_permute(c, table, 1, 0, 2, 3));
        auto * kq_b = ggml_get_rows(c, table, ids);
        kq_b = ggml_reshape_4d(c, kq_b, H, n, NQ, 1);
        kq_b = ggml_cont(c, ggml_permute(c, kq_b, 2, 0, 1, 3));
        ggml_set_name(kq_b, "kq_b-5");
        m = ggml_cast(c, ggml_add(c, kq_b, ggml_cast(c, raw, GGML_TYPE_F32)), GGML_TYPE_F16);
    }
    ggml_set_name(m, "kq_mask_rel_pos-5");
    auto * si = ggml_fill(c, ggml_new_tensor_1d(c, GGML_TYPE_F32, H), 0.0f);
    auto * y = ggml_flash_attn_ext(c, q, k, v, m, 1.0f/std::sqrt(float(D)), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(y, si);
    ggml_set_output(y);
    auto * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, y);
    need(ggml_backend_supports_op(be, y), "unsupported");
    auto * buf = ggml_backend_alloc_ctx_tensors(c, be);
    need(buf, "device allocation");

    {
        std::vector<ggml_fp16_t> x(size_t(D)*G*n);
        for (auto * t : {kr, vr}) {
            for (auto & z : x) { z = ggml_fp32_to_fp16(gen()); }
            ggml_backend_tensor_set(t, x.data(), 0, x.size()*2);
        }
        std::vector<float> Q(size_t(D)*H*NQ);
        for (auto & z : Q) { z = 2.0f*gen(); }
        ggml_backend_tensor_set(qr, Q.data(), 0, Q.size()*4);
    }
    if (dense) {
        std::vector<ggml_fp16_t> M(size_t(n)*NQ*H);
        std::vector<float> far(size_t(H)*NQ), near(size_t(H)*E);
        for (auto & z : far)  { z = gen(); }
        for (auto & z : near) { z = 2.0f*gen(); }
        for (int h = 0; h < H; ++h) for (int t = 0; t < NQ; ++t) for (int row = 0; row < n; ++row) {
            const int d = distance(t, row);
            M[row + size_t(n)*(t + NQ*h)] = !visible(t, row) ? 0xfc00 : ggml_fp32_to_fp16(d == E - 1 ? far[h*NQ + t] : near[h*E + d] + far[h*NQ + t]);
        }
        ggml_backend_tensor_set(m, M.data(), 0, M.size()*2);
    } else {
        std::vector<float> a;
        for (auto * t : {weight, r, sw, sx}) {
            a.resize(ggml_nelements(t));
            for (auto & z : a) { z = gen(); }
            ggml_backend_tensor_set(t, a.data(), 0, a.size()*4);
        }
        std::vector<int32_t> ix(size_t(n)*NQ);
        std::vector<ggml_fp16_t> M(size_t(n)*NQ);
        for (int t = 0; t < NQ; ++t) for (int row = 0; row < n; ++row) {
            ix[row + size_t(n)*t] = distance(t, row);
            M [row + size_t(n)*t] = visible(t, row) ? 0 : 0xfc00;
        }
        ggml_backend_tensor_set(ids, ix.data(), 0, ix.size()*4);
        ggml_backend_tensor_set(raw, M.data(), 0, M.size()*2);
    }

    std::vector<float> Y(ggml_nelements(y)), Y0;
    auto step = [&]() { need(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute"); ggml_backend_synchronize(be); };
    auto save = [&]() {
        ggml_backend_tensor_get(y, Y.data(), 0, Y.size()*4);
        need(std::all_of(Y.begin(), Y.end(), [](float f) { return std::isfinite(f); }), "nonfinite output");
        need(std::fwrite(Y.data(), 4, Y.size(), out) == Y.size(), "output save");
    };
    step(); save(); Y0 = Y;
    step();
    std::vector<double> t(reps);
    for (int j = 0; j < reps; ++j) {
        const auto b = std::chrono::steady_clock::now();
        step();
        t[j] = std::chrono::duration<double>(std::chrono::steady_clock::now() - b).count();
    }
    // the last half of the runs shows the speed at the power limit of the machine
    std::vector<double> late(t.begin() + reps/2, t.end());
    std::sort(late.begin(), late.end());
    std::sort(t.begin(), t.end());
    save();
    need(Y == Y0, "the last run differs from the first run");
    uint64_t hash = 1469598103934665603ull;
    for (float f : Y) { uint32_t u; std::memcpy(&u, &f, 4); hash = (hash ^ u)*1099511628211ull; }
    std::printf("%s keys=%d reps=%d ms_median=%.3f ms_best=%.3f ms_late=%.3f fnv=%016llx\n", mode.c_str(), n, reps, t[reps/2]*1000, t[0]*1000, late[late.size()/2]*1000, (unsigned long long) hash);
    std::fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_free(c);
}

int main(int argc, char ** argv) {
    try {
        need(argc >= 5, "LIB_DIR OUTPUT_FILE REPS CASE...");
        ggml_backend_load_all_from_path(argv[1]);
        auto * b = ggml_backend_init_by_name("ROCm0", nullptr);
        need(b, "no ROCm0");
        std::ifstream maps("/proc/self/maps");
        std::string l;
        bool hip = false;
        while (std::getline(maps, l)) {
            if (l.find("/libggml-hip.so") != l.npos) {
                need(l.find(std::string(argv[1]) + "/libggml-hip.so") != l.npos, "foreign mapped HIP");
                hip = true;
            }
        }
        need(hip, "missing HIP");
        auto * out = std::fopen(argv[2], "wbx");
        need(out, "output exists");
        const int reps = std::atoi(argv[3]);
        for (int i = 4; i < argc; ++i) {
            const std::string s = argv[i];
            const auto p = s.find(':');
            need(p != s.npos, "CASE is mode:keys");
            run(b, out, s.substr(0, p), std::atoi(s.c_str() + p + 1), reps);
        }
        need(!std::fclose(out), "close");
        ggml_backend_free(b);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "pfattn: %s\n", e.what());
        return 1;
    }
}

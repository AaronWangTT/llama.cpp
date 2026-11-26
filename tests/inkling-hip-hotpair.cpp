// C++17/Linux ROCm; link -lggml -lggml-base -ldl. No private ggml headers.
// fixture LIBDIR GGUF gate|down ROUTE_FILE OUT   (OUT must not exist)
// Use separate processes with the selected build's LD_LIBRARY_PATH and GGML_KPERF_OFF arm (inkling.hotpair_gate / hotpair_down).
// HOTPAIR_PASS=check (default): raw F32 [case][token][slot][output row], no header.
//   Run independently against the pinned d82+layout parent, ORIGINAL, and candidate.
//   Compare with check.py. No reference values are read by this program.
// HOTPAIR_PASS=time: TSV laps of synchronized single-op graphs, not model throughput.
// HOTPAIR_PASS=profile: one ROCTX range per case after warmup; check device trace for
//   inkling_hotpair_j32 vs the original MMQ kernel. A marker alone proves no dispatch.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <dlfcn.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr int T = 256, S = 8, E = 128, N = 2304, J = 32, D = 4;
constexpr const char *names[D] = {"route", "no_pair", "single_hot", "offset_tail127"};
using Row = std::array<int32_t, S>;
using Routes = std::array<Row, T>;
static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);

static void need(bool ok, const std::string &why) {
    if (!ok) throw std::runtime_error(why);
}

static Routes read_routes(const char *file) {
    std::ifstream in(file);
    need(in.is_open(), "cannot open route file");
    Routes rows;
    std::string word;
    for (auto &row : rows) for (auto &id : row) {
        need(bool(in >> word), "route file has fewer than 2048 IDs");
        auto end = word.data() + word.size();
        auto rc = std::from_chars(word.data(), end, id);
        need(rc.ec == std::errc{} && rc.ptr == end, "invalid route ID: " + word);
    }
    need(!(in >> word) && in.eof() && !in.bad(), "trailing route data or read error");
    return rows;
}

static Routes synthetic(int d) {
    Routes rows;
    for (int t = 0; t < T; ++t) {
        Row r{};
        if (d == 1) {
            for (int s = 0; s < S; ++s) r[s] = (5*t + 17*s) % E; // 16/expert
        } else if (d == 2) {
            r[0] = 0;
            for (int s = 1; s < S; ++s) r[s] = 1 + (7*t+s-1)%127;
        } else {
            int n = 0;
            if (t < 3)   r[n++] = 0;
            if (t < 131) r[n++] = 1;    // compact [3,134)
            if (t < 97)  r[n++] = 127;  // compact [1951,2048), final singleton
            for (int j = 0; n < S; ++j) r[n++] = 2 + (8*t+j)%125;
        }
        std::rotate(r.begin(), r.begin() + t%S, r.end());
        rows[t] = r;
    }
    return rows;
}

static std::vector<int32_t> pack(const Routes &rows, int d) {
    std::vector<int32_t> strided(T*E, -1);
    std::array<int,E> count{};
    for (int t = 0; t < T; ++t) {
        std::array<bool,E> used{};
        for (int s = 0; s < S; ++s) {
            int id = rows[t][s];
            need(id >= 0 && id < E && !used[id], "invalid top-8 at token " + std::to_string(t));
            used[id] = true;
            ++count[id];
            strided[t*E+s] = id;
        }
    }
    int base = 0, pairs = 0, unaligned = 0, hot = 0;
    for (int n : count) {
        int p = n/(2*J);
        pairs += p;
        if (base%J) unaligned += p;
        hot += p != 0;
        base += n;
    }
    if (d == 1) need(pairs == 0, "no_pair construction");
    if (d == 2) need(hot == 1 && pairs == 4, "single_hot construction");
    if (d == 3) need(count[0] == 3 && count[1] == 131 && count[127] == 97 &&
                     hot == 2 && pairs == 3 && unaligned == 3, "offset/tail construction");
    std::fprintf(stderr, "case=%s hot_experts=%d disjoint_full_pairs=%d unaligned_pairs=%d expert127_cols=%d\n",
                 names[d], hot, pairs, unaligned, count[127]);
    return strided;
}

static void provenance(const fs::path &lib) {
    std::ifstream maps("/proc/self/maps");
    need(maps.is_open(), "cannot inspect mapped ggml libraries");
    std::set<fs::path> paths;
    bool hip = false, base = false, core = false;
    for (std::string line; std::getline(maps,line);) {
        auto i = line.find('/');
        if (i == line.npos) continue;
        fs::path p(line.substr(i));
        auto name = p.filename().string();
        if (name.rfind("libggml", 0) != 0) continue;
        need(p.parent_path() == lib && fs::canonical(p) == p, "foreign/stale ggml mapping: " + p.string());
        auto so = [&](const std::string &s) { return name == s || name.rfind(s+".", 0) == 0; };
        hip |= so("libggml-hip.so"); base |= so("libggml-base.so"); core |= so("libggml.so");
        paths.insert(p);
    }
    need(maps.eof() && !maps.bad() && hip && core && base, "missing HIP/core/base mapping");
    for (const auto &p : paths) std::fprintf(stderr, "mapped %s\n", p.c_str());
}

struct Markers { // Load ROCTX only in the separate profiler process.
    void *handle = dlopen("librocprofiler-sdk-roctx.so", RTLD_NOW | RTLD_LOCAL);
    int (*push)(const char *);
    int (*pop)();
    Markers() {
        need(handle, "cannot load librocprofiler-sdk-roctx.so");
        push = reinterpret_cast<decltype(push)>(dlsym(handle,"roctxRangePushA"));
        pop = reinterpret_cast<decltype(pop)>(dlsym(handle,"roctxRangePop"));
        need(push && pop, "missing ROCTX range API");
    }
    ~Markers() { dlclose(handle); }
};

static void load_packed(ggml_tensor *w, const char *file, const gguf_context *meta, int64_t wi) {
    const size_t size = ggml_nbytes(w);
    need(size == gguf_get_tensor_size(meta,wi), "wrong packed byte count");
    std::ifstream f(file, std::ios::binary);
    f.seekg(gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta,wi));
    std::vector<char> chunk(1u<<20);
    for (size_t off = 0; off < size;) {
        size_t n = std::min(chunk.size(), size-off);
        need(bool(f.read(chunk.data(), n)), "short GGUF tensor read");
        ggml_backend_tensor_set(w, chunk.data(), off, n);
        off += n;
    }
}

static int run(int argc, char **argv) {
    need(argc == 6, "LIBDIR GGUF gate|down ROUTE_FILE OUT");
    std::string mode = argv[3];
    need(mode == "gate" || mode == "down", "expected gate or down");
    const char *env = std::getenv("HOTPAIR_PASS");
    std::string pass = env ? env : "check";
    need(pass == "check" || pass == "time" || pass == "profile", "bad HOTPAIR_PASS");
    std::array<std::vector<int32_t>,D> ids;
    for (int d = 0; d < D; ++d) ids[d] = pack(d == 0 ? read_routes(argv[4]) : synthetic(d), d);
    bool down = mode == "down";
    int k = down ? 1152 : 2304, channels = down ? S : 1;
    const char *weight = down ? "blk.6.ffn_down_exps.weight" : "blk.6.ffn_gate_up_exps.weight";
    auto *meta = gguf_init_from_file(argv[2], {true,nullptr});
    need(meta, "cannot open GGUF");
    auto wi = gguf_find_tensor(meta,weight);
    need(wi >= 0, "missing weight");
    auto *ne = gguf_get_tensor_ne(meta,wi);
    need(gguf_get_tensor_type(meta,wi) == GGML_TYPE_IQ4_NL && ne[0] == k && ne[1] == N &&
         ne[2] == E && ne[3] == 1, "wrong GGUF tensor shape/type");

    auto lib = fs::canonical(argv[1]);
    ggml_backend_load_all_from_path(lib.c_str());
    auto *be = ggml_backend_init_by_name("ROCm0",nullptr);
    need(be, "no ROCm0");
    provenance(lib);
    auto *ctx = ggml_init({2u<<20,nullptr,true});
    need(ctx, "cannot allocate graph context");
    auto *w = ggml_new_tensor_3d(ctx,GGML_TYPE_IQ4_NL,k,N,E);
    ggml_set_name(w,weight);
    auto *x = ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,channels,T);
    auto *raw = ggml_new_tensor_2d(ctx,GGML_TYPE_I32,E,T);
    auto *view = ggml_view_2d(ctx,raw,S,T,raw->nb[1],0);
    auto *y = ggml_mul_mat_id(ctx,w,x,view);
    ggml_set_name(y,down ? "ffn_moe_down-24" : "ffn_moe_gate_up-24");
    need(y->type == GGML_TYPE_F32 && ggml_is_contiguous(y) && y->ne[0] == N && y->ne[1] == S &&
         y->ne[2] == T && y->ne[3] == 1 && ggml_backend_supports_op(be,y), "unsupported output");
    auto *g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g,y);
    auto *buf = ggml_backend_alloc_ctx_tensors(ctx,be);
    need(buf, "device allocation failed");
    load_packed(w,argv[2],meta,wi);
    gguf_free(meta);

    // Hold activation bytes fixed across routes and builds. Every token (and for down,
    // every slot) has its own nonconstant input, so scatter errors aren't hidden by a broadcast.
    std::vector<float> X(ggml_nelements(x));
    uint32_t seed = 451;
    for (auto &v : X) {
        seed = seed*1664525u + 1013904223u;
        v = (int(seed>>16)-32768)/8192.f;
    }
    for (int row = 0; row < T*channels; ++row) X[size_t(row)*k] = (row+1)/512.f;
    ggml_backend_tensor_set(x,X.data(),0,ggml_nbytes(x));
    auto select = [&](int d) {
        ggml_backend_tensor_set(raw,ids[d].data(),0,ggml_nbytes(raw));
        ggml_backend_synchronize(be);
    };
    auto step = [&] {
        if (ggml_backend_graph_compute(be,g) != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed");
        ggml_backend_synchronize(be);
    };
    FILE *out = std::fopen(argv[5],"wbx");
    need(out, "OUT exists or cannot be created");
    for (int d = 0; d < D; ++d) { select(d); step(); step(); }
    provenance(lib); // also catch dependencies first loaded by compute

    if (pass == "check") {
        std::vector<uint32_t> words(ggml_nelements(y));
        const uint32_t salt = uint32_t(getpid()) * 2654435761u; // distinct poison across processes
        for (int d = 0; d < D; ++d) {
            select(d);
            // Poison every word before this one graph (not before a loop of graphs).
            for (size_t i = 0; i < words.size(); ++i)
                words[i] = 0x7fc00000u | ((salt + uint32_t(i) + 100003u*d) & 0x003fffffu);
            ggml_backend_tensor_set(y,words.data(),0,ggml_nbytes(y));
            step();
            ggml_backend_tensor_get(y,words.data(),0,ggml_nbytes(y));
            auto bad = std::find_if(words.begin(),words.end(),[](uint32_t u) {
                return (u & 0x7f800000u) == 0x7f800000u;
            });
            need(bad == words.end(), std::string(names[d])+": unwritten/nonfinite F32 at word "+std::to_string(bad-words.begin()));
            need(std::fwrite(words.data(),sizeof(uint32_t),words.size(),out) == words.size(), "short F32 write");
        }
    } else if (pass == "time") {
        std::fprintf(out,"mode\tround\tposition\tcase\treps\tgraph_ms_per_rep\n");
        constexpr int reps = 4;
        // Two complete position rotations. Upload, reporting and marker API are untimed.
        for (int round = 0; round < 2*D; ++round) for (int pos = 0; pos < D; ++pos) {
            int d = (round+pos)%D;
            select(d);
            auto a = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) step();
            double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count();
            std::fprintf(out,"%s\t%d\t%d\t%s\t%d\t%.9f\n",mode.c_str(),round,pos,names[d],reps,ms/reps);
        }
    } else {
        Markers m;
        for (int d = 0; d < D; ++d) {
            select(d);
            std::string tag = "hotpair_fixture/" + mode + "/" + names[d];
            m.push(tag.c_str()); step(); m.pop();
            std::fprintf(out,"%s\n",tag.c_str());
        }
    }
    bool wrote = !std::ferror(out);
    need(std::fclose(out) == 0 && wrote, "output write/close failed");
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    return 0;
}

int main(int argc, char **argv) {
    try { return run(argc,argv); }
    catch (const std::exception &e) { std::fprintf(stderr,"hotpair fixture: %s\n",e.what()); return 1; }
}

// Fixture for GGML_Q35MOE_PFMMN: the F32 products of qwen35moe (router with 256 rows, beta and alpha with 32 rows, k = 2048) for
// every token count from 1 to 2048. Each product is computed twice in one graph: on a float4 aligned x (the kernel, when the switch
// is set and the batch has 128 tokens or more) and on an x that starts one float into its buffer (always the stock path).
//   q35moe-hip-pfmmn ABS_LIBDIR [rep1]                                            stock path only, prints the hashes (rep1: only the second data set)
//   GGML_Q35MOE_PFMMN=1 GGML_Q35MOE_PFMMN_TRACE=1 q35moe-hip-pfmmn ABS_LIBDIR      kernel against SGEMM, counts the kernel calls
//   [GGML_Q35MOE_PFMMN=1] q35moe-hip-pfmmn ABS_LIBDIR bench                       time of the 99 products of a prompt ubatch, aligned and shifted x in turn
// Checks, from 9 tokens on (up to 8 the stock path is the vector kernel): aligned = shifted for every value; a token column of
// SGEMM has the same bits whatever the token count (against the product with 2048 tokens); the product with 2048 tokens against a
// fused multiply-add chain on the CPU in the order of the rocBLAS kernel. The switch must take the aligned products from T_MIN
// tokens on and no other: below it SGEMM is faster. The bench with the switch on and off shows from which token count the kernel
// wins; run against a library without the lower bound (commit 5f3b5a7) it times the kernel for the small batches too.
// The hash_aligned of a data set must be the same with and without the switch.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

constexpr int K     = 2048;
constexpr int T_MAX = 2048;
constexpr int T_MIN = 128; // q35moe_pfmmn_min_tokens
constexpr int N_W   = 3;

static const char * const w_name[N_W] = { "blk.7.ffn_gate_inp.weight", "blk.7.ssm_beta.weight", "blk.7.ssm_alpha.weight" };
static const int          w_rows[N_W] = { 256, 32, 32 };

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::printf("FAIL: %s\n", what); std::exit(1); }
}

static long n_hit = 0;

static void log_cb(enum ggml_log_level, const char * text, void *) {
    if (std::strstr(text, "Q35MOE_PFMMN_HIT")) {
        n_hit++;
    } else {
        std::fputs(text, stderr);
    }
}

static uint64_t fnv(uint64_t h, const std::vector<float> & v) {
    const unsigned char * p = (const unsigned char *) v.data();
    for (size_t i = 0; i < v.size()*sizeof(float); ++i) { h = (h ^ p[i]) * 1099511628211ull; }
    return h;
}

// rocBLAS order: one chain per value, k rises from 64*(row tile & 31) and wraps
static std::vector<float> cpu_ref(const std::vector<float> & w, const std::vector<float> & x, int rows, int n) {
    std::vector<float> r((size_t) rows*n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < rows; ++i) {
            const int k0 = ((i/32) & 31)*64;
            float acc = 0.0f;
            for (int q = 0; q < K; ++q) {
                const int k = (k0 + q) % K;
                acc = std::fma(w[(size_t) i*K + k], x[(size_t) j*K + k], acc);
            }
            r[(size_t) j*rows + i] = acc;
        }
    }
    return r;
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the F32 products of one prompt ubatch (39 router products, 30 beta, 30 alpha, each with its own weight) as one graph;
// the graph on the aligned x and the graph on the shifted x run in turn
static void bench(ggml_backend_t hip, bool on) {
    constexpr int n_router = 39, n_rec = 30, n_w = n_router + 2*n_rec;
    ggml_context * cs = ggml_init({(n_w + 8)*ggml_tensor_overhead(), nullptr, true});
    std::vector<ggml_tensor *> w;
    for (int i = 0; i < n_w; ++i) {
        const bool router = i < n_router;
        w.push_back(ggml_new_tensor_2d(cs, GGML_TYPE_F32, K, router ? 256 : 32));
        const std::string name = "blk." + std::to_string(i) + (router ? ".ffn_gate_inp.weight" : i < n_router + n_rec ? ".ssm_beta.weight" : ".ssm_alpha.weight");
        ggml_set_name(w.back(), name.c_str());
    }
    ggml_tensor * pool_a = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) K*T_MAX);
    ggml_tensor * pool_s = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) K*T_MAX + 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(cs, hip);
    check(buf, "weight buffer");
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> d((size_t) K*T_MAX);
    for (auto & v : d) { v = nd(rng); }
    ggml_backend_tensor_set(pool_a, d.data(), 0, d.size()*sizeof(float));
    ggml_backend_tensor_set(pool_s, d.data(), sizeof(float), d.size()*sizeof(float));
    for (ggml_tensor * t : w) {
        ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
    }
    for (int n : { 9, 16, 32, 64, 96, 112, 127, 128, 129, 160, 188, 256, 316, 508, 512, 828, 1020, 1024, 2048 }) {
        ggml_context * c = ggml_init({(4*n_w + 8)*ggml_tensor_overhead() + 2*ggml_graph_overhead(), nullptr, true});
        ggml_cgraph * g[2];
        ggml_gallocr_t ga[2];
        for (int v = 0; v < 2; ++v) {
            ggml_tensor * x = v == 0 ? ggml_view_2d(c, pool_a, K, n, K*sizeof(float), 0) : ggml_view_2d(c, pool_s, K, n, K*sizeof(float), sizeof(float));
            g[v] = ggml_new_graph(c);
            for (ggml_tensor * t : w) {
                ggml_tensor * out = ggml_mul_mat(c, t, x);
                ggml_set_output(out);
                ggml_build_forward_expand(g[v], out);
            }
            ga[v] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(hip));
            check(ggml_gallocr_alloc_graph(ga[v], g[v]), "alloc");
        }
        std::vector<double> ms[2];
        for (int it = 0; it < 14; ++it) {
            for (int q = 0; q < 2; ++q) {
                const int v = (it + q) % 2; // a s s a a s ...
                const double t0 = now();
                check(ggml_backend_graph_compute(hip, g[v]) == GGML_STATUS_SUCCESS, "compute");
                ggml_backend_synchronize(hip);
                if (it >= 2) { ms[v].push_back(1e3*(now() - t0)); }
            }
        }
        for (int v = 0; v < 2; ++v) { std::sort(ms[v].begin(), ms[v].end()); }
        std::printf("bench tokens %4d: %d products, aligned x (switch %s) median %.2f ms (min %.2f, max %.2f), shifted x (SGEMM) median %.2f ms (min %.2f, max %.2f), %zu runs each\n",
            n, n_w, on ? "on" : "off", ms[0][ms[0].size()/2], ms[0].front(), ms[0].back(), ms[1][ms[1].size()/2], ms[1].front(), ms[1].back(), ms[0].size());
        ggml_gallocr_free(ga[0]);
        ggml_gallocr_free(ga[1]);
        ggml_free(c);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(cs);
}

int main(int argc, char ** argv) {
    const bool do_bench = argc == 3 && std::strcmp(argv[2], "bench") == 0;
    const bool rep1     = argc == 3 && std::strcmp(argv[2], "rep1") == 0;
    check(argc == 2 || do_bench || rep1, "usage: q35moe-hip-pfmmn ABS_LIBDIR [bench|rep1]");
    const char * on_env = std::getenv("GGML_Q35MOE_PFMMN");
    const bool   on     = on_env && std::strcmp(on_env, "1") == 0;
    const char * tr_env = std::getenv("GGML_Q35MOE_PFMMN_TRACE");
    check(do_bench || !on || (tr_env && std::strcmp(tr_env, "1") == 0), "set GGML_Q35MOE_PFMMN_TRACE=1 with the switch");
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t hip = ggml_backend_init_by_name("ROCm0", nullptr);
    check(hip, "backend");
    if (do_bench) {
        bench(hip, on);
        ggml_backend_free(hip);
        return 0;
    }

    // weights and the token pool live in one buffer, like model weights; the graphs take views of the pool
    ggml_context * cs = ggml_init({16*ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * w[N_W];
    for (int i = 0; i < N_W; ++i) {
        w[i] = ggml_new_tensor_2d(cs, GGML_TYPE_F32, K, w_rows[i]);
        ggml_set_name(w[i], w_name[i]);
    }
    ggml_tensor * pool_a = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) K*T_MAX);
    ggml_tensor * pool_s = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) K*T_MAX + 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(cs, hip);
    check(buf, "weight buffer");
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    long n_val = 0, d_as = 0, d_col = 0, d_cpu = 0, n_kernel = 0, n_graphs = 0, n_lines = 0;
    for (int rep = rep1 ? 1 : 0; rep < 2; ++rep) {
        uint64_t h_a = 1469598103934665603ull, h_s = h_a;
        // rep 0: every token count; rep 1: other magnitudes (each token has its own scale), a part of the counts
        std::mt19937 rng(20261124 + rep);
        std::vector<float> wd[N_W], xd((size_t) K*T_MAX);
        for (int i = 0; i < N_W; ++i) {
            wd[i].resize((size_t) K*w_rows[i]);
            for (auto & v : wd[i]) { v = 0.03f*nd(rng); }
            ggml_backend_tensor_set(w[i], wd[i].data(), 0, wd[i].size()*sizeof(float));
        }
        for (int j = 0; j < T_MAX; ++j) {
            const float amp = rep == 0 ? 1.0f : std::exp(3.0f*nd(rng));
            for (int k = 0; k < K; ++k) { xd[(size_t) j*K + k] = amp*nd(rng); }
        }
        ggml_backend_tensor_set(pool_a, xd.data(), 0, xd.size()*sizeof(float));
        ggml_backend_tensor_set(pool_s, xd.data(), sizeof(float), xd.size()*sizeof(float));

        std::vector<float> full[N_W]; // SGEMM with 2048 tokens
        for (int n = T_MAX; n >= 1; --n) {
            if (rep == 1 && n != T_MAX && n % 37 != 0 && n % 128 > 2 && n % 128 < 126 && n > 40) {
                continue;
            }
            ggml_context * c = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
            ggml_tensor * x_a = ggml_view_2d(c, pool_a, K, n, K*sizeof(float), 0);
            ggml_tensor * x_s = ggml_view_2d(c, pool_s, K, n, K*sizeof(float), sizeof(float));
            ggml_cgraph * gf = ggml_new_graph(c);
            // up to 8 tokens the stock path is the vector kernel: only the aligned product, the switch must leave it alone
            const bool sgemm = n > 8;
            ggml_tensor * out_a[N_W], * out_s[N_W];
            for (int i = 0; i < N_W; ++i) {
                out_a[i] = ggml_mul_mat(c, w[i], x_a); ggml_set_output(out_a[i]); ggml_build_forward_expand(gf, out_a[i]);
                if (sgemm) {
                    out_s[i] = ggml_mul_mat(c, w[i], x_s); ggml_set_output(out_s[i]); ggml_build_forward_expand(gf, out_s[i]);
                }
            }
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(hip));
            check(ggml_gallocr_alloc_graph(ga, gf), "alloc");
            const long hit0 = n_hit;
            check(ggml_backend_graph_compute(hip, gf) == GGML_STATUS_SUCCESS, "compute");
            n_graphs++;
            // the kernel takes the aligned products of T_MIN tokens and more
            const long want = on && n >= T_MIN ? N_W : 0;
            if (n_hit - hit0 != want) {
                std::printf("tokens %d: %ld kernel calls, expected %ld\n", n, n_hit - hit0, want);
                check(false, "kernel call count");
            }
            n_kernel += n_hit - hit0;
            for (int i = 0; i < N_W; ++i) {
                std::vector<float> ra((size_t) w_rows[i]*n), rs(ra.size());
                ggml_backend_tensor_get(out_a[i], ra.data(), 0, ra.size()*sizeof(float));
                h_a = fnv(h_a, ra);
                if (!sgemm) {
                    n_val += ra.size();
                    continue;
                }
                ggml_backend_tensor_get(out_s[i], rs.data(), 0, rs.size()*sizeof(float));
                h_s = fnv(h_s, rs);
                if (n == T_MAX) {
                    full[i] = rs;
                    const auto ref = cpu_ref(wd[i], xd, w_rows[i], n);
                    long d = 0;
                    for (size_t q = 0; q < rs.size(); ++q) {
                        check(std::isfinite(rs[q]), "finite");
                        d += std::memcmp(&rs[q], &ref[q], sizeof(float)) != 0;
                    }
                    std::printf("rep %d %s: SGEMM with %d tokens against the CPU chain: %ld of %zu values differ\n", rep, w_name[i], n, d, rs.size());
                    d_cpu += d;
                }
                long da = 0, dc = 0;
                for (size_t q = 0; q < rs.size(); ++q) {
                    da += std::memcmp(&ra[q], &rs[q], sizeof(float)) != 0;
                    // a column of SGEMM must not depend on the token count
                    dc += std::memcmp(&rs[q], &full[i][q], sizeof(float)) != 0;
                }
                if ((da != 0 || dc != 0) && n_lines++ < 60) {
                    std::printf("rep %d %s tokens %d: of %zu values differ: aligned against shifted %ld, SGEMM against SGEMM with %d tokens %ld\n",
                        rep, w_name[i], n, rs.size(), da, T_MAX, dc);
                }
                n_val += rs.size(); d_as += da; d_col += dc;
            }
            ggml_gallocr_free(ga);
            ggml_free(c);
        }
        std::printf("rep %d hash_aligned %016llx hash_shifted %016llx\n", rep, (unsigned long long) h_a, (unsigned long long) h_s);
    }
    std::printf("switch %s: %ld graphs, %ld values, %ld kernel calls\n", on ? "on" : "off", n_graphs, n_val, n_kernel);
    std::printf("differences: aligned against shifted %ld, SGEMM column against the 2048 token product %ld, SGEMM against the CPU chain %ld\n", d_as, d_col, d_cpu);
    check(d_as == 0 && d_col == 0 && d_cpu == 0, "not bit-exact");
    check(!on || n_kernel > 0, "the kernel was never used");
    std::printf("PASS bit-exact\n");
    ggml_backend_buffer_free(buf);
    ggml_free(cs);
    ggml_backend_free(hip);
    return 0;
}

// Fixture for the single-token depth mix op. Compares the stock chain on ROCm0, the fused op on ROCm0 and the CPU reference.
//   inkling-hip-mix1 ABS_LIBDIR    the fused op must equal the stock chain bit for bit
// It also reports the sum order of the last product of the stock chain.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-inkling-mix1.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

constexpr int D = 2304;

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

struct result {
    bool               ran = false;
    std::vector<float> out;    // [D, nq]
    std::vector<float> normed; // [D, ns], stock chain only
    std::vector<float> probs;  // [ns, nq], stock chain only
};

// probe keeps the normed states and the softmax of the stock chain
static result run(ggml_backend_t be, int ns, int nq, bool fused, bool probe, const std::vector<std::vector<float>> & data, const std::vector<float> & qdata) {
    ggml_context * c = ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    std::vector<ggml_tensor *> rows(ns);
    for (auto & v : rows) { v = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 1); ggml_set_input(v); }
    ggml_tensor * query = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, nq); ggml_set_input(query);
    ggml_tensor * out;
    ggml_tensor * states = nullptr;
    ggml_tensor * scores = nullptr;
    result res;
    if (fused) {
        out = ggml_inkling_try_mix1(c, rows.data(), ns, query);
        check(out && ggml_inkling_is_mix1(out), "factory/identity");
        if (!ggml_backend_supports_op(be, out)) {
            ggml_free(c);
            return res;
        }
    } else {
        // same as graph::mix: the last row is cur, the blocks are prepended
        states = ggml_reshape_3d(c, rows[ns - 1], D, 1, 1);
        for (int s = ns - 2; s >= 0; --s) {
            states = ggml_concat(c, ggml_reshape_3d(c, rows[s], D, 1, 1), states, 1);
        }
        states = ggml_rms_norm(c, states, 1e-6f);
        ggml_tensor * queries = ggml_repeat(c, query, ggml_new_tensor_3d(c, GGML_TYPE_F32, D, nq, 1));
        scores = ggml_mul_mat(c, states, queries);
        scores = ggml_soft_max_ext(c, scores, nullptr, 1.0f/sqrtf(D), 0.0f);
        out = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, states)), scores);
        if (probe) {
            ggml_set_output(states);
            ggml_set_output(scores);
        }
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    check(ggml_gallocr_alloc_graph(ga, gf), "alloc");
    for (int s = 0; s < ns; ++s) { ggml_backend_tensor_set(rows[s], data[s].data(), 0, D*sizeof(float)); }
    ggml_backend_tensor_set(query, qdata.data(), 0, (size_t) D*nq*sizeof(float));
    check(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute");
    res.ran = true;
    res.out.resize((size_t) D*nq);
    ggml_backend_tensor_get(out, res.out.data(), 0, res.out.size()*sizeof(float));
    if (probe && !fused) {
        res.normed.resize((size_t) D*ns);
        res.probs.resize((size_t) ns*nq);
        ggml_backend_tensor_get(states, res.normed.data(), 0, res.normed.size()*sizeof(float));
        ggml_backend_tensor_get(scores, res.probs.data(),  0, res.probs.size()*sizeof(float));
    }
    ggml_gallocr_free(ga);
    ggml_free(c);
    return res;
}

static size_t count_diff(const std::vector<float> & a, const std::vector<float> & b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    return n;
}

// lane 0 of a warp sum when only the first n lanes are not zero
static float lanes(const float * v, int n) {
    float a[8];
    for (int i = 0; i < 8; ++i) { a[i] = (i < n ? v[i] : 0.0f) + 0.0f; }
    return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
}

// candidate sum orders of out[i, j] = sum over s of normed[s][i]*probs[s][j]
enum { ORDER_PAIRS, ORDER_LANES, ORDER_CHAIN, ORDER_PLAIN, ORDER_COUNT };
static const char * order_name[ORDER_COUNT] = { "fma pairs + warp sum", "products + warp sum", "fma chain", "plain sum" };

static float product(int order, const float * x, const float * p, int ns) {
    float t[8] = {};
    switch (order) {
        case ORDER_PAIRS:
            for (int m = 0; m < (ns + 1)/2; ++m) {
                t[m] = std::fma(x[2*m], p[2*m], 0.0f);
                if (2*m + 1 < ns) { t[m] = std::fma(x[2*m + 1], p[2*m + 1], t[m]); }
            }
            return lanes(t, (ns + 1)/2);
        case ORDER_LANES:
            for (int s = 0; s < ns; ++s) { t[s] = std::fma(x[s], p[s], 0.0f); }
            return lanes(t, ns);
        case ORDER_CHAIN:
            for (int s = 0; s < ns; ++s) { t[0] = std::fma(x[s], p[s], t[0]); }
            return t[0];
        default:
            for (int s = 0; s < ns; ++s) { t[0] += x[s]*p[s]; }
            return t[0];
    }
}

int main(int argc, char ** argv) {
    check(argc == 2, "usage: inkling-hip-mix1 ABS_LIBDIR");
    check(ggml_inkling_mix1_enabled(), "inkling.mix1 is off (GGML_KPERF_OFF)");
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t hip = ggml_backend_init_by_name("ROCm0", nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    check(hip && cpu, "backends");
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    constexpr int n_rep = 6;
    double worst      = 0.0;
    size_t total_diff = 0;
    for (int ns = 2; ns <= ggml_inkling_mix1_max_states; ++ns) {
        for (int nq = 1; nq <= ggml_inkling_mix1_max_query; ++nq) {
            size_t n_diff = 0, n_probe_diff = 0, n_order[ORDER_COUNT] = {};
            int    n_fused = 0;
            double rel_fused = 0.0, rel_host = 0.0;
            for (int rep = 0; rep < n_rep; ++rep) {
                std::vector<std::vector<float>> data(ns, std::vector<float>(D));
                for (int s = 0; s < ns; ++s) {
                    const float amp = std::exp(nd(rng)); // states of different norms
                    for (auto & v : data[s]) { v = amp*nd(rng); }
                    if (rep == n_rep - 1) {
                        // signed zeros, and one state that is all zero
                        for (auto & v : data[s]) { v = rng() % 4 == 0 || s == 1 ? (rng() % 2 ? 0.0f : -0.0f) : v; }
                    }
                }
                std::vector<float> q((size_t) D*nq);
                // large queries make the softmax selective, small ones keep it flat
                const float qamp = rep % 2 ? 8.0f : 0.5f;
                for (auto & v : q) { v = qamp*nd(rng); }
                const result ref   = run(hip, ns, nq, false, false, data, q);
                const result probe = run(hip, ns, nq, false, true,  data, q);
                const result fused = run(hip, ns, nq, true,  false, data, q);
                const result host  = run(cpu, ns, nq, true,  false, data, q);
                check(host.ran, "cpu reference");
                n_probe_diff += count_diff(ref.out, probe.out);
                for (int j = 0; j < nq; ++j) {
                    for (int i = 0; i < D; ++i) {
                        float x[8], p[8];
                        for (int s = 0; s < ns; ++s) { x[s] = probe.normed[(size_t) s*D + i]; p[s] = probe.probs[(size_t) j*ns + s]; }
                        for (int o = 0; o < ORDER_COUNT; ++o) {
                            const float v = product(o, x, p, ns);
                            n_order[o] += std::memcmp(&v, &ref.out[(size_t) j*D + i], sizeof(float)) != 0;
                        }
                    }
                }
                double amax = 0.0, d_fused = 0.0, d_host = 0.0;
                for (size_t i = 0; i < ref.out.size(); ++i) {
                    check(std::isfinite(ref.out[i]) && std::isfinite(host.out[i]), "finite");
                    amax   = std::max(amax,   (double) std::fabs(ref.out[i]));
                    d_host = std::max(d_host, (double) std::fabs(ref.out[i] - host.out[i]));
                    if (fused.ran) { d_fused = std::max(d_fused, (double) std::fabs(ref.out[i] - fused.out[i])); }
                }
                if (fused.ran) {
                    n_fused++;
                    n_diff   += count_diff(ref.out, fused.out);
                    rel_fused = std::max(rel_fused, d_fused/amax);
                }
                rel_host = std::max(rel_host, d_host/amax);
            }
            int order = -1;
            for (int o = ORDER_COUNT - 1; o >= 0; --o) {
                if (n_order[o] == 0) { order = o; }
            }
            check(n_fused == 0 || n_fused == n_rep, "fused op support changed between runs");
            std::printf("ns %d nq %d: stock order: %-20s probe diff %zu | ", ns, nq, order < 0 ? "unknown" : order_name[order], n_probe_diff);
            if (n_fused) {
                std::printf("fused: %zu of %d values differ, rel %.2e | cpu ref rel %.2e\n", n_diff, n_rep*D*nq, rel_fused, rel_host);
            } else {
                std::printf("fused: not supported, stock path | cpu ref rel %.2e\n", rel_host);
            }
            total_diff += n_diff;
            worst = std::max(worst, std::max(rel_fused, rel_host));
        }
    }
    check(worst < 2e-5, "relative error above 2e-5");
    check(total_diff == 0, "the exact op differs from the stock chain");
    std::printf("PASS %zu values differ, worst relative error %.2e\n", total_diff, worst);
    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    return 0;
}

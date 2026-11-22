// Fixture for the single-token depth mix op. Compares the stock chain on ROCm0, the fused op on ROCm0 and the CPU reference.
//   GGML_INKLING_MIX1=1 inkling-hip-mix1 ABS_LIBDIR
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-inkling-mix1.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

static std::vector<float> run(ggml_backend_t be, int ns, int nq, bool fused, const std::vector<std::vector<float>> & data, const std::vector<float> & qdata) {
    constexpr int D = 2304;
    ggml_context * c = ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    std::vector<ggml_tensor *> rows(ns);
    for (auto & v : rows) { v = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 1); ggml_set_input(v); }
    ggml_tensor * query = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, nq); ggml_set_input(query);
    ggml_tensor * out;
    if (fused) {
        out = ggml_inkling_try_mix1(c, rows.data(), ns, query);
        check(out && ggml_inkling_is_mix1(out), "factory/identity");
    } else {
        // same as graph::mix: the last row is cur, the blocks are prepended
        ggml_tensor * states = ggml_reshape_3d(c, rows[ns - 1], D, 1, 1);
        for (int s = ns - 2; s >= 0; --s) {
            states = ggml_concat(c, ggml_reshape_3d(c, rows[s], D, 1, 1), states, 1);
        }
        states = ggml_rms_norm(c, states, 1e-6f);
        ggml_tensor * queries = ggml_repeat(c, query, ggml_new_tensor_3d(c, GGML_TYPE_F32, D, nq, 1));
        ggml_tensor * scores  = ggml_mul_mat(c, states, queries);
        scores = ggml_soft_max_ext(c, scores, nullptr, 1.0f/sqrtf(D), 0.0f);
        out = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, states)), scores);
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    check(ggml_gallocr_alloc_graph(ga, gf), "alloc");
    for (int s = 0; s < ns; ++s) { ggml_backend_tensor_set(rows[s], data[s].data(), 0, D*sizeof(float)); }
    ggml_backend_tensor_set(query, qdata.data(), 0, (size_t) D*nq*sizeof(float));
    check(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute");
    std::vector<float> res((size_t) D*nq);
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(c);
    return res;
}

int main(int argc, char ** argv) {
    check(argc == 2, "usage: inkling-hip-mix1 ABS_LIBDIR");
    check(ggml_inkling_mix1_enabled(), "set GGML_INKLING_MIX1=1");
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t hip = ggml_backend_init_by_name("ROCm0", nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    check(hip && cpu, "backends");
    constexpr int D = 2304;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    double worst = 0.0;
    for (int rep = 0; rep < 4; ++rep) {
        for (int ns = 2; ns <= ggml_inkling_mix1_max_states; ++ns) {
            for (int nq : {1, 3, 4}) {
                std::vector<std::vector<float>> data(ns, std::vector<float>(D));
                for (int s = 0; s < ns; ++s) {
                    const float amp = std::exp(nd(rng)); // states of different norms
                    for (auto & v : data[s]) { v = amp*nd(rng); }
                }
                std::vector<float> q((size_t) D*nq);
                // large queries make the softmax selective, small ones keep it flat
                const float qamp = rep % 2 ? 8.0f : 0.5f;
                for (auto & v : q) { v = qamp*nd(rng); }
                const auto ref   = run(hip, ns, nq, false, data, q);
                const auto fused = run(hip, ns, nq, true,  data, q);
                const auto host  = run(cpu, ns, nq, true,  data, q);
                double amax = 0.0, d_fused = 0.0, d_host = 0.0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    check(std::isfinite(ref[i]) && std::isfinite(fused[i]) && std::isfinite(host[i]), "finite");
                    amax    = std::max(amax,    (double) std::fabs(ref[i]));
                    d_fused = std::max(d_fused, (double) std::fabs(ref[i] - fused[i]));
                    d_host  = std::max(d_host,  (double) std::fabs(ref[i] - host[i]));
                }
                std::printf("rep %d ns %d nq %d: max|ref| %.4f, fused vs stock %.3e (rel %.2e), cpu ref vs stock %.3e (rel %.2e)\n",
                    rep, ns, nq, amax, d_fused, d_fused/amax, d_host, d_host/amax);
                worst = std::max(worst, std::max(d_fused, d_host)/amax);
            }
        }
    }
    check(worst < 2e-5, "relative error above 2e-5");
    std::printf("PASS worst relative error %.2e\n", worst);
    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    return 0;
}

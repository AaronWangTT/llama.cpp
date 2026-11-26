// Fixture for the prefill float product kernel. The products of the three weight shapes of the model must match rocBLAS SGEMM
// bit for bit, with F32 weights and with BF16 weights that hold the same values.
//   inkling-hip-pfmm ABS_LIBDIR
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-kperf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

constexpr int T = 256;

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

// dst = w x. A shifted x is a view that starts one float into its buffer: the kernel needs float4 loads, so this product runs SGEMM.
static std::vector<float> run(ggml_backend_t be, ggml_type wtype, bool shifted, int n_k, int n_rows,
        const std::vector<float> & w_data, const std::vector<float> & x_data) {
    ggml_context * c = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * w   = ggml_new_tensor_2d(c, wtype, n_k, n_rows);              ggml_set_input(w);
    ggml_tensor * buf = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t) n_k*T + 4); ggml_set_input(buf);
    ggml_tensor * x   = ggml_view_2d(c, buf, n_k, T, n_k*sizeof(float), shifted ? sizeof(float) : 0);
    ggml_tensor * out = ggml_mul_mat(c, w, x);
    // same as the router and graph::control
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    check(ggml_gallocr_alloc_graph(ga, gf), "alloc");
    if (wtype == GGML_TYPE_F32) {
        ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size()*sizeof(float));
    } else {
        std::vector<uint16_t> h(w_data.size());
        for (size_t i = 0; i < h.size(); ++i) {
            uint32_t bits;
            std::memcpy(&bits, &w_data[i], sizeof(bits));
            check((bits & 0xffff) == 0, "weight is not exact in BF16");
            h[i] = bits >> 16;
        }
        ggml_backend_tensor_set(w, h.data(), 0, h.size()*sizeof(uint16_t));
    }
    ggml_backend_tensor_set(buf, x_data.data(), shifted ? sizeof(float) : 0, x_data.size()*sizeof(float));
    check(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute");
    std::vector<float> res((size_t) n_rows*T);
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(c);
    return res;
}

int main(int argc, char ** argv) {
    check(argc == 2, "usage: inkling-hip-pfmm ABS_LIBDIR");
    check(ggml_kperf_on(GGML_KPERF_INKLING_PFMM), "inkling.pfmm is off (GGML_KPERF_OFF)");
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t hip = ggml_backend_init_by_name("ROCm0", nullptr);
    check(hip, "backend");
    std::mt19937 rng(9876);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int shapes[3][2] = { { 2304, 128 }, { 4608, 64 }, { 64, 2304 } }; // router, controller down, controller gain and forget
    int n_diff = 0;
    for (int rep = 0; rep < 12; ++rep) {
        const int n_k = shapes[rep % 3][0], n_rows = shapes[rep % 3][1];
        std::vector<float> w((size_t) n_k*n_rows), x((size_t) n_k*T);
        const float amp_w = 0.03f*std::exp(nd(rng)), amp_x = std::exp(2.0f*nd(rng));
        for (auto & v : w) {
            v = amp_w*nd(rng);
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            bits &= 0xffff0000u;
            std::memcpy(&v, &bits, sizeof(bits));
        }
        for (auto & v : x) { v = amp_x*nd(rng); }
        const auto ref    = run(hip, GGML_TYPE_F32,  true,  n_k, n_rows, w, x); // SGEMM
        const auto f32    = run(hip, GGML_TYPE_F32,  false, n_k, n_rows, w, x); // kernel
        const auto bf16   = run(hip, GGML_TYPE_BF16, false, n_k, n_rows, w, x); // kernel
        const auto widen  = run(hip, GGML_TYPE_BF16, true,  n_k, n_rows, w, x); // SGEMM on the widened weight
        int d_f32 = 0, d_bf16 = 0, d_widen = 0;
        double amax = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            check(std::isfinite(ref[i]), "finite");
            d_f32   += std::memcmp(&ref[i], &f32[i],   sizeof(float)) != 0;
            d_bf16  += std::memcmp(&ref[i], &bf16[i],  sizeof(float)) != 0;
            d_widen += std::memcmp(&ref[i], &widen[i], sizeof(float)) != 0;
            amax = std::max(amax, (double) std::fabs(ref[i]));
        }
        std::printf("rep %2d k %4d rows %4d: max|ref| %.4f, of %zu values differ: F32 kernel %d, BF16 kernel %d, BF16 SGEMM %d\n",
            rep, n_k, n_rows, amax, ref.size(), d_f32, d_bf16, d_widen);
        n_diff += d_f32 + d_bf16 + d_widen;
    }
    check(n_diff == 0, "not bit-exact");
    std::printf("PASS bit-exact\n");
    ggml_backend_free(hip);
    return 0;
}

// Fixture for the single-token loop controller op. The fused op on ROCm0 must match the stock chain on ROCm0 bit for bit.
//   GGML_INKLING_CTRL1=1 [GGML_INKLING_CTRL_TAIL=1] inkling-hip-ctrl1 ABS_LIBDIR
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-inkling-ctrl1.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

constexpr int D = ggml_inkling_ctrl1_n_embd;
constexpr int R = ggml_inkling_ctrl1_rank;

struct inputs {
    std::vector<float> a, b; // [D, 4] each, cur and message are columns of them
    std::vector<float> down, gain, forget;
    int ja, jb;
};

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

static std::vector<float> run(ggml_backend_t be, bool fused, const inputs & in) {
    ggml_context * c = ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * a      = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 4);   ggml_set_input(a);
    ggml_tensor * b      = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 4);   ggml_set_input(b);
    ggml_tensor * down   = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2*D, R); ggml_set_input(down);
    ggml_tensor * gain   = ggml_new_tensor_2d(c, GGML_TYPE_F32, R, D);   ggml_set_input(gain);
    ggml_tensor * forget = ggml_new_tensor_2d(c, GGML_TYPE_F32, R, D);   ggml_set_input(forget);
    // the names let the stock control tail fusion match
    ggml_set_name(down,   "blk.6.loop_q_ctrl_down.weight");
    ggml_set_name(gain,   "blk.6.loop_q_ctrl_gain.weight");
    ggml_set_name(forget, "blk.6.loop_q_ctrl_forget.weight");
    // same as the column views of graph::build_layer
    ggml_tensor * cur = ggml_reshape_2d(c, ggml_view_2d(c, a, D, 1, a->nb[1], in.ja*a->nb[1]), D, 1);
    ggml_tensor * msg = ggml_reshape_2d(c, ggml_view_2d(c, b, D, 1, b->nb[1], in.jb*b->nb[1]), D, 1);
    ggml_tensor * out;
    if (fused) {
        out = ggml_inkling_try_ctrl1(c, cur, msg, down, gain, forget);
        check(out && ggml_inkling_is_ctrl1(out), "factory/identity");
    } else {
        // same as graph::control
        ggml_tensor * gate = ggml_concat(c, ggml_rms_norm(c, cur, 1e-6f), ggml_rms_norm(c, msg, 1e-6f), 0);
        gate = ggml_mul_mat(c, down, gate);
        ggml_tensor * g = ggml_tanh(c, ggml_mul_mat(c, gain,   gate));
        ggml_tensor * f = ggml_tanh(c, ggml_mul_mat(c, forget, gate));
        out = ggml_add(c, ggml_sub(c, cur, ggml_mul(c, f, cur)), ggml_mul(c, g, msg));
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    check(ggml_gallocr_alloc_graph(ga, gf), "alloc");
    ggml_backend_tensor_set(a,      in.a.data(),      0, in.a.size()*sizeof(float));
    ggml_backend_tensor_set(b,      in.b.data(),      0, in.b.size()*sizeof(float));
    ggml_backend_tensor_set(down,   in.down.data(),   0, in.down.size()*sizeof(float));
    ggml_backend_tensor_set(gain,   in.gain.data(),   0, in.gain.size()*sizeof(float));
    ggml_backend_tensor_set(forget, in.forget.data(), 0, in.forget.size()*sizeof(float));
    check(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute");
    std::vector<float> res(D);
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(c);
    return res;
}

int main(int argc, char ** argv) {
    check(argc == 2, "usage: inkling-hip-ctrl1 ABS_LIBDIR");
    check(ggml_inkling_ctrl1_enabled(), "set GGML_INKLING_CTRL1=1");
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t hip = ggml_backend_init_by_name("ROCm0", nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    check(hip && cpu, "backends");
    std::mt19937 rng(4321);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int    n_diff = 0;
    double worst  = 0.0;
    for (int rep = 0; rep < 32; ++rep) {
        inputs in;
        in.a.resize(4*D); in.b.resize(4*D); in.down.resize(2*D*R); in.gain.resize(R*D); in.forget.resize(R*D);
        in.ja = rep % 4;
        in.jb = (rep / 4) % 4;
        // inputs of different norms; small and large projections cover the linear and the saturated range of tanh
        const float amp_a = std::exp(2.0f*nd(rng)), amp_b = std::exp(2.0f*nd(rng));
        const float amp_d = 0.02f*std::exp(nd(rng)), amp_g = 0.05f*std::exp(nd(rng));
        for (auto & v : in.a)      { v = amp_a*nd(rng); }
        for (auto & v : in.b)      { v = amp_b*nd(rng); }
        for (auto & v : in.down)   { v = amp_d*nd(rng); }
        for (auto & v : in.gain)   { v = amp_g*nd(rng); }
        for (auto & v : in.forget) { v = amp_g*nd(rng); }
        const auto ref   = run(hip, false, in);
        const auto fused = run(hip, true,  in);
        const auto host  = run(cpu, true,  in);
        int    diff = 0;
        double amax = 0.0, d_host = 0.0;
        for (int i = 0; i < D; ++i) {
            check(std::isfinite(ref[i]) && std::isfinite(fused[i]) && std::isfinite(host[i]), "finite");
            diff  += std::memcmp(&ref[i], &fused[i], sizeof(float)) != 0;
            amax   = std::max(amax,   (double) std::fabs(ref[i]));
            d_host = std::max(d_host, (double) std::fabs(ref[i] - host[i]));
        }
        std::printf("rep %2d cols %d %d: max|ref| %.4f, fused vs stock %d of %d values differ, cpu ref vs stock %.3e (rel %.2e)\n",
            rep, in.ja, in.jb, amax, diff, D, d_host, d_host/amax);
        n_diff += diff;
        worst   = std::max(worst, d_host/amax);
    }
    check(n_diff == 0, "fused op is not bit-exact");
    check(worst < 1e-4, "cpu reference error above 1e-4");
    std::printf("PASS bit-exact, worst cpu reference error %.2e\n", worst);
    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    return 0;
}

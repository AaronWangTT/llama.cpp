// Fixture for the single-token loop controller op. The fused op on ROCm0 must match the stock chain on ROCm0 bit for bit.
// BF16 weights that hold the same values must give the bits of the F32 weights, in the fused op and in the stock chain of any batch size.
//   GGML_INKLING_CTRL1=1 [GGML_INKLING_CTRL_TAIL=1] inkling-hip-ctrl1 ABS_LIBDIR
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-inkling-ctrl1.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

constexpr int D = ggml_inkling_ctrl1_n_embd;
constexpr int R = ggml_inkling_ctrl1_rank;

struct inputs {
    std::vector<float> a, b; // one token: [D, 4] each, cur and message are columns of them; else [D, n_tokens] each
    std::vector<float> down, gain, forget;
    int ja, jb;
    int n_tokens = 1;
};

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

// F32 values to a weight of type F32 or BF16
static void set_weight(ggml_tensor * w, const std::vector<float> & v) {
    if (w->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(w, v.data(), 0, v.size()*sizeof(float));
        return;
    }
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], sizeof(bits));
        check((bits & 0xffff) == 0, "weight is not exact in BF16");
        h[i] = bits >> 16;
    }
    ggml_backend_tensor_set(w, h.data(), 0, h.size()*sizeof(uint16_t));
}

// the products of a BF16 weight accumulate in F32, same as graph::control
static ggml_tensor * project(ggml_context * c, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * res = ggml_mul_mat(c, w, x);
    if (w->type == GGML_TYPE_BF16) {
        ggml_prec_set_acc(res, GGML_PREC_F32);
    }
    return res;
}

static std::vector<float> run(ggml_backend_t be, bool fused, const inputs & in, ggml_type wtype = GGML_TYPE_F32) {
    const int T = in.n_tokens;
    ggml_context * c = ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * a      = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T == 1 ? 4 : T); ggml_set_input(a);
    ggml_tensor * b      = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T == 1 ? 4 : T); ggml_set_input(b);
    ggml_tensor * down   = ggml_new_tensor_2d(c, wtype, 2*D, R); ggml_set_input(down);
    ggml_tensor * gain   = ggml_new_tensor_2d(c, wtype, R, D);   ggml_set_input(gain);
    ggml_tensor * forget = ggml_new_tensor_2d(c, wtype, R, D);   ggml_set_input(forget);
    // the names let the stock control tail fusion match
    ggml_set_name(down,   "blk.6.loop_q_ctrl_down.weight");
    ggml_set_name(gain,   "blk.6.loop_q_ctrl_gain.weight");
    ggml_set_name(forget, "blk.6.loop_q_ctrl_forget.weight");
    // same as the column views of graph::build_layer
    ggml_tensor * cur = T == 1 ? ggml_reshape_2d(c, ggml_view_2d(c, a, D, 1, a->nb[1], in.ja*a->nb[1]), D, 1) : a;
    ggml_tensor * msg = T == 1 ? ggml_reshape_2d(c, ggml_view_2d(c, b, D, 1, b->nb[1], in.jb*b->nb[1]), D, 1) : b;
    ggml_tensor * out;
    if (fused) {
        check(T == 1, "the fused op takes one token");
        out = ggml_inkling_try_ctrl1(c, cur, msg, down, gain, forget);
        check(out && ggml_inkling_is_ctrl1(out), "factory/identity");
    } else {
        // same as graph::control
        ggml_tensor * gate = ggml_concat(c, ggml_rms_norm(c, cur, 1e-6f), ggml_rms_norm(c, msg, 1e-6f), 0);
        gate = project(c, down, gate);
        ggml_tensor * g = ggml_tanh(c, project(c, gain,   gate));
        ggml_tensor * f = ggml_tanh(c, project(c, forget, gate));
        out = ggml_add(c, ggml_sub(c, cur, ggml_mul(c, f, cur)), ggml_mul(c, g, msg));
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    check(ggml_gallocr_alloc_graph(ga, gf), "alloc");
    ggml_backend_tensor_set(a,      in.a.data(),      0, in.a.size()*sizeof(float));
    ggml_backend_tensor_set(b,      in.b.data(),      0, in.b.size()*sizeof(float));
    set_weight(down,   in.down);
    set_weight(gain,   in.gain);
    set_weight(forget, in.forget);
    check(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute");
    std::vector<float> res(D*T);
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

    // BF16 weights: the values of the F32 weights with the low 16 bits cleared, stored in both types
    const int batches[] = { 1, 1, 1, 1, 2, 3, 4, 5, 8, 9, 32, 33, 256 };
    int n_diff_bf16 = 0;
    for (int rep = 0; rep < (int) std::size(batches); ++rep) {
        inputs in;
        in.n_tokens = batches[rep];
        const int T = in.n_tokens;
        in.a.resize((T == 1 ? 4 : T)*D); in.b.resize(in.a.size()); in.down.resize(2*D*R); in.gain.resize(R*D); in.forget.resize(R*D);
        in.ja = rep % 4;
        in.jb = (rep + 1) % 4;
        const float amp_d = 0.02f*std::exp(nd(rng)), amp_g = 0.05f*std::exp(nd(rng));
        for (auto & v : in.a) { v = nd(rng); }
        for (auto & v : in.b) { v = nd(rng); }
        for (auto * w : { &in.down, &in.gain, &in.forget }) {
            for (auto & v : *w) {
                v = (w == &in.down ? amp_d : amp_g)*nd(rng);
                uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                bits &= 0xffff0000u;
                std::memcpy(&v, &bits, sizeof(bits));
            }
        }
        const auto ref   = run(hip, false, in);
        const auto chain = run(hip, false, in, GGML_TYPE_BF16);
        int diff_chain = 0, diff_fused = 0;
        double d_host = 0.0, amax = 0.0;
        for (int i = 0; i < D*T; ++i) {
            check(std::isfinite(ref[i]) && std::isfinite(chain[i]), "finite");
            diff_chain += std::memcmp(&ref[i], &chain[i], sizeof(float)) != 0;
        }
        if (T == 1) {
            const auto fused = run(hip, true, in, GGML_TYPE_BF16);
            const auto host  = run(cpu, true, in, GGML_TYPE_BF16);
            for (int i = 0; i < D; ++i) {
                check(std::isfinite(fused[i]) && std::isfinite(host[i]), "finite");
                diff_fused += std::memcmp(&ref[i], &fused[i], sizeof(float)) != 0;
                amax   = std::max(amax,   (double) std::fabs(ref[i]));
                d_host = std::max(d_host, (double) std::fabs(ref[i] - host[i]));
            }
            check(d_host/amax < 1e-4, "cpu reference error above 1e-4 with BF16 weights");
        }
        std::printf("bf16 rep %2d tokens %3d: stock chain %d of %d values differ, fused %d, cpu ref rel %.2e\n",
            rep, T, diff_chain, D*T, diff_fused, T == 1 ? d_host/amax : 0.0);
        n_diff_bf16 += diff_chain + diff_fused;
    }
    check(n_diff_bf16 == 0, "BF16 weights are not bit-exact");
    std::printf("PASS bit-exact, worst cpu reference error %.2e\n", worst);
    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    return 0;
}

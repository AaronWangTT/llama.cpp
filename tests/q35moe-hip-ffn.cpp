// Fixture for the one-token feed-forward op of the qwen35moe blocks (ggml-q35moe-ffn.h).
// The fused op on ROCm0 must match the stock nodes of build_moe_ffn() and build_ffn() bit for bit, for every weight type with a kernel.
// Both graphs run through the scheduler as in llama.cpp: only then the backend fuses the top-k chain of the stock nodes into
// topk_moe_cuda, the form that the op replays (without the scheduler the chain runs node by node and rounds differently).
// Cases with a zero router, a zero shared expert, zero routed experts and a zero input show which part differs.
// The CPU callback must stay close to the stock nodes on CPU.
//   GGML_Q35MOE_FFN1=1 q35moe-hip-ffn ABS_LIBDIR
#include "ggml-backend.h"
#include "ggml-q35moe-ffn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

constexpr int N      = ggml_q35moe_n_embd;
constexpr int N_FF   = ggml_q35moe_n_ff;
constexpr int N_EXP  = ggml_q35moe_n_expert;
constexpr int N_USED = ggml_q35moe_n_used;
constexpr int N_W    = GGML_Q35MOE_FFN1_N_SRC - GGML_Q35MOE_FFN1_GATE_INP;

// weight index = source of the op - GGML_Q35MOE_FFN1_GATE_INP
enum { W_GATE_INP, W_UP_EXPS, W_GATE_EXPS, W_DOWN_EXPS, W_GATE_INP_SHEXP, W_UP_SHEXP, W_GATE_SHEXP, W_DOWN_SHEXP };

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

struct types {
    ggml_type up, down, shexp_up, shexp_down;
};

struct model {
    ggml_backend_t be;
    ggml_backend_t cpu; // last backend of the scheduler
    ggml_context * ctx;
    ggml_tensor *  w[N_W];
};

// random blocks of a quantized type: random bytes with scales of a sane size
static std::vector<char> random_blocks(ggml_type type, size_t nbytes, std::mt19937_64 & rng) {
    std::vector<char> buf(nbytes);
    for (size_t i = 0; i + 8 <= nbytes; i += 8) {
        const uint64_t v = rng();
        std::memcpy(buf.data() + i, &v, 8);
    }
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    const size_t bs = ggml_type_size(type);
    auto put = [&](size_t offs, float v) {
        const ggml_fp16_t h = ggml_fp32_to_fp16((rng() & 1) ? v : -v);
        std::memcpy(buf.data() + offs, &h, sizeof(h));
    };
    for (size_t b = 0; b + bs <= nbytes; b += bs) {
        switch (type) {
            case GGML_TYPE_Q8_0: put(b, 2e-4f*ud(rng)); break;
            case GGML_TYPE_Q4_K: put(b, 1.5e-4f*ud(rng)); put(b + 2, 1e-3f*ud(rng)); break;
            case GGML_TYPE_Q5_K: put(b, 1e-4f*ud(rng));   put(b + 2, 1e-3f*ud(rng)); break;
            case GGML_TYPE_Q6_K: put(b + bs - 2, 1e-4f*ud(rng)); break; // d is the last field
            default: check(false, "weight type");
        }
    }
    return buf;
}

static std::vector<std::vector<char>> make_data(const types & t, std::mt19937_64 & rng) {
    std::normal_distribution<float> nd(0.0f, 0.05f);
    auto f32 = [&](size_t n) {
        std::vector<char> buf(n*sizeof(float));
        for (size_t i = 0; i < n; ++i) {
            const float v = nd(rng);
            std::memcpy(buf.data() + i*sizeof(float), &v, sizeof(v));
        }
        return buf;
    };
    auto q = [&](ggml_type type, size_t n) {
        return random_blocks(type, n/ggml_blck_size(type)*ggml_type_size(type), rng);
    };
    std::vector<std::vector<char>> d(N_W);
    d[W_GATE_INP]       = f32((size_t) N*N_EXP);
    d[W_UP_EXPS]        = q(t.up,   (size_t) N*N_FF*N_EXP);
    d[W_GATE_EXPS]      = q(t.up,   (size_t) N*N_FF*N_EXP);
    d[W_DOWN_EXPS]      = q(t.down, (size_t) N*N_FF*N_EXP);
    d[W_GATE_INP_SHEXP] = f32(N);
    d[W_UP_SHEXP]       = q(t.shexp_up,   (size_t) N*N_FF);
    d[W_GATE_SHEXP]     = q(t.shexp_up,   (size_t) N*N_FF);
    d[W_DOWN_SHEXP]     = q(t.shexp_down, (size_t) N*N_FF);
    return d;
}

static model make_model(ggml_backend_t be, ggml_backend_t cpu, const types & t, const std::vector<std::vector<char>> & data) {
    model m;
    m.be  = be;
    m.cpu = cpu;
    m.ctx = ggml_init({16*ggml_tensor_overhead(), nullptr, true});
    m.w[W_GATE_INP]       = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, N, N_EXP);
    m.w[W_UP_EXPS]        = ggml_new_tensor_3d(m.ctx, t.up,   N,    N_FF, N_EXP);
    m.w[W_GATE_EXPS]      = ggml_new_tensor_3d(m.ctx, t.up,   N,    N_FF, N_EXP);
    m.w[W_DOWN_EXPS]      = ggml_new_tensor_3d(m.ctx, t.down, N_FF, N,    N_EXP);
    m.w[W_GATE_INP_SHEXP] = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, N);
    m.w[W_UP_SHEXP]       = ggml_new_tensor_2d(m.ctx, t.shexp_up,   N,    N_FF);
    m.w[W_GATE_SHEXP]     = ggml_new_tensor_2d(m.ctx, t.shexp_up,   N,    N_FF);
    m.w[W_DOWN_SHEXP]     = ggml_new_tensor_2d(m.ctx, t.shexp_down, N_FF, N);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(m.ctx, be);
    check(buf != nullptr, "allocation");
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int j = 0; j < N_W; ++j) {
        check(data[j].size() == ggml_nbytes(m.w[j]), "weight size");
        ggml_backend_tensor_set(m.w[j], data[j].data(), 0, data[j].size());
    }
    return m;
}

static std::vector<float> run(const model & m, bool fused, const std::vector<float> & x_data, const std::vector<float> & res_data) {
    ggml_backend_t be = m.be;
    ggml_context * c  = ggml_init({128*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_cgraph *  gf = ggml_new_graph(c);

    ggml_tensor * x   = ggml_new_tensor_2d(c, GGML_TYPE_F32, N, 1); ggml_set_input(x);
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_F32, N, 1); ggml_set_input(res);

    ggml_tensor * out;
    if (fused) {
        ggml_tensor * w[N_W];
        std::memcpy(w, m.w, sizeof(w));
        out = ggml_q35moe_try_ffn1(c, x, res, w);
        check(out && ggml_q35moe_is_ffn1(out) && ggml_backend_supports_op(be, out), "the backend must take the fused op");
    } else {
        // build_moe_ffn() with the settings of qwen35moe
        ggml_tensor * logits   = ggml_mul_mat(c, m.w[W_GATE_INP], x);
        ggml_tensor * probs    = ggml_soft_max(c, logits);
        ggml_tensor * selected = ggml_argsort_top_k(c, probs, N_USED);
        probs = ggml_reshape_3d(c, probs, 1, N_EXP, 1);
        ggml_tensor * weights = ggml_get_rows(c, probs, selected);
        weights = ggml_reshape_2d(c, weights, N_USED, 1);
        ggml_tensor * weights_sum = ggml_sum_rows(c, weights);
        weights_sum = ggml_clamp(c, weights_sum, 6.103515625e-5, INFINITY);
        weights = ggml_div(c, weights, weights_sum);
        weights = ggml_reshape_3d(c, weights, 1, N_USED, 1);
        ggml_build_forward_expand(gf, weights);
        ggml_build_forward_expand(gf, x);

        ggml_tensor * cur  = ggml_reshape_3d(c, x, N, 1, 1);
        ggml_tensor * up   = ggml_mul_mat_id(c, m.w[W_UP_EXPS], cur, selected);
        ggml_tensor * gate = ggml_mul_mat_id(c, m.w[W_GATE_EXPS], cur, selected);
        cur = ggml_swiglu_split(c, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(c, m.w[W_DOWN_EXPS], cur, selected);
        experts = ggml_mul(c, experts, weights);
        ggml_build_forward_expand(gf, experts);

        ggml_tensor * cur_experts[N_USED];
        for (int i = 0; i < N_USED; ++i) {
            cur_experts[i] = ggml_view_2d(c, experts, N, 1, experts->nb[2], i*experts->nb[1]);
            ggml_build_forward_expand(gf, cur_experts[i]);
        }
        ggml_tensor * moe_out = cur_experts[0];
        for (int i = 1; i < N_USED; ++i) {
            moe_out = ggml_add(c, moe_out, cur_experts[i]);
            ggml_build_forward_expand(gf, moe_out);
        }

        // build_ffn() of the shared expert and its gate, as in build_layer_ffn()
        ggml_tensor * tmp = ggml_mul_mat(c, m.w[W_UP_SHEXP], x);
        cur = ggml_mul_mat(c, m.w[W_GATE_SHEXP], x);
        cur = ggml_swiglu_split(c, cur, tmp);
        cur = ggml_mul_mat(c, m.w[W_DOWN_SHEXP], cur);
        ggml_tensor * shared_gate = ggml_sigmoid(c, ggml_mul_mat(c, m.w[W_GATE_INP_SHEXP], x));
        cur = ggml_mul(c, cur, shared_gate);
        cur = ggml_add(c, moe_out, cur);
        out = ggml_add(c, cur, res);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backends[2] = { be, m.cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, be == m.cpu ? 1 : 2, 256, false, true);
    check(sched && ggml_backend_sched_alloc_graph(sched, gf), "alloc");
    ggml_backend_tensor_set(x,   x_data.data(),   0, N*sizeof(float));
    ggml_backend_tensor_set(res, res_data.data(), 0, N*sizeof(float));
    check(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS, "compute");

    std::vector<float> result(N);
    ggml_backend_tensor_get(out, result.data(), 0, N*sizeof(float));
    ggml_backend_sched_free(sched);
    ggml_free(c);
    return result;
}

int main(int argc, char ** argv) {
    check(argc == 2, "usage: q35moe-hip-ffn ABS_LIBDIR");
    check(ggml_q35moe_ffn1_enabled(), "set GGML_Q35MOE_FFN1=1");
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t hip = ggml_backend_init_by_name("ROCm0", nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    check(hip && cpu, "backends");

    // the first set has the types of Qwen3.6-35B-A3B-UD-Q4_K_XL, every type is in every place once
    static const types sets[] = {
        { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K },
        { GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K },
    };
    // weight set to zero in a case, -1: none
    static const int          zero_w[]    = { -1, W_GATE_INP, W_UP_SHEXP, W_UP_EXPS };
    static const char * const case_name[] = { "all", "zero router", "zero shared expert", "zero routed experts" };

    std::mt19937_64 rng(4096);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    long   n_diff    = 0;
    double worst_cpu = 0.0;
    for (size_t s = 0; s < sizeof(sets)/sizeof(sets[0]); ++s) {
        const types & t = sets[s];
        const auto data = make_data(t, rng);
        const model m   = make_model(hip, cpu, t, data);

        for (int cs = 0; cs < 4; ++cs) {
            if (zero_w[cs] >= 0) {
                const std::vector<char> zero(data[zero_w[cs]].size(), 0);
                ggml_backend_tensor_set(m.w[zero_w[cs]], zero.data(), 0, zero.size());
            }
            for (int rep = 0; rep < 3; ++rep) {
                std::vector<float> x(N), res(N);
                // the last repetition of a case has a zero input
                const float amp = rep == 2 ? 0.0f : 0.3f*std::exp(0.5f*nd(rng));
                for (auto & v : x)   { v = amp*nd(rng); }
                for (auto & v : res) { v = nd(rng); }

                const auto ref = run(m, false, x, res);
                const auto out = run(m, true,  x, res);
                int    diff = 0, first = -1;
                double dmax = 0.0, amax = 0.0;
                for (int i = 0; i < N; ++i) {
                    check(std::isfinite(ref[i]) && std::isfinite(out[i]), "finite");
                    if (std::memcmp(&ref[i], &out[i], sizeof(float)) != 0) {
                        first = first < 0 ? i : first;
                        diff++;
                    }
                    dmax = std::max(dmax, (double) std::fabs(ref[i] - out[i]));
                    amax = std::max(amax, (double) std::fabs(ref[i] - res[i]));
                }
                std::printf("types %s %s %s %s, %s, rep %d: %d/%d differ, max diff %.3e, max|ffn| %.4f",
                    ggml_type_name(t.up), ggml_type_name(t.down), ggml_type_name(t.shexp_up), ggml_type_name(t.shexp_down),
                    case_name[cs], rep, diff, N, dmax, amax);
                if (first >= 0) {
                    std::printf(", first at %d: %a stock, %a fused", first, ref[first], out[first]);
                }
                std::printf("\n");
                n_diff += diff;

                // CPU: the callback against the stock nodes, which quantize the input of a product
                if (cs == 0 && rep == 0 && s < 2) {
                    const model m_cpu  = make_model(cpu, cpu, t, data);
                    const auto ref_cpu = run(m_cpu, false, x, res);
                    const auto out_cpu = run(m_cpu, true,  x, res);
                    double dmax_cpu = 0.0, amax_cpu = 0.0, dmax_dev = 0.0;
                    for (int i = 0; i < N; ++i) {
                        amax_cpu = std::max(amax_cpu, (double) std::fabs(ref_cpu[i] - res[i]));
                        dmax_cpu = std::max(dmax_cpu, (double) std::fabs(ref_cpu[i] - out_cpu[i]));
                        dmax_dev = std::max(dmax_dev, (double) std::fabs(ref_cpu[i] - ref[i]));
                    }
                    std::printf("cpu callback: max|ffn| %.4f, max diff to the stock nodes on cpu %.3e, stock cpu to stock device %.3e\n",
                        amax_cpu, dmax_cpu, dmax_dev);
                    worst_cpu = std::max(worst_cpu, dmax_cpu/amax_cpu);
                    ggml_backend_buffer_free(m_cpu.w[0]->buffer);
                    ggml_free(m_cpu.ctx);
                }
            }
            if (zero_w[cs] >= 0) {
                ggml_backend_tensor_set(m.w[zero_w[cs]], data[zero_w[cs]].data(), 0, data[zero_w[cs]].size());
            }
        }
        ggml_backend_buffer_free(m.w[0]->buffer);
        ggml_free(m.ctx);
    }
    check(n_diff == 0, "the fused op is not bit-exact");
    check(worst_cpu < 5e-2, "cpu reference error above 5e-2");
    std::printf("PASS bit-exact, worst cpu reference error %.2e\n", worst_cpu);
    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    return 0;
}

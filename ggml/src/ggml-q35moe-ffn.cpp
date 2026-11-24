#include "ggml-q35moe-ffn.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int n_embd   = ggml_q35moe_n_embd;
constexpr int n_ff     = ggml_q35moe_n_ff;
constexpr int n_expert = ggml_q35moe_n_expert;
constexpr int n_used   = ggml_q35moe_n_used;

static_assert(GGML_Q35MOE_FFN1_N_SRC == GGML_MAX_SRC, "the op fills every source of a node");

bool env_on(const char * name) {
    const char * p = std::getenv(name);
    return p && std::strcmp(p, "1") == 0;
}

bool f32_vec(const ggml_tensor * x) {
    return x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[0] == n_embd && ggml_nelements(x) == n_embd;
}

// a weight leaf of this shape
bool weight(const ggml_tensor * w, int64_t ne0, int64_t ne1, int64_t ne2) {
    return w && w->op == GGML_OP_NONE && !w->view_src && ggml_is_contiguous(w) &&
        w->ne[0] == ne0 && w->ne[1] == ne1 && w->ne[2] == ne2 && w->ne[3] == 1;
}

// the weights of a block
bool weights_layout(const ggml_tensor * const * src) {
    const ggml_tensor * gate_inp       = src[GGML_Q35MOE_FFN1_GATE_INP];
    const ggml_tensor * up_exps        = src[GGML_Q35MOE_FFN1_UP_EXPS];
    const ggml_tensor * gate_exps      = src[GGML_Q35MOE_FFN1_GATE_EXPS];
    const ggml_tensor * down_exps      = src[GGML_Q35MOE_FFN1_DOWN_EXPS];
    const ggml_tensor * gate_inp_shexp = src[GGML_Q35MOE_FFN1_GATE_INP_SHEXP];
    const ggml_tensor * up_shexp       = src[GGML_Q35MOE_FFN1_UP_SHEXP];
    const ggml_tensor * gate_shexp     = src[GGML_Q35MOE_FFN1_GATE_SHEXP];
    const ggml_tensor * down_shexp     = src[GGML_Q35MOE_FFN1_DOWN_SHEXP];

    return weight(gate_inp, n_embd, n_expert, 1) && gate_inp->type == GGML_TYPE_F32 &&
        weight(up_exps, n_embd, n_ff, n_expert) && ggml_q35moe_ffn1_type(up_exps->type) &&
        weight(gate_exps, n_embd, n_ff, n_expert) && gate_exps->type == up_exps->type &&
        weight(down_exps, n_ff, n_embd, n_expert) && ggml_q35moe_ffn1_type(down_exps->type) &&
        weight(gate_inp_shexp, n_embd, 1, 1) && gate_inp_shexp->type == GGML_TYPE_F32 &&
        weight(up_shexp, n_embd, n_ff, 1) && ggml_q35moe_ffn1_type(up_shexp->type) &&
        weight(gate_shexp, n_embd, n_ff, 1) && gate_shexp->type == up_shexp->type &&
        weight(down_shexp, n_ff, n_embd, 1) && ggml_q35moe_ffn1_type(down_shexp->type);
}

bool ffn1_layout(const ggml_tensor * const * src) {
    return f32_vec(src[GGML_Q35MOE_FFN1_X]) && f32_vec(src[GGML_Q35MOE_FFN1_RES]) && weights_layout(src);
}

// n_embd values for each of 2 to ggml_q35moe_ffnm_max tokens, one more with GGML_Q35MOE_FFNM5
bool f32_cols(const ggml_tensor * x) {
    const int64_t n_max = ggml_q35moe_ffnm5_enabled() ? ggml_q35moe_ffnm5_max : ggml_q35moe_ffnm_max;
    return x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[0] == n_embd &&
        x->ne[1] >= 2 && x->ne[1] <= n_max && x->ne[2] == 1 && x->ne[3] == 1;
}

bool ffnm_layout(const ggml_tensor * const * src) {
    const ggml_tensor * x   = src[GGML_Q35MOE_FFN1_X];
    const ggml_tensor * res = src[GGML_Q35MOE_FFN1_RES];
    return f32_cols(x) && f32_cols(res) && x->ne[1] == res->ne[1] && weights_layout(src);
}

// y = m[e]*x with the dequantized rows of expert e
void matvec(const ggml_tensor * m, int64_t e, const float * x, float * y, std::vector<float> & row) {
    const int64_t n = m->ne[0];
    for (int64_t r = 0; r < m->ne[1]; ++r) {
        const char * p = static_cast<const char *>(m->data) + e*m->nb[2] + r*m->nb[1];
        if (m->type == GGML_TYPE_F32) {
            std::memcpy(row.data(), p, n*sizeof(float));
        } else {
            ggml_get_type_traits(m->type)->to_float(p, row.data(), n);
        }
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += row[i]*x[i];
        }
        y[r] = sum;
    }
}

// reference, one token
void ffn_cpu_token(ggml_tensor ** src, const float * x, const float * res, float * out) {
    std::vector<float> row(n_embd);

    // softmax of the router logits
    float p[n_expert];
    matvec(src[GGML_Q35MOE_FFN1_GATE_INP], 0, x, p, row);
    float p_max = p[0];
    for (int i = 1; i < n_expert; ++i) {
        p_max = std::fmax(p_max, p[i]);
    }
    float p_sum = 0.0f;
    for (int i = 0; i < n_expert; ++i) {
        p[i] = expf(p[i] - p_max);
        p_sum += p[i];
    }

    // the n_used largest in order, the lower index wins a tie
    int   ids[n_used];
    float w[n_used];
    float w_sum = 0.0f;
    for (int k = 0; k < n_used; ++k) {
        int best = 0;
        for (int i = 1; i < n_expert; ++i) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        ids[k]  = best;
        w[k]    = p[best]/p_sum;
        w_sum  += w[k];
        p[best] = -INFINITY;
    }
    w_sum = std::fmax(w_sum, ggml_q35moe_sum_min);

    std::vector<float> acc(n_embd, 0.0f), up(n_ff), gate(n_ff), down(n_embd);
    auto expert = [&](int j_up, int j_gate, int j_down, int64_t e, float scale) {
        matvec(src[j_up],   e, x, up.data(),   row);
        matvec(src[j_gate], e, x, gate.data(), row);
        for (int i = 0; i < n_ff; ++i) {
            up[i] *= gate[i]/(1.0f + expf(-gate[i]));
        }
        matvec(src[j_down], e, up.data(), down.data(), row);
        for (int i = 0; i < n_embd; ++i) {
            acc[i] += scale*down[i];
        }
    };
    for (int k = 0; k < n_used; ++k) {
        expert(GGML_Q35MOE_FFN1_UP_EXPS, GGML_Q35MOE_FFN1_GATE_EXPS, GGML_Q35MOE_FFN1_DOWN_EXPS, ids[k], w[k]/w_sum);
    }

    float g;
    matvec(src[GGML_Q35MOE_FFN1_GATE_INP_SHEXP], 0, x, &g, row);
    expert(GGML_Q35MOE_FFN1_UP_SHEXP, GGML_Q35MOE_FFN1_GATE_SHEXP, GGML_Q35MOE_FFN1_DOWN_SHEXP, 0, 1.0f/(1.0f + expf(-g)));

    for (int i = 0; i < n_embd; ++i) {
        out[i] = acc[i] + res[i];
    }
}

void ffn1_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    ggml_tensor ** src = dst->src;
    ffn_cpu_token(src, static_cast<const float *>(src[GGML_Q35MOE_FFN1_X]->data), static_cast<const float *>(src[GGML_Q35MOE_FFN1_RES]->data),
        static_cast<float *>(dst->data));
}

void ffnm_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    ggml_tensor ** src = dst->src;
    const float * x   = static_cast<const float *>(src[GGML_Q35MOE_FFN1_X]->data);
    const float * res = static_cast<const float *>(src[GGML_Q35MOE_FFN1_RES]->data);
    float *       out = static_cast<float *>(dst->data);
    for (int64_t t = 0; t < dst->ne[1]; ++t) {
        ffn_cpu_token(src, x + t*n_embd, res + t*n_embd, out + t*n_embd);
    }
}

bool custom_op(const ggml_tensor * op, ggml_custom_op_t fun, int n_src) {
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || op->type != GGML_TYPE_F32) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != fun || p.userdata || p.n_tasks != 1) {
        return false;
    }
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if ((op->src[j] != nullptr) != (j < n_src)) {
            return false;
        }
    }
    return true;
}

// the op node for n_tokens tokens, nullptr if layout refuses the sources
ggml_tensor * ffn_node(ggml_context * ctx, ggml_tensor * x, ggml_tensor * res, ggml_tensor ** w, int64_t n_tokens,
        bool (*layout)(const ggml_tensor * const *), ggml_custom_op_t fun) {
    ggml_tensor * args[GGML_Q35MOE_FFN1_N_SRC] = { x, res };
    for (int j = GGML_Q35MOE_FFN1_GATE_INP; j < GGML_Q35MOE_FFN1_N_SRC; ++j) {
        args[j] = w[j - GGML_Q35MOE_FFN1_GATE_INP];
    }
    if (!layout(args)) {
        return nullptr;
    }
    // ggml_custom_4d takes one source less than a node holds
    constexpr int last = GGML_Q35MOE_FFN1_N_SRC - 1;
    ggml_tensor * op = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tokens, 1, 1, args, last, fun, 1, nullptr);
    op->src[last] = args[last];
    return op;
}
} // namespace

bool ggml_q35moe_ffn1_enabled() {
    static const bool enabled = env_on("GGML_Q35MOE_FFN1");
    return enabled;
}

ggml_tensor * ggml_q35moe_try_ffn1(ggml_context * ctx, ggml_tensor * x, ggml_tensor * res, ggml_tensor ** w) {
    if (!ggml_q35moe_ffn1_enabled()) {
        return nullptr;
    }
    return ffn_node(ctx, x, res, w, 1, ffn1_layout, ffn1_cpu);
}

bool ggml_q35moe_is_ffn1(const ggml_tensor * op) {
    return custom_op(op, ffn1_cpu, GGML_Q35MOE_FFN1_N_SRC) && op->ne[0] == n_embd && op->ne[1] == 1 && op->ne[2] == 1 && op->ne[3] == 1 &&
        ffn1_layout(op->src);
}

bool ggml_q35moe_ffnm_enabled() {
    static const bool enabled = env_on("GGML_Q35MOE_FFNM");
    return enabled;
}

ggml_tensor * ggml_q35moe_try_ffnm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * res, ggml_tensor ** w) {
    if (!ggml_q35moe_ffnm_enabled() || x == nullptr) {
        return nullptr;
    }
    return ffn_node(ctx, x, res, w, x->ne[1], ffnm_layout, ffnm_cpu);
}

bool ggml_q35moe_is_ffnm(const ggml_tensor * op) {
    return custom_op(op, ffnm_cpu, GGML_Q35MOE_FFN1_N_SRC) && ffnm_layout(op->src) && op->ne[0] == n_embd &&
        op->ne[1] == op->src[GGML_Q35MOE_FFN1_X]->ne[1] && op->ne[2] == 1 && op->ne[3] == 1;
}

bool ggml_q35moe_exponce_enabled() {
    static const bool enabled = env_on("GGML_Q35MOE_EXPONCE");
    return enabled;
}

bool ggml_q35moe_ffnm5_enabled() {
    static const bool enabled = env_on("GGML_Q35MOE_FFNM5");
    return enabled;
}

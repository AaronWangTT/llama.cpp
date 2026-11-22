#include "ggml-inkling-mix1.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int max_embd = 4096;

bool vec_ok(const ggml_tensor * x, int64_t n) {
    return x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_nelements(x) == n;
}

// reference; src[0] is the query, src[1..] are the states
void mix1_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const int n_embd = dst->ne[0];
    const int nq     = dst->ne[1];
    int ns = 0;
    while (ns < ggml_inkling_mix1_max_states && dst->src[ns + 1]) {
        ns++;
    }
    const float * q   = static_cast<const float *>(dst->src[0]->data);
    float *       out = static_cast<float *>(dst->data);

    float scale[ggml_inkling_mix1_max_states];
    for (int s = 0; s < ns; ++s) {
        const float * x = static_cast<const float *>(dst->src[s + 1]->data);
        float sum = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            sum += x[i]*x[i];
        }
        scale[s] = 1.0f/sqrtf(sum/n_embd + ggml_inkling_mix1_eps);
    }
    const float kq = 1.0f/sqrtf(n_embd);
    for (int j = 0; j < nq; ++j) {
        float p[ggml_inkling_mix1_max_states];
        float max = -INFINITY;
        for (int s = 0; s < ns; ++s) {
            const float * x = static_cast<const float *>(dst->src[s + 1]->data);
            float dot = 0.0f;
            for (int i = 0; i < n_embd; ++i) {
                dot += x[i]*scale[s]*q[j*n_embd + i];
            }
            p[s] = dot*kq;
            max  = fmaxf(max, p[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < ns; ++s) {
            p[s] = expf(p[s] - max);
            sum += p[s];
        }
        for (int i = 0; i < n_embd; ++i) {
            float acc = 0.0f;
            for (int s = 0; s < ns; ++s) {
                const float * x = static_cast<const float *>(dst->src[s + 1]->data);
                acc += x[i]*scale[s]*(p[s]/sum);
            }
            out[j*n_embd + i] = acc;
        }
    }
}
} // namespace

bool ggml_inkling_mix1_enabled() {
    static const bool enabled = [] {
        const char * p = std::getenv("GGML_INKLING_MIX1");
        return p && std::strcmp(p, "1") == 0;
    }();
    return enabled;
}

bool ggml_inkling_viewcol_enabled() {
    static const bool enabled = [] {
        const char * p = std::getenv("GGML_INKLING_VIEWCOL");
        return p && std::strcmp(p, "1") == 0;
    }();
    return enabled;
}

ggml_tensor * ggml_inkling_try_mix1(ggml_context * ctx, ggml_tensor ** states, int n_states, ggml_tensor * query) {
    if (!ggml_inkling_mix1_enabled() || n_states < 2 || n_states > ggml_inkling_mix1_max_states || !query) {
        return nullptr;
    }
    const int64_t n_embd = query->ne[0];
    const int64_t nq     = query->ne[1];
    if (n_embd % 4 != 0 || n_embd > max_embd || nq < 1 || nq > ggml_inkling_mix1_max_query || !vec_ok(query, n_embd*nq)) {
        return nullptr;
    }
    ggml_tensor * args[1 + ggml_inkling_mix1_max_states] = { query };
    for (int s = 0; s < n_states; ++s) {
        if (!vec_ok(states[s], n_embd)) {
            return nullptr;
        }
        args[s + 1] = ggml_reshape_1d(ctx, states[s], n_embd);
    }
    return ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, nq, 1, 1, args, n_states + 1, mix1_cpu, 1, nullptr);
}

bool ggml_inkling_is_mix1(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || op->type != GGML_TYPE_F32 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != mix1_cpu || p.userdata || p.n_tasks != 1) {
        return false;
    }
    const int64_t n_embd = op->ne[0];
    const int64_t nq     = op->ne[1];
    if (n_embd % 4 != 0 || n_embd > max_embd || nq < 1 || nq > ggml_inkling_mix1_max_query || !vec_ok(op->src[0], n_embd*nq)) {
        return false;
    }
    int ns = 0;
    for (int j = 1; j < GGML_MAX_SRC; ++j) {
        if (op->src[j]) {
            if (ns != j - 1 || !vec_ok(op->src[j], n_embd)) {
                return false;
            }
            ns++;
        }
    }
    return ns >= 2 && ns <= ggml_inkling_mix1_max_states;
}

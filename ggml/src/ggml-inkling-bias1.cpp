#include "ggml-inkling-bias1.h"
#include "ggml-kperf.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
bool f32_block(const ggml_tensor * x, int64_t n) {
    return x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_nelements(x) == n;
}

bool layout(const ggml_tensor * bias, const ggml_tensor * rcur, const ggml_tensor * sinks) {
    if (!bias || !rcur) {
        return false;
    }
    const int64_t n_rel = bias->ne[0], n_extent = bias->ne[1], n_head = rcur->ne[1];
    return n_rel >= 1 && n_rel <= 64 && n_extent >= 1 && n_head >= 1 && f32_block(bias, n_rel*n_extent) &&
        rcur->ne[0] == n_rel && f32_block(rcur, n_rel*n_head) && f32_block(sinks, n_head);
}

// reference
void bias1_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const ggml_tensor * bias_t = dst->src[0];
    const int64_t n_rel = bias_t->ne[0], n_extent = bias_t->ne[1], n_head = dst->ne[0];
    const float * bias  = static_cast<const float *>(bias_t->data);
    const float * rcur  = static_cast<const float *>(dst->src[1]->data);
    const float * sinks = static_cast<const float *>(dst->src[2]->data);
    float *       out   = static_cast<float *>(dst->data);
    for (int64_t d = 0; d <= n_extent; ++d) {
        for (int64_t h = 0; h < n_head; ++h) {
            float acc = 0.0f;
            for (int64_t r = 0; r < n_rel && d < n_extent; ++r) {
                acc = fmaf(bias[d*n_rel + r], rcur[h*n_rel + r], acc);
            }
            out[d*n_head + h] = acc - sinks[h];
        }
    }
}
} // namespace

bool ggml_inkling_bias1_enabled() {
    return ggml_kperf_on(GGML_KPERF_INKLING_BIAS1);
}

ggml_tensor * ggml_inkling_try_bias1(ggml_context * ctx, ggml_tensor * bias, ggml_tensor * rcur, ggml_tensor * sinks) {
    if (!ggml_inkling_bias1_enabled() || !layout(bias, rcur, sinks)) {
        return nullptr;
    }
    ggml_tensor * args[] = { bias, rcur, sinks };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, rcur->ne[1], bias->ne[1] + 1, 1, 1, args, 3, bias1_cpu, 1, nullptr);
}

bool ggml_inkling_is_bias1(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || op->type != GGML_TYPE_F32 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != bias1_cpu || p.userdata || p.n_tasks != 1) {
        return false;
    }
    for (int j = 3; j < GGML_MAX_SRC; ++j) {
        if (op->src[j]) {
            return false;
        }
    }
    return layout(op->src[0], op->src[1], op->src[2]) && op->ne[0] == op->src[1]->ne[1] && op->ne[1] == op->src[0]->ne[1] + 1;
}

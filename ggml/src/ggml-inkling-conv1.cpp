#include "ggml-inkling-conv1.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int n_conv = ggml_inkling_conv1_n_conv;

bool f32_block(const ggml_tensor * x, int64_t n) {
    return x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_nelements(x) == n;
}

bool layout(const ggml_tensor * cur, const ggml_tensor * past, const ggml_tensor * weight, const ggml_tensor * cell) {
    if (!cur) {
        return false;
    }
    const int64_t n = ggml_nelements(cur);
    return n > 0 && n % 128 == 0 && f32_block(cur, n) &&
        f32_block(past, (n_conv - 1)*n) && past->ne[0] == n_conv - 1 && past->ne[1] == n &&
        f32_block(weight, n_conv*n) && weight->ne[0] == n_conv && weight->ne[1] == n &&
        f32_block(cell, (n_conv - 1)*n);
}

// reference; same order of operations as SSM_CONV, ADD and SCALE
void conv1_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const int64_t n      = dst->ne[0];
    const float * cur    = static_cast<const float *>(dst->src[0]->data);
    const float * past   = static_cast<const float *>(dst->src[1]->data);
    const float * weight = static_cast<const float *>(dst->src[2]->data);
    float *       cell   = static_cast<float *>(dst->src[3]->data);
    float *       out    = static_cast<float *>(dst->data);
    const float   scale  = 1.0f/sqrtf(n_conv + 1);
    for (int64_t c = 0; c < n; ++c) {
        float x[n_conv];
        for (int k = 0; k < n_conv - 1; ++k) {
            x[k] = past[c*(n_conv - 1) + k];
        }
        x[n_conv - 1] = cur[c];
        float sum = 0.0f;
        for (int k = 0; k < n_conv; ++k) {
            sum += x[k]*weight[c*n_conv + k];
        }
        out[c] = scale*(cur[c] + sum);
        for (int k = 0; k < n_conv - 1; ++k) {
            cell[c*(n_conv - 1) + k] = x[k + 1];
        }
    }
}
} // namespace

bool ggml_inkling_conv1_enabled() {
    static const bool enabled = [] {
        const char * p = std::getenv("GGML_INKLING_CONV1");
        return p && std::strcmp(p, "1") == 0;
    }();
    return enabled;
}

ggml_tensor * ggml_inkling_try_conv1(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * past, ggml_tensor * weight, ggml_tensor * cell) {
    if (!ggml_inkling_conv1_enabled() || !layout(cur, past, weight, cell)) {
        return nullptr;
    }
    ggml_tensor * args[] = { cur, past, weight, cell };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, ggml_nelements(cur), 1, 1, 1, args, 4, conv1_cpu, 1, nullptr);
}

bool ggml_inkling_is_conv1(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || op->type != GGML_TYPE_F32 || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != conv1_cpu || p.userdata || p.n_tasks != 1) {
        return false;
    }
    for (int j = 4; j < GGML_MAX_SRC; ++j) {
        if (op->src[j]) {
            return false;
        }
    }
    return layout(op->src[0], op->src[1], op->src[2], op->src[3]) && ggml_nelements(op->src[0]) == op->ne[0];
}

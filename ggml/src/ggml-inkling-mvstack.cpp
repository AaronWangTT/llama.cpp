#include "ggml-inkling-mvstack.h"
#include "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
// number of stacked rows, 0 if the op can not take these tensors
int64_t layout(const ggml_tensor * x, const ggml_tensor * const * w, int n_w) {
    if (!x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != ggml_nelements(x) || x->ne[0] % QK8_0 != 0 ||
            n_w < 2 || n_w > ggml_inkling_mvstack_max) {
        return 0;
    }
    // the kernel counts the blocks of one matrix and the stacked rows in 32 bits
    const int64_t max_rows = INT32_MAX/(x->ne[0]/QK8_0);
    int64_t n_rows = 0;
    for (int j = 0; j < n_w; ++j) {
        const ggml_tensor * m = w[j];
        if (!m || m->type != GGML_TYPE_Q8_0 || m->op != GGML_OP_NONE || m->view_src || !ggml_is_matrix(m) || !ggml_is_contiguous(m) ||
                m->ne[0] != x->ne[0] || m->ne[1] > max_rows) {
            return 0;
        }
        n_rows += m->ne[1];
    }
    return n_rows <= INT32_MAX ? n_rows : 0;
}

// reference; the input is quantized per block like the q8_1 input of the GPU matvec
void mvstack_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const ggml_tensor * x_t      = dst->src[0];
    const int64_t       n_blocks = x_t->ne[0]/QK8_0;
    const float *       x        = static_cast<const float *>(x_t->data);
    float *             out      = static_cast<float *>(dst->data);

    std::vector<int8_t> q(x_t->ne[0]);
    std::vector<float>  d(n_blocks);
    for (int64_t b = 0; b < n_blocks; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < QK8_0; ++i) {
            amax = fmaxf(amax, fabsf(x[b*QK8_0 + i]));
        }
        const float scale = amax/127.0f;
        for (int i = 0; i < QK8_0; ++i) {
            q[b*QK8_0 + i] = amax == 0.0f ? 0 : (int8_t) roundf(x[b*QK8_0 + i]/scale);
        }
        d[b] = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(scale));
    }

    for (int j = 1; j < GGML_MAX_SRC && dst->src[j]; ++j) {
        const ggml_tensor * m    = dst->src[j];
        const block_q8_0 *  rows = static_cast<const block_q8_0 *>(m->data);
        for (int64_t r = 0; r < m->ne[1]; ++r) {
            float sum = 0.0f;
            for (int64_t b = 0; b < n_blocks; ++b) {
                const block_q8_0 & blk = rows[r*n_blocks + b];
                int sumi = 0;
                for (int i = 0; i < QK8_0; ++i) {
                    sumi += blk.qs[i]*q[b*QK8_0 + i];
                }
                sum += GGML_FP16_TO_FP32(blk.d)*d[b]*(float) sumi;
            }
            *out++ = sum;
        }
    }
}
} // namespace

bool ggml_inkling_mvstack_enabled() {
    static const bool enabled = [] {
        const char * p = std::getenv("GGML_INKLING_MVSTACK");
        return p && std::strcmp(p, "1") == 0;
    }();
    return enabled;
}

ggml_tensor * ggml_inkling_try_mvstack(ggml_context * ctx, ggml_tensor ** w, int n_w, ggml_tensor * x) {
    const int64_t n_rows = ggml_inkling_mvstack_enabled() ? layout(x, w, n_w) : 0;
    if (n_rows == 0) {
        return nullptr;
    }
    ggml_tensor * args[1 + ggml_inkling_mvstack_max] = { x };
    for (int j = 0; j < n_w; ++j) {
        args[j + 1] = w[j];
    }
    return ggml_custom_4d(ctx, GGML_TYPE_F32, n_rows, 1, 1, 1, args, n_w + 1, mvstack_cpu, 1, nullptr);
}

bool ggml_inkling_is_mvstack(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || op->type != GGML_TYPE_F32 || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != mvstack_cpu || p.userdata || p.n_tasks != 1) {
        return false;
    }
    int n_w = 0;
    for (int j = 1; j < GGML_MAX_SRC; ++j) {
        if (op->src[j]) {
            if (n_w != j - 1) {
                return false;
            }
            n_w++;
        }
    }
    const int64_t n_rows = layout(op->src[0], op->src + 1, n_w);
    return n_rows > 0 && n_rows == op->ne[0];
}

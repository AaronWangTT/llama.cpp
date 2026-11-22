#include "ggml-inkling-ctrl1.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int n_embd = ggml_inkling_ctrl1_n_embd;
constexpr int rank   = ggml_inkling_ctrl1_rank;

bool f32_block(const ggml_tensor * x, int64_t n) {
    return x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_nelements(x) == n;
}

// the stock norm kernel reads float4, which needs 16 byte alignment; allocations are aligned, so only the view offset counts
bool norm_input(const ggml_tensor * x) {
    return f32_block(x, n_embd) && x->view_offs % 16 == 0;
}

bool layout(const ggml_tensor * cur, const ggml_tensor * message, const ggml_tensor * down, const ggml_tensor * gain, const ggml_tensor * forget) {
    return norm_input(cur) && norm_input(message) &&
        f32_block(down, 2*n_embd*rank) && down->ne[0] == 2*n_embd && down->ne[1] == rank &&
        f32_block(gain, rank*n_embd) && gain->ne[0] == rank && gain->ne[1] == n_embd &&
        f32_block(forget, rank*n_embd) && forget->ne[0] == rank && forget->ne[1] == n_embd;
}

// reference
void ctrl1_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const float * in[2]  = { static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data) };
    const float * down   = static_cast<const float *>(dst->src[2]->data);
    const float * gain   = static_cast<const float *>(dst->src[3]->data);
    const float * forget = static_cast<const float *>(dst->src[4]->data);
    float *       out    = static_cast<float *>(dst->data);

    float scale[2];
    for (int s = 0; s < 2; ++s) {
        float sum = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            sum += in[s][i]*in[s][i];
        }
        scale[s] = 1.0f/sqrtf(sum/n_embd + ggml_inkling_ctrl1_eps);
    }
    float gate[rank];
    for (int r = 0; r < rank; ++r) {
        float sum = 0.0f;
        for (int s = 0; s < 2; ++s) {
            for (int i = 0; i < n_embd; ++i) {
                sum += down[(2*r + s)*n_embd + i]*(scale[s]*in[s][i]);
            }
        }
        gate[r] = sum;
    }
    for (int i = 0; i < n_embd; ++i) {
        float g = 0.0f;
        float f = 0.0f;
        for (int r = 0; r < rank; ++r) {
            g += gain[i*rank + r]*gate[r];
            f += forget[i*rank + r]*gate[r];
        }
        out[i] = in[0][i] - tanhf(f)*in[0][i] + tanhf(g)*in[1][i];
    }
}
} // namespace

bool ggml_inkling_ctrl1_enabled() {
    static const bool enabled = [] {
        const char * p = std::getenv("GGML_INKLING_CTRL1");
        return p && std::strcmp(p, "1") == 0;
    }();
    return enabled;
}

ggml_tensor * ggml_inkling_try_ctrl1(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * message,
        ggml_tensor * down, ggml_tensor * gain, ggml_tensor * forget) {
    if (!ggml_inkling_ctrl1_enabled() || !layout(cur, message, down, gain, forget)) {
        return nullptr;
    }
    ggml_tensor * args[] = { cur, message, down, gain, forget };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, 1, 1, 1, args, 5, ctrl1_cpu, 1, nullptr);
}

bool ggml_inkling_is_ctrl1(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || op->type != GGML_TYPE_F32 ||
            op->ne[0] != n_embd || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != ctrl1_cpu || p.userdata || p.n_tasks != 1) {
        return false;
    }
    for (int j = 5; j < GGML_MAX_SRC; ++j) {
        if (op->src[j]) {
            return false;
        }
    }
    return layout(op->src[0], op->src[1], op->src[2], op->src[3], op->src[4]);
}

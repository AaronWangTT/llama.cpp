#include "ggml-qwen35-convpf.h"
#include "ggml-kperf.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int n_conv = ggml_qwen35_convpf_n_conv;

bool layout(const ggml_tensor * cur, const ggml_tensor * past, const ggml_tensor * weight) {
    if (!cur || !past || !weight || cur->type != GGML_TYPE_F32 || past->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t n_channels = cur->ne[0];
    const int64_t n_tokens   = cur->ne[1];
    return n_channels > 0 && n_channels % 128 == 0 && n_tokens >= ggml_qwen35_convpf_min_n_tokens && cur->ne[2] == 1 && cur->ne[3] == 1 &&
        cur->nb[0] == sizeof(float) && cur->nb[1] >= (size_t) n_channels*sizeof(float) && cur->nb[1] % sizeof(float) == 0 &&
        ggml_is_contiguous(past) && ggml_nelements(past) == (n_conv - 1)*n_channels && past->ne[0] == n_conv - 1 && past->ne[1] == n_channels &&
        ggml_is_contiguous(weight) && ggml_nelements(weight) == n_conv*n_channels && weight->ne[0] == n_conv && weight->ne[1] == n_channels;
}

// reference; same order of operations as SSM_CONV and SILU
void convpf_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const ggml_tensor * cur_t = dst->src[0];
    const int64_t n_channels  = dst->ne[0];
    const int64_t n_tokens    = dst->ne[1];
    const size_t  stride      = cur_t->nb[1]/sizeof(float);
    const float * cur    = static_cast<const float *>(cur_t->data);
    const float * past   = static_cast<const float *>(dst->src[1]->data);
    const float * weight = static_cast<const float *>(dst->src[2]->data);
    float *       out    = static_cast<float *>(dst->data);
    for (int64_t c = 0; c < n_channels; ++c) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            float sum = 0.0f;
            for (int j = 0; j < n_conv; ++j) {
                const int64_t tj = t + j - (n_conv - 1);
                const float   x  = tj < 0 ? past[c*(n_conv - 1) + tj + (n_conv - 1)] : cur[tj*stride + c];
                sum += x*weight[c*n_conv + j];
            }
            out[t*n_channels + c] = sum/(1.0f + expf(-sum));
        }
    }
}
} // namespace

bool ggml_qwen35_convpf_enabled() {
    return ggml_kperf_on(GGML_KPERF_QWEN35_CONVPF);
}

bool ggml_q35moe_convpf_enabled() {
    return ggml_kperf_on(GGML_KPERF_QWEN35MOE_CONVPF);
}

bool ggml_q35moe_convpfrs_enabled() {
    return ggml_kperf_on(GGML_KPERF_QWEN35MOE_CONVPFRS);
}

ggml_tensor * ggml_qwen35_try_convpf(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * past, ggml_tensor * weight) {
    // each graph checks its own switch before it asks for the op
    if ((!ggml_qwen35_convpf_enabled() && !ggml_q35moe_convpf_enabled()) || !layout(cur, past, weight)) {
        return nullptr;
    }
    ggml_tensor * args[] = { cur, past, weight };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1], 1, 1, args, 3, convpf_cpu, 1, nullptr);
}

bool ggml_qwen35_is_convpf(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || op->type != GGML_TYPE_F32 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != convpf_cpu || p.userdata || p.n_tasks != 1) {
        return false;
    }
    for (int j = 3; j < GGML_MAX_SRC; ++j) {
        if (op->src[j]) {
            return false;
        }
    }
    return layout(op->src[0], op->src[1], op->src[2]) && op->src[0]->ne[0] == op->ne[0] && op->src[0]->ne[1] == op->ne[1];
}

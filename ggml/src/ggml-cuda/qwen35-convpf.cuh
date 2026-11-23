#pragma once
#include "ggml-qwen35-convpf.h"
#include "unary.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private CONVPF kernel for gfx1103 only"
#endif

// One thread per channel, one block per 128 channels and 32 tokens. The inputs come straight from cur and past, there is no concat tensor.
// The arithmetic follows ssm_conv_long_token_f32<true, 128, 4, 32> step by step.
static __global__ void qwen35_convpf(const float * cur_ptr, const float * past_ptr, const float * weight_ptr, float * dst_ptr,
        const int stride_cur, const int n_channels, const int n_t) {
    constexpr int n_conv    = ggml_qwen35_convpf_n_conv;
    constexpr int split_n_t = 32;
    const float * GGML_CUDA_RESTRICT cur    = cur_ptr;
    const float * GGML_CUDA_RESTRICT past   = past_ptr;
    const float * GGML_CUDA_RESTRICT weight = weight_ptr;
    float       * GGML_CUDA_RESTRICT dst    = dst_ptr;

    const int c  = blockIdx.x*blockDim.x + threadIdx.x;
    const int t0 = blockIdx.y*split_n_t;

    float w[n_conv];
#pragma unroll
    for (int j = 0; j < n_conv; j++) {
        w[j] = weight[c*n_conv + j];
    }

    // The inputs of the block: the n_conv - 1 tokens before it and its own tokens.
    float x[n_conv - 1 + split_n_t];
#pragma unroll
    for (int i = 0; i < n_conv - 1 + split_n_t; i++) {
        const int t = t0 + i - (n_conv - 1);
        if (t < 0) {
            x[i] = past[c*(n_conv - 1) + i];
        } else if (t < n_t) {
            x[i] = cur[int64_t(t)*stride_cur + c];
        } else {
            x[i] = 0.0f;
        }
    }

    const float b = 0.0f;
#pragma unroll
    for (int i = 0; i < split_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < n_conv; j++) {
            sumf += x[i + j] * w[j];
        }
        sumf += b;
        if (t0 + i < n_t) {
            dst[int64_t(t0 + i)*n_channels + c] = ggml_cuda_op_silu_single(sumf);
        }
    }
}

static bool qwen35_convpf_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    return info.cc == GGML_CUDA_CC_OFFSET_AMD + 0x1103 && info.warp_size == 32 && (ggml_qwen35_convpf_enabled() || ggml_q35moe_convpf_enabled()) &&
        ggml_qwen35_is_convpf(op);
}

static void qwen35_convpf_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const int n_channels = op->ne[0];
    const int n_t        = op->ne[1];
    const dim3 blocks(n_channels/128, (n_t + 31)/32, 1);
    qwen35_convpf<<<blocks, 128, 0, ctx.stream()>>>(
        static_cast<const float *>(op->src[0]->data), static_cast<const float *>(op->src[1]->data), static_cast<const float *>(op->src[2]->data),
        static_cast<float *>(op->data), (int) (op->src[0]->nb[1]/sizeof(float)), n_channels, n_t);
    CUDA_CHECK(cudaGetLastError());
}

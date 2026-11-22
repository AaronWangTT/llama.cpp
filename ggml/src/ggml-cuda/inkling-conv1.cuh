#pragma once
#include "ggml-inkling-conv1.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private CONV1 experiment for gfx1103 only"
#endif

// one thread per channel; the arithmetic follows ssm_conv_f32, op_add and scale_f32 step by step
static __global__ void inkling_conv1(const float * cur_ptr, const float * past_ptr, const float * weight_ptr, float * cell_ptr, float * dst_ptr,
        const float scale, const float bias) {
    constexpr int n_conv = ggml_inkling_conv1_n_conv;
    const float * GGML_CUDA_RESTRICT cur    = cur_ptr;
    const float * GGML_CUDA_RESTRICT past   = past_ptr;
    const float * GGML_CUDA_RESTRICT weight = weight_ptr;
    float       * GGML_CUDA_RESTRICT cell   = cell_ptr;
    float       * GGML_CUDA_RESTRICT dst    = dst_ptr;

    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    float x[n_conv];
    float w[n_conv];
#pragma unroll
    for (int j = 0; j < n_conv; j++) {
        w[j] = weight[c*n_conv + j];
    }
#pragma unroll
    for (int j = 0; j < n_conv - 1; j++) {
        x[j] = past[c*(n_conv - 1) + j];
    }
    x[n_conv - 1] = cur[c];

    const float b = 0.0f;
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < n_conv; j++) {
        sumf += x[j] * w[j];
    }
    sumf += b;

    const float sum = x[n_conv - 1] + sumf;
    dst[c] = scale * sum + bias;

#pragma unroll
    for (int j = 0; j < n_conv - 1; j++) {
        cell[c*(n_conv - 1) + j] = x[j + 1];
    }
}

static bool inkling_conv1_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    return info.cc == GGML_CUDA_CC_OFFSET_AMD + 0x1103 && info.warp_size == 32 && ggml_inkling_conv1_enabled() && ggml_inkling_is_conv1(op);
}

static void inkling_conv1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const int n = op->ne[0];
    inkling_conv1<<<n/128, 128, 0, ctx.stream()>>>(
        static_cast<const float *>(op->src[0]->data), static_cast<const float *>(op->src[1]->data), static_cast<const float *>(op->src[2]->data),
        static_cast<float *>(op->src[3]->data), static_cast<float *>(op->data), 1.0f/sqrtf(ggml_inkling_conv1_n_conv + 1), 0.0f);
}

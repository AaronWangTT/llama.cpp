#pragma once
#include "ggml-inkling-bias1.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "inkling.bias1: compile this kernel for gfx1103 only"
#endif

// one thread per output element out[h, d]
static __global__ void inkling_bias1(const float * bias, const float * rcur, const float * sinks, float * dst,
        const int n_rel, const int n_extent, const int n_head) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (n_extent + 1)*n_head) {
        return;
    }
    const int d = i / n_head;
    const int h = i - d*n_head;

    float acc = 0.0f;
    if (d < n_extent) {
        for (int r = 0; r < n_rel; ++r) {
            acc = fmaf(bias[d*n_rel + r], rcur[h*n_rel + r], acc);
        }
    }
    dst[i] = acc - sinks[h];
}

static bool inkling_bias1_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    return info.cc == GGML_CUDA_CC_OFFSET_AMD + 0x1103 && info.warp_size == 32 && ggml_inkling_bias1_enabled() && ggml_inkling_is_bias1(op);
}

static void inkling_bias1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const int n_rel = op->src[0]->ne[0], n_extent = op->src[0]->ne[1], n_head = op->ne[0];
    const dim3 grid(((n_extent + 1)*n_head + 255) / 256);
    const float * bias  = static_cast<const float *>(op->src[0]->data);
    const float * rcur  = static_cast<const float *>(op->src[1]->data);
    const float * sinks = static_cast<const float *>(op->src[2]->data);
    inkling_bias1<<<grid, 256, 0, ctx.stream()>>>(bias, rcur, sinks, static_cast<float *>(op->data), n_rel, n_extent, n_head);
}

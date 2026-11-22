#pragma once
#include "ggml-inkling-concat4.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private CONCAT4 experiment for gfx1103 only"
#endif

static __global__ void inkling_concat4_pack(const float * a, const float * b, const float * c, const float * d, float * dst) {
    constexpr int D = 2304;
    const int s = blockIdx.y, t = blockIdx.z, i = blockIdx.x*256 + threadIdx.x;
    const float * src = s == 0 ? a : s == 1 ? b : s == 2 ? c : d;
    __builtin_memcpy(dst + (4*t + s)*D + i, src + t*D + i, sizeof(float));
}

static bool inkling_concat4_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    if (info.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || info.warp_size != 32 ||
            !ggml_inkling_concat4_enabled() || !ggml_inkling_is_concat4(op)) {
        return false;
    }
    // Null data is normal during placement. A view's byte offset is checked by is_concat4.
    for (int j = 0; j < 4; ++j) {
        if (uintptr_t(op->src[j]->data) % alignof(float)) {
            return false;
        }
    }
    return uintptr_t(op->data) % alignof(float) == 0;
}

static void inkling_concat4_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    auto src = [&](int s) { return static_cast<const float *>(op->src[s]->data); };
    inkling_concat4_pack<<<dim3(9, 4, 256), 256, 0, ctx.stream()>>>(src(0), src(1), src(2), src(3), static_cast<float *>(op->data));
}

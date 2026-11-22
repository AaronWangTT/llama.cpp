#pragma once
#include "ggml-inkling-depthmix.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private MMVF experiment for gfx1103 only"
#endif

static __device__ __forceinline__ float depthmix_add(float a, float b) {
    float out;
    asm("v_add_f32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

static __device__ __forceinline__ float depthmix_mad(float z, float a, float b) {
    asm("v_fmac_f32_e32 %0, %1, %2" : "+v"(z) : "v"(a), "v"(b));
    return z;
}

template<int S, int Q>
static __global__ void inkling_depthmix_cont(const float * __restrict__ packed, const float * __restrict__ scores, float * __restrict__ out) {
    constexpr int D = 2304;
    const int d = blockIdx.x * 128 + threadIdx.x, t = blockIdx.y;
    using pack = std::conditional_t<S == 2, float2, float4>;
    const pack v = reinterpret_cast<const pack *>(packed)[t*D + d];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float * a = scores + (t*Q + q)*S;
        float even = depthmix_mad(depthmix_mad(0.0f, v.x, a[0]), v.y, a[1]);
        float odd = 0.0f;
        if constexpr (S == 4) {
            odd = depthmix_mad(depthmix_mad(0.0f, v.z, a[2]), v.w, a[3]);
            odd = depthmix_add(odd, 0.0f);
        }
        // The empty-lane steps canonicalize negative zero before the offset-1 XOR.
        even = depthmix_add(even, 0.0f);
        out[(t*Q + q)*D + d] = (q & 1) ? depthmix_add(odd, even) : depthmix_add(even, odd);
    }
}

static bool depthmix_layout(const ggml_tensor * x, int a, int b, int c) {
    return x && x->type == GGML_TYPE_F32 && !x->view_src && x->ne[0] == a && x->ne[1] == b &&
        x->ne[2] == c && x->ne[3] == 1 && x->nb[0] == 4 && x->nb[1] == 4u*a &&
        x->nb[2] == 4u*a*b && x->nb[3] == 4u*a*b*c;
}

static bool depthmix_mm(const ggml_tensor * x) {
    if (!x || x->op != GGML_OP_MUL_MAT || x->op_params[1] != GGML_HINT_NONE || x->src[2]) {
        return false;
    }
    auto f32 = [](ggml_prec p) { return p == GGML_PREC_UNDEFINED || p == GGML_PREC_F32; };
    return f32(ggml_prec_get_acc(x)) && f32(ggml_prec_get_src(x, 1));
}

static bool inkling_depthmix_try(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, ggml_tensor * y) {
    constexpr int D = 2304, T = 256;
    if ((x->ne[0] != 2 && x->ne[0] != 4) || (w->ne[1] != 1 && w->ne[1] != 3 && w->ne[1] != 4)) {
        return false;
    }
    const int s = x->ne[0], q = w->ne[1];
    if (!depthmix_mm(y) || y->src[0] != x || y->src[1] != w || !depthmix_layout(y, D, q, T) ||
            !depthmix_layout(x, s, D, T) || x->op != GGML_OP_CONT || !depthmix_layout(w, s, q, T) || w->op != GGML_OP_SOFT_MAX ||
            w->src[1] || w->src[2] || ggml_get_op_params_f32(w, 0) != 1.0f/48 || ggml_get_op_params_f32(w, 1) != 0.0f) {
        return false;
    }
    const auto * tr = x->src[0];
    const auto * n = tr ? tr->src[0] : nullptr;
    if (!depthmix_layout(n, D, s, T) || n->op != GGML_OP_RMS_NORM || ggml_get_op_params_f32(n, 0) != 1.e-6f ||
            !depthmix_layout(n->src[0], D, s, T) || n->src[0]->op != GGML_OP_CONCAT || ggml_get_op_params_i32(n->src[0], 0) != 1 ||
            tr->op != GGML_OP_TRANSPOSE || tr->type != GGML_TYPE_F32 || tr->view_src != n || tr->view_offs ||
            tr->ne[0] != s || tr->ne[1] != D || tr->ne[2] != T || tr->ne[3] != 1 ||
            tr->nb[0] != n->nb[1] || tr->nb[1] != n->nb[0] || tr->nb[2] != n->nb[2] || tr->nb[3] != n->nb[3]) {
        return false;
    }
    const auto * dot = w->src[0];
    if (!depthmix_mm(dot) || dot->src[0] != n || !depthmix_layout(dot, s, q, T)) {
        return false;
    }
    const auto * rep = dot->src[1];
    if (!depthmix_layout(rep, D, q, T) || rep->op != GGML_OP_REPEAT || !depthmix_layout(rep->src[0], D, q, 1) || rep->src[0]->op != GGML_OP_NONE) {
        return false;
    }
    // Ancestors above are metadata only. Pass only live MUL_MAT operands to the device.
    const size_t alignment = s == 2 ? alignof(float2) : alignof(float4);
    if (!x->data || !w->data || !y->data || uintptr_t(x->data) % alignment || uintptr_t(w->data) % 4 || uintptr_t(y->data) % 4) {
        return false;
    }
    auto launch = [&](auto S, auto Q) {
        inkling_depthmix_cont<S.value, Q.value><<<dim3(D/128, T), 128, 0, ctx.stream()>>>(
                static_cast<const float *>(x->data), static_cast<const float *>(w->data), static_cast<float *>(y->data));
    };
    auto for_s = [&](auto S) {
        if (q == 1) { launch(S, std::integral_constant<int, 1>{}); }
        if (q == 3) { launch(S, std::integral_constant<int, 3>{}); }
        if (q == 4) { launch(S, std::integral_constant<int, 4>{}); }
    };
    if (s == 2) { for_s(std::integral_constant<int, 2>{}); }
    else        { for_s(std::integral_constant<int, 4>{}); }
    return true;
}

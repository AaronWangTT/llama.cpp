#pragma once
#include "ggml-inkling-ctrl1.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private CTRL1 experiment for gfx1103 only"
#endif

// weights col2 and col2 + 1 of a row; a BF16 weight is the high half of the F32 value
template <typename T>
static __device__ __forceinline__ float2 inkling_ctrl1_weight2(const T * row, const int col2) {
    if constexpr (std::is_same_v<T, float>) {
        return ((const float2 *) row)[col2];
    } else {
        static_assert(std::is_same_v<T, nv_bfloat16>, "unsupported type");
        const uint32_t pair = ((const uint32_t *) row)[col2];
        return make_float2(__uint_as_float(pair << 16), __uint_as_float(pair & 0xffff0000u));
    }
}

// one block of 1024 threads for four rows of down
// the input scales follow rms_norm_f32<1024, 1, true>, each quarter of the block follows mul_mat_vec_f<float, float, 1, 256>
template <typename T>
static __global__ void inkling_ctrl1_gate(const float * cur_ptr, const float * msg_ptr, const T * down_ptr, float * gate_ptr,
        const int ncols, const float eps) {
    constexpr int norm_size = 1024; // block size of the stock norm kernel
    constexpr int mmv_size  = 256;  // block size of the stock product kernel
    constexpr int mmv_warps = mmv_size/WARP_SIZE;
    const float * GGML_CUDA_RESTRICT cur  = cur_ptr;
    const float * GGML_CUDA_RESTRICT msg  = msg_ptr;
    const T     * GGML_CUDA_RESTRICT down = down_ptr;
    float       * GGML_CUDA_RESTRICT gate = gate_ptr;

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int part = tid / mmv_size; // the row of this block
    const int vtid = tid % mmv_size; // the thread of the stock product kernel
    const int row  = blockIdx.x*(norm_size/mmv_size) + part;

    __shared__ float s_sum[2][WARP_SIZE];
    __shared__ float buf_iw[WARP_SIZE];

    const float * in[2]  = { cur, msg };
    const int     ncols4 = ncols / 4;

#pragma unroll
    for (int s = 0; s < 2; ++s) {
        float tmp = 0.0f;
        if (tid < ncols4) {
            const float4 xs = ((const float4 *) in[s])[tid];
            tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
        }
        tmp = warp_reduce_sum(tmp);
        if (lane == 0) {
            s_sum[s][warp] = tmp;
        }
    }
    __syncthreads();

    float scale[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const float tmp = warp_reduce_sum(s_sum[s][lane]);
        scale[s] = rsqrtf(tmp / ncols + eps);
    }

    // the columns of the stock concat: scaled cur, then scaled message
    const int ncols2 = ncols;
    const T * x      = down + (int64_t) row*2*ncols;

    float sumf = 0.0f;
    for (int col2 = vtid; col2 < ncols2; col2 += mmv_size) {
        const int    s    = col2 < ncols2/2 ? 0 : 1;
        const float2 raw  = ((const float2 *) in[s])[col2 - s*(ncols2/2)];
        const float2 tmpx = inkling_ctrl1_weight2(x, col2);
        const float2 tmpy = make_float2(scale[s]*raw.x, scale[s]*raw.y);
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }

    sumf = warp_reduce_sum(sumf);
    if (lane == 0) {
        buf_iw[warp] = sumf;
    }
    __syncthreads();
    if (vtid >= WARP_SIZE) {
        return;
    }
    // the stock kernel adds 32 slots, all but the first 8 are zero
    sumf = vtid < mmv_warps ? buf_iw[part*mmv_warps + vtid] : 0.0f;
    sumf = warp_reduce_sum(sumf);
    if (vtid == 0) {
        gate[row] = sumf;
    }
}

// one block of 32 threads per channel
// both products follow mul_mat_vec_f<float, float, 1, 32>, the tail follows inkling_control_three (forget, then gain)
template <typename T>
static __global__ void inkling_ctrl1_tail(const float * cur_ptr, const float * msg_ptr, const T * gain_ptr, const T * forget_ptr,
        const float * gate_ptr, float * dst_ptr, const int ncols2) {
    const float * GGML_CUDA_RESTRICT cur    = cur_ptr;
    const float * GGML_CUDA_RESTRICT msg    = msg_ptr;
    const T     * GGML_CUDA_RESTRICT gain   = gain_ptr;
    const T     * GGML_CUDA_RESTRICT forget = forget_ptr;
    const float * GGML_CUDA_RESTRICT gate   = gate_ptr;
    float       * GGML_CUDA_RESTRICT dst    = dst_ptr;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float2 * y2 = (const float2 *) gate;
    const T      * f  = forget + (int64_t) row*2*ncols2;
    const T      * g  = gain   + (int64_t) row*2*ncols2;

    float sumf = 0.0f;
    float sumg = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += WARP_SIZE) {
        const float2 tmpy = y2[col2];
        const float2 tmpf = inkling_ctrl1_weight2(f, col2);
        const float2 tmpg = inkling_ctrl1_weight2(g, col2);
        ggml_cuda_mad(sumf, tmpf.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpf.y, tmpy.y);
        ggml_cuda_mad(sumg, tmpg.x, tmpy.x);
        ggml_cuda_mad(sumg, tmpg.y, tmpy.y);
    }
    sumf = warp_reduce_sum(sumf);
    sumg = warp_reduce_sum(sumg);
    if (tid != 0) {
        return;
    }

    const float x = cur[row];
    const float m = msg[row];
    const float tf = tanhf(sumf);
    const float tg = tanhf(sumg);
    float prod, base, y;
    // explicit instructions as in inkling_control_three: no contraction, src[0] before src[1]
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(prod) : "v"(tf), "v"(x));
    asm("v_sub_f32_e32 %0, %1, %2" : "=v"(base) : "v"(x), "v"(prod));
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(prod) : "v"(tg), "v"(m));
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(y) : "v"(base), "v"(prod));
    dst[row] = y;
}

static bool inkling_ctrl1_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    return info.cc == GGML_CUDA_CC_OFFSET_AMD + 0x1103 && info.warp_size == 32 && ggml_inkling_ctrl1_enabled() && ggml_inkling_is_ctrl1(op);
}

static void inkling_ctrl1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    constexpr int n_embd = ggml_inkling_ctrl1_n_embd;
    constexpr int rank   = ggml_inkling_ctrl1_rank;
    const float * cur    = static_cast<const float *>(op->src[0]->data);
    const float * msg    = static_cast<const float *>(op->src[1]->data);
    const void  * down   = op->src[2]->data;
    const void  * gain   = op->src[3]->data;
    const void  * forget = op->src[4]->data;
    // the stock norm kernel sums in float4 order for aligned inputs only
    GGML_ASSERT((uintptr_t) cur % 16 == 0 && (uintptr_t) msg % 16 == 0);
    // the kernels load two weights as one word
    GGML_ASSERT((uintptr_t) down % 4 == 0 && (uintptr_t) gain % 4 == 0 && (uintptr_t) forget % 4 == 0);

    ggml_cuda_pool_alloc<float> gate(ctx.pool(), rank);
    if (op->src[2]->type == GGML_TYPE_BF16) {
        inkling_ctrl1_gate<nv_bfloat16><<<rank/4, 1024, 0, ctx.stream()>>>(cur, msg, static_cast<const nv_bfloat16 *>(down), gate.get(), n_embd, ggml_inkling_ctrl1_eps);
    } else {
        inkling_ctrl1_gate<float><<<rank/4, 1024, 0, ctx.stream()>>>(cur, msg, static_cast<const float *>(down), gate.get(), n_embd, ggml_inkling_ctrl1_eps);
    }
    float * dst = static_cast<float *>(op->data);
    if (op->src[3]->type == GGML_TYPE_BF16) {
        inkling_ctrl1_tail<nv_bfloat16><<<n_embd, 32, 0, ctx.stream()>>>(cur, msg, static_cast<const nv_bfloat16 *>(gain), static_cast<const nv_bfloat16 *>(forget), gate.get(), dst, rank/2);
    } else {
        inkling_ctrl1_tail<float><<<n_embd, 32, 0, ctx.stream()>>>(cur, msg, static_cast<const float *>(gain), static_cast<const float *>(forget), gate.get(), dst, rank/2);
    }
}

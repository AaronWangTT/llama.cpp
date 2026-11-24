#pragma once
#include "ggml-q35moe-ffn.h"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cfloat>
#include <type_traits>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private one-token op for gfx1103 only"
#endif

// One token of the feed-forward in five launches. Every value follows the stock kernel of its node step by step:
//   router, gate of the shared expert   mul_mat_vec_f<float, float, 1, 256>, op_sigmoid
//   quantized vectors                   quantize_q8_1
//   top-k                               topk_moe_cuda<256, false> with softmax and weight norm
//   quantized products                  mul_mat_vec_q<type, 1> with the RDNA2 parameters (one warp per row), SWIGLU as in its fused form
//   sum                                 moe_weighted_reduction_f32, MUL, ADD of three values
// A launch holds the parts that wait for the same inputs.

constexpr int q35moe_n_embd   = ggml_q35moe_n_embd;
constexpr int q35moe_n_ff     = ggml_q35moe_n_ff;
constexpr int q35moe_n_expert = ggml_q35moe_n_expert;
constexpr int q35moe_n_used   = ggml_q35moe_n_used;
constexpr int q35moe_warp     = 32;
constexpr int q35moe_vdr      = 2;

// routed experts per warp in the last launch: 4 in each of 2 warps is the fastest layout on gfx1103 (8 in 1 warp: +7%, 2 in 4: the same, 1 in 8: +25%)
constexpr int q35moe_down_per_warp = 4;
constexpr int q35moe_down_warps    = q35moe_n_used / q35moe_down_per_warp;

static_assert(VDR_Q8_0_Q8_1_MMVQ == q35moe_vdr && VDR_Q4_K_Q8_1_MMVQ == q35moe_vdr && VDR_Q5_K_Q8_1_MMVQ == q35moe_vdr &&
              VDR_Q6_K_Q8_1_MMVQ == q35moe_vdr, "the four weight types share one thread layout");
static_assert(q35moe_n_expert % WARP_SIZE == 0 && q35moe_n_used <= WARP_SIZE && WARP_SIZE == q35moe_warp, "top-k layout");

typedef float (*q35moe_vec_dot_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ q35moe_vec_dot_t q35moe_vec_dot(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q4_K: return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K: return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K: return vec_dot_q6_K_q8_1;
        default:             return nullptr;
    }
}

// warp sums of n products as mul_mat_vec_q<type, 1> makes them: row i starts at block kbx0[i] of vx[i] and takes the ncols quantized values y[i]
template <ggml_type type, int ncols, int n>
static __device__ __forceinline__ void q35moe_dots(const void * const * vx, const block_q8_1 * const * y, const int * kbx0, float * tmp) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = q35moe_vdr;

    constexpr q35moe_vec_dot_t vec_dot = q35moe_vec_dot(type);

    const     int tid = threadIdx.x;
    constexpr int blocks_per_row_x = ncols / qk;
    constexpr int blocks_per_iter  = vdr * q35moe_warp / qi;

#pragma unroll
    for (int i = 0; i < n; ++i) {
        tmp[i] = 0.0f;
    }

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int i = 0; i < n; ++i) {
            tmp[i] += vec_dot(vx[i], &y[i][kby], kbx0[i] + kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < n; ++i) {
        tmp[i] = warp_reduce_sum<q35moe_warp>(tmp[i]);
    }
}

// row kbx0 of up and gate times y, with SWIGLU; lane 0 stores the value
template <ggml_type type>
static __device__ __forceinline__ void q35moe_glu_row(const void * vup, const void * vgate, const block_q8_1 * y, const int kbx0, float * dst) {
    const void *       vx[2]  = { vup, vgate };
    const block_q8_1 * vy[2]  = { y, y };
    const int          kbx[2] = { kbx0, kbx0 };

    float tmp[2];
    q35moe_dots<type, q35moe_n_embd, 2>(vx, vy, kbx, tmp);

    if (threadIdx.x == 0) {
        // the stock kernel adds a zero bias to both sums: no effect, a sum that starts at +0 is never -0
        float result = tmp[0];
        result *= ggml_cuda_op_silu_single(tmp[1]);
        dst[0] = result;
    }
}

// one warp writes the q8_1 block ib of x
static __device__ __forceinline__ void q35moe_quantize(const float * x, block_q8_1 * y, const int ib) {
    const int   iqs = threadIdx.x % QK8_1;
    const float xi  = x[ib*QK8_1 + iqs];
    float amax = fabsf(xi);

    amax = warp_reduce_max<QK8_1>(amax);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    const int sum = warp_reduce_sum<QK8_1>((int) q);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, d * sum);
}

// one warp: softmax of the logits, the n_used largest in order, their weights divided by their sum
static __device__ __forceinline__ void q35moe_topk(const float * logits, int32_t * ids, float * weights) {
    constexpr int n_experts          = q35moe_n_expert;
    constexpr int n_expert_used      = q35moe_n_used;
    constexpr int experts_per_thread = n_experts / WARP_SIZE;

    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = logits[expert];
    }

    {
        float max_val = -INFINITY;

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            max_val = max(max_val, wt[i]);
        }

        max_val = warp_reduce_max(max_val);

        float sum = 0.f;

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            const float val = expf(wt[i] - max_val);
            wt[i]           = val;
            sum += val;
        }

        sum = warp_reduce_sum(sum);

        const float inv_sum = 1.0f / sum;

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            wt[i] *= inv_sum;
        }
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float wt_sum = 0.f;

    // the n_used weights are in the first lanes
    float output_weight = 0.f;

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * WARP_SIZE;
            if (wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }

        if (k == (int) threadIdx.x) {
            output_weight = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            wt_sum += max_val;
        }
    }

    wt_sum              = warp_reduce_sum(wt_sum);
    wt_sum              = max(wt_sum, ggml_q35moe_sum_min);
    const float inv_sum = 1.0f / wt_sum;

    output_weight *= inv_sum;

    if (threadIdx.x < n_expert_used) {
        weights[threadIdx.x] = output_weight;
    }
}

// launch 1, blocks of 256 threads: n_expert router rows, the row of the shared expert gate, then n_embd/256 blocks that quantize x
__launch_bounds__(256, 1)
static __global__ void q35moe_ffn_router(
        const float * w_router, const float * w_gate, const float * y_ptr, float * logits, float * gate, block_q8_1 * xq) {
    constexpr int warp_size  = q35moe_warp;
    constexpr int block_size = 256;
    constexpr int ncols2     = q35moe_n_embd / 2;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row > q35moe_n_expert) {
        q35moe_quantize(y_ptr, xq, (block_size*(row - q35moe_n_expert - 1) + tid) / QK8_1);
        return;
    }

    const float * GGML_CUDA_RESTRICT x = row < q35moe_n_expert ? w_router + row*q35moe_n_embd : w_gate;
    const float * GGML_CUDA_RESTRICT y = y_ptr;

    __shared__ float buf_iw[warp_size];

    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf = 0.0f;

    const float2 * y2 = (const float2 *) y;
    const float2 * x2 = (const float2 *) x;

    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }

    sumf = warp_reduce_sum<warp_size>(sumf);

    buf_iw[tid/warp_size] = sumf;
    __syncthreads();
    if (tid < warp_size) {
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<warp_size>(sumf);
    }

    if (tid >= 1) {
        return;
    }

    if (row < q35moe_n_expert) {
        logits[row] = sumf;
    } else {
        gate[0] = 1.0f / (1.0f + expf(-sumf));
    }
}

// launch 2: one warp per row of the shared expert gate/up, the last block is the top-k
template <ggml_type type>
__launch_bounds__(q35moe_warp, 1)
static __global__ void q35moe_ffn_shexp_up(
        const void * vup, const void * vgate, const block_q8_1 * xq, const float * logits,
        float * h, int32_t * ids, float * weights) {
    constexpr int blocks_per_row = q35moe_n_embd / ggml_cuda_type_traits<type>::qk;

    if (blockIdx.x == q35moe_n_ff) {
        q35moe_topk(logits, ids, weights);
        return;
    }

    q35moe_glu_row<type>(vup, vgate, xq, blockIdx.x*blocks_per_row, h + blockIdx.x);
}

// launch 3, grid (n_ff + 2, n_used): one warp per row of a routed expert gate/up
// the last two columns quantize the hidden vector of the shared expert
template <ggml_type type>
__launch_bounds__(q35moe_warp, 1)
static __global__ void q35moe_ffn_exps_up(
        const void * vup, const void * vgate, const block_q8_1 * xq, const int32_t * ids,
        const float * h_shexp, block_q8_1 * hq_shexp, float * h) {
    constexpr int blocks_per_row = q35moe_n_embd / ggml_cuda_type_traits<type>::qk;
    static_assert(2*q35moe_n_used == q35moe_n_ff / QK8_1, "two columns hold the blocks of the shared hidden vector");

    const int row = blockIdx.x;
    const int k   = blockIdx.y;

    if (row >= q35moe_n_ff) {
        q35moe_quantize(h_shexp, hq_shexp, (row - q35moe_n_ff)*q35moe_n_used + k);
        return;
    }

    q35moe_glu_row<type>(vup, vgate, xq, (ids[k]*q35moe_n_ff + row)*blocks_per_row, h + k*q35moe_n_ff + row);
}

// launch 4: one warp per row of the shared expert down, the last blocks quantize the hidden vectors of the routed experts
template <ggml_type type>
__launch_bounds__(q35moe_warp, 1)
static __global__ void q35moe_ffn_shexp_down(
        const void * vdown, const block_q8_1 * hq_shexp, const float * h_exps, block_q8_1 * hq_exps, float * dst) {
    constexpr int blocks_per_row = q35moe_n_ff / ggml_cuda_type_traits<type>::qk;

    const int row = blockIdx.x;

    if (row >= q35moe_n_embd) {
        q35moe_quantize(h_exps, hq_exps, row - q35moe_n_embd);
        return;
    }

    const int kbx0 = row*blocks_per_row;

    float tmp;
    q35moe_dots<type, q35moe_n_ff, 1>(&vdown, &hq_shexp, &kbx0, &tmp);

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

// launch 5, one block per output value: the routed experts down, then the weighted sum, the gated shared expert and the residual
template <ggml_type type>
__launch_bounds__(q35moe_warp * q35moe_down_warps, 1)
static __global__ void q35moe_ffn_exps_down(
        const void * vdown, const block_q8_1 * hq, const int32_t * ids, const float * weights,
        const float * d_shexp, const float * gate, const float * res, float * dst) {
    constexpr int n_per_warp     = q35moe_down_per_warp;
    constexpr int blocks_per_row = q35moe_n_ff / ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_y   = q35moe_n_ff / QK8_1;

    const int row = blockIdx.x;

    const void *       vx[n_per_warp];
    const block_q8_1 * vy[n_per_warp];
    int                kbx[n_per_warp];
#pragma unroll
    for (int i = 0; i < n_per_warp; ++i) {
        const int k = threadIdx.y*n_per_warp + i;
        vx[i]  = vdown;
        vy[i]  = hq + k*blocks_per_y;
        kbx[i] = (ids[k]*q35moe_n_embd + row)*blocks_per_row;
    }

    float tmp[n_per_warp];
    q35moe_dots<type, q35moe_n_ff, n_per_warp>(vx, vy, kbx, tmp);

    __shared__ float experts[q35moe_n_used];

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < n_per_warp; ++i) {
            experts[threadIdx.y*n_per_warp + i] = tmp[i];
        }
    }
    __syncthreads();

    if (threadIdx.x != 0 || threadIdx.y != 0) {
        return;
    }

    float sum = experts[0] * weights[0];
#pragma unroll
    for (int k = 1; k < q35moe_n_used; ++k) {
        sum = fmaf(experts[k], weights[k], sum);
    }

    const float shexp = d_shexp[row];
    const float g     = gate[0];
    const float r     = res[row];
    float prod, part, out;
    // explicit instructions, the stock nodes are a MUL and two ADD: no contraction
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(prod) : "v"(shexp), "v"(g));
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(part) : "v"(sum), "v"(prod));
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(out) : "v"(part), "v"(r));
    dst[row] = out;
}

static bool q35moe_ffn1_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    if (info.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || info.warp_size != q35moe_warp || !ggml_q35moe_ffn1_enabled() || !ggml_q35moe_is_ffn1(op)) {
        return false;
    }
    // weights in memory of this device only: the scheduler gives an op with a weight in host memory to the CPU backend
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);
    for (int j = GGML_Q35MOE_FFN1_GATE_INP; j < GGML_Q35MOE_FFN1_N_SRC; ++j) {
        const ggml_backend_buffer_t buf = op->src[j]->buffer;
        if (!buf || buf->buft != buft || ggml_backend_buffer_get_usage(buf) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return false;
        }
    }
    return true;
}

// calls f with the type as a constant
template <typename F>
static void q35moe_ffn_with_type(ggml_type type, F && f) {
    switch (type) {
        case GGML_TYPE_Q8_0: f(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{}); break;
        case GGML_TYPE_Q4_K: f(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K: f(std::integral_constant<ggml_type, GGML_TYPE_Q5_K>{}); break;
        case GGML_TYPE_Q6_K: f(std::integral_constant<ggml_type, GGML_TYPE_Q6_K>{}); break;
        default:             GGML_ABORT("fatal error");
    }
}

static void q35moe_ffn1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    cudaStream_t stream = ctx.stream();

    const float * x   = static_cast<const float *>(op->src[GGML_Q35MOE_FFN1_X]->data);
    const float * res = static_cast<const float *>(op->src[GGML_Q35MOE_FFN1_RES]->data);

    const ggml_tensor * up_exps    = op->src[GGML_Q35MOE_FFN1_UP_EXPS];
    const ggml_tensor * down_exps  = op->src[GGML_Q35MOE_FFN1_DOWN_EXPS];
    const ggml_tensor * up_shexp   = op->src[GGML_Q35MOE_FFN1_UP_SHEXP];
    const ggml_tensor * down_shexp = op->src[GGML_Q35MOE_FFN1_DOWN_SHEXP];

    const void * w_router     = op->src[GGML_Q35MOE_FFN1_GATE_INP]->data;
    const void * w_gate       = op->src[GGML_Q35MOE_FFN1_GATE_INP_SHEXP]->data;
    const void * w_gate_exps  = op->src[GGML_Q35MOE_FFN1_GATE_EXPS]->data;
    const void * w_gate_shexp = op->src[GGML_Q35MOE_FFN1_GATE_SHEXP]->data;

    // scratch: router logits, gate, weights and ids of the selected experts, hidden vectors, shared expert output, quantized vectors
    static_assert(sizeof(block_q8_1) == 9*sizeof(float), "a q8_1 block takes 9 words of the scratch");
    constexpr int n_h      = q35moe_n_ff*q35moe_n_used;
    constexpr int n_floats = q35moe_n_expert + 3*q35moe_n_used + q35moe_n_ff + n_h + q35moe_n_embd;
    constexpr int n_q8     = (q35moe_n_embd + q35moe_n_ff + n_h) / QK8_1;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_floats + 9*n_q8);

    float *      logits   = scratch.get();
    float *      gate     = logits + q35moe_n_expert;
    float *      weights  = gate + q35moe_n_used;
    int32_t *    ids      = reinterpret_cast<int32_t *>(weights + q35moe_n_used);
    float *      h_shexp  = weights + 2*q35moe_n_used;
    float *      h_exps   = h_shexp + q35moe_n_ff;
    float *      d_shexp  = h_exps + n_h;
    block_q8_1 * xq       = reinterpret_cast<block_q8_1 *>(d_shexp + q35moe_n_embd);
    block_q8_1 * hq_shexp = xq + q35moe_n_embd/QK8_1;
    block_q8_1 * hq_exps  = hq_shexp + q35moe_n_ff/QK8_1;

    q35moe_ffn_router<<<q35moe_n_expert + 1 + q35moe_n_embd/256, 256, 0, stream>>>(
        static_cast<const float *>(w_router), static_cast<const float *>(w_gate), x, logits, gate, xq);
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffn_with_type(up_shexp->type, [&](auto t) {
        q35moe_ffn_shexp_up<decltype(t)::value><<<q35moe_n_ff + 1, q35moe_warp, 0, stream>>>(
            up_shexp->data, w_gate_shexp, xq, logits, h_shexp, ids, weights);
    });
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffn_with_type(up_exps->type, [&](auto t) {
        q35moe_ffn_exps_up<decltype(t)::value><<<dim3(q35moe_n_ff + 2, q35moe_n_used), q35moe_warp, 0, stream>>>(
            up_exps->data, w_gate_exps, xq, ids, h_shexp, hq_shexp, h_exps);
    });
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffn_with_type(down_shexp->type, [&](auto t) {
        q35moe_ffn_shexp_down<decltype(t)::value><<<q35moe_n_embd + n_h/QK8_1, q35moe_warp, 0, stream>>>(
            down_shexp->data, hq_shexp, h_exps, hq_exps, d_shexp);
    });
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffn_with_type(down_exps->type, [&](auto t) {
        q35moe_ffn_exps_down<decltype(t)::value><<<q35moe_n_embd, dim3(q35moe_warp, q35moe_down_warps), 0, stream>>>(
            down_exps->data, hq_exps, ids, weights, d_shexp, gate, res, static_cast<float *>(op->data));
    });
    CUDA_CHECK(cudaGetLastError());
}

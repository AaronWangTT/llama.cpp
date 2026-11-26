#pragma once
#include "q35moe-ffn.cuh"

// The feed-forward of 2 to ggml_q35moe_ffnm_max tokens in six launches (GGML_Q35MOE_FFNM). Every token gets the value that the stock kernels of a batch of that size give it:
//   router, gate of the shared expert   mul_mat_vec_f<float, float, n, 256>: the sum of a column does not depend on the other columns
//   quantized vectors, top-k            quantize_q8_1, topk_moe_cuda<256, false>: one warp per block or row, as for one token
//   routed experts                      mul_mat_vec_q_moe<type, 2, fusion>: one warp sums a row for a token as mul_mat_vec_q<type, 1> does
//   shared expert                       mul_mat_vec_q<type, n, false> and the SWIGLU kernel: the same sums and the same product as the fused one-token form
//   sum                                 moe_weighted_reduction_f32, MUL, ADD of three values, per token
// The device functions are those of the one-token op.
// GGML_Q35MOE_EXPONCE: tokens of the batch that selected the same expert share one warp, which loads the rows of the expert once. The sums of a token stay the same.

constexpr int q35moe_ffnm_max  = ggml_q35moe_ffnm_max;
constexpr int q35moe_ffnm_rows = 2; // rows of a routed expert per warp, as in mul_mat_vec_q_moe

// The owner word of slot s = token*n_used + place: 0 if an earlier token selected the same expert.
// Else bits 0..2 are the number of tokens m that share the expert. Sharer 0 is the slot itself, bits 3 + 6*(j - 1) are the slot of sharer j.
constexpr int q35moe_ffnm_slot_bits = 6;
static_assert(q35moe_ffnm_max*q35moe_n_used <= (1 << q35moe_ffnm_slot_bits) && q35moe_ffnm_max < 8 &&
              3 + q35moe_ffnm_slot_bits*(q35moe_ffnm_max - 1) <= 32, "owner word layout");

// slot of sharer j of the owner slot s
static __device__ __forceinline__ int q35moe_ffnm_slot(const uint32_t own, const int s, const int j) {
    return j == 0 ? s : (own >> (3 + q35moe_ffnm_slot_bits*(j - 1))) & ((1 << q35moe_ffnm_slot_bits) - 1);
}

// one warp makes the owner words, lane l those of the slots l, l + 32, ..; once: tokens share experts
static __device__ __forceinline__ void q35moe_ffnm_owners(const int32_t * ids, int32_t * own, const int n_tok, const bool once) {
    for (int s = threadIdx.x; s < n_tok*q35moe_n_used; s += q35moe_warp) {
        const int t = s / q35moe_n_used;
        const int e = ids[s];

        uint32_t word  = 0;
        int      m     = 1;
        bool     first = true;
        for (int j = 0; once && j < n_tok; ++j) {
            for (int k = 0; j != t && k < q35moe_n_used; ++k) {
                if (ids[j*q35moe_n_used + k] == e) {
                    first = first && j > t;
                    word |= (uint32_t) (j*q35moe_n_used + k) << (3 + q35moe_ffnm_slot_bits*(m - 1));
                    m++;
                }
            }
        }
        own[s] = first ? word | m : 0;
    }
}

// launch 1, blocks of 256 threads: n_expert router rows and the row of the shared expert gate for n_tok tokens, then n_tok*n_embd/256 blocks that quantize x
template <int n_tok>
__launch_bounds__(256, 1)
static __global__ void q35moe_ffnm_router(
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

    float sumf[n_tok] = {0.0f};

    const float2 * y2 = (const float2 *) y;
    const float2 * x2 = (const float2 *) x;

    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];

#pragma unroll
        for (int j = 0; j < n_tok; ++j) {
            const float2 tmpy = y2[j*ncols2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }

#pragma unroll
    for (int j = 0; j < n_tok; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        buf_iw[tid/warp_size] = sumf[j];
        __syncthreads();
        if (tid < warp_size) {
            sumf[j] = buf_iw[tid];
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        }
        __syncthreads();
    }

    if (tid >= n_tok) {
        return;
    }

    if (row < q35moe_n_expert) {
        logits[tid*q35moe_n_expert + row] = sumf[tid];
    } else {
        gate[tid] = 1.0f / (1.0f + expf(-sumf[tid]));
    }
}

// launch 2: one warp per row of the shared expert gate/up for n_tok tokens, the last block is the top-k of every token and the owner words
template <ggml_type type, int n_tok>
__launch_bounds__(q35moe_warp, 1)
static __global__ void q35moe_ffnm_shexp_up(
        const void * vup, const void * vgate, const block_q8_1 * xq, const float * logits,
        float * h, int32_t * ids, float * weights, int32_t * own, const bool once) {
    constexpr int blocks_per_row = q35moe_n_embd / ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_y   = q35moe_n_embd / QK8_1;

    if (blockIdx.x == q35moe_n_ff) {
#pragma unroll
        for (int j = 0; j < n_tok; ++j) {
            q35moe_topk(logits + j*q35moe_n_expert, ids + j*q35moe_n_used, weights + j*q35moe_n_used);
        }
        q35moe_ffnm_owners(ids, own, n_tok, once);
        return;
    }

    const int row = blockIdx.x;

    const void *       vx[2*n_tok];
    const block_q8_1 * vy[2*n_tok];
    int                kbx[2*n_tok];
#pragma unroll
    for (int j = 0; j < n_tok; ++j) {
        vx[2*j]      = vup;
        vx[2*j + 1]  = vgate;
        vy[2*j]      = xq + j*blocks_per_y;
        vy[2*j + 1]  = xq + j*blocks_per_y;
        kbx[2*j]     = row*blocks_per_row;
        kbx[2*j + 1] = row*blocks_per_row;
    }

    float tmp[2*n_tok];
    q35moe_dots<type, q35moe_n_embd, 2*n_tok>(vx, vy, kbx, tmp);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < n_tok; ++j) {
            float result = tmp[2*j];
            result *= ggml_cuda_op_silu_single(tmp[2*j + 1]);
            h[j*q35moe_n_ff + row] = result;
        }
    }
}

// rows row0 and row0 + 1 of the gate/up of one expert for the m tokens of an owner word, with SWIGLU
template <ggml_type type, int m>
static __device__ __forceinline__ void q35moe_ffnm_up_rows(
        const void * vup, const void * vgate, const block_q8_1 * xq, const int kbx0, const uint32_t own, const int s, const int row0, float * h) {
    constexpr int rows           = q35moe_ffnm_rows;
    constexpr int blocks_per_row = q35moe_n_embd / ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_y   = q35moe_n_embd / QK8_1;

    const void *       vx[2*rows*m];
    const block_q8_1 * vy[2*rows*m];
    int                kbx[2*rows*m];
#pragma unroll
    for (int j = 0; j < m; ++j) {
        const block_q8_1 * y = xq + (q35moe_ffnm_slot(own, s, j)/q35moe_n_used)*blocks_per_y;
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            const int l = 2*(j*rows + i);
            vx[l]      = vup;
            vx[l + 1]  = vgate;
            vy[l]      = y;
            vy[l + 1]  = y;
            kbx[l]     = kbx0 + i*blocks_per_row;
            kbx[l + 1] = kbx0 + i*blocks_per_row;
        }
    }

    float tmp[2*rows*m];
    q35moe_dots<type, q35moe_n_embd, 2*rows*m>(vx, vy, kbx, tmp);

#pragma unroll
    for (int i = 0; i < rows; ++i) {
        if (threadIdx.x == i) {
#pragma unroll
            for (int j = 0; j < m; ++j) {
                const int l = 2*(j*rows + i);
                float result = tmp[l];
                result *= ggml_cuda_op_silu_single(tmp[l + 1]);
                h[q35moe_ffnm_slot(own, s, j)*q35moe_n_ff + row0 + i] = result;
            }
        }
    }
}

// launch 3, grid (n_ff/2 + 2, n_used), one warp per token: the warp of an owner makes two rows of the gate/up of its expert for the tokens that share it
// the last two columns quantize the hidden vectors of the shared expert
template <ggml_type type>
__launch_bounds__(q35moe_warp * q35moe_ffnm_max, 1)
static __global__ void q35moe_ffnm_exps_up(
        const void * vup, const void * vgate, const block_q8_1 * xq, const int32_t * ids, const int32_t * own_ptr,
        const float * h_shexp, block_q8_1 * hq_shexp, float * h) {
    constexpr int rows           = q35moe_ffnm_rows;
    constexpr int blocks_per_row = q35moe_n_embd / ggml_cuda_type_traits<type>::qk;
    static_assert(2*q35moe_n_used == q35moe_n_ff / QK8_1, "two columns hold the blocks of a shared hidden vector");
    static_assert(q35moe_ffnm_max == 4, "one case for each number of tokens that share an expert");

    const int t = threadIdx.y;
    const int k = blockIdx.y;

    if (blockIdx.x >= q35moe_n_ff/rows) {
        q35moe_quantize(h_shexp, hq_shexp, t*(q35moe_n_ff/QK8_1) + (blockIdx.x - q35moe_n_ff/rows)*q35moe_n_used + k);
        return;
    }

    const int      s   = t*q35moe_n_used + k;
    const uint32_t own = own_ptr[s];

    const int row0 = rows*blockIdx.x;
    const int kbx0 = (ids[s]*q35moe_n_ff + row0)*blocks_per_row;

    switch (own & 7) {
        case 1: q35moe_ffnm_up_rows<type, 1>(vup, vgate, xq, kbx0, own, s, row0, h); break;
        case 2: q35moe_ffnm_up_rows<type, 2>(vup, vgate, xq, kbx0, own, s, row0, h); break;
        case 3: q35moe_ffnm_up_rows<type, 3>(vup, vgate, xq, kbx0, own, s, row0, h); break;
        case 4: q35moe_ffnm_up_rows<type, 4>(vup, vgate, xq, kbx0, own, s, row0, h); break;
        default: break;
    }
}

// launch 4: one warp per row of the shared expert down for n_tok tokens, the last blocks quantize the hidden vectors of the routed experts
template <ggml_type type, int n_tok>
__launch_bounds__(q35moe_warp, 1)
static __global__ void q35moe_ffnm_shexp_down(
        const void * vdown, const block_q8_1 * hq_shexp, const float * h_exps, block_q8_1 * hq_exps, float * dst) {
    constexpr int blocks_per_row = q35moe_n_ff / ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_y   = q35moe_n_ff / QK8_1;

    const int row = blockIdx.x;

    if (row >= q35moe_n_embd) {
        q35moe_quantize(h_exps, hq_exps, row - q35moe_n_embd);
        return;
    }

    const void *       vx[n_tok];
    const block_q8_1 * vy[n_tok];
    int                kbx[n_tok];
#pragma unroll
    for (int j = 0; j < n_tok; ++j) {
        vx[j]  = vdown;
        vy[j]  = hq_shexp + j*blocks_per_y;
        kbx[j] = row*blocks_per_row;
    }

    float tmp[n_tok];
    q35moe_dots<type, q35moe_n_ff, n_tok>(vx, vy, kbx, tmp);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < n_tok; ++j) {
            dst[j*q35moe_n_embd + row] = tmp[j];
        }
    }
}

// rows row0 and row0 + 1 of the down of one expert for the m tokens of an owner word
template <ggml_type type, int m>
static __device__ __forceinline__ void q35moe_ffnm_down_rows(
        const void * vdown, const block_q8_1 * hq, const int kbx0, const uint32_t own, const int s, const int row0, float * d) {
    constexpr int rows           = q35moe_ffnm_rows;
    constexpr int blocks_per_row = q35moe_n_ff / ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_y   = q35moe_n_ff / QK8_1;

    const void *       vx[rows*m];
    const block_q8_1 * vy[rows*m];
    int                kbx[rows*m];
#pragma unroll
    for (int j = 0; j < m; ++j) {
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            vx[j*rows + i]  = vdown;
            vy[j*rows + i]  = hq + q35moe_ffnm_slot(own, s, j)*blocks_per_y;
            kbx[j*rows + i] = kbx0 + i*blocks_per_row;
        }
    }

    float tmp[rows*m];
    q35moe_dots<type, q35moe_n_ff, rows*m>(vx, vy, kbx, tmp);

#pragma unroll
    for (int i = 0; i < rows; ++i) {
        if (threadIdx.x == i) {
#pragma unroll
            for (int j = 0; j < m; ++j) {
                d[q35moe_ffnm_slot(own, s, j)*q35moe_n_embd + row0 + i] = tmp[j*rows + i];
            }
        }
    }
}

// launch 5, grid (n_embd/2, n_used), one warp per token: the warp of an owner makes two rows of the down of its expert for the tokens that share it
template <ggml_type type>
__launch_bounds__(q35moe_warp * q35moe_ffnm_max, 1)
static __global__ void q35moe_ffnm_exps_down(
        const void * vdown, const block_q8_1 * hq, const int32_t * ids, const int32_t * own_ptr, float * d) {
    constexpr int rows           = q35moe_ffnm_rows;
    constexpr int blocks_per_row = q35moe_n_ff / ggml_cuda_type_traits<type>::qk;

    const int      s   = threadIdx.y*q35moe_n_used + blockIdx.y;
    const uint32_t own = own_ptr[s];

    const int row0 = rows*blockIdx.x;
    const int kbx0 = (ids[s]*q35moe_n_embd + row0)*blocks_per_row;

    switch (own & 7) {
        case 1: q35moe_ffnm_down_rows<type, 1>(vdown, hq, kbx0, own, s, row0, d); break;
        case 2: q35moe_ffnm_down_rows<type, 2>(vdown, hq, kbx0, own, s, row0, d); break;
        case 3: q35moe_ffnm_down_rows<type, 3>(vdown, hq, kbx0, own, s, row0, d); break;
        case 4: q35moe_ffnm_down_rows<type, 4>(vdown, hq, kbx0, own, s, row0, d); break;
        default: break;
    }
}

// launch 6, grid (n_embd/256, n_tok), one thread per output value: the weighted sum of the routed experts, the gated shared expert and the residual
__launch_bounds__(256, 1)
static __global__ void q35moe_ffnm_sum(
        const float * d_exps, const float * weights_ptr, const float * d_shexp, const float * gate, const float * res, float * dst) {
    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    const int t   = blockIdx.y;

    const float * experts = d_exps + t*q35moe_n_used*q35moe_n_embd + row;
    const float * weights = weights_ptr + t*q35moe_n_used;

    float sum = experts[0] * weights[0];
#pragma unroll
    for (int k = 1; k < q35moe_n_used; ++k) {
        sum = fmaf(experts[k*q35moe_n_embd], weights[k], sum);
    }

    const float shexp = d_shexp[t*q35moe_n_embd + row];
    const float g     = gate[t];
    const float r     = res[t*q35moe_n_embd + row];
    float prod, part, out;
    // explicit instructions, the stock nodes are a MUL and two ADD: no contraction
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(prod) : "v"(shexp), "v"(g));
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(part) : "v"(sum), "v"(prod));
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(out) : "v"(part), "v"(r));
    dst[t*q35moe_n_embd + row] = out;
}

static bool q35moe_ffnm_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    if (info.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || info.warp_size != q35moe_warp || !ggml_q35moe_ffnm_enabled() || !ggml_q35moe_is_ffnm(op)) {
        return false;
    }
    // as for the one-token op: the kernels replay the fused kernels of the stock nodes
    if (ggml_cuda_fusion_disabled()) {
        return false;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);
    for (int j = GGML_Q35MOE_FFN1_GATE_INP; j < GGML_Q35MOE_FFN1_N_SRC; ++j) {
        const ggml_backend_buffer_t buf = op->src[j]->buffer;
        if (!buf || buf->buft != buft || ggml_backend_buffer_get_usage(buf) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return false;
        }
    }
    return true;
}

// calls f with the token count as a constant
template <typename F>
static void q35moe_ffnm_with_count(int n_tok, F && f) {
    static_assert(q35moe_ffnm_max == 4, "one case for each token count");
    switch (n_tok) {
        case 2:  f(std::integral_constant<int, 2>{}); break;
        case 3:  f(std::integral_constant<int, 3>{}); break;
        case 4:  f(std::integral_constant<int, 4>{}); break;
        default: GGML_ABORT("fatal error");
    }
}

static void q35moe_ffnm_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    cudaStream_t stream = ctx.stream();

    const int  n_tok = op->ne[1];
    const bool once  = ggml_q35moe_exponce_enabled();

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

    // scratch, each part holds the tokens one after the other:
    // router logits, gates, weights, ids and owner words of the selected experts, hidden vectors, outputs of the shared and the routed experts, quantized vectors
    constexpr int n_h      = q35moe_n_ff*q35moe_n_used;
    constexpr int n_d      = q35moe_n_embd*q35moe_n_used;
    constexpr int n_floats = q35moe_n_expert + 4*q35moe_n_used + q35moe_n_ff + n_h + q35moe_n_embd + n_d;
    constexpr int n_q8     = (q35moe_n_embd + q35moe_n_ff + n_h) / QK8_1;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_tok*(n_floats + 9*n_q8));

    float *      logits   = scratch.get();
    float *      gate     = logits + n_tok*q35moe_n_expert;
    float *      weights  = gate + n_tok*q35moe_n_used;
    int32_t *    ids      = reinterpret_cast<int32_t *>(weights + n_tok*q35moe_n_used);
    int32_t *    own      = ids + n_tok*q35moe_n_used;
    float *      h_shexp  = weights + 3*n_tok*q35moe_n_used;
    float *      h_exps   = h_shexp + n_tok*q35moe_n_ff;
    float *      d_shexp  = h_exps + n_tok*n_h;
    float *      d_exps   = d_shexp + n_tok*q35moe_n_embd;
    block_q8_1 * xq       = reinterpret_cast<block_q8_1 *>(d_exps + n_tok*n_d);
    block_q8_1 * hq_shexp = xq + n_tok*(q35moe_n_embd/QK8_1);
    block_q8_1 * hq_exps  = hq_shexp + n_tok*(q35moe_n_ff/QK8_1);

    q35moe_ffnm_with_count(n_tok, [&](auto n) {
        q35moe_ffnm_router<decltype(n)::value><<<q35moe_n_expert + 1 + n_tok*(q35moe_n_embd/256), 256, 0, stream>>>(
            static_cast<const float *>(w_router), static_cast<const float *>(w_gate), x, logits, gate, xq);
    });
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffn_with_type(up_shexp->type, [&](auto t) {
        q35moe_ffnm_with_count(n_tok, [&](auto n) {
            q35moe_ffnm_shexp_up<decltype(t)::value, decltype(n)::value><<<q35moe_n_ff + 1, q35moe_warp, 0, stream>>>(
                up_shexp->data, w_gate_shexp, xq, logits, h_shexp, ids, weights, own, once);
        });
    });
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffn_with_type(up_exps->type, [&](auto t) {
        q35moe_ffnm_exps_up<decltype(t)::value><<<dim3(q35moe_n_ff/q35moe_ffnm_rows + 2, q35moe_n_used), dim3(q35moe_warp, n_tok), 0, stream>>>(
            up_exps->data, w_gate_exps, xq, ids, own, h_shexp, hq_shexp, h_exps);
    });
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffn_with_type(down_shexp->type, [&](auto t) {
        q35moe_ffnm_with_count(n_tok, [&](auto n) {
            q35moe_ffnm_shexp_down<decltype(t)::value, decltype(n)::value><<<q35moe_n_embd + n_tok*(n_h/QK8_1), q35moe_warp, 0, stream>>>(
                down_shexp->data, hq_shexp, h_exps, hq_exps, d_shexp);
        });
    });
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffn_with_type(down_exps->type, [&](auto t) {
        q35moe_ffnm_exps_down<decltype(t)::value><<<dim3(q35moe_n_embd/q35moe_ffnm_rows, q35moe_n_used), dim3(q35moe_warp, n_tok), 0, stream>>>(
            down_exps->data, hq_exps, ids, own, d_exps);
    });
    CUDA_CHECK(cudaGetLastError());

    q35moe_ffnm_sum<<<dim3(q35moe_n_embd/256, n_tok), 256, 0, stream>>>(
        d_exps, weights, d_shexp, gate, res, static_cast<float *>(op->data));
    CUDA_CHECK(cudaGetLastError());
}

#pragma once

// include after mmf.cuh and mmvf.cuh
#include "inkling-pfmm.cuh"

// GGML_Q35MOE_PFMMN=1: F32 products of a prefill ubatch of qwen35moe (router, beta, alpha) for q35moe_pfmmn_min_tokens to q35moe_pfmmn_max_tokens tokens, not only the 512 of GGML_Q35MOE_PFMM.
// rocBLAS has one SGEMM kernel for gfx1103 (Tensile MT32x32x8, StaggerU 32 on the row tile), so a token column has the same bits whatever the token count.

// the largest batch of the server; the fixture tests/q35moe-hip-pfmmn.cpp compares every token count up to it with SGEMM
constexpr int q35moe_pfmmn_max_tokens = 2048;
// a partial block does the work of 128 tokens, so SGEMM is faster for small batches (bench of the fixture): they stay with SGEMM
constexpr int q35moe_pfmmn_min_tokens = 128;

static bool q35moe_pfmmn_enabled() {
    static const bool enabled = [] { const char * p = getenv("GGML_Q35MOE_PFMMN"); return p && strcmp(p, "1") == 0; }();
    return enabled;
}

// inkling_pfmm<float> with a token bound: the last block of 128 tokens may be partial. 256 threads, each owns 4 rows x 4 tokens.
static __global__ void q35moe_pfmmn(const float * w_ptr, const float * x_ptr, float * dst_ptr,
        const int n_k, const int n_rows, const int n_tokens, const int stagger_mask) {
    constexpr int depth = inkling_pfmm_depth;
    constexpr int rows  = inkling_pfmm_rows;
    constexpr int cols  = inkling_pfmm_cols;
    constexpr int nw    = depth*rows/256; // values of w and of x that one thread loads
    constexpr int nx    = depth*cols/256;
    const float * GGML_CUDA_RESTRICT w   = w_ptr;
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;

    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x*rows;
    const int col0 = blockIdx.y*cols;

    // k major, so that a thread reads its 4 rows or 4 tokens of one k as one float4
    __shared__ float4 s_w[depth*rows/4];
    __shared__ float4 s_x[depth*cols/4];
    float * s_w1 = (float *) s_w;
    float * s_x1 = (float *) s_x;

    // loads: nw k of one row of w, nx k of one token of x; past the last token the last one is read again, its products are not stored
    const int     lw_row = tid % rows;
    const int     lw_k   = nw*(tid / rows);
    const int     lx_col = tid % cols;
    const int     lx_k   = nx*(tid / cols);
    const float * lw     = w + (int64_t) (row0 + lw_row)*n_k + lw_k;
    const float * lx     = x + (int64_t) min(col0 + lx_col, n_tokens - 1)*n_k + lx_k;

    // products: rows 4*pi .. 4*pi + 3, tokens 4*pj .. 4*pj + 3
    const int pi = tid % (rows/4);
    const int pj = tid / (rows/4);

    float acc[4][4] = {};

    // the rocBLAS kernel starts tile t at iteration 8*(t & mask) of 8 k each
    const int k_start = (blockIdx.x & stagger_mask)*64;
    for (int kb = 0; kb < n_k; kb += depth) {
        int k = k_start + kb;
        if (k >= n_k) {
            k -= n_k;
        }
        float tw[nw];
        float tx[nx];
#pragma unroll
        for (int q = 0; q < nw; q += 2) {
            const float2 v = *(const float2 *) (lw + k + q);
            tw[q + 0] = v.x;
            tw[q + 1] = v.y;
        }
#pragma unroll
        for (int q = 0; q < nx; q += 4) {
            const float4 v = *(const float4 *) (lx + k + q);
            tx[q + 0] = v.x;
            tx[q + 1] = v.y;
            tx[q + 2] = v.z;
            tx[q + 3] = v.w;
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < nw; ++q) {
            s_w1[(lw_k + q)*rows + lw_row] = tw[q];
        }
#pragma unroll
        for (int q = 0; q < nx; ++q) {
            s_x1[(lx_k + q)*cols + lx_col] = tx[q];
        }
        __syncthreads();
#pragma unroll
        for (int d = 0; d < depth; ++d) {
            const float4 a4 = s_w[d*(rows/4) + pi];
            const float4 b4 = s_x[d*(cols/4) + pj];
            const float  a[4] = { a4.x, a4.y, a4.z, a4.w };
            const float  b[4] = { b4.x, b4.y, b4.z, b4.w };
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[j][i] = fmaf(a[i], b[j], acc[j][i]);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = col0 + 4*pj + j;
        if (col < n_tokens) {
            float4 * out = (float4 *) (dst + (int64_t) col*n_rows + row0 + 4*pi);
            *out = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        }
    }
}

// the F32 weights of the architecture, [2048, rows]: router with 256 rows, beta and alpha with 32 rows; token counts from q35moe_pfmmn_min_tokens that the stock path gives to SGEMM
static bool q35moe_pfmmn_supported(int device, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * dst) {
    const auto & info = ggml_cuda_info().devices[device];
    if (info.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || info.warp_size != 32 || !q35moe_pfmmn_enabled()) {
        return false;
    }
    const auto ends_with = [w](const char * suffix) {
        const size_t n = strlen(w->name), m = strlen(suffix);
        return n >= m && strcmp(w->name + n - m, suffix) == 0;
    };
    const int64_t n_k = w->ne[0], n_rows = w->ne[1], n_tokens = x->ne[1];
    if (w->type != GGML_TYPE_F32 || w->view_src || n_k != 2048) {
        return false;
    }
    if (!((n_rows == 256 && ends_with(".ffn_gate_inp.weight")) || (n_rows == 32 && (ends_with(".ssm_beta.weight") || ends_with(".ssm_alpha.weight"))))) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || x->ne[0] != n_k || dst->ne[0] != n_rows || dst->ne[1] != n_tokens ||
            n_tokens < q35moe_pfmmn_min_tokens || n_tokens > q35moe_pfmmn_max_tokens) {
        return false;
    }
    // few tokens take the vector kernels (other bits than SGEMM), see ggml_cuda_mul_mat
    if (ggml_cuda_should_use_mmvf(w, info.cc, info.warp_size, n_tokens) ||
            ggml_cuda_should_use_mmf(w->type, info.cc, info.warp_size, w->ne, w->nb, n_tokens, /*mul_mat_id =*/ false)) {
        return false;
    }
    for (const ggml_tensor * t : { w, x, dst }) {
        // the kernel loads float4 from x and stores float4 to dst
        if (t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t) || (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    return inkling_pfmm_stock_sgemm("GGML_Q35MOE_PFMMN");
}

static void q35moe_pfmmn_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * dst) {
    const int n_k = w->ne[0], n_rows = w->ne[1], n_tokens = x->ne[1];
    // start offsets as rocBLAS sets them: up to 32, halved while k/64 is smaller
    int stagger = 32;
    while (stagger > 1 && n_k/64 < stagger) {
        stagger /= 2;
    }
    const dim3 grid(n_rows/inkling_pfmm_rows, (n_tokens + inkling_pfmm_cols - 1)/inkling_pfmm_cols);
    q35moe_pfmmn<<<grid, 256, 0, ctx.stream()>>>(static_cast<const float *>(w->data), static_cast<const float *>(x->data),
        static_cast<float *>(dst->data), n_k, n_rows, n_tokens, stagger - 1);
    CUDA_CHECK(cudaGetLastError());
    // for the fixture: one line per product
    static const bool trace = [] { const char * p = getenv("GGML_Q35MOE_PFMMN_TRACE"); return p && strcmp(p, "1") == 0; }();
    if (trace) {
        GGML_LOG_INFO("Q35MOE_PFMMN_HIT %s rows=%d tokens=%d\n", w->name, n_rows, n_tokens);
    }
}

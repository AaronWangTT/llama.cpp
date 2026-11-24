#pragma once

#include <cctype>
#include <string>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private PFMM experiment for gfx1103 only"
#endif

// Float products of a full prefill batch: dst[i, j] = sum_k w[k, i]*x[k, j], for the router and the loop controller weights.
// Same bits as the rocBLAS kernel that SGEMM takes for these sizes (Tensile Cijk_Alik_Bljk_SB_MT32x32x8, StaggerU 32, stride 256 B):
// each value is one fused multiply-add chain from zero, k rises from a start that depends on the 32 row tile and wraps at the end.
constexpr int inkling_pfmm_n_tokens = 256;
constexpr int inkling_pfmm_rows     = 32;  // rows of w per block; rocBLAS sets one start offset per 32 rows
constexpr int inkling_pfmm_cols     = 128; // tokens per block
constexpr int inkling_pfmm_depth    = 16;  // k per load

static bool inkling_pfmm_enabled() {
    static const bool enabled = [] { const char * p = getenv("GGML_INKLING_PFMM"); return p && strcmp(p, "1") == 0; }();
    return enabled;
}

// The kernel replays the F32 SGEMM that rocBLAS takes by default. False if the environment gives the stock product another compute type or
// another GEMM; this goes to the log once. name is the switch that asks.
static bool inkling_pfmm_stock_sgemm(const char * name) {
    const char * var = nullptr;
    for (const char * v : { "ROCBLAS_USE_HIPBLASLT", "ROCBLAS_TENSILE_LIBPATH", "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH" }) {
        if (getenv(v) != nullptr) {
            var = v;
        }
    }
    // read on each call and in lower case, as ggml_cuda_mul_mat_cublas does
    if (const char * p = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")) {
        std::string type = p;
        for (char & c : type) {
            c = std::tolower(c);
        }
        if (type == "f16" || type == "fp16" || type == "bf16") {
            var = "GGML_CUDA_CUBLAS_COMPUTE_TYPE";
        }
    }
    if (var == nullptr) {
        return true;
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_WARN("%s: %s is not used, %s is set\n", __func__, name, var);
    }
    return false;
}

// weights k and k + 1 of a row; a BF16 weight is the high half of the F32 value
template <typename T>
static __device__ __forceinline__ float2 inkling_pfmm_weight2(const T * row, const int k) {
    if constexpr (std::is_same_v<T, float>) {
        return *(const float2 *) (row + k);
    } else {
        static_assert(std::is_same_v<T, nv_bfloat16>, "unsupported type");
        const uint32_t pair = *(const uint32_t *) (row + k);
        return make_float2(__uint_as_float(pair << 16), __uint_as_float(pair & 0xffff0000u));
    }
}

// 256 threads, each owns 4 rows x 4 tokens
template <typename T>
static __global__ void inkling_pfmm(const T * w_ptr, const float * x_ptr, float * dst_ptr,
        const int n_k, const int n_rows, const int stagger_mask) {
    constexpr int depth = inkling_pfmm_depth;
    constexpr int rows  = inkling_pfmm_rows;
    constexpr int cols  = inkling_pfmm_cols;
    constexpr int nw    = depth*rows/256; // values of w and of x that one thread loads
    constexpr int nx    = depth*cols/256;
    const T     * GGML_CUDA_RESTRICT w   = w_ptr;
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

    // loads: nw k of one row of w, nx k of one token of x
    const int     lw_row = tid % rows;
    const int     lw_k   = nw*(tid / rows);
    const int     lx_col = tid % cols;
    const int     lx_k   = nx*(tid / cols);
    const T     * lw     = w + (int64_t) (row0 + lw_row)*n_k + lw_k;
    const float * lx     = x + (int64_t) (col0 + lx_col)*n_k + lx_k;

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
            const float2 v = inkling_pfmm_weight2(lw, k + q);
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
        float4 * out = (float4 *) (dst + (int64_t) (col0 + 4*pj + j)*n_rows + row0 + 4*pi);
        *out = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
    }
}

// the three weight shapes of the model, [k, rows]: router, controller down, controller gain and forget
static bool inkling_pfmm_supported(int device, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * dst) {
    const auto & info = ggml_cuda_info().devices[device];
    if (info.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || info.warp_size != 32 || !inkling_pfmm_enabled()) {
        return false;
    }
    // BF16 rounds x in the stock kernels, unless the product accumulates in F32
    if (w->type != GGML_TYPE_F32 && (w->type != GGML_TYPE_BF16 || ggml_prec_get_acc(dst) != GGML_PREC_F32)) {
        return false;
    }
    const int64_t n_k = w->ne[0], n_rows = w->ne[1];
    if (!((n_k == 2304 && n_rows == 128) || (n_k == 4608 && n_rows == 64) || (n_k == 64 && n_rows == 2304))) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || x->ne[0] != n_k || x->ne[1] != inkling_pfmm_n_tokens ||
            dst->ne[0] != n_rows || dst->ne[1] != inkling_pfmm_n_tokens) {
        return false;
    }
    for (const ggml_tensor * t : { w, x, dst }) {
        // the kernel loads float4 from x and stores float4 to dst
        if (t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t) || (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    return inkling_pfmm_stock_sgemm("GGML_INKLING_PFMM");
}

static void inkling_pfmm_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * dst) {
    const int n_k = w->ne[0], n_rows = w->ne[1];
    // start offsets as rocBLAS sets them: up to 32, halved while k/64 is smaller
    int stagger = 32;
    while (stagger > 1 && n_k/64 < stagger) {
        stagger /= 2;
    }
    const dim3 grid(n_rows/inkling_pfmm_rows, inkling_pfmm_n_tokens/inkling_pfmm_cols);
    const float * xd = static_cast<const float *>(x->data);
    float       * dd = static_cast<float *>(dst->data);
    if (w->type == GGML_TYPE_BF16) {
        inkling_pfmm<nv_bfloat16><<<grid, 256, 0, ctx.stream()>>>(static_cast<const nv_bfloat16 *>(w->data), xd, dd, n_k, n_rows, stagger - 1);
    } else {
        inkling_pfmm<float><<<grid, 256, 0, ctx.stream()>>>(static_cast<const float *>(w->data), xd, dd, n_k, n_rows, stagger - 1);
    }
    CUDA_CHECK(cudaGetLastError());
    static const bool trace = [] { const char * p = getenv("GGML_INKLING_PFMM_TRACE"); return p && strcmp(p, "1") == 0; }();
    if (trace) {
        GGML_LOG_INFO("INKLING_PFMM_HIT %s %s k=%d rows=%d\n", w->name, ggml_type_name(w->type), n_k, n_rows);
    }
}

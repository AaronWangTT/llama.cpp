#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#if defined(GGML_USE_HIP)

// One-token attention with grouped query heads, F16 K and V.
// One block takes one KV head and one KV split. It computes the G query heads of this KV head from one load of each K and V row.
// For each head the float operations and their order are the same as in flash_attn_ext_vec<D, 1, F16, F16, false, true>.
template<int D, int G> // D == head size, G == query heads per KV head
__launch_bounds__(128, 1)
static __global__ void inkling_fagqa_vec(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int64_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(V_DOT2_F32_F16_AVAILABLE)
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    // The caller makes sure that max_bias is 0 (mask slope 1), that there is a mask with one row per head and no logit softcap.
    GGML_UNUSED_VARS(KV_rows_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13, ne31, ne32);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int nthreads    = 128;
    constexpr int nwarps      = nthreads / WARP_SIZE;
    constexpr int nthreads_KQ = 128 / cpy_nb;
    constexpr int nthreads_V  = 128 / cpy_nb;

    constexpr int V_rows_per_thread = 2*cpy_ne;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr int ne_KQ_thread  = (D/2)/nthreads_KQ; // half2 of one K row per thread
    constexpr int ne_VKQ_thread = (D/2)/nthreads_V;  // half2 of one V row per thread

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");
    static_assert(nthreads_KQ*sizeof(half) == cpy_nb, "the mask of one thread group must be one copy");

    const int ic0 = blockIdx.x; // Index of the Q column to work on.

    const int sequence   = blockIdx.z / ne12;
    const int head0      = (blockIdx.z - sequence*ne12)*G; // First query head of the KV head.
    const unsigned split = blockIdx.y;

    Q += nb03*sequence + nb02* head0      + nb01*ic0;
    K += nb13*sequence + nb12*(head0 / G);
    V += nb23*sequence + nb22*(head0 / G);

    const half * maskh      = (const half *) (mask + nb33*(sequence % ne33) + nb32*head0 + nb31*ic0);
    const int64_t mask_step = nb32 / sizeof(half); // From one head to the next.

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
    static_assert(G*nthreads <= ne_combine, "KQ too small");
    half2           VKQ[G][ne_VKQ_thread] = {{{0.0f, 0.0f}}};
    __shared__ half KQ[ne_combine];

    float KQ_max[G];
    float KQ_sum[G];
#pragma unroll
    for (int j = 0; j < G; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to half2 and store in registers, one set per head:
    half2 Q_reg[G][ne_KQ_thread];
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const float2 * Q_j = (const float2 *) (Q + j*nb02);
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
            const int i = i0 + (threadIdx.x % nthreads_KQ)*cpy_ne;

            __align__(16) float2 tmp[cpy_ne];
            ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
            ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne; ++i1) {
                Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
            }
        }
#pragma unroll
        for (int k = 0; k < ne_KQ_thread; ++k) {
            Q_reg[j][k] *= scale_h2;
        }
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += split*nthreads * nb11;
    V     += split*nthreads * nb21;
    maskh += split*nthreads;
    for (int k_VKQ_0 = split*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Mask of the rows of this thread group, one 16 byte load per head.
        __align__(16) half mask_reg[G][nthreads_KQ];
        int visible = 0;
#pragma unroll
        for (int j = 0; j < G; ++j) {
            ggml_cuda_memcpy_1<cpy_nb>(mask_reg[j], maskh + j*mask_step + (tid & ~(nthreads_KQ-1)));
#pragma unroll
            for (int i = 0; i < nthreads_KQ; ++i) {
                visible |= __half_as_ushort(mask_reg[j][i]) != uint16_t(0xfc00);
            }
        }

        // Skip the tile if no head can see a row of it. This does not change the accumulators.
        if (!__syncthreads_or(visible)) {
            continue;
        }

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[G]; // KQ in registers.

        float KQ_max_new[G];
#pragma unroll
        for (int j = 0; j < G; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

        // Keep this loop unrolled with one K row per step. Loops that are not unrolled and steps of more rows were slower on gfx1103.
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (threadIdx.x & ~(nthreads_KQ-1)) + i_KQ_0;

            // One load of the K row for all heads:
            __align__(16) half2 K_reg[ne_KQ_thread];
            const half2 * K_h2 = (const half2 *) (K + i_KQ*nb11);
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads_KQ*cpy_ne) {
                ggml_cuda_memcpy_1<cpy_nb>(&K_reg[k_KQ_0/nthreads_KQ], K_h2 + k_KQ_0 + (threadIdx.x % nthreads_KQ)*cpy_ne);
            }

#pragma unroll
            for (int j = 0; j < G; ++j) {
                float sum = 0.0f;
#pragma unroll
                for (int k = 0; k < ne_KQ_thread; ++k) {
                    ggml_cuda_mad(sum, K_reg[k], Q_reg[j][k]);
                }
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                sum += __half2float(mask_reg[j][i_KQ_0]);

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if (threadIdx.x % nthreads_KQ == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < G; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i = 0; i < ne_VKQ_thread; ++i) {
                VKQ[j][i] *= KQ_max_scale_h2;
            }
        }

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + threadIdx.x / nthreads_V;

            half2 KQ_k[G];
#pragma unroll
            for (int j = 0; j < G; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                // One load of the V row for all heads:
                half2 tmp[V_rows_per_thread/2];
                dequantize_V_f16<half, V_rows_per_thread>(V + k*nb21, tmp, 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < G; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
        }
    }

    // Only the first warp takes the sinks, as with one query column per block.
    if (sinks && split == 0 && threadIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < G; ++j) {
            const float sink = ((const float *) sinks)[head0 + j];

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i = 0; i < ne_VKQ_thread; ++i) {
                VKQ[j][i] *= KQ_max_scale_h2;
            }
        }
    }

    __shared__ float KQ_max_shared[G][WARP_SIZE];
    __shared__ float KQ_sum_shared[G][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < G; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < G; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < G; ++j) {
        float kqmax_new = KQ_max_shared[j][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j] - kqmax_new);
        KQ_max[j] = kqmax_new;

        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2) + (threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i = 0; i < ne_VKQ_thread; ++i) {
            VKQ[j][i] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j][i_VKQ_0/nthreads_V]);
        }

        KQ_sum[j] *= kqmax_scale;
        KQ_sum[j] = warp_reduce_sum(KQ_sum[j]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j][threadIdx.y] = KQ_sum[j];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j] = KQ_sum_shared[j][threadIdx.x];
            KQ_sum[j] = warp_reduce_sum(KQ_sum[j]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j];
                }
                dst[(((sequence*int(ne01.z) + ic0)*ne02 + head0 + j)*gridDim.y + split)*D + i0 + tid] = dst_val;
            }
        }

        if (j < G-1) {
            __syncthreads();
        }
    }

    // Same layout as with one block per head, flash_attn_combine_results reads it.
    if (gridDim.y != 1 && tid == 0) {
#pragma unroll
        for (int j = 0; j < G; ++j) {
            dst_meta[((sequence*int(ne01.z) + ic0)*ne02 + head0 + j)*gridDim.y + split] = make_float2(KQ_max[j], KQ_sum[j]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(V_DOT2_F32_F16_AVAILABLE)
}

#endif // defined(GGML_USE_HIP)

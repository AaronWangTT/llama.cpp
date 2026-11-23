#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#if defined(GGML_USE_HIP)

// One-token attention with grouped query heads, F16 K and V.
// One block takes one KV head and one KV split. It computes the G query heads of this KV head from one load of each K and V row.
// For each head the float operations and their order are the same as in flash_attn_ext_vec<D, 1, F16, F16, false, true>.
//
// VSKIP == n > 0 does not load a V row if its stored weight is below 2^-n for all G heads, so the result is not the same.
// The stored weight is relative to the running maximum plus FATTN_KQ_MAX_OFFSET: the best row has 2^-3, the cutoff is 2^-(n-3) of the best row.
// The V rows of a tile are then loaded one tile later, together with the K rows of the next tile.
// The weights have to be known first, and V rows that wait for the weights of their own tile are slow.
template<int D, int G, int VSKIP = 0> // D == head size, G == query heads per KV head
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
    static_assert(VSKIP >= 0 && VSKIP <= 14, "bad VSKIP");
    static_assert(VSKIP == 0 || nthreads_KQ*V_cols_per_iter == WARP_SIZE, "VSKIP needs as many K steps as V steps");

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
    __shared__ uint8_t V_row[nthreads]; // VSKIP: the V row to load for a row of the tile

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

    // VSKIP: the tile whose V rows are still to be added, its weights are in KQ. Before the first tile the weights are zero.
    [[maybe_unused]] const char * V_pend = nullptr;
    if constexpr (VSKIP > 0) {
#pragma unroll
        for (int j = 0; j < G; ++j) {
            KQ[j*nthreads + tid] = __float2half(0.0f);
        }
        V_row[tid] = tid;
    }

    // The V rows of one tile, with the weights in KQ.
    const auto add_V_tile = [&](const char * GGML_CUDA_RESTRICT V_tile) {
#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + threadIdx.x / nthreads_V;

            half2 KQ_k[G];
#pragma unroll
            for (int j = 0; j < G; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
            const int k_V = VSKIP > 0 ? int(V_row[k]) : k;
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                // One load of the V row for all heads:
                half2 tmp[V_rows_per_thread/2];
                dequantize_V_f16<half, V_rows_per_thread>(V_tile + k_V*nb21, tmp, 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < G; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
        }
    };

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

            // VSKIP: one step of V rows of the tile before, loaded together with the K row.
            // The first tile takes its own V rows with weights of zero.
            [[maybe_unused]] half2 V_reg[ne_VKQ_thread];
            [[maybe_unused]] half2 KQ_V[G];
            if constexpr (VSKIP > 0) {
                const char * V_step = V_pend ? V_pend : V;
                const int k_step = threadIdx.y*WARP_SIZE + i_KQ_0*V_cols_per_iter + threadIdx.x / nthreads_V;
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    dequantize_V_f16<half, V_rows_per_thread>(V_step + int(V_row[k_step])*nb21, &V_reg[i_VKQ_0/nthreads_V],
                        2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int j = 0; j < G; ++j) {
                    KQ_V[j] = __half2half2(KQ[j*nthreads + k_step]);
                }
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

            if constexpr (VSKIP > 0) {
#pragma unroll
                for (int i = 0; i < ne_VKQ_thread; ++i) {
#pragma unroll
                    for (int j = 0; j < G; ++j) {
                        VKQ[j][i] += V_reg[i]*KQ_V[j];
                    }
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

        if constexpr (VSKIP > 0) {
            // A row with small weights for all heads gets weights of zero and takes the first V row of its warp instead.
            // The weights are not negative, their order is the order of their bits.
            constexpr uint16_t w_min = (15 - VSKIP) << 10;
            bool live = false;
#pragma unroll
            for (int j = 0; j < G; ++j) {
                live |= __half_as_ushort(KQ[j*nthreads + tid]) >= w_min;
            }
#pragma unroll
            for (int j = 0; j < G; ++j) {
                KQ[j*nthreads + tid] = live ? KQ[j*nthreads + tid] : __float2half(0.0f);
            }
            V_row[tid] = live ? tid : threadIdx.y*WARP_SIZE;
            V_pend = V;
        } else {
            add_V_tile(V);
        }
    }

    if constexpr (VSKIP > 0) {
        // The V rows of the last tile.
        if (V_pend) {
            add_V_tile(V_pend);
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

// Load ne_raw x 4 bytes of a Q8_0 row with as few loads as possible. A row is some blocks of (half d, 32 int8). It needs no alignment.
template <int ne_raw>
static __device__ __forceinline__ void inkling_q8_row_load(const char * __restrict__ row, uint32_t * __restrict__ raw) {
    typedef int v4i __attribute__((ext_vector_type(4)));
    typedef int v2i __attribute__((ext_vector_type(2)));
    static_assert(ne_raw % 4 == 1 || ne_raw % 4 == 2, "bad row size");
#pragma unroll
    for (int i = 0; i < ne_raw/4; ++i) {
        v4i tmp;
        __builtin_memcpy(&tmp, row + 4*i*sizeof(int), sizeof(tmp));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            raw[4*i + l] = tmp[l];
        }
    }
    if constexpr (ne_raw % 4 == 2) {
        v2i tmp;
        __builtin_memcpy(&tmp, row + (ne_raw - 2)*sizeof(int), sizeof(tmp));
        raw[ne_raw - 2] = tmp[0];
        raw[ne_raw - 1] = tmp[1];
    } else {
        __builtin_memcpy(&raw[ne_raw - 1], row + (ne_raw - 1)*sizeof(int), sizeof(int));
    }
}

// Take the scales and the groups of 4 int8 out of the loaded blocks of a Q8_0 row. The first block starts at raw[0].
template <int nblocks>
static __device__ __forceinline__ void inkling_q8_row_split(const uint32_t * __restrict__ raw, int * __restrict__ qs, half * __restrict__ d) {
#pragma unroll
    for (int ib = 0; ib < nblocks; ++ib) {
        const int b0 = ib*sizeof(block_q8_0); // First byte of the block. It is a multiple of 2.
        if (b0 % 4 == 0) {
            d[ib] = __ushort_as_half(uint16_t(raw[b0/4]));
#pragma unroll
            for (int i = 0; i < QI8_0; ++i) {
                qs[ib*QI8_0 + i] = (raw[b0/4 + i] >> 16) | (raw[b0/4 + i + 1] << 16);
            }
        } else {
            d[ib] = __ushort_as_half(uint16_t(raw[b0/4] >> 16));
#pragma unroll
            for (int i = 0; i < QI8_0; ++i) {
                qs[ib*QI8_0 + i] = raw[b0/4 + 1 + i];
            }
        }
    }
}

// One-token attention with grouped query heads, Q8_0 K and V.
// One block takes HB KV heads and one KV split. 4 warps take one KV head, they do the work of one block of inkling_fagqa_vec.
// Each value of the result has the bits that flash_attn_ext_vec<D, 1, Q8_0, Q8_0, false> gives: the same integer and float operations in the same order.
// Only the threads that do them are different: one thread loads one K row, two threads load one V row.
// The rows of the KV heads of one block are side by side in memory. A row has 136 bytes, so two heads have a 64 byte line in common.
// On gfx1103 a block with 4 KV heads was faster than 4 blocks with one KV head.
template<int D, int G, int H, int HB> // D == head size, G == query heads per KV head, H == KV heads, HB == KV heads per block
__launch_bounds__(128*HB, 1)
static __global__ void inkling_fagqa_q8_vec(
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
#if defined(FLASH_ATTN_AVAILABLE) && defined(V_DOT2_F32_F16_AVAILABLE) && defined(RDNA)
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    // The caller makes sure that max_bias is 0 (mask slope 1), that there is a mask with one row per head and no logit softcap.
    GGML_UNUSED_VARS(KV_rows_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13, ne31, ne32, nb11, nb21);

    typedef short          v2s __attribute__((ext_vector_type(2)));
    typedef unsigned short v2u __attribute__((ext_vector_type(2)));

    constexpr int nthreads = 128;
    constexpr int nwarps   = nthreads / WARP_SIZE;

    constexpr int ne_K_i32 = D/sizeof(int);              // groups of 4 int8 in one row
    constexpr int nblocks  = D/QK8_0;
    constexpr int nb_head  = nblocks*sizeof(block_q8_0); // bytes of one row of one KV head
    constexpr int nb_row   = H*nb_head;                  // The caller makes sure that this is the step from one row to the next: the heads of one row are side by side.
    constexpr int ne_raw   = nb_head/sizeof(int);

    // Two threads load one V row, one half each, and put it in shared memory. Then each thread takes its 4 values of each row.
    // The index of this memory has threadIdx.y, the warp of the block.
    constexpr int V_rows_shared = WARP_SIZE/2; // V rows of one warp in shared memory

    static_assert(D == 4*WARP_SIZE, "one warp must take one Q row and one V row with 4 values per thread");
    static_assert(G <= nwarps, "one warp converts the Q row of one head");
    static_assert(H % HB == 0, "bad number of KV heads per block");
    static_assert(nblocks % 2 == 0 && (nb_head/2) % sizeof(int) == 0, "half a row must be whole blocks and whole 4 byte words");

    const int ic0 = blockIdx.x; // Index of the Q column to work on.

    const int head_block = threadIdx.y / nwarps; // KV head of this warp in the block
    const int warp       = threadIdx.y % nwarps; // Warp of this KV head

    const int sequence   = blockIdx.z / (ne12/HB);
    const int head_kv    = (blockIdx.z - sequence*(ne12/HB))*HB + head_block;
    const int head0      = head_kv*G; // First query head of the KV head.
    const unsigned split = blockIdx.y;

    Q += nb03*sequence + nb02*head0   + nb01*ic0;
    K += nb13*sequence + nb12*head_kv;
    V += nb23*sequence + nb22*head_kv;

    const half * maskh      = (const half *) (mask + nb33*(sequence % ne33) + nb32*head0 + nb31*ic0);
    const int64_t mask_step = nb32 / sizeof(half); // From one head to the next.

    const int tid = WARP_SIZE*warp + threadIdx.x; // Thread of this KV head, also its row of the tile.
    __builtin_assume(tid < nthreads);

    constexpr int ne_combine = nwarps*D;
    static_assert(G*D <= ne_combine, "KQ too small");
    // Each thread has 4 values of the result per head. VKQ[j][0] has the values 0 and 2, VKQ[j][1] the values 1 and 3.
    half2           VKQ[G][2] = {{{0.0f, 0.0f}}};
    __shared__ half KQ_shared[HB*ne_combine];
    // The softmax weights of the tile, each value two times: the threads of a warp read them for each V row.
    __shared__ half2 KQ_h2_shared[HB*G*nthreads];
    __shared__ __align__(16) int   V_qs_shared[HB*nwarps*V_rows_shared*ne_K_i32];
    __shared__ __align__(16) half2 V_d_shared[HB*nwarps*nblocks*V_rows_shared];
    half  * KQ    = KQ_shared    + head_block*ne_combine;
    half2 * KQ_h2 = KQ_h2_shared + head_block*G*nthreads;

    float KQ_max[G];
    float KQ_sum[G];
#pragma unroll
    for (int j = 0; j < G; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to q8_1 as the reference does, warp j takes head j. KQ is the temporary storage.
    if (warp < G) {
        int    * tmp_q_i32 = (int    *) &KQ[warp*D];
        float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + ne_K_i32);
        quantize_q8_1_to_shared<float2, WARP_SIZE>((const float *) (Q + warp*nb02), scale, tmp_q_i32, tmp_q_ds);
    }

    __syncthreads();

    int   Q_i32[G][ne_K_i32];
    float Q_d[G][nblocks];
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int    * tmp_q_i32 = (const int    *) &KQ[j*D];
        const float2 * tmp_q_ds  = (const float2 *) (tmp_q_i32 + ne_K_i32);
#pragma unroll
        for (int i = 0; i < ne_K_i32; ++i) {
            Q_i32[j][i] = tmp_q_i32[i];
        }
#pragma unroll
        for (int ib = 0; ib < nblocks; ++ib) {
            Q_d[j][ib] = tmp_q_ds[ib].x;
        }
    }

    __syncthreads();

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += split*nthreads * nb_row;
    V     += split*nthreads * nb_row;
    maskh += split*nthreads;
    for (int k_VKQ_0 = split*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb_row, V += gridDim.y*nthreads*nb_row, maskh += gridDim.y*nthreads) {

        // Mask of the row of this thread, one load per head.
        half mask_reg[G];
        int visible = 0;
#pragma unroll
        for (int j = 0; j < G; ++j) {
            mask_reg[j] = maskh[j*mask_step + tid];
            visible |= __half_as_ushort(mask_reg[j]) != uint16_t(0xfc00);
        }

        // Skip the rows of this warp if no head can see one of them. This does not change the accumulators.
        // No value goes from one warp to another in this loop, so each warp can do this alone.
        if (!__any(visible)) {
            continue;
        }

        // Pass p takes V_rows_shared rows of this warp: the half row that this thread loads.
        uint32_t V_raw[2][ne_raw/2];
        const auto load_V_half = [&](const int p) {
            const int row = warp*WARP_SIZE + p*V_rows_shared + threadIdx.x/2;
            inkling_q8_row_load<ne_raw/2>(V + row*nb_row + (threadIdx.x % 2)*(nb_head/2), V_raw[p]);
        };
        load_V_half(0);

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[G]; // KQ in registers.

        float KQ_max_new[G];

        // One load of the whole K row for all heads:
        uint32_t K_raw[ne_raw];
        inkling_q8_row_load<ne_raw>(K + tid*nb_row, K_raw);

        int   K_i32[ne_K_i32];
        half  K_dh[nblocks];
        float K_d[nblocks];
        inkling_q8_row_split<nblocks>(K_raw, K_i32, K_dh);
#pragma unroll
        for (int ib = 0; ib < nblocks; ++ib) {
            K_d[ib] = __half2float(K_dh[ib]);
        }

#pragma unroll
        for (int j = 0; j < G; ++j) {
            // The reference adds the products of the groups with even index in one thread, the ones with odd index in the next thread, then both sums.
            float sum2[2] = {0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < ne_K_i32; ++k) {
                sum2[k % 2] += vec_dot_q8_0_q8_1_impl<float, 1>(&K_i32[k], &Q_i32[j][k], K_d[k / QI8_0], Q_d[j][k / QI8_0]);
            }
            float sum = sum2[0] + sum2[1];

            sum += __half2float(mask_reg[j]);

            KQ_max_new[j] = fmaxf(KQ_max[j], sum + FATTN_KQ_MAX_OFFSET);
            KQ_reg[j] = sum;
        }

#pragma unroll
        for (int j = 0; j < G; ++j) {
            KQ_max_new[j] = warp_reduce_max(KQ_max_new[j]);
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ_h2[j*nthreads + tid] = __half2half2(KQ_reg[j]);

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                VKQ[j][i] *= KQ_max_scale_h2;
            }
        }

        __align__(16) half2 V_d_part[V_rows_shared];
#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; ++k0) {
            const int k = warp*WARP_SIZE + k0;

            half2 KQ_k[G];
#pragma unroll
            for (int j = 0; j < G; ++j) {
                KQ_k[j] = KQ_h2[j*nthreads + k];
            }

            if (k0 % V_rows_shared == 0) {
                const int p = k0 / V_rows_shared;
                if (p == 0) {
                    // The loads of the second half of the rows go out before the work on the first half.
                    load_V_half(1);
                }
                // Put the half row in shared memory. Only this warp reads it. As with the softmax weights there is no barrier.
                __align__(16) int V_half_qs[ne_K_i32/2];
                half V_half_d[nblocks/2];
                inkling_q8_row_split<nblocks/2>(V_raw[p], V_half_qs, V_half_d);
                const int slot = threadIdx.y*V_rows_shared + threadIdx.x/2;
#pragma unroll
                for (int i = 0; i < ne_K_i32/2; i += 4) {
                    ggml_cuda_memcpy_1<4*sizeof(int)>(&V_qs_shared[slot*ne_K_i32 + (threadIdx.x % 2)*(ne_K_i32/2) + i], &V_half_qs[i]);
                }
#pragma unroll
                for (int ib = 0; ib < nblocks/2; ++ib) {
                    V_d_shared[(threadIdx.y*nblocks + (threadIdx.x % 2)*(nblocks/2) + ib)*V_rows_shared + threadIdx.x/2] = __half2half2(V_half_d[ib]);
                }
                // The scales of the block of this thread for all rows of the pass:
#pragma unroll
                for (int i = 0; i < V_rows_shared; i += 4) {
                    ggml_cuda_memcpy_1<4*sizeof(half2)>(&V_d_part[i], &V_d_shared[(threadIdx.y*nblocks + threadIdx.x / QI8_0)*V_rows_shared + i]);
                }
            }

            // Sign extension of the 4 int8 of this thread with shifts of 16 bit pairs. This gives the values 0 and 2 in one pair and the values 1 and 3 in the other.
            const int qs_i32 = V_qs_shared[(threadIdx.y*V_rows_shared + k0 % V_rows_shared)*ne_K_i32 + threadIdx.x];
            v2u qs;
            __builtin_memcpy(&qs, &qs_i32, sizeof(qs));
            const v2s qs_02 = (v2s) (qs << 8) >> 8;
            const v2s qs_13 = (v2s) qs >> 8;
            // Each product has the value that dequantize_V_q8_0 gives.
            const half2 d = V_d_part[k0 % V_rows_shared];
            const half2 tmp[2] = {d * make_half2(qs_02[0], qs_02[1]), d * make_half2(qs_13[0], qs_13[1])};
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < G; ++j) {
                    VKQ[j][i] += tmp[i]*KQ_k[j];
                }
            }
        }
    }

    // Only the first warp takes the sinks, as with one query column per block.
    if (sinks && split == 0 && warp == 0) {
#pragma unroll
        for (int j = 0; j < G; ++j) {
            const float sink = ((const float *) sinks)[head0 + j];

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                VKQ[j][i] *= KQ_max_scale_h2;
            }
        }
    }

    __shared__ float KQ_max_shared[HB][G][WARP_SIZE];
    __shared__ float KQ_sum_shared[HB][G][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < G; ++j) {
        if (warp == 0) {
            KQ_max_shared[head_block][j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[head_block][j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < G; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[head_block][j][warp] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < G; ++j) {
        float kqmax_new = KQ_max_shared[head_block][j][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j] - kqmax_new);
        KQ_max[j] = kqmax_new;

        half2 * VKQ_tmp = (half2 *) KQ + warp*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            VKQ[j][i] *= kqmax_scale_h2;
        }
        // Back to the order of the values in the row.
        const half2 VKQ_out[2] = {
            make_half2(__low2half(VKQ[j][0]),  __low2half(VKQ[j][1])),
            make_half2(__high2half(VKQ[j][0]), __high2half(VKQ[j][1]))};
        ggml_cuda_memcpy_1<2*sizeof(half2)>(VKQ_tmp + threadIdx.x*2, VKQ_out);

        KQ_sum[j] *= kqmax_scale;
        KQ_sum[j] = warp_reduce_sum(KQ_sum[j]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[head_block][j][warp] = KQ_sum[j];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j] = KQ_sum_shared[head_block][j][threadIdx.x];
            KQ_sum[j] = warp_reduce_sum(KQ_sum[j]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
                    dst_val += float(KQ[w*D + i0 + tid]);
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
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(V_DOT2_F32_F16_AVAILABLE) && defined(RDNA)
}

#endif // defined(GGML_USE_HIP)

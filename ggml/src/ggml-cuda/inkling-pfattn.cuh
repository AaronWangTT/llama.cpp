#pragma once

// Include after fattn-tile.cuh: it has the tile config and the tile load functions, and it has no include guard.
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn.cuh"

#if defined(GGML_USE_HIP)

// Kernel variants, bits of pf:
// FAR: KV_rows_ptr is the buffer of inkling_pfmask_24. A set flag means that every mask value of the tile is the far bias of its (head, query).
// FAST: same results with less work. The softmax weights go to shared memory as pairs, so the V loop needs no broadcast. One K.Q step goes over all accumulators before the next step. A warp with no new maximum in a tile does not reduce the maxima and does not rescale: the scale would be expf(0) == 1.
// ZERO: skip work that adds +0 to the accumulators. The V loop skips a K row if its softmax weight is +0 in half precision for the 8 queries of the warp. A block skips a tile that none of its queries can see. This needs finite K, Q and V. Only the sign of a result that is zero can differ.
#define INKLING_PF_FAR  1
#define INKLING_PF_FAST 2
#define INKLING_PF_ZERO 4

// Prefill attention, head size 128, F16 K and V, one block per 64 queries of one head, mask with one bias per head.
// For each query the float operations and their order are the same as in flash_attn_tile<128, 128, 64, 1, false>.
// The caller makes sure that there is a mask, that max_bias is 0 and that ne11 is a multiple of 64.
template<bool skip_empty_prefix, int pf>
__launch_bounds__(ggml_cuda_fattn_tile_get_nthreads(128, 128, 64), ggml_cuda_fattn_tile_get_occupancy(128, 128, 64))
static __global__ void inkling_pfattn_tile(
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
#if defined(FLASH_ATTN_AVAILABLE) && defined(FAST_FP16_AVAILABLE) && defined(V_DOT2_F32_F16_AVAILABLE) && defined(RDNA)
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    const int  * GGML_CUDA_RESTRICT aux      = KV_rows_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    GGML_UNUSED_VARS(logit_softcap, ne00, ne03, ne10, ne13, ne31, aux);

    constexpr int DKQ       = 128;
    constexpr int DV        = 128;
    constexpr int ncols     = 64;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols) / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_get_nbatch_K (DKQ, DV, ncols);
    constexpr int cpw       = ncols/nwarps; // Q columns per warp.
    constexpr int nrow      = nbatch_fa/warp_size; // K rows per thread.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    static_assert(nwarps == 8 && nbatch_fa == 64 && nbatch_K == 64 && cpw == 2*cpy_ne && cpy_ne == 4, "the kernel is written for this tile config");

    const int col_Q_0 = blockIdx.x * ncols; // Index of the first Q column for this block to work on.

    const int sequence = blockIdx.z / ne02;
    const int head0 = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);
    const half2 * K_h2 = (const half2 *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));

    const half * maskh = (const half *) (mask + nb33*(sequence % ne33) + nb32*(head0 % ne32));

    const int stride_K2   = nb11 / sizeof(half2);
    const int stride_V2   = nb21 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const float slope = get_alibi_slope(max_bias, head0, n_head_log2, m0, m1);

    // Q_tmp == Q data for the lifetime of the kernel. KV_tmp == fragments of K/V data. KQ == softmax weights between the two products.
    constexpr int KQ_ne = (pf & INKLING_PF_FAST) ? 2 : 1; // half values per softmax weight
    __shared__ half2 Q_tmp[ncols * DKQ/2];
    __shared__ __align__(16) half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne)];
    __shared__ __align__(16) half  KQ[ncols * nbatch_fa * KQ_ne];
    __align__(16) half2 VKQ[cpw * ((DV/2)/warp_size)] = {{0.0f, 0.0f}};

    float KQ_max[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max[jc0] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    ggml_cuda_pdl_sync();

    // Load Q data, convert to FP16:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + threadIdx.y*cpw;

        __align__(16) float tmp_f[cpy_ne] = {0.0f};
        ggml_cuda_memcpy_1<sizeof(tmp_f)>
            (tmp_f, &Q_f[fastmodulo(col_Q_0 + jc, ne01)*(nb01/sizeof(float)) + threadIdx.x*cpy_ne]);

#pragma unroll
        for (int i1 = 0; i1 < cpy_ne; ++i1) {
            tmp_f[i1] *= scale;
        }

        __align__(16) half2 tmp_h2[cpy_ne/2];
#pragma unroll
        for (int i1 = 0; i1 < cpy_ne; i1 += 2) {
            tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
        }
        ggml_cuda_memcpy_1<sizeof(tmp_h2)>(&Q_tmp[jc*(DKQ/2) + threadIdx.x*(cpy_ne/2)], tmp_h2);
    }

    // The bias of the far tiles, one per Q column of the block. It stays in shared memory to keep registers free.
    __shared__ __align__(16) float bias_far[(pf & INKLING_PF_FAR) ? ncols : cpy_ne];
    if constexpr (pf & INKLING_PF_FAR) {
        if (threadIdx.x < cpw) {
            const int jc = threadIdx.x + threadIdx.y*cpw;
            bias_far[jc] = slope*((const float *) aux)[head0*INKLING_PF_AUX_QUERIES + fastmodulo(col_Q_0 + jc, ne01)];
        }
    }

    __syncthreads();

    // flash_attn_combine_results adds the parts of all blocks to +0, so the sign of a zero part has no effect.
    [[maybe_unused]] const bool zero_skip = gridDim.y != 1;

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    int k_VKQ_0 = blockIdx.y*nbatch_fa;
    // True if the mask tile of this block at K row k is all -inf, the same answer in all threads. Same test as in flash_attn_tile.
    // A far tile is visible and its mask values are not written.
    [[maybe_unused]] const auto tile_empty = [&] __device__ (const int k) -> bool {
        if constexpr (pf & INKLING_PF_FAR) {
            if (aux[INKLING_PF_AUX_FLAGS + (k/nbatch_fa)*(INKLING_PF_AUX_QUERIES/ncols) + blockIdx.x] != 0) { return false; }
        }
        const int lane = threadIdx.y*warp_size + threadIdx.x;
        const half * m = maskh + int64_t(col_Q_0)*stride_mask;
        if (reinterpret_cast<const uint16_t *>(m)[k] != 0xfc00u) { return false; }
        uint32_t diff = 0;
#pragma unroll
        for (int off = 0; off < ncols*(nbatch_fa/2); off += nwarps*warp_size) {
            const int i = off + lane;
            if (i < ncols*(nbatch_fa/2)) {
                uint32_t pair;
                ggml_cuda_memcpy_1<4>(&pair, m + int64_t(i/(nbatch_fa/2))*stride_mask + k + 2*(i%(nbatch_fa/2)));
                diff |= pair ^ 0xfc00fc00u;
            }
        }
        return !__syncthreads_or(diff != 0);
    };
    if constexpr (skip_empty_prefix) {
        // Skip leading tiles of this split that no query of the block can see: no accumulator has a value yet.
        while (k_VKQ_0 <= k_VKQ_max - nbatch_fa && tile_empty(k_VKQ_0)) {
            k_VKQ_0 += gridDim.y*nbatch_fa;
        }
    }
    for (; k_VKQ_0 <= k_VKQ_max - nbatch_fa; k_VKQ_0 += gridDim.y*nbatch_fa) {
        if constexpr (pf & INKLING_PF_ZERO) {
            // An empty tile leaves the maxima and the sums as they are (expf(0) == 1), and all its V rows have the weight +0.
            if (zero_skip && tile_empty(k_VKQ_0)) { continue; }
        }

        float KQ_max_new[cpw];
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            KQ_max_new[jc0] = KQ_max[jc0];
        }

        float KQ_acc[nrow * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

        // The tile flag is the same for all threads. The load goes out here, the K.Q product hides its latency.
        bool far = false;
        if constexpr (pf & INKLING_PF_FAR) {
            far = aux[INKLING_PF_AUX_FLAGS + (k_VKQ_0/nbatch_fa)*(INKLING_PF_AUX_QUERIES/ncols) + blockIdx.x] != 0;
        }

        // KQ = K @ Q matrix multiplication:
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < DKQ; k_KQ_0 += nbatch_K) {
            flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, false>
                (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, nbatch_fa);
            __syncthreads();

#pragma unroll
            for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
                __align__(16) half2 K_k[nrow][cpy_ne];
                __align__(16) half2 Q_k[cpw][cpy_ne];
#pragma unroll
                for (int i = 0; i < nrow; ++i) {
                    const int i_KQ = i*warp_size + threadIdx.x;
                    ggml_cuda_memcpy_1<cpy_nb>(&K_k[i], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
                }
#pragma unroll
                for (int jc0 = 0; jc0 < cpw; ++jc0) {
                    const int jc = jc0 + threadIdx.y*cpw;
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
                }

                if constexpr (pf & INKLING_PF_FAST) {
                    // Each accumulator takes its terms in the same order. Two steps of one accumulator are 16 instructions apart.
#pragma unroll
                    for (int k = 0; k < cpy_ne; ++k) {
#pragma unroll
                        for (int i = 0; i < nrow; ++i) {
#pragma unroll
                            for (int jc0 = 0; jc0 < cpw; ++jc0) {
                                ggml_cuda_mad(KQ_acc[i*cpw + jc0], K_k[i][k], Q_k[jc0][k]);
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < nrow; ++i) {
#pragma unroll
                        for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                            for (int k = 0; k < cpy_ne; ++k) {
                                ggml_cuda_mad(KQ_acc[i*cpw + jc0], K_k[i][k], Q_k[jc0][k]);
                            }
                        }
                    }
                }
            }

            if (k_KQ_0 + nbatch_K < DKQ) {
                __syncthreads(); // Sync not needed on last iteration.
            }
        }

        // Apply mask, update KQ_max:
        if (far) {
            __align__(16) float bias_k[cpw];
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; jc0 += cpy_ne) {
                ggml_cuda_memcpy_1<cpy_nb>(&bias_k[jc0], &bias_far[threadIdx.y*cpw + jc0]);
            }
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int i = 0; i < nrow; ++i) {
                    KQ_acc[i*cpw + jc0] += bias_k[jc0];
                }
            }
        } else {
            // All loads go out before the first use.
            half mask_k[nrow][cpw];
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
                const int j = fastmodulo(col_Q_0 + jc0 + threadIdx.y*cpw, ne01);
#pragma unroll
                for (int i = 0; i < nrow; ++i) {
                    mask_k[i][jc0] = maskh[j*stride_mask + k_VKQ_0 + i*warp_size + threadIdx.x];
                }
            }
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int i = 0; i < nrow; ++i) {
                    KQ_acc[i*cpw + jc0] += slope*__half2float(mask_k[i][jc0]);
                }
            }
        }
        // False if no query of the warp has a new maximum in this tile. Then the reduction gives the old maxima and the scale is expf(0) == 1.
        bool rescale = true;
        if constexpr (pf & INKLING_PF_FAST) {
            bool larger = false;
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int i = 0; i < nrow; ++i) {
                    KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[i*cpw + jc0] + FATTN_KQ_MAX_OFFSET);
                }
                larger |= KQ_max_new[jc0] != KQ_max[jc0];
            }
            rescale = __ballot(larger) != 0;
            if (rescale) {
#pragma unroll
                for (int jc0 = 0; jc0 < cpw; ++jc0) {
                    KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
                }
            }
        } else {
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int i = 0; i < nrow; ++i) {
                    KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[i*cpw + jc0] + FATTN_KQ_MAX_OFFSET);
                }
                KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
            }
        }

        __syncthreads();

        // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
        [[maybe_unused]] uint32_t rows_used[nrow];
        {
            [[maybe_unused]] uint32_t nonzero[nrow] = {0};

            // One 16 byte store per K row and group of Q columns.
            constexpr int cpg = cpw/KQ_ne;
#pragma unroll
            for (int g = 0; g < KQ_ne; ++g) {
                __align__(16) half tmp[nrow][cpw];

#pragma unroll
                for (int jc = g*cpg; jc < (g + 1)*cpg; ++jc) {
                    float KQ_max_scale = 1.0f;
                    if (rescale) {
                        KQ_max_scale = expf(KQ_max[jc] - KQ_max_new[jc]);
                    }
                    KQ_max[jc] = KQ_max_new[jc];

                    float KQ_sum_add = 0.0f;
#pragma unroll
                    for (int i = 0; i < nrow; ++i) {
                        const float val = expf(KQ_acc[i*cpw + jc] - KQ_max[jc]);
                        KQ_sum_add += val;
                        const half val_h = val;
#pragma unroll
                        for (int l = 0; l < KQ_ne; ++l) {
                            tmp[i][KQ_ne*(jc - g*cpg) + l] = val_h;
                        }
                        if constexpr (pf & INKLING_PF_ZERO) {
                            nonzero[i] |= __half_as_ushort(val_h);
                        }
                    }
                    KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

                    if (rescale) {
                        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                        for (int i0 = 0; i0 < DV/2; i0 += warp_size) {
                            VKQ[jc*((DV/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
                        }
                    }
                }

#pragma unroll
                for (int i = 0; i < nrow; ++i) {
                    const int i_KQ = i*warp_size + threadIdx.x;
                    ggml_cuda_memcpy_1<cpy_nb>(KQ + (threadIdx.y*nbatch_fa + i_KQ)*(cpw*KQ_ne) + g*cpw, tmp[i]);
                }
            }
            if constexpr (pf & INKLING_PF_ZERO) {
                // One bit per K row of the tile, the same value in all threads of the warp.
#pragma unroll
                for (int i = 0; i < nrow; ++i) {
                    rows_used[i] = zero_skip ? uint32_t(__ballot(nonzero[i] != 0)) : 0xFFFFFFFFu;
                }
            }
        }

        // VKQ = V @ KQ matrix multiplication:
        constexpr int nbatch_V = nbatch_K * nbatch_fa / DV; // Number of V rows that fit in SRAM for K.
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
            flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, false>
                (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, nbatch_fa - k0);
            __syncthreads();

#pragma unroll
            for (int k1 = 0; k1 < nbatch_V; ++k1) {
                if constexpr (pf & INKLING_PF_ZERO) {
                    // V*0 is a zero for a finite V, so the accumulators keep their values.
                    if (((rows_used[(k0 + k1)/warp_size] >> ((k0 + k1) % warp_size)) & 1) == 0) { continue; }
                }
                __align__(16) half2 V_k[(DV/2)/warp_size];
                __align__(16) half2 KQ_k[cpw];

                ggml_cuda_memcpy_1<sizeof(V_k)>(V_k, &KV_tmp[k1*(DV/2) + threadIdx.x*((DV/2)/warp_size)]);
                if constexpr (pf & INKLING_PF_FAST) {
#pragma unroll
                    for (int l = 0; l < KQ_ne; ++l) {
                        ggml_cuda_memcpy_1<cpy_nb>(&KQ_k[l*cpy_ne], KQ + (threadIdx.y*nbatch_fa + k0 + k1)*(cpw*KQ_ne) + l*cpw);
                    }
                } else {
                    __align__(16) half tmp[cpw];
                    ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, KQ + (threadIdx.y*nbatch_fa + k0 + k1)*cpw);
#pragma unroll
                    for (int jc = 0; jc < cpw; ++jc) {
                        KQ_k[jc] = __half2half2(tmp[jc]);
                    }
                }

#pragma unroll
                for (int i0 = 0; i0 < DV/2; i0 += warp_size) {
#pragma unroll
                    for (int jc = 0; jc < cpw; ++jc) {
                        VKQ[jc*((DV/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc];
                    }
                }
            }

            __syncthreads();
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<warp_size>(KQ_sum[jc0]);
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const float sink = ((const float *) sinks)[head0];

            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = expf(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DV/2; i0 += warp_size) {
                VKQ[jc0*((DV/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
        }
    }

    // Write back results:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int j = jc0 + threadIdx.y*cpw;

        if (col_Q_0 + j >= int(ne01.z)) {
            return;
        }

        const float scale = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;

        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + j)*ne02 + head0)*gridDim.y + blockIdx.y;

        constexpr int cpy_ne_D = (DV/2)/warp_size;
        __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
        for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
            tmp[i1] = __half22float2(VKQ[jc0*((DV/2)/warp_size) + i1]);
            tmp[i1].x *= scale;
            tmp[i1].y *= scale;
        }
        ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + threadIdx.x*(2*cpy_ne_D)], tmp);

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
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
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(FAST_FP16_AVAILABLE) && defined(V_DOT2_F32_F16_AVAILABLE) && defined(RDNA)
}

#endif // defined(GGML_USE_HIP)

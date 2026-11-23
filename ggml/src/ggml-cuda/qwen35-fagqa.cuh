#pragma once

// Include after fattn-mma-f16.cuh: it has the WMMA tiles and the config of flash_attn_ext_f16, and it has no include guard.
#include "common.cuh"
#include "fattn-common.cuh"

#if defined(GGML_USE_HIP)

#if defined(AMD_WMMA_256B_AVAILABLE)
// The part of one KV head that one block of flash_attn_ext_f16<256, 256, 4, 4, false, false, false> takes: the K/V row tiles kb0_start to kb0_stop.
// The WMMA and float operations and their order are the ones of flash_attn_ext_f16_process_tile, so each value of the result has the same bits.
// The loads are not the same. That kernel waits for each 16 byte load of a K or V row before it sends the next one, and all rows go through shared memory.
// Here a warp sends the loads of NB WMMA steps of K and of NB rows of V before it waits. K goes to the WMMA registers, V to the rows of the warp in the shared tile.
// Q is in shared memory and not in registers: with fewer registers in use the compiler has no need to put values in scratch memory, which was slow.
// The two warps of a block need no barrier in the loop over the K/V rows.
template<bool needs_fixup, bool is_fixup, int NB>
static __device__ __forceinline__ void qwen35_fagqa_wmma_tile(
        const float2  * const __restrict__ Q_f2,
        const half2   * const __restrict__ K_h2,
        const half2   * const __restrict__ V_h2,
        const half    * const __restrict__ mask_h,
        const int32_t * const __restrict__ rows,
        float2        * const __restrict__ dstk,
        float2        * const __restrict__ dstk_fixup,
        half2         * const __restrict__ tile,
        const float scale,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int kb0_start,
        const int kb0_stop) {
    constexpr int DKQ    = 256;
    constexpr int DV     = 256;
    constexpr int ncols2 = 4;
    constexpr int ncols  = 16;
    using     T_A_KQ     = typename mma_tile_sizes<DKQ, ncols>::T_A_KQ;
    using     T_B_KQ     = typename mma_tile_sizes<DKQ, ncols>::T_B_KQ;
    using     T_C_KQ     = typename mma_tile_sizes<DKQ, ncols>::T_C_KQ;
    using     T_A_VKQ    = typename mma_tile_sizes<DKQ, ncols>::T_A_VKQ;
    using     T_B_VKQ    = typename mma_tile_sizes<DKQ, ncols>::T_B_VKQ;
    using     T_C_VKQ    = typename mma_tile_sizes<DKQ, ncols>::T_C_VKQ;

    constexpr int warp_size      = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps         = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols) / warp_size;
    constexpr int cols_per_warp  = T_B_KQ::I;
    constexpr int np             = nwarps * cols_per_warp/ncols; // Number of parallel warps per Q column.
    constexpr int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);

    constexpr int stride_tile_V = DV/2;   // rows of V in the shared tile, no padding
    constexpr int stride_tile_Q = DKQ/2;  // rows of Q after the rows of V
    constexpr int tile_stride   = nbatch_combine + 4;

    half2 * const tile_V = tile;
    half2 * const tile_Q = tile + nbatch_fa*stride_tile_V;

    // The config of flash_attn_ext_f16<256, 256, 4, 4> that this code is made for.
    static_assert(warp_size == 32 && nwarps == 2 && np == 2 && cols_per_warp == 16 && nbatch_fa == 32, "other split of the KV rows");
    static_assert(nbatch_V2 == DV/2 && nbatch_combine == DV/2, "other tile sizes");
    static_assert(nwarps*cols_per_warp*tile_stride <= nbatch_fa*stride_tile_V + ncols2*stride_tile_Q, "shared tile too small for the result");
    static_assert(T_A_KQ::I == 16 && T_A_KQ::J == 8 && T_B_KQ::J == 8 && T_C_KQ::ne == 8 && T_A_VKQ::I == 32 && T_C_VKQ::J == 16, "other WMMA tiles");
    static_assert(DKQ/2 == T_A_KQ::I*T_A_KQ::J, "a warp must have as many WMMA steps of K as rows of V");
    static_assert(NB == 2 || NB == 4 || NB == 8, "bad NB");
    static_assert(ncols2 % nwarps == 0, "each warp must put the same number of Q rows in shared memory");

    const int lane   = threadIdx.x % T_A_KQ::I;             // K row of this thread in the rows of the warp, also its Q column.
    const int i_warp = (threadIdx.y % np)*T_A_KQ::I;         // First row of this warp in a tile of nbatch_fa rows.

    T_C_VKQ VKQ_C[DV/(2*T_C_VKQ::J)];

    float KQ_rowsum = 0.0f;
    float KQ_max    = -FLT_MAX/2.0f;

    // Q of the ncols2 query heads to shared memory, each warp takes ncols2/nwarps heads and each thread 16 bytes of a row.
    {
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int c0 = 0; c0 < ncols2; c0 += nwarps) {
            const int c = c0 + threadIdx.y;
            __align__(16) float2 tmp[4];
            ggml_cuda_memcpy_1<16>(tmp + 0, Q_f2 + c*stride_Q2 + threadIdx.x*4 + 0);
            ggml_cuda_memcpy_1<16>(tmp + 2, Q_f2 + c*stride_Q2 + threadIdx.x*4 + 2);
            __align__(16) half2 q[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                q[l] = scale_h2 * make_half2(tmp[l].x, tmp[l].y);
            }
            ggml_cuda_memcpy_1<16>(tile_Q + c*stride_tile_Q + threadIdx.x*4, q);
        }
    }

    __syncthreads();

    // The WMMA tiles have 16 Q columns. Column c takes head c % ncols2, the kernel that this one replaces has zeros from column ncols2.
    // No value of these columns goes to the result.
    const half2 * const Q_lane = tile_Q + (lane % ncols2)*stride_tile_Q;

    // The last tile is done also if kb0_start >= kb0_stop, as in flash_attn_ext_f16_process_tile.
    for (int kb0 = kb0_start; ; ++kb0) {
        const int k_warp = kb0*nbatch_fa + i_warp; // First list entry of this warp.

        // K row of this thread, V rows and mask of the warp. A negative row is padding, its mask is -inf.
        const int row_K = max(rows[k_warp + lane], 0);
        __align__(16) int row_V[T_A_KQ::I];
#pragma unroll
        for (int i = 0; i < T_A_KQ::I; i += 4) {
            ggml_cuda_memcpy_1<16>(row_V + i, rows + k_warp + i);
        }
        __align__(16) half mask_w[T_A_KQ::I];
#pragma unroll
        for (int i = 0; i < T_A_KQ::I; i += 8) {
            ggml_cuda_memcpy_1<16>(mask_w + i, mask_h + k_warp + i);
        }

        // Calculate tile of KQ. On the way the V rows of this warp go to its rows of the shared tile, each thread takes 16 bytes of a row.
        T_C_KQ KQ_C;
        const half2 * K_row = K_h2 + int64_t(row_K)*stride_K;
#pragma unroll
        for (int i0 = 0; i0 < T_A_KQ::I; i0 += NB) {
            T_A_KQ K_A[NB];
            __align__(16) half2 V_tmp[NB][4];
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                ggml_cuda_memcpy_1<16>(K_A[b].x + 0, K_row + (i0 + b)*T_A_KQ::J + 0);
                ggml_cuda_memcpy_1<16>(K_A[b].x + 4, K_row + (i0 + b)*T_A_KQ::J + 4);
            }
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                ggml_cuda_memcpy_1<16>(V_tmp[b], V_h2 + int64_t(max(row_V[i0 + b], 0))*stride_V + threadIdx.x*4);
            }

            // No code, only a use of each load. It keeps the compiler from a wait after each load: all loads above go out before the first wait.
            if constexpr (NB == 2) {
                asm volatile("" : : "v"(K_A[0].x[0]), "v"(K_A[0].x[4]), "v"(K_A[1].x[0]), "v"(K_A[1].x[4]), "v"(V_tmp[0][0]), "v"(V_tmp[1][0]));
            } else if constexpr (NB == 4) {
                asm volatile("" : :
                    "v"(K_A[0].x[0]), "v"(K_A[0].x[4]), "v"(K_A[1].x[0]), "v"(K_A[1].x[4]), "v"(K_A[2].x[0]), "v"(K_A[2].x[4]), "v"(K_A[3].x[0]), "v"(K_A[3].x[4]),
                    "v"(V_tmp[0][0]), "v"(V_tmp[1][0]), "v"(V_tmp[2][0]), "v"(V_tmp[3][0]));
            } else {
                asm volatile("" : :
                    "v"(K_A[0].x[0]), "v"(K_A[0].x[4]), "v"(K_A[1].x[0]), "v"(K_A[1].x[4]), "v"(K_A[2].x[0]), "v"(K_A[2].x[4]), "v"(K_A[3].x[0]), "v"(K_A[3].x[4]),
                    "v"(K_A[4].x[0]), "v"(K_A[4].x[4]), "v"(K_A[5].x[0]), "v"(K_A[5].x[4]), "v"(K_A[6].x[0]), "v"(K_A[6].x[4]), "v"(K_A[7].x[0]), "v"(K_A[7].x[4]),
                    "v"(V_tmp[0][0]), "v"(V_tmp[1][0]), "v"(V_tmp[2][0]), "v"(V_tmp[3][0]), "v"(V_tmp[4][0]), "v"(V_tmp[5][0]), "v"(V_tmp[6][0]), "v"(V_tmp[7][0]));
            }

#pragma unroll
            for (int b = 0; b < NB; ++b) {
                ggml_cuda_memcpy_1<16>(tile_V + (i_warp + i0 + b)*stride_tile_V + threadIdx.x*4, V_tmp[b]);
            }
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                T_B_KQ Q_B;
                ggml_cuda_memcpy_1<16>(Q_B.x + 0, Q_lane + (i0 + b)*T_B_KQ::J + 0);
                ggml_cuda_memcpy_1<16>(Q_B.x + 4, Q_lane + (i0 + b)*T_B_KQ::J + 4);
                mma(KQ_C, K_A[b], Q_B);
            }
        }

        // Value l of KQ_C is row 2*l + threadIdx.x/16 of the warp, see T_C_KQ::get_j.
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            KQ_C.x[l] += __half2float(threadIdx.x < 16 ? mask_w[2*l] : mask_w[2*l + 1]);
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        float KQ_max_new = KQ_max;
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            KQ_max_new = fmaxf(KQ_max_new, KQ_C.x[l] + FATTN_KQ_MAX_OFFSET);
        }
        // Values per KQ column are spread across 2 threads:
        KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, 16, warp_size));

        float KQ_rowsum_add = 0.0f;
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            KQ_C.x[l] = expf(KQ_C.x[l] - KQ_max_new);
            KQ_rowsum_add += KQ_C.x[l];
        }

        {
            const float KQ_max_diff = KQ_max - KQ_max_new;
            float KQ_max_scale = expf(KQ_max_diff);
            KQ_max = KQ_max_new;

            *((uint32_t *) &KQ_max_scale) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum = KQ_max_scale*KQ_rowsum + KQ_rowsum_add;

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        }

        // Convert KQ C tile into B tile for VKQ calculation:
        const T_B_VKQ B = get_half2(KQ_C);

        // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < DV; i_VKQ_0 += T_A_VKQ::I) {
            // As load_ldmatrix_trans_swizzled: value k of the first half of A is element i_VKQ_0 + 2*lane of V row k, the second half has element i_VKQ_0 + 2*lane + 1.
            T_A_VKQ A;
#pragma unroll
            for (int l0 = 0; l0 < T_A_VKQ::ne/2; ++l0) {
                const half2 tmp0 = tile_V[(i_warp + 2*l0 + 0)*stride_tile_V + i_VKQ_0/2 + lane];
                const half2 tmp1 = tile_V[(i_warp + 2*l0 + 1)*stride_tile_V + i_VKQ_0/2 + lane];

                A.x[l0]                 =  __lows2half2(tmp0, tmp1);
                A.x[l0 + T_A_VKQ::ne/2] = __highs2half2(tmp0, tmp1);
            }
            mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B);
        }

        if (kb0 >= kb0_stop - 1) {
            break;
        }
    }

    // Finally, sum up partial KQ rowsums. The partial sums are spread across 2 threads.
    KQ_rowsum += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum, 16, warp_size);

    // Combine VKQ accumulator values of the np warps as flash_attn_ext_f16 does it, in the same shared memory layout.
    // This takes the whole shared tile, so the other warp must be done with its rows.
    __syncthreads();

    {
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max, KQ_rowsum);

        if (threadIdx.x / 16 < 1) {
            ((float2 *) tile)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();
    }

    {
        float KQ_cmn;
        float KQ_cms;
        float KQ_crs;

        const int jc_meta = threadIdx.y*cols_per_warp + threadIdx.x;
        float2 * const meta_ptr = ((float2 *) tile) + jc_meta*(tile_stride/2) + nbatch_combine/2;

        if (threadIdx.y % np == 0) {
            // Combine the meta data for parallel warps via shared memory.
            const float2 meta = meta_ptr[0];

            KQ_cmn = meta.x; // KQ combine max new, max between all parallel warps.
            KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, 16, warp_size));

            KQ_cms = expf(meta.x - KQ_cmn);

            KQ_crs = KQ_cms*meta.y; // KQ combine rowsum, scaled sum of all parallel warps.
            KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, 16, warp_size);
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // Write back combined meta data: combined KQ max scale + rowsum.
            meta_ptr[0] = make_float2(KQ_cms, KQ_crs);

            // Combined KQ max + rowsum.
            if (needs_fixup && threadIdx.x < cols_per_warp) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
            if (is_fixup && threadIdx.x < cols_per_warp) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
        }
    }

    {
        const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
        for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const int j = j0 + T_C_VKQ::get_i(l);
                const int k = k1 + T_C_VKQ::get_j(l);

                tile[j*tile_stride + k] = VKQ_C[k1/T_C_VKQ::J].x[l];
            }
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

            // Only the columns of the ncols2 query heads, flash_attn_stream_k_fixup_uniform reads no other column.
#pragma unroll
            for (int jc_dst = 0; jc_dst < ncols2; ++jc_dst) {
                const float * meta_j = (const float *) tile + jc_dst*tile_stride + nbatch_combine;
#pragma unroll
                for (int k0 = 0; k0 < nbatch_combine; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                    for (int ip = 0; ip < np; ++ip) {
                        const float KQ_crs = meta_j[ip*cols_per_warp * tile_stride + 0];
                        const float2 dstk_val_add = __half22float2(tile[(jc_dst + ip*cols_per_warp) * tile_stride + k]);
                        dstk_val.x += dstk_val_add.x*KQ_crs;
                        dstk_val.y += dstk_val_add.y*KQ_crs;
                    }

                    if (is_fixup) {
                        dstk_fixup_data[jc_dst*(DV/2) + k] = dstk_val;
                    } else {
                        dstk[jc_dst*(DV/2) + k] = dstk_val;
                    }
                }
            }
        }
    }
}
#endif // defined(AMD_WMMA_256B_AVAILABLE)

// One-token attention over a list of K/V rows for 4 query heads per KV head, head size 256, F16 K and V, one mask row for all heads.
// It takes the place of flash_attn_ext_f16<256, 256, 4, 4, false, false, false> with the Stream-K plan of launch_fattn: same grid, same KV rows per block.
// The caller makes sure that each block has its KV rows in one KV head and that a KV head has more than one block.
// The result has the same bits after flash_attn_stream_k_fixup_uniform, see qwen35_fagqa_wmma_tile.
template<int NB> // NB == WMMA steps of K and rows of V whose loads go out together
__launch_bounds__(64, 2)
static __global__ void qwen35_fagqa_wmma(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const int32_t ne02, const int32_t ne03, const int32_t nb02, const int32_t nb03,
        const int32_t ne11, const int32_t ne12,
        const int32_t nb11, const int32_t nb12,
        const int32_t nb21, const int32_t nb22,
        const int32_t ne33, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_256B_AVAILABLE)
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    const int  * GGML_CUDA_RESTRICT KV_rows  = KV_rows_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    constexpr int DV        = 256;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(256, DV, 16);

    // The V rows of nbatch_fa K/V rows and the Q rows of the 4 query heads.
    __shared__ half2 tile[nbatch_fa*(DV/2) + 4*(256/2)];

    const int gqa_ratio = ne02 / ne12;
    const int iter_k    = ne11 / nbatch_fa;

    // kbc == k block continuous, current index in continuous ijk space.
    const int kbc      = int64_t(blockIdx.x + 0)*(iter_k*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*ne12*ne03) / gridDim.x;

    // kb0 == k start index when in the output tile.
    const int kb0_start = kbc % iter_k;
    int       kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);
    const bool last     = kb0_stop == iter_k; // This block finishes the KV head, its part goes to dst.

    const int sequence =  kbc /(iter_k*ne12);
    const int z_KV     = (kbc - iter_k*ne12 * sequence)/iter_k;
    const int zt_Q     = z_KV*gqa_ratio; // First query head of the KV head.

    const float2  * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2   * K_h2   = (const half2  *) (K + nb12*z_KV);
    const half2   * V_h2   = (const half2  *) (V + nb22*z_KV);
    const half    * mask_h = (const half   *) (mask + nb33*(sequence % ne33));
    const int32_t * rows   = KV_rows + int64_t(sequence)*ne11;
    float2        * dstk   = ((float2 *) dst) + (sequence*ne02 + zt_Q) * (DV/2);

    kb0_stop = min(kb0_stop, KV_max[sequence] / nbatch_fa);

    const int stride_Q2 = nb02 / sizeof(float2);
    const int stride_K  = nb11 / sizeof(half2);
    const int stride_V  = nb21 / sizeof(half2);

    if (last) {
        qwen35_fagqa_wmma_tile<true, false, NB>
            (Q_f2, K_h2, V_h2, mask_h, rows, dstk, dst_meta, tile, scale, stride_Q2, stride_K, stride_V, kb0_start, kb0_stop);
    } else {
        qwen35_fagqa_wmma_tile<false, true, NB>
            (Q_f2, K_h2, V_h2, mask_h, rows, dstk, dst_meta, tile, scale, stride_Q2, stride_K, stride_V, kb0_start, kb0_stop);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        ne02, ne03, nb02, nb03,
        ne11, ne12,
        nb11, nb12,
        nb21, nb22,
        ne33, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_256B_AVAILABLE)
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // fattn.cu

// GGML_QWEN35_FAGQA=1 takes the one-token attention of qwen35 (head size 256, 4 query heads per KV head, K/V rows) on gfx1103.
// mma_f16 says that the op goes to flash_attn_ext_f16 without the switch. The result has the same bits as with that kernel.
// GGML_QWEN35_FAGQA_CHECK=1 with GGML_CUDA_DISABLE_GRAPHS=1 also runs that kernel and logs the number of values with other bits. This is slow, it is for tests.
static bool ggml_cuda_flash_attn_ext_qwen35_fagqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool mma_f16) {
    static const bool enable = [] { const char * s = getenv("GGML_QWEN35_FAGQA"); return s && strcmp(s, "1") == 0; }();
    if (!enable) {
        return false;
    }

    constexpr int DKQ    = 256;
    constexpr int DV     = 256;
    constexpr int ncols1 = 4;
    constexpr int ncols2 = 4;
    constexpr int ncols  = ncols1*ncols2;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * rows = dst->src[5];

    // The op must be one that goes to flash_attn_ext_f16<256, 256, 4, 4, false, false, false> with its GQA path.
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (!mma_f16 || cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || !mask || !rows || dst->src[4]) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (Q->ne[0] != DKQ || K->ne[0] != DKQ || V->ne[0] != DV || Q->ne[1] != 1 || Q->ne[2] != ncols2*K->ne[2] || V->ne[2] != K->ne[2]) {
        return false;
    }
    if (rows->ne[0] % FATTN_KQ_STRIDE != 0 || rows->ne[1] != Q->ne[3] || mask->ne[0] != rows->ne[0] || mask->ne[2] != 1) {
        return false;
    }
    if (ggml_get_op_params_f32(dst, 1) != 0.0f || ggml_get_op_params_f32(dst, 2) != 0.0f) {
        return false;
    }
    // The kernel loads 16 bytes at a time.
    if (!ggml_cuda_is_aligned(Q, 16) || !ggml_cuda_is_aligned(K, 16) || !ggml_cuda_is_aligned(V, 16) || !ggml_cuda_is_aligned(mask, 16) ||
            !ggml_cuda_is_aligned(rows, 16)) {
        return false;
    }

    // The plan of launch_fattn for that kernel.
    const int nsm       = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const int nthreads  = ggml_cuda_fattn_mma_get_nthreads (DKQ, DV, ncols, cc);
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc);
    const int nwarps    = nthreads / warp_size;
    if (warp_size != WARP_SIZE || nwarps != 2 || nbatch_fa != 32 || !ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, ncols, cc) ||
            ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc) > 1 || ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols, cc)) {
        return false;
    }

    const int64_t n_kv       = rows->ne[0];
    const int     ntiles_dst = K->ne[2]*Q->ne[3]; // One output tile is one KV head of one sequence.
    const int     ntiles_KV  = n_kv / nbatch_fa;

    // Shared memory of that kernel, as in ggml_cuda_flash_attn_ext_mma_f16_case. The number of blocks comes from it.
    const int    nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int    nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int    nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const size_t nbytes_shared_KV      = nbatch_fa                                     * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q       = ncols                                         * (DKQ/2 + 4)                            * sizeof(half2);
    const size_t nbytes_shared_mask    = ncols1                                        * (nbatch_fa/2 + 4)                      * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps*std::min(ncols, get_cols_per_warp(cc)) * (nbatch_combine + 4)                   * sizeof(half2);
    const size_t nbytes_shared_total   = std::max(nbytes_shared_combine, std::max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask));

    const fattn_kernel_t ref = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, false, false>;
    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, ref, nthreads, nbytes_shared_total));
    GGML_ASSERT(max_blocks_per_sm > 0);

    const int max_blocks               = max_blocks_per_sm*nsm;
    const int tiles_nwaves             = (ntiles_dst + max_blocks - 1) / max_blocks;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);
    if (tiles_efficiency_percent >= 75) {
        return false; // No Stream-K.
    }
    const int nblocks_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
    const int nblocks     = (nblocks_raw / ntiles_dst) * ntiles_dst;
    if (nblocks <= ntiles_dst || 100 * (nblocks_raw - nblocks) / nblocks_raw > 5) {
        return false; // The blocks of that plan are not whole parts of one KV head.
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    // The check reads the result back. A stream that captures a graph cannot do this, so it needs GGML_CUDA_DISABLE_GRAPHS=1.
    static const bool check = getenv("GGML_QWEN35_FAGQA_CHECK") != nullptr && getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr;
    std::vector<float> out_ref;
    if (check) {
        ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
        out_ref.resize(ggml_nelements(dst));
        CUDA_CHECK(cudaMemcpyAsync(out_ref.data(), dst->data, ggml_nbytes(dst), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // The mask scan of launch_fattn, a list of K/V rows can end with padding.
    ggml_cuda_pool_alloc<int> KV_max(pool, Q->ne[3]);
    {
        const dim3 blocks_num_KV_max(1, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, stream),
            (const half2 *) mask->data, KV_max.ptr, int(n_kv / FATTN_KQ_STRIDE), int(Q->ne[1]), int(mask->ne[3]),
            int64_t(mask->nb[1] / sizeof(half2)), int64_t(mask->nb[3] / sizeof(half2)));
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool, size_t(nblocks) * ncols * (2 + DV/2));

    // GGML_QWEN35_FAGQA_BURST=2 or 8 sets NB of the kernel for tests, the result is the same.
    static const int burst = [] { const char * s = getenv("GGML_QWEN35_FAGQA_BURST"); return s ? atoi(s) : 4; }();
    auto kernel = qwen35_fagqa_wmma<4>;
    if (burst == 2) {
        kernel = qwen35_fagqa_wmma<2>;
    } else if (burst == 8) {
        kernel = qwen35_fagqa_wmma<8>;
    }

    const dim3 blocks_num(nblocks, 1, 1);
    const dim3 block_dim(warp_size, nwarps, 1);
    ggml_cuda_kernel_launch(kernel, ggml_cuda_kernel_launch_params(blocks_num, block_dim, 0, stream),
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
        (const int *) KV_max.ptr, (const int *) rows->data, (float *) dst->data, dst_tmp_meta.ptr, ggml_get_op_params_f32(dst, 0),
        Q->ne[2], Q->ne[3], Q->nb[2], Q->nb[3],
        n_kv, K->ne[2],
        K->nb[1], K->nb[2],
        V->nb[1], V->nb[2],
        mask->ne[3], mask->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    // The fixup of launch_fattn for blocks that are whole parts of an output tile.
    {
        const uint3 fd0 = init_fastdiv_values(K->ne[2]);
        const uint3 fd1 = init_fastdiv_values(1);
        const uint3 fd2 = init_fastdiv_values(1);

        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(ntiles_dst, ncols1, ncols2);
        ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>,
            ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, stream),
            (float *) dst->data, dst_tmp_meta.ptr,
            int(Q->ne[1]), int(Q->ne[2]), int(K->ne[2]), nblocks,
            int(Q->ne[2] / K->ne[2]), nblocks / ntiles_dst, fd0, fd1, fd2);
        CUDA_CHECK(cudaGetLastError());
    }

    if (check) {
        std::vector<float> out(out_ref.size());
        CUDA_CHECK(cudaMemcpyAsync(out.data(), dst->data, ggml_nbytes(dst), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        size_t n_diff = 0;
        float  d_max  = 0.0f;
        for (size_t i = 0; i < out.size(); ++i) {
            if (memcmp(&out[i], &out_ref[i], sizeof(float)) != 0) {
                n_diff++;
                d_max = std::max(d_max, fabsf(out[i] - out_ref[i]));
            }
        }
        static long n_calls = 0;
        static long n_calls_diff = 0;
        n_calls++;
        n_calls_diff += n_diff > 0;
        if (n_calls <= 8 || n_calls % 256 == 0 || (n_diff > 0 && n_calls_diff <= 32)) {
            GGML_LOG_INFO("%s: call %ld: %zu of %zu values differ (max %.3g), kv = %lld, sequences = %lld, blocks = %d, calls with differences so far: %ld\n",
                __func__, n_calls, n_diff, out.size(), d_max, (long long) n_kv, (long long) Q->ne[3], nblocks, n_calls_diff);
        }
    }

    return true;
}

#endif // defined(GGML_USE_HIP)

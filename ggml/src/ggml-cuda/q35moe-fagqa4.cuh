#pragma once

// Include after q35moe-fagqa.cuh: it has the Q8_0 row parts of the one-token kernel and the declaration of the WMMA path.
#include "common.cuh"
#include "fattn-common.cuh"

#if defined(GGML_USE_HIP)

#if defined(AMD_WMMA_256B_AVAILABLE)
// Shared memory of q35moe_fagqa4_wmma_tile: the A tiles of the VKQ steps, one record of SA half2 for each half2 of half a V row (16 half2 are used),
// and the rows of Q with SQ half2 (128 are used). With SA > 16 the records of each group of 16 are rotated by the group number.
// With padding the 16 lanes of a half warp are at different offsets modulo 256 bytes when they load and when they store, as in qwen35-pfattn.cuh.
template<int SA>
static __device__ __forceinline__ int q35moe_fagqa4_rec(const int r) {
    return SA > 16 ? (r & ~15) | ((r + (r >> 4)) & 15) : r;
}

// Warps of a block: the kernel that this one replaces has nwarps_ref warps, and here one warp takes G groups of 16 Q columns.
template<int ncols1, int G>
static constexpr __device__ int q35moe_fagqa4_nwarps() {
    return ggml_cuda_fattn_mma_get_nthreads(256, 256, ncols1*8) / ggml_cuda_get_physical_warp_size() / G;
}

// half2 of the shared tile. At the end it has the result of all warps in the layout of flash_attn_ext_f16, which can need more.
template<int ncols1, int G, int SA, int SQ>
static constexpr __device__ int q35moe_fagqa4_tile_ne() {
    const int ncols      = ncols1*8;
    const int nwarps_ref = ggml_cuda_fattn_mma_get_nthreads(256, 256, ncols) / ggml_cuda_get_physical_warp_size();
    const int ne_loop    = q35moe_fagqa4_nwarps<ncols1, G>()*(256/4)*SA + ncols*SQ;
    const int ne_combine = nwarps_ref*16*(ggml_cuda_fattn_mma_get_nbatch_combine(256, 256, ncols) + 4);
    return ne_loop > ne_combine ? ne_loop : ne_combine;
}

// As q35moe_store_A, for records of S half2: v0 and v1 are 4 half2 of two V rows, the records r0 to r0 + 3 get this pair of rows at place i.
template<int S>
static __device__ __forceinline__ void q35moe_fagqa4_store_A(
        half2 * __restrict__ tile_A, const int r0, const int i, const half2 * __restrict__ v0, const half2 * __restrict__ v1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        half2 * dst = tile_A + q35moe_fagqa4_rec<S>(r0 + c)*S + i;
        dst[0] =  __lows2half2(v0[c], v1[c]);
        dst[8] = __highs2half2(v0[c], v1[c]);
    }
}

// As q35moe_q8_store_A, for records of S half2: the same block a and b of two Q8_0 rows goes to the records r0 to r0 + 15 at place i.
template<int S>
static __device__ __forceinline__ void q35moe_fagqa4_q8_store_A(
        half2 * __restrict__ tile_A, const int r0, const int i, const q35moe_q8_blk & a, const q35moe_q8_blk & b) {
    const uint32_t ds = q35moe_perm_b32(b.word(0), a.word(0), 0x05040100u);
    half2 d;
    __builtin_memcpy(&d, &ds, sizeof(d));
#pragma unroll
    for (int p = 0; p < QK8_0/2; ++p) {
        const int  w  = (2 + 2*p)/4;
        const bool lo = (2 + 2*p) % 4 == 0; // the two int8 are the low half of their word
        const uint32_t x0 = q35moe_perm_b32(b.word(w), a.word(w), lo ? 0x0C040C00u : 0x0C060C02u) ^ 0x64806480u;
        const uint32_t x1 = q35moe_perm_b32(b.word(w), a.word(w), lo ? 0x0C050C01u : 0x0C070C03u) ^ 0x64806480u;
        half2 q0;
        half2 q1;
        __builtin_memcpy(&q0, &x0, sizeof(q0));
        __builtin_memcpy(&q1, &x1, sizeof(q1));
        half2 * dst = tile_A + q35moe_fagqa4_rec<S>(r0 + p)*S + i;
        dst[0] = (q0 + make_half2(-1152.0f, -1152.0f)) * d;
        dst[8] = (q1 + make_half2(-1152.0f, -1152.0f)) * d;
    }
}

// The K/V row tiles kb0_start to kb0_stop of one KV head for ncols1 query tokens of one sequence and the 8 query heads of the KV head (ne01 tokens are there).
// It takes the place of flash_attn_ext_f16_process_tile<256, 256, ncols1, 8> with K/V rows from a list: for each Q column the WMMA and float operations
// and their order are the same, so each value of the result has the same bits. That kernel has nwarps_ref warps: a warp there has 16 Q columns
// (2 tokens) and 16 of the K/V rows of a tile, and np warps have the same columns. Here a warp has the same 16 K/V rows and G groups of 16 columns.
// The loads and the use of shared memory are the ones of q35moe_fagqa_wmma_tile: a warp sends the loads of NB WMMA steps of K before it waits,
// K goes to the WMMA registers, Q is in shared memory, the values of V go to shared memory as A tiles of the VKQ steps, half of a row at a time.
// The warps of a block need no barrier in the loop over the K/V rows. With G == 2 a K/V row is loaded once for all 4 tokens.
// Q8 == K and V are Q8_0 rows, stride_K and stride_V are still in units of 4 bytes. Each thread makes the halves of dequantize_block_q8_0_f16 in registers.
// The caller gives a tile of q35moe_fagqa4_tile_ne() half2.
template<int ncols1, int G, int NB, int SA, int SQ, bool needs_fixup, bool is_fixup, bool Q8>
static __device__ __forceinline__ void q35moe_fagqa4_wmma_tile(
        const float2  * const __restrict__ Q_f2,
        const half2   * const __restrict__ K_h2,
        const half2   * const __restrict__ V_h2,
        const half    * const __restrict__ mask_h,
        const int32_t * const __restrict__ rows,
        float2        * const __restrict__ dstk,
        float2        * const __restrict__ dstk_fixup,
        half2         * const __restrict__ tile,
        const float scale,
        const int ne01,
        const int ne02,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int kb0_start,
        const int kb0_stop) {
    constexpr int DKQ    = 256;
    constexpr int DV     = 256;
    constexpr int ncols2 = 8;
    constexpr int ncols  = ncols1*ncols2;
    using     T_A_KQ     = typename mma_tile_sizes<DKQ, ncols>::T_A_KQ;
    using     T_B_KQ     = typename mma_tile_sizes<DKQ, ncols>::T_B_KQ;
    using     T_C_KQ     = typename mma_tile_sizes<DKQ, ncols>::T_C_KQ;
    using     T_A_VKQ    = typename mma_tile_sizes<DKQ, ncols>::T_A_VKQ;
    using     T_B_VKQ    = typename mma_tile_sizes<DKQ, ncols>::T_B_VKQ;
    using     T_C_VKQ    = typename mma_tile_sizes<DKQ, ncols>::T_C_VKQ;

    constexpr int warp_size      = ggml_cuda_get_physical_warp_size();
    constexpr int cols_per_warp  = T_B_KQ::I;
    constexpr int nwarps_ref     = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols) / warp_size;
    constexpr int np             = nwarps_ref * cols_per_warp/ncols; // Number of parallel warps per Q column in the kernel that this one replaces.
    constexpr int nwarps         = q35moe_fagqa4_nwarps<ncols1, G>();
    constexpr int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);

    constexpr int S             = SA;                            // half2 of one A tile record
    constexpr int stride_tile_Q = SQ;                            // rows of Q after the A tiles
    constexpr int ne_tile_A     = nwarps*(DV/4)*S;               // A tiles of all warps: one record for each half2 of half a V row
    constexpr int tile_stride   = nbatch_combine + 4;

    // The config of flash_attn_ext_f16<256, 256, ncols1, 8> that this code is made for.
    static_assert(warp_size == 32 && cols_per_warp == 16 && ncols % cols_per_warp == 0 && (ncols/cols_per_warp) % G == 0, "other split of the Q columns");
    static_assert(np >= 2 && np*T_A_KQ::I == nbatch_fa && nwarps*G == nwarps_ref, "other split of the KV rows");
    static_assert(nbatch_K2 == DKQ/2 && nbatch_V2 == DV/2, "other tile sizes");
    static_assert((DV/2) % nbatch_combine == 0 && nbatch_combine % warp_size == 0 && np*cols_per_warp % warp_size == 0, "other layout of the result");
    static_assert(T_A_KQ::I == 16 && T_A_KQ::J == 8 && T_B_KQ::J == 8 && T_C_KQ::ne == 8 && T_A_VKQ::I == 32 && T_A_VKQ::ne == 16 && T_C_VKQ::J == 16, "other WMMA tiles");
    static_assert(DKQ/2 == T_A_KQ::I*T_A_KQ::J, "a warp must have as many WMMA steps of K as rows of V");
    static_assert(T_A_KQ::I % NB == 0 && NB % 4 == 0, "a burst must have whole parts of a Q8_0 row and pairs of F16 V rows");
    static_assert(ncols % nwarps == 0, "each warp must put the same number of Q rows in shared memory");
    static_assert(S >= T_A_VKQ::ne && stride_tile_Q >= DKQ/2 && stride_tile_Q % 4 == 0, "bad strides of the shared tile");

    const int lane   = threadIdx.x % T_A_KQ::I;         // K row of this thread in the rows of the warp, also its Q column in each group.
    const int ip_w   = threadIdx.y % np;                // The parallel warp of the kernel that this one replaces whose K/V rows this warp has.
    const int g0     = (threadIdx.y / np)*G;            // First group of 16 Q columns of this warp.
    const int i_warp = ip_w*T_A_KQ::I;                  // First row of this warp in a tile of nbatch_fa rows.

    half2 * const tile_A = tile + threadIdx.y*(DV/4)*S; // A tiles of this warp
    half2 * const tile_Q = tile + ne_tile_A;

    T_C_VKQ VKQ_C[G][DV/(2*T_C_VKQ::J)];

    float KQ_rowsum[G];
    float KQ_max[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        KQ_rowsum[g] = 0.0f;
        KQ_max[g]    = -FLT_MAX/2.0f;
    }

    // Q of all columns to shared memory, each thread takes 16 bytes of a row. A column of a token that is not there takes the last token,
    // the kernel that this one replaces has zeros there. No value of these columns goes to the result.
    {
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps) {
            const int jc = jc0 + threadIdx.y;
            const int j  = min(jc / ncols2, ne01 - 1);
            const int c  = jc % ncols2;
            const float2 * src = Q_f2 + j*stride_Q1 + c*stride_Q2 + threadIdx.x*4;
            __align__(16) float2 tmp[4];
            ggml_cuda_memcpy_1<16>(tmp + 0, src + 0);
            ggml_cuda_memcpy_1<16>(tmp + 2, src + 2);
            __align__(16) half2 q[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                q[l] = scale_h2 * make_half2(tmp[l].x, tmp[l].y);
            }
            ggml_cuda_memcpy_1<16>(tile_Q + jc*stride_tile_Q + threadIdx.x*4, q);
        }
    }

    __syncthreads();

    // Q row and mask row of this thread for each of its groups.
    const half2 * Q_lane[G];
    const half  * mask_lane[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int jc = (g0 + g)*cols_per_warp + lane;
        Q_lane[g]    = tile_Q + jc*stride_tile_Q;
        mask_lane[g] = mask_h + int64_t(min(jc / ncols2, ne01 - 1))*stride_mask;
    }

    // The last tile is done also if kb0_start >= kb0_stop, as in flash_attn_ext_f16_process_tile.
    for (int kb0 = kb0_start; ; ++kb0) {
        const int k_warp = kb0*nbatch_fa + i_warp; // First list entry of this warp.

        // K row of this thread and the mask of the rows of the warp. A negative row is padding, its mask is -inf.
        const int row_K = max(rows[k_warp + lane], 0);
        __align__(16) half mask_w[G][T_A_KQ::I];
#pragma unroll
        for (int g = 0; g < G; ++g) {
#pragma unroll
            for (int i = 0; i < T_A_KQ::I; i += 8) {
                ggml_cuda_memcpy_1<16>(mask_w[g] + i, mask_lane[g] + k_warp + i);
            }
        }

        // F16: this thread takes 4 half2 of each half of 8 V rows of the warp.
        [[maybe_unused]] const int i_Vh = (threadIdx.x / T_A_KQ::I)*(T_A_KQ::I/2);
        [[maybe_unused]] const int c_Vh = (threadIdx.x % T_A_KQ::I)*4;
        [[maybe_unused]] __align__(16) int row_V[T_A_KQ::I/2];
        // Q8: this thread takes one block of each half of two V rows of the warp that are side by side in the A tiles.
        [[maybe_unused]] const int m_Vq = threadIdx.x / 4;
        [[maybe_unused]] const int u_Vq = threadIdx.x % 4;
        [[maybe_unused]] const char * V_row[2] = {nullptr, nullptr};
        [[maybe_unused]] q35moe_q8_blk V_b[2];

        // Calculate tile of KQ. On the way the first half of the values of the V rows of this warp goes to the shared tile.
        T_C_KQ KQ_C[G];
        if constexpr (!Q8) {
            const half2 * K_row = K_h2 + int64_t(row_K)*stride_K;
#pragma unroll
            for (int i = 0; i < T_A_KQ::I/2; i += 4) {
                ggml_cuda_memcpy_1<16>(row_V + i, rows + k_warp + i_Vh + i);
            }
#pragma unroll
            for (int i0 = 0; i0 < T_A_KQ::I; i0 += NB) {
                T_A_KQ K_A[NB];
                __align__(16) half2 V_tmp[NB/2][4];
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    ggml_cuda_memcpy_1<16>(K_A[b].x + 0, K_row + (i0 + b)*T_A_KQ::J + 0);
                    ggml_cuda_memcpy_1<16>(K_A[b].x + 4, K_row + (i0 + b)*T_A_KQ::J + 4);
                }
#pragma unroll
                for (int b = 0; b < NB/2; ++b) {
                    ggml_cuda_memcpy_1<16>(V_tmp[b], V_h2 + int64_t(max(row_V[i0/2 + b], 0))*stride_V + c_Vh);
                }

                // No code, only a use of each load: all loads above go out before the first wait.
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    asm volatile("" : : "v"(K_A[b].x[0]), "v"(K_A[b].x[4]));
                }
#pragma unroll
                for (int b = 0; b < NB/2; ++b) {
                    asm volatile("" : : "v"(V_tmp[b][0]));
                }

#pragma unroll
                for (int b = 0; b < NB/2; b += 2) {
                    q35moe_fagqa4_store_A<S>(tile_A, c_Vh, (i_Vh + i0/2 + b)/2, V_tmp[b], V_tmp[b + 1]);
                }
#pragma unroll
                for (int b = 0; b < NB; ++b) {
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        T_B_KQ Q_B;
                        ggml_cuda_memcpy_1<16>(Q_B.x + 0, Q_lane[g] + (i0 + b)*T_B_KQ::J + 0);
                        ggml_cuda_memcpy_1<16>(Q_B.x + 4, Q_lane[g] + (i0 + b)*T_B_KQ::J + 4);
                        mma(KQ_C[g], K_A[b], Q_B);
                    }
                }
            }
        } else {
            // A K row is 4 parts of 2 blocks, one part has the values of 4 WMMA steps. One burst has the K parts of NB WMMA steps.
            // The V blocks of this thread go out with the first burst.
            static_assert(DKQ == 4*QK8_0*2 && DV == DKQ && T_A_KQ::J*4 == QK8_0, "bad row size");

            const char * K_row = (const char *) (K_h2 + int64_t(row_K)*stride_K);
            __align__(8) int r2[2];
            ggml_cuda_memcpy_1<8>(r2, rows + k_warp + 2*m_Vq);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                V_row[i] = (const char *) (V_h2 + int64_t(max(r2[i], 0))*stride_V) + u_Vq*sizeof(block_q8_0);
            }

#pragma unroll
            for (int i0 = 0; i0 < T_A_KQ::I; i0 += NB) {
                q35moe_q8_part K_p[NB/4];
#pragma unroll
                for (int p = 0; p < NB/4; ++p) {
                    K_p[p].load(K_row + (i0/4 + p)*2*sizeof(block_q8_0));
                }
                if (i0 == 0) {
                    V_b[0].load(V_row[0]);
                    V_b[1].load(V_row[1]);
                }

                // All loads above go out before the first wait.
#pragma unroll
                for (int p = 0; p < NB/4; ++p) {
                    K_p[p].use();
                }
                if (i0 == 0) {
                    V_b[0].use();
                    V_b[1].use();
                    q35moe_fagqa4_q8_store_A<S>(tile_A, u_Vq*(QK8_0/2), m_Vq, V_b[0], V_b[1]);
                }
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    const int s  = i0 + b;                        // WMMA step, it has T_A_KQ::J half2 of the K row
                    const int bs = ((s/2) % 2)*sizeof(block_q8_0); // first byte of the block of this step in its part
                    const half2 d = K_p[b/4].d2(bs);
                    T_A_KQ K_A;
#pragma unroll
                    for (int l = 0; l < T_A_KQ::J; ++l) {
                        K_A.x[l] = K_p[b/4].h2(bs + 2 + 2*((s % 2)*T_A_KQ::J + l), d);
                    }
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        T_B_KQ Q_B;
                        ggml_cuda_memcpy_1<16>(Q_B.x + 0, Q_lane[g] + s*T_B_KQ::J + 0);
                        ggml_cuda_memcpy_1<16>(Q_B.x + 4, Q_lane[g] + s*T_B_KQ::J + 4);
                        mma(KQ_C[g], K_A, Q_B);
                    }
                }
            }
        }

        T_B_VKQ B[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            // Value l of KQ_C is row 2*l + threadIdx.x/16 of the warp, see T_C_KQ::get_j.
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[g].x[l] += __half2float(threadIdx.x < 16 ? mask_w[g][2*l] : mask_w[g][2*l + 1]);
            }

            // Calculate softmax for each KQ column using the current max. value.
            // The divisor is stored in KQ_rowsum and will be applied at the end.
            float KQ_max_new = KQ_max[g];
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_max_new = fmaxf(KQ_max_new, KQ_C[g].x[l] + FATTN_KQ_MAX_OFFSET);
            }
            // Values per KQ column are spread across 2 threads:
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, 16, warp_size));

            float KQ_rowsum_add = 0.0f;
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[g].x[l] = expf(KQ_C[g].x[l] - KQ_max_new);
                KQ_rowsum_add += KQ_C[g].x[l];
            }

            {
                const float KQ_max_diff = KQ_max[g] - KQ_max_new;
                float KQ_max_scale = expf(KQ_max_diff);
                KQ_max[g] = KQ_max_new;

                *((uint32_t *) &KQ_max_scale) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum[g] = KQ_max_scale*KQ_rowsum[g] + KQ_rowsum_add;

                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        VKQ_C[g][i].x[l] *= KQ_max_scale_h2;
                    }
                }
            }

            // Convert KQ C tile into B tile for VKQ calculation:
            B[g] = get_half2(KQ_C[g]);
        }

        // The loads of the second half of the values of the V rows go out here. Their data goes to the shared tile when the first half is done.
        [[maybe_unused]] __align__(16) half2 V_tmp[T_A_KQ::I/2][4];
        if constexpr (Q8) {
            V_b[0].load(V_row[0] + (DV/(2*QK8_0))*sizeof(block_q8_0));
            V_b[1].load(V_row[1] + (DV/(2*QK8_0))*sizeof(block_q8_0));
        } else {
#pragma unroll
            for (int i = 0; i < T_A_KQ::I/2; ++i) {
                ggml_cuda_memcpy_1<16>(V_tmp[i], V_h2 + int64_t(max(row_V[i], 0))*stride_V + DV/4 + c_Vh);
            }
        }

        // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < DV; i_VKQ_0 += T_A_VKQ::I) {
            if (i_VKQ_0 == DV/2) {
                if constexpr (Q8) {
                    q35moe_fagqa4_q8_store_A<S>(tile_A, u_Vq*(QK8_0/2), m_Vq, V_b[0], V_b[1]);
                } else {
#pragma unroll
                    for (int i = 0; i < T_A_KQ::I/2; i += 2) {
                        q35moe_fagqa4_store_A<S>(tile_A, c_Vh, (i_Vh + i)/2, V_tmp[i], V_tmp[i + 1]);
                    }
                }
            }

            // The A tile of this lane for the step, see q35moe_store_A.
            const half2 * src = tile_A + q35moe_fagqa4_rec<S>((i_VKQ_0 % (DV/2))/2 + lane)*S;
            T_A_VKQ A;
#pragma unroll
            for (int l = 0; l < T_A_VKQ::ne; l += 4) {
                ggml_cuda_memcpy_1<16>(A.x + l, src + l);
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                mma(VKQ_C[g][i_VKQ_0/T_A_VKQ::I], A, B[g]);
            }
        }

        if (kb0 >= kb0_stop - 1) {
            break;
        }
    }

    // Finally, sum up partial KQ rowsums. The partial sums are spread across 2 threads.
#pragma unroll
    for (int g = 0; g < G; ++g) {
        KQ_rowsum[g] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[g], 16, warp_size);
    }

    // Combine VKQ accumulator values of the np parallel warps as flash_attn_ext_f16 does it, in the same shared memory layout:
    // the warp with the number w there is group w / np and parallel warp w % np.
    // This takes the whole shared tile, so the other warps must be done with their rows.
    __syncthreads();

    {
        if (threadIdx.x / 16 < 1) {
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int jc_cwm = ((g0 + g)*np + ip_w)*cols_per_warp + T_C_VKQ::get_i(0);
                const float2 KQ_cmr = make_float2(KQ_max[g], KQ_rowsum[g]);
                ((float2 *) tile)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
            }
        }

        __syncthreads();
    }

    {
        constexpr int nmeta = np*cols_per_warp/warp_size;

        float KQ_cmn[G];
        float KQ_cms[G][nmeta];
        float KQ_crs[G];

        if (ip_w == 0) {
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int jc_meta = (g0 + g)*np*cols_per_warp + threadIdx.x;
                const float2 * const meta_ptr = ((const float2 *) tile) + jc_meta*(tile_stride/2) + nbatch_combine/2;

                // Combine the meta data for parallel warps via shared memory.
                float2 meta[nmeta];
#pragma unroll
                for (int imeta = 0; imeta < nmeta; ++imeta) {
                    meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
                }

                KQ_cmn[g] = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
                for (int imeta = 1; imeta < nmeta; ++imeta) {
                    KQ_cmn[g] = fmaxf(KQ_cmn[g], meta[imeta].x);
                }
#pragma unroll
                for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                    if (offset < warp_size) {
                        KQ_cmn[g] = fmaxf(KQ_cmn[g], __shfl_xor_sync(0xFFFFFFFF, KQ_cmn[g], offset, warp_size));
                    }
                }

#pragma unroll
                for (int imeta = 0; imeta < nmeta; ++imeta) {
                    KQ_cms[g][imeta] = expf(meta[imeta].x - KQ_cmn[g]);
                }

                KQ_crs[g] = KQ_cms[g][0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
                for (int imeta = 1; imeta < nmeta; ++imeta) {
                    KQ_crs[g] += KQ_cms[g][imeta]*meta[imeta].y;
                }
#pragma unroll
                for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                    if (offset < warp_size) {
                        KQ_crs[g] += __shfl_xor_sync(0xFFFFFFFF, KQ_crs[g], offset, warp_size);
                    }
                }
            }
        }

        __syncthreads();

        if (ip_w == 0) {
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int jc_meta = (g0 + g)*np*cols_per_warp + threadIdx.x;
                float2 * const meta_ptr = ((float2 *) tile) + jc_meta*(tile_stride/2) + nbatch_combine/2;

                // Write back combined meta data: combined KQ max scale + rowsum.
#pragma unroll
                for (int imeta = 0; imeta < nmeta; ++imeta) {
                    meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[g][imeta], KQ_crs[g]);
                }

                // Combined KQ max + rowsum.
                if (needs_fixup && threadIdx.x < cols_per_warp) {
                    float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                    dstk_fixup_meta[(g0 + g)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn[g], KQ_crs[g]);
                }
                if (is_fixup && threadIdx.x < cols_per_warp) {
                    float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                    dstk_fixup_meta[(g0 + g)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn[g], KQ_crs[g]);
                }
            }
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const int j0 = ((g0 + g)*np + ip_w)*cols_per_warp;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    const int j = j0 + T_C_VKQ::get_i(l);
                    const int k = k1 + T_C_VKQ::get_j(l);

                    tile[j*tile_stride + k] = VKQ_C[g][(k00 + k1)/T_C_VKQ::J].x[l];
                }
            }
        }

        __syncthreads();

        if (ip_w == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int jc0_dst = 0; jc0_dst < G*cols_per_warp; ++jc0_dst) {
                const int jc_dst = g0*cols_per_warp + jc0_dst;

                const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                const int j_dst = jc_dst / ncols2;
                const int c_dst = jc_dst % ncols2;

                // Only the columns of the tokens that are there, the fixup kernels read no other column.
                if (j_dst >= ne01) {
                    continue;
                }

                const float * meta_j = (const float *) tile + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                for (int k0 = 0; k0 < nbatch_combine; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                    for (int ip = 0; ip < np; ++ip) {
                        const float KQ_crs = meta_j[ip*cols_per_warp * tile_stride + 0];
                        const float2 dstk_val_add = __half22float2(tile[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                        dstk_val.x += dstk_val_add.x*KQ_crs;
                        dstk_val.y += dstk_val_add.y*KQ_crs;
                    }

                    if (!needs_fixup && !is_fixup) {
                        const float KQ_rowsum_j = meta_j[1];
                        dstk_val.x /= KQ_rowsum_j;
                        dstk_val.y /= KQ_rowsum_j;
                    }

                    if (is_fixup) {
                        dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                    } else {
                        dstk[(j_dst*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                    }
                }
            }
        }

        __syncthreads();
    }
}

// The K/V row tiles kb0_start to kb0_stop of the KV head that has the tile kbc.
template<int ncols1, int G, int NB, int SA, int SQ, bool needs_fixup, bool is_fixup, bool Q8>
static __device__ __forceinline__ void q35moe_fagqa4_wmma_part(
        const char * const __restrict__ Q,
        const char * const __restrict__ K,
        const char * const __restrict__ V,
        const char * const __restrict__ mask,
        const int  * const __restrict__ KV_max,
        const int  * const __restrict__ KV_rows,
        float      * const __restrict__ dst,
        float2     * const __restrict__ dst_meta,
        half2      * const __restrict__ tile,
        const float scale,
        const int32_t ne01, const int32_t ne02, const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11, const int32_t ne12,
        const int32_t nb11, const int32_t nb12,
        const int32_t nb21, const int32_t nb22,
        const int32_t ne33, const int32_t nb31, const int64_t nb33,
        const int kbc, const int kb0_start, const int kb0_stop) {
    constexpr int DV        = 256;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(256, DV, ncols1*8);

    const int gqa_ratio = ne02 / ne12;
    const int iter_k    = ne11 / nbatch_fa;

    const int sequence =  kbc /(iter_k*ne12);
    const int z_KV     = (kbc - iter_k*ne12 * sequence)/iter_k;
    const int zt_Q     = z_KV*gqa_ratio; // First query head of the KV head.

    const float2  * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2   * K_h2   = (const half2  *) (K + nb12*z_KV);
    const half2   * V_h2   = (const half2  *) (V + nb22*z_KV);
    const half    * mask_h = (const half   *) (mask + nb33*(sequence % ne33));
    const int32_t * rows   = KV_rows + int64_t(sequence)*ne11;
    float2        * dstk   = ((float2 *) dst) + (sequence*ne01*ne02 + zt_Q) * (DV/2);

    q35moe_fagqa4_wmma_tile<ncols1, G, NB, SA, SQ, needs_fixup, is_fixup, Q8>
        (Q_f2, K_h2, V_h2, mask_h, rows, dstk, dst_meta, tile, scale, ne01, ne02,
         nb01 / sizeof(float2), nb02 / sizeof(float2), nb11 / sizeof(half2), nb21 / sizeof(half2), nb31 / sizeof(half),
         kb0_start, min(kb0_stop, KV_max[sequence] / nbatch_fa));
}
#endif // defined(AMD_WMMA_256B_AVAILABLE)

// Attention over a list of K/V rows for 2 to ncols1 query tokens of each sequence (ncols1 is 2 or 4), 8 query heads per KV head, head size 256,
// one mask row per token for all heads. K and V are F16 or Q8_0 rows (Q8).
// It takes the place of flash_attn_ext_f16<256, 256, ncols1, 8, false, false, false> with the Stream-K plan of launch_fattn: same grid, same KV rows per block,
// same blocks that start or end a KV head. The result has the same bits after the fixup kernel of that plan, see q35moe_fagqa4_wmma_tile.
// G, NB, SA and SQ do not change the result: Q column groups per warp, WMMA steps of K per burst of loads, strides of the shared tile.
template<int ncols1, int G, int NB, int SA, int SQ, bool Q8>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(256, 256, ncols1*8) / G, 1)
static __global__ void q35moe_fagqa4_wmma(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const int32_t ne01, const int32_t ne02, const int32_t ne03, const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11, const int32_t ne12,
        const int32_t nb11, const int32_t nb12,
        const int32_t nb21, const int32_t nb22,
        const int32_t ne33, const int32_t nb31, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_256B_AVAILABLE)
    constexpr int DV        = 256;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(256, DV, ncols1*8);

    __shared__ half2 tile[q35moe_fagqa4_tile_ne<ncols1, G, SA, SQ>()];

    const int iter_k = ne11 / nbatch_fa;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*ne12*ne03) / gridDim.x;

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // This block finishes a KV head, that part goes to dst.
        if (kb0_start == 0) {
            // The block has the whole KV head.
            q35moe_fagqa4_wmma_part<ncols1, G, NB, SA, SQ, false, false, Q8>
                (Q_ptr, K_ptr, V_ptr, mask_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, tile, scale,
                 ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb21, nb22, ne33, nb31, nb33, kbc, kb0_start, kb0_stop);
        } else {
            q35moe_fagqa4_wmma_part<ncols1, G, NB, SA, SQ, true, false, Q8>
                (Q_ptr, K_ptr, V_ptr, mask_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, tile, scale,
                 ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb21, nb22, ne33, nb31, nb33, kbc, kb0_start, kb0_stop);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    q35moe_fagqa4_wmma_part<ncols1, G, NB, SA, SQ, false, true, Q8>
        (Q_ptr, K_ptr, V_ptr, mask_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, tile, scale,
         ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb21, nb22, ne33, nb31, nb33, kbc, kb0_start, kb0_stop);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        ne01, ne02, ne03, nb01, nb02, nb03,
        ne11, ne12,
        nb11, nb12,
        nb21, nb22,
        ne33, nb31, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_256B_AVAILABLE)
}

typedef void (* q35moe_fagqa4_kernel_t)(
        const char *, const char *, const char *, const char *, const int *, const int *, float *, float2 *, const float,
        const int32_t, const int32_t, const int32_t, const int32_t, const int32_t, const int32_t,
        const int32_t, const int32_t, const int32_t, const int32_t, const int32_t, const int32_t,
        const int32_t, const int32_t, const int64_t);

// The plan of launch_fattn and the launches for flash_attn_ext_f16<256, 256, ncols1, 8, false, false, false>, with q35moe_fagqa4_wmma in its place.
template<int ncols1>
static void ggml_cuda_flash_attn_ext_q35moe_fagqa4_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool q8) {
    constexpr int DKQ    = 256;
    constexpr int DV     = 256;
    constexpr int ncols2 = 8;
    constexpr int ncols  = ncols1*ncols2;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * rows = dst->src[5];

    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const int warp_size      = ggml_cuda_info().devices[id].warp_size;
    const int nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const int nwarps_ref     = nthreads / warp_size;

    // The layout of the kernel (G, NB, SA, SQ of q35moe_fagqa4_wmma), the fastest one of those tried at a list of 102400 rows.
    // 2 tokens: no padding. With it a block has more shared memory than a block of the kernel that this one replaces, and the kernel is slower than that one.
    // 3 and 4 tokens: one warp has both groups of Q columns, padding.
    constexpr int G  = ncols1 == 2 ? 1 : 2;
    constexpr int NB = 4;
    constexpr int SA = ncols1 == 2 ? 16 : 20;
    constexpr int SQ = ncols1 == 2 ? 128 : 132;
    const q35moe_fagqa4_kernel_t kernel = q8 ? q35moe_fagqa4_wmma<ncols1, G, NB, SA, SQ, true> : q35moe_fagqa4_wmma<ncols1, G, NB, SA, SQ, false>;
    const int nwarps = nwarps_ref / G;

    const int64_t n_kv       = rows->ne[0];
    const int     ntiles_dst = K->ne[2]*Q->ne[3]; // One output tile is one KV head of one sequence.
    const int     ntiles_KV  = n_kv / nbatch_fa;

    // Shared memory of the kernel that this one replaces, as in ggml_cuda_flash_attn_ext_mma_f16_case. The number of blocks comes from it.
    const size_t nbytes_shared_KV      = nbatch_fa                                         * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q       = ncols                                             * (DKQ/2 + 4)                            * sizeof(half2);
    const size_t nbytes_shared_mask    = ncols1                                            * (nbatch_fa/2 + 4)                      * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps_ref*std::min(ncols, get_cols_per_warp(cc)) * (nbatch_combine + 4)                   * sizeof(half2);
    const size_t nbytes_shared_total   = std::max(nbytes_shared_combine, std::max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask));

    const fattn_kernel_t ref = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, false, false>;
    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, ref, nthreads, nbytes_shared_total));
    GGML_ASSERT(max_blocks_per_sm > 0);

    // Stream-K as in launch_fattn.
    const int max_blocks               = max_blocks_per_sm*nsm;
    const int tiles_nwaves             = (ntiles_dst + max_blocks - 1) / max_blocks;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

    int nblocks = ntiles_dst;
    if (tiles_efficiency_percent < 75) {
        const int nblocks_raw     = std::min(max_blocks, ntiles_KV*ntiles_dst);
        const int nblocks_rounded = (nblocks_raw / ntiles_dst) * ntiles_dst;
        nblocks = nblocks_rounded > 0 && 100 * (nblocks_raw - nblocks_rounded) / nblocks_raw <= 5 ? nblocks_rounded : nblocks_raw;
    }
    const bool fixup_uniform = nblocks % ntiles_dst == 0 && nblocks > ntiles_dst; // Each block is in one KV head.
    const bool fixup_general = !fixup_uniform && ntiles_dst % nblocks != 0;

    static bool logged_use = false;
    if (!logged_use) {
        logged_use = true;
        GGML_LOG_INFO("%s: GGML_Q35MOE_FAGQA4 is used, K and V %s, kv = %lld, tokens = %lld, sequences = %lld, blocks = %d\n",
            __func__, ggml_type_name(K->type), (long long) n_kv, (long long) Q->ne[1], (long long) Q->ne[3], nblocks);
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

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

    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);
    if (ntiles_dst % nblocks != 0) {
        dst_tmp_meta.alloc(size_t(nblocks) * ncols * (2 + DV/2));
    }

    const dim3 blocks_num(nblocks, 1, 1);
    const dim3 block_dim(warp_size, nwarps, 1);
    ggml_cuda_kernel_launch(kernel, ggml_cuda_kernel_launch_params(blocks_num, block_dim, 0, stream),
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
        (const int *) KV_max.ptr, (const int *) rows->data, (float *) dst->data, dst_tmp_meta.ptr, ggml_get_op_params_f32(dst, 0),
        int32_t(Q->ne[1]), int32_t(Q->ne[2]), int32_t(Q->ne[3]), int32_t(Q->nb[1]), int32_t(Q->nb[2]), int32_t(Q->nb[3]),
        int32_t(n_kv), int32_t(K->ne[2]),
        int32_t(K->nb[1]), int32_t(K->nb[2]),
        int32_t(V->nb[1]), int32_t(V->nb[2]),
        int32_t(mask->ne[3]), int32_t(mask->nb[1]), int64_t(mask->nb[3]));
    CUDA_CHECK(cudaGetLastError());

    // The fixup of launch_fattn.
    const dim3 block_dim_combine(DV, 1, 1);
    if (fixup_uniform) {
        const uint3 fd0 = init_fastdiv_values(K->ne[2]);
        const uint3 fd1 = init_fastdiv_values(1);
        const uint3 fd2 = init_fastdiv_values(1);

        const dim3 blocks_num_combine(ntiles_dst, ncols1, ncols2);
        ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>,
            ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, stream),
            (float *) dst->data, dst_tmp_meta.ptr,
            int(Q->ne[1]), int(Q->ne[2]), int(K->ne[2]), nblocks,
            int(Q->ne[2] / K->ne[2]), nblocks / ntiles_dst, fd0, fd1, fd2);
    } else if (fixup_general) {
        const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * K->ne[2]);
        const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

        const dim3 blocks_num_combine(nblocks, ncols1, ncols2);
        ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>,
            ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, stream),
            (float *) dst->data, dst_tmp_meta.ptr,
            int(Q->ne[1]), int(Q->ne[2]), int(Q->ne[2] / K->ne[2]), ntiles_KV * ntiles_dst,
            fd_k_j_z_ne12, fd_k, fd_k, fd_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

// GGML_Q35MOE_FAGQA4=1 takes the attention of qwen35moe for 2 to 4 query tokens of each sequence (head size 256, 8 query heads per KV head, K/V rows,
// F16 or Q8_0 K and V) on gfx1103. These are the batches that verify the drafts of speculative decoding. One token is GGML_Q35MOE_FAGQA, 5 and more GGML_Q35MOE_PFATTN.
// mma_f16 says that the op goes to flash_attn_ext_f16 without the switch. The result has the same bits as with that kernel, also for Q8_0 K and V:
// that path makes F16 copies of the whole K and V first.
// GGML_Q35MOE_FAGQA4_CHECK=1 with GGML_CUDA_DISABLE_GRAPHS=1 also runs that kernel and logs the number of values with other bits. This is slow, it is for tests.
static bool ggml_cuda_flash_attn_ext_q35moe_fagqa4(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool mma_f16) {
    static const bool enable = [] { const char * s = getenv("GGML_Q35MOE_FAGQA4"); return s && strcmp(s, "1") == 0; }();
    if (!enable) {
        return false;
    }

    constexpr int DKQ    = 256;
    constexpr int DV     = 256;
    constexpr int ncols2 = 8;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * rows = dst->src[5];

    // An op of this shape is the attention of the model for 2 to 4 tokens: each reason that sends it to the stock kernel goes to the log once.
    // Ops of other shapes (other models, one token, prompt batches) come here too, they say nothing.
    const bool   candidate = Q->ne[0] == DKQ && Q->ne[1] >= 2 && Q->ne[1] <= 4 && Q->ne[2] == ncols2*K->ne[2] && !dst->src[4];
    const char * func      = __func__;
    static bool  logged[10] = {false};
    const auto stock = [&](const int reason, const char * why) {
        if (candidate && !logged[reason]) {
            logged[reason] = true;
            GGML_LOG_WARN("%s: GGML_Q35MOE_FAGQA4 is not used: %s (tokens = %lld, sequences = %lld, K %s, V %s)\n",
                func, why, (long long) Q->ne[1], (long long) Q->ne[3], ggml_type_name(K->type), ggml_type_name(V->type));
        }
        return false;
    };

    // The op must be one that goes to flash_attn_ext_f16<256, 256, ncols1, 8, false, false, false> with its GQA path.
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103) {
        return stock(0, "the device is not gfx1103");
    }
    if (!mask || !rows) {
        return stock(1, "no list of K/V rows (it needs --kv-unified, and LLAMA_KV_ROWS_DISABLE not set)");
    }
    if (!mma_f16 || dst->src[4]) {
        return stock(2, "the op does not go to the WMMA kernel");
    }
    const bool q8 = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0;
    if (Q->type != GGML_TYPE_F32 || mask->type != GGML_TYPE_F16 || !(q8 || (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16))) {
        return stock(3, "K and V must both be F16 or both be Q8_0");
    }
    if (Q->ne[0] != DKQ || K->ne[0] != DKQ || V->ne[0] != DV || Q->ne[1] < 2 || Q->ne[1] > 4 || Q->ne[2] != ncols2*K->ne[2] || V->ne[2] != K->ne[2]) {
        return stock(4, "the shape of K or V");
    }
    if (rows->ne[0] <= 0 || rows->ne[0] % FATTN_KQ_STRIDE != 0 || rows->ne[1] != Q->ne[3] || mask->ne[0] != rows->ne[0] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1) {
        return stock(5, "the shape of the row list or of the mask");
    }
    if (ggml_get_op_params_f32(dst, 1) != 0.0f || ggml_get_op_params_f32(dst, 2) != 0.0f) {
        return stock(6, "max_bias or logit_softcap is set");
    }
    // The kernel loads 16 bytes at a time. A Q8_0 row must be whole blocks, one after the other.
    if (!ggml_cuda_is_aligned(Q, 16) || !ggml_cuda_is_aligned(K, 16) || !ggml_cuda_is_aligned(V, 16) || !ggml_cuda_is_aligned(mask, 16) ||
            !ggml_cuda_is_aligned(rows, 16)) {
        return stock(7, "a tensor is not aligned to 16 bytes");
    }
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return stock(8, "the strides of K or V");
    }

    // The config of that kernel that q35moe_fagqa4_wmma_tile is made for.
    const int ncols1    = Q->ne[1] <= 2 ? 2 : 4;
    const int ncols     = ncols1*ncols2;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const int nwarps    = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols, cc) / warp_size;
    if (warp_size != WARP_SIZE || nwarps != (ncols1 == 2 ? 2 : 8) || ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc) != 2*ncols ||
            !ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, ncols, cc) || ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc) > 1 ||
            ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols, cc)) {
        return stock(9, "the configuration of the WMMA kernel");
    }

    cudaStream_t stream = ctx.stream();

    // The check reads the result back. A stream that captures a graph cannot do this, so it needs GGML_CUDA_DISABLE_GRAPHS=1.
    static const bool check = [] {
        const char * s = getenv("GGML_Q35MOE_FAGQA4_CHECK");
        if (!s || strcmp(s, "1") != 0) {
            return false;
        }
        if (getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
            GGML_LOG_WARN("GGML_Q35MOE_FAGQA4_CHECK=1 does nothing, it needs GGML_CUDA_DISABLE_GRAPHS=1\n");
            return false;
        }
        return true;
    }();
    std::vector<float> out_ref;
    if (check) {
        ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
        out_ref.resize(ggml_nelements(dst));
        CUDA_CHECK(cudaMemcpyAsync(out_ref.data(), dst->data, ggml_nbytes(dst), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Every byte 0xff: a value that the new kernel does not write then has other bits than the reference.
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0xff, ggml_nbytes(dst), stream));
    }

    if (ncols1 == 2) {
        ggml_cuda_flash_attn_ext_q35moe_fagqa4_launch<2>(ctx, dst, q8);
    } else {
        ggml_cuda_flash_attn_ext_q35moe_fagqa4_launch<4>(ctx, dst, q8);
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
        // The totals go to stderr when the process ends, with the calls for 2, 3 and 4 tokens and for 1 to 8 sequences.
        static struct check_stats {
            long n_calls      = 0;
            long n_calls_diff = 0;
            long n_calls_tok[3] = {0};
            long n_calls_seq[8] = {0};
            ~check_stats() {
                fprintf(stderr, "ggml_cuda_flash_attn_ext_q35moe_fagqa4: check done, %ld calls, %ld calls with differences, calls by tokens: %ld %ld %ld, by sequences:",
                    n_calls, n_calls_diff, n_calls_tok[0], n_calls_tok[1], n_calls_tok[2]);
                for (const long n : n_calls_seq) {
                    fprintf(stderr, " %ld", n);
                }
                fprintf(stderr, "\n");
            }
        } stats;
        stats.n_calls++;
        stats.n_calls_diff += n_diff > 0;
        stats.n_calls_tok[Q->ne[1] - 2]++;
        stats.n_calls_seq[std::min<int64_t>(Q->ne[3], 8) - 1]++;
        if (stats.n_calls <= 8 || (n_diff > 0 && stats.n_calls_diff <= 32)) {
            GGML_LOG_INFO("%s: call %ld: %zu of %zu values differ (max %.3g), kv = %lld, tokens = %lld, sequences = %lld, heads = %lld, %s\n",
                __func__, stats.n_calls, n_diff, out.size(), d_max, (long long) rows->ne[0], (long long) Q->ne[1], (long long) Q->ne[3],
                (long long) K->ne[2], ggml_type_name(K->type));
        }
    }

    return true;
}

#endif // defined(GGML_USE_HIP)

#pragma once

// Include from fattn-mma-f16.cuh after flash_attn_ext_f16: it has the tile types and the mask tile load, and it has no include guard.

#if defined(GGML_USE_HIP)

// Rows in shared memory are padded by 16 bytes like in flash_attn_ext_f16: the lanes of a warp read rows that are one stride apart, and without the padding these loads are 7 times slower.
#define QWEN35_PF_STRIDE (128 + 4)
// A V half tile is transposed in shared memory, so a warp reads a WMMA operand with 4 loads of 16 bytes: 128 pairs of elements, 2*16 half2 (32 rows) and the padding per pair.
#define QWEN35_PF_STRIDE_VT (2*16 + 4)

// Shared memory: Q of 64 columns, one half tile of K or V (32 rows, or 128 pairs of elements), the mask tile of up to 16 queries.
#define QWEN35_PF_NBYTES_SHARED ((64*QWEN35_PF_STRIDE + 128*QWEN35_PF_STRIDE_VT)*sizeof(half2) + 16*(64 + 8)*sizeof(half))

// K/V row numbers of 4 rows of a half tile. Warp w takes the rows 4*w .. 4*w + 3 from row0.
// pfload: no branch, the compiler makes one 16-byte load of the 4 numbers; indices must be set and 16-byte aligned, and qwen35_pf_load_rows<true> clamps the padding rows.
template<bool pfload>
static __device__ __forceinline__ void qwen35_pf_load_index(
        int32_t * const __restrict__ idx, const int row0, const int32_t * const __restrict__ indices) {
    if constexpr (pfload) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            idx[q] = indices[row0 + threadIdx.y*4 + q];
        }
    } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int i = row0 + threadIdx.y*4 + q;
            // K/V rows given per mask column, negative rows are padding
            idx[q] = indices ? max(indices[i], 0) : i;
        }
    }
}

// Global loads of 4 rows of K or V, 16 bytes per lane and row.
template<bool pfload>
static __device__ __forceinline__ void qwen35_pf_load_rows(
        half2 (* __restrict__ R)[4], const half2 * const __restrict__ KV, const int stride_KV, const int32_t * const __restrict__ idx) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int i = pfload ? max(idx[q], 0) : idx[q];
        ggml_cuda_memcpy_1<16>(R[q], KV + int64_t(i)*stride_KV + threadIdx.x*4);
    }
}

// Global loads of the mask tile, 2 queries per warp and 2 keys per lane. Same layout in shared memory as flash_attn_ext_f16_load_mask.
template<int ncols1, int nwarps>
static __device__ __forceinline__ void qwen35_pf_load_mask(
        uint32_t * const __restrict__ M, const half * const __restrict__ mask_h, const int stride_mask, const int k_VKQ_0, const int j0, const uint3 ne01) {
#pragma unroll
    for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
        const int j_vram = fastmodulo(j0 + j1 + threadIdx.y, ne01);
        ggml_cuda_memcpy_1<sizeof(half2)>(M + j1/nwarps, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*threadIdx.x);
    }
}

template<int ncols1, int nwarps, int nbatch_fa>
static __device__ __forceinline__ void qwen35_pf_store_mask(half * const __restrict__ tile_mask, const uint32_t * const __restrict__ M) {
#pragma unroll
    for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
        ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + (j1 + threadIdx.y)*(nbatch_fa + 8) + 2*threadIdx.x, M + j1/nwarps);
    }
}

// K rows as they are: tile[row][element].
static __device__ __forceinline__ void qwen35_pf_store_rows(half2 * const __restrict__ tile, const half2 (* __restrict__ R)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        ggml_cuda_memcpy_1<16>(tile + (threadIdx.y*4 + q)*QWEN35_PF_STRIDE + threadIdx.x*4, R[q]);
    }
}

// Place of the pair of elements p in the transposed V half tile. The 16 lanes of a half warp store the pairs 4*lane + d and load the pairs p0 + lane.
// The pairs of each group of 16 are rotated by the group number, so that both sets of 16 accesses are at different offsets modulo 256 bytes (the same offset costs 4 times as much).
static __device__ __forceinline__ int qwen35_pf_pair_pos(const int p) {
    return (p & ~15) | ((p + (p >> 4)) & 15);
}

// V rows transposed: tile[pair of elements][element%2][row], 32 rows. One store of 8 bytes holds an element of the 4 rows of the warp.
static __device__ __forceinline__ void qwen35_pf_store_rows_t(half2 * const __restrict__ tile, const half2 (* __restrict__ R)[4]) {
    const uint32_t * r = (const uint32_t *) R;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const uint32_t a = r[0*4 + d];
        const uint32_t b = r[1*4 + d];
        const uint32_t c = r[2*4 + d];
        const uint32_t g = r[3*4 + d];
        __align__(8) const uint32_t lo[2] = {(a & 0xFFFFu) | (b << 16), (c & 0xFFFFu) | (g << 16)};
        __align__(8) const uint32_t hi[2] = {(a >> 16) | (b & 0xFFFF0000u), (c >> 16) | (g & 0xFFFF0000u)};
        half2 * dst = tile + qwen35_pf_pair_pos(threadIdx.x*4 + d)*QWEN35_PF_STRIDE_VT + threadIdx.y*2;
        ggml_cuda_memcpy_1<8>(dst +  0, lo);
        ggml_cuda_memcpy_1<8>(dst + 16, hi);
    }
}

// pfqk: a K stage is the elements 0-127 or 128-255 of all 64 rows of a tile, 64 half2 per row and the padding.
#define QWEN35_PF_STRIDE_KS (64 + 4)

// pfqk: global loads of stage s of K for the 8 rows of a warp. Load q: lanes 0-15 take the row q of the first half tile, lanes 16-31 the row q of the second one, 16 bytes per lane.
static __device__ __forceinline__ void qwen35_pf_load_rows_ks(
        half2 (* __restrict__ R)[4], const half2 * const __restrict__ K, const int stride_K, const int32_t (* __restrict__ idx)[4], const int s) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        // both numbers first and a select of the values: a select of idx[0][q] or idx[1][q] as places becomes a lane-dependent index into idx (compares and selects)
        const int i0 = idx[0][q];
        const int i1 = idx[1][q];
        const int i  = max(threadIdx.x < 16 ? i0 : i1, 0);
        ggml_cuda_memcpy_1<16>(R[q], K + int64_t(i)*stride_K + s*64 + (threadIdx.x % 16)*4);
    }
}

// pfqk: one stage of K as tile[row][element of the stage], rows 0-31 from the first half tile, 32-63 from the second one.
static __device__ __forceinline__ void qwen35_pf_store_rows_ks(half2 * const __restrict__ tile, const half2 (* __restrict__ R)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int row = (threadIdx.x / 16)*32 + threadIdx.y*4 + q;
        ggml_cuda_memcpy_1<16>(tile + row*QWEN35_PF_STRIDE_KS + (threadIdx.x % 16)*4, R[q]);
    }
}

// One output tile of 64 Q columns: 16 queries and 4 query heads of a KV head, or 8 queries and 8 query heads.
// For each query the float operations and their order are the same as in flash_attn_ext_f16_process_tile<256, 256, ncols1, ncols2, 8, false, false, false>.
// Q stays in shared memory. K and V come in half tiles of 32 rows. The global loads of a half tile go out before the product of the half tile before it, and their row numbers one step earlier.
// pfload (qwen35.pfload, qwen35moe.pfload): the row numbers come in 16-byte loads and the loads of the second K half tile stay before the QK products of the first one, the same results.
// pfqk (qwen35.pfqk, qwen35moe.pfqk, needs pfload): K comes in two stages of 128 elements of all 64 rows instead of two half tiles, so that each Q fragment is read once for the
// products of both row slices of a warp (rows r0 and nrows + r0). Each KQ_C[h] still sums its 16 k steps in the same order, the same results.
template<int ncols1, int ncols2, bool needs_fixup, bool is_fixup, bool pfload, bool pfqk>
static __device__ __forceinline__ void qwen35_pfattn_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_256B_AVAILABLE)
    constexpr int DKQ    = 256;
    constexpr int DV     = 256;
    constexpr int ncols  = ncols1 * ncols2;
    constexpr int nwarps = 8;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    using     T_A_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = nwarps * cols_per_warp/ncols; // Number of parallel warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr int  nrows           = nbatch_fa/2; // K/V rows of a half tile.

    static_assert(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols) == nwarps*warp_size && np == 2 && nbatch_fa == 64 && nbatch_combine == 64 &&
        cols_per_warp == 16 && cols_per_thread == 1 && warp_size == 32, "the kernel is written for this config");
    static_assert(nrows == np*T_A_KQ::I && nrows == np*2*T_A_VKQ::J && nrows == nwarps*4, "a half tile is one WMMA tile per parallel warp");
    static_assert(ncols1 % nwarps == 0 && ncols1 <= 16 && nbatch_fa == 2*warp_size, "the mask tile is 1 or 2 queries per warp and 2 keys per lane");

    constexpr int stride_tile_Q = QWEN35_PF_STRIDE;
    constexpr int stride_tile_K = QWEN35_PF_STRIDE;
    static_assert(stride_tile_Q == DKQ/2 + 4 && QWEN35_PF_STRIDE_VT == nrows + 4 && nrows*stride_tile_K <= (DV/2)*QWEN35_PF_STRIDE_VT, "bad tile strides");
    static_assert(!pfqk || (pfload && QWEN35_PF_STRIDE_KS == DKQ/4 + 4 && nbatch_fa*QWEN35_PF_STRIDE_KS <= (DV/2)*QWEN35_PF_STRIDE_VT),
        "pfqk: needs the pfload loads, and a K stage must fit into the K/V buffer");

    extern __shared__ half2 tile_Q[];
    half2 * tile_KV   = tile_Q + ncols * stride_tile_Q;
    half  * tile_mask = (half *) (tile_KV + (DV/2) * QWEN35_PF_STRIDE_VT);

    T_C_VKQ VKQ_C[DV/(2*T_C_VKQ::J)];

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, it stays there for all K/V rows.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int jc0 = 0; jc0 < ncols; jc0 += nwarps) {
        const int jc = jc0 + threadIdx.y;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        if (jt*ncols1 + j < int(ne01.z) && zt_gqa*ncols2 + c < gqa_ratio) {
#pragma unroll
            for (int k0 = 0; k0 < DKQ/2; k0 += warp_size) {
                const int k = k0 + threadIdx.x;

                const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
            }
        } else {
#pragma unroll
            for (int k0 = 0; k0 < DKQ/2; k0 += warp_size) {
                const int k = k0 + threadIdx.x;

                tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
            }
        }
    }

    const int j0 = (threadIdx.y / np)*cols_per_warp; // First Q column of the warp.
    const int r0 = (threadIdx.y % np)*T_A_KQ::I;     // First row of the warp in a half tile.

    int kb0 = kb0_start;

    // Rows and mask values on their way from global to shared memory, and the row numbers of the two half tiles.
    __align__(16) half2 R[4][4];
    uint32_t M[ncols1/nwarps];
    int32_t idx[2][4];
    qwen35_pf_load_index<pfload>(idx[0], kb0*nbatch_fa,         indices);
    qwen35_pf_load_index<pfload>(idx[1], kb0*nbatch_fa + nrows, indices);
    if constexpr (pfqk) {
        qwen35_pf_load_rows_ks(R, K_h2, stride_K, idx, 0);
    } else {
        qwen35_pf_load_rows<pfload>(R, K_h2, stride_K, idx[0]);
    }
    qwen35_pf_load_mask<ncols1, nwarps>(M, mask_h, stride_mask, kb0*nbatch_fa, jt*ncols1, ne01);

    __syncthreads();

    // The first iteration runs unconditionally, like the last one of flash_attn_ext_f16_process_tile.
    do {
        const int  k_VKQ_0  = kb0 * nbatch_fa;
        const bool has_next = kb0 + 1 < kb0_stop;
        T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];

        qwen35_pf_store_mask<ncols1, nwarps, nbatch_fa>(tile_mask, M);

        // Calculate tile of KQ:
        if constexpr (pfqk) {
            // Stage s holds the k steps 8*s .. 8*s + 7 of both row slices: one Q fragment for two products, each KQ_C[h] gets its k steps in order.
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                qwen35_pf_store_rows_ks(tile_KV, R);
                __syncthreads();

                if (s == 0) {
                    qwen35_pf_load_rows_ks(R, K_h2, stride_K, idx, 1);
                    // otherwise the compiler moves these loads behind the QK products
                    __builtin_amdgcn_sched_barrier(0);
                } else {
                    qwen35_pf_load_rows<pfload>(R, V_h2, stride_V, idx[0]);
                    if (has_next) {
                        qwen35_pf_load_index<pfload>(idx[0], k_VKQ_0 + nbatch_fa, indices);
                    }
                }

#pragma unroll
                for (int k_KQ_0 = 0; k_KQ_0 < DKQ/4; k_KQ_0 += T_A_KQ::J) {
                    T_B_KQ Q_B;
                    load_ldmatrix(Q_B, tile_Q + j0*stride_tile_Q + s*(DKQ/4) + k_KQ_0, stride_tile_Q);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        T_A_KQ K_A;
                        load_ldmatrix(K_A, tile_KV + (h*nrows + r0)*QWEN35_PF_STRIDE_KS + k_KQ_0, QWEN35_PF_STRIDE_KS);
                        mma(KQ_C[h], K_A, Q_B);
                    }
                }

                if (s == 0) {
                    __syncthreads();
                }
            }
        } else {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                qwen35_pf_store_rows(tile_KV, R);
                __syncthreads();

                if (h == 0) {
                    qwen35_pf_load_rows<pfload>(R, K_h2, stride_K, idx[1]);
                    if constexpr (pfload) {
                        // otherwise the compiler moves these loads behind the QK products
                        __builtin_amdgcn_sched_barrier(0);
                    }
                } else {
                    qwen35_pf_load_rows<pfload>(R, V_h2, stride_V, idx[0]);
                    if (has_next) {
                        qwen35_pf_load_index<pfload>(idx[0], k_VKQ_0 + nbatch_fa, indices);
                    }
                }

#pragma unroll
                for (int k_KQ_0 = 0; k_KQ_0 < DKQ/2; k_KQ_0 += T_A_KQ::J) {
                    T_B_KQ Q_B;
                    load_ldmatrix(Q_B, tile_Q + j0*stride_tile_Q + k_KQ_0, stride_tile_Q);
                    T_A_KQ K_A;
                    load_ldmatrix(K_A, tile_KV + r0*stride_tile_K + k_KQ_0, stride_tile_K);
                    mma(KQ_C[h], K_A, Q_B);
                }

                if (h == 0) {
                    __syncthreads();
                }
            }
        }

        float KQ_max_new[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_max_new[col] = KQ_max[col];
        }
        float KQ_rowsum_add[cols_per_thread] = {0.0f};

#pragma unroll
        for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
            const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;

#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                const int i = i0 + T_C_KQ::get_j(l);
                const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / ncols2;

                KQ_C[i00/(np*T_C_KQ::J)].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                constexpr int KQ_idx = 0;
                KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 16;
            constexpr int offset_last  = 16;
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                constexpr int KQ_idx = 0;
                KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
            }
        }

        {
            float KQ_max_scale[cols_per_thread];
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
                KQ_max_scale[col] = expf(KQ_max_diff);
                KQ_max[col] = KQ_max_new[col];

                *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
            }

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        }

        // Convert KQ C tiles into B tiles for VKQ calculation:
        T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }

        __syncthreads();

        // Calculate VKQ tile:
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            qwen35_pf_store_rows_t(tile_KV, R);
            __syncthreads();

            if (h == 0) {
                qwen35_pf_load_rows<pfload>(R, V_h2, stride_V, idx[1]);
                if (has_next) {
                    qwen35_pf_load_index<pfload>(idx[1], k_VKQ_0 + nbatch_fa + nrows, indices);
                }
            } else if (has_next) {
                if constexpr (pfqk) {
                    qwen35_pf_load_rows_ks(R, K_h2, stride_K, idx, 0);
                } else {
                    qwen35_pf_load_rows<pfload>(R, K_h2, stride_K, idx[0]);
                }
                qwen35_pf_load_mask<ncols1, nwarps>(M, mask_h, stride_mask, k_VKQ_0 + nbatch_fa, jt*ncols1, ne01);
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < DV; i_VKQ_0 += T_A_VKQ::I) {
                // The elements i_VKQ_0 + 2*(lane % 16) and the one after it, 16 rows each.
                const half2 * src = tile_KV + qwen35_pf_pair_pos(i_VKQ_0/2 + threadIdx.x % 16)*QWEN35_PF_STRIDE_VT + r0/2;
                T_A_VKQ A;
                ggml_cuda_memcpy_1<16>(A.x +  0, src);
                ggml_cuda_memcpy_1<16>(A.x +  4, src + 4);
                ggml_cuda_memcpy_1<16>(A.x +  8, src + nrows/2);
                ggml_cuda_memcpy_1<16>(A.x + 12, src + nrows/2 + 4);
                mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[h]);
            }

            __syncthreads();
        }
    } while (++kb0 < kb0_stop);

    // Finally, sum up partial KQ rowsums.
    {
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // Combine VKQ accumulator values of the parallel warps.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;

        if (thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();
    }

    {
        constexpr int nmeta = np*cols_per_warp/warp_size;
        static_assert(nmeta == 1, "bad nmeta");

        float KQ_cmn;
        float KQ_cms[nmeta];
        float KQ_crs;

        const int jc_meta = threadIdx.y*cols_per_warp + threadIdx.x;
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;

        if (threadIdx.y % np == 0) {
            // Combine the meta data for parallel warps via shared memory.
            float2 meta[nmeta];
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
            }

            KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, warp_size));
            }

#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
            }

            KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, warp_size);
            }
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // Write back combined meta data:
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }

            // Combined KQ max + rowsum.
            if (needs_fixup && threadIdx.x < cols_per_warp) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
            if (is_fixup && threadIdx.x < cols_per_warp) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
        for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const int j = j0 + T_C_VKQ::get_i(l);
                const int k = k1 + T_C_VKQ::get_j(l);

                tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
            }
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += nwarps/np) {
                const int jc_dst = jc0_dst + threadIdx.y/np;

                const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                const int j_dst = jc_dst / ncols2;
                const int c_dst = jc_dst % ncols2;

                if (!is_fixup && (jt*ncols1 + j_dst >= int(ne01.z) || zt_gqa*ncols2 + c_dst >= gqa_ratio)) {
                    continue;
                }

                const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                for (int k0 = 0; k0 < nbatch_combine; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                    for (int ip = 0; ip < np; ++ip) {
                        const float KQ_crs = meta_j[ip*cols_per_warp * tile_stride + 0];
                        const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
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
                        dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_256B_AVAILABLE)
}

// Same work split over the blocks as flash_attn_ext_f16<256, 256, ncols1, ncols2, false, false, false>. The caller makes sure that there is a mask, no sinks and no logit softcap.
// With pfload the caller makes sure that KV_rows is set, 16-byte aligned, and that ne11 is a multiple of 4. pfqk needs pfload.
template<int DKQ, int DV, int ncols1, int ncols2, bool pfload = false, bool pfqk = false>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void qwen35_pfattn_f16(
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
    ggml_cuda_pdl_sync();
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_256B_AVAILABLE)
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    GGML_UNUSED_VARS(sinks_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne10, ne13, ne31);

    static_assert(DKQ == 256 && DV == 256 && ((ncols1 == 16 && ncols2 == 4) || (ncols1 == 8 && ncols2 == 8)), "the kernel is written for these shapes");
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols1*ncols2);

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_V    = nb21 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    // If the seams of 2 blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        const half   * mask_h = (const half *) (mask + nb33*(sequence % ne33) + nb32*(zt_Q % ne32));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

        const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*z_KV);
        const int32_t * indices = KV_rows_ptr ? KV_rows_ptr + int64_t(sequence)*ne11 : nullptr;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // The block is working on an entire tile.
            qwen35_pfattn_process_tile<ncols1, ncols2, needs_fixup, is_fixup, pfload, pfqk>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dst_meta, scale,
                 ne01, ne02, gqa_ratio, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        } else {
            constexpr bool needs_fixup = true; // The block is missing the beginning of a tile.
            qwen35_pfattn_process_tile<ncols1, ncols2, needs_fixup, is_fixup, pfload, pfqk>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dst_meta, scale,
                 ne01, ne02, gqa_ratio, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = (const half *) (mask + nb33*(sequence % ne33) + nb32*(zt_Q % ne32));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*z_KV);
    const int32_t * indices = KV_rows_ptr ? KV_rows_ptr + int64_t(sequence)*ne11 : nullptr;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    qwen35_pfattn_process_tile<ncols1, ncols2, needs_fixup, is_fixup, pfload, pfqk>
        (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dst_meta, scale,
         ne01, ne02, gqa_ratio, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop);
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
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(AMD_WMMA_256B_AVAILABLE)
}

#endif // defined(GGML_USE_HIP)

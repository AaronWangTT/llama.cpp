#pragma once

// Include after fattn-mma-f16.cuh: it has the WMMA tiles and the config of flash_attn_ext_f16, and it has no include guard.
#include "common.cuh"
#include "fattn-common.cuh"

#if defined(GGML_USE_HIP)

#if defined(AMD_WMMA_256B_AVAILABLE)
// Byte i of the result is byte number sel[i] of the 8 bytes of s1 (0 to 3) and s0 (4 to 7).
static __device__ __forceinline__ uint32_t q35moe_perm_b32(const uint32_t s0, const uint32_t s1, const uint32_t sel) {
    uint32_t d;
    asm("v_perm_b32 %0, %1, %2, %3" : "=v"(d) : "v"(s0), "v"(s1), "v"(sel));
    return d;
}

// 2 blocks of a Q8_0 row: 2 times (half d, 32 int8), 68 bytes. They need no alignment above 4 bytes.
struct q35moe_q8_part {
    typedef int v4i __attribute__((ext_vector_type(4)));

    v4i      c[4];
    uint32_t t;

    __device__ __forceinline__ void load(const char * __restrict__ src) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __builtin_memcpy(&c[i], src + 16*i, sizeof(v4i));
        }
        __builtin_memcpy(&t, src + 64, sizeof(t));
    }

    // No code, only a use of each load: a wait for all of them.
    __device__ __forceinline__ void use() const {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            asm volatile("" : : "v"(c[i]));
        }
        asm volatile("" : : "v"(t));
    }

    __device__ __forceinline__ uint32_t word(const int w) const {
        return w < 16 ? uint32_t(c[w/4][w%4]) : t;
    }

    // The scale of the block that starts at byte b, two times.
    __device__ __forceinline__ half2 d2(const int b) const {
        const uint32_t w = word(b/4);
        const uint32_t d = q35moe_perm_b32(w, w, b % 4 == 0 ? 0x01000100u : 0x03020302u);
        half2 r;
        __builtin_memcpy(&r, &d, sizeof(r));
        return r;
    }

    // The two int8 at byte b times the scale d of their block, the same halves as dequantize_block_q8_0_f16 gives.
    // The bits 0x6400 | (q ^ 0x80) are the half 1152 + q. The sum with -1152 is exact, it is the half that make_half2 gives for q.
    __device__ __forceinline__ half2 h2(const int b, const half2 d) const {
        const uint32_t p = q35moe_perm_b32(0x64646464u, word(b/4) ^ 0x80808080u, b % 4 == 0 ? 0x04010400u : 0x04030402u);
        half2 q;
        __builtin_memcpy(&q, &p, sizeof(q));
        return (q + make_half2(-1152.0f, -1152.0f)) * d;
    }
};

// half2 of the shared tile of q35moe_fagqa_wmma_tile. It has the A tiles of V and the rows of Q, and at the end the result of the two warps, which needs more.
static constexpr __device__ int q35moe_fagqa_tile_ne() {
    return 2*16*(256/2 + 4);
}

// The shared tile has the values of V as the A tiles of the VKQ steps, 16 half2 for each lane of a step.
// As load_ldmatrix_trans: half2 l0 of an A tile has the low halves of the same half2 of the V rows 2*l0 and 2*l0 + 1, half2 8 + l0 has the high halves.
// v0 and v1 are 4 half2 of two such V rows, dst is the place of this pair of rows in the A tile of the first one.
static __device__ __forceinline__ void q35moe_store_A(half2 * __restrict__ dst, const half2 * __restrict__ v0, const half2 * __restrict__ v1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        dst[c*16 + 0] =  __lows2half2(v0[c], v1[c]);
        dst[c*16 + 8] = __highs2half2(v0[c], v1[c]);
    }
}

// One block of a Q8_0 row: half d and 32 int8, 34 bytes. It needs no alignment above 2 bytes.
struct q35moe_q8_blk {
    typedef int v4i __attribute__((ext_vector_type(4)));

    v4i      c[2];
    uint32_t t;

    __device__ __forceinline__ void load(const char * __restrict__ src) {
        __builtin_memcpy(&c[0], src +  0, sizeof(v4i));
        __builtin_memcpy(&c[1], src + 16, sizeof(v4i));
        uint16_t t16;
        __builtin_memcpy(&t16, src + 32, sizeof(t16));
        t = t16;
    }

    // No code, only a use of each load: a wait for all of them.
    __device__ __forceinline__ void use() const {
        asm volatile("" : : "v"(c[0]));
        asm volatile("" : : "v"(c[1]));
        asm volatile("" : : "v"(t));
    }

    __device__ __forceinline__ uint32_t word(const int w) const {
        return w < 8 ? uint32_t(c[w/4][w%4]) : t;
    }
};

// As q35moe_store_A for the same block a and b of two Q8_0 rows: 16 half2 of each row.
// The byte numbers 12 of q35moe_perm_b32 give zero, so its result has one int8 of each row. The other steps are the ones of q35moe_q8_part::h2 for both of them.
static __device__ __forceinline__ void q35moe_q8_store_A(half2 * __restrict__ dst, const q35moe_q8_blk & a, const q35moe_q8_blk & b) {
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
        dst[p*16 + 0] = (q0 + make_half2(-1152.0f, -1152.0f)) * d;
        dst[p*16 + 8] = (q1 + make_half2(-1152.0f, -1152.0f)) * d;
    }
}

// The part of one KV head that one block of flash_attn_ext_f16<256, 256, 2, 8, false, false, false> takes: the K/V row tiles kb0_start to kb0_stop.
// The WMMA and float operations and their order are the ones of flash_attn_ext_f16_process_tile, so each value of the result has the same bits.
// The loads and the use of shared memory are not the same, as in qwen35_fagqa_wmma_tile: a warp sends the loads of some WMMA steps before it waits,
// K goes to the WMMA registers and Q is in shared memory. The two warps of a block need no barrier in the loop over the K/V rows.
// The shared tile has the A tiles of VKQ for half of the values of the V rows at a time. The second half is loaded after the softmax and takes the place of the first half.
// So the lanes need no work to make their A tiles, and the shared memory of a block stays below the one of flash_attn_ext_f16.
// With more shared memory fewer blocks run at the same time, and the kernel is slower than flash_attn_ext_f16.
// Q8 == K and V are Q8_0 rows, stride_K and stride_V are still in units of 4 bytes. The kernel that this one replaces then gets K and V as F16 from dequantize_block_q8_0_f16.
// Here each thread makes the same halves in registers from the int8 that it loads.
// The caller gives a tile of q35moe_fagqa_tile_ne() half2.
template<bool needs_fixup, bool is_fixup, bool Q8>
static __device__ __forceinline__ void q35moe_fagqa_wmma_tile(
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
    constexpr int ncols2 = 8;
    constexpr int ncols  = 16;
    constexpr int NB     = 4; // WMMA steps of K whose loads go out together
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

    constexpr int ne_tile_A     = np*(DV/4)*T_A_VKQ::ne; // A tiles of both warps: one for each half2 of half a V row
    constexpr int stride_tile_Q = DKQ/2;                 // rows of Q after the A tiles
    constexpr int tile_stride   = nbatch_combine + 4;

    half2 * const tile_A = tile + (threadIdx.y % np)*(DV/4)*T_A_VKQ::ne; // A tiles of this warp
    half2 * const tile_Q = tile + ne_tile_A;

    // The config of flash_attn_ext_f16<256, 256, 2, 8> that this code is made for.
    static_assert(warp_size == 32 && nwarps == 2 && np == 2 && cols_per_warp == 16 && nbatch_fa == 32, "other split of the KV rows");
    static_assert(nbatch_V2 == DV/2 && nbatch_combine == DV/2, "other tile sizes");
    static_assert(ne_tile_A + ncols2*stride_tile_Q <= q35moe_fagqa_tile_ne(), "shared tile too small for V and Q");
    static_assert(nwarps*cols_per_warp*tile_stride <= q35moe_fagqa_tile_ne(), "shared tile too small for the result");
    static_assert(T_A_KQ::I == 16 && T_A_KQ::J == 8 && T_B_KQ::J == 8 && T_C_KQ::ne == 8 && T_A_VKQ::I == 32 && T_A_VKQ::ne == 16 && T_C_VKQ::J == 16, "other WMMA tiles");
    static_assert(DKQ/2 == T_A_KQ::I*T_A_KQ::J, "a warp must have as many WMMA steps of K as rows of V");
    static_assert(T_A_KQ::I % NB == 0 && NB % 4 == 0, "a burst must have whole parts of a Q8_0 row and pairs of F16 V rows");
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

        // K row of this thread and mask of the warp. A negative row is padding, its mask is -inf.
        const int row_K = max(rows[k_warp + lane], 0);
        __align__(16) half mask_w[T_A_KQ::I];
#pragma unroll
        for (int i = 0; i < T_A_KQ::I; i += 8) {
            ggml_cuda_memcpy_1<16>(mask_w + i, mask_h + k_warp + i);
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
        T_C_KQ KQ_C;
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
                    q35moe_store_A(tile_A + c_Vh*T_A_VKQ::ne + (i_Vh + i0/2 + b)/2, V_tmp[b], V_tmp[b + 1]);
                }
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    T_B_KQ Q_B;
                    ggml_cuda_memcpy_1<16>(Q_B.x + 0, Q_lane + (i0 + b)*T_B_KQ::J + 0);
                    ggml_cuda_memcpy_1<16>(Q_B.x + 4, Q_lane + (i0 + b)*T_B_KQ::J + 4);
                    mma(KQ_C, K_A[b], Q_B);
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
                    q35moe_q8_store_A(tile_A + u_Vq*(QK8_0/2)*T_A_VKQ::ne + m_Vq, V_b[0], V_b[1]);
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
                    T_B_KQ Q_B;
                    ggml_cuda_memcpy_1<16>(Q_B.x + 0, Q_lane + s*T_B_KQ::J + 0);
                    ggml_cuda_memcpy_1<16>(Q_B.x + 4, Q_lane + s*T_B_KQ::J + 4);
                    mma(KQ_C, K_A, Q_B);
                }
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
                    q35moe_q8_store_A(tile_A + u_Vq*(QK8_0/2)*T_A_VKQ::ne + m_Vq, V_b[0], V_b[1]);
                } else {
#pragma unroll
                    for (int i = 0; i < T_A_KQ::I/2; i += 2) {
                        q35moe_store_A(tile_A + c_Vh*T_A_VKQ::ne + (i_Vh + i)/2, V_tmp[i], V_tmp[i + 1]);
                    }
                }
            }

            // The A tile of this lane for the step, see q35moe_store_A.
            T_A_VKQ A;
#pragma unroll
            for (int l = 0; l < T_A_VKQ::ne; l += 4) {
                ggml_cuda_memcpy_1<16>(A.x + l, tile_A + ((i_VKQ_0 % (DV/2))/2 + lane)*T_A_VKQ::ne + l);
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

// The K/V row tiles kb0_start to kb0_stop of the KV head that has the tile kbc.
template<bool needs_fixup, bool is_fixup, bool Q8>
static __device__ __forceinline__ void q35moe_fagqa_wmma_part(
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
        const int32_t ne02, const int32_t nb02, const int32_t nb03,
        const int32_t ne11, const int32_t ne12,
        const int32_t nb11, const int32_t nb12,
        const int32_t nb21, const int32_t nb22,
        const int32_t ne33, const int64_t nb33,
        const int kbc, const int kb0_start, const int kb0_stop) {
    constexpr int DV        = 256;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(256, DV, 16);

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
    float2        * dstk   = ((float2 *) dst) + (sequence*ne02 + zt_Q) * (DV/2);

    q35moe_fagqa_wmma_tile<needs_fixup, is_fixup, Q8>
        (Q_f2, K_h2, V_h2, mask_h, rows, dstk, dst_meta, tile, scale, nb02 / sizeof(float2), nb11 / sizeof(half2), nb21 / sizeof(half2),
         kb0_start, min(kb0_stop, KV_max[sequence] / nbatch_fa));
}
#endif // defined(AMD_WMMA_256B_AVAILABLE)

// One-token attention over a list of K/V rows for 8 query heads per KV head, head size 256, one mask row for all heads. K and V are F16 or Q8_0 rows (Q8).
// It takes the place of flash_attn_ext_f16<256, 256, 2, 8, false, false, false> with the Stream-K plan of launch_fattn: same grid, same KV rows per block.
// A block can have the end of one KV head and then the start of the next one, as in that kernel. The caller makes sure that a block has less than one KV head.
// The result has the same bits after the fixup kernel of that plan, see q35moe_fagqa_wmma_tile.
template<bool Q8>
__launch_bounds__(64, 2)
static __global__ void q35moe_fagqa_wmma(
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
    constexpr int DV        = 256;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(256, DV, 16);

    __shared__ half2 tile[q35moe_fagqa_tile_ne()];

    const int iter_k = ne11 / nbatch_fa;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*ne12*ne03) / gridDim.x;

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    if (kb0_stop == iter_k) {
        // This block finishes a KV head, that part goes to dst.
        q35moe_fagqa_wmma_part<true, false, Q8>
            (Q_ptr, K_ptr, V_ptr, mask_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, tile, scale,
             ne02, nb02, nb03, ne11, ne12, nb11, nb12, nb21, nb22, ne33, nb33, kbc, kb0_start, kb0_stop);

        kbc += iter_k;
        kbc -= kbc % iter_k;
        if (kbc >= kbc_stop) {
            return;
        }

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);

        // The first warp still reads the shared tile for dst.
        __syncthreads();
    }

    q35moe_fagqa_wmma_part<false, true, Q8>
        (Q_ptr, K_ptr, V_ptr, mask_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, tile, scale,
         ne02, nb02, nb03, ne11, ne12, nb11, nb12, nb21, nb22, ne33, nb33, kbc, kb0_start, kb0_stop);
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

// GGML_Q35MOE_FAGQA=1 takes the one-token attention of qwen35moe (head size 256, 8 query heads per KV head, K/V rows, F16 or Q8_0 K and V) on gfx1103.
// mma_f16 says that the op goes to flash_attn_ext_f16 without the switch. The result has the same bits as with that kernel, also for Q8_0 K and V.
static bool ggml_cuda_flash_attn_ext_q35moe_fagqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool mma_f16) {
    static const bool enable = [] { const char * s = getenv("GGML_Q35MOE_FAGQA"); return s && strcmp(s, "1") == 0; }();
    if (!enable) {
        return false;
    }

    constexpr int DKQ    = 256;
    constexpr int DV     = 256;
    constexpr int ncols1 = 2;
    constexpr int ncols2 = 8;
    constexpr int ncols  = ncols1*ncols2;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * rows = dst->src[5];

    // An op of this shape is the one-token attention of the model: each reason that sends it to the stock kernel goes to the log once.
    // Ops of other shapes (other models, prompt batches) come here too, they say nothing.
    const bool   candidate = Q->ne[0] == DKQ && Q->ne[1] == 1 && Q->ne[2] == ncols2*K->ne[2] && !dst->src[4];
    const char * func      = __func__;
    static bool  logged[10] = {false};
    const auto stock = [&](const int reason, const char * why) {
        if (candidate && !logged[reason]) {
            logged[reason] = true;
            GGML_LOG_WARN("%s: GGML_Q35MOE_FAGQA is not used: %s (sequences = %lld, K %s, V %s)\n",
                func, why, (long long) Q->ne[3], ggml_type_name(K->type), ggml_type_name(V->type));
        }
        return false;
    };

    // The op must be one that goes to flash_attn_ext_f16<256, 256, 2, 8, false, false, false> with its GQA path.
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
    if (Q->ne[0] != DKQ || K->ne[0] != DKQ || V->ne[0] != DV || Q->ne[1] != 1 || Q->ne[2] != ncols2*K->ne[2] || V->ne[2] != K->ne[2]) {
        return stock(4, "the shape of K or V");
    }
    if (rows->ne[0] % FATTN_KQ_STRIDE != 0 || rows->ne[1] != Q->ne[3] || mask->ne[0] != rows->ne[0] || mask->ne[2] != 1) {
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

    // The plan of launch_fattn for that kernel.
    const int nsm       = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const int nthreads  = ggml_cuda_fattn_mma_get_nthreads (DKQ, DV, ncols, cc);
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc);
    const int nwarps    = nthreads / warp_size;
    if (warp_size != WARP_SIZE || nwarps != 2 || nbatch_fa != 32 || !ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, ncols, cc) ||
            ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc) > 1 || ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols, cc)) {
        return stock(9, "the configuration of the WMMA kernel");
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

    // The two tests of the plan depend on the number of sequences in the batch, so they do not go to the log.
    const int max_blocks               = max_blocks_per_sm*nsm;
    const int tiles_nwaves             = (ntiles_dst + max_blocks - 1) / max_blocks;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);
    if (tiles_efficiency_percent >= 75) {
        return false; // No Stream-K.
    }
    const int nblocks_raw     = std::min(max_blocks, ntiles_KV*ntiles_dst);
    const int nblocks_rounded = (nblocks_raw / ntiles_dst) * ntiles_dst;
    const int nblocks         = nblocks_rounded > 0 && 100 * (nblocks_raw - nblocks_rounded) / nblocks_raw <= 5 ? nblocks_rounded : nblocks_raw;
    if (nblocks < 2*ntiles_dst) {
        return false; // A block of that plan can have a whole KV head.
    }
    const bool uniform = nblocks % ntiles_dst == 0; // Each block is in one KV head.

    static bool logged_use = false;
    if (!logged_use) {
        logged_use = true;
        GGML_LOG_INFO("%s: GGML_Q35MOE_FAGQA is used, K and V %s, kv = %lld, sequences = %lld, blocks = %d\n",
            __func__, ggml_type_name(K->type), (long long) n_kv, (long long) Q->ne[3], nblocks);
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

    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool, size_t(nblocks) * ncols * (2 + DV/2));

    const dim3 blocks_num(nblocks, 1, 1);
    const dim3 block_dim(warp_size, nwarps, 1);
    ggml_cuda_kernel_launch(q8 ? q35moe_fagqa_wmma<true> : q35moe_fagqa_wmma<false>, ggml_cuda_kernel_launch_params(blocks_num, block_dim, 0, stream),
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
        (const int *) KV_max.ptr, (const int *) rows->data, (float *) dst->data, dst_tmp_meta.ptr, ggml_get_op_params_f32(dst, 0),
        Q->ne[2], Q->ne[3], Q->nb[2], Q->nb[3],
        n_kv, K->ne[2],
        K->nb[1], K->nb[2],
        V->nb[1], V->nb[2],
        mask->ne[3], mask->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    // The fixup of launch_fattn.
    const dim3 block_dim_combine(DV, 1, 1);
    if (uniform) {
        const uint3 fd0 = init_fastdiv_values(K->ne[2]);
        const uint3 fd1 = init_fastdiv_values(1);
        const uint3 fd2 = init_fastdiv_values(1);

        const dim3 blocks_num_combine(ntiles_dst, ncols1, ncols2);
        ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>,
            ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, stream),
            (float *) dst->data, dst_tmp_meta.ptr,
            int(Q->ne[1]), int(Q->ne[2]), int(K->ne[2]), nblocks,
            int(Q->ne[2] / K->ne[2]), nblocks / ntiles_dst, fd0, fd1, fd2);
    } else {
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

    return true;
}

#endif // defined(GGML_USE_HIP)

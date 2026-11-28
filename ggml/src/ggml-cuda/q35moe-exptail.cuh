#pragma once

// Include from mmq.cu after mmq.cuh.

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private kernel for gfx1103 only"
#endif

// qwen35moe.exptail: expert products of qwen35moe (256 experts, 8 used) with K-quant weights.
// The tile specs are 64 columns wide. A tile takes the narrowest body that holds its valid columns, so the weights of an expert are
// unpacked once for up to 64 columns and the products and the y loads follow the columns in steps of 16. Same results as mul_mat_q.
constexpr int q35moe_exptail_width = 64;

static bool q35moe_exptail_enabled() {
    return ggml_kperf_on(GGML_KPERF_QWEN35MOE_EXPTAIL);
}

template <ggml_type type, int W>
static __device__ __forceinline__ void q35moe_exptail_body(const char * x, const int * y, const int * ids,
        float * dst, int xoff, int sx, int sy, int sd, int imax, int useful, int nbk) {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr auto c = ggml_cuda_mmq_get_config(type, W, false, GGML_PREC_Q8);
    static_assert(c.type == type && c.J == W && W % 16 == 0 && W <= q35moe_exptail_width && !c.fallback && !c.stream_k, "bad config");
    static_assert(c.I == 64 && c.nthreads == 128 && ggml_cuda_get_physical_warp_size() == 32 && c.rows_per_warp() == 16 && c.use_mma_data_layout(), "bad tile");
    // a narrow body runs in the shared memory of the launch for 64 columns
    static_assert(ggml_cuda_mmq_get_sram_stride(c.sram_layout) == 76 && MMQ_TILE_Y_K == 36 && sizeof(block_q8_1_mmq) == 144, "bad tile sizes");
#endif
    mul_mat_q_process_tile<type, W, false, false, GGML_PREC_Q8, false>(
        x, nullptr, xoff, y, ids, dst, nullptr, nullptr, GGML_GLU_OP_SWIGLU, 0.0f, nullptr, nullptr,
        sx, sy, sd, imax, useful, 0, nbk);
}

// Grid: 64 row tiles of x times the tile specs of mm_ids_helper for a tile width of 64, like mul_mat_q for MoE on HIP.
template <ggml_type type>
__launch_bounds__(128, 1)
static __global__ void q35moe_exptail(const char * __restrict__ x, const int * __restrict__ y,
        const int * __restrict__ ids, const ggml_cuda_mm_id_tile_spec * __restrict__ specs, float * __restrict__ dst,
        int sx, int sex, int sy, int sd, int nrow, int nbk) {
    const ggml_cuda_mm_id_tile_spec spec = specs[blockIdx.y];
    const int useful = spec.col_high - spec.col_low;
    if (useful == 0) {
        return;
    }
    const int row  = int(blockIdx.x)*64;
    const int xoff = spec.expert*sex + row*sx;
    const int imax = nrow - row - 1;
    y   += spec.col_low*(sizeof(block_q8_1_mmq)/sizeof(int));
    dst += row;

    extern __shared__ int data_mul_mat_q[];
    const int j = threadIdx.y*32 + threadIdx.x;
    if (j < useful) {
        data_mul_mat_q[j] = ids[spec.col_low + j];
    }
    __syncthreads();

    if (useful <= 16) {
        q35moe_exptail_body<type, 16>(x, y, data_mul_mat_q, dst, xoff, sx, sy, sd, imax, useful, nbk);
    } else if (useful <= 32) {
        q35moe_exptail_body<type, 32>(x, y, data_mul_mat_q, dst, xoff, sx, sy, sd, imax, useful, nbk);
    } else if (useful <= 48) {
        q35moe_exptail_body<type, 48>(x, y, data_mul_mat_q, dst, xoff, sx, sy, sd, imax, useful, nbk);
    } else {
        q35moe_exptail_body<type, 64>(x, y, data_mul_mat_q, dst, xoff, sx, sy, sd, imax, useful, nbk);
    }
}

// qwen35moe.exptail_pipe: the same tiles with the loads of the next k step in flight during the dot products of the current one.
// A block takes 2 row tiles of 64 rows (the k steps of both tiles run as one sequence of steps). At the start of a step the registers of the
// step are stored into shared memory and the weights and the first activation half of the next step are loaded into registers; the second
// activation half of the next step is loaded after the second half of the current one is stored. The barriers only wait for shared memory,
// so these loads stay in flight across them (__syncthreads also waits for every global load and invalidates L0).
// The weight words are unpacked with the expressions of ggml_cuda_mmq_load_tiles_q4_K / q5_K / q6_K (MMA layout, 64 rows, 4 warps of 32)
// and every output keeps its k order, dot products and write-back, so the results are those of q35moe_exptail.
constexpr int q35moe_exptail_pipe_rows = 2;

static bool q35moe_exptail_pipe_enabled() {
    return ggml_kperf_on(GGML_KPERF_QWEN35MOE_EXPTAIL_PIPE);
}

static __device__ __forceinline__ void q35moe_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// the weight words of one k step (one super block of 64 rows) in registers
template <ggml_type type> struct q35moe_xraw;

template <> struct q35moe_xraw<GGML_TYPE_Q4_K> {
    int   qs[16];
    int   sc[4]; // the words that unpack_scales_q45_K reads for ksc and ksc + 2, no register array indexed at run time
    half2 dm;
    __device__ __forceinline__ void fetch(const char * __restrict__ x, const int kbx0, const int stride) {
#pragma unroll
        for (int it = 0; it < 16; ++it) {
            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + (it*4 + threadIdx.y)*stride;
            qs[it] = get_int_b4(bxi->qs, threadIdx.x);
        }
        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + (threadIdx.y*16 + threadIdx.x/2)*stride;
        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % 2;
        sc[0] = scales[((ksc + 0)%2) + ((ksc + 0) != 0)];
        sc[1] = scales[(ksc + 0)/2];
        sc[2] = scales[((ksc + 2)%2) + ((ksc + 2) != 0)];
        sc[3] = scales[(ksc + 2)/2];
        dm = bxi->dm;
    }
    __device__ __forceinline__ void store(int * __restrict__ x_tile) const {
        constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1);
        int   * x_qs = (int   *)  x_tile;
        half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
        const int txi = threadIdx.x;
#pragma unroll
        for (int it = 0; it < 16; ++it) {
            const int i = it*4 + threadIdx.y;
            x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 0] = (qs[it] >> 0) & 0x0F0F0F0F;
            x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 8] = (qs[it] >> 4) & 0x0F0F0F0F;
        }
        const int i   = threadIdx.y*16 + threadIdx.x/2;
        const int ksc = threadIdx.x % 2;
        const int sc32 = ((sc[0] >> (4 * ((ksc + 0) & ((ksc + 0)/2)))) & 0x0F0F0F0F) | ((sc[1] >> (2 * ((ksc + 0) % 2))) & 0x30303030);
        const int  m32 = ((sc[2] >> (4 * ((ksc + 2) & ((ksc + 2)/2)))) & 0x0F0F0F0F) | ((sc[3] >> (2 * ((ksc + 2) % 2))) & 0x30303030);
        const uint8_t * sc8 = (const uint8_t *) &sc32;
        const uint8_t *  m8 = (const uint8_t *)  &m32;
        const half2 dmx = dm * make_half2(1.0f, -1.0f);
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_dm[i*sram_stride + sizeof(int)*ksc + l] = dmx*make_half2(sc8[l], m8[l]);
        }
    }
};

template <> struct q35moe_xraw<GGML_TYPE_Q5_K> {
    int   ql[16];
    int   qh[16];
    int   sc[4];
    half2 dm;
    __device__ __forceinline__ void fetch(const char * __restrict__ x, const int kbx0, const int stride) {
        const int txi = threadIdx.x;
#pragma unroll
        for (int it = 0; it < 16; ++it) {
            const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + (it*4 + threadIdx.y)*stride;
            ql[it] = get_int_b4(bxi->qs, txi);
            qh[it] = get_int_b4(bxi->qh, txi % (QI5_K/4));
        }
        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + (threadIdx.y*16 + threadIdx.x/2)*stride;
        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % 2;
        sc[0] = scales[((ksc + 0)%2) + ((ksc + 0) != 0)];
        sc[1] = scales[(ksc + 0)/2];
        sc[2] = scales[((ksc + 2)%2) + ((ksc + 2) != 0)];
        sc[3] = scales[(ksc + 2)/2];
        dm = bxi->dm;
    }
    __device__ __forceinline__ void store(int * __restrict__ x_tile) const {
        constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1);
        int   * x_qs = (int   *)  x_tile;
        half2 * x_dm = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
        const int txi = threadIdx.x;
#pragma unroll
        for (int it = 0; it < 16; ++it) {
            const int i  = it*4 + threadIdx.y;
            const int ky = QR5_K*txi;
            const int ql0 = (ql[it] >> 0) & 0x0F0F0F0F;
            const int ql1 = (ql[it] >> 4) & 0x0F0F0F0F;
            const int qh0 = ((qh[it] >> (2 * (txi / (QI5_K/4)) + 0)) << 4) & 0x10101010;
            const int qh1 = ((qh[it] >> (2 * (txi / (QI5_K/4)) + 1)) << 4) & 0x10101010;
            const int kq0 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + 0;
            const int kq1 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + QI5_K/4;
            x_qs[i*sram_stride + kq0] = ql0 | qh0;
            x_qs[i*sram_stride + kq1] = ql1 | qh1;
        }
        const int i   = threadIdx.y*16 + threadIdx.x/2;
        const int ksc = threadIdx.x % 2;
        const int sc32 = ((sc[0] >> (4 * ((ksc + 0) & ((ksc + 0)/2)))) & 0x0F0F0F0F) | ((sc[1] >> (2 * ((ksc + 0) % 2))) & 0x30303030);
        const int  m32 = ((sc[2] >> (4 * ((ksc + 2) & ((ksc + 2)/2)))) & 0x0F0F0F0F) | ((sc[3] >> (2 * ((ksc + 2) % 2))) & 0x30303030);
        const uint8_t * sc8 = (const uint8_t *) &sc32;
        const uint8_t *  m8 = (const uint8_t *)  &m32;
        const half2 dmx = dm * make_half2(1.0f, -1.0f);
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_dm[i*sram_stride + sizeof(int)*ksc + l] = dmx*make_half2(sc8[l], m8[l]);
        }
    }
};

template <> struct q35moe_xraw<GGML_TYPE_Q6_K> {
    int  ql[16];
    int  qh[16];
    int  sc[2];
    half d;
    __device__ __forceinline__ void fetch(const char * __restrict__ x, const int kbx0, const int stride) {
        const int txi = threadIdx.x;
#pragma unroll
        for (int it = 0; it < 16; ++it) {
            const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + (it*4 + threadIdx.y)*stride;
            ql[it] = get_int_b2(bxi->ql, txi);
            qh[it] = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        }
        d = ((const block_q6_K *) x + kbx0 + ((threadIdx.y*32 + threadIdx.x) % 64)*stride)->d;
#pragma unroll
        for (int i0 = 0; i0 < 64; i0 += 32) {
            const int i = (i0 + threadIdx.y*8 + threadIdx.x/(MMQ_TILE_NE_K/8)) % 64;
            const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / 4;
            sc[i0/32] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
        }
    }
    __device__ __forceinline__ void store(int * __restrict__ x_tile) const {
        constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K);
        int   * x_qs = (int   *)  x_tile;
        float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
        int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);
        const int txi = threadIdx.x;
#pragma unroll
        for (int it = 0; it < 16; ++it) {
            const int i = it*4 + threadIdx.y;
            const int ql0 = (ql[it] >> 0) & 0x0F0F0F0F;
            const int ql1 = (ql[it] >> 4) & 0x0F0F0F0F;
            const int qh0 = ((qh[it] >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
            const int qh1 =  (qh[it] >> ((txi & 0x08) >> 2))       & 0x30303030;
            const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
            const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;
            x_qs[i*sram_stride + kq0] = __vsub4(ql0 | qh0, 0x20202020);
            x_qs[i*sram_stride + kq1] = __vsub4(ql1 | qh1, 0x20202020);
        }
        x_df[((threadIdx.y*32 + threadIdx.x) % 64)*sram_stride] = d;
#pragma unroll
        for (int i0 = 0; i0 < 64; i0 += 32) {
            const int i = (i0 + threadIdx.y*8 + threadIdx.x/(MMQ_TILE_NE_K/8)) % 64;
            x_sc[i*sram_stride + threadIdx.x%4] = sc[i0/32];
        }
    }
};

// Row tiles [row, row + 64*R) of one tile spec; W as in q35moe_exptail_body.
template <ggml_type type, int W, int R>
static __device__ __forceinline__ void q35moe_exptail_pipe_body(const char * __restrict__ x, const int * __restrict__ y, const int * __restrict__ ids,
        float * __restrict__ dst, const int xoff, const int sx, const int sy, const int sd, const int imax, const int useful, const int nbk) {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr auto c = ggml_cuda_mmq_get_config(type, W, false, GGML_PREC_Q8);
    static_assert(c.type == type && c.J == W && W % 16 == 0 && W <= q35moe_exptail_width && !c.fallback && !c.stream_k, "bad config");
    static_assert(c.I == 64 && c.nthreads == 128 && ggml_cuda_get_physical_warp_size() == 32 && c.rows_per_warp() == 16 && c.use_mma_data_layout(), "bad tile");
    static_assert(ggml_cuda_mmq_get_sram_stride(c.sram_layout) == 76 && MMQ_TILE_Y_K == 36 && sizeof(block_q8_1_mmq) == 144, "bad tile sizes");
#endif
    constexpr int nthreads = 128;
    constexpr int I        = 64;
    constexpr int sz       = sizeof(block_q8_1_mmq) / sizeof(int);
    constexpr int NY       = GGML_PAD(W*MMQ_TILE_Y_K, nthreads) / nthreads;
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, W, false, GGML_PREC_Q8>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, W, false, GGML_PREC_Q8, false>();

    // same shared memory layout as mul_mat_q_process_tile
    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + W;
    int * tile_x = tile_y + GGML_PAD(W*MMQ_TILE_Y_K, nthreads);

    const int tid = threadIdx.y*32 + threadIdx.x;

    q35moe_xraw<type> xr;
    int y0r[NY];
    int y1r[NY];
    xr.fetch(x, xoff, sx);
#pragma unroll
    for (int l0 = 0; l0 < W*MMQ_TILE_Y_K; l0 += nthreads) {
        y0r[l0/nthreads] = y[l0 + tid];
    }
#pragma unroll
    for (int l0 = 0; l0 < W*MMQ_TILE_Y_K; l0 += nthreads) {
        y1r[l0/nthreads] = y[sy*sz + l0 + tid];
    }

    float sum[W*I/nthreads] = {0.0f};
    const int nstep = R*nbk;
    for (int s = 0; s < nstep; ++s) {
        const int r   = s / nbk;
        const int kb0 = s - r*nbk;
        xr.store(tile_x);
#pragma unroll
        for (int l0 = 0; l0 < W*MMQ_TILE_Y_K; l0 += nthreads) {
            tile_y[l0 + tid] = y0r[l0/nthreads];
        }
        const int rn  = (s + 1) / nbk;
        const int kbn = (s + 1) - rn*nbk;
        if (s + 1 < nstep) {
            xr.fetch(x, xoff + rn*I*sx + kbn, sx);
            const int * by0 = y + sy*(kbn*2)*sz;
#pragma unroll
            for (int l0 = 0; l0 < W*MMQ_TILE_Y_K; l0 += nthreads) {
                y0r[l0/nthreads] = by0[l0 + tid];
            }
        }
        q35moe_lds_barrier();
        vec_dot(tile_x, tile_y, sum, 0);
        q35moe_lds_barrier();
#pragma unroll
        for (int l0 = 0; l0 < W*MMQ_TILE_Y_K; l0 += nthreads) {
            tile_y[l0 + tid] = y1r[l0/nthreads];
        }
        if (s + 1 < nstep) {
            const int * by1 = y + sy*((kbn*2)*sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < W*MMQ_TILE_Y_K; l0 += nthreads) {
                y1r[l0/nthreads] = by1[l0 + tid];
            }
        }
        q35moe_lds_barrier();
        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        q35moe_lds_barrier();
        if (kb0 == nbk - 1) {
            write_back(sum, ids, dst + r*I, nullptr, sd, imax - r*I, useful, GGML_GLU_OP_SWIGLU, 0.0f, nullptr, nullptr);
#pragma unroll
            for (int l = 0; l < W*I/nthreads; ++l) {
                sum[l] = 0.0f;
            }
        }
    }
}

// Grid: blocks of q35moe_exptail_pipe_rows row tiles times the tile specs, the order of q35moe_exptail.
template <ggml_type type>
__launch_bounds__(128, 1)
static __global__ void q35moe_exptail_pipe(const char * __restrict__ x, const int * __restrict__ y,
        const int * __restrict__ ids, const ggml_cuda_mm_id_tile_spec * __restrict__ specs, float * __restrict__ dst,
        int sx, int sex, int sy, int sd, int nrow, int nbk) {
    constexpr int R = q35moe_exptail_pipe_rows;
    const ggml_cuda_mm_id_tile_spec spec = specs[blockIdx.y];
    const int useful = spec.col_high - spec.col_low;
    if (useful == 0) {
        return;
    }
    const int row  = int(blockIdx.x)*64*R;
    const int xoff = spec.expert*sex + row*sx;
    const int imax = nrow - row - 1;
    y   += spec.col_low*(sizeof(block_q8_1_mmq)/sizeof(int));
    dst += row;

    extern __shared__ int data_mul_mat_q[];
    const int j = threadIdx.y*32 + threadIdx.x;
    if (j < useful) {
        data_mul_mat_q[j] = ids[spec.col_low + j];
    }
    q35moe_lds_barrier();

    if (useful <= 16) {
        q35moe_exptail_pipe_body<type, 16, R>(x, y, data_mul_mat_q, dst, xoff, sx, sy, sd, imax, useful, nbk);
    } else if (useful <= 32) {
        q35moe_exptail_pipe_body<type, 32, R>(x, y, data_mul_mat_q, dst, xoff, sx, sy, sd, imax, useful, nbk);
    } else if (useful <= 48) {
        q35moe_exptail_pipe_body<type, 48, R>(x, y, data_mul_mat_q, dst, xoff, sx, sy, sd, imax, useful, nbk);
    } else {
        q35moe_exptail_pipe_body<type, 64, R>(x, y, data_mul_mat_q, dst, xoff, sx, sy, sd, imax, useful, nbk);
    }
}

// the expert tensors of the architecture: [n_in, n_out, 256 experts]
static bool q35moe_exptail_weight(const ggml_tensor * w) {
    const auto ends_with = [w](const char * suffix) {
        const size_t n = strlen(w->name), m = strlen(suffix);
        return n >= m && strcmp(w->name + n - m, suffix) == 0;
    };
    if (w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q5_K && w->type != GGML_TYPE_Q6_K) {
        return false;
    }
    if (w->view_src || !ggml_is_contiguous(w) || w->ne[2] != 256 || w->ne[3] != 1) {
        return false;
    }
    if (w->ne[0] == 2048 && w->ne[1] == 512) {
        return ends_with(".ffn_gate_exps.weight") || ends_with(".ffn_up_exps.weight");
    }
    return w->ne[0] == 512 && w->ne[1] == 2048 && ends_with(".ffn_down_exps.weight");
}

// True if the product takes the kernel; the caller then asks mm_ids_helper for tile specs of q35moe_exptail_width columns.
static bool q35moe_exptail_wanted(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst,
        const int cc, const int warp_size, const bool fallback, const ggml_prec prec_src1, const bool has_fusion) {
    if (!q35moe_exptail_enabled() || cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || warp_size != 32 || has_fusion || fallback || prec_src1 != GGML_PREC_Q8) {
        return false;
    }
    // gate and up read one activation per token, down reads one per used expert
    if (!q35moe_exptail_weight(src0) || dst->op != GGML_OP_MUL_MAT_ID || dst->src[0] != src0 || dst->src[1] != src1 || dst->src[2] != ids ||
            !ggml_is_contiguous(dst) || ids->ne[0] != 8 || src1->ne[0] != src0->ne[0] || src1->ne[1] != (src0->ne[0] == 2048 ? 1 : 8)) {
        return false;
    }
    for (int J = 16; J <= q35moe_exptail_width; J += 16) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, false, cc, prec_src1);
        if (config.type != src0->type || config.I != 64 || config.nthreads != 128 || config.stream_k) {
            return false;
        }
    }
    return true;
}

template <ggml_type type>
static void q35moe_exptail_launch_type(const mmq_args & args, const dim3 grid, const size_t bytes, const bool pipe, cudaStream_t stream) {
    const dim3 block(32, 4);
    const int  nbk = args.ncols_x/ggml_cuda_type_traits<type>::qk;
    if (pipe) {
        const dim3 grid_pipe(grid.x/q35moe_exptail_pipe_rows, grid.y);
        q35moe_exptail_pipe<type><<<grid_pipe, block, bytes, stream>>>(args.x, args.y, args.ids_dst, args.tile_spec, args.dst,
            args.stride_row_x, args.stride_channel_x, args.ncols_y, args.nrows_dst, args.nrows_x, nbk);
        return;
    }
    q35moe_exptail<type><<<grid, block, bytes, stream>>>(args.x, args.y, args.ids_dst, args.tile_spec, args.dst,
        args.stride_row_x, args.stride_channel_x, args.ncols_y, args.nrows_dst, args.nrows_x, nbk);
}

// args are the ones of mul_mat_q for tile specs of q35moe_exptail_width columns
static void q35moe_exptail_launch(const mmq_args & args, const int cc, const ggml_prec prec_src1, cudaStream_t stream) {
    GGML_ASSERT(args.J_best == q35moe_exptail_width && args.nchannels_y == 1 && args.nsamples_y == 1 && args.nrows_x % 64 == 0);
    const dim3   grid(args.nrows_x/64, args.ntj);
    const size_t bytes = mmq_get_nbytes_shared(ggml_cuda_mmq_get_config(args.type_x, args.J_best, false, cc, prec_src1), cc);
    // qwen35moe.exptail_pipe: the weight row counts of q35moe_exptail_weight (512, 2048) are whole blocks of row tiles
    const bool pipe = q35moe_exptail_pipe_enabled() && args.nrows_x % (64*q35moe_exptail_pipe_rows) == 0;
    if (pipe && ggml_kperf_trace(GGML_KPERF_QWEN35MOE_EXPTAIL_PIPE)) {
        GGML_LOG_INFO("Q35MOE_EXPTAIL_PIPE_HIT type=%s rows=%lld k=%lld cols=%lld\n", ggml_type_name(args.type_x),
            (long long) args.nrows_x, (long long) args.ncols_x, (long long) args.ncols_y);
    }
    switch (args.type_x) {
        case GGML_TYPE_Q4_K: q35moe_exptail_launch_type<GGML_TYPE_Q4_K>(args, grid, bytes, pipe, stream); break;
        case GGML_TYPE_Q5_K: q35moe_exptail_launch_type<GGML_TYPE_Q5_K>(args, grid, bytes, pipe, stream); break;
        case GGML_TYPE_Q6_K: q35moe_exptail_launch_type<GGML_TYPE_Q6_K>(args, grid, bytes, pipe, stream); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

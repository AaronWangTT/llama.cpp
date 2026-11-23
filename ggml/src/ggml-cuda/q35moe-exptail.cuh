#pragma once

// Include from mmq.cu after mmq.cuh.

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private kernel for gfx1103 only"
#endif

// GGML_Q35MOE_EXPTAIL=1: expert products of qwen35moe (256 experts, 8 used) with K-quant weights.
// The tile specs are 64 columns wide. A tile takes the narrowest body that holds its valid columns, so the weights of an expert are
// unpacked once for up to 64 columns and the products and the y loads follow the columns in steps of 16. Same results as mul_mat_q.
constexpr int q35moe_exptail_width = 64;

static bool q35moe_exptail_enabled() {
    static const bool enabled = [] { const char * p = getenv("GGML_Q35MOE_EXPTAIL"); return p && strcmp(p, "1") == 0; }();
    return enabled;
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
static void q35moe_exptail_launch_type(const mmq_args & args, const dim3 grid, const size_t bytes, cudaStream_t stream) {
    const dim3 block(32, 4);
    const int  nbk = args.ncols_x/ggml_cuda_type_traits<type>::qk;
    q35moe_exptail<type><<<grid, block, bytes, stream>>>(args.x, args.y, args.ids_dst, args.tile_spec, args.dst,
        args.stride_row_x, args.stride_channel_x, args.ncols_y, args.nrows_dst, args.nrows_x, nbk);
}

// args are the ones of mul_mat_q for tile specs of q35moe_exptail_width columns
static void q35moe_exptail_launch(const mmq_args & args, const int cc, const ggml_prec prec_src1, cudaStream_t stream) {
    GGML_ASSERT(args.J_best == q35moe_exptail_width && args.nchannels_y == 1 && args.nsamples_y == 1 && args.nrows_x % 64 == 0);
    const dim3   grid(args.nrows_x/64, args.ntj);
    const size_t bytes = mmq_get_nbytes_shared(ggml_cuda_mmq_get_config(args.type_x, args.J_best, false, cc, prec_src1), cc);
    switch (args.type_x) {
        case GGML_TYPE_Q4_K: q35moe_exptail_launch_type<GGML_TYPE_Q4_K>(args, grid, bytes, stream); break;
        case GGML_TYPE_Q5_K: q35moe_exptail_launch_type<GGML_TYPE_Q5_K>(args, grid, bytes, stream); break;
        case GGML_TYPE_Q6_K: q35moe_exptail_launch_type<GGML_TYPE_Q6_K>(args, grid, bytes, stream); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

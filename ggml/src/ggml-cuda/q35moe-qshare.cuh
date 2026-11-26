#pragma once

// Include from mmq.cu after mmq.cuh and q35moe-exptail.cuh.

// qwen35moe.qshare: MMQ products of qwen35moe that read the same input share one Q8_1 copy of it, and the gate and up products
// of the routed experts share their column lists. In a prompt graph: the joint projection and its gate, q / k / v of an attention
// block, up and gate of the shared expert, up and gate of the routed experts.
// The first product quantizes into buffers of the backend context (ggml_cuda_qshare in common.cuh). The next product of the same
// graph run with the same src1 (tensor, data, shape, strides) and block layout, and for the experts the same ids and tile width,
// skips its quantization (and mm_ids_helper) and reads these buffers. The product kernels and the bytes they read do not change.
// The graph loop drops the kept data when a node may have written into the memory of the kept src1 or ids: any node that is not
// a plain product and has that memory as its output or as a source (ggml_cuda_qshare_written in ggml-cuda.cu).

static bool q35moe_qshare_enabled() {
    return ggml_kperf_on(GGML_KPERF_QWEN35MOE_QSHARE);
}

// the dense Q8_0 weights of the architecture that read the input of a block: [2048, n_out]
static bool q35moe_qshare_weight(const ggml_tensor * w) {
    static const char * const suffixes[] = {
        ".attn_qkv.weight", ".attn_gate.weight", ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".ffn_up_shexp.weight", ".ffn_gate_shexp.weight",
    };
    if (w->type != GGML_TYPE_Q8_0 || w->view_src || !ggml_is_contiguous(w) || w->ne[0] != 2048 || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    const size_t n = strlen(w->name);
    for (const char * suffix : suffixes) {
        const size_t m = strlen(suffix);
        if (n >= m && strcmp(w->name + n - m, suffix) == 0) {
            return true;
        }
    }
    return false;
}

// True if the product is the node that the graph loop computes and src1 has the form of a block input.
// A product inside a fused op or on a slice of a tensor is not that node.
static bool q35moe_qshare_node(const ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!q35moe_qshare_enabled()) {
        return false;
    }
    if (cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: qwen35moe.qshare is not used: the device is not gfx1103\n", __func__);
        }
        return false;
    }
    if (ctx.qshare.node != dst || ctx.curr_stream_no != 0 || dst->src[1] != src1) {
        return false;
    }
    return src1->type == GGML_TYPE_F32 && src1->data != nullptr && ggml_is_contiguous(src1) && src1->ne[0] == 2048 && src1->ne[3] == 1;
}

// the last condition of a product that passed all others
static bool q35moe_qshare_not_captured(const ggml_backend_cuda_context & ctx) {
    if (ctx.qshare.captured) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: qwen35moe.qshare is not used in a graph that runs as a captured graph\n", __func__);
        }
        return false;
    }
    return true;
}

// a dense product of the architecture on the plain MMQ path
static bool q35moe_qshare_dense_wanted(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * dst, const int cc, const bool fallback, const ggml_prec prec_src1, const bool has_fusion) {
    if (!q35moe_qshare_node(ctx, src1, dst, cc) || has_fusion || fallback || prec_src1 != GGML_PREC_Q8) {
        return false;
    }
    if (dst->op != GGML_OP_MUL_MAT || dst->src[0] != src0 || !q35moe_qshare_weight(src0) || src1->ne[2] != 1) {
        return false;
    }
    return q35moe_qshare_not_captured(ctx);
}

// gate and up of the routed experts (one activation per token) on the kernel of q35moe-exptail.cuh, which fixes the tile width
static bool q35moe_qshare_experts_wanted(const ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, const int cc, const bool exptail, const bool dedup_bcast) {
    if (!dedup_bcast || !q35moe_qshare_node(ctx, src1, dst, cc)) {
        return false;
    }
    if (!exptail) {
        static bool logged = false;
        if (!logged && !q35moe_exptail_enabled()) {
            logged = true;
            GGML_LOG_INFO("%s: qwen35moe.qshare is not used for the routed experts: it needs qwen35moe.exptail\n", __func__);
        }
        return false;
    }
    return ids->data != nullptr && q35moe_qshare_not_captured(ctx);
}

// A larger buffer replaces the old one once the kernels that read it are done.
static void q35moe_qshare_reserve(ggml_cuda_qshare::buffer & b, const size_t size, cudaStream_t stream) {
    if (b.size >= size) {
        return;
    }
    if (b.ptr != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(b.ptr));
        b.ptr  = nullptr;
        b.size = 0;
    }
    // room for the padding of any tile width, so that a product with another padding still fits
    const size_t size_alloc = GGML_PAD(size + 128*sizeof(block_q8_1_mmq), 4096);
    CUDA_CHECK(cudaMalloc(&b.ptr, size_alloc));
    b.size = size_alloc;
}

// True if the kept data was made from the same input: the product reads it. Otherwise the product becomes the one that makes the
// kept data: the buffers get the sizes it needs and the caller fills them. ids, n_experts, tile_width and the list sizes: routed experts only.
static bool q35moe_qshare_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_tensor * ids,
        const int ds_layout, const int64_t ne10_padded, const int64_t n_experts, const int tile_width,
        const size_t nbytes_q8, const size_t nbytes_ids_src1, const size_t nbytes_ids_dst, const size_t nbytes_tile_spec, cudaStream_t stream) {
    ggml_cuda_qshare & qs = ctx.qshare;

    bool same = qs.valid && qs.src1 == src1 && qs.src1_lo == (const char *) src1->data && qs.ds_layout == ds_layout &&
        qs.ne10_padded == ne10_padded && qs.ids == ids && qs.q8.size >= nbytes_q8;
    for (int i = 0; same && i < GGML_MAX_DIMS; ++i) {
        same = qs.src1_ne[i] == src1->ne[i] && qs.src1_nb[i] == src1->nb[i];
    }
    if (same && ids != nullptr) {
        same = qs.ids_lo == (const char *) ids->data && qs.ids_ne[0] == ids->ne[0] && qs.ids_ne[1] == ids->ne[1] && qs.ids_nb1 == ids->nb[1] &&
            qs.n_experts == n_experts && qs.tile_width == tile_width &&
            qs.ids_src1.size >= nbytes_ids_src1 && qs.ids_dst.size >= nbytes_ids_dst && qs.tile_spec.size >= nbytes_tile_spec;
    }
    if (same) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: qwen35moe.qshare is used: a product takes the Q8_1 input of the product before it\n", __func__);
        }
        qs.n_taken++;
        qs.n_taken_graph++;
        return true;
    }

    qs.valid = false;
    q35moe_qshare_reserve(qs.q8, nbytes_q8, stream);
    if (ids != nullptr) {
        q35moe_qshare_reserve(qs.ids_src1,  nbytes_ids_src1,  stream);
        q35moe_qshare_reserve(qs.ids_dst,   nbytes_ids_dst,   stream);
        q35moe_qshare_reserve(qs.tile_spec, nbytes_tile_spec, stream);
    }

    qs.src1    = src1;
    qs.src1_lo = (const char *) src1->data;
    qs.src1_hi = qs.src1_lo + ggml_nbytes(src1);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        qs.src1_ne[i] = src1->ne[i];
        qs.src1_nb[i] = src1->nb[i];
    }
    qs.ds_layout   = ds_layout;
    qs.ne10_padded = ne10_padded;
    qs.ids         = ids;
    if (ids != nullptr) {
        qs.ids_lo     = (const char *) ids->data;
        qs.ids_hi     = qs.ids_lo + ggml_nbytes(ids);
        qs.ids_ne[0]  = ids->ne[0];
        qs.ids_ne[1]  = ids->ne[1];
        qs.ids_nb1    = ids->nb[1];
        qs.n_experts  = n_experts;
        qs.tile_width = tile_width;
    }
    qs.valid = true;
    qs.n_kept++;
    qs.n_kept_graph++;
    return false;
}

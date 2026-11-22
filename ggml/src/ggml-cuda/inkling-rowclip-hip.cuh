// The optional Q/R channel scale is the producer of x, not part of this fusion.
#include "reduce_rows.cuh"

static __global__ void inkling_clip_one(const float * x, float * y, int cols, float threshold) {
    __shared__ float squared[128];
    __shared__ float factor;
    const int row = blockIdx.x, lane = threadIdx.x;
    for (int j = lane; j < cols; j += blockDim.x) {
        float v = x[row*cols+j];
        squared[j] = v*v; // Materialize the original F32 SQR (no contracted multiply-add).
    }
    __syncthreads();
    // The very same helper as SUM_ROWS, including the 8-way unroll, padding and XOR tree.
    float sum = reduce_row_f32(squared, cols);
    if (lane == 0) {
        const float den = sqrtf(sum) + 1e-6f;
        factor = fminf(fmaxf(threshold / den, 0.0f), 1.0f);
    }
    __syncthreads();
    // Allocator reuse at exactly x is safe: no other row/block reads this row;
    // all cross-lane input reads precede the barrier, final products are lane-local.
    for (int j = lane; j < cols; j += blockDim.x) {
        y[row*cols+j] = x[row*cols+j] * factor;
    }
}

static bool inkling_clip_switch(const char * key) {
    const char * v = getenv(key);
    return v && strcmp(v, "1") == 0;
}

// Require real Inkling weight provenance, not just a popular ggml shape or Kcur name.
static bool inkling_clip_weight(const ggml_tensor * t, const char * suffix) {
    if (!t || t->op != GGML_OP_NONE || strncmp(t->name, "blk.", 4)) { return false; }
    const char * p = t->name + 4;
    if (*p < '0' || *p > '9') { return false; }
    int n = *p++ - '0';
    if (*p >= '0' && *p <= '9') { n = 10*n + *p++ - '0'; }
    return n < 24 && strcmp(p, suffix) == 0;
}

static bool inkling_clip_origin(const ggml_tensor * x, const ggml_tensor * out) {
    if (x->ne[1] == 24) {
        return x->op == GGML_OP_MUL && x->src[0] && x->src[0]->op == GGML_OP_RESHAPE &&
            inkling_clip_weight(x->src[1], x->ne[0] == 128 ? ".attn_q_clip.weight" : ".attn_rel_clip.weight");
    }
    // K is never channel-scaled. Walk only the bounded, specific residual conv topology
    // of inkling2.cpp::convolve: optional per-lag ADDs before the outer residual ADD.
    if (strncmp(out->name, "Kcur-", 5) || x->op != GGML_OP_RESHAPE) { return false; }
    const auto * scale = x->src[0];
#if defined(GGML_INKLING_DEPTHMIX_GFX1103)
    // the one-token conv op stands for the whole SSM_CONV, ADD, SCALE chain
    if (ggml_inkling_is_conv1(scale)) { return inkling_clip_weight(scale->src[2], ".attn_k_conv.weight"); }
#endif
    const auto * add = scale ? scale->src[0] : nullptr;
    if (!scale || scale->op != GGML_OP_SCALE || !add || add->op != GGML_OP_ADD) { return false; }
    const auto * v = add->src[1];
    for (int i = 0; i < 16 && v && v->op == GGML_OP_ADD; ++i) { v = v->src[0]; }
    const auto * conv = v && v->op == GGML_OP_RESHAPE ? v->src[0] : nullptr;
    return conv && conv->op == GGML_OP_SSM_CONV && inkling_clip_weight(conv->src[1], ".attn_k_conv.weight");
}

static bool inkling_clip_dispatch(ggml_backend_cuda_context & ctx, ggml_cgraph * g, int i) {
    static const bool enabled = inkling_clip_switch("GGML_INKLING_ROWCLIP");
    if (!enabled || g->nodes[i]->op != GGML_OP_SQR || i + 7 >= g->n_nodes ||
        !ctx.stream_context().concurrent_events.empty()) { return false; }
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    if (dev.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || dev.warp_size != 32 || dev.nsm <= 0) { return false; }
    auto ** n = g->nodes+i;
    const auto * x = n[0]->src[0];
    if (!x || x->ne[2] != 1 || x->ne[3] != 1 || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
        !((x->ne[0] == 128 && (x->ne[1] == 24 || x->ne[1] == 8)) || (x->ne[0] == 16 && x->ne[1] == 24))) { return false; }
    static const ggml_op ops[] = { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE,
                                   GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL };
    const int output = i+7;
    // Checks compute flags, all intermediate uses (including both consumers of SCALE),
    // watched outputs, and graph-view boundaries. Reject in-place op nodes as well.
    if (!ggml_can_fuse_subgraph(g, i, 8, ops, &output, 1) || !inkling_clip_origin(x,n[7])) { return false; }
    const ggml_tensor * s0[] = {x,n[0],n[1],n[2],n[3],n[4],n[5],x};
    const ggml_tensor * s1[] = {nullptr,nullptr,nullptr,nullptr,nullptr,n[3],nullptr,n[6]};
    for (int j=0;j<8;++j) {
        const auto * t = n[j];
        if (t->src[0] != s0[j] || t->src[1] != s1[j] || t->view_src || t->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(t) || t->ne[0] != ((j==0 || j==7) ? x->ne[0] : 1) || t->ne[1] != x->ne[1] ||
            t->ne[2] != 1 || t->ne[3] != 1 || (t->flags & (GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_PARAM|GGML_TENSOR_FLAG_LOSS))) { return false; }
    }
    // op_params is int32_t[]: bit checks distinguish a -0 lower clamp from +0.
    if (n[3]->op_params[0] != 0x3f800000 || ggml_get_op_params_f32(n[3],1) != 1e-6f ||
        n[6]->op_params[0] != 0 || n[6]->op_params[1] != 0x3f800000) { return false; }
    // No new alloc dependencies: x is already src[0] of the last MUL. Nothing
    // else is read during the span. Refuse a shifted overlap or foreign storage.
    auto buft = ggml_backend_cuda_buffer_type(ctx.device);
    if (!x->data || !n[7]->data || !x->buffer || !n[7]->buffer || x->buffer->buft != buft ||
        n[7]->buffer->buft != buft || (x->data != n[7]->data && !inkling_maskepi_separate(x,n[7]))) { return false; }
    const int rows = x->ne[1], cols = x->ne[0];
    const int threads = rows / dev.nsm < 2 ? 512 : 32; // exact sumrows.cu decision for cols < 1024
    inkling_clip_one<<<rows,threads,0,ctx.stream()>>>((const float *)x->data, (float *)n[7]->data,
                                                       cols, ggml_get_op_params_f32(n[4],0));
    CUDA_CHECK(cudaGetLastError());
    static const bool trace = inkling_clip_switch("GGML_INKLING_ROWCLIP_TRACE");
    if (trace) GGML_LOG_INFO("INKLING_ROWCLIP_HIT cols=%d rows=%d threads=%d\n",cols,rows,threads);
    return true;
}

// Subtract the sink at the gathered distance, not in the full E x H x T table.
// One wave covers 32 consecutive output keys of a single head. The actual
// distance is loaded for *every* key: repeats, wraps and the padded row are legal.
static __global__ void inkling_rawmask_24(const float * proj, const float * sinks,
        const int32_t * ids, const half * mask, half * out, int K, int E, int Q) {
    int k = blockIdx.x*32 + (threadIdx.x & 31), h = blockIdx.z*4 + threadIdx.x/32;
    if (k >= K) { return; }
    int t = blockIdx.y;
    int64_t key = int64_t(t)*K+k, ht = int64_t(t)*24+h;
    uint32_t row = ids[key];
    float shifted = 0.0f;
    if (row < uint32_t(E)) {
        // gfx1103 binbcast V_DUAL_SUB_F32: projection is src0, sink is src1.
        asm("v_sub_f32_e32 %0, %1, %2" : "=v"(shifted) : "v"(proj[ht*E+row]), "v"(sinks[ht]));
    }
    float sum, m = ggml_cuda_cast<float>(mask[key]);
    // ctrl.o MASKEPI: LDS bias first, widened half mask second. No commuting
    // two NaNs and no reassociation across the former F32 store/load boundary.
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(sum) : "v"(shifted), "v"(m));
    out[((int64_t(t/Q)*24+h)*Q+t%Q)*K+k] = ggml_cuda_cast<half>(sum);
}

static bool inkling_rawmask_layout(ggml_cgraph * g, int i) {
    if (!inkling_rawmask_enabled() || i+9 >= g->n_nodes) { return false; }
    auto ** n = g->nodes+i;
    if (n[0]->op != GGML_OP_SUB || n[1]->op != GGML_OP_PERMUTE || n[2]->op != GGML_OP_CONT ||
        n[1]->src[0] != n[0] || n[2]->src[0] != n[1] || n[3]->src[0] != n[2] || !inkling_maskepi_match(g,i+3)) { return false; }
    auto * p = n[0]->src[0], * s = n[0]->src[1];
    if (!p || !s || p->op != GGML_OP_MUL_MAT || s->op != GGML_OP_RESHAPE || !s->src[0] ||
        s->view_src != s->src[0] || s->view_offs || s->src[0]->view_src || p->view_src ||
        n[0]->view_src || n[1]->view_src != n[0] || n[1]->view_offs || n[2]->view_src ||
        n[3]->src[1]->op != GGML_OP_NONE || n[3]->src[1]->view_src ||
        n[7]->src[0]->op != GGML_OP_NONE || n[7]->src[0]->view_src) { return false; }
    const int64_t E = n[2]->ne[1], T = n[2]->ne[2];
    // Bound binbcast to the exact inspected non-unravel gfx1103 specialization.
    if (E != 1025 || T <= 1 || T > 256 || p->ne[0]!=E || p->ne[1]!=24 || p->ne[2]!=T || p->ne[3]!=1 ||
        s->ne[0]!=1 || s->ne[1]!=24 || s->ne[2]!=T || s->ne[3]!=1 ||
        !ggml_are_same_shape(n[0],p) || !ggml_are_same_shape(n[1],n[2])) { return false; }
    for (auto * x : {p,s,s->src[0],n[0],n[2]}) {
        if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) { return false; }
    }
    for (int j=0; j<10; ++j) {
        if (!(n[j]->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            (n[j]->flags & (GGML_TENSOR_FLAG_PARAM | GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_LOSS)) ||
            (j<9 && (n[j]->flags & GGML_TENSOR_FLAG_OUTPUT))) { return false; }
    }
    return n[1]->type == GGML_TYPE_F32 && n[1]->nb[0] == n[0]->nb[1] && n[1]->nb[1] == n[0]->nb[0] &&
           n[1]->nb[2] == n[0]->nb[2] && n[1]->nb[3] == n[0]->nb[3] &&
           ggml_node_has_n_uses(g,i,1) && ggml_node_get_use_count(g,i+1)==1 && ggml_node_has_n_uses(g,i+2,1);
}

// Called with exactly the same predicate before capture/replay and in the eager loop.
static int inkling_rawmask_site(ggml_backend_cuda_context & ctx, ggml_cgraph * g, int i, bool launch) {
    if (!ctx.stream_context().concurrent_events.empty() || !inkling_rawmask_layout(g,i) ||
        ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103) { return 0; }
    auto ** n = g->nodes+i;
    auto * p=n[0]->src[0], * s=n[0]->src[1], * ids=n[3]->src[1], * mask=n[7]->src[0], * out=n[9];
    auto buft = ggml_backend_cuda_buffer_type(ctx.device);
    if (!out->buffer || out->buffer->buft != buft || s->data != s->view_src->data) { return 0; }
    // n[2] check also requires that the old MASKEPI (our numerical reference)
    // actually would have run here. Reject exact AND partial read/write overlap.
    for (auto * x : {p,s->view_src,ids,mask,n[2]}) {
        if (!x->buffer || x->buffer->buft != buft || !inkling_maskepi_separate(x,out)) { return 0; }
    }
    if (launch) {
        inkling_rawmask_24<<<dim3((ids->ne[0]+31)/32,ids->ne[1],6),128,0,ctx.stream()>>>(
                (const float *)p->data,(const float *)s->data,(const int32_t *)ids->data,(const half *)mask->data,
                (half *)out->data,(int)ids->ne[0],(int)p->ne[0],(int)out->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        static const bool trace = inkling_clip_switch("GGML_INKLING_RAWMASK_TRACE");
        if (trace) { GGML_LOG_INFO("INKLING_RAWMASK_HIT %s K=%lld T=%lld\n",out->name,(long long)ids->ne[0],(long long)ids->ne[1]); }
    }
    return 9;
}

#ifdef USE_CUDA_GRAPH
// A captured shorter MASKEPI or GATHER can also elide a freshly tapped result
// while the long fusion remains ineligible. Track their actual launch choices.
static int inkling_mask_replay_choice(ggml_backend_cuda_context & ctx, ggml_cgraph * g, int i) {
    auto * t = g->nodes[i];
    if (t->op == GGML_OP_SUB) { return inkling_rawmask_site(ctx,g,i,false); }
    if (t->op != GGML_OP_GET_ROWS || !ctx.stream_context().concurrent_events.empty()) { return 0; }
    if (inkling_epi_ready(g,i)) { return 6; }
    return inkling_gather_ready(g,i) ? 3 : 0;
}
#endif

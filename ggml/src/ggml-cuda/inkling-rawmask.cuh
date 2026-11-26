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

// Mask of a full-attention layer for the Inkling prefill kernel (inkling-pfattn.cuh), 256 queries. One block takes 64 keys and 64 queries.
// If every key is at the padded distance E-1 and no key is masked, each mask value is the far bias of its (head, query): the block sets the tile flag and writes no mask.
// Else it writes the tile like inkling_rawmask_24 does. The blocks of the first key tile also write the far bias.
static __global__ void inkling_pfmask_24(const float * proj, const float * sinks, const int32_t * ids, const half * mask,
        half * out, int * aux, int K, int E) {
    constexpr int Q = INKLING_PF_AUX_QUERIES, H = INKLING_PF_AUX_HEADS;
    const int tid = threadIdx.x;
    const int t = blockIdx.y*64 + tid/4, k0 = blockIdx.x*64 + (tid%4)*16; // one query and 16 keys per thread
    const int64_t key = int64_t(t)*K + k0;
    __align__(16) int32_t row[16];
    __align__(16) half m[16];
#pragma unroll
    for (int l = 0; l < 16; l += 4) { ggml_cuda_memcpy_1<16>(row + l, ids + key + l); }
#pragma unroll
    for (int l = 0; l < 16; l += 8) { ggml_cuda_memcpy_1<16>(m + l, mask + key + l); }
    int near = 0;
#pragma unroll
    for (int l = 0; l < 16; ++l) { near |= int(row[l] != E - 1) | int(__half_as_ushort(m[l]) != 0); }
    if (blockIdx.x == 0) {
        // The value that inkling_rawmask_24 gives for the row E-1 and the mask +0, 6 heads per thread.
#pragma unroll
        for (int h = (tid%4)*6; h < (tid%4)*6 + 6; ++h) {
            const int64_t ht = int64_t(t)*H + h;
            float shifted, sum, zero = 0.0f;
            asm("v_sub_f32_e32 %0, %1, %2" : "=v"(shifted) : "v"(proj[ht*E + E - 1]), "v"(sinks[ht]));
            asm("v_add_f32_e32 %0, %1, %2" : "=v"(sum) : "v"(shifted), "v"(zero));
            ((float *) aux)[h*Q + t] = ggml_cuda_cast<float>(ggml_cuda_cast<half>(sum));
        }
    }
    const bool dense = __syncthreads_or(near);
    if (tid == 0) { aux[INKLING_PF_AUX_FLAGS + blockIdx.x*(Q/64) + blockIdx.y] = !dense; }
    if (!dense) { return; }
    for (int h = 0; h < H; ++h) {
        const int64_t ht = int64_t(t)*H + h;
        __align__(16) half o[16];
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            float shifted = 0.0f;
            if (uint32_t(row[l]) < uint32_t(E)) {
                asm("v_sub_f32_e32 %0, %1, %2" : "=v"(shifted) : "v"(proj[ht*E + row[l]]), "v"(sinks[ht]));
            }
            float sum, ml = ggml_cuda_cast<float>(m[l]);
            asm("v_add_f32_e32 %0, %1, %2" : "=v"(sum) : "v"(shifted), "v"(ml));
            o[l] = ggml_cuda_cast<half>(sum);
        }
#pragma unroll
        for (int l = 0; l < 16; l += 8) { ggml_cuda_memcpy_1<16>(out + (int64_t(h)*Q + t)*K + k0 + l, o + l); }
    }
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

// True if the mask of the site i can go to its attention op as tile flags plus the tiles that are not far.
// Full-attention layers only: a sliding window layer has no far tile.
// The attention op must be the only reader of the mask (use count 2: a ggml_cast CPY also counts itself).
static bool inkling_pfmask_ready(ggml_backend_cuda_context & ctx, ggml_cgraph * g, int i) {
    if (!ggml_kperf_on(GGML_KPERF_INKLING_PFMASK)) { return false; }
    auto ** n = g->nodes+i;
    const auto * ids=n[3]->src[1], * mask=n[7]->src[0], * out=n[9];
    // The scheduler copies the input, the name of the copy has the backend name in front.
    if (!strstr(ids->name,"attn_inp_rel_pos") || strstr(ids->name,"attn_inp_rel_pos_swa") ||
        ids->ne[1]!=INKLING_PF_AUX_QUERIES || ids->ne[0]%64!=0 ||
        ids->ne[0]<2048 || ids->ne[0]>INKLING_PF_AUX_KV_MAX || mask->ne[0]!=ids->ne[0] || mask->ne[1]!=ids->ne[1] ||
        mask->ne[2]!=1 || mask->ne[3]!=1 || out->ne[0]!=ids->ne[0] || out->ne[1]!=ids->ne[1] || out->ne[2]!=INKLING_PF_AUX_HEADS ||
        (out->flags & GGML_TENSOR_FLAG_OUTPUT) || out->view_src || ggml_node_get_use_count(g,i+9)!=2) { return false; }
    for (int j=i+10; j<g->n_nodes && j<=i+12; ++j) {
        const auto * fa = g->nodes[j];
        if (fa->op == GGML_OP_FLASH_ATTN_EXT) { return fa->src[3]==out && ggml_cuda_flash_attn_ext_inkling_pf(ctx.device,fa); }
    }
    return false;
}

// Called with exactly the same predicate before capture/replay and in the eager loop.
// Returns 0 if the site does not fuse, 10 if the mask goes out with tile flags, else 9.
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
    const bool flags = inkling_pfmask_ready(ctx,g,i);
    if (launch) {
        if (flags) {
            // The attention op of this mask takes the buffer and clears the pointer.
            GGML_ASSERT(ctx.inkling_pf_mask == nullptr);
            if (!ctx.inkling_pf_aux) { CUDA_CHECK(cudaMalloc(&ctx.inkling_pf_aux, INKLING_PF_AUX_SIZE*sizeof(int))); }
            inkling_pfmask_24<<<dim3(ids->ne[0]/64,INKLING_PF_AUX_QUERIES/64),256,0,ctx.stream()>>>(
                    (const float *)p->data,(const float *)s->data,(const int32_t *)ids->data,(const half *)mask->data,
                    (half *)out->data,ctx.inkling_pf_aux,(int)ids->ne[0],(int)p->ne[0]);
            ctx.inkling_pf_mask = out;
        } else {
            inkling_rawmask_24<<<dim3((ids->ne[0]+31)/32,ids->ne[1],6),128,0,ctx.stream()>>>(
                    (const float *)p->data,(const float *)s->data,(const int32_t *)ids->data,(const half *)mask->data,
                    (half *)out->data,(int)ids->ne[0],(int)p->ne[0],(int)out->ne[1]);
        }
        CUDA_CHECK(cudaGetLastError());
        static const bool trace = inkling_clip_switch("GGML_INKLING_RAWMASK_TRACE");
        if (trace) { GGML_LOG_INFO("INKLING_RAWMASK_HIT %s K=%lld T=%lld flags=%d\n",out->name,(long long)ids->ne[0],(long long)ids->ne[1],int(flags)); }
    }
    return flags ? 10 : 9;
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

// Gather the F32 per-head bias, add the broadcast mask in F32, then cast once for FA.
static __global__ void inkling_maskepi_24(const uint32_t * __restrict__ table, const int32_t * __restrict__ ids,
                                          const half * __restrict__ mask, half * __restrict__ dst,
                                          int K, int E, int Q) {
    __shared__ uint32_t tile[32][25];
    __shared__ float mask32[32];
    const int t = blockIdx.y, first = blockIdx.x*32, tid = threadIdx.x;
    if (tid < 32 && first + tid < K) {
        mask32[tid] = ggml_cuda_cast<float>(mask[int64_t(t)*K + first+tid]);
    }
    for (int j=tid; j<32*24; j+=128) {
        const int k=j/24, head=j%24;
        if (first+k < K) {
            const uint32_t row=ids[int64_t(t)*K + first+k];
            tile[k][head]=row<(uint32_t)E ? table[(int64_t(t)*E+row)*24+head] : 0;
        }
    }
    __syncthreads();
    for (int j=tid; j<24*32; j+=128) {
        const int head=j/32,k=j%32;
        if (first+k < K) {
            // In particular, never replace the add by choosing -inf at masked locations.
            const float sum = __uint_as_float(tile[k][head]) + mask32[k];
            dst[((int64_t(t/Q)*24+head)*Q+t%Q)*K+first+k] = ggml_cuda_cast<half>(sum);
        }
    }
}

static bool inkling_maskepi_match(ggml_cgraph * g, int i) {
    static const bool enable=[] { const char * e=getenv("GGML_INKLING_MASKEPI"); return e && strcmp(e,"1")==0; }();
    if (!enable || i+6>=g->n_nodes || !inkling_gather_layout(g,i,1)) { return false; }
    const auto * bias=g->nodes[i+3]; const auto * cast=g->nodes[i+4];
    const auto * add=g->nodes[i+5]; const auto * out=g->nodes[i+6];
    const auto * raw=cast->src[0];
    if (cast->op!=GGML_OP_CPY || add->op!=GGML_OP_ADD || out->op!=GGML_OP_CPY ||
        cast->src[1]!=cast || !raw || add->src[0]!=bias || add->src[1]!=cast || out->src[0]!=add || out->src[1]!=out ||
        cast->view_src || add->view_src || out->view_src ||
        cast->type!=GGML_TYPE_F32 || raw->type!=GGML_TYPE_F16 || add->type!=GGML_TYPE_F32 || out->type!=GGML_TYPE_F16 ||
        strncmp(out->name,"kq_mask_rel_pos-",16)!=0 ||
        !ggml_is_contiguous(raw) || !ggml_is_contiguous(cast) || !ggml_is_contiguous(add) || !ggml_is_contiguous(out) ||
        !ggml_are_same_shape(raw,cast) || !ggml_are_same_shape(bias,add) || !ggml_are_same_shape(bias,out) ||
        raw->ne[0]!=out->ne[0] || raw->ne[1]!=out->ne[1] || raw->ne[2]!=1 || raw->ne[3]!=out->ne[3]) { return false; }
    // A ggml_cast CPY consumes itself as src[1], in addition to the edge to ADD.
    for (auto * n : {cast,add,out}) { if (!(n->flags & GGML_TENSOR_FLAG_COMPUTE)) { return false; } }
    return !((bias->flags | cast->flags | add->flags) & GGML_TENSOR_FLAG_OUTPUT) &&
           ggml_node_get_use_count(g,i+3)==1 && ggml_node_get_use_count(g,i+4)==2 && ggml_node_get_use_count(g,i+5)==1;
}

static bool inkling_maskepi_separate(const ggml_tensor * a,const ggml_tensor * b) {
    const uintptr_t p=reinterpret_cast<uintptr_t>(a->data), q=reinterpret_cast<uintptr_t>(b->data);
    return p && q && (p<q ? q-p>=ggml_nbytes(a) : p-q>=ggml_nbytes(b));
}

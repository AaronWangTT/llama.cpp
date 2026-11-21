// Experimental HIP-only bit gather into final [K, Q, H, S] attention layout.
// A CTA reads a [32 keys, 24 heads] tile by head and writes it by key.
static __global__ void inkling_bias_gather_24(const uint32_t * __restrict__ table, const int32_t * __restrict__ ids,
                                             uint32_t * __restrict__ dst, int K, int E, int Q) {
    __shared__ uint32_t tile[32][25];
    const int t = blockIdx.y, first = blockIdx.x * 32, tid = threadIdx.x;
    for (int j = tid; j < 32*24; j += 128) {
        const int k = j/24, h = j%24;
        if (first+k < K) {
            const uint32_t row = ids[int64_t(t)*K + first+k];
            tile[k][h] = row < (uint32_t) E ? table[(int64_t(t)*E + row)*24 + h] : 0;
        }
    }
    __syncthreads();
    for (int j = tid; j < 24*32; j += 128) {
        const int h = j/32, k = j%32;
        if (first+k < K) {
            dst[((int64_t(t/Q)*24 + h)*Q + t%Q)*K + first+k] = tile[k][h];
        }
    }
}

static bool inkling_gather_layout(ggml_cgraph * g, int i, int min_tokens) {
    if (g->nodes[i]->op != GGML_OP_GET_ROWS || i+3 >= g->n_nodes) { return false; }
    const auto * get = g->nodes[i]; const auto * reshape = g->nodes[i+1]; const auto * perm = g->nodes[i+2];
    const auto * out = g->nodes[i+3]; const auto * tab = get->src[0]; const auto * ids = get->src[1];
    if (!ggml_can_fuse_subgraph(g, i, {GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT}, {i+3}) ||
        reshape->src[0] != get || perm->src[0] != reshape || out->src[0] != perm || reshape->view_offs || perm->view_offs ||
        reshape->view_src != get || perm->view_src != get || out->view_src || !tab || !ids ||
        strstr(ids->name,"attn_inp_rel_pos") == nullptr || strncmp(out->name,"kq_b-",5) != 0 ||
        tab->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || get->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(tab) || !ggml_is_contiguous(ids) || !ggml_is_contiguous(get) ||
        !ggml_is_contiguous(reshape) || !ggml_is_contiguous(out)) { return false; }
    const int64_t H=24, K=ids->ne[0], T=ids->ne[1], E=tab->ne[1], Q=out->ne[1], S=out->ne[3];
    if (K < 1 || K > (1<<25) || T < min_tokens || T > 65535 || E < 1 || E > (1<<23) || Q<1 || S<1 || Q>T || S>T || Q*S!=T ||
        ids->ne[2]!=1 || ids->ne[3]!=1 || tab->ne[0]!=H || tab->ne[2]!=T || tab->ne[3]!=1 ||
        get->ne[0]!=H || get->ne[1]!=K || get->ne[2]!=T || get->ne[3]!=1 ||
        reshape->ne[0]!=H || reshape->ne[1]!=K || reshape->ne[2]!=Q || reshape->ne[3]!=S ||
        out->ne[0]!=K || out->ne[2]!=H || !ggml_are_same_shape(out, perm)) { return false; }
    return perm->nb[0]==reshape->nb[1] && perm->nb[1]==reshape->nb[2] &&
           perm->nb[2]==reshape->nb[0] && perm->nb[3]==reshape->nb[3];
}

static bool inkling_gather_match(ggml_cgraph * g, int i) {
    static const bool enable = [] { const char * e = getenv("GGML_INKLING_GATHER"); return e && strcmp(e, "1") == 0; }();
    return enable && inkling_gather_layout(g, i, 2);
}

// Two independent, three-node tails. Never fuse across a projection, even when
// both weights share a controller. Addressing is one F32 per lane (gfx1103).
template<bool gain>
static __global__ void inkling_control_three(const float * proj, const float * x, const float * base, float * out) {
    const int j = blockIdx.x * 128 + threadIdx.x;
    const float t = tanhf(proj[j]);  // same device library / flags as op_tanh
    float prod, y;
    // Canonical binbcast's gfx1103 v_dual_{mul,sub,add}_f32 all take
    // src[0] before src[1]. Keep the rounding boundary and NaN operand order.
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(prod) : "v"(t), "v"(x[j]));
    if constexpr (gain) {
        asm("v_add_f32_e32 %0, %1, %2" : "=v"(y) : "v"(base[j]), "v"(prod));
    } else {
        asm("v_sub_f32_e32 %0, %1, %2" : "=v"(y) : "v"(x[j]), "v"(prod));
    }
    out[j] = y;
}

static bool inkling_control_shape(const ggml_tensor * t, int w, int rows, bool weight = false) {
    // The tail reads no weight, so a weight may also hold its F32 values as BF16.
    return t && (t->type == GGML_TYPE_F32 || (weight && t->type == GGML_TYPE_BF16)) && t->ne[0] == w && t->ne[1] == rows &&
           t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t);
}

// Match blk.6..17.loop_{q,k,v,ffn}_ctrl_, exactly, on the down weight.
static int inkling_control_prefix(const char * name) {
    if (strncmp(name, "blk.", 4)) { return 0; }
    int k = 4;
    if (name[k] == '1' && name[k+1] >= '0' && name[k+1] <= '7') { k += 2; }
    else if (name[k] >= '6' && name[k] <= '9') { ++k; }
    else { return 0; }
    if (strncmp(name+k, ".loop_", 6)) { return 0; }
    k += 6;
    if (name[k]=='q' || name[k]=='k' || name[k]=='v') { ++k; }
    else if (!strncmp(name+k,"ffn",3)) { k += 3; }
    else { return 0; }
    return strcmp(name+k, "_ctrl_down.weight") == 0 ? k+6 : 0;
}

// 0 = fallback, 1 = forget, 2 = gain (stored as such in the replay guard).
static int inkling_control_site(ggml_backend_cuda_context & ctx, ggml_cgraph * g, int i, bool launch) {
    if (i+2 >= g->n_nodes || !ctx.stream_context().concurrent_events.empty()) { return 0; }
    auto * t = g->nodes[i]; auto * m = g->nodes[i+1]; auto * out = g->nodes[i+2];
    bool gain = out->op == GGML_OP_ADD;
    if (t->op != GGML_OP_UNARY || ggml_get_unary_op(t) != GGML_UNARY_OP_TANH ||
        (!gain && out->op != GGML_OP_SUB) || m->op != GGML_OP_MUL || m->src[0] != t || out->src[1] != m ||
        !ggml_node_has_n_uses(g,i,1) || !ggml_node_has_n_uses(g,i+1,1)) { return 0; }
    if (t->ne[1] != 1 && t->ne[1] != 256) { return 0; }
    const int rows = (int)t->ne[1];
    auto * p = t->src[0]; auto * x = m->src[1]; auto * base = out->src[0];
    for (auto * v : {t,m,out,p,x,base}) {
        if (!inkling_control_shape(v,2304,rows) || !v->data || !v->buffer ||
            v->buffer->buft != ggml_backend_cuda_buffer_type(ctx.device)) { return 0; }
    }
    for (auto * v : {t,m,out}) {
        if (v->view_src || !(v->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            (v->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_PARAM | GGML_TENSOR_FLAG_LOSS))) { return 0; }
    }
    if (p->op != GGML_OP_MUL_MAT || !inkling_control_shape(p->src[0],64,2304,true) ||
        !inkling_control_shape(p->src[1],64,rows)) { return 0; }
    auto * down = p->src[1];
    if (down->op != GGML_OP_MUL_MAT || !inkling_control_shape(down->src[0],4608,64,true) ||
        !inkling_control_shape(down->src[1],4608,rows)) { return 0; }
    const char * dw = down->src[0]->name;
    const int k = inkling_control_prefix(dw);
    auto named = [&](const ggml_tensor * w, const char * suffix) {
        return w && k && !strncmp(w->name,dw,k) && !strcmp(w->name+k,suffix);
    };
    if (!named(p->src[0],gain ? "gain.weight" : "forget.weight")) { return 0; }
    auto * cat = down->src[1];
    if (cat->op != GGML_OP_CONCAT || cat->op_params[0] != 0 || !cat->src[0] || !cat->src[1] ||
        cat->src[0]->op != GGML_OP_RMS_NORM || cat->src[1]->op != GGML_OP_RMS_NORM) { return 0; }
    auto * cur = cat->src[0]->src[0]; auto * msg = cat->src[1]->src[0];
    if (!gain) {
        if (base != cur || x != cur) { return 0; }
    } else {
        if (x != msg || base->op != GGML_OP_SUB || base->src[0] != cur) { return 0; }
        auto * fm = base->src[1];
        if (!fm || fm->op != GGML_OP_MUL || fm->src[1] != cur || !fm->src[0] ||
            fm->src[0]->op != GGML_OP_UNARY || ggml_get_unary_op(fm->src[0]) != GGML_UNARY_OP_TANH) { return 0; }
        auto * fp = fm->src[0]->src[0];
        if (!fp || fp->op != GGML_OP_MUL_MAT || fp->src[1] != down || !named(fp->src[0],"forget.weight")) { return 0; }
    }
    // All three reads are complete predecessors outside the consumed span.
    // An allocator may recycle an input at exactly the same address: safe
    // lane-for-lane, but a partial overlap allows another lane to clobber it.
    for (auto * v : {p,x,base}) {
        if (v->data != out->data && !inkling_maskepi_separate(v,out)) { return 0; }
    }
    if (ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103) { return 0; }
    if (launch) {
        auto * pv = (const float *)p->data; auto * xv = (const float *)x->data;
        auto * bv = (const float *)base->data; auto * y = (float *)out->data;
        if (gain) { inkling_control_three<true><<<18*rows,128,0,ctx.stream()>>>(pv,xv,bv,y); }
        else { inkling_control_three<false><<<18*rows,128,0,ctx.stream()>>>(pv,xv,bv,y); }
        CUDA_CHECK(cudaGetLastError());
        if (ggml_kperf_trace(GGML_KPERF_INKLING_CTRL_TAIL)) { GGML_LOG_INFO("INKLING_CTRL_TAIL_HIT %s T=%d\n",p->src[0]->name,rows); }
    }
    return gain ? 2 : 1;
}

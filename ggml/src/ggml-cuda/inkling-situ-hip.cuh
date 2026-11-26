// Exact 1152x8 expert SITU on gfx1103. Keep each former kernel's F32 result
// opaque to the next one (register only, no spill or global memory round trip).
static __device__ __forceinline__ float inkling_situ_f32(float x) {
    asm("" : "+v"(x));
    return x;
}

static __global__ void inkling_situ_one(const float * packed, float * dst,
        float gi, float go, float ui, float uo, float bias) {
    // 9 blocks of 128 lanes per expert row; no padding or partial rows.
    const int i = blockIdx.x * 128 + threadIdx.x;
    const int j = i + (i / 1152) * 1152;
    const float g = packed[j], u = packed[j + 1152];
    // SCALE is scale*x + bias, not x/beta or a bare multiplication. Pass
    // the actual rounded graph parameters (including +0) as kernel arguments.
    float a = inkling_situ_f32(gi * g + bias);
    a = inkling_situ_f32(tanhf(a));
    a = inkling_situ_f32(go * a + bias);
    // Baseline unary_gated_op_kernel applies sigmoid to its *first* input.
    const float sig = 1.0f / (1.0f + expf(-g));
    // gfx1103 codegen of the existing unary_gated kernel loads g[j1] as
    // source 0 of V_MUL_F32, sigmoid as source 1. Do not commute two NaNs.
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(a) : "v"(a), "v"(sig));
    float b = inkling_situ_f32(ui * u + bias);
    b = inkling_situ_f32(tanhf(b));
    b = inkling_situ_f32(uo * b + bias);
    // binbcast's gfx1103 V_DUAL_MUL_F32 takes src0 (act) then src1 (up).
    float y;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(y) : "v"(a), "v"(b));
    dst[i] = y;
}

static int inkling_situ_dispatch(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool launch) {
    static const bool enabled = ggml_kperf_on(GGML_KPERF_INKLING_SITU);
    if (!enabled || i + 8 >= graph->n_nodes || (launch && !ctx.stream_context().concurrent_events.empty())) { return 0; }
    auto ** n = graph->nodes + i;
    auto * gate = n[0]->src[0];
    if (!gate || gate->op != GGML_OP_VIEW || gate->ne[0] != 1152 || gate->ne[1] != 8 || gate->ne[3] != 1 ||
        !(gate->ne[2] == 1 || gate->ne[2] == 128 || gate->ne[2] == 256)) { return 0; }
    // Depth-first graph construction can place the up VIEW between the two
    // branches. Never step over an unrelated (even metadata-only) node.
    const bool has_view = n[5]->op == GGML_OP_VIEW;
    const int k = 5 + has_view, last = k + 3;
    if (i + last >= graph->n_nodes || n[last]->op != GGML_OP_MUL) { return 0; }
    auto * out = n[last];
    auto * up = n[k]->src[0];
    auto * pack = gate->view_src;
    if (!up || up->op != GGML_OP_VIEW || !pack || pack->view_src || pack->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(pack) || pack->ne[0] != 2304 || pack->ne[1] != 8 || pack->ne[2] != gate->ne[2] || pack->ne[3] != 1 ||
        up->view_src != pack || up == gate || (has_view && (n[5] != up || ggml_node_get_use_count(graph,i+5) != 1)) ||
        gate->view_offs != 0 || up->view_offs != 1152*sizeof(float) || !ggml_are_same_shape(gate,up)) { return 0; }
    for (auto * t : {gate,up}) {
        if (t->src[0] != pack || t->type != GGML_TYPE_F32 || memcmp(t->nb,pack->nb,sizeof(t->nb)) ||
            (t->flags & (GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_PARAM | GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_LOSS))) { return 0; }
    }
    static constexpr char prefix[] = "ffn_moe_situ-";
    if (strncmp(out->name,prefix,sizeof(prefix)-1) != 0) { return 0; }
    const char * layer = out->name + sizeof(prefix)-1;
    if (!(*layer >= '0' && *layer <= '9') || (layer[1] &&
        (layer[0]=='0' || layer[1]<'0' || layer[1]>'9' || layer[2] || atoi(layer)>=48))) { return 0; }
    if (strncmp(gate->name,"ffn_moe_gate-",13) || strcmp(gate->name+13,layer) ||
        strncmp(up->name,"ffn_moe_up-",11) || strcmp(up->name+11,layer)) { return 0; }
    static const ggml_op ops[] = {GGML_OP_SCALE,GGML_OP_UNARY,GGML_OP_SCALE,GGML_OP_UNARY,GGML_OP_MUL,
                                  GGML_OP_SCALE,GGML_OP_UNARY,GGML_OP_SCALE,GGML_OP_MUL};
    const ggml_tensor * s0[] = {gate,n[0],n[1],gate,n[2],up,n[k],n[k+1],n[4]};
    const ggml_tensor * s1[] = {nullptr,nullptr,nullptr,nullptr,n[3],nullptr,nullptr,nullptr,n[k+2]};
    for (int j=0; j<9; ++j) {
        const int at = j + (has_view && j>=5);
        const auto * t = n[at];
        if (t->op != ops[j] || t->src[0] != s0[j] || t->src[1] != s1[j] || t->view_src ||
            t->type != GGML_TYPE_F32 || !ggml_are_same_shape(t,gate) || !ggml_is_contiguous(t) ||
            !(t->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            (t->flags & (GGML_TENSOR_FLAG_PARAM|GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_LOSS)) ||
            (j != 8 && !ggml_node_has_n_uses(graph,i+at,1))) { return 0; }
    }
    if (ggml_get_unary_op(n[1]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(n[3]) != GGML_UNARY_OP_SIGMOID ||
        ggml_get_unary_op(n[k+1]) != GGML_UNARY_OP_TANH) { return 0; }
    float scales[4];
    const int at[] = {0,2,k,k+2};
    static constexpr float expected[] = {1.0f/4,4,1.0f/25,25};
    for (int j=0;j<4;++j) {
        scales[j] = ggml_get_op_params_f32(n[at[j]],0);
        if (scales[j] != expected[j] || n[at[j]]->op_params[1] != 0) { return 0; }
    }
    // At the FIRST SCALE, the packed producer must be live (gate reads it).
    // Both inputs are slices of that same completed allocation. No intervening
    // non-SITU work or outside use of an omitted result. The only early write
    // is to the LAST MUL's storage: reject *any* overlap with the whole pack.
    const auto buft = ggml_backend_cuda_buffer_type(ctx.device);
    if (!pack->buffer || !out->buffer || pack->buffer->buft != buft || out->buffer->buft != buft ||
        !inkling_maskepi_separate(pack,out) || gate->data != pack->data ||
        up->data != (const char *)pack->data + up->view_offs ||
        ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103) { return 0; }
    if (!launch) { return last; }
    inkling_situ_one<<<9*8*(int)gate->ne[2],128,0,ctx.stream()>>>((const float *)pack->data,(float *)out->data,
        scales[0],scales[1],scales[2],scales[3],ggml_get_op_params_f32(n[0],1));
    CUDA_CHECK(cudaGetLastError());
    static const bool trace = inkling_clip_switch("GGML_INKLING_SITU_TRACE");
    if (trace) { GGML_LOG_INFO("INKLING_SITU_HIT %s T=%d\n",out->name,(int)gate->ne[2]); }
    return last;
}

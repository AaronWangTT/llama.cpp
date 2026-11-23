#include "models.h"
#include "llama-memory-recurrent.h"

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
#include "ggml-qwen35-convpf.h"
#include "ggml-qwen35-decode.h"

#include <cstring>

// the fused ops exist on a single ROCm device only; returns that device or nullptr
// not with tensor overrides: an op next to a weight in host memory can go to the CPU backend, the new state is then lost
static ggml_backend_dev_t qwen35_rocm_dev(const llama_cparams & cparams, const llama_model & model, const llama_adapter_loras * loras) {
    if (cparams.cb_eval || cparams.optimizer_graph || !loras->empty() || model.devices.size() != 1 || model.devices.front().is_meta ||
            model.has_tensor_overrides() ||
            (model.split_mode() != LLAMA_SPLIT_MODE_NONE && model.split_mode() != LLAMA_SPLIT_MODE_LAYER)) {
        return nullptr;
    }
    const auto dev = model.devices.front().dev;
    return std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "ROCm") == 0 ? dev : nullptr;
}
#endif

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // optional projection of the embeddings output
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), { n_embd, hparams.n_embd_out() }, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   { hparams.n_embd_out() },         TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            const int separate_flags = flags & TENSOR_SKIP ? flags | TENSOR_NOT_REQUIRED : flags;
            layer.wqkv_z = create_tensor(tn(LLM_TENSOR_ATTN_QKV_GATE, "weight", il), { n_embd, key_dim * 2 + value_dim * 2 }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.wqkv_z) {
                layer.wqkv      = create_tensor(tn(LLM_TENSOR_ATTN_QKV,  "weight", il), { n_embd, key_dim * 2 + value_dim }, separate_flags);
                layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, value_dim }, separate_flags);
            }
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta_alpha_cat = create_tensor(tn(LLM_TENSOR_SSM_BETA_ALPHA_CAT, "weight", il), { n_embd, n_v_heads * 2 }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.ssm_beta_alpha_cat) {
                layer.ssm_beta  = create_tensor(tn(LLM_TENSOR_SSM_BETA,  "weight", il), { n_embd, n_v_heads }, separate_flags);
                layer.ssm_alpha = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "weight", il), { n_embd, n_v_heads }, separate_flags);
            }
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

llama_model_qwen35::graph::graph(const llm_graph_params & params) :
    llm_build_delta_net_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.cls_out) {
        ggml_tensor * embd = build_lora_mm(model.cls_out, cur);
        if (model.cls_out_b) {
            embd = ggml_add(ctx0, embd, model.cls_out_b);
        }
        cb(embd, "result_embd_proj", -1);
        res->t_embd = embd;
        ggml_build_forward_expand(gf, embd);
    }

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv_z) {
        ggml_tensor * qkvz = build_lora_mm(model.layers[il].wqkv_z, input);
        cb(qkvz, "linear_attn_qkvz", il);

        const int64_t n_z   = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t n_qkv = qkvz->ne[0] - n_z;

        ggml_tensor * qkv_mixed = ggml_view_3d(ctx0, qkvz, n_qkv, n_seq_tokens, n_seqs, qkvz->nb[1], qkvz->nb[1] * n_seq_tokens, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, n_z, n_seq_tokens * n_seqs, qkvz->nb[1], ggml_row_size(qkvz->type, n_qkv));
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv_mixed = nullptr;
    ggml_tensor * z         = nullptr;
    ggml_tensor * beta      = nullptr;
    ggml_tensor * alpha     = nullptr;

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    const float eps_norm = hparams.f_norm_rms_eps;

    // one token of one sequence on a single ROCm device takes the fused ops of ggml-qwen35-decode.h
    ggml_tensor * prep   = nullptr; // [q, k, v, beta, gate]
    ggml_tensor * output = nullptr;
#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    const auto dev_one = n_seq_tokens == 1 && n_seqs == 1 && cparams.n_rs_seq == 0 && mctx_cur->get_n_rs() == 1 ?
        qwen35_rocm_dev(cparams, model, loras) : nullptr;
    // a layer that is not on that device keeps the stock ops
    const auto dev = dev_one && model.dev_layer(il) == dev_one ? dev_one : nullptr;
    const size_t fs = sizeof(float);

    // the zero state of build_rs(); the fused ops read the state row of the sequence from the cache
    auto clear_zero_state = [&](ggml_tensor * states) {
        const int32_t rs_zero = mctx_cur->get_rs_z();
        ggml_tensor * state_zero = ggml_view_1d(ctx0, states, states->ne[0]*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
        ggml_build_forward_expand(gf, ggml_fill_inplace(ctx0, state_zero, 0.0f));
    };
    auto cell_of = [&](ggml_context * ctx, ggml_tensor * states) {
        return ggml_view_2d(ctx, states, states->ne[0], n_seqs, states->nb[1], mctx_cur->get_head()*states->nb[1]);
    };

    // the four products of cur share one quantized input
    if (dev && ggml_qwen35_mvshare_enabled() && !model.layers[il].wqkv_z && !model.layers[il].ssm_beta_alpha_cat) {
        const auto & layer = model.layers[il];
        std::vector<ggml_tensor *> w = { layer.wqkv, layer.wqkv_gate, layer.ssm_beta, layer.ssm_alpha };
        bool plain = !layer.wqkv_s && !layer.wqkv_gate_s && !layer.ssm_beta_s && !layer.ssm_alpha_s;
        for (const ggml_tensor * m : w) {
            plain = plain && m && (!hdmd || !hdmd->rot.count(m)) && (!prec_policy || !prec_policy->prec_src1.count(m));
        }
        ggml_context_ptr probe{ggml_init({2*ggml_tensor_overhead(), nullptr, true})};
        auto * candidate = plain ? ggml_qwen35_try_mvshare(probe.get(), w.data(), w.size(), cur) : nullptr;
        if (candidate && ggml_backend_dev_supports_op(dev, candidate)) {
            ggml_tensor * shared = ggml_qwen35_try_mvshare(ctx0, w.data(), w.size(), cur);
            cb(shared, "linear_attn_shared", il);

            const int64_t n_qkv = layer.wqkv->ne[1];
            const int64_t n_z   = layer.wqkv_gate->ne[1];

            qkv_mixed = ggml_view_3d(ctx0, shared, n_qkv, n_seq_tokens, n_seqs, n_qkv*fs, n_qkv*fs, 0);
            z         = ggml_view_2d(ctx0, shared, n_z, n_seq_tokens * n_seqs, n_z*fs, n_qkv*fs);
            beta      = ggml_view_4d(ctx0, shared, 1, num_v_heads, n_seq_tokens, n_seqs, fs, num_v_heads*fs, num_v_heads*fs, (n_qkv + n_z)*fs);
            alpha     = ggml_view_3d(ctx0, shared, num_v_heads, n_seq_tokens, n_seqs, num_v_heads*fs, num_v_heads*fs, (n_qkv + n_z + num_v_heads)*fs);
        }
    }
#endif

    if (!qkv_mixed) {
        // Input projections
        auto qkvz = build_qkvz(cur, il);
        qkv_mixed = qkvz.first;
        z         = qkvz.second;

        if (model.layers[il].ssm_beta_alpha_cat) {
            ggml_tensor * ba = build_lora_mm(model.layers[il].ssm_beta_alpha_cat, cur);
            cb(ba, "beta_alpha", il);

            beta  = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                    ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, 0);
            alpha = ggml_view_3d(ctx0, ba, num_v_heads, n_seq_tokens, n_seqs,
                    ba->nb[1], ba->nb[1] * n_seq_tokens, ggml_row_size(ba->type, num_v_heads));
        } else {
            beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
            beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);

            alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
            alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
        }
    }
    cb(beta, "beta", il);
    cb(alpha, "alpha", il);

    ggml_tensor * gate   = nullptr;
    ggml_tensor * q_conv = nullptr;
    ggml_tensor * k_conv = nullptr;
    ggml_tensor * v_conv = nullptr;

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    // conv, SiLU, L2 norm, beta and gate in one op, which also updates the conv state
    if (dev && ggml_qwen35_prep1_enabled() && head_k_dim == ggml_qwen35_head_dim && head_v_dim == head_k_dim) {
        const int64_t key_dim = head_k_dim * num_k_heads;
        auto make = [&](ggml_context * ctx) {
            return ggml_qwen35_try_prep1(ctx, qkv_mixed, beta, alpha, conv_kernel, model.layers[il].ssm_dt, model.layers[il].ssm_a,
                    conv_states_all, inp->s_copy_main, cell_of(ctx, conv_states_all),
                    2*key_dim, eps_norm/(float) head_k_dim, 1.0f/sqrtf((float) head_k_dim));
        };
        ggml_context_ptr probe{ggml_init({3*ggml_tensor_overhead(), nullptr, true})};
        auto * candidate = conv_channels == qkv_mixed->ne[0] ? make(probe.get()) : nullptr;
        if (candidate && ggml_backend_dev_supports_op(dev, candidate)) {
            clear_zero_state(conv_states_all);
            prep = make(ctx0);
            cb(prep, "linear_attn_prep", il);

            q_conv = ggml_view_4d(ctx0, prep, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, head_k_dim*fs, key_dim*fs, key_dim*fs, 0);
            k_conv = ggml_view_4d(ctx0, prep, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, head_k_dim*fs, key_dim*fs, key_dim*fs, key_dim*fs);
            v_conv = ggml_view_4d(ctx0, prep, head_v_dim, num_v_heads, n_seq_tokens, n_seqs, head_v_dim*fs, d_inner*fs, d_inner*fs, 2*key_dim*fs);
            beta   = ggml_view_4d(ctx0, prep, 1, num_v_heads, n_seq_tokens, n_seqs, fs, num_v_heads*fs, num_v_heads*fs, conv_channels*fs);
            gate   = ggml_view_4d(ctx0, prep, 1, num_v_heads, n_seq_tokens, n_seqs, fs, num_v_heads*fs, num_v_heads*fs, (conv_channels + num_v_heads)*fs);
        }
    }
#endif

    if (!prep) {
        beta = ggml_sigmoid(ctx0, beta);
        cb(beta, "beta_sigmoid", il);

        ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
        cb(alpha_softplus, "a_softplus", il);

        gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
        cb(gate, "gate", il);

        gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * conv_output_silu = nullptr;

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    // many tokens of one sequence: one op reads the conv state and qkv_mixed, there is no concat tensor
    if (!prep && ggml_qwen35_convpf_enabled() && n_seqs == 1 && n_seq_tokens >= ggml_qwen35_convpf_min_n_tokens && cparams.n_rs_seq == 0 &&
            conv_kernel_size == ggml_qwen35_convpf_n_conv && qkv_mixed->ne[0] == conv_channels) {
        // a layer that is not on that device keeps the stock ops
        if (const auto dev_pf = qwen35_rocm_dev(cparams, model, loras); dev_pf && model.dev_layer(il) == dev_pf) {
            ggml_context_ptr probe{ggml_init({3*ggml_tensor_overhead(), nullptr, true})};
            ggml_tensor * past = ggml_new_tensor_2d(probe.get(), GGML_TYPE_F32, conv_kernel_size - 1, conv_channels);
            auto * candidate = ggml_qwen35_try_convpf(probe.get(), qkv_mixed, past, conv_kernel);
            if (candidate && ggml_backend_dev_supports_op(dev_pf, candidate)) {
                ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
                cb(conv_states, "conv_states", il);

                conv_states = ggml_reshape_2d(ctx0, conv_states, conv_kernel_size - 1, conv_channels);

                conv_output_silu = ggml_qwen35_try_convpf(ctx0, qkv_mixed, conv_states, conv_kernel);
                GGML_ASSERT(conv_output_silu != nullptr);
                // the op reads the old state, so it goes into the graph before the store of the new state
                ggml_build_forward_expand(gf, conv_output_silu);

                // new state: the last tokens of qkv_mixed, the same values as the tail of the concat tensor
                ggml_tensor * last = ggml_view_3d(ctx0, qkv_mixed, conv_channels, conv_kernel_size - 1, n_seqs,
                        qkv_mixed->nb[1], qkv_mixed->nb[2], (n_seq_tokens - (conv_kernel_size - 1))*qkv_mixed->nb[1]);
                ggml_tensor * cell = ggml_view_2d(ctx0, conv_states_all, (conv_kernel_size - 1)*conv_channels, n_seqs,
                        conv_states_all->nb[1], mctx_cur->get_head()*conv_states_all->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, last), cell));
            }
        }
    }
#endif

    ggml_tensor * conv_input = prep || conv_output_silu ? nullptr : build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    // with the fused delta net the state stays in the cache: no copy of it here
    const bool gdn_one = dev && ggml_qwen35_gdn1_enabled() && head_k_dim == ggml_qwen35_head_dim && head_v_dim == head_k_dim;
#else
    const bool gdn_one = false;
#endif

    ggml_tensor * state = nullptr;
    if (!gdn_one) {
        state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_predelta", il);
    }

    if (!prep) {
        if (!conv_output_silu) {
            ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
            cb(conv_output_proper, "conv_output_raw", il);

            conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        }
        cb(conv_output_silu, "conv_output_silu", il);

        ggml_tensor * conv_qkv_mix = conv_output_silu;

        // Calculate the total conv dimension
        int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
        int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

        // Extract the convolved Q, K, V from conv_output
        q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                ggml_row_size(conv_qkv_mix->type, head_k_dim),
                nb1_qkv,
                nb1_qkv * n_seq_tokens,
                0);

        k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                ggml_row_size(conv_qkv_mix->type, head_k_dim),
                nb1_qkv,
                nb1_qkv * n_seq_tokens,
                head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

        v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
                ggml_row_size(conv_qkv_mix->type, head_v_dim),
                nb1_qkv,
                nb1_qkv * n_seq_tokens,
                ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

        cb(q_conv, "q_conv", il);
        cb(k_conv, "k_conv", il);
        cb(v_conv, "v_conv", il);

        q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    if (gdn_one) {
        auto make = [&](ggml_context * ctx) {
            return ggml_qwen35_try_gdn1(ctx, q_conv, k_conv, v_conv, gate, beta, ssm_states_all, inp->s_copy_main, cell_of(ctx, ssm_states_all));
        };
        ggml_context_ptr probe{ggml_init({3*ggml_tensor_overhead(), nullptr, true})};
        auto * candidate = make(probe.get());
        if (candidate && ggml_backend_dev_supports_op(dev, candidate)) {
            clear_zero_state(ssm_states_all);
            output = make(ctx0);
            cb(output, "attn_output", il);
        }
    }
#endif

    if (!output) {
        if (!state) {
            state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
            cb(state, "state_predelta", il);
        }
        output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
    }

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = nullptr;
#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    if (dev && ggml_qwen35_post1_enabled() && head_v_dim == ggml_qwen35_head_dim) {
        ggml_context_ptr probe{ggml_init({2*ggml_tensor_overhead(), nullptr, true})};
        auto * candidate = ggml_qwen35_try_post1(probe.get(), output, model.layers[il].ssm_norm, z_2d, eps_norm);
        if (candidate && ggml_backend_dev_supports_op(dev, candidate)) {
            attn_out_norm = ggml_qwen35_try_post1(ctx0, output, model.layers[il].ssm_norm, z_2d, eps_norm);
        }
    }
#endif
    if (!attn_out_norm) {
        attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);
    }

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);

        // a Hadamard-latent table stores rotated rows; restore the primal basis
        if (hdmd) {
            if (const auto it = hdmd->inv.find(tok_embd_w); it != hdmd->inv.end()) {
                tok_embd = llama_mul_mat_hadamard(ctx0, tok_embd, it->second.rot);
                if (it->second.signs) {
                    tok_embd = ggml_mul(ctx0, tok_embd, it->second.signs);
                }
            }
        }
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd);

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}

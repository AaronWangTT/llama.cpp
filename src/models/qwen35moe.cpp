#include "models.h"
#include "llama-memory-recurrent.h"

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
#include "ggml-q35moe-ffn.h"
#include "ggml-qwen35-convpf.h"
#include "ggml-qwen35-decode.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

// the fused ops exist on a single ROCm device only; returns that device if layer il is on it, else nullptr and the reason in why
// not with tensor overrides: an op next to a weight in host memory can go to the CPU backend, the new state is then lost
// rs: the op reads or writes the recurrent state, it does not keep the copies for a rollback
static ggml_backend_dev_t q35moe_rocm_dev(const llama_cparams & cparams, const llama_model & model, const llama_adapter_loras * loras,
        const int il, const bool rs, const char ** why) {
    if (cparams.cb_eval) {
        *why = "an eval callback is set";
    } else if (cparams.optimizer_graph) {
        *why = "the graph is for training";
    } else if (!loras->empty()) {
        *why = "a LoRA adapter is in use";
    } else if (model.devices.size() != 1 || model.devices.front().is_meta) {
        *why = "the model is not on one plain device";
    } else if (model.has_tensor_overrides()) {
        *why = "tensor overrides are set (-ot, --cpu-moe)";
    } else if (model.split_mode() != LLAMA_SPLIT_MODE_NONE && model.split_mode() != LLAMA_SPLIT_MODE_LAYER) {
        *why = "the split mode is not none or layer";
    } else if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(model.devices.front().dev)), "ROCm") != 0) {
        *why = "the device is not a ROCm device";
    } else if (rs && cparams.n_rs_seq != 0) {
        *why = "n_rs_seq > 0 (state rollback for speculative decoding)";
    } else if (model.dev_layer(il) != model.devices.front().dev) {
        *why = "the block is not on the device";
    } else {
        return model.devices.front().dev;
    }
    return nullptr;
}

// a switch that is set but can not act in this context goes to the log, once per switch and reason (both are string literals)
static void q35moe_log_off(const char * switches, const char * why) {
    static std::mutex mutex;
    static std::vector<std::pair<const char *, const char *>> seen;
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & s : seen) {
        if (s.first == switches && s.second == why) {
            return;
        }
    }
    seen.emplace_back(switches, why);
    LLAMA_LOG_WARN("qwen35moe: no effect of %s: %s\n", switches, why);
}
#endif

void llama_model_qwen35moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
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
        case 40: type = LLM_TYPE_35B_A3B; break;
        case 48: type = LLM_TYPE_122B_A10B; break;
        case 60: type = LLM_TYPE_397B_A17B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

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
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // MTP block looks like a full-attention Qwen3.5 decoder block with MoE FFN.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, mtp_flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, mtp_flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, mtp_flags);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, mtp_flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, mtp_flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, mtp_flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, mtp_flags);

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

std::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

llama_model_qwen35moe::graph::graph(const llm_graph_params & params) :
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
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
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

        // MOE FFN layer; one token on a ROCm device can take the feed-forward and its residual as one op
        cur = build_layer_ffn_one(attn_post_norm, ffn_residual, il);
        if (cur == nullptr) {
            cur = build_layer_ffn(attn_post_norm, il);
            cb(cur, "ffn_out", il);

            // Residual connection for FFN - add to the tensor from before post_attention_layernorm
            cur = ggml_add(ctx0, cur, ffn_residual);
        }
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // post-norm hidden state feeds both the LM head and the MTP seed below
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35moe::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35moe::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn(
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

    // Apply IMRoPE
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

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn_linear(
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

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    if (ggml_tensor * one = build_layer_attn_linear_one(inp, cur, il)) {
        return one;
    }
    if (ggml_tensor * few = build_layer_attn_linear_few(inp, cur, il)) {
        return few;
    }
#endif

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_output_silu = nullptr;

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    // GGML_Q35MOE_CONVPF=1, many tokens of one sequence: one op reads the conv state and qkv_mixed, there is no concat tensor
    if (ggml_q35moe_convpf_enabled()) {
        // a reason that holds for the whole context goes to the log once; the shape of the batch does not
        // GGML_Q35MOE_CONVPFRS=1: also with rollback snapshots, the store below then writes the windows of all slots
        const char * why = nullptr;
        const auto dev_pf = q35moe_rocm_dev(cparams, model, loras, il, !ggml_q35moe_convpfrs_enabled(), &why);
        if (dev_pf && (conv_kernel_size != ggml_qwen35_convpf_n_conv || qkv_mixed->ne[0] != conv_channels)) {
            why = "the conv kernel size or the channel count";
        }
        // slots that build_conv_state_store() writes; all their windows must lie in qkv_mixed
        const int64_t n_slots = std::min<int64_t>(n_seq_tokens, (int64_t) cparams.n_rs_seq + 1);
        if (why) {
            q35moe_log_off("GGML_Q35MOE_CONVPF", why);
        } else if (n_seqs == 1 && n_seq_tokens >= ggml_qwen35_convpf_min_n_tokens && n_seq_tokens >= n_slots + conv_kernel_size - 1) {
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
                // slot s is the state of s tokens before the end, its row is s*size + head as in build_conv_state_store()
                for (int64_t slot = 0; slot < n_slots; ++slot) {
                    ggml_tensor * last = ggml_view_3d(ctx0, qkv_mixed, conv_channels, conv_kernel_size - 1, n_seqs,
                            qkv_mixed->nb[1], qkv_mixed->nb[2], (n_seq_tokens - slot - (conv_kernel_size - 1))*qkv_mixed->nb[1]);
                    ggml_tensor * cell = ggml_view_2d(ctx0, conv_states_all, (conv_kernel_size - 1)*conv_channels, n_seqs,
                            conv_states_all->nb[1], (slot*mctx_cur->get_size() + mctx_cur->get_head())*conv_states_all->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, last), cell));
                }
            }
        }
    }
#endif

    ggml_tensor * conv_input = conv_output_silu ? nullptr : build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

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
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
// build_layer_attn_linear() for one token of one sequence on a single ROCm device, with the ops of ggml-qwen35-decode.h
// each op has its own GGML_Q35MOE_* switch; a part whose op is off or not usable gets the stock nodes
ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn_linear_one(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;
    const auto & layer    = model.layers[il];

    if (!ggml_q35moe_mvshare_enabled() && !ggml_q35moe_ba1_enabled() && !ggml_q35moe_prep1_enabled() && !ggml_q35moe_gdn1_enabled() &&
            !ggml_q35moe_post1_enabled()) {
        return nullptr;
    }
    // a reason that holds for the whole context goes to the log once; the shape of the batch does not
    // GGML_Q35MOE_RS1=1: also with rollback snapshots, one token reads the row that the memory names and writes slot 0
    const char * why = nullptr;
    const auto dev = q35moe_rocm_dev(cparams, model, loras, il, !ggml_q35moe_rs1_enabled(), &why);

    const int64_t d_inner       = hparams.ssm_d_inner;
    const int64_t head_k_dim    = hparams.ssm_d_state;
    const int64_t num_k_heads   = hparams.ssm_n_group;
    const int64_t num_v_heads   = hparams.ssm_dt_rank;
    const int64_t head_v_dim    = d_inner / num_v_heads;
    const int64_t key_dim       = head_k_dim * num_k_heads;
    const int64_t conv_channels = d_inner + 2 * key_dim;

    // the kernels replay the stock kernels of this head size only
    if (dev && (head_k_dim != ggml_qwen35_head_dim || head_v_dim != head_k_dim || d_inner != head_v_dim * num_v_heads)) {
        why = "the head size of the recurrent blocks";
    }
    if (why) {
        q35moe_log_off("GGML_Q35MOE_MVSHARE/BA1/PREP1/GDN1/POST1", why);
        return nullptr;
    }
    // PREP1 and GDN1 write the state cache, the backend takes them only with the cache in its memory
    if (!cparams.offload_kqv && (ggml_q35moe_prep1_enabled() || ggml_q35moe_gdn1_enabled())) {
        q35moe_log_off("GGML_Q35MOE_PREP1/GDN1", "the recurrent state is not in device memory (--no-kv-offload)");
    }
    if (ubatch.n_seq_tokens != 1 || ubatch.n_seqs != 1 || mctx_cur->get_n_rs() != 1) {
        return nullptr;
    }

    const size_t fs = sizeof(float);

    // a weight that build_lora_mm() gives to one plain MUL_MAT
    auto plain = [&](const ggml_tensor * w, const ggml_tensor * w_s) {
        return w && !w_s && (!hdmd || !hdmd->rot.count(w)) && (!prec_policy || !prec_policy->prec_src1.count(w));
    };
    auto usable = [&](const ggml_tensor * candidate) {
        return candidate && ggml_backend_dev_supports_op(dev, candidate);
    };
    // the zero state of build_rs(); the fused ops read the state row of the sequence from the cache
    auto clear_zero_state = [&](ggml_tensor * states) {
        const int32_t rs_zero = mctx_cur->get_rs_z();
        ggml_tensor * state_zero = ggml_view_1d(ctx0, states, states->ne[0]*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
        ggml_build_forward_expand(gf, ggml_fill_inplace(ctx0, state_zero, 0.0f));
    };
    auto cell_of = [&](ggml_context * ctx, ggml_tensor * states) {
        return ggml_view_2d(ctx, states, states->ne[0], 1, states->nb[1], mctx_cur->get_head()*states->nb[1]);
    };

    ggml_tensor * qkv_mixed = nullptr;
    ggml_tensor * z         = nullptr;
    ggml_tensor * beta      = nullptr;
    ggml_tensor * alpha     = nullptr;

    const bool ba_plain = plain(layer.ssm_beta, layer.ssm_beta_s) && plain(layer.ssm_alpha, layer.ssm_alpha_s) &&
        layer.ssm_beta->ne[1] == num_v_heads && layer.ssm_alpha->ne[1] == num_v_heads;
    const bool ba_quant = ba_plain && ggml_is_quantized(layer.ssm_beta->type) && ggml_is_quantized(layer.ssm_alpha->type);

    // the quantized products of cur share one quantized input
    if (ggml_q35moe_mvshare_enabled() && plain(layer.wqkv, layer.wqkv_s) && plain(layer.wqkv_gate, layer.wqkv_gate_s)) {
        std::vector<ggml_tensor *> w = { layer.wqkv, layer.wqkv_gate };
        if (ba_quant) {
            w.push_back(layer.ssm_beta);
            w.push_back(layer.ssm_alpha);
        }
        ggml_context_ptr probe{ggml_init({2*ggml_tensor_overhead(), nullptr, true})};
        if (usable(ggml_qwen35_try_mvshare(probe.get(), w.data(), w.size(), cur))) {
            ggml_tensor * shared = ggml_qwen35_try_mvshare(ctx0, w.data(), w.size(), cur);
            cb(shared, "linear_attn_shared", il);

            const int64_t n_qkv = layer.wqkv->ne[1];
            const int64_t n_z   = layer.wqkv_gate->ne[1];

            qkv_mixed = ggml_view_3d(ctx0, shared, n_qkv, 1, 1, n_qkv*fs, n_qkv*fs, 0);
            z         = ggml_view_2d(ctx0, shared, n_z, 1, n_z*fs, n_qkv*fs);
            if (ba_quant) {
                beta  = ggml_view_4d(ctx0, shared, 1, num_v_heads, 1, 1, fs, num_v_heads*fs, num_v_heads*fs, (n_qkv + n_z)*fs);
                alpha = ggml_view_3d(ctx0, shared, num_v_heads, 1, 1, num_v_heads*fs, num_v_heads*fs, (n_qkv + n_z + num_v_heads)*fs);
            }
        }
    }
    if (!qkv_mixed) {
        auto qkvz = build_qkvz(cur, il);
        qkv_mixed = qkvz.first;
        z         = qkvz.second;
    }

    // the two F32 products of cur in one launch
    if (!beta && ggml_q35moe_ba1_enabled() && ba_plain) {
        ggml_tensor * w[] = { layer.ssm_beta, layer.ssm_alpha };
        ggml_context_ptr probe{ggml_init({2*ggml_tensor_overhead(), nullptr, true})};
        if (usable(ggml_q35moe_try_mvf(probe.get(), w, 2, cur))) {
            ggml_tensor * ba = ggml_q35moe_try_mvf(ctx0, w, 2, cur);
            cb(ba, "beta_alpha", il);

            beta  = ggml_view_4d(ctx0, ba, 1, num_v_heads, 1, 1, fs, num_v_heads*fs, num_v_heads*fs, 0);
            alpha = ggml_view_3d(ctx0, ba, num_v_heads, 1, 1, num_v_heads*fs, num_v_heads*fs, num_v_heads*fs);
        }
    }
    if (!beta) {
        beta = build_lora_mm(layer.ssm_beta, cur, layer.ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, 1, 1);

        alpha = build_lora_mm(layer.ssm_alpha, cur, layer.ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, 1, 1);
    }
    cb(beta, "beta", il);
    cb(alpha, "alpha", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
    ggml_tensor * conv_kernel     = layer.ssm_conv1d;

    const float eps_norm = hparams.f_norm_rms_eps;

    ggml_tensor * gate   = nullptr;
    ggml_tensor * q_conv = nullptr;
    ggml_tensor * k_conv = nullptr;
    ggml_tensor * v_conv = nullptr;

    // conv, SiLU, L2 norm, beta and gate in one op, which also updates the conv state
    if (ggml_q35moe_prep1_enabled() && conv_channels == qkv_mixed->ne[0]) {
        auto make = [&](ggml_context * ctx) {
            return ggml_qwen35_try_prep1(ctx, qkv_mixed, beta, alpha, conv_kernel, layer.ssm_dt, layer.ssm_a,
                    conv_states_all, inp->s_copy_main, cell_of(ctx, conv_states_all),
                    2*key_dim, eps_norm/(float) head_k_dim, 1.0f/sqrtf((float) head_k_dim));
        };
        ggml_context_ptr probe{ggml_init({3*ggml_tensor_overhead(), nullptr, true})};
        if (usable(make(probe.get()))) {
            clear_zero_state(conv_states_all);
            ggml_tensor * prep = make(ctx0);
            cb(prep, "linear_attn_prep", il);

            q_conv = ggml_view_4d(ctx0, prep, head_k_dim, num_k_heads, 1, 1, head_k_dim*fs, key_dim*fs, key_dim*fs, 0);
            k_conv = ggml_view_4d(ctx0, prep, head_k_dim, num_k_heads, 1, 1, head_k_dim*fs, key_dim*fs, key_dim*fs, key_dim*fs);
            v_conv = ggml_view_4d(ctx0, prep, head_v_dim, num_v_heads, 1, 1, head_v_dim*fs, d_inner*fs, d_inner*fs, 2*key_dim*fs);
            beta   = ggml_view_4d(ctx0, prep, 1, num_v_heads, 1, 1, fs, num_v_heads*fs, num_v_heads*fs, conv_channels*fs);
            gate   = ggml_view_4d(ctx0, prep, 1, num_v_heads, 1, 1, fs, num_v_heads*fs, num_v_heads*fs, (conv_channels + num_v_heads)*fs);
        }
    }

    // with the fused delta net the state stays in the cache: no copy of it here
    const bool gdn_one = ggml_q35moe_gdn1_enabled();

    ggml_tensor * state = nullptr;
    if (!q_conv) {
        beta = ggml_sigmoid(ctx0, beta);
        cb(beta, "beta_sigmoid", il);

        ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, layer.ssm_dt);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
        cb(alpha_softplus, "a_softplus", il);

        gate = ggml_mul(ctx0, alpha_softplus, layer.ssm_a);  // -A_log.exp() * softplus
        cb(gate, "gate", il);

        gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, 1, 1);

        ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel->ne[0], conv_channels, il);

        if (!gdn_one) {
            state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), 1);
            state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, 1);
            cb(state, "state_predelta", il);
        }

        ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        cb(conv_output_silu, "conv_output_silu", il);

        const int64_t nb1_qkv = ggml_row_size(conv_output_silu->type, conv_channels);

        q_conv = ggml_view_4d(ctx0, conv_output_silu, head_k_dim, num_k_heads, 1, 1, head_k_dim*fs, nb1_qkv, nb1_qkv, 0);
        k_conv = ggml_view_4d(ctx0, conv_output_silu, head_k_dim, num_k_heads, 1, 1, head_k_dim*fs, nb1_qkv, nb1_qkv, key_dim*fs);
        v_conv = ggml_view_4d(ctx0, conv_output_silu, head_v_dim, num_v_heads, 1, 1, head_v_dim*fs, nb1_qkv, nb1_qkv, 2*key_dim*fs);

        cb(q_conv, "q_conv", il);
        cb(k_conv, "k_conv", il);
        cb(v_conv, "v_conv", il);

        q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = nullptr;
    if (gdn_one) {
        auto make = [&](ggml_context * ctx) {
            return ggml_qwen35_try_gdn1(ctx, q_conv, k_conv, v_conv, gate, beta, ssm_states_all, inp->s_copy_main, cell_of(ctx, ssm_states_all));
        };
        ggml_context_ptr probe{ggml_init({3*ggml_tensor_overhead(), nullptr, true})};
        if (usable(make(probe.get()))) {
            clear_zero_state(ssm_states_all);
            output = make(ctx0);
            cb(output, "attn_output", il);
        }
    }
    if (!output) {
        if (!state) {
            state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), 1);
            state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, 1);
            cb(state, "state_predelta", il);
        }
        output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
    }

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, 1, 1);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = nullptr;
    if (ggml_q35moe_post1_enabled()) {
        ggml_context_ptr probe{ggml_init({2*ggml_tensor_overhead(), nullptr, true})};
        if (usable(ggml_qwen35_try_post1(probe.get(), output, layer.ssm_norm, z_2d, eps_norm))) {
            attn_out_norm = ggml_qwen35_try_post1(ctx0, output, layer.ssm_norm, z_2d, eps_norm);
        }
    }
    if (!attn_out_norm) {
        attn_out_norm = build_norm_gated(output, layer.ssm_norm, z_2d, il);
    }

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, 1);

    // Output projection
    cur = build_lora_mm(layer.ssm_out, final_output, layer.ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}
#endif

#ifdef GGML_INKLING_DEPTHMIX_GFX1103
// build_layer_attn_linear() for 2 to n_rs_seq + 1 tokens of one sequence in a context with rollback snapshots (a draft verification
// batch) on a single ROCm device. GGML_Q35MOE_PREPN, GDNN and POSTN give one part each to an op of ggml-qwen35-decode.h,
// a part whose op is off or not usable gets the stock nodes
ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn_linear_few(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;
    const auto & layer    = model.layers[il];

    if (!ggml_q35moe_prepn_enabled() && !ggml_q35moe_gdnn_enabled() && !ggml_q35moe_postn_enabled()) {
        return nullptr;
    }
    // a reason that holds for the whole context goes to the log once; the shape of the batch does not
    const char * why = nullptr;
    const auto dev = q35moe_rocm_dev(cparams, model, loras, il, false, &why);

    const int64_t d_inner       = hparams.ssm_d_inner;
    const int64_t head_k_dim    = hparams.ssm_d_state;
    const int64_t num_k_heads   = hparams.ssm_n_group;
    const int64_t num_v_heads   = hparams.ssm_dt_rank;
    const int64_t head_v_dim    = d_inner / num_v_heads;
    const int64_t key_dim       = head_k_dim * num_k_heads;
    const int64_t conv_channels = d_inner + 2 * key_dim;
    const int64_t n_t           = ubatch.n_seq_tokens;

    // the kernels replay the stock kernels of this head size only
    if (dev && (head_k_dim != ggml_qwen35_head_dim || head_v_dim != head_k_dim || d_inner != head_v_dim * num_v_heads)) {
        why = "the head size of the recurrent blocks";
    }
    if (dev && !why && cparams.n_rs_seq == 0) {
        why = "the context keeps no rollback snapshots (n_rs_seq is 0)";
    }
    if (why) {
        q35moe_log_off("GGML_Q35MOE_PREPN/GDNN/POSTN", why);
        return nullptr;
    }
    // PREPN and GDNN write the state cache, the backend takes them only with the cache in its memory
    if (!cparams.offload_kqv && (ggml_q35moe_prepn_enabled() || ggml_q35moe_gdnn_enabled())) {
        q35moe_log_off("GGML_Q35MOE_PREPN/GDNN", "the recurrent state is not in device memory (--no-kv-offload)");
    }
    // up to n_rs_seq + 1 tokens every token writes a slot and the stock delta net is the plain kernel
    // 2 to 5 tokens are checked against the stock graph (fixture, spec-ab.sh); the ops could take up to ggml_q35moe_few_max
    constexpr int64_t n_t_checked = 5;
    if (n_t < 2 || n_t > (int64_t) cparams.n_rs_seq + 1 || n_t > n_t_checked || n_t > ggml_q35moe_few_max || ubatch.n_seqs != 1 ||
            mctx_cur->get_n_rs() != 1) {
        return nullptr;
    }

    const size_t fs = sizeof(float);

    auto usable = [&](const ggml_tensor * candidate) {
        return candidate && ggml_backend_dev_supports_op(dev, candidate);
    };
    // the zero state of build_rs(); the fused ops read the state row of the sequence from the cache
    auto clear_zero_state = [&](ggml_tensor * states) {
        const int32_t rs_zero = mctx_cur->get_rs_z();
        ggml_tensor * state_zero = ggml_view_1d(ctx0, states, states->ne[0]*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
        ggml_build_forward_expand(gf, ggml_fill_inplace(ctx0, state_zero, 0.0f));
    };
    // slot 0 of the cell of the sequence
    auto cell_of = [&](ggml_context * ctx, ggml_tensor * states) {
        return ggml_view_2d(ctx, states, states->ne[0], 1, states->nb[1], mctx_cur->get_head()*states->nb[1]);
    };

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(layer.ssm_beta, cur, layer.ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_t, 1);
    cb(beta, "beta", il);

    ggml_tensor * alpha = build_lora_mm(layer.ssm_alpha, cur, layer.ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_t, 1);
    cb(alpha, "alpha", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
    ggml_tensor * conv_kernel     = layer.ssm_conv1d;

    const float eps_norm = hparams.f_norm_rms_eps;

    // n_t <= n_rs_seq + 1: the stock graph writes one slot per token, see build_conv_state_store() and build_recurrent_attn()
    const int64_t slot_rows = mctx_cur->get_size();

    ggml_tensor * gate   = nullptr;
    ggml_tensor * q_conv = nullptr;
    ggml_tensor * k_conv = nullptr;
    ggml_tensor * v_conv = nullptr;

    // conv, SiLU, L2 norm, beta and gate of all tokens in one op, which also writes the rollback windows of the conv state
    if (ggml_q35moe_prepn_enabled() && conv_channels == qkv_mixed->ne[0]) {
        auto make = [&](ggml_context * ctx) {
            return ggml_q35moe_try_prepn(ctx, qkv_mixed, beta, alpha, conv_kernel, layer.ssm_dt, layer.ssm_a,
                    conv_states_all, inp->s_copy_main, cell_of(ctx, conv_states_all),
                    2*key_dim, eps_norm/(float) head_k_dim, 1.0f/sqrtf((float) head_k_dim), slot_rows);
        };
        ggml_context_ptr probe{ggml_init({3*ggml_tensor_overhead(), nullptr, true})};
        if (usable(make(probe.get()))) {
            clear_zero_state(conv_states_all);
            ggml_tensor * prep = make(ctx0);
            cb(prep, "linear_attn_prep", il);

            const size_t nb_t = conv_channels*fs;
            const size_t nb_h = num_v_heads*fs;

            q_conv = ggml_view_4d(ctx0, prep, head_k_dim, num_k_heads, n_t, 1, head_k_dim*fs, nb_t, nb_t*n_t, 0);
            k_conv = ggml_view_4d(ctx0, prep, head_k_dim, num_k_heads, n_t, 1, head_k_dim*fs, nb_t, nb_t*n_t, key_dim*fs);
            v_conv = ggml_view_4d(ctx0, prep, head_v_dim, num_v_heads, n_t, 1, head_v_dim*fs, nb_t, nb_t*n_t, 2*key_dim*fs);
            beta   = ggml_view_4d(ctx0, prep, 1, num_v_heads, n_t, 1, fs, nb_h, nb_h*n_t, nb_t*n_t);
            gate   = ggml_view_4d(ctx0, prep, 1, num_v_heads, n_t, 1, fs, nb_h, nb_h*n_t, nb_t*n_t + nb_h*n_t);
        }
    }

    // with the fused delta net the state stays in the cache: no copy of it here
    const bool gdn_few = ggml_q35moe_gdnn_enabled();

    ggml_tensor * state = nullptr;
    if (!q_conv) {
        beta = ggml_sigmoid(ctx0, beta);
        cb(beta, "beta_sigmoid", il);

        ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, layer.ssm_dt);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
        cb(alpha_softplus, "a_softplus", il);

        gate = ggml_mul(ctx0, alpha_softplus, layer.ssm_a);  // -A_log.exp() * softplus
        cb(gate, "gate", il);

        gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_t, 1);

        ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel->ne[0], conv_channels, il);

        if (!gdn_few) {
            state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), 1);
            state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, 1);
            cb(state, "state_predelta", il);
        }

        ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        cb(conv_output_silu, "conv_output_silu", il);

        const int64_t nb1_qkv = ggml_row_size(conv_output_silu->type, conv_channels);

        q_conv = ggml_view_4d(ctx0, conv_output_silu, head_k_dim, num_k_heads, n_t, 1, head_k_dim*fs, nb1_qkv, nb1_qkv*n_t, 0);
        k_conv = ggml_view_4d(ctx0, conv_output_silu, head_k_dim, num_k_heads, n_t, 1, head_k_dim*fs, nb1_qkv, nb1_qkv*n_t, key_dim*fs);
        v_conv = ggml_view_4d(ctx0, conv_output_silu, head_v_dim, num_v_heads, n_t, 1, head_v_dim*fs, nb1_qkv, nb1_qkv*n_t, 2*key_dim*fs);

        cb(q_conv, "q_conv", il);
        cb(k_conv, "k_conv", il);
        cb(v_conv, "v_conv", il);

        q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // the delta net reads the state row of the sequence and writes the snapshots of the tokens
    ggml_tensor * output = nullptr;
    if (gdn_few) {
        auto make = [&](ggml_context * ctx) {
            return ggml_q35moe_try_gdnn(ctx, q_conv, k_conv, v_conv, gate, beta, ssm_states_all, inp->s_copy_main, cell_of(ctx, ssm_states_all),
                    slot_rows);
        };
        ggml_context_ptr probe{ggml_init({3*ggml_tensor_overhead(), nullptr, true})};
        if (usable(make(probe.get()))) {
            clear_zero_state(ssm_states_all);
            output = make(ctx0);
            cb(output, "attn_output", il);
        }
    }
    if (!output) {
        if (!state) {
            state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), 1);
            state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, 1);
            cb(state, "state_predelta", il);
        }
        output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
    }

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_t, 1);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = nullptr;
    if (ggml_q35moe_postn_enabled()) {
        ggml_context_ptr probe{ggml_init({2*ggml_tensor_overhead(), nullptr, true})};
        if (usable(ggml_qwen35_try_post1(probe.get(), output, layer.ssm_norm, z_2d, eps_norm))) {
            attn_out_norm = ggml_qwen35_try_post1(ctx0, output, layer.ssm_norm, z_2d, eps_norm);
        }
    }
    if (!attn_out_norm) {
        attn_out_norm = build_norm_gated(output, layer.ssm_norm, z_2d, il);
    }

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_t);

    // Output projection
    cur = build_lora_mm(layer.ssm_out, final_output, layer.ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}
#endif

ggml_tensor * llama_model_qwen35moe::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // Add shared experts if present - following Qwen3Next reference implementation
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // Apply shared expert gating as in the reference implementation
        // The shared expert has its own gate that is sigmoided
        // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        // Apply sigmoid to the gate
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);


        // Apply the gate to the shared expert output
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// build_layer_ffn() plus the residual as a fused op of ggml-q35moe-ffn.h, nullptr if no op takes this layer
// GGML_Q35MOE_FFN1 is the op for a graph of one token, GGML_Q35MOE_FFNM the op for a graph of 2 to ggml_q35moe_ffnm_max tokens (GGML_Q35MOE_FFNM5: one more)
ggml_tensor * llama_model_qwen35moe::graph::build_layer_ffn_one(ggml_tensor * cur, ggml_tensor * residual, const int il) {
#ifdef GGML_INKLING_DEPTHMIX_GFX1103
    if (ggml_q35moe_exponce_enabled() && !ggml_q35moe_ffnm_enabled()) {
        q35moe_log_off("GGML_Q35MOE_EXPONCE", "it acts inside the op of GGML_Q35MOE_FFNM, which is not set");
    }
    if (ggml_q35moe_ffnm5_enabled() && !ggml_q35moe_ffnm_enabled()) {
        q35moe_log_off("GGML_Q35MOE_FFNM5", "it extends the op of GGML_Q35MOE_FFNM, which is not set");
    }
    if (!ggml_q35moe_ffn1_enabled() && !ggml_q35moe_ffnm_enabled()) {
        return nullptr;
    }

    constexpr int n_w = GGML_Q35MOE_FFN1_N_SRC - GGML_Q35MOE_FFN1_GATE_INP;

    using try_op_t = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor **);

    // why block i does not take the op for the input x and the residual r, nullptr if it takes it; w gets the weights of the block
    const auto refusal = [&](const int i, ggml_tensor * x, ggml_tensor * r, ggml_tensor ** w, const try_op_t try_op) -> const char * {
        const auto & layer = model.layers[i];

        if (moe_cache != nullptr) {
            return "the MoE cache is on";
        }
        const char * why = nullptr;
        const auto dev = q35moe_rocm_dev(cparams, model, loras, i, false, &why);
        if (dev == nullptr) {
            return why;
        }

        // the op replays build_moe_ffn() and build_ffn() with these settings only
        constexpr float eps = 1e-6f;
        const bool plain_graph = n_expert == ggml_q35moe_n_expert && n_expert_used == ggml_q35moe_n_used &&
            hparams.n_expert_used(i) == (uint32_t) ggml_q35moe_n_used && hparams.n_expert_groups <= 1 &&
            (hparams.expert_weights_scale == 0.0f || hparams.expert_weights_scale == 1.0f) && hparams.expert_weights_norm_eps <= 0.0f &&
            !hparams.expert_router_f32 && !(hparams.swiglu_clamp_exp[i] > eps) && !(hparams.swiglu_clamp_shexp[i] > eps) &&
            !layer.ffn_gate_up_exps && !layer.ffn_up_exps_s && !layer.ffn_gate_exps_s && !layer.ffn_down_exps_s &&
            !layer.ffn_up_shexp_s && !layer.ffn_gate_shexp_s && !layer.ffn_down_shexp_s;
        if (!plain_graph) {
            return "feed-forward settings that the op does not replay";
        }

        ggml_tensor * const ws[n_w] = {
            layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
            layer.ffn_gate_inp_shexp, layer.ffn_up_shexp, layer.ffn_gate_shexp, layer.ffn_down_shexp,
        };
        for (int j = 0; j < n_w; ++j) {
            const ggml_tensor * m = ws[j];
            // no rotated input and no precision request for a product
            if (m == nullptr || (hdmd && hdmd->rot.count(m)) || (prec_policy && (prec_policy->prec_src1.count(m) || prec_policy->prec_acc.count(m)))) {
                return "a weight is missing, rotated or has a precision request";
            }
            w[j] = ws[j];
        }

        ggml_context_ptr probe{ggml_init({2*ggml_tensor_overhead(), nullptr, true})};
        auto * candidate = try_op(probe.get(), x, r, w);
        if (candidate == nullptr) {
            return "weight types or sizes without a kernel";
        }
        if (!ggml_backend_dev_supports_op(dev, candidate)) {
            return "the backend refuses the op (a weight is not in device memory, or GGML_CUDA_DISABLE_FUSION is set)";
        }
        return nullptr;
    };

    // the number of blocks that take the op for n_cols tokens, and the first reason against it
    const auto report = [&](const char * name, const char * graph, const int64_t n_cols, const try_op_t try_op) {
        int          n_take    = 0;
        int          first     = -1;
        const char * first_why = nullptr;
        for (int i = 0; i < n_layer; ++i) {
            ggml_context_ptr inputs{ggml_init({3*ggml_tensor_overhead(), nullptr, true})};
            ggml_tensor * x = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, ggml_q35moe_n_embd, n_cols);
            ggml_tensor * r = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, ggml_q35moe_n_embd, n_cols);
            ggml_tensor * w[n_w];
            const char *  why = refusal(i, x, r, w, try_op);
            if (why == nullptr) {
                n_take++;
            } else if (first_why == nullptr) {
                first     = i;
                first_why = why;
            }
        }
        if (first_why == nullptr) {
            LLAMA_LOG_INFO("qwen35moe: %s: %d of %d blocks take the op in a graph of %s\n", name, n_take, (int) n_layer, graph);
        } else {
            LLAMA_LOG_WARN("qwen35moe: %s: %d of %d blocks take the op in a graph of %s, block %d does not: %s\n",
                    name, n_take, (int) n_layer, graph, first, first_why);
        }
    };

    // once per process and switch
    if (!model.no_alloc()) {
        if (ggml_q35moe_ffn1_enabled()) {
            static std::once_flag reported;
            std::call_once(reported, [&] { report("GGML_Q35MOE_FFN1", "one token", 1, ggml_q35moe_try_ffn1); });
        }
        if (ggml_q35moe_ffnm_enabled()) {
            static std::once_flag reported;
            std::call_once(reported, [&] {
                const char * graph = ggml_q35moe_ffnm5_enabled() ? "2 to 5 tokens (5 tokens: not the stock result)" : "2 to 4 tokens";
                report(ggml_q35moe_exponce_enabled() ? "GGML_Q35MOE_FFNM with GGML_Q35MOE_EXPONCE" : "GGML_Q35MOE_FFNM", graph, 2, ggml_q35moe_try_ffnm);
            });
        }
    }
    static_assert(ggml_q35moe_ffnm_max == 4 && ggml_q35moe_ffnm5_max == 5, "the log line names the largest graph");

    // An op is for graphs of its token count only: the last block of a larger graph with few output rows has a small input too.
    // An op there would change the topology between batches (one slot generates, another reads a prompt), and the scheduler then plans the memory again.
    try_op_t try_op = nullptr;
    if (ubatch.n_tokens == 1) {
        try_op = ggml_q35moe_ffn1_enabled() ? ggml_q35moe_try_ffn1 : nullptr;
    } else if (ubatch.n_tokens <= (uint32_t) ggml_q35moe_ffnm5_max && cur->ne[1] == (int64_t) ubatch.n_tokens) {
        // the op itself refuses the largest count without GGML_Q35MOE_FFNM5
        try_op = ggml_q35moe_ffnm_enabled() ? ggml_q35moe_try_ffnm : nullptr;
    }
    if (try_op == nullptr) {
        return nullptr;
    }

    ggml_tensor * w[n_w];
    if (refusal(il, cur, residual, w, try_op) != nullptr) {
        return nullptr;
    }

    return try_op(ctx0, cur, residual, w);
#else
    GGML_UNUSED(cur);
    GGML_UNUSED(residual);
    GGML_UNUSED(il);
    return nullptr;
#endif
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 MoE
llama_model_qwen35moe::graph_mtp::graph_mtp(const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35MOE MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35MOE MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj    && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm      && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm      && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp     && "MTP block missing ffn_gate_inp");

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

    // MoE FFN — routed experts plus gated shared expert (mirrors qwen35moe).
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "mtp_ffn_moe_out", il);

    if (layer.ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        ggml_tensor * shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, cur);
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "mtp_shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "mtp_ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    } else {
        cur = moe_out;
    }
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35MOE MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn= cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)");

    // row list (llama_set_logits_rows()): the head as one 1-row matrix per token id, the list picks the rows
    // the products of the picked rows are the ones of the whole head; the samplers map the list entries back to token ids
    const int64_t n_out = cur->ne[1];
    const bool head_plain = head_s == nullptr && loras->empty() && hdmd == nullptr &&
        (prec_policy == nullptr || (prec_policy->prec_acc.count(head_w) == 0 && prec_policy->prec_src1.count(head_w) == 0));
    // the raw logits of an output without a backend sampler go to the host with one value per token id
    bool all_sampled = !samplers.empty();
    for (uint32_t i = 0; i < ubatch.n_tokens && all_sampled; ++i) {
        all_sampled = !ubatch.output[i] || (ubatch.n_seq_id[i] == 1 && samplers.count(ubatch.seq_id[i][0]) > 0);
    }
    if (logits_rows != nullptr && head_plain && all_sampled && n_out >= 1 && n_out <= logits_rows->ne[1] &&
            ggml_n_dims(head_w) == 2 && ggml_is_contiguous(head_w)) {
        const int64_t n_rows = logits_rows->ne[0];

        ggml_tensor * head_rows = ggml_reshape_3d(ctx0, head_w, head_w->ne[0], 1, head_w->ne[1]);
        ggml_tensor * ids       = ggml_view_2d(ctx0, logits_rows, n_rows, n_out, logits_rows->nb[1], 0);

        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, n_out);
        cur = ggml_mul_mat_id(ctx0, head_rows, cur, ids);
        cur = ggml_reshape_2d(ctx0, cur, n_rows, n_out);

        res->t_logits_rows = ggml_view_1d(ctx0, logits_rows, n_rows, 0);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}

#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. Feed-forward of a qwen35moe block in one op, for one token and for a few tokens.
extern "C" {
// GGML_Q35MOE_FFN1=1: router, top-k, routed experts, shared expert and residual of one token.
// x: n_embd F32 (the normed block input), res: n_embd F32, w: the 8 weight leafs from GGML_Q35MOE_FFN1_GATE_INP on
// result: n_embd F32 = sum_k p_k*down_k(silu(gate_k*x)*(up_k*x)) + sigmoid(gate_inp_shexp*x)*down_sh(silu(gate_sh*x)*(up_sh*x)) + res
// k: the n_used experts with the largest router probability, p_k: their probabilities divided by their sum
GGML_API bool ggml_q35moe_ffn1_enabled();
GGML_API ggml_tensor * ggml_q35moe_try_ffn1(ggml_context * ctx, ggml_tensor * x, ggml_tensor * res, ggml_tensor ** w);
GGML_API bool ggml_q35moe_is_ffn1(const ggml_tensor * op);

// GGML_Q35MOE_FFNM=1: the same feed-forward for 2 to ggml_q35moe_ffnm_max tokens in one op.
// x, res and the result: [n_embd, n_tokens] F32, the sources are those of the one-token op. Every token gets the value of the one-token formula.
GGML_API bool ggml_q35moe_ffnm_enabled();
GGML_API ggml_tensor * ggml_q35moe_try_ffnm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * res, ggml_tensor ** w);
GGML_API bool ggml_q35moe_is_ffnm(const ggml_tensor * op);

// GGML_Q35MOE_EXPONCE=1, inside the many-token op: the rows of an expert are loaded once for all tokens of the batch that selected it. Same values.
GGML_API bool ggml_q35moe_exponce_enabled();

// GGML_Q35MOE_FFNM5=1: the many-token op also takes ggml_q35moe_ffnm5_max tokens. NOT the stock result of that batch size:
// there the stock routed products are matrix kernels that round in another way. Every token gets the value that a smaller batch gives it.
GGML_API bool ggml_q35moe_ffnm5_enabled();
}

// sources of the op
enum ggml_q35moe_ffn1_src {
    GGML_Q35MOE_FFN1_X,
    GGML_Q35MOE_FFN1_RES,
    GGML_Q35MOE_FFN1_GATE_INP,       // [n_embd, n_expert] F32
    GGML_Q35MOE_FFN1_UP_EXPS,        // [n_embd, n_ff, n_expert]
    GGML_Q35MOE_FFN1_GATE_EXPS,      // [n_embd, n_ff, n_expert], type of UP_EXPS
    GGML_Q35MOE_FFN1_DOWN_EXPS,      // [n_ff, n_embd, n_expert]
    GGML_Q35MOE_FFN1_GATE_INP_SHEXP, // [n_embd] F32
    GGML_Q35MOE_FFN1_UP_SHEXP,       // [n_embd, n_ff]
    GGML_Q35MOE_FFN1_GATE_SHEXP,     // [n_embd, n_ff], type of UP_SHEXP
    GGML_Q35MOE_FFN1_DOWN_SHEXP,     // [n_ff, n_embd]
    GGML_Q35MOE_FFN1_N_SRC,
};

// the kernels replay the stock kernels of these sizes only
constexpr int ggml_q35moe_n_embd   = 2048;
constexpr int ggml_q35moe_n_ff     = 512;
constexpr int ggml_q35moe_n_expert = 256;
constexpr int ggml_q35moe_n_used   = 8;

// most tokens of the many-token op: up to here the stock routed products of Q4_K, Q5_K and Q6_K weights are vector kernels on gfx1103
constexpr int ggml_q35moe_ffnm_max  = 4;
// most tokens with GGML_Q35MOE_FFNM5
constexpr int ggml_q35moe_ffnm5_max = 5;

// lower bound of the sum of the selected probabilities, the CLAMP node of build_moe_ffn
constexpr float ggml_q35moe_sum_min = 6.103515625e-5f;

// quantized weight types with a kernel
constexpr bool ggml_q35moe_ffn1_type(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

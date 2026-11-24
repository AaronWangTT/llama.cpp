#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. One-token feed-forward of a qwen35moe block in one op.
extern "C" {
// GGML_Q35MOE_FFN1=1: router, top-k, routed experts, shared expert and residual of one token.
// x: n_embd F32 (the normed block input), res: n_embd F32, w: the 8 weight leafs from GGML_Q35MOE_FFN1_GATE_INP on
// result: n_embd F32 = sum_k p_k*down_k(silu(gate_k*x)*(up_k*x)) + sigmoid(gate_inp_shexp*x)*down_sh(silu(gate_sh*x)*(up_sh*x)) + res
// k: the n_used experts with the largest router probability, p_k: their probabilities divided by their sum
GGML_API bool ggml_q35moe_ffn1_enabled();
GGML_API ggml_tensor * ggml_q35moe_try_ffn1(ggml_context * ctx, ggml_tensor * x, ggml_tensor * res, ggml_tensor ** w);
GGML_API bool ggml_q35moe_is_ffn1(const ggml_tensor * op);
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

// lower bound of the sum of the selected probabilities, the CLAMP node of build_moe_ffn
constexpr float ggml_q35moe_sum_min = 6.103515625e-5f;

// quantized weight types with a kernel
constexpr bool ggml_q35moe_ffn1_type(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

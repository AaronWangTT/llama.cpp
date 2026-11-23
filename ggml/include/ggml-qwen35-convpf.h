#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. Causal convolution of the recurrent block for many tokens of one sequence in one op:
// out[c, t] = silu(sum_j x[t + j, c]*weight[j, c]), x = past followed by cur along the tokens. It replaces TRANSPOSE, CONCAT, SSM_CONV and SILU.
extern "C" {
GGML_API bool ggml_qwen35_convpf_enabled();
// GGML_Q35MOE_CONVPF=1: the same op in the qwen35moe graph
GGML_API bool ggml_q35moe_convpf_enabled();
// cur: [n_channels, n_tokens] (rows may have a stride), past: [n_conv - 1, n_channels], weight: [n_conv, n_channels], all F32
GGML_API ggml_tensor * ggml_qwen35_try_convpf(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * past, ggml_tensor * weight);
GGML_API bool ggml_qwen35_is_convpf(const ggml_tensor * op);
}

constexpr int ggml_qwen35_convpf_n_conv       = 4;
constexpr int ggml_qwen35_convpf_min_n_tokens = 32;

#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. Relative bias table of one token in one op:
// out[h, d] = sum_r bias[r, d]*rcur[r, h] - sinks[h], with one extra all-zero distance row at the end.
extern "C" {
// the sum over r is a chain of fused multiply-adds in ascending r, which matches the rocBLAS product bit for bit
GGML_API bool ggml_inkling_bias1_enabled();
// bias: [n_rel, n_extent], rcur: [n_rel, n_head], sinks: n_head, all F32; result [n_head, n_extent + 1]
GGML_API ggml_tensor * ggml_inkling_try_bias1(ggml_context * ctx, ggml_tensor * bias, ggml_tensor * rcur, ggml_tensor * sinks);
GGML_API bool ggml_inkling_is_bias1(const ggml_tensor * op);
}

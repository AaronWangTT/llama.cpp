#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. Causal short convolution of one token in one op:
// out = (cur + conv([past, cur], weight)) / sqrt(n_conv + 1), and the new state [past[1..], cur] is written to cell.
extern "C" {
GGML_API bool ggml_inkling_conv1_enabled();
// cur: n_channels, past: [n_conv - 1, n_channels], weight: [n_conv, n_channels], cell: (n_conv - 1)*n_channels, all F32
GGML_API ggml_tensor * ggml_inkling_try_conv1(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * past, ggml_tensor * weight, ggml_tensor * cell);
GGML_API bool ggml_inkling_is_conv1(const ggml_tensor * op);
}

constexpr int ggml_inkling_conv1_n_conv = 4;

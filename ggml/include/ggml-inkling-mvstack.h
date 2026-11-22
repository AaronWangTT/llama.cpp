#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. Several Q8_0 matrices times the same vector in one op:
// out = [w[0]*x, w[1]*x, ...], the rows of all matrices one after the other.
extern "C" {
GGML_API bool ggml_inkling_mvstack_enabled();
// x: n_cols F32, w[i]: [n_cols, n_rows_i] Q8_0 weight leafs; result: sum of n_rows_i F32
GGML_API ggml_tensor * ggml_inkling_try_mvstack(ggml_context * ctx, ggml_tensor ** w, int n_w, ggml_tensor * x);
GGML_API bool ggml_inkling_is_mvstack(const ggml_tensor * op);
}

constexpr int ggml_inkling_mvstack_max = 6;

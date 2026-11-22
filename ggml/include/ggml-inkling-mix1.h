#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. Depth mix of one token in one op:
// rms norm of each state, scores against the queries, softmax over the states, weighted sum.
extern "C" {
// GGML_INKLING_MIX1: 0 = off, 1 = bit-exact against the stock chain, 2 = faster kernel that sums in a different order
GGML_API int  ggml_inkling_mix1_mode();
GGML_API bool ggml_inkling_mix1_enabled();
// states are in the row order of the stock concat (newest block first, cur last), each n_embd F32
GGML_API ggml_tensor * ggml_inkling_try_mix1(ggml_context * ctx, ggml_tensor ** states, int n_states, ggml_tensor * query);
GGML_API bool ggml_inkling_is_mix1(const ggml_tensor * op);
// take a column of the mixed states as a view when it is one block of memory
GGML_API bool ggml_inkling_viewcol_enabled();
}

constexpr int   ggml_inkling_mix1_max_states = 6;
constexpr int   ggml_inkling_mix1_max_query  = 4;
constexpr float ggml_inkling_mix1_eps        = 1e-6f;

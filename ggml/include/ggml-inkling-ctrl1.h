#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. Loop controller of one token in one op:
// gate = down*[rms_norm(cur), rms_norm(message)], out = cur - tanh(forget*gate)*cur + tanh(gain*gate)*message.
extern "C" {
GGML_API bool ggml_inkling_ctrl1_enabled();
// cur, message: n_embd, F32; down: [2*n_embd, rank]; gain, forget: [rank, n_embd], both of one type; the weights are F32 or BF16
GGML_API ggml_tensor * ggml_inkling_try_ctrl1(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * message,
        ggml_tensor * down, ggml_tensor * gain, ggml_tensor * forget);
GGML_API bool ggml_inkling_is_ctrl1(const ggml_tensor * op);
}

// the kernels replay the stock kernels of these sizes only
constexpr int   ggml_inkling_ctrl1_n_embd = 2304;
constexpr int   ggml_inkling_ctrl1_rank   = 64;
constexpr float ggml_inkling_ctrl1_eps    = 1e-6f;

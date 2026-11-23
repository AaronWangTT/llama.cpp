#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. One-token ops of the gated delta net blocks (qwen35), one switch per op.
extern "C" {
// GGML_QWEN35_MVSHARE=1: several quantized matrices times the same vector, the vector is quantized once.
// x: n_cols F32, w[i]: [n_cols, n_rows_i] quantized weight leafs; result: [w[0]*x, w[1]*x, ...], sum of n_rows_i F32
GGML_API bool ggml_qwen35_mvshare_enabled();
GGML_API ggml_tensor * ggml_qwen35_try_mvshare(ggml_context * ctx, ggml_tensor ** w, int n_w, ggml_tensor * x);
GGML_API bool ggml_qwen35_is_mvshare(const ggml_tensor * op);

// GGML_QWEN35_PREP1=1: inputs of the gated delta net in one op.
// y = silu(conv([past, qkv], conv_w)), the first n_norm channels get the L2 norm per head of ggml_qwen35_head_dim channels.
// result: [y, sigmoid(beta), softplus(alpha + dt)*a]. past is row ids[0] of states, the new conv state is written to cell.
// qkv: n_channels, conv_w: [n_conv, n_channels], beta, alpha, dt, a: n_heads, states: [(n_conv - 1)*n_channels, n_cells],
// ids: 1 I32, cell: (n_conv - 1)*n_channels; eps and scale are the values of the stock RMS_NORM and SCALE nodes
GGML_API bool ggml_qwen35_prep1_enabled();
GGML_API ggml_tensor * ggml_qwen35_try_prep1(ggml_context * ctx, ggml_tensor * qkv, ggml_tensor * beta, ggml_tensor * alpha,
        ggml_tensor * conv_w, ggml_tensor * dt, ggml_tensor * a, ggml_tensor * states, ggml_tensor * ids, ggml_tensor * cell,
        int64_t n_norm, float eps, float scale);
GGML_API bool ggml_qwen35_is_prep1(const ggml_tensor * op);

// GGML_QWEN35_GDN1=1: gated delta net step of one token without a copy of the state.
// The state is row ids[0] of states, the new state is written to cell (the same row or another one).
// q, k: [head_dim, n_heads_k], v: [head_dim, n_heads], g, b: n_heads, states: [head_dim*head_dim*n_heads, n_cells],
// ids: 1 I32, cell: head_dim*head_dim*n_heads; result: [head_dim, n_heads]
GGML_API bool ggml_qwen35_gdn1_enabled();
GGML_API ggml_tensor * ggml_qwen35_try_gdn1(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * g, ggml_tensor * b, ggml_tensor * states, ggml_tensor * ids, ggml_tensor * cell);
GGML_API bool ggml_qwen35_is_gdn1(const ggml_tensor * op);

// GGML_QWEN35_POST1=1: gated norm of the delta net output, result: silu(z)*(rms_norm(x)*w) per head.
// x, z: [head_dim, n_heads], w: head_dim; eps is the value of the stock RMS_NORM node; result: head_dim*n_heads
GGML_API bool ggml_qwen35_post1_enabled();
GGML_API ggml_tensor * ggml_qwen35_try_post1(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * z, float eps);
GGML_API bool ggml_qwen35_is_post1(const ggml_tensor * op);
}

constexpr int ggml_qwen35_mvshare_max = 8;
// the kernels replay the stock kernels of these sizes only
constexpr int ggml_qwen35_head_dim = 128;
constexpr int ggml_qwen35_n_conv   = 4;

// op_params slots after ggml_custom_op_params
constexpr int ggml_qwen35_prep1_slot_n_norm = 8;
constexpr int ggml_qwen35_prep1_slot_eps    = 10;
constexpr int ggml_qwen35_prep1_slot_scale  = 11;
constexpr int ggml_qwen35_post1_slot_eps    = 8;

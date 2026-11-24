#pragma once

#include "ggml.h"

// Private to the native gfx1103 build. One-token ops of the gated delta net blocks (qwen35, qwen35moe), one switch per op.
// A GGML_QWEN35_* switch turns an op on in the qwen35 graph, the GGML_Q35MOE_* switch of the same name in the qwen35moe graph.
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

// GGML_Q35MOE_MVSHARE=1, GGML_Q35MOE_PREP1=1, GGML_Q35MOE_GDN1=1, GGML_Q35MOE_POST1=1: the four ops above in the qwen35moe graph
GGML_API bool ggml_q35moe_mvshare_enabled();
GGML_API bool ggml_q35moe_prep1_enabled();
GGML_API bool ggml_q35moe_gdn1_enabled();
GGML_API bool ggml_q35moe_post1_enabled();

// GGML_Q35MOE_BA1=1: several F32 matrices times the same vector in one launch (the beta and alpha products of qwen35moe).
// x: n_cols F32, w[i]: [n_cols, n_rows_i] F32 weight leafs; result: [w[0]*x, w[1]*x, ...], sum of n_rows_i F32
GGML_API bool ggml_q35moe_ba1_enabled();
GGML_API ggml_tensor * ggml_q35moe_try_mvf(ggml_context * ctx, ggml_tensor ** w, int n_w, ggml_tensor * x);
GGML_API bool ggml_q35moe_is_mvf(const ggml_tensor * op);

// GGML_Q35MOE_RS1=1: the five one-token ops of qwen35moe also in a context that keeps rollback snapshots (n_rs_seq > 0).
// The ops read the row that the memory names (a rollback slot or slot 0) and write slot 0, as the stock graph of one token does.
GGML_API bool ggml_q35moe_rs1_enabled();

// The ops below take 2 to ggml_q35moe_few_max tokens of one sequence in a context with rollback snapshots (a draft verification batch).
// A state tensor holds slot s of cell c in row s*slot_rows + c. cell is the row of slot 0, a view of states. An op writes one slot
// per token: slot s gets the state of s tokens before the end of the batch. The row ids[0] can be one of these rows.

// GGML_Q35MOE_PREPN=1: ggml_qwen35_try_prep1 for n_tokens tokens, with the rollback windows of the conv state.
// qkv: [n_channels, n_tokens], beta, alpha: n_heads*n_tokens; result: [y of all tokens, sigmoid(beta), softplus(alpha + dt)*a],
// n_channels*n_tokens + 2*n_heads*n_tokens F32
GGML_API bool ggml_q35moe_prepn_enabled();
GGML_API ggml_tensor * ggml_q35moe_try_prepn(ggml_context * ctx, ggml_tensor * qkv, ggml_tensor * beta, ggml_tensor * alpha,
        ggml_tensor * conv_w, ggml_tensor * dt, ggml_tensor * a, ggml_tensor * states, ggml_tensor * ids, ggml_tensor * cell,
        int64_t n_norm, float eps, float scale, int64_t slot_rows);
GGML_API bool ggml_q35moe_is_prepn(const ggml_tensor * op);

// GGML_Q35MOE_GDNN=1: gated delta net over n_tokens tokens without a copy of the state, the op writes the snapshots.
// q, k: [head_dim, n_heads_k, n_tokens], v: [head_dim, n_heads, n_tokens] (the tokens may have a stride), g, b: n_heads*n_tokens;
// result: [head_dim, n_heads, n_tokens]
GGML_API bool ggml_q35moe_gdnn_enabled();
GGML_API ggml_tensor * ggml_q35moe_try_gdnn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * g, ggml_tensor * b, ggml_tensor * states, ggml_tensor * ids, ggml_tensor * cell, int64_t slot_rows);
GGML_API bool ggml_q35moe_is_gdnn(const ggml_tensor * op);

// GGML_Q35MOE_POSTN=1: ggml_qwen35_try_post1 takes the heads of several tokens in the qwen35moe graph
GGML_API bool ggml_q35moe_postn_enabled();

// GGML_Q35MOE_GDNPF=1: a batch of more than n_rs_seq + 1 tokens per sequence in the qwen35moe graph (a prompt batch).
// The memory takes no rollback after such a batch, so the graph writes rollback slot 0 only, like a context without snapshots.
// The graph marks its GGML_OP_GATED_DELTA_NET node (one slot, head_dim 128, scalar gate) for the kernel of q35moe-gdnpf.cuh.
GGML_API bool ggml_q35moe_gdnpf_enabled();
GGML_API void ggml_q35moe_gdnpf_mark(ggml_tensor * gdn);
GGML_API bool ggml_q35moe_is_gdnpf(const ggml_tensor * gdn);
}

constexpr int ggml_qwen35_mvshare_max = 8;
constexpr int ggml_q35moe_mvf_max     = 4;
// most tokens of a batch that PREPN and GDNN take; the dense products of such a batch are vector kernels
constexpr int ggml_q35moe_few_max     = 8;
// the kernels replay the stock kernels of these sizes only
constexpr int ggml_qwen35_head_dim = 128;
constexpr int ggml_qwen35_n_conv   = 4;

// op_params slots after ggml_custom_op_params
constexpr int ggml_qwen35_prep1_slot_n_norm = 8;
constexpr int ggml_qwen35_prep1_slot_eps    = 10;
constexpr int ggml_qwen35_prep1_slot_scale  = 11;
constexpr int ggml_qwen35_post1_slot_eps    = 8;
constexpr int ggml_q35moe_prepn_slot_slot_rows = 12;
constexpr int ggml_q35moe_gdnn_slot_slot_rows  = 8;
// op_params slot of GGML_OP_GATED_DELTA_NET after the slot count
constexpr int ggml_q35moe_gdnpf_slot_mark      = 1;

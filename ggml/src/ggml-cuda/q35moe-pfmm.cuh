#pragma once

#include "inkling-pfmm.cuh"

// qwen35moe.pfmm: F32 products of a full prefill ubatch of qwen35moe (router, beta, alpha) with the kernel of inkling-pfmm.cuh.
// rocBLAS takes the same Tensile kernel for these sizes (MT32x32x8, StaggerU 32, stride 256 B), so the results have the same bits.
constexpr int q35moe_pfmm_n_tokens = 512;

static bool q35moe_pfmm_enabled() {
    return ggml_kperf_on(GGML_KPERF_QWEN35MOE_PFMM);
}

// the F32 weights of the architecture, [2048, rows]: router with 256 rows, beta and alpha with 32 rows
static bool q35moe_pfmm_supported(int device, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * dst) {
    const auto & info = ggml_cuda_info().devices[device];
    if (info.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || info.warp_size != 32 || !q35moe_pfmm_enabled()) {
        return false;
    }
    const auto ends_with = [w](const char * suffix) {
        const size_t n = strlen(w->name), m = strlen(suffix);
        return n >= m && strcmp(w->name + n - m, suffix) == 0;
    };
    const int64_t n_k = w->ne[0], n_rows = w->ne[1];
    if (w->type != GGML_TYPE_F32 || w->view_src || n_k != 2048) {
        return false;
    }
    if (!((n_rows == 256 && ends_with(".ffn_gate_inp.weight")) || (n_rows == 32 && (ends_with(".ssm_beta.weight") || ends_with(".ssm_alpha.weight"))))) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || x->ne[0] != n_k || x->ne[1] != q35moe_pfmm_n_tokens ||
            dst->ne[0] != n_rows || dst->ne[1] != q35moe_pfmm_n_tokens) {
        return false;
    }
    for (const ggml_tensor * t : { w, x, dst }) {
        // the kernel loads float4 from x and stores float4 to dst
        if (t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t) || (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    return inkling_pfmm_stock_sgemm("qwen35moe.pfmm");
}

static void q35moe_pfmm_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * dst) {
    const int n_k = w->ne[0], n_rows = w->ne[1];
    // start offsets as rocBLAS sets them: up to 32, halved while k/64 is smaller
    int stagger = 32;
    while (stagger > 1 && n_k/64 < stagger) {
        stagger /= 2;
    }
    const dim3 grid(n_rows/inkling_pfmm_rows, q35moe_pfmm_n_tokens/inkling_pfmm_cols);
    inkling_pfmm<float><<<grid, 256, 0, ctx.stream()>>>(static_cast<const float *>(w->data), static_cast<const float *>(x->data),
        static_cast<float *>(dst->data), n_k, n_rows, stagger - 1);
    CUDA_CHECK(cudaGetLastError());
}

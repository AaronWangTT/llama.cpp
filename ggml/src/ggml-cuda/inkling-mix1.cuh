#pragma once
#include "ggml-inkling-mix1.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private MIX1 experiment for gfx1103 only"
#endif

// one block of 1024 threads, thread t owns elements 4*t .. 4*t + 3
static __global__ void inkling_mix1(const float * query, const float * s0, const float * s1, const float * s2, const float * s3,
        const float * s4, const float * s5, float * dst, const int ns, const int nq, const int n_embd, const float kq) {
    constexpr int max_s = ggml_inkling_mix1_max_states;
    constexpr int max_q = ggml_inkling_mix1_max_query;
    constexpr int n_red = max_s + max_s*max_q;

    const float * st[max_s] = { s0, s1, s2, s3, s4, s5 };
    static_assert(max_s == 6, "one pointer argument per state");
    const int  tid  = threadIdx.x;
    const int  warp = tid / WARP_SIZE;
    const int  lane = tid % WARP_SIZE;
    const int  i0   = 4*tid;
    const bool live = i0 < n_embd;

    __shared__ float red[n_red][WARP_SIZE];

    float x[max_s][4];
    float acc[n_red];
#pragma unroll
    for (int v = 0; v < n_red; ++v) {
        acc[v] = 0.0f;
    }

#pragma unroll
    for (int s = 0; s < max_s; ++s) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x[s][k] = live && s < ns ? st[s][i0 + k] : 0.0f;
            acc[s] += x[s][k]*x[s][k];
        }
    }
#pragma unroll
    for (int j = 0; j < max_q; ++j) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float w = live && j < nq ? query[j*n_embd + i0 + k] : 0.0f;
#pragma unroll
            for (int s = 0; s < max_s; ++s) {
                acc[max_s + s*max_q + j] += x[s][k]*w;
            }
        }
    }

    // block sums of all values with one barrier
#pragma unroll
    for (int v = 0; v < n_red; ++v) {
        acc[v] = warp_reduce_sum(acc[v]);
        if (lane == 0) {
            red[v][warp] = acc[v];
        }
    }
    __syncthreads();
#pragma unroll
    for (int v = 0; v < n_red; ++v) {
        acc[v] = warp_reduce_sum(red[v][lane]);
    }

    if (!live) {
        return;
    }

    float scale[max_s];
#pragma unroll
    for (int s = 0; s < max_s; ++s) {
        scale[s] = rsqrtf(acc[s] / n_embd + ggml_inkling_mix1_eps);
    }

#pragma unroll
    for (int j = 0; j < max_q; ++j) {
        if (j >= nq) {
            continue;
        }
        float p[max_s];
        float max = -INFINITY;
#pragma unroll
        for (int s = 0; s < max_s; ++s) {
            p[s] = s < ns ? acc[max_s + s*max_q + j]*scale[s]*kq : -INFINITY;
            max  = fmaxf(max, p[s]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < max_s; ++s) {
            p[s] = s < ns ? expf(p[s] - max) : 0.0f;
            sum += p[s];
        }
#pragma unroll
        for (int s = 0; s < max_s; ++s) {
            p[s] = p[s] / sum * scale[s];
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float out = 0.0f;
#pragma unroll
            for (int s = 0; s < max_s; ++s) {
                out += x[s][k]*p[s];
            }
            dst[j*n_embd + i0 + k] = out;
        }
    }
}

static bool inkling_mix1_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    return info.cc == GGML_CUDA_CC_OFFSET_AMD + 0x1103 && info.warp_size == 32 && ggml_inkling_mix1_enabled() && ggml_inkling_is_mix1(op);
}

static void inkling_mix1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const float * s[ggml_inkling_mix1_max_states] = {};
    int ns = 0;
    for (; ns < ggml_inkling_mix1_max_states && op->src[ns + 1]; ++ns) {
        s[ns] = static_cast<const float *>(op->src[ns + 1]->data);
    }
    const int n_embd = op->ne[0];
    inkling_mix1<<<1, 1024, 0, ctx.stream()>>>(static_cast<const float *>(op->src[0]->data), s[0], s[1], s[2], s[3], s[4], s[5],
        static_cast<float *>(op->data), ns, (int) op->ne[1], n_embd, 1.0f/sqrtf(n_embd));
}

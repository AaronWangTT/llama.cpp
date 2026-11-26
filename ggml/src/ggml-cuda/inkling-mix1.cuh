#pragma once
#include "ggml-inkling-mix1.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private MIX1 experiment for gfx1103 only"
#endif

// the bit-exact kernel copies the stock kernels of this row size
constexpr int inkling_mix1x_embd = 2304;

struct inkling_mix1_states {
    const float * s[ggml_inkling_mix1_max_states];
};

// value of lane 0 after a warp sum when only the first n lanes are not zero
template <int n>
static __device__ __forceinline__ float inkling_mix1x_lanes(const float * v) {
    static_assert(n >= 1 && n <= 8, "eight lanes");
    float a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = (i < n ? v[i] : 0.0f) + 0.0f;
    }
    return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
}

// GGML_INKLING_MIX1=1, bit-exact against the stock chain. Each stage keeps the thread layout and the sum order of its stock kernel:
// - norm: rms_norm_f32<1024, 1, true>
// - scores: mul_mat_vec_f<float, float, nq, 256>
// - softmax: soft_max_f32<true, 0, 0>
// - weighted sum: mul_mat_vec_f<float, float, nq, 32> (even ns), rocBLAS gemv (odd ns, one query), Tensile gemm (odd ns, more queries)
// One block of 576 threads, thread t owns elements 4*t .. 4*t + 3.
template <int ns, int nq>
static __global__ void inkling_mix1x(const float * query, const inkling_mix1_states st, float * dst, const int n_embd, const float kq) {
    constexpr int norm_warps  = inkling_mix1x_embd / 4 / WARP_SIZE;
    constexpr int score_block = 256;
    constexpr int score_warps = score_block / WARP_SIZE;
    constexpr int score_iter  = (inkling_mix1x_embd/2 + score_block - 1) / score_block;
    static_assert(inkling_mix1x_embd % (4*WARP_SIZE) == 0 && norm_warps <= WARP_SIZE, "full warps of float4");

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    __shared__ float red_norm[ns][WARP_SIZE];
    __shared__ float red_score[ns*nq][score_warps];

    // all loads first, so that they overlap (loads in a chain took most of the kernel time)
    float x[ns][4];
#pragma unroll
    for (int s = 0; s < ns; ++s) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x[s][k] = st.s[s][4*tid + k];
        }
    }
    // threads 0 .. 255 are the block of the stock score kernel, thread t owns the pairs t, t + 256, ...
    float sq[nq][score_iter][2];
    float sx[ns][score_iter][2];
    if (tid < score_block) {
#pragma unroll
        for (int i = 0; i < score_iter; ++i) {
            const int col = 2*(tid + i*score_block);
            if (col < inkling_mix1x_embd) {
#pragma unroll
                for (int j = 0; j < nq; ++j) {
                    sq[j][i][0] = query[j*n_embd + col];
                    sq[j][i][1] = query[j*n_embd + col + 1];
                }
#pragma unroll
                for (int s = 0; s < ns; ++s) {
                    sx[s][i][0] = st.s[s][col];
                    sx[s][i][1] = st.s[s][col + 1];
                }
            }
        }
    }

#pragma unroll
    for (int s = 0; s < ns; ++s) {
        float tmp = fmaf(x[s][3], x[s][3], fmaf(x[s][2], x[s][2], fmaf(x[s][0], x[s][0], x[s][1]*x[s][1])));
        tmp = warp_reduce_sum(tmp);
        if (lane == 0) {
            red_norm[s][warp] = tmp;
        }
    }
    __syncthreads();

    float scale[ns];
#pragma unroll
    for (int s = 0; s < ns; ++s) {
        float tmp = lane < norm_warps ? red_norm[s][lane] : 0.0f;
        tmp = warp_reduce_sum(tmp);
        scale[s] = rsqrtf(tmp / n_embd + ggml_inkling_mix1_eps);
    }

    if (tid < score_block) {
        float sum[ns][nq];
#pragma unroll
        for (int s = 0; s < ns; ++s) {
#pragma unroll
            for (int j = 0; j < nq; ++j) {
                sum[s][j] = 0.0f;
            }
        }
#pragma unroll
        for (int i = 0; i < score_iter; ++i) {
            const int col = 2*(tid + i*score_block);
            if (col < inkling_mix1x_embd) {
#pragma unroll
                for (int s = 0; s < ns; ++s) {
                    const float x0 = scale[s]*sx[s][i][0];
                    const float x1 = scale[s]*sx[s][i][1];
#pragma unroll
                    for (int j = 0; j < nq; ++j) {
                        sum[s][j] = fmaf(x0, sq[j][i][0], sum[s][j]);
                        sum[s][j] = fmaf(x1, sq[j][i][1], sum[s][j]);
                    }
                }
            }
        }
#pragma unroll
        for (int s = 0; s < ns; ++s) {
#pragma unroll
            for (int j = 0; j < nq; ++j) {
                const float v = warp_reduce_sum(sum[s][j]);
                if (lane == 0) {
                    red_score[s*nq + j][warp] = v;
                }
            }
        }
    }
    __syncthreads();

    float prob[ns][nq];
#pragma unroll
    for (int j = 0; j < nq; ++j) {
        float val[ns];
        float max_val = -INFINITY;
#pragma unroll
        for (int s = 0; s < ns; ++s) {
            val[s]  = fmaf(inkling_mix1x_lanes<score_warps>(red_score[s*nq + j]), kq, 0.0f);
            max_val = fmaxf(max_val, val[s]);
        }
#pragma unroll
        for (int s = 0; s < ns; ++s) {
            val[s] = expf(val[s] - max_val);
        }
        const float inv_sum = 1.0f / inkling_mix1x_lanes<ns>(val);
#pragma unroll
        for (int s = 0; s < ns; ++s) {
            prob[s][j] = val[s] * inv_sum;
        }
    }

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float xn[ns];
#pragma unroll
        for (int s = 0; s < ns; ++s) {
            xn[s] = scale[s]*x[s][k];
        }
#pragma unroll
        for (int j = 0; j < nq; ++j) {
            float out;
            if constexpr (ns % 2 == 0) {
                // lane m owns the states 2*m and 2*m + 1
                float pair[ns/2];
#pragma unroll
                for (int m = 0; m < ns/2; ++m) {
                    pair[m] = fmaf(xn[2*m + 1], prob[2*m + 1][j], fmaf(xn[2*m], prob[2*m][j], 0.0f));
                }
                out = inkling_mix1x_lanes<ns/2>(pair);
            } else if constexpr (nq == 1) {
                // lane s owns the state s
                float term[ns];
#pragma unroll
                for (int s = 0; s < ns; ++s) {
                    term[s] = fmaf(xn[s], prob[s][j], 0.0f);
                }
                out = inkling_mix1x_lanes<ns>(term);
            } else {
                out = 0.0f;
#pragma unroll
                for (int s = 0; s < ns; ++s) {
                    out = fmaf(xn[s], prob[s][j], out);
                }
            }
            dst[j*n_embd + 4*tid + k] = out;
        }
    }
}

static bool inkling_mix1_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    if (info.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || info.warp_size != 32 || !ggml_inkling_mix1_enabled() || !ggml_inkling_is_mix1(op)) {
        return false;
    }
    return op->ne[0] == inkling_mix1x_embd;
}

template <int ns>
static void inkling_mix1x_launch(const int nq, const float * query, const inkling_mix1_states & st, float * dst, cudaStream_t stream) {
    const int   n_embd = inkling_mix1x_embd;
    const float kq     = 1.0f/sqrtf(n_embd);
    const dim3  block(n_embd/4);
    static_assert(ggml_inkling_mix1_max_query == 4, "one case per query count");
    switch (nq) {
        case 1: inkling_mix1x<ns, 1><<<1, block, 0, stream>>>(query, st, dst, n_embd, kq); break;
        case 2: inkling_mix1x<ns, 2><<<1, block, 0, stream>>>(query, st, dst, n_embd, kq); break;
        case 3: inkling_mix1x<ns, 3><<<1, block, 0, stream>>>(query, st, dst, n_embd, kq); break;
        case 4: inkling_mix1x<ns, 4><<<1, block, 0, stream>>>(query, st, dst, n_embd, kq); break;
        default: GGML_ABORT("fatal error");
    }
}

static void inkling_mix1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    inkling_mix1_states st = {};
    int ns = 0;
    for (; ns < ggml_inkling_mix1_max_states && op->src[ns + 1]; ++ns) {
        st.s[ns] = static_cast<const float *>(op->src[ns + 1]->data);
    }
    const float * query = static_cast<const float *>(op->src[0]->data);
    float *       dst   = static_cast<float *>(op->data);
    const int     nq    = op->ne[1];
    static_assert(ggml_inkling_mix1_max_states == 6, "one case per state count");
    switch (ns) {
        case 2: inkling_mix1x_launch<2>(nq, query, st, dst, ctx.stream()); break;
        case 3: inkling_mix1x_launch<3>(nq, query, st, dst, ctx.stream()); break;
        case 4: inkling_mix1x_launch<4>(nq, query, st, dst, ctx.stream()); break;
        case 5: inkling_mix1x_launch<5>(nq, query, st, dst, ctx.stream()); break;
        case 6: inkling_mix1x_launch<6>(nq, query, st, dst, ctx.stream()); break;
        default: GGML_ABORT("fatal error");
    }
}

#pragma once
#include "ggml-qwen35-decode.h"
#include "mmq-few.cuh"
#include "unary.cuh"

// needs mmvq.cuh, which has no include guard: include this file after it and after mmvf.cuh

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile these private one-token ops for gfx1103 only"
#endif

static bool qwen35_decode_device(int device) {
    const auto & info = ggml_cuda_info().devices[device];
    return info.cc == GGML_CUDA_CC_OFFSET_AMD + 0x1103 && info.warp_size == 32;
}

// a state cache in memory of this device; the ops write to it, so a copy made by the scheduler would lose the new state
static bool qwen35_state_on_device(int device, const ggml_tensor * t) {
    const ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    return buf && buf->buft == ggml_backend_cuda_buffer_type(device);
}

// a weight in memory of this device; the scheduler gives an op with a weight in host memory to the CPU backend
static bool qwen35_weight_on_device(int device, const ggml_tensor * t) {
    const ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    return buf && buf->buft == ggml_backend_cuda_buffer_type(device) && ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

//
// mvshare
//

static bool qwen35_mvshare_supported(int device, const ggml_tensor * op) {
    if (!qwen35_decode_device(device) || !(ggml_qwen35_mvshare_enabled() || ggml_q35moe_mvshare_enabled()) || !ggml_qwen35_is_mvshare(op)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);
    for (int j = 1; j < GGML_MAX_SRC && op->src[j]; ++j) {
        const ggml_tensor * w = op->src[j];
        // weights of this device only: no padding to clear, no copy by the scheduler
        if (!w->buffer || w->buffer->buft != buft || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return false;
        }
        // only the products that ggml_cuda_mul_mat gives to ggml_cuda_mul_mat_vec_q
        if (ggml_cuda_should_use_mmq_few(w, op->src[0], cc) || !ggml_cuda_should_use_mmvq(w->type, cc, 1)) {
            return false;
        }
    }
    return true;
}

static void qwen35_mvshare_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    int n_w = 0;
    while (n_w + 1 < GGML_MAX_SRC && op->src[n_w + 1]) {
        n_w++;
    }
    ggml_cuda_mul_mat_vec_q_shared(ctx, op->src + 1, n_w, op->src[0], static_cast<float *>(op->data));
}

//
// mvsharen
//

static bool q35moe_mvsharen_supported(int device, const ggml_tensor * op) {
    if (!qwen35_decode_device(device) || !ggml_q35moe_mvsharen_enabled() || !ggml_q35moe_is_mvsharen(op)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    for (int j = 1; j < GGML_MAX_SRC && op->src[j]; ++j) {
        const ggml_tensor * w = op->src[j];
        // only the products that ggml_cuda_mul_mat gives to ggml_cuda_mul_mat_vec_q, with the weight in memory of this device
        if (!qwen35_weight_on_device(device, w) || ggml_cuda_should_use_mmq_few(w, op->src[0], cc) || !ggml_cuda_should_use_mmvq(w->type, cc, op->src[0]->ne[1])) {
            return false;
        }
    }
    return true;
}

static void q35moe_mvsharen_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    int n_w = 0;
    while (n_w + 1 < GGML_MAX_SRC && op->src[n_w + 1]) {
        n_w++;
    }
    if (ggml_kperf_trace(GGML_KPERF_QWEN35MOE_MVSHAREN)) {
        GGML_LOG_INFO("Q35MOE_MVSHAREN_HIT matrices=%d rows=%lld tokens=%lld\n", n_w, (long long) (op->ne[0]/op->src[0]->ne[1]), (long long) op->src[0]->ne[1]);
    }
    ggml_cuda_mul_mat_vec_q_stack(ctx, op->src + 1, n_w, op->src[0], static_cast<float *>(op->data));
}

//
// prep1
//

// one warp per head of 128 channels, a thread owns 4 channels in a row
// conv and SiLU follow ssm_conv_f32<true>, the L2 norm follows rms_norm_f32<32, 1, true, false, false, true>,
// the head values follow op_sigmoid, op_add and unary_gated_op_kernel<op_softplus>
// past and cell can be the same row: a thread reads its part of the row before it writes it
__launch_bounds__(WARP_SIZE, 1)
static __global__ void qwen35_prep1(
        const float * qkv_ptr, const float * beta, const float * alpha, const float * weight_ptr, const float * dt, const float * a,
        const float * states, const int32_t * ids, float * cell_ptr, float * dst_ptr,
        const int64_t state_stride, const int n_channels, const int n_norm, const int n_heads,
        const int ncols, const float eps, const float scale_out) {
    constexpr int n_conv = ggml_qwen35_n_conv;
    constexpr int vec    = ggml_qwen35_head_dim / WARP_SIZE;
    const float * GGML_CUDA_RESTRICT qkv    = qkv_ptr;
    const float * GGML_CUDA_RESTRICT weight = weight_ptr;
    float       * GGML_CUDA_RESTRICT dst    = dst_ptr;

    const int     c0   = blockIdx.x*ggml_qwen35_head_dim + threadIdx.x*vec;
    const float * past = states + ids[0]*state_stride + (int64_t) c0*(n_conv - 1);
    float       * cell = cell_ptr + (int64_t) c0*(n_conv - 1);

    float x[vec][n_conv];
    float w[vec][n_conv];
#pragma unroll
    for (int i = 0; i < vec; i++) {
#pragma unroll
        for (int j = 0; j < n_conv; j++) {
            w[i][j] = weight[(c0 + i)*n_conv + j];
        }
#pragma unroll
        for (int j = 0; j < n_conv - 1; j++) {
            x[i][j] = past[i*(n_conv - 1) + j];
        }
        x[i][n_conv - 1] = qkv[c0 + i];
    }

    float y[vec];
#pragma unroll
    for (int i = 0; i < vec; i++) {
        const float b = 0.0f;
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < n_conv; j++) {
            sumf += x[i][j] * w[i][j];
        }
        sumf += b;
        y[i] = ggml_cuda_op_silu_single(sumf);
    }

#pragma unroll
    for (int i = 0; i < vec; i++) {
#pragma unroll
        for (int j = 0; j < n_conv - 1; j++) {
            cell[i*(n_conv - 1) + j] = x[i][j + 1];
        }
    }

    static_assert(vec == 4, "the norm reads one float4 per thread");
    float4 r = make_float4(y[0], y[1], y[2], y[3]);
    if ((int) blockIdx.x*ggml_qwen35_head_dim < n_norm) {
        const float4 xs = r;
        float tmp = 0.0f;
        tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(tmp / ncols + eps);

        r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
        r.x *= scale_out;
        r.y *= scale_out;
        r.z *= scale_out;
        r.w *= scale_out;
    }
    dst[c0 + 0] = r.x;
    dst[c0 + 1] = r.y;
    dst[c0 + 2] = r.z;
    dst[c0 + 3] = r.w;

    if (blockIdx.x == 0) {
        for (int h = threadIdx.x; h < n_heads; h += WARP_SIZE) {
            dst[n_channels + h] = 1.0f / (1.0f + expf(-beta[h]));

            const float t = alpha[h] + dt[h];
            dst[n_channels + n_heads + h] = ((t > 20.0f) ? t : log1pf(expf(t))) * a[h];
        }
    }
}

static bool qwen35_prep1_supported(int device, const ggml_tensor * op) {
    return qwen35_decode_device(device) && (ggml_qwen35_prep1_enabled() || ggml_q35moe_prep1_enabled()) && ggml_qwen35_is_prep1(op) &&
        qwen35_weight_on_device(device, op->src[3]) && qwen35_weight_on_device(device, op->src[4]) && qwen35_weight_on_device(device, op->src[5]) &&
        qwen35_state_on_device(device, op->src[6]) && qwen35_state_on_device(device, op->src[8]);
}

static void qwen35_prep1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const int n_channels = ggml_nelements(op->src[0]);
    const int n_heads    = ggml_nelements(op->src[1]);
    qwen35_prep1<<<n_channels/ggml_qwen35_head_dim, WARP_SIZE, 0, ctx.stream()>>>(
        static_cast<const float *>(op->src[0]->data), static_cast<const float *>(op->src[1]->data), static_cast<const float *>(op->src[2]->data),
        static_cast<const float *>(op->src[3]->data), static_cast<const float *>(op->src[4]->data), static_cast<const float *>(op->src[5]->data),
        static_cast<const float *>(op->src[6]->data), static_cast<const int32_t *>(op->src[7]->data), static_cast<float *>(op->src[8]->data),
        static_cast<float *>(op->data), op->src[6]->ne[0], n_channels, ggml_get_op_params_i32(op, ggml_qwen35_prep1_slot_n_norm), n_heads,
        ggml_qwen35_head_dim, ggml_get_op_params_f32(op, ggml_qwen35_prep1_slot_eps), ggml_get_op_params_f32(op, ggml_qwen35_prep1_slot_scale));
    CUDA_CHECK(cudaGetLastError());
}

//
// gdn1
//

// gated_delta_net_cuda<S_v, false, false> of one sequence, statement by statement; the input state is row ids[0] of states.
// states and state can be the same row: a thread reads its part of the state before it writes it
template <int S_v, int cols_per_warp>
__launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
static __global__ void qwen35_gdn1(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * states, const int32_t * ids, float * dst, float * state,
        const int64_t state_stride, const int64_t H, const int64_t n_tokens, const int64_t neqk1, const uint3 neqk1_magic, const float scale) {
    const uint32_t h_idx = blockIdx.x;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);

    float * attn_data = dst;

    const float * curr_state = states + ids[0] * state_stride + h_idx * S_v * S_v + col * S_v;
    state += h_idx * S_v * S_v;
    attn_data += h_idx * S_v;

    float s_shard[rows_per_lane];

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + (t * neqk1 + iq1) * S_v;
        const float * k_t = k + (t * neqk1 + iq1) * S_v;
        const float * v_t = v + (t * H + h_idx) * S_v;

        const int64_t gb_offset = t * H + h_idx;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset;

        const float beta_val = *beta_t;

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        const float g_val = expf(*g_t);

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }

        float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

        if (lane_in_col == 0) {
            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * lanes_per_col + lane_in_col;
        state[col * S_v + i] = s_shard[r];
    }
}

static bool qwen35_gdn1_supported(int device, const ggml_tensor * op) {
    return qwen35_decode_device(device) && (ggml_qwen35_gdn1_enabled() || ggml_q35moe_gdn1_enabled()) && ggml_qwen35_is_gdn1(op) &&
        qwen35_state_on_device(device, op->src[5]) && qwen35_state_on_device(device, op->src[7]);
}

static void qwen35_gdn1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    constexpr int S_v           = ggml_qwen35_head_dim;
    constexpr int cols_per_warp = 4; // gdn_cols_per_warp
    const int64_t H     = op->src[2]->ne[1];
    const int64_t neqk1 = op->src[0]->ne[1];

    // the block layout of launch_gated_delta_net
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nsm       = ggml_cuda_info().devices[ctx.device].nsm;
    int num_warps = 4;
    while (num_warps > 1 && H*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    const dim3 grid_dims(H, 1, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    const dim3 block_dims(warp_size, num_warps, 1);

    qwen35_gdn1<S_v, cols_per_warp><<<grid_dims, block_dims, 0, ctx.stream()>>>(
        static_cast<const float *>(op->src[0]->data), static_cast<const float *>(op->src[1]->data), static_cast<const float *>(op->src[2]->data),
        static_cast<const float *>(op->src[3]->data), static_cast<const float *>(op->src[4]->data),
        static_cast<const float *>(op->src[5]->data), static_cast<const int32_t *>(op->src[6]->data),
        static_cast<float *>(op->data), static_cast<float *>(op->src[7]->data),
        op->src[5]->ne[0], H, 1, neqk1, init_fastdiv_values(neqk1), 1.0f / sqrtf((float) S_v));
    CUDA_CHECK(cudaGetLastError());
}

//
// prepn
//

// qwen35_prep1 for N tokens of one sequence, token by token; ssm_conv_f32<true> runs the same sums over the tokens.
// Slot s of the cell gets the conv state of s tokens before the end of the batch, one slot per token.
// past can be one of these rows: a thread reads its part of the row before it writes it
template <int N>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void q35moe_prepn(
        const float * qkv_ptr, const float * beta, const float * alpha, const float * weight_ptr, const float * dt, const float * a,
        const float * states, const int32_t * ids, float * cell_ptr, float * dst_ptr,
        const int64_t state_stride, const int64_t slot_stride, const int n_channels, const int n_norm, const int n_heads,
        const int ncols, const float eps, const float scale_out) {
    constexpr int n_conv = ggml_qwen35_n_conv;
    constexpr int vec    = ggml_qwen35_head_dim / WARP_SIZE;
    const float * GGML_CUDA_RESTRICT qkv    = qkv_ptr;
    const float * GGML_CUDA_RESTRICT weight = weight_ptr;
    float       * GGML_CUDA_RESTRICT dst    = dst_ptr;

    const int     c0   = blockIdx.x*ggml_qwen35_head_dim + threadIdx.x*vec;
    const float * past = states + ids[0]*state_stride + (int64_t) c0*(n_conv - 1);
    float       * cell = cell_ptr + (int64_t) c0*(n_conv - 1);

    float x[vec][n_conv - 1 + N];
    float w[vec][n_conv];
#pragma unroll
    for (int i = 0; i < vec; i++) {
#pragma unroll
        for (int j = 0; j < n_conv; j++) {
            w[i][j] = weight[(c0 + i)*n_conv + j];
        }
#pragma unroll
        for (int j = 0; j < n_conv - 1; j++) {
            x[i][j] = past[i*(n_conv - 1) + j];
        }
#pragma unroll
        for (int t = 0; t < N; t++) {
            x[i][n_conv - 1 + t] = qkv[t*n_channels + c0 + i];
        }
    }

    static_assert(vec == 4, "the norm reads one float4 per thread");
#pragma unroll
    for (int t = 0; t < N; t++) {
        float y[vec];
#pragma unroll
        for (int i = 0; i < vec; i++) {
            const float b = 0.0f;
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < n_conv; j++) {
                sumf += x[i][t + j] * w[i][j];
            }
            sumf += b;
            y[i] = ggml_cuda_op_silu_single(sumf);
        }

        float4 r = make_float4(y[0], y[1], y[2], y[3]);
        if ((int) blockIdx.x*ggml_qwen35_head_dim < n_norm) {
            const float4 xs = r;
            float tmp = 0.0f;
            tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
            tmp = warp_reduce_sum(tmp);
            const float scale = rsqrtf(tmp / ncols + eps);

            r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
            r.x *= scale_out;
            r.y *= scale_out;
            r.z *= scale_out;
            r.w *= scale_out;
        }
        dst[t*n_channels + c0 + 0] = r.x;
        dst[t*n_channels + c0 + 1] = r.y;
        dst[t*n_channels + c0 + 2] = r.z;
        dst[t*n_channels + c0 + 3] = r.w;
    }

#pragma unroll
    for (int s = 0; s < N; s++) {
#pragma unroll
        for (int i = 0; i < vec; i++) {
#pragma unroll
            for (int j = 0; j < n_conv - 1; j++) {
                cell[s*slot_stride + i*(n_conv - 1) + j] = x[i][N - s + j];
            }
        }
    }

    if (blockIdx.x == 0) {
        float * dst_beta = dst + N*n_channels;
        float * dst_gate = dst_beta + N*n_heads;
        for (int t = 0; t < N; t++) {
            for (int h = threadIdx.x; h < n_heads; h += WARP_SIZE) {
                dst_beta[t*n_heads + h] = 1.0f / (1.0f + expf(-beta[t*n_heads + h]));

                const float v = alpha[t*n_heads + h] + dt[h];
                dst_gate[t*n_heads + h] = ((v > 20.0f) ? v : log1pf(expf(v))) * a[h];
            }
        }
    }
}

static bool q35moe_prepn_supported(int device, const ggml_tensor * op) {
    return qwen35_decode_device(device) && ggml_q35moe_prepn_enabled() && ggml_q35moe_is_prepn(op) &&
        qwen35_weight_on_device(device, op->src[3]) && qwen35_weight_on_device(device, op->src[4]) && qwen35_weight_on_device(device, op->src[5]) &&
        qwen35_state_on_device(device, op->src[6]) && qwen35_state_on_device(device, op->src[8]);
}

template <int N>
static void q35moe_prepn_launch_n(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const int     n_channels  = op->src[0]->ne[0];
    const int     n_heads     = ggml_nelements(op->src[4]);
    const int64_t slot_stride = ggml_get_op_params_i32(op, ggml_q35moe_prepn_slot_slot_rows)*op->src[6]->ne[0];
    q35moe_prepn<N><<<n_channels/ggml_qwen35_head_dim, WARP_SIZE, 0, ctx.stream()>>>(
        static_cast<const float *>(op->src[0]->data), static_cast<const float *>(op->src[1]->data), static_cast<const float *>(op->src[2]->data),
        static_cast<const float *>(op->src[3]->data), static_cast<const float *>(op->src[4]->data), static_cast<const float *>(op->src[5]->data),
        static_cast<const float *>(op->src[6]->data), static_cast<const int32_t *>(op->src[7]->data), static_cast<float *>(op->src[8]->data),
        static_cast<float *>(op->data), op->src[6]->ne[0], slot_stride, n_channels, ggml_get_op_params_i32(op, ggml_qwen35_prep1_slot_n_norm), n_heads,
        ggml_qwen35_head_dim, ggml_get_op_params_f32(op, ggml_qwen35_prep1_slot_eps), ggml_get_op_params_f32(op, ggml_qwen35_prep1_slot_scale));
    CUDA_CHECK(cudaGetLastError());
}

static void q35moe_prepn_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    static_assert(ggml_q35moe_few_max == 5, "one kernel per token count");
    switch (op->src[0]->ne[1]) {
        case 2: q35moe_prepn_launch_n<2>(ctx, op); break;
        case 3: q35moe_prepn_launch_n<3>(ctx, op); break;
        case 4: q35moe_prepn_launch_n<4>(ctx, op); break;
        case 5: q35moe_prepn_launch_n<5>(ctx, op); break;
        default: GGML_ABORT("fatal error");
    }
}

//
// gdnn
//

// gated_delta_net_cuda<S_v, false, true> of one sequence, statement by statement; the input state is row ids[0] of states.
// Slot s of the cell gets the state of s tokens before the end of the batch, one slot per token.
// The input state can be one of these rows: a thread reads its part of the state before it writes it
template <int S_v, int cols_per_warp>
__launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
static __global__ void q35moe_gdnn(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * states, const int32_t * ids, float * dst, float * state,
        const int64_t state_stride, const int64_t slot_stride, const int64_t H, const int64_t n_tokens,
        const int64_t sq2, const int64_t sv2, const uint3 neqk1_magic, const float scale) {
    const uint32_t h_idx = blockIdx.x;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);

    float * attn_data = dst;

    const float * curr_state = states + ids[0] * state_stride + h_idx * S_v * S_v + col * S_v;
    state += h_idx * S_v * S_v;
    attn_data += h_idx * S_v;

    float s_shard[rows_per_lane];

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + t * sq2 + iq1 * S_v;
        const float * k_t = k + t * sq2 + iq1 * S_v;
        const float * v_t = v + t * sv2 + h_idx * S_v;

        const int64_t gb_offset = t * H + h_idx;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset;

        const float beta_val = *beta_t;

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        const float g_val = expf(*g_t);

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }

        float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

        if (lane_in_col == 0) {
            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;

        const int target_slot = (int) n_tokens - 1 - t;
        float * snapshot = state + target_slot * slot_stride;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            snapshot[col * S_v + i] = s_shard[r];
        }
    }
}

// qwen35moe.gdnn_pre: q35moe_gdnn for N tokens with the inputs of all tokens in registers before the token loop.
// The statements of a token and their order are those of q35moe_gdnn; only the loads move.
template <int S_v, int cols_per_warp, int N>
__launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
static __global__ void q35moe_gdnn_pre(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * states, const int32_t * ids, float * dst, float * state,
        const int64_t state_stride, const int64_t slot_stride, const int64_t H,
        const int64_t sq2, const int64_t sv2, const uint3 neqk1_magic, const float scale) {
    const uint32_t h_idx = blockIdx.x;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);

    float * attn_data = dst;

    const float * curr_state = states + ids[0] * state_stride + h_idx * S_v * S_v + col * S_v;
    state += h_idx * S_v * S_v;
    attn_data += h_idx * S_v;

    float s_shard[rows_per_lane];

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    float k_reg[N][rows_per_lane];
    float q_reg[N][rows_per_lane];
    float v_val[N];
    float g_in[N];
    float beta_in[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float * q_t = q + t * sq2 + iq1 * S_v;
        const float * k_t = k + t * sq2 + iq1 * S_v;
        const float * v_t = v + t * sv2 + h_idx * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[t][r] = k_t[i];
            q_reg[t][r] = q_t[i];
        }
        v_val[t]   = v_t[col];
        g_in[t]    = g[t * H + h_idx];
        beta_in[t] = beta[t * H + h_idx];
    }

#pragma unroll
    for (int t = 0; t < N; t++) {
        const float beta_val = beta_in[t];

        const float g_val = expf(g_in[t]);

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[t][r];
        }
        float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

        float delta_col = (v_val[t] - g_val * kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]  = g_val * s_shard[r] + k_reg[t][r] * delta_col;
            attn_partial += s_shard[r] * q_reg[t][r];
        }

        float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

        if (lane_in_col == 0) {
            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;

        const int target_slot = N - 1 - t;
        float * snapshot = state + target_slot * slot_stride;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            snapshot[col * S_v + i] = s_shard[r];
        }
    }
}

static bool q35moe_gdnn_supported(int device, const ggml_tensor * op) {
    return qwen35_decode_device(device) && ggml_q35moe_gdnn_enabled() && ggml_q35moe_is_gdnn(op) &&
        qwen35_state_on_device(device, op->src[5]) && qwen35_state_on_device(device, op->src[7]);
}

static void q35moe_gdnn_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    constexpr int S_v           = ggml_qwen35_head_dim;
    constexpr int cols_per_warp = 4; // gdn_cols_per_warp
    const int64_t H           = op->src[2]->ne[1];
    const int64_t neqk1       = op->src[0]->ne[1];
    const int64_t slot_stride = ggml_get_op_params_i32(op, ggml_q35moe_gdnn_slot_slot_rows)*op->src[5]->ne[0];

    // the block layout of launch_gated_delta_net
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nsm       = ggml_cuda_info().devices[ctx.device].nsm;
    int num_warps = 4;
    while (num_warps > 1 && H*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    const dim3 grid_dims(H, 1, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    const dim3 block_dims(warp_size, num_warps, 1);

    const int64_t n_tokens = op->src[2]->ne[2];
    if (ggml_kperf_on(GGML_KPERF_QWEN35MOE_GDNN_PRE) && n_tokens >= 2 && n_tokens <= ggml_q35moe_few_max) {
        static_assert(ggml_q35moe_few_max == 5, "one kernel per token count");
        decltype(&q35moe_gdnn_pre<S_v, cols_per_warp, 2>) kernel = nullptr;
        switch (n_tokens) {
            case 2: kernel = q35moe_gdnn_pre<S_v, cols_per_warp, 2>; break;
            case 3: kernel = q35moe_gdnn_pre<S_v, cols_per_warp, 3>; break;
            case 4: kernel = q35moe_gdnn_pre<S_v, cols_per_warp, 4>; break;
            default: kernel = q35moe_gdnn_pre<S_v, cols_per_warp, 5>; break;
        }
        if (ggml_kperf_trace(GGML_KPERF_QWEN35MOE_GDNN_PRE)) {
            GGML_LOG_INFO("Q35MOE_GDNN_PRE_HIT tokens=%lld heads=%lld\n", (long long) n_tokens, (long long) H);
        }
        kernel<<<grid_dims, block_dims, 0, ctx.stream()>>>(
            static_cast<const float *>(op->src[0]->data), static_cast<const float *>(op->src[1]->data), static_cast<const float *>(op->src[2]->data),
            static_cast<const float *>(op->src[3]->data), static_cast<const float *>(op->src[4]->data),
            static_cast<const float *>(op->src[5]->data), static_cast<const int32_t *>(op->src[6]->data),
            static_cast<float *>(op->data), static_cast<float *>(op->src[7]->data),
            op->src[5]->ne[0], slot_stride, H,
            (int64_t) (op->src[0]->nb[2]/sizeof(float)), (int64_t) (op->src[2]->nb[2]/sizeof(float)), init_fastdiv_values(neqk1),
            1.0f / sqrtf((float) S_v));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    q35moe_gdnn<S_v, cols_per_warp><<<grid_dims, block_dims, 0, ctx.stream()>>>(
        static_cast<const float *>(op->src[0]->data), static_cast<const float *>(op->src[1]->data), static_cast<const float *>(op->src[2]->data),
        static_cast<const float *>(op->src[3]->data), static_cast<const float *>(op->src[4]->data),
        static_cast<const float *>(op->src[5]->data), static_cast<const int32_t *>(op->src[6]->data),
        static_cast<float *>(op->data), static_cast<float *>(op->src[7]->data),
        op->src[5]->ne[0], slot_stride, H, op->src[2]->ne[2],
        (int64_t) (op->src[0]->nb[2]/sizeof(float)), (int64_t) (op->src[2]->nb[2]/sizeof(float)), init_fastdiv_values(neqk1),
        1.0f / sqrtf((float) S_v));
    CUDA_CHECK(cudaGetLastError());
}

//
// post1
//

// one warp per head, a thread owns 4 values in a row
// the norm follows rms_norm_f32<32, 1, true, true>, the gate follows unary_gated_op_kernel<op_silu>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void qwen35_post1(const float * x_ptr, const float * w_ptr, const float * z_ptr, float * dst_ptr, const int ncols, const float eps) {
    constexpr int vec = ggml_qwen35_head_dim / WARP_SIZE;
    static_assert(vec == 4, "the norm reads one float4 per thread");
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    const float * GGML_CUDA_RESTRICT w   = w_ptr;
    const float * GGML_CUDA_RESTRICT z   = z_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;

    const int i0 = blockIdx.x*ggml_qwen35_head_dim + threadIdx.x*vec;

    const float4 xs = make_float4(x[i0 + 0], x[i0 + 1], x[i0 + 2], x[i0 + 3]);
    float tmp = 0.0f;
    tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    tmp = warp_reduce_sum(tmp);
    const float scale = rsqrtf(tmp / ncols + eps);

    float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
    const float4 m = make_float4(w[threadIdx.x*vec + 0], w[threadIdx.x*vec + 1], w[threadIdx.x*vec + 2], w[threadIdx.x*vec + 3]);
    r.x *= m.x;
    r.y *= m.y;
    r.z *= m.z;
    r.w *= m.w;

    dst[i0 + 0] = ggml_cuda_op_silu_single(z[i0 + 0]) * r.x;
    dst[i0 + 1] = ggml_cuda_op_silu_single(z[i0 + 1]) * r.y;
    dst[i0 + 2] = ggml_cuda_op_silu_single(z[i0 + 2]) * r.z;
    dst[i0 + 3] = ggml_cuda_op_silu_single(z[i0 + 3]) * r.w;
}

static bool qwen35_post1_supported(int device, const ggml_tensor * op) {
    return qwen35_decode_device(device) && (ggml_qwen35_post1_enabled() || ggml_q35moe_post1_enabled() || ggml_q35moe_postn_enabled()) &&
        ggml_qwen35_is_post1(op) &&
        qwen35_weight_on_device(device, op->src[1]);
}

static void qwen35_post1_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const int n = ggml_nelements(op->src[0]);
    qwen35_post1<<<n/ggml_qwen35_head_dim, WARP_SIZE, 0, ctx.stream()>>>(
        static_cast<const float *>(op->src[0]->data), static_cast<const float *>(op->src[1]->data), static_cast<const float *>(op->src[2]->data),
        static_cast<float *>(op->data), ggml_qwen35_head_dim, ggml_get_op_params_f32(op, ggml_qwen35_post1_slot_eps));
    CUDA_CHECK(cudaGetLastError());
}

//
// mvf
//

// the block size of mul_mat_vec_f for the row lengths that this op takes
constexpr int q35moe_mvf_block_size = 256;

struct q35moe_mvf_weights {
    const float * w[ggml_q35moe_mvf_max];
    int           row_begin[ggml_q35moe_mvf_max]; // first row of matrix j in the result, INT32_MAX for an unused slot
};

// one block per row of one of the matrices, the product follows mul_mat_vec_f<float, float, 1, 256> without fusion
static __global__ void q35moe_mvf(const q35moe_mvf_weights weights, const float * y_ptr, float * dst_ptr, const int ncols2, const int stride_row) {
    constexpr int block_size = q35moe_mvf_block_size;
    constexpr int warp_size  = WARP_SIZE;
    const float * GGML_CUDA_RESTRICT y   = y_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int tid = threadIdx.x;

    int j = 0;
#pragma unroll
    for (int i = 1; i < ggml_q35moe_mvf_max; i++) {
        j += (int) blockIdx.x >= weights.row_begin[i];
    }
    const int row = blockIdx.x - weights.row_begin[j];
    const float * GGML_CUDA_RESTRICT x = weights.w[j] + (int64_t) row*stride_row;

    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    const float2 * y2 = (const float2 *) y;
    const float2 * x2 = (const float2 *) x;

    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }

    sumf = warp_reduce_sum<warp_size>(sumf);
    buf_iw[tid/warp_size] = sumf;
    __syncthreads();
    if (tid >= warp_size) {
        return;
    }
    sumf = buf_iw[tid];
    sumf = warp_reduce_sum<warp_size>(sumf);

    if (tid == 0) {
        dst[blockIdx.x] = sumf;
    }
}

// qwen35moe.ban: one block per stacked row, ncols columns; the product follows mul_mat_vec_f<float, float, ncols, 256> without fusion:
// the same column loop, the reduction of each column through warp_reduce_sum, the shared buffer and warp_reduce_sum, and the write by thread j
template <int ncols>
static __global__ void q35moe_mvfn(const q35moe_mvf_weights weights, const float * y_ptr, float * dst_ptr, const int ncols2, const int stride_row,
        const int stride_col_y2) {
    constexpr int block_size = q35moe_mvf_block_size;
    constexpr int warp_size  = WARP_SIZE;
    const float * GGML_CUDA_RESTRICT y   = y_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int tid = threadIdx.x;

    int j = 0;
#pragma unroll
    for (int i = 1; i < ggml_q35moe_mvf_max; i++) {
        j += (int) blockIdx.x >= weights.row_begin[i];
    }
    const int row = blockIdx.x - weights.row_begin[j];
    const float * GGML_CUDA_RESTRICT x = weights.w[j] + (int64_t) row*stride_row;
    // the [rows of matrix j, ncols] block of matrix j
    int rows_j = 0;
    int base   = 0;
#pragma unroll
    for (int i = 0; i < ggml_q35moe_mvf_max; i++) {
        const int b = weights.row_begin[i];
        const int e = i + 1 < ggml_q35moe_mvf_max && weights.row_begin[i + 1] != INT32_MAX ? weights.row_begin[i + 1] : (int) gridDim.x;
        if (i == j) {
            rows_j = e - b;
            base   = b*ncols;
        }
    }
    dst += base;

    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf[ncols] = {0.0f};

    const float2 * y2 = (const float2 *) y;
    const float2 * x2 = (const float2 *) x;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const float2 tmpy = y2[c*stride_col_y2 + col2];
            ggml_cuda_mad(sumf[c], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[c], tmpx.y, tmpy.y);
        }
    }

#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        sumf[c] = warp_reduce_sum<warp_size>(sumf[c]);
        buf_iw[tid/warp_size] = sumf[c];
        __syncthreads();
        if (tid < warp_size) {
            sumf[c] = buf_iw[tid];
            sumf[c] = warp_reduce_sum<warp_size>(sumf[c]);
        }
        __syncthreads();
    }

    if (tid >= ncols) {
        return;
    }
    dst[tid*rows_j + row] = sumf[tid];
}

// loop turns of a thread of mul_mat_vec_f with this block size
static int64_t q35moe_mvf_niter(int64_t ncols, int block_size) {
    return (ncols + 2*block_size - 1) / (2*block_size);
}

static bool q35moe_mvf_supported(int device, const ggml_tensor * op) {
    if (!qwen35_decode_device(device) || !ggml_q35moe_ba1_enabled() || !ggml_q35moe_is_mvf(op)) {
        return false;
    }
    // launch_mul_mat_vec_f_cuda takes the biggest block only when the next smaller one needs more loop turns
    const int64_t ncols = op->src[0]->ne[0];
    if (q35moe_mvf_niter(ncols, q35moe_mvf_block_size) >= q35moe_mvf_niter(ncols, q35moe_mvf_block_size - WARP_SIZE)) {
        return false;
    }
    const auto & info = ggml_cuda_info().devices[device];
    for (int j = 1; j < GGML_MAX_SRC && op->src[j]; ++j) {
        // only the products that ggml_cuda_mul_mat gives to ggml_cuda_mul_mat_vec_f, with the weight in memory of this device
        if (!qwen35_weight_on_device(device, op->src[j]) || !ggml_cuda_should_use_mmvf(op->src[j], info.cc, info.warp_size, 1)) {
            return false;
        }
    }
    return true;
}

static bool q35moe_mvfn_supported(int device, const ggml_tensor * op) {
    if (!qwen35_decode_device(device) || !ggml_q35moe_ban_enabled() || !ggml_q35moe_is_mvfn(op)) {
        return false;
    }
    // launch_mul_mat_vec_f_cuda takes the biggest block only when the next smaller one needs more loop turns
    const int64_t ncols = op->src[0]->ne[0];
    if (q35moe_mvf_niter(ncols, q35moe_mvf_block_size) >= q35moe_mvf_niter(ncols, q35moe_mvf_block_size - WARP_SIZE)) {
        return false;
    }
    const auto & info = ggml_cuda_info().devices[device];
    for (int j = 1; j < GGML_MAX_SRC && op->src[j]; ++j) {
        // only the products that ggml_cuda_mul_mat gives to ggml_cuda_mul_mat_vec_f, with the weight in memory of this device
        if (!qwen35_weight_on_device(device, op->src[j]) || !ggml_cuda_should_use_mmvf(op->src[j], info.cc, info.warp_size, op->src[0]->ne[1])) {
            return false;
        }
    }
    return true;
}

static void q35moe_mvfn_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    q35moe_mvf_weights weights;
    int n_rows = 0;
    for (int j = 0; j < ggml_q35moe_mvf_max; ++j) {
        const ggml_tensor * w = op->src[j + 1];
        weights.w[j]         = w ? static_cast<const float *>(w->data) : nullptr;
        weights.row_begin[j] = w ? n_rows : INT32_MAX;
        n_rows += w ? (int) w->ne[1] : 0;
    }
    if (ggml_kperf_trace(GGML_KPERF_QWEN35MOE_BAN)) {
        GGML_LOG_INFO("Q35MOE_BAN_HIT rows=%d tokens=%lld\n", n_rows, (long long) x->ne[1]);
    }
    const int ncols2 = (int) (x->ne[0]/2);
    const int stride_col_y2 = (int) (x->nb[1]/sizeof(float2));
    static_assert(ggml_q35moe_few_max == 5, "one kernel per column count");
    switch (x->ne[1]) {
        case 2: q35moe_mvfn<2><<<n_rows, q35moe_mvf_block_size, 0, ctx.stream()>>>(weights, static_cast<const float *>(x->data), static_cast<float *>(op->data), ncols2, (int) x->ne[0], stride_col_y2); break;
        case 3: q35moe_mvfn<3><<<n_rows, q35moe_mvf_block_size, 0, ctx.stream()>>>(weights, static_cast<const float *>(x->data), static_cast<float *>(op->data), ncols2, (int) x->ne[0], stride_col_y2); break;
        case 4: q35moe_mvfn<4><<<n_rows, q35moe_mvf_block_size, 0, ctx.stream()>>>(weights, static_cast<const float *>(x->data), static_cast<float *>(op->data), ncols2, (int) x->ne[0], stride_col_y2); break;
        case 5: q35moe_mvfn<5><<<n_rows, q35moe_mvf_block_size, 0, ctx.stream()>>>(weights, static_cast<const float *>(x->data), static_cast<float *>(op->data), ncols2, (int) x->ne[0], stride_col_y2); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

static void q35moe_mvf_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    q35moe_mvf_weights weights;
    int n_rows = 0;
    for (int j = 0; j < ggml_q35moe_mvf_max; ++j) {
        const ggml_tensor * w = op->src[j + 1];
        weights.w[j]         = w ? static_cast<const float *>(w->data) : nullptr;
        weights.row_begin[j] = w ? n_rows : INT32_MAX;
        n_rows += w ? (int) w->ne[1] : 0;
    }
    q35moe_mvf<<<n_rows, q35moe_mvf_block_size, 0, ctx.stream()>>>(
        weights, static_cast<const float *>(x->data), static_cast<float *>(op->data), (int) (x->ne[0]/2), (int) x->ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

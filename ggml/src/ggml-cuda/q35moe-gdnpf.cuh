#pragma once

#include "common.cuh"
#include "ggml-qwen35-decode.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private kernel for gfx1103 only"
#endif

// qwen35moe.gdnpf: gated delta net of a prompt batch of qwen35moe (heads of 128, scalar gate, one state slot).
// The kernel is gated_delta_net_cuda<128, false, true, 4>, statement by statement, so the results have the same bits.
// That instance stores the state in the token loop. It loads the values of a token before it needs them, the instance
// without snapshots waits for two loads in the middle of a token. The lane exchange of the two sums is a DPP move here.

// lane i of a row of 16 lanes gets the value of lane i ^ mask (row_xmask); all lanes of the warp must be active
template <int mask>
static __device__ __forceinline__ float q35moe_gdnpf_xor(const float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x160 + mask, 0xF, 0xF, true));
}

// warp_reduce_sum<8> without ds_bpermute: the same partner lanes in the same order
static __device__ __forceinline__ float q35moe_gdnpf_reduce8(float x) {
    x += q35moe_gdnpf_xor<4>(x);
    x += q35moe_gdnpf_xor<2>(x);
    x += q35moe_gdnpf_xor<1>(x);
    return x;
}

constexpr int q35moe_gdnpf_S_v           = 128;
constexpr int q35moe_gdnpf_cols_per_warp = 4; // gdn_cols_per_warp

// qk_groups (qwen35moe.gdnpf_order): 4 warps per block, H = 2 x the q/k heads, grid (8, sequences, 8 column slices x H/8).
// A group of 64 blocks takes 4 q/k heads and the 2 value heads of each: the 16 blocks that read the same q and k rows run together.
template <bool qk_groups>
__launch_bounds__(ggml_cuda_get_physical_warp_size() * 4, 2)
static __global__ void q35moe_gdnpf(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * curr_state, float * dst, float * state,
        int64_t H, int64_t n_tokens,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K) {
    uint32_t       h_idx    = blockIdx.x;
    uint32_t       slice    = blockIdx.z;
    const uint32_t sequence = blockIdx.y;
    if constexpr (qk_groups) {
        constexpr uint32_t n_slices = q35moe_gdnpf_S_v / (q35moe_gdnpf_cols_per_warp * 4);
        const uint32_t grp  = blockIdx.z / n_slices;
        const uint32_t half = gridDim.x / 2;
        slice = blockIdx.z % n_slices;
        h_idx = grp * half + blockIdx.x % half + (blockIdx.x / half) * neqk1_magic.z;
    }

    constexpr int S_v           = q35moe_gdnpf_S_v;
    constexpr int cols_per_warp = q35moe_gdnpf_cols_per_warp;
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == 32, "the lane exchange is written for warps of 32 lanes");
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(lanes_per_col == 8, "q35moe_gdnpf_reduce8 takes 8 lanes");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col = (slice * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
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
        float kv_col = q35moe_gdnpf_reduce8(kv_shard);

        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }

        float attn_col = q35moe_gdnpf_reduce8(attn_partial);

        if (lane_in_col == 0) {
            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;

        // slot s gets the state of s tokens before the end; the launch passes one slot
        const int target_slot = (int) n_tokens - 1 - t;
        if (target_slot >= 0 && target_slot < K) {
            float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                curr_state[col * S_v + i] = s_shard[r];
            }
        }
    }
}

static bool q35moe_gdnpf_supported(const int device, const ggml_tensor * gdn) {
    const auto & info = ggml_cuda_info().devices[device];
    return info.cc == GGML_CUDA_CC_OFFSET_AMD + 0x1103 && info.warp_size == 32 && ggml_q35moe_is_gdnpf(gdn);
}

// the arguments and the block layout of launch_gated_delta_net
static void q35moe_gdnpf_launch(
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d, const int64_t H, const int64_t n_tokens, const int64_t n_seqs,
        const int64_t sq1, const int64_t sq2, const int64_t sq3, const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3, const int64_t neqk1, const int64_t rq3, const float scale, cudaStream_t stream) {
    constexpr int S_v           = q35moe_gdnpf_S_v;
    constexpr int cols_per_warp = q35moe_gdnpf_cols_per_warp;
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    const dim3 grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    const dim3 block_dims(32, num_warps, 1);

    static bool logged_use = false;
    if (!logged_use) {
        logged_use = true;
        GGML_LOG_INFO("%s: qwen35moe.gdnpf is used, heads = %lld, tokens = %lld, sequences = %lld\n", __func__,
            (long long) H, (long long) n_tokens, (long long) n_seqs);
    }

    // qwen35moe.gdnpf_order: value heads h and h + neqk1 read the same q and k rows; groups of 8 heads keep them in L2
    const bool qk_groups = ggml_kperf_on(GGML_KPERF_QWEN35MOE_GDNPF_ORDER) && num_warps == 4 && H == 2 * neqk1 && neqk1 % 4 == 0;

    // one slot: the state after the last token
    if (qk_groups) {
        if (ggml_kperf_trace(GGML_KPERF_QWEN35MOE_GDNPF_ORDER)) {
            GGML_LOG_INFO("Q35MOE_GDNPF_ORDER_HIT heads=%lld tokens=%lld sequences=%lld\n", (long long) H, (long long) n_tokens, (long long) n_seqs);
        }
        const dim3 grid_groups(8, n_seqs, grid_dims.z * (H / 8));
        q35moe_gdnpf<true><<<grid_groups, block_dims, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, init_fastdiv_values(neqk1), init_fastdiv_values(rq3), scale, 0, 1);
    } else {
        q35moe_gdnpf<false><<<grid_dims, block_dims, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, init_fastdiv_values(neqk1), init_fastdiv_values(rq3), scale, 0, 1);
    }
    CUDA_CHECK(cudaGetLastError());
}

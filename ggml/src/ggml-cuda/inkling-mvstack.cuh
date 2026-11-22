#pragma once
#include "ggml-inkling-mvstack.h"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <climits>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1103__)
#error "Compile this private MVSTACK experiment for gfx1103 only"
#endif

struct inkling_mvstack_args {
    const void * w[ggml_inkling_mvstack_max];
    int          first[ggml_inkling_mvstack_max]; // first stacked row of each matrix, INT_MAX when unused
};

// one warp per row; the arithmetic of a row follows mul_mat_vec_q<GGML_TYPE_Q8_0, 1, false> with the RDNA2 parameters step by step
__launch_bounds__(32, 1)
static __global__ void inkling_mvstack(const inkling_mvstack_args args, const void * vy_ptr, float * dst_ptr, const int blocks_per_row_x) {
    constexpr int qk        = QK8_0;
    constexpr int qi        = QI8_0;
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = 32;

    // the matrix of this row
    int m = 0;
#pragma unroll
    for (int j = 1; j < ggml_inkling_mvstack_max; j++) {
        if ((int) blockIdx.x >= args.first[j]) {
            m = j;
        }
    }
    const void * GGML_CUDA_RESTRICT vx  = args.w[m];
    const void * GGML_CUDA_RESTRICT vy  = vy_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;

    const int tid             = threadIdx.x;
    const int blocks_per_iter = vdr * warp_size / qi;

    float tmp = 0.0f;

    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = ((int) blockIdx.x - args.first[m])*blocks_per_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

        tmp += vec_dot_q8_0_q8_1(vx, &y[kby], kbx_offset + kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);

    if (threadIdx.x == 0) {
        dst[blockIdx.x] = tmp;
    }
}

static bool inkling_mvstack_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info().devices[device];
    if (info.cc != GGML_CUDA_CC_OFFSET_AMD + 0x1103 || info.warp_size != 32 || !ggml_inkling_mvstack_enabled() || !ggml_inkling_is_mvstack(op)) {
        return false;
    }
    // weights of this device only: no padding to clear, no copy by the scheduler
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);
    for (int j = 1; j < GGML_MAX_SRC && op->src[j]; ++j) {
        const ggml_backend_buffer_t buf = op->src[j]->buffer;
        if (!buf || buf->buft != buft || ggml_backend_buffer_get_usage(buf) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return false;
        }
    }
    return true;
}

static void inkling_mvstack_launch(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    const ggml_tensor * x      = op->src[0];
    cudaStream_t        stream = ctx.stream();

    // the same quantized input as in ggml_cuda_mul_mat_vec_q
    const int64_t ne10        = x->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> x_q8_1(ctx.pool(), ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = x->nb[1] / sizeof(float);
        const int64_t s12 = x->nb[2] / sizeof(float);
        const int64_t s13 = x->nb[3] / sizeof(float);
        quantize_row_q8_1_cuda(static_cast<const float *>(x->data), x_q8_1.get(), ne10, s11, s12, s13, ne10_padded, 1, 1, 1, stream);
    }

    inkling_mvstack_args args;
    int n_rows = 0;
    for (int j = 0; j < ggml_inkling_mvstack_max; ++j) {
        const ggml_tensor * w = op->src[j + 1];
        args.w[j]     = w ? w->data : nullptr;
        args.first[j] = w ? n_rows : INT_MAX;
        n_rows       += w ? w->ne[1] : 0;
    }
    inkling_mvstack<<<n_rows, 32, 0, stream>>>(args, x_q8_1.get(), static_cast<float *>(op->data), (int) (ne10 / QK8_0));
    CUDA_CHECK(cudaGetLastError());
}

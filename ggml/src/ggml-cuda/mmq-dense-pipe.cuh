#pragma once

// Include from mmq.cu after mmq.cuh.

// hip.mmq_dense_pipe: dense Q8_0 products on gfx1103 with tiles of 64 rows x 128 columns in 8 warps of 16 x 64 that take K in stages
// of 128 values. Each output gets the same values in the same order as in mul_mat_q: for each block of 32 values in K order
// sum += float(C)*dA*dB, C being the int32 product of the block. Only the work layout differs:
//   - the loads of the next stage go to registers before the products of the current stage;
//   - a stage takes 27 KiB of LDS instead of 38 KiB, so 4 tiles fit in a WGP instead of 3;
//   - the row and stage parts of the load addresses are scalar;
//   - C starts at the bits of 1.5*2^23 in the WMMA and float(C) is as_float(C) - 1.5*2^23, exact for |C| < 2^22 (32 products of two
//     int8 are at most 2^19 in size): a subtraction that pairs with the multiplications (VOPD) instead of v_cvt_f32_i32.
// With dbuf the kernel has two LDS buffers and one barrier per stage, for products of few tiles (see mmq_dense_pipe_launch).

#if defined(GGML_USE_HIP)

#include <type_traits>

static constexpr int   mmq_dense_pipe_J       = 128;
static constexpr int   mmq_dense_pipe_xs      = 36;           // x tile row stride (ints): 32 qs + 4 float scales
static constexpr int   mmq_dense_pipe_ys      = MMQ_TILE_Y_K; // 4 float scales + 32 qs per column
static constexpr int   mmq_dense_pipe_magic_i = 0x4B400000;
static constexpr float mmq_dense_pipe_magic_f = 12582912.0f;

template <int nwr, int nwc>
struct mmq_dense_pipe_regs {
    static constexpr int nt  = 32*nwr*nwc;
    static constexpr int I   = 16*nwr;
    static constexpr int nxq = I / (nwr*nwc);
    static constexpr int nxd = (I*4 + nt - 1) / nt;
    static constexpr bool y2 = mmq_dense_pipe_J*mmq_dense_pipe_ys % (2*nt) == 0;
    static constexpr int ny  = y2 ? mmq_dense_pipe_J*mmq_dense_pipe_ys / (2*nt) : mmq_dense_pipe_J*mmq_dense_pipe_ys / nt;
    static_assert(mmq_dense_pipe_J*mmq_dense_pipe_ys % nt == 0, "y tile");
    typedef typename std::conditional<y2, int2, int>::type ytype;

    int   xq[nxq];
    half  xd[nxd];
    ytype yv[ny];
};

// Byte offsets of the loads of a thread, the same for every stage. The row and stage parts of an address are uniform (scalar), so
// each load is one instruction with a scalar base and this offset.
struct mmq_dense_pipe_offsets {
    uint32_t xq;
    uint32_t xd;
    uint32_t y;
};

template <int nwr, int nwc>
static __device__ __forceinline__ mmq_dense_pipe_offsets mmq_dense_pipe_get_offsets(const int stride_row_x) {
    typedef mmq_dense_pipe_regs<nwr, nwc> R;
    const int w    = threadIdx.y;
    const int lane = threadIdx.x;
    const int t    = w*32 + lane;
    mmq_dense_pipe_offsets o;
    o.xq = (w*stride_row_x + lane/QI8_0)*sizeof(block_q8_0) + offsetof(block_q8_0, qs) + (lane % QI8_0)*sizeof(int);
    o.xd = ((t/4)*stride_row_x + t%4)*sizeof(block_q8_0) + offsetof(block_q8_0, d);
    o.y  = t*sizeof(typename R::ytype);
    return o;
}

// stage s: the blocks 4*s .. 4*s + 3 of the I rows from row0, and the y blocks of 128 values of the 128 columns of the tile
template <int nwr, int nwc>
static __device__ __forceinline__ void mmq_dense_pipe_load(mmq_dense_pipe_regs<nwr, nwc> & r, const char * __restrict__ x,
        const int * __restrict__ y, const mmq_dense_pipe_offsets & o, const int row0, const int stride_row_x, const int s, const int ncols_y) {
    typedef mmq_dense_pipe_regs<nwr, nwc> R;
    const char * xs = x + (size_t) s*4*sizeof(block_q8_0);
#pragma unroll
    for (int p = 0; p < R::nxq; ++p) {
        const char * xp = xs + (size_t) (row0 + p*nwr*nwc)*stride_row_x*sizeof(block_q8_0);
        r.xq[p] = get_int_b2(xp + o.xq, 0);
    }
#pragma unroll
    for (int p = 0; p < R::nxd; ++p) {
        if (R::I*4 % R::nt == 0 || p*R::nt + (int) (threadIdx.y*32 + threadIdx.x) < R::I*4) {
            const char * xp = xs + (size_t) (row0 + p*R::nt/4)*stride_row_x*sizeof(block_q8_0);
            r.xd[p] = *(const half *) (xp + o.xd);
        }
    }
    const char * ys = (const char *) (y + (size_t) s*ncols_y*mmq_dense_pipe_ys);
#pragma unroll
    for (int p = 0; p < R::ny; ++p) {
        r.yv[p] = *(const typename R::ytype *) (ys + p*R::nt*sizeof(typename R::ytype) + o.y);
    }
}

template <int nwr, int nwc>
static __device__ __forceinline__ void mmq_dense_pipe_store(const mmq_dense_pipe_regs<nwr, nwc> & r, int * __restrict__ xs, int * __restrict__ ys) {
    typedef mmq_dense_pipe_regs<nwr, nwc> R;
    const int w    = threadIdx.y;
    const int lane = threadIdx.x;
#pragma unroll
    for (int p = 0; p < R::nxq; ++p) {
        const int i = p*(nwr*nwc) + w;
        xs[i*mmq_dense_pipe_xs + lane] = r.xq[p];
    }
    const int t = w*32 + lane;
#pragma unroll
    for (int p = 0; p < R::nxd; ++p) {
        const int e = p*R::nt + t;
        if (R::I*4 % R::nt == 0 || e < R::I*4) {
            ((float *) xs)[(e/4)*mmq_dense_pipe_xs + 32 + e%4] = __half2float(r.xd[p]);
        }
    }
#pragma unroll
    for (int p = 0; p < R::ny; ++p) {
        ((typename R::ytype *) ys)[p*R::nt + t] = r.yv[p];
    }
}

// The products of one stage (4 blocks of 32 in K order) for the 16 rows x 128/nwc columns of a warp.
template <int nwr, int nwc>
static __device__ __forceinline__ void mmq_dense_pipe_compute(const int * __restrict__ xs, const int * __restrict__ ys, float * __restrict__ sum) {
    typedef tile<16,  8, int, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_A;
    typedef tile<16,  8, int, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR>          tile_C;
    constexpr int JW = mmq_dense_pipe_J / nwc;
    constexpr int NJ = JW / tile_C::J;

    const int wr = threadIdx.y % nwr;
    const int wc = threadIdx.y / nwr;
    const float * xf = (const float *) xs;
    const float * yf = (const float *) ys;

#pragma unroll 1
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        tile_A A;
        load_ldmatrix(A, xs + (wr*tile_A::I)*mmq_dense_pipe_xs + k01, mmq_dense_pipe_xs);
        float dA[tile_C::ne];
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            dA[l] = xf[(wr*tile_A::I + tile_C::get_i(l))*mmq_dense_pipe_xs + 32 + k01/QI8_0];
        }
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const int j0 = wc*JW + jj*tile_C::J;
            tile_B B;
            load_ldmatrix(B, ys + j0*mmq_dense_pipe_ys + 4 + k01, mmq_dense_pipe_ys);
            const float dB = yf[(j0 + tile_C::get_j(0))*mmq_dense_pipe_ys + k01/QI8_1];

            tile_C C;
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] = mmq_dense_pipe_magic_i;
            }
            mma(C, A, B);
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const float c = __int_as_float(C.x[l]) - mmq_dense_pipe_magic_f;
                sum[jj*tile_C::ne + l] += c*dA[l]*dB;
            }
        }
    }
}

// grid (G, row tiles, column groups): block (x, y, z) takes column tile z*G + x and row tile y, see mmq_dense_col_group
template <int nwr, int nwc, bool dbuf, int min_waves>
__launch_bounds__(32*nwr*nwc, min_waves)
static __global__ void mmq_dense_pipe(const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int ncols, const int stride_row_x, const int ncols_y, const int stride_col_dst, const int nkb, const int ntj) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
    constexpr int I  = 16*nwr;
    constexpr int JW = mmq_dense_pipe_J / nwc;
    constexpr int NJ = JW / tile_C::J;

    const int jt = blockIdx.z*gridDim.x + blockIdx.x;
    const int it = blockIdx.y;
    if (jt >= ntj) {
        return;
    }
    const int col_low = jt*mmq_dense_pipe_J;
    const int j_sup   = min(mmq_dense_pipe_J, ncols - col_low);

    extern __shared__ int mmq_dense_pipe_smem[];
    int * xs0 = mmq_dense_pipe_smem;
    int * ys0 = xs0 + I*mmq_dense_pipe_xs;
    int * xs1 = ys0 + mmq_dense_pipe_J*mmq_dense_pipe_ys;
    int * ys1 = xs1 + I*mmq_dense_pipe_xs;

    const int * yt = y + col_low*mmq_dense_pipe_ys;
    const int row0 = it*I;
    const int ns   = nkb / 4;

    float sum[NJ*tile_C::ne] = {0.0f};
    mmq_dense_pipe_regs<nwr, nwc> r;
    const mmq_dense_pipe_offsets o = mmq_dense_pipe_get_offsets<nwr, nwc>(stride_row_x);

    if constexpr (dbuf) {
        mmq_dense_pipe_load(r, x, yt, o, row0, stride_row_x, 0, ncols_y);
        mmq_dense_pipe_store(r, xs0, ys0);
        __syncthreads();
        for (int s = 0; s < ns; ++s) {
            if (s + 1 < ns) {
                mmq_dense_pipe_load(r, x, yt, o, row0, stride_row_x, s + 1, ncols_y);
            }
            mmq_dense_pipe_compute<nwr, nwc>(s % 2 == 0 ? xs0 : xs1, s % 2 == 0 ? ys0 : ys1, sum);
            if (s + 1 < ns) {
                mmq_dense_pipe_store(r, s % 2 == 0 ? xs1 : xs0, s % 2 == 0 ? ys1 : ys0);
            }
            __syncthreads();
        }
    } else {
        mmq_dense_pipe_load(r, x, yt, o, row0, stride_row_x, 0, ncols_y);
        for (int s = 0; s < ns; ++s) {
            if (s > 0) {
                __syncthreads();
            }
            mmq_dense_pipe_store(r, xs0, ys0);
            __syncthreads();
            if (s + 1 < ns) {
                mmq_dense_pipe_load(r, x, yt, o, row0, stride_row_x, s + 1, ncols_y);
            }
            mmq_dense_pipe_compute<nwr, nwc>(xs0, ys0, sum);
        }
    }

    const int wr = threadIdx.y % nwr;
    const int wc = threadIdx.y / nwr;
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int j = wc*JW + jj*tile_C::J + tile_C::get_j(l);
            if (j >= j_sup) {
                continue;
            }
            const int i = row0 + wr*16 + tile_C::get_i(l);
            dst[(col_low + j)*stride_col_dst + i] = sum[jj*tile_C::ne + l];
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, ncols, stride_row_x, ncols_y, stride_col_dst, nkb, ntj);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
}

#endif // defined(GGML_USE_HIP)

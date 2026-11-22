#pragma once

// Full-width tiles live in the seam section. The expert's extra descriptor always contains
// its final compact-column endpoint, even when the extra descriptor is empty.
template<typename Spec>
#if defined(__HIP__) || defined(__CUDACC__)
__host__ __device__
#endif
static inline int inkling_hotpair_role(const Spec * specs, int index, int seams) {
    const auto s = specs[index];
    if (index >= seams || s.col_high - s.col_low != 32) { return 0; }
    const int remaining = (specs[seams + s.expert].col_high - s.col_low) / 32;
    if ((remaining & 1) == 0) {
        if (index + 1 >= seams) { return 0; }
        const auto next = specs[index + 1];
        return next.expert == s.expert && next.col_low == s.col_high && next.col_high - next.col_low == 32 ? 1 : 0;
    }
    if (index == 0) { return 0; }
    const auto prev = specs[index - 1];
    return prev.expert == s.expert && prev.col_high == s.col_low && prev.col_high - prev.col_low == 32 ? -1 : 0;
}

#ifndef INKLING_HOTPAIR_MAP_ONLY
// Two independent J32 accumulators, unchanged K order and quantization, one packed IQ4
// weight load/unpack per K tile. The two column groups take turns using the ordinary tile_y LDS.
static __device__ __forceinline__ void inkling_hotpair_two32(const char * x, const int * y, const int * ids,
        float * dst, int xoff, int sx, int sy, int sd, int imax, int nbk) {
    constexpr int J = 32, nth = 128, sz = sizeof(block_q8_1_mmq)/sizeof(int);
    constexpr auto load = ggml_cuda_mmq_get_load_tiles<GGML_TYPE_IQ4_NL,J,false,GGML_PREC_Q8,false>();
    constexpr auto dot = ggml_cuda_mmq_get_vec_dot<GGML_TYPE_IQ4_NL,J,false,GGML_PREC_Q8>();
    constexpr auto store = ggml_cuda_mmq_get_write_back<GGML_TYPE_IQ4_NL,J,false,GGML_PREC_Q8,false>();
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr auto c = ggml_cuda_mmq_get_config(GGML_TYPE_IQ4_NL,J,false,GGML_PREC_Q8);
    static_assert(c.I==64 && c.nthreads==nth && c.K_vram==256 && !c.stream_k && !c.fallback);
    static_assert(QK8_1_MMQ==128 && ggml_cuda_type_traits<GGML_TYPE_IQ4_NL>::qk==32 && sz==36 && MMQ_TILE_Y_K==36);
#endif
    extern __shared__ int data_mul_mat_q[];
    int * ty = data_mul_mat_q + J;
    int * tx = ty + GGML_PAD(J*MMQ_TILE_Y_K,nth);
    const int tid = threadIdx.y*32+threadIdx.x;
    float sum[2][J*64/nth] = {};
    for (int kb=0; kb<nbk; kb+=8) {
        load(x,nullptr,tx,xoff+kb,imax,sx);
#pragma unroll
        for(int half=0; half<2; ++half) {
#pragma unroll
            for(int group=0; group<2; ++group) {
                const int * by = y + (sy*(kb/4+half)+group*J)*sz;
#pragma unroll
                for(int j=0;j<J*MMQ_TILE_Y_K;j+=nth) { ty[j+tid]=by[j+tid]; }
                __syncthreads();
                dot(tx,ty,sum[group],half*MMQ_TILE_NE_K);
                __syncthreads();
            }
        }
    }
#pragma unroll
    for(int group=0;group<2;++group) {
        if(tid<J) { data_mul_mat_q[tid]=ids[group*J+tid]; }
        __syncthreads();
        store(sum[group],data_mul_mat_q,dst,nullptr,sd,imax,J,GGML_GLU_OP_SWIGLU,0.f,nullptr,nullptr);
        __syncthreads();
    }
}

__launch_bounds__(128, 2)
static __global__ void inkling_hotpair_j32(const char * __restrict__ x, const int * __restrict__ y,
        const int * __restrict__ ids, const ggml_cuda_mm_id_tile_spec * __restrict__ specs, float * __restrict__ dst,
        int sx, int sex, int sy, int sd, int nrow, int nbk) {
    const int index=blockIdx.y;
    const auto s=specs[index];
    const int cols=s.col_high-s.col_low;
    if(!cols) { return; }
    const int role=inkling_hotpair_role(specs,index,sy/32);
    if(role<0) { return; }
    const int row=blockIdx.x*64, xoff=s.expert*sex+row*sx;
    y+=s.col_low*(sizeof(block_q8_1_mmq)/sizeof(int)); ids+=s.col_low; dst+=row;
    if(role>0) { inkling_hotpair_two32(x,y,ids,dst,xoff,sx,sy,sd,nrow-row-1,nbk); return; }
    extern __shared__ int data_mul_mat_q[];
    const int tid=threadIdx.y*32+threadIdx.x;
    if(tid<cols) { data_mul_mat_q[tid]=ids[tid]; }
    __syncthreads();
    if(cols<=16) { inkling_tail_body<16>(x,y,data_mul_mat_q,dst,xoff,sx,sy,sd,nrow-row-1,cols,nbk); }
    else { inkling_tail_body<32>(x,y,data_mul_mat_q,dst,xoff,sx,sy,sd,nrow-row-1,cols,nbk); }
}
#endif

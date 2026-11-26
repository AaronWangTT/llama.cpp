#include "common.cuh"
#include "fattn-tile.cuh"
#include "inkling-pfattn.cuh"

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        default:
            GGML_ABORT("Unsupported head size");
    }
}

#ifdef GGML_USE_HIP
fattn_kernel_t ggml_hip_inkling_ppskip_kernel() {
    return flash_attn_tile<128, 128, 64, 1, false, false, true>;
}

int ggml_hip_inkling_pfattn_mode(ggml_backend_cuda_context & ctx, const ggml_tensor * mask, const int ** aux) {
    const bool fast = ggml_kperf_on(GGML_KPERF_INKLING_PFATTN);
    const bool zero = ggml_kperf_on(GGML_KPERF_INKLING_PFZERO);
    int pf = fast ? INKLING_PF_FAST : 0;
    // The zero skip is compiled with the fast kernel only. Its empty tile test loads the mask in pairs.
    if (fast && zero && (mask->nb[1] | mask->nb[2] | mask->nb[3] | uintptr_t(mask->data)) % 4 == 0) {
        pf |= INKLING_PF_ZERO;
    }
    if (ctx.inkling_pf_mask == mask) {
        pf |= INKLING_PF_FAR;
        *aux = ctx.inkling_pf_aux;
        ctx.inkling_pf_mask = nullptr;
    }
    const bool trace = ggml_kperf_trace(GGML_KPERF_INKLING_PFATTN);
    if (trace && pf) {
        GGML_LOG_INFO("INKLING_PFATTN_HIT %s K=%lld fast=%d flags=%d zero=%d\n", mask->name, (long long) mask->ne[0],
            int((pf & INKLING_PF_FAST) != 0), int((pf & INKLING_PF_FAR) != 0), int((pf & INKLING_PF_ZERO) != 0));
    }
    return pf;
}

#define INKLING_PFATTN_CASE(pf_) case pf_: return skip_empty_prefix ? inkling_pfattn_tile<true, pf_> : inkling_pfattn_tile<false, pf_>;
fattn_kernel_t ggml_hip_inkling_pfattn_kernel(const bool skip_empty_prefix, const int pf) {
    switch (pf) {
        INKLING_PFATTN_CASE(INKLING_PF_FAR)
        INKLING_PFATTN_CASE(INKLING_PF_FAST)
        INKLING_PFATTN_CASE(INKLING_PF_FAR | INKLING_PF_FAST)
        INKLING_PFATTN_CASE(INKLING_PF_FAST | INKLING_PF_ZERO)
        INKLING_PFATTN_CASE(INKLING_PF_FAR | INKLING_PF_FAST | INKLING_PF_ZERO)
        default: GGML_ABORT("no Inkling prefill kernel for this mode");
    }
}
#endif

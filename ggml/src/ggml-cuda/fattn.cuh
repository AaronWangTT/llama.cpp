#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

#ifdef GGML_USE_HIP
// True if the op runs on the Inkling prefill kernel, which can take a mask with tile flags (inkling_pfmask_24).
bool ggml_cuda_flash_attn_ext_inkling_pf(int device, const ggml_tensor * dst);

// Layout of the int buffer that inkling_pfmask_24 fills for the attention op of its mask.
// First one float bias per (head, query), then one flag per (K tile of 64 rows, tile of 64 queries).
#define INKLING_PF_AUX_HEADS   24
#define INKLING_PF_AUX_QUERIES 256
#define INKLING_PF_AUX_FLAGS   (INKLING_PF_AUX_HEADS*INKLING_PF_AUX_QUERIES)
#define INKLING_PF_AUX_KV_MAX  262144
#define INKLING_PF_AUX_SIZE    (INKLING_PF_AUX_FLAGS + (INKLING_PF_AUX_KV_MAX/64)*(INKLING_PF_AUX_QUERIES/64))
#endif

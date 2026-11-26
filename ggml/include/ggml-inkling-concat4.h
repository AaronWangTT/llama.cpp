#pragma once

#include "ggml.h"

// inkling.concat4, native gfx1103 HIP builds only. One ggml-base owns the CPU callback
// and checks its identity, including when llama and the HIP plugin are separate DSOs.
extern "C" {
GGML_API bool ggml_inkling_concat4_enabled();
GGML_API ggml_tensor * ggml_inkling_try_concat4(ggml_context * ctx, ggml_tensor * oldest,
        ggml_tensor * middle, ggml_tensor * newest, ggml_tensor * cur);
GGML_API bool ggml_inkling_is_concat4(const ggml_tensor * op);
}

#pragma once

#include "ggml-backend.h"
#include "ggml-kperf.h"

// inkling.depthmix_raw and inkling.concat4: MUL_MAT words 0..3 remain standard precision and hint fields.
constexpr int inkling_depthmix_slot = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;

// Optional, versioned constructor exported only by the native gfx1103 HIP module.
// A null result leaves the original MUL_MAT graph and scheduler untouched.
using inkling_depthmix_make_v2_t = ggml_tensor * (*)(ggml_context *, ggml_backend_sched_t, ggml_backend_t, ggml_tensor *);
constexpr char inkling_depthmix_make_v2_name[] = "ggml_backend_inkling_depthmix_make_v2";
constexpr int32_t inkling_depthmix_raw_v2_tag = 0x444d5632; // OUT_PROD only
static inline bool inkling_depthmix_raw_v2_enabled() {
    return ggml_kperf_on(GGML_KPERF_INKLING_DEPTHMIX_RAW);
}

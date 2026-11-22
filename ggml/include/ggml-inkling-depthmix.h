#pragma once

#include "ggml.h"
#include <cstdlib>
#include <cstring>

// Private experiment: MUL_MAT words 0..3 remain standard precision and hint fields.
constexpr int inkling_depthmix_slot = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
constexpr int32_t inkling_depthmix_tag = 0x444d5831;
static inline bool inkling_depthmix_enabled() {
    static const bool enabled = [] {
        const char * p = std::getenv("GGML_INKLING_DEPTHMIX_CONT");
        return p && std::strcmp(p, "1") == 0;
    }();
    return enabled;
}

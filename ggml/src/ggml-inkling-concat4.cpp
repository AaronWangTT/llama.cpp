#include "ggml-inkling-concat4.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>

namespace {
constexpr int D = 2304, T = 256;

bool layout(const ggml_tensor * x, size_t s) {
    return x && x->type == GGML_TYPE_F32 && x->ne[0] == D && x->ne[1] == int64_t(s) &&
        x->ne[2] == T && x->ne[3] == 1 && x->nb[0] == 4 && x->nb[1] == 4*D &&
        x->nb[2] == 4*D*s && x->nb[3] == 4*D*s*T && x->view_offs % 4 == 0;
}

void concat4_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    for (int t = ith; t < T; t += nth) {
        for (int s = 0; s < 4; ++s) {
            const ggml_tensor * x = dst->src[s];
            std::memcpy(static_cast<char *>(dst->data) + t*dst->nb[2] + s*dst->nb[1],
                        static_cast<const char *>(x->data) + t*x->nb[2], 4*D);
        }
    }
}
} // namespace

bool ggml_inkling_concat4_enabled() {
    static const bool enabled = [] {
        const char * p = std::getenv("GGML_INKLING_CONCAT4");
        return p && std::strcmp(p, "1") == 0;
    }();
    return enabled;
}

ggml_tensor * ggml_inkling_try_concat4(ggml_context * ctx, ggml_tensor * oldest,
        ggml_tensor * middle, ggml_tensor * newest, ggml_tensor * cur) {
    if (!ggml_inkling_concat4_enabled()) {
        return nullptr;
    }
    // Each iteration of the stock loop PREPENDS one block. cur is always last.
    ggml_tensor * args[] = { newest, middle, oldest, cur };
    for (auto * x : args) {
        if (!x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
                ggml_nelements(x) != D*T || x->view_offs % 4 != 0) {
            return nullptr;
        }
    }
    for (auto & x : args) {
        x = ggml_reshape_3d(ctx, x, D, 1, T);
    }
    return ggml_custom_4d(ctx, GGML_TYPE_F32, D, 4, T, 1, args, 4, concat4_cpu, GGML_N_TASKS_MAX, nullptr);
}

bool ggml_inkling_is_concat4(const ggml_tensor * op) {
    // Metadata only: this is also used after the pack has ceased to be live.
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || !layout(op, 4)) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != concat4_cpu || p.userdata || p.n_tasks != GGML_N_TASKS_MAX) {
        return false;
    }
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (j < 4 ? !layout(op->src[j], 1) : op->src[j] != nullptr) {
            return false;
        }
    }
    return true;
}

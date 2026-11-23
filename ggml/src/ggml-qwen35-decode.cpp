#include "ggml-qwen35-decode.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int head_dim = ggml_qwen35_head_dim;
constexpr int n_conv   = ggml_qwen35_n_conv;

bool env_on(const char * name) {
    const char * p = std::getenv(name);
    return p && std::strcmp(p, "1") == 0;
}

bool f32_block(const ggml_tensor * x, int64_t n) {
    return x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_nelements(x) == n;
}

// one I32 row index
bool row_id(const ggml_tensor * ids) {
    return ids && ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) && ggml_nelements(ids) == 1;
}

// states: [n, n_cells], cell: one row of n values
bool state_rows(const ggml_tensor * states, const ggml_tensor * cell, int64_t n) {
    return states && states->type == GGML_TYPE_F32 && ggml_is_matrix(states) && ggml_is_contiguous(states) && states->ne[0] == n &&
        states->ne[1] > 0 && f32_block(cell, n);
}

bool custom_op(const ggml_tensor * op, ggml_custom_op_t fun, int n_src) {
    if (!op || op->op != GGML_OP_CUSTOM || op->view_src || op->type != GGML_TYPE_F32) {
        return false;
    }
    ggml_custom_op_params p;
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "custom op params must fit");
    std::memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != fun || p.userdata || p.n_tasks != 1) {
        return false;
    }
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if ((op->src[j] != nullptr) != (j < n_src)) {
            return false;
        }
    }
    return true;
}

// source row of the state; a bad index reads row 0
int64_t state_row(const ggml_tensor * states, const ggml_tensor * ids) {
    const int32_t row = *static_cast<const int32_t *>(ids->data);
    return row >= 0 && row < states->ne[1] ? row : 0;
}

//
// mvshare
//

// the types with a mul_mat_vec_q kernel and a plain block layout
bool mvshare_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// number of stacked rows, 0 if the op can not take these tensors
int64_t mvshare_layout(const ggml_tensor * x, const ggml_tensor * const * w, int n_w) {
    if (!x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != ggml_nelements(x) || n_w < 2 || n_w > ggml_qwen35_mvshare_max) {
        return 0;
    }
    int64_t n_rows = 0;
    for (int j = 0; j < n_w; ++j) {
        const ggml_tensor * m = w[j];
        if (!m || !mvshare_type(m->type) || m->op != GGML_OP_NONE || m->view_src || !ggml_is_matrix(m) || !ggml_is_contiguous(m) ||
                m->ne[0] != x->ne[0] || m->ne[0] % ggml_blck_size(m->type) != 0 || m->ne[1] > INT32_MAX) {
            return 0;
        }
        n_rows += m->ne[1];
    }
    return n_rows <= INT32_MAX ? n_rows : 0;
}

// reference: float product with the dequantized rows
void mvshare_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const ggml_tensor * x_t = dst->src[0];
    const int64_t       n   = x_t->ne[0];
    const float *       x   = static_cast<const float *>(x_t->data);
    float *             out = static_cast<float *>(dst->data);

    std::vector<float> row(n);
    for (int j = 1; j < GGML_MAX_SRC && dst->src[j]; ++j) {
        const ggml_tensor * m = dst->src[j];
        const auto * traits = ggml_get_type_traits(m->type);
        for (int64_t r = 0; r < m->ne[1]; ++r) {
            traits->to_float(static_cast<const char *>(m->data) + r*m->nb[1], row.data(), n);
            float sum = 0.0f;
            for (int64_t i = 0; i < n; ++i) {
                sum += row[i]*x[i];
            }
            *out++ = sum;
        }
    }
}

//
// prep1
//

bool prep1_layout(const ggml_tensor * qkv, const ggml_tensor * beta, const ggml_tensor * alpha, const ggml_tensor * conv_w,
        const ggml_tensor * dt, const ggml_tensor * a, const ggml_tensor * states, const ggml_tensor * ids, const ggml_tensor * cell,
        int64_t n_norm) {
    if (!qkv || !beta) {
        return false;
    }
    const int64_t n   = ggml_nelements(qkv);
    const int64_t n_h = ggml_nelements(beta);
    return n > 0 && n % head_dim == 0 && n <= INT32_MAX/n_conv && n_norm >= 0 && n_norm <= n && n_norm % head_dim == 0 && n_h > 0 && n_h <= n &&
        f32_block(qkv, n) && f32_block(beta, n_h) && f32_block(alpha, n_h) && f32_block(dt, n_h) && f32_block(a, n_h) &&
        f32_block(conv_w, n_conv*n) && conv_w->ne[0] == n_conv && conv_w->ne[1] == n &&
        state_rows(states, cell, (n_conv - 1)*n) && row_id(ids);
}

// reference
void prep1_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const ggml_tensor * states = dst->src[6];
    const int64_t n      = ggml_nelements(dst->src[0]);
    const int64_t n_h    = ggml_nelements(dst->src[1]);
    const int64_t n_norm = ggml_get_op_params_i32(dst, ggml_qwen35_prep1_slot_n_norm);
    const float   eps    = ggml_get_op_params_f32(dst, ggml_qwen35_prep1_slot_eps);
    const float   scale  = ggml_get_op_params_f32(dst, ggml_qwen35_prep1_slot_scale);
    const float * qkv    = static_cast<const float *>(dst->src[0]->data);
    const float * beta   = static_cast<const float *>(dst->src[1]->data);
    const float * alpha  = static_cast<const float *>(dst->src[2]->data);
    const float * weight = static_cast<const float *>(dst->src[3]->data);
    const float * dt     = static_cast<const float *>(dst->src[4]->data);
    const float * a      = static_cast<const float *>(dst->src[5]->data);
    const float * past   = static_cast<const float *>(states->data) + state_row(states, dst->src[7])*states->ne[0];
    float *       cell   = static_cast<float *>(dst->src[8]->data);
    float *       out    = static_cast<float *>(dst->data);

    for (int64_t c = 0; c < n; ++c) {
        float x[n_conv];
        for (int j = 0; j < n_conv - 1; ++j) {
            x[j] = past[c*(n_conv - 1) + j];
        }
        x[n_conv - 1] = qkv[c];
        float sum = 0.0f;
        for (int j = 0; j < n_conv; ++j) {
            sum += x[j]*weight[c*n_conv + j];
        }
        out[c] = sum/(1.0f + expf(-sum));
        for (int j = 0; j < n_conv - 1; ++j) {
            cell[c*(n_conv - 1) + j] = x[j + 1];
        }
    }
    for (int64_t h = 0; h < n_norm/head_dim; ++h) {
        float * y = out + h*head_dim;
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            sum += y[i]*y[i];
        }
        const float norm = 1.0f/sqrtf(sum/head_dim + eps);
        for (int i = 0; i < head_dim; ++i) {
            y[i] = scale*(norm*y[i]);
        }
    }
    for (int64_t h = 0; h < n_h; ++h) {
        const float t = alpha[h] + dt[h];
        out[n + h]       = 1.0f/(1.0f + expf(-beta[h]));
        out[n + n_h + h] = (t > 20.0f ? t : log1pf(expf(t)))*a[h];
    }
}

//
// gdn1
//

bool gdn1_layout(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * g, const ggml_tensor * b,
        const ggml_tensor * states, const ggml_tensor * ids, const ggml_tensor * cell) {
    if (!q || !v) {
        return false;
    }
    const int64_t n_h   = v->ne[1];
    const int64_t n_h_k = q->ne[1];
    return v->ne[0] == head_dim && n_h > 0 && n_h <= INT32_MAX/(head_dim*head_dim) && f32_block(v, head_dim*n_h) &&
        q->ne[0] == head_dim && n_h_k > 0 && n_h % n_h_k == 0 && f32_block(q, head_dim*n_h_k) && f32_block(k, head_dim*n_h_k) &&
        k->ne[0] == head_dim && f32_block(g, n_h) && f32_block(b, n_h) &&
        state_rows(states, cell, head_dim*head_dim*n_h) && row_id(ids);
}

// reference; the state is stored transposed, row col of a head holds S[..][col]
void gdn1_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const ggml_tensor * states = dst->src[5];
    const int64_t n_h   = dst->src[2]->ne[1];
    const int64_t n_h_k = dst->src[0]->ne[1];
    const float * q     = static_cast<const float *>(dst->src[0]->data);
    const float * k     = static_cast<const float *>(dst->src[1]->data);
    const float * v     = static_cast<const float *>(dst->src[2]->data);
    const float * g     = static_cast<const float *>(dst->src[3]->data);
    const float * b     = static_cast<const float *>(dst->src[4]->data);
    const float * src   = static_cast<const float *>(states->data) + state_row(states, dst->src[6])*states->ne[0];
    float *       cell  = static_cast<float *>(dst->src[7]->data);
    float *       out   = static_cast<float *>(dst->data);
    const float   scale = 1.0f/sqrtf((float) head_dim);

    for (int64_t h = 0; h < n_h; ++h) {
        const float * q_h   = q + (h % n_h_k)*head_dim;
        const float * k_h   = k + (h % n_h_k)*head_dim;
        const float   decay = expf(g[h]);
        for (int col = 0; col < head_dim; ++col) {
            const float * s_in  = src  + (h*head_dim + col)*head_dim;
            float *       s_out = cell + (h*head_dim + col)*head_dim;
            float kv = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                kv += s_in[i]*k_h[i];
            }
            const float delta = (v[h*head_dim + col] - decay*kv)*b[h];
            float attn = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                const float s = decay*s_in[i] + k_h[i]*delta;
                s_out[i] = s;
                attn += s*q_h[i];
            }
            out[h*head_dim + col] = attn*scale;
        }
    }
}

//
// post1
//

bool post1_layout(const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * z) {
    if (!x) {
        return false;
    }
    const int64_t n = ggml_nelements(x);
    return n > 0 && n % head_dim == 0 && n <= INT32_MAX && f32_block(x, n) && f32_block(z, n) && f32_block(w, head_dim);
}

// reference
void post1_cpu(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) {
        return;
    }
    const int64_t n   = ggml_nelements(dst->src[0]);
    const float   eps = ggml_get_op_params_f32(dst, ggml_qwen35_post1_slot_eps);
    const float * x   = static_cast<const float *>(dst->src[0]->data);
    const float * w   = static_cast<const float *>(dst->src[1]->data);
    const float * z   = static_cast<const float *>(dst->src[2]->data);
    float *       out = static_cast<float *>(dst->data);

    for (int64_t h = 0; h < n/head_dim; ++h) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            sum += x[h*head_dim + i]*x[h*head_dim + i];
        }
        const float norm = 1.0f/sqrtf(sum/head_dim + eps);
        for (int i = 0; i < head_dim; ++i) {
            const float g = z[h*head_dim + i];
            out[h*head_dim + i] = g/(1.0f + expf(-g))*(norm*x[h*head_dim + i]*w[i]);
        }
    }
}
} // namespace

bool ggml_qwen35_mvshare_enabled() {
    static const bool enabled = env_on("GGML_QWEN35_MVSHARE");
    return enabled;
}

ggml_tensor * ggml_qwen35_try_mvshare(ggml_context * ctx, ggml_tensor ** w, int n_w, ggml_tensor * x) {
    const int64_t n_rows = ggml_qwen35_mvshare_enabled() ? mvshare_layout(x, w, n_w) : 0;
    if (n_rows == 0) {
        return nullptr;
    }
    ggml_tensor * args[1 + ggml_qwen35_mvshare_max] = { x };
    for (int j = 0; j < n_w; ++j) {
        args[j + 1] = w[j];
    }
    return ggml_custom_4d(ctx, GGML_TYPE_F32, n_rows, 1, 1, 1, args, n_w + 1, mvshare_cpu, 1, nullptr);
}

bool ggml_qwen35_is_mvshare(const ggml_tensor * op) {
    int n_w = 0;
    for (int j = 1; op && j < GGML_MAX_SRC && op->src[j]; ++j) {
        n_w++;
    }
    if (!custom_op(op, mvshare_cpu, n_w + 1) || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    const int64_t n_rows = mvshare_layout(op->src[0], op->src + 1, n_w);
    return n_rows > 0 && n_rows == op->ne[0];
}

bool ggml_qwen35_prep1_enabled() {
    static const bool enabled = env_on("GGML_QWEN35_PREP1");
    return enabled;
}

ggml_tensor * ggml_qwen35_try_prep1(ggml_context * ctx, ggml_tensor * qkv, ggml_tensor * beta, ggml_tensor * alpha,
        ggml_tensor * conv_w, ggml_tensor * dt, ggml_tensor * a, ggml_tensor * states, ggml_tensor * ids, ggml_tensor * cell,
        int64_t n_norm, float eps, float scale) {
    if (!ggml_qwen35_prep1_enabled() || !prep1_layout(qkv, beta, alpha, conv_w, dt, a, states, ids, cell, n_norm)) {
        return nullptr;
    }
    ggml_tensor * args[] = { qkv, beta, alpha, conv_w, dt, a, states, ids, cell };
    ggml_tensor * op = ggml_custom_4d(ctx, GGML_TYPE_F32, ggml_nelements(qkv) + 2*ggml_nelements(beta), 1, 1, 1, args, 9, prep1_cpu, 1, nullptr);
    ggml_set_op_params_i32(op, ggml_qwen35_prep1_slot_n_norm, (int32_t) n_norm);
    ggml_set_op_params_f32(op, ggml_qwen35_prep1_slot_eps,    eps);
    ggml_set_op_params_f32(op, ggml_qwen35_prep1_slot_scale,  scale);
    return op;
}

bool ggml_qwen35_is_prep1(const ggml_tensor * op) {
    if (!custom_op(op, prep1_cpu, 9) || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    return prep1_layout(op->src[0], op->src[1], op->src[2], op->src[3], op->src[4], op->src[5], op->src[6], op->src[7], op->src[8],
            ggml_get_op_params_i32(op, ggml_qwen35_prep1_slot_n_norm)) &&
        op->ne[0] == ggml_nelements(op->src[0]) + 2*ggml_nelements(op->src[1]);
}

bool ggml_qwen35_gdn1_enabled() {
    static const bool enabled = env_on("GGML_QWEN35_GDN1");
    return enabled;
}

ggml_tensor * ggml_qwen35_try_gdn1(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * g, ggml_tensor * b, ggml_tensor * states, ggml_tensor * ids, ggml_tensor * cell) {
    if (!ggml_qwen35_gdn1_enabled() || !gdn1_layout(q, k, v, g, b, states, ids, cell)) {
        return nullptr;
    }
    ggml_tensor * args[] = { q, k, v, g, b, states, ids, cell };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, head_dim, v->ne[1], 1, 1, args, 8, gdn1_cpu, 1, nullptr);
}

bool ggml_qwen35_is_gdn1(const ggml_tensor * op) {
    return custom_op(op, gdn1_cpu, 8) && op->ne[0] == head_dim && op->ne[2] == 1 && op->ne[3] == 1 &&
        gdn1_layout(op->src[0], op->src[1], op->src[2], op->src[3], op->src[4], op->src[5], op->src[6], op->src[7]) &&
        op->ne[1] == op->src[2]->ne[1];
}

bool ggml_qwen35_post1_enabled() {
    static const bool enabled = env_on("GGML_QWEN35_POST1");
    return enabled;
}

ggml_tensor * ggml_qwen35_try_post1(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * z, float eps) {
    if (!ggml_qwen35_post1_enabled() || !post1_layout(x, w, z)) {
        return nullptr;
    }
    ggml_tensor * args[] = { x, w, z };
    ggml_tensor * op = ggml_custom_4d(ctx, GGML_TYPE_F32, ggml_nelements(x), 1, 1, 1, args, 3, post1_cpu, 1, nullptr);
    ggml_set_op_params_f32(op, ggml_qwen35_post1_slot_eps, eps);
    return op;
}

bool ggml_qwen35_is_post1(const ggml_tensor * op) {
    return custom_op(op, post1_cpu, 3) && op->ne[1] == 1 && op->ne[2] == 1 && op->ne[3] == 1 &&
        post1_layout(op->src[0], op->src[1], op->src[2]) && op->ne[0] == ggml_nelements(op->src[0]);
}

// Fixture for the one-token ops of the gated delta net blocks (ggml-qwen35-decode.h).
// Each fused op on ROCm0 must match the stock ops it replaces bit for bit, also when the state row of the sequence
// is not the cell that gets the new state. The CPU callbacks must stay close to the stock ops on CPU.
// The switches select the shapes: the four GGML_QWEN35_* switches run qwen3.5-4b, the five GGML_Q35MOE_* switches run qwen3.6-35b.
//   GGML_QWEN35_MVSHARE=1 GGML_QWEN35_PREP1=1 GGML_QWEN35_GDN1=1 GGML_QWEN35_POST1=1 qwen35-hip-decode ABS_LIBDIR
//   GGML_Q35MOE_MVSHARE=1 GGML_Q35MOE_BA1=1 GGML_Q35MOE_PREP1=1 GGML_Q35MOE_GDN1=1 GGML_Q35MOE_POST1=1 qwen35-hip-decode ABS_LIBDIR
// With GGML_Q35MOE_PREPN=1 GGML_Q35MOE_GDNN=1 GGML_Q35MOE_POSTN=1 in addition to the five GGML_Q35MOE_* switches: state caches with
// rollback slots. One token (the one-token ops, as with GGML_Q35MOE_RS1) and 2 to 5 tokens (the ops for a few tokens) against the
// stock graph that writes the snapshots, also when the state row is one of the rows that the batch writes.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-qwen35-decode.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// sizes of the gated delta net of qwen3.5-4b and of qwen3.6-35b
constexpr int S       = ggml_qwen35_head_dim;
constexpr int H_K     = 16;
constexpr int H       = 32;
constexpr int KEY     = S*H_K;
constexpr int VAL     = S*H;
constexpr int C       = 2*KEY + VAL;
constexpr int N_CONV  = ggml_qwen35_n_conv;
constexpr int N_CELLS = 3;
constexpr int D_R     = (N_CONV - 1)*C;
constexpr int D_S     = S*S*H;
constexpr float EPS   = 1e-6f;

enum { FUSE_MVSHARE = 1, FUSE_PREP1 = 2, FUSE_GDN1 = 4, FUSE_POST1 = 8, FUSE_BA1 = 16 };

// qwen35: input of 2560, four quantized products of the input. qwen35moe: input of 2048, the beta and alpha weights are F32
// or, in files with other types for the dense weights, quantized as in qwen35
struct shape {
    const char * name;
    int          n;
    ggml_type    types[4]; // qkv, z, beta, alpha
    bool         ba_f32;   // beta and alpha are F32: not in mvshare, GGML_Q35MOE_BA1 takes them
};
static const shape SHAPE_QWEN35 = { "qwen35",      2560, { GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_0 }, false };
static const shape SHAPE_MOE    = { "qwen35moe",   2048, { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32,  GGML_TYPE_F32  }, true  };
static const shape SHAPE_MOE_Q  = { "qwen35moe-q", 2048, { GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 }, false };

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

// weights and state caches of one backend
struct model {
    const shape *   sh;
    ggml_backend_t  be;
    ggml_context *  ctx_w;
    ggml_context *  ctx_s;
    ggml_tensor *   w[4]; // qkv, z, beta, alpha
    ggml_tensor *   conv_w, * dt, * a, * norm_w;
    ggml_tensor *   r, * s;
};

struct inputs {
    std::vector<float> x, r, s;
    int row;  // state row of the sequence
    int cell; // row that gets the new state
};

// one result vector: products, q and k, v, beta, gate, attention output, gated norm, conv state row, delta net state row
constexpr int N_SEG = 9;
constexpr int SEG_N[N_SEG] = { C + VAL + 2*H, 2*KEY, VAL, H, H, VAL, VAL, D_R, D_S };
static const char * const SEG_NAME[N_SEG] = { "products", "qk", "v", "beta", "gate", "attn", "post", "conv state", "state" };

static std::vector<float> run(const model & m, int fuse, const inputs & in) {
    ggml_backend_t be = m.be;
    ggml_context * c  = ggml_init({128*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_cgraph *  gf = ggml_new_graph(c);
    const size_t   fs = sizeof(float);
    const int      N  = m.sh->n;
    // F32 beta and alpha weights are not in mvshare
    const int n_share = m.sh->ba_f32 ? 2 : 4;

    ggml_tensor * x   = ggml_new_tensor_2d(c, GGML_TYPE_F32, N, 1); ggml_set_input(x);
    ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);    ggml_set_input(ids);

    auto cell_of = [&](ggml_tensor * states) {
        return ggml_view_2d(c, states, states->ne[0], 1, states->nb[1], in.cell*states->nb[1]);
    };
    auto fused_op = [&](ggml_tensor * op, bool is_op) {
        check(op && is_op && ggml_backend_supports_op(be, op), "the backend must take the fused op");
        ggml_set_output(op);
        return op;
    };

    // products, as in build_qkvz() and build_layer_attn_linear()
    std::vector<ggml_tensor *> prod;
    ggml_tensor * qkv, * z, * beta, * alpha;
    if (fuse & FUSE_MVSHARE) {
        ggml_tensor * w[4] = { m.w[0], m.w[1], m.w[2], m.w[3] };
        ggml_tensor * shared = ggml_qwen35_try_mvshare(c, w, n_share, x);
        prod  = { fused_op(shared, ggml_qwen35_is_mvshare(shared)) };
        qkv   = ggml_view_3d(c, shared, C, 1, 1, C*fs, C*fs, 0);
        z     = ggml_view_2d(c, shared, VAL, 1, VAL*fs, C*fs);
        if (n_share == 4) {
            beta  = ggml_view_4d(c, shared, 1, H, 1, 1, fs, H*fs, H*fs, (C + VAL)*fs);
            alpha = ggml_view_3d(c, shared, H, 1, 1, H*fs, H*fs, (C + VAL + H)*fs);
        }
    } else {
        for (int j = 0; j < n_share; ++j) {
            prod.push_back(ggml_mul_mat(c, m.w[j], x));
            ggml_set_output(prod.back());
        }
        qkv   = ggml_reshape_3d(c, prod[0], C, 1, 1);
        z     = prod[1];
        if (n_share == 4) {
            beta  = ggml_reshape_4d(c, prod[2], 1, H, 1, 1);
            alpha = ggml_reshape_3d(c, prod[3], H, 1, 1);
        }
    }
    if (n_share == 2 && (fuse & FUSE_BA1)) {
        ggml_tensor * w[2] = { m.w[2], m.w[3] };
        ggml_tensor * ba = ggml_q35moe_try_mvf(c, w, 2, x);
        prod.push_back(fused_op(ba, ggml_q35moe_is_mvf(ba)));
        beta  = ggml_view_4d(c, ba, 1, H, 1, 1, fs, H*fs, H*fs, 0);
        alpha = ggml_view_3d(c, ba, H, 1, 1, H*fs, H*fs, H*fs);
    } else if (n_share == 2) {
        for (int j = 2; j < 4; ++j) {
            prod.push_back(ggml_mul_mat(c, m.w[j], x));
            ggml_set_output(prod.back());
        }
        beta  = ggml_reshape_4d(c, prod[prod.size() - 2], 1, H, 1, 1);
        alpha = ggml_reshape_3d(c, prod[prod.size() - 1], H, 1, 1);
    }
    ggml_build_forward_expand(gf, z);

    // inputs of the delta net
    std::vector<ggml_tensor *> prep;
    ggml_tensor * q, * k, * v, * gate;
    if (fuse & FUSE_PREP1) {
        ggml_tensor * op = ggml_qwen35_try_prep1(c, qkv, beta, alpha, m.conv_w, m.dt, m.a, m.r, ids, cell_of(m.r),
                2*KEY, EPS/(float) S, 1.0f/sqrtf((float) S));
        prep = { fused_op(op, ggml_qwen35_is_prep1(op)) };
        q    = ggml_view_4d(c, op, S, H_K, 1, 1, S*fs, KEY*fs, KEY*fs, 0);
        k    = ggml_view_4d(c, op, S, H_K, 1, 1, S*fs, KEY*fs, KEY*fs, KEY*fs);
        v    = ggml_view_4d(c, op, S, H,   1, 1, S*fs, VAL*fs, VAL*fs, 2*KEY*fs);
        beta = ggml_view_4d(c, op, 1, H, 1, 1, fs, H*fs, H*fs, C*fs);
        gate = ggml_view_4d(c, op, 1, H, 1, 1, fs, H*fs, H*fs, (C + H)*fs);
    } else {
        beta = ggml_sigmoid(c, beta);
        gate = ggml_mul(c, ggml_softplus(c, ggml_add(c, alpha, m.dt)), m.a);
        ggml_tensor * gate_out = gate;
        gate = ggml_reshape_4d(c, gate, 1, H, 1, 1);

        // build_conv_state()
        ggml_tensor * past = ggml_get_rows(c, m.r, ids);
        ggml_build_forward_expand(gf, past);
        past = ggml_reshape_3d(c, past, N_CONV - 1, C, 1);
        ggml_tensor * conv_x = ggml_concat(c, past, ggml_transpose(c, qkv), 0);
        ggml_tensor * last   = ggml_view_3d(c, conv_x, N_CONV - 1, C, 1, conv_x->nb[1], conv_x->nb[2], fs);
        ggml_build_forward_expand(gf, ggml_cpy(c, last, cell_of(m.r)));

        ggml_tensor * conv = ggml_silu(c, ggml_ssm_conv(c, conv_x, m.conv_w));
        q = ggml_view_4d(c, conv, S, H_K, 1, 1, S*fs, C*fs, C*fs, 0);
        k = ggml_view_4d(c, conv, S, H_K, 1, 1, S*fs, C*fs, C*fs, KEY*fs);
        v = ggml_view_4d(c, conv, S, H,   1, 1, S*fs, C*fs, C*fs, 2*KEY*fs);
        q = ggml_scale(c, ggml_rms_norm(c, q, EPS/(float) S), 1.0f/sqrtf((float) S));
        k = ggml_scale(c, ggml_rms_norm(c, k, EPS/(float) S), 1.0f/sqrtf((float) S));
        // only the last node of each backend fusion is an output
        prep = { q, k, conv, beta, gate_out };
        for (ggml_tensor * t : prep) {
            ggml_set_output(t);
        }
    }

    // delta net
    ggml_tensor * out;
    if (fuse & FUSE_GDN1) {
        out = ggml_qwen35_try_gdn1(c, q, k, v, gate, beta, m.s, ids, cell_of(m.s));
        fused_op(out, ggml_qwen35_is_gdn1(out));
    } else {
        // build_rs() and build_recurrent_attn()
        ggml_tensor * state = ggml_get_rows(c, m.s, ids);
        ggml_build_forward_expand(gf, state);
        state = ggml_reshape_4d(c, state, S, S, H, 1);
        ggml_tensor * res = ggml_gated_delta_net(c, q, k, v, gate, beta, state, 1);
        ggml_tensor * new_state = ggml_view_4d(c, res, S, S, H, 1, S*fs, S*S*fs, D_S*fs, VAL*fs);
        ggml_build_forward_expand(gf, ggml_cpy(c, new_state, cell_of(m.s)));
        out = ggml_view_4d(c, res, S, H, 1, 1, S*fs, VAL*fs, VAL*fs, 0);
    }

    // gated norm, as in build_norm_gated()
    ggml_tensor * z_2d = ggml_view_4d(c, z, S, H, 1, 1, S*fs, VAL*fs, VAL*fs, 0);
    ggml_tensor * post;
    if (fuse & FUSE_POST1) {
        post = ggml_qwen35_try_post1(c, out, m.norm_w, z_2d, EPS);
        fused_op(post, ggml_qwen35_is_post1(post));
    } else {
        post = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, out, EPS), m.norm_w), ggml_silu(c, z_2d));
        ggml_set_output(post);
    }
    ggml_build_forward_expand(gf, post);
    // the delta net output is a view of the op result in the stock graph: read it through a copy
    ggml_tensor * out_copy = ggml_cont(c, out);
    ggml_set_output(out_copy);
    ggml_build_forward_expand(gf, out_copy);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    check(ggml_gallocr_alloc_graph(ga, gf), "alloc");
    const int32_t row = in.row;
    ggml_backend_tensor_set(x,   in.x.data(), 0, N*fs);
    ggml_backend_tensor_set(ids, &row,        0, sizeof(row));
    ggml_backend_tensor_set(m.r, in.r.data(), 0, in.r.size()*fs);
    ggml_backend_tensor_set(m.s, in.s.data(), 0, in.s.size()*fs);
    check(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute");

    std::vector<float> res;
    auto append = [&](const ggml_tensor * t, size_t offs, size_t n) {
        res.resize(res.size() + n);
        ggml_backend_tensor_get(t, res.data() + res.size() - n, offs*fs, n*fs);
    };
    for (const ggml_tensor * t : prod) {
        append(t, 0, ggml_nelements(t));
    }
    if (fuse & FUSE_PREP1) {
        append(prep[0], 0, C + 2*H);
    } else {
        append(prep[0], 0,     KEY);
        append(prep[1], 0,     KEY);
        append(prep[2], 2*KEY, VAL);
        append(prep[3], 0,     H);
        append(prep[4], 0,     H);
    }
    append(out_copy, 0,         VAL);
    append(post, 0,             VAL);
    append(m.r, in.cell*D_R,    D_R);
    append(m.s, (size_t) in.cell*D_S, D_S);

    ggml_gallocr_free(ga);
    ggml_free(c);
    return res;
}

static model make_model(const shape & sh, ggml_backend_t be, std::mt19937 & rng) {
    static const int  rows[4] = { C, VAL, H, H };
    const ggml_type * types   = sh.types;
    const int         N       = sh.n;
    std::normal_distribution<float> nd(0.0f, 1.0f);

    model m;
    m.sh    = &sh;
    m.be    = be;
    m.ctx_w = ggml_init({16*ggml_tensor_overhead(), nullptr, true});
    m.ctx_s = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
    for (int j = 0; j < 4; ++j) {
        m.w[j] = ggml_new_tensor_2d(m.ctx_w, types[j], N, rows[j]);
    }
    m.conv_w = ggml_new_tensor_2d(m.ctx_w, GGML_TYPE_F32, N_CONV, C);
    m.dt     = ggml_new_tensor_1d(m.ctx_w, GGML_TYPE_F32, H);
    m.a      = ggml_new_tensor_1d(m.ctx_w, GGML_TYPE_F32, H);
    m.norm_w = ggml_new_tensor_1d(m.ctx_w, GGML_TYPE_F32, S);
    m.r      = ggml_new_tensor_2d(m.ctx_s, GGML_TYPE_F32, D_R, N_CELLS);
    m.s      = ggml_new_tensor_2d(m.ctx_s, GGML_TYPE_F32, D_S, N_CELLS);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, be);
    ggml_backend_buffer_t buf_s = ggml_backend_alloc_ctx_tensors(m.ctx_s, be);
    check(buf_w && buf_s, "allocation");
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::mt19937 wrng(rng()); // the same weights on every backend need the same seed, see main()
    for (int j = 0; j < 4; ++j) {
        std::vector<float> f((size_t) N*rows[j]);
        for (auto & val : f) { val = 0.03f*nd(wrng); }
        if (types[j] == GGML_TYPE_F32) {
            ggml_backend_tensor_set(m.w[j], f.data(), 0, f.size()*sizeof(float));
            continue;
        }
        std::vector<char> qd(ggml_nbytes(m.w[j]));
        check(ggml_quantize_chunk(types[j], f.data(), qd.data(), 0, rows[j], N, nullptr) == qd.size(), "quantize");
        ggml_backend_tensor_set(m.w[j], qd.data(), 0, qd.size());
    }
    std::vector<float> f(N_CONV*C);
    for (auto & val : f) { val = 0.5f*nd(wrng); }
    ggml_backend_tensor_set(m.conv_w, f.data(), 0, f.size()*sizeof(float));
    float dt[H], a[H];
    for (int h = 0; h < H; ++h) {
        dt[h] = nd(wrng);
        a[h]  = -std::exp(0.5f*nd(wrng));
    }
    // one head on the linear branch of softplus
    dt[H - 1] = 40.0f;
    ggml_backend_tensor_set(m.dt, dt, 0, sizeof(dt));
    ggml_backend_tensor_set(m.a,  a,  0, sizeof(a));
    float norm_w[S];
    for (auto & val : norm_w) { val = 1.0f + 0.3f*nd(wrng); }
    ggml_backend_tensor_set(m.norm_w, norm_w, 0, sizeof(norm_w));
    return m;
}

// all checks for one set of shapes, returns the number of values that differ
static long run_shape(const shape & sh, ggml_backend_t hip, ggml_backend_t cpu, double & worst_cpu) {
    const int N = sh.n;
    std::mt19937 rng(8192);
    std::mt19937 rng_cpu = rng;
    const model m_hip = make_model(sh, hip, rng);
    const model m_cpu = make_model(sh, cpu, rng_cpu);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    static const int rows[][2] = { {1, 1}, {0, 0}, {0, 2}, {2, 1} }; // {state row, cell}
    const int fuse_all = FUSE_MVSHARE | FUSE_PREP1 | FUSE_GDN1 | FUSE_POST1 | (sh.ba_f32 ? FUSE_BA1 : 0);
    std::vector<int> modes = { FUSE_MVSHARE, FUSE_PREP1, FUSE_GDN1, FUSE_POST1 };
    if (sh.ba_f32) {
        modes.push_back(FUSE_BA1);
        modes.push_back(FUSE_BA1 | FUSE_PREP1);
    }
    modes.push_back(fuse_all);

    long n_diff = 0;
    for (int rep = 0; rep < 8; ++rep) {
        inputs in;
        in.x.resize(N); in.r.resize((size_t) D_R*N_CELLS); in.s.resize((size_t) D_S*N_CELLS);
        in.row  = rows[rep % 4][0];
        in.cell = rows[rep % 4][1];
        const float amp_s = 0.05f*std::exp(nd(rng));
        for (auto & val : in.x) { val = nd(rng); }
        for (auto & val : in.r) { val = nd(rng); }
        for (auto & val : in.s) { val = amp_s*nd(rng); }

        const auto ref = run(m_hip, 0, in);
        for (int mode : modes) {
            const auto res = run(m_hip, mode, in);
            check(res.size() == ref.size(), "result size");
            size_t offs = 0;
            std::printf("%s rep %d row %d cell %d fused %d:", sh.name, rep, in.row, in.cell, mode);
            for (int seg = 0; seg < N_SEG; ++seg) {
                int diff = 0;
                for (int i = 0; i < SEG_N[seg]; ++i) {
                    check(std::isfinite(ref[offs + i]) && std::isfinite(res[offs + i]), "finite");
                    diff += std::memcmp(&ref[offs + i], &res[offs + i], sizeof(float)) != 0;
                }
                std::printf(" %s %d/%d", SEG_NAME[seg], diff, SEG_N[seg]);
                n_diff += diff;
                offs   += SEG_N[seg];
            }
            std::printf("\n");
        }

        // CPU: the callbacks of prep1, gdn1 and post1 against the stock ops with the same products
        if (rep < 2) {
            const auto ref_cpu = run(m_cpu, 0, in);
            const auto res_cpu = run(m_cpu, FUSE_PREP1 | FUSE_GDN1 | FUSE_POST1, in);
            size_t offs = SEG_N[0];
            for (int seg = 1; seg < N_SEG; ++seg) {
                double amax = 0.0, dmax = 0.0;
                for (int i = 0; i < SEG_N[seg]; ++i) {
                    amax = std::max(amax, (double) std::fabs(ref_cpu[offs + i]));
                    dmax = std::max(dmax, (double) std::fabs(ref_cpu[offs + i] - res_cpu[offs + i]));
                }
                std::printf("%s rep %d cpu callbacks, %s: max|ref| %.4f, max diff %.3e\n", sh.name, rep, SEG_NAME[seg], amax, dmax);
                worst_cpu = std::max(worst_cpu, dmax/amax);
                offs += SEG_N[seg];
            }
            // the mvshare callback takes the unquantized input, the stock CPU product quantizes it
            const auto mv_cpu = run(m_cpu, FUSE_MVSHARE, in);
            double amax = 0.0, dmax = 0.0;
            for (int i = 0; i < SEG_N[0]; ++i) {
                amax = std::max(amax, (double) std::fabs(ref_cpu[i]));
                dmax = std::max(dmax, (double) std::fabs(ref_cpu[i] - mv_cpu[i]));
            }
            std::printf("%s rep %d cpu callback, products: max|ref| %.4f, max diff %.3e\n", sh.name, rep, amax, dmax);
            check(dmax/amax < 5e-2, "cpu reference of mvshare is off");
            // the mvf callback against the stock F32 products on CPU
            if (sh.ba_f32) {
                const auto ba_cpu = run(m_cpu, FUSE_BA1, in);
                amax = 0.0, dmax = 0.0;
                for (int i = C + VAL; i < SEG_N[0]; ++i) {
                    amax = std::max(amax, (double) std::fabs(ref_cpu[i]));
                    dmax = std::max(dmax, (double) std::fabs(ref_cpu[i] - ba_cpu[i]));
                }
                std::printf("%s rep %d cpu callback, beta and alpha products: max|ref| %.4f, max diff %.3e\n", sh.name, rep, amax, dmax);
                worst_cpu = std::max(worst_cpu, dmax/amax);
            }
        }
    }
    return n_diff;
}

//
// state caches with rollback slots: slot s of cell c is row s*N_CELLS + c, slot s is the state of s tokens before the end of a batch
//

constexpr int N_SLOTS = 5;
constexpr int N_ROWS  = N_CELLS*N_SLOTS;

struct states_keep {
    ggml_context * ctx;
    ggml_tensor *  r, * s;
};

static states_keep make_states_keep(ggml_backend_t be) {
    states_keep st;
    st.ctx = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
    st.r   = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, D_R, N_ROWS);
    st.s   = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, D_S, N_ROWS);
    check(ggml_backend_alloc_ctx_tensors(st.ctx, be) != nullptr, "allocation");
    return st;
}

// result vector for n_t tokens: q and k, v, beta, gate, attention output, gated norm, all conv state rows, all delta net state rows
constexpr int N_SEG_KEEP = 8;
static const char * const SEG_KEEP_NAME[N_SEG_KEEP] = { "qk", "v", "beta", "gate", "attn", "post", "conv states", "states" };

static void seg_keep_sizes(int n_t, size_t * n) {
    const size_t sizes[N_SEG_KEEP] = { (size_t) 2*KEY*n_t, (size_t) VAL*n_t, (size_t) H*n_t, (size_t) H*n_t, (size_t) VAL*n_t, (size_t) VAL*n_t,
        (size_t) D_R*N_ROWS, (size_t) D_S*N_ROWS };
    std::copy(sizes, sizes + N_SEG_KEEP, n);
}

// n_t tokens of one sequence, the stock nodes of a context with n_rs_seq = N_SLOTS - 1 or the fused ops
static std::vector<float> run_keep(const model & m, const states_keep & st, int fuse, const inputs & in, int n_t) {
    ggml_backend_t be = m.be;
    ggml_context * c  = ggml_init({256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_cgraph *  gf = ggml_new_graph(c);
    const size_t   fs = sizeof(float);
    const int      N  = m.sh->n;
    const int64_t  K  = N_SLOTS;
    check(n_t >= 1 && n_t <= K, "every token writes a slot");

    ggml_tensor * x   = ggml_new_tensor_2d(c, GGML_TYPE_F32, N, n_t); ggml_set_input(x);
    ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);      ggml_set_input(ids);

    auto cell_of = [&](ggml_tensor * states) {
        return ggml_view_2d(c, states, states->ne[0], 1, states->nb[1], in.cell*states->nb[1]);
    };
    auto fused_op = [&](ggml_tensor * op, bool is_op) {
        check(op && is_op && ggml_backend_supports_op(be, op), "the backend must take the fused op");
        ggml_set_output(op);
        return op;
    };

    // products, as in build_qkvz() and build_layer_attn_linear()
    ggml_tensor * qkv   = ggml_reshape_3d(c, ggml_mul_mat(c, m.w[0], x), C, n_t, 1);
    ggml_tensor * z     = ggml_mul_mat(c, m.w[1], x);
    ggml_tensor * beta  = ggml_reshape_4d(c, ggml_mul_mat(c, m.w[2], x), 1, H, n_t, 1);
    ggml_tensor * alpha = ggml_reshape_3d(c, ggml_mul_mat(c, m.w[3], x), H, n_t, 1);
    ggml_build_forward_expand(gf, z);

    // inputs of the delta net
    std::vector<ggml_tensor *> prep;
    ggml_tensor * q, * k, * v, * gate;
    if (fuse & FUSE_PREP1) {
        ggml_tensor * op = n_t == 1 ?
            ggml_qwen35_try_prep1(c, qkv, beta, alpha, m.conv_w, m.dt, m.a, st.r, ids, cell_of(st.r), 2*KEY, EPS/(float) S, 1.0f/sqrtf((float) S)) :
            ggml_q35moe_try_prepn(c, qkv, beta, alpha, m.conv_w, m.dt, m.a, st.r, ids, cell_of(st.r), 2*KEY, EPS/(float) S, 1.0f/sqrtf((float) S), N_CELLS);
        prep = { fused_op(op, n_t == 1 ? ggml_qwen35_is_prep1(op) : ggml_q35moe_is_prepn(op)) };
        q    = ggml_view_4d(c, op, S, H_K, n_t, 1, S*fs, C*fs, C*n_t*fs, 0);
        k    = ggml_view_4d(c, op, S, H_K, n_t, 1, S*fs, C*fs, C*n_t*fs, KEY*fs);
        v    = ggml_view_4d(c, op, S, H,   n_t, 1, S*fs, C*fs, C*n_t*fs, 2*KEY*fs);
        beta = ggml_view_4d(c, op, 1, H, n_t, 1, fs, H*fs, H*n_t*fs, (size_t) C*n_t*fs);
        gate = ggml_view_4d(c, op, 1, H, n_t, 1, fs, H*fs, H*n_t*fs, (size_t) (C + H)*n_t*fs);
    } else {
        beta = ggml_sigmoid(c, beta);
        gate = ggml_mul(c, ggml_softplus(c, ggml_add(c, alpha, m.dt)), m.a);
        ggml_tensor * gate_out = gate;
        gate = ggml_reshape_4d(c, gate, 1, H, n_t, 1);

        // build_conv_state()
        ggml_tensor * past = ggml_get_rows(c, st.r, ids);
        ggml_build_forward_expand(gf, past);
        past = ggml_reshape_3d(c, past, N_CONV - 1, C, 1);
        ggml_tensor * conv_x = ggml_concat(c, past, ggml_transpose(c, qkv), 0);

        // build_conv_state_store() with snapshots
        const int64_t Kw = std::min<int64_t>(n_t, K);
        ggml_tensor * order = ggml_cast(c, ggml_scale_bias(c, ggml_arange(c, 0.0f, (float) K, 1.0f), -1.0f, (float) (K - 1)), GGML_TYPE_I32);
        order = ggml_view_1d(c, order, Kw, (K - Kw)*sizeof(int32_t));
        ggml_tensor * windows = ggml_view_4d(c, conv_x, N_CONV - 1, Kw, C, 1, fs, conv_x->nb[1], conv_x->nb[2], (n_t - (Kw - 1))*fs);
        ggml_tensor * slots   = ggml_view_4d(c, st.r, N_CONV - 1, K, C, 1, N_CELLS*st.r->nb[1], (N_CONV - 1)*fs, st.r->nb[1], in.cell*st.r->nb[1]);
        ggml_build_forward_expand(gf, ggml_set_rows(c, slots, windows, order));

        ggml_tensor * conv = ggml_silu(c, ggml_ssm_conv(c, conv_x, m.conv_w));
        q = ggml_view_4d(c, conv, S, H_K, n_t, 1, S*fs, C*fs, C*n_t*fs, 0);
        k = ggml_view_4d(c, conv, S, H_K, n_t, 1, S*fs, C*fs, C*n_t*fs, KEY*fs);
        v = ggml_view_4d(c, conv, S, H,   n_t, 1, S*fs, C*fs, C*n_t*fs, 2*KEY*fs);
        q = ggml_scale(c, ggml_rms_norm(c, q, EPS/(float) S), 1.0f/sqrtf((float) S));
        k = ggml_scale(c, ggml_rms_norm(c, k, EPS/(float) S), 1.0f/sqrtf((float) S));
        // only the last node of each backend fusion is an output
        prep = { q, k, conv, beta, gate_out };
        for (ggml_tensor * t : prep) {
            ggml_set_output(t);
        }
    }

    // delta net
    ggml_tensor * out;
    if (fuse & FUSE_GDN1) {
        out = n_t == 1 ? ggml_qwen35_try_gdn1(c, q, k, v, gate, beta, st.s, ids, cell_of(st.s)) :
                         ggml_q35moe_try_gdnn(c, q, k, v, gate, beta, st.s, ids, cell_of(st.s), N_CELLS);
        fused_op(out, n_t == 1 ? ggml_qwen35_is_gdn1(out) : ggml_q35moe_is_gdnn(out));
    } else {
        // build_rs() and build_recurrent_attn() with snapshots
        ggml_tensor * state = ggml_get_rows(c, st.s, ids);
        ggml_build_forward_expand(gf, state);
        state = ggml_reshape_4d(c, state, S, S, H, 1);
        ggml_tensor * res = ggml_gated_delta_net(c, q, k, v, gate, beta, state, K);
        const int64_t n_written = std::min<int64_t>(n_t, K);
        ggml_tensor * src = ggml_view_3d(c, res, D_S, 1, n_written, D_S*fs, D_S*fs, (size_t) VAL*n_t*fs);
        ggml_tensor * dst = ggml_view_3d(c, st.s, D_S, 1, n_written, st.s->nb[1], N_CELLS*st.s->nb[1], in.cell*st.s->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(c, src, dst));
        out = ggml_view_4d(c, res, S, H, n_t, 1, S*fs, VAL*fs, (size_t) VAL*n_t*fs, 0);
    }

    // gated norm, as in build_norm_gated()
    ggml_tensor * z_2d = ggml_reshape_4d(c, z, S, H, n_t, 1);
    ggml_tensor * post;
    if (fuse & FUSE_POST1) {
        post = ggml_qwen35_try_post1(c, out, m.norm_w, z_2d, EPS);
        fused_op(post, ggml_qwen35_is_post1(post));
    } else {
        post = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, out, EPS), m.norm_w), ggml_silu(c, z_2d));
        ggml_set_output(post);
    }
    ggml_build_forward_expand(gf, post);
    // the delta net output is a view of the op result in the stock graph: read it through a copy
    ggml_tensor * out_copy = ggml_cont(c, out);
    ggml_set_output(out_copy);
    ggml_build_forward_expand(gf, out_copy);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    check(ggml_gallocr_alloc_graph(ga, gf), "alloc");
    const int32_t row = in.row;
    ggml_backend_tensor_set(x,    in.x.data(), 0, in.x.size()*fs);
    ggml_backend_tensor_set(ids,  &row,        0, sizeof(row));
    ggml_backend_tensor_set(st.r, in.r.data(), 0, in.r.size()*fs);
    ggml_backend_tensor_set(st.s, in.s.data(), 0, in.s.size()*fs);
    check(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute");

    std::vector<float> res;
    auto append = [&](const ggml_tensor * t, size_t offs, size_t n) {
        res.resize(res.size() + n);
        ggml_backend_tensor_get(t, res.data() + res.size() - n, offs*fs, n*fs);
    };
    // the fused op and the stock SiLU node hold C values per token, the stock norms hold the q and k heads only
    const bool   prep_fused = fuse & FUSE_PREP1;
    const size_t s_qk       = prep_fused ? C : KEY;
    for (int t = 0; t < n_t; ++t) {
        append(prep[0], t*s_qk, KEY);
    }
    for (int t = 0; t < n_t; ++t) {
        append(prep_fused ? prep[0] : prep[1], t*s_qk + (prep_fused ? KEY : 0), KEY);
    }
    for (int t = 0; t < n_t; ++t) {
        append(prep_fused ? prep[0] : prep[2], (size_t) t*C + 2*KEY, VAL);
    }
    if (prep_fused) {
        append(prep[0], (size_t) C*n_t, 2*H*n_t);
    } else {
        append(prep[3], 0, H*n_t);
        append(prep[4], 0, H*n_t);
    }
    append(out_copy, 0, (size_t) VAL*n_t);
    append(post, 0,     (size_t) VAL*n_t);
    append(st.r, 0,     (size_t) D_R*N_ROWS);
    append(st.s, 0,     (size_t) D_S*N_ROWS);

    ggml_gallocr_free(ga);
    ggml_free(c);
    return res;
}

// 1 to N_SLOTS tokens in a cache with rollback slots, returns the number of values that differ
static long run_shape_keep(const shape & sh, ggml_backend_t hip, ggml_backend_t cpu, double & worst_cpu) {
    const int N = sh.n;
    std::mt19937 rng(4096);
    std::mt19937 rng_cpu = rng;
    const model m_hip = make_model(sh, hip, rng);
    const model m_cpu = make_model(sh, cpu, rng_cpu);
    const states_keep st_hip = make_states_keep(hip);
    const states_keep st_cpu = make_states_keep(cpu);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // {slot of the state row, cell of the state row, cell of the batch}: no rollback, rollbacks into rows that the batch writes,
    // the state of another cell, a slot of another cell
    static const int rows[][3] = { {0, 1, 1}, {1, 2, 2}, {3, 0, 0}, {0, 0, 2}, {2, 1, 0}, {4, 1, 1} };
    const std::vector<int> modes = { FUSE_PREP1, FUSE_GDN1, FUSE_POST1, FUSE_PREP1 | FUSE_GDN1 | FUSE_POST1 };

    long n_diff = 0;
    int  rep    = 0;
    for (int n_t = 1; n_t <= N_SLOTS; ++n_t) {
        size_t seg_n[N_SEG_KEEP];
        seg_keep_sizes(n_t, seg_n);
        for (const auto & rc : rows) {
            inputs in;
            in.x.resize((size_t) N*n_t); in.r.resize((size_t) D_R*N_ROWS); in.s.resize((size_t) D_S*N_ROWS);
            in.row  = rc[0]*N_CELLS + rc[1];
            in.cell = rc[2];
            const float amp_s = 0.05f*std::exp(nd(rng));
            for (auto & val : in.x) { val = nd(rng); }
            for (auto & val : in.r) { val = nd(rng); }
            for (auto & val : in.s) { val = amp_s*nd(rng); }

            const auto ref = run_keep(m_hip, st_hip, 0, in, n_t);
            for (int mode : modes) {
                const auto res = run_keep(m_hip, st_hip, mode, in, n_t);
                check(res.size() == ref.size(), "result size");
                size_t offs = 0;
                std::printf("%s keep tokens %d row %d cell %d fused %d:", sh.name, n_t, in.row, in.cell, mode);
                for (int seg = 0; seg < N_SEG_KEEP; ++seg) {
                    long diff = 0;
                    for (size_t i = 0; i < seg_n[seg]; ++i) {
                        check(std::isfinite(ref[offs + i]) && std::isfinite(res[offs + i]), "finite");
                        diff += std::memcmp(&ref[offs + i], &res[offs + i], sizeof(float)) != 0;
                    }
                    std::printf(" %s %ld/%zu", SEG_KEEP_NAME[seg], diff, seg_n[seg]);
                    n_diff += diff;
                    offs   += seg_n[seg];
                }
                std::printf("\n");
            }

            // CPU: the callbacks against the stock ops with the same products, one case per token count
            if (rep % 6 == 1) {
                const auto ref_cpu = run_keep(m_cpu, st_cpu, 0, in, n_t);
                const auto res_cpu = run_keep(m_cpu, st_cpu, FUSE_PREP1 | FUSE_GDN1 | FUSE_POST1, in, n_t);
                size_t offs = 0;
                for (int seg = 0; seg < N_SEG_KEEP; ++seg) {
                    double amax = 0.0, dmax = 0.0;
                    for (size_t i = 0; i < seg_n[seg]; ++i) {
                        amax = std::max(amax, (double) std::fabs(ref_cpu[offs + i]));
                        dmax = std::max(dmax, (double) std::fabs(ref_cpu[offs + i] - res_cpu[offs + i]));
                    }
                    std::printf("%s keep tokens %d cpu callbacks, %s: max|ref| %.4f, max diff %.3e\n", sh.name, n_t, SEG_KEEP_NAME[seg], amax, dmax);
                    worst_cpu = std::max(worst_cpu, dmax/amax);
                    offs += seg_n[seg];
                }
            }
            rep++;
        }
    }
    return n_diff;
}

int main(int argc, char ** argv) {
    check(argc == 2, "usage: qwen35-hip-decode ABS_LIBDIR");
    const bool run_qwen35 = ggml_qwen35_mvshare_enabled() && ggml_qwen35_prep1_enabled() && ggml_qwen35_gdn1_enabled() && ggml_qwen35_post1_enabled();
    const bool run_moe    = ggml_q35moe_mvshare_enabled() && ggml_q35moe_ba1_enabled() && ggml_q35moe_prep1_enabled() && ggml_q35moe_gdn1_enabled() &&
        ggml_q35moe_post1_enabled();
    check(run_qwen35 || run_moe, "set GGML_QWEN35_MVSHARE=1 GGML_QWEN35_PREP1=1 GGML_QWEN35_GDN1=1 GGML_QWEN35_POST1=1 or the five GGML_Q35MOE_* switches");
    const bool run_keep  = run_moe && ggml_q35moe_prepn_enabled() && ggml_q35moe_gdnn_enabled() && ggml_q35moe_postn_enabled();
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t hip = ggml_backend_init_by_name("ROCm0", nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    check(hip && cpu, "backends");

    long   n_diff    = 0;
    double worst_cpu = 0.0;
    if (run_qwen35) {
        n_diff += run_shape(SHAPE_QWEN35, hip, cpu, worst_cpu);
    }
    if (run_moe) {
        n_diff += run_shape(SHAPE_MOE, hip, cpu, worst_cpu);
        n_diff += run_shape(SHAPE_MOE_Q, hip, cpu, worst_cpu);
    }
    if (run_keep) {
        n_diff += run_shape_keep(SHAPE_MOE, hip, cpu, worst_cpu);
    }
    check(n_diff == 0, "a fused op is not bit-exact");
    check(worst_cpu < 1e-3, "cpu reference error above 1e-3");
    std::printf("PASS bit-exact%s%s%s, worst cpu reference error %.2e\n", run_qwen35 ? " qwen35" : "", run_moe ? " qwen35moe" : "",
            run_keep ? " qwen35moe-snapshots" : "", worst_cpu);
    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    return 0;
}

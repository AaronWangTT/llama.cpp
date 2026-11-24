// Fixture for GGML_Q35MOE_QSHARE (ggml/src/ggml-cuda/q35moe-qshare.cuh): MMQ products of one graph run that read the same input
// share one Q8_1 copy of it, gate and up of the routed experts also their column lists.
// Every case builds a graph in which products share an input, and a reference graph in which every product reads its own copy
// of that input (other tensors, so nothing can be shared). Both must give the same bits.
// Cases in which a kept copy would be stale: the input is changed in place between two products (scale, copy into it, scale of
// the larger tensor it is a view of), the ids of the routed experts differ or are changed in place, the same graph runs again
// with other data, the graphs get larger and smaller.
// With GGML_Q35MOE_QSHARE_LOG=1 the backend logs per graph how many products took a kept copy. The fixture checks these counts:
// the expected number with the switch on, none in the reference graphs and with the switch off.
// A second run of a graph is a captured graph, which takes no part; with GGML_CUDA_DISABLE_GRAPHS=1 it runs node by node and
// must not see the copy that the first run kept.
// "hash" covers all results; it must be the same with the switch on and off.
//   GGML_Q35MOE_EXPTAIL=1 [GGML_Q35MOE_QSHARE=1 GGML_Q35MOE_QSHARE_LOG=1] [GGML_CUDA_DISABLE_GRAPHS=1] q35moe-hip-qshare ABS_LIBDIR [BACKEND]
// BACKEND (default ROCm0): with CPU the fixture only checks its own graphs against their references.
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

constexpr int N      = 2048;
constexpr int N_FF   = 512;
constexpr int N_EXP  = 256;
constexpr int N_USED = 8;

enum { W_Q, W_K, W_V, W_OUT, W_ROUTER, W_UP_EXPS, W_GATE_EXPS, W_DOWN_EXPS, W_UP_SHEXP, W_GATE_SHEXP, W_DOWN_SHEXP, N_W };

struct weight_def {
    const char * name;
    ggml_type    type;
    int64_t      ne[3];
};

// names and types of Qwen3.6-35B-A3B-UD-Q4_K_XL; ssm_out is a Q8_0 weight that takes no part
static const weight_def defs[N_W] = {
    { "blk.0.attn_q.weight",         GGML_TYPE_Q8_0, { N,    2*N_FF, 1     } },
    { "blk.0.attn_k.weight",         GGML_TYPE_Q8_0, { N,    N_FF,   1     } },
    { "blk.0.attn_v.weight",         GGML_TYPE_Q8_0, { N,    N_FF,   1     } },
    { "blk.0.ssm_out.weight",        GGML_TYPE_Q8_0, { N,    N_FF,   1     } },
    { "blk.0.ffn_gate_inp.weight",   GGML_TYPE_F32,  { N,    N_EXP,  1     } },
    { "blk.0.ffn_up_exps.weight",    GGML_TYPE_Q4_K, { N,    N_FF,   N_EXP } },
    { "blk.0.ffn_gate_exps.weight",  GGML_TYPE_Q4_K, { N,    N_FF,   N_EXP } },
    { "blk.0.ffn_down_exps.weight",  GGML_TYPE_Q5_K, { N_FF, N,      N_EXP } },
    { "blk.0.ffn_up_shexp.weight",   GGML_TYPE_Q8_0, { N,    N_FF,   1     } },
    { "blk.0.ffn_gate_shexp.weight", GGML_TYPE_Q8_0, { N,    N_FF,   1     } },
    { "blk.0.ffn_down_shexp.weight", GGML_TYPE_Q8_0, { N_FF, N,      1     } },
};

static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

// counts of the log lines of the backend since the last reset
static int g_kept  = 0;
static int g_taken = 0;

static void log_callback(ggml_log_level level, const char * text, void *) {
    int nodes = 0, kept = 0, taken = 0;
    const char * p = std::strstr(text, "GGML_Q35MOE_QSHARE: graph of ");
    if (p && std::sscanf(p, "GGML_Q35MOE_QSHARE: graph of %d nodes, %d products quantized their input, %d took", &nodes, &kept, &taken) == 3) {
        g_kept  += kept;
        g_taken += taken;
        return;
    }
    if (level != GGML_LOG_LEVEL_DEBUG) {
        std::fputs(text, stderr);
    }
}

// random blocks of a quantized type: random bytes with scales of a sane size
static std::vector<char> random_weight(ggml_type type, size_t nbytes, std::mt19937_64 & rng) {
    std::vector<char> buf(nbytes);
    if (type == GGML_TYPE_F32) {
        std::normal_distribution<float> nd(0.0f, 0.05f);
        for (size_t i = 0; i + sizeof(float) <= nbytes; i += sizeof(float)) {
            const float v = nd(rng);
            std::memcpy(buf.data() + i, &v, sizeof(v));
        }
        return buf;
    }
    for (size_t i = 0; i + 8 <= nbytes; i += 8) {
        const uint64_t v = rng();
        std::memcpy(buf.data() + i, &v, 8);
    }
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    const size_t bs = ggml_type_size(type);
    auto put = [&](size_t offs, float v) {
        const ggml_fp16_t h = ggml_fp32_to_fp16((rng() & 1) ? v : -v);
        std::memcpy(buf.data() + offs, &h, sizeof(h));
    };
    for (size_t b = 0; b + bs <= nbytes; b += bs) {
        switch (type) {
            case GGML_TYPE_Q8_0: put(b, 2e-4f*ud(rng)); break;
            case GGML_TYPE_Q4_K: put(b, 1.5e-4f*ud(rng)); put(b + 2, 1e-3f*ud(rng)); break;
            case GGML_TYPE_Q5_K: put(b, 1e-4f*ud(rng));   put(b + 2, 1e-3f*ud(rng)); break;
            default: check(false, "weight type");
        }
    }
    return buf;
}

struct model {
    ggml_backend_t be;
    ggml_backend_t cpu;
    ggml_tensor *  w[N_W];
};

// a graph with its inputs and outputs
struct graph {
    ggml_context * c;
    ggml_cgraph *  gf;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>>   in_f32;
    std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> in_i32;
    std::vector<ggml_tensor *> outputs;

    graph() {
        c  = ggml_init({256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        gf = ggml_new_graph(c);
    }
    ~graph() { ggml_free(c); }

    ggml_tensor * in(const std::vector<float> & data, int64_t ne0, int64_t ne1) {
        check((int64_t) data.size() == ne0*ne1, "input size");
        ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, ne0, ne1);
        ggml_set_input(t);
        in_f32.emplace_back(t, data);
        return t;
    }
    ggml_tensor * in_ids(const std::vector<int32_t> & data) {
        ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_I32, N_USED, data.size()/N_USED);
        ggml_set_input(t);
        in_i32.emplace_back(t, data);
        return t;
    }
    // a node whose value is read back
    void out(ggml_tensor * t) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
        outputs.push_back(t);
    }
    // a node that only changes a tensor in place
    void step(ggml_tensor * t) {
        ggml_build_forward_expand(gf, t);
    }
};

static std::vector<float> read_outputs(const graph & g) {
    std::vector<float> res;
    for (ggml_tensor * t : g.outputs) {
        const size_t n = ggml_nelements(t);
        const size_t o = res.size();
        res.resize(o + n);
        ggml_backend_tensor_get(t, res.data() + o, 0, n*sizeof(float));
    }
    return res;
}

// Runs the graph through the scheduler as llama.cpp does. again: second set of values for the first input, the graph then runs
// a second time without a new allocation and the results of both runs are returned.
static std::vector<float> run(const model & m, graph & g, const std::vector<float> * again = nullptr) {
    ggml_backend_t backends[2] = { m.be, m.cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, m.be == m.cpu ? 1 : 2, 512, false, true);
    check(sched && ggml_backend_sched_alloc_graph(sched, g.gf), "alloc");
    for (auto & [t, data] : g.in_f32) {
        ggml_backend_tensor_set(t, data.data(), 0, data.size()*sizeof(float));
    }
    for (auto & [t, data] : g.in_i32) {
        ggml_backend_tensor_set(t, data.data(), 0, data.size()*sizeof(int32_t));
    }
    check(ggml_backend_sched_graph_compute(sched, g.gf) == GGML_STATUS_SUCCESS, "compute");
    std::vector<float> res = read_outputs(g);
    if (again) {
        ggml_backend_tensor_set(g.in_f32[0].first, again->data(), 0, again->size()*sizeof(float));
        check(ggml_backend_sched_graph_compute(sched, g.gf) == GGML_STATUS_SUCCESS, "second compute");
        const std::vector<float> res2 = read_outputs(g);
        res.insert(res.end(), res2.begin(), res2.end());
    }
    ggml_backend_sched_free(sched);
    return res;
}

// the values of one case
struct data {
    int64_t n; // tokens
    std::vector<float>   x, z;       // [N, n]
    std::vector<float>   base;       // [N, 2n]
    std::vector<int32_t> sel, sel2;  // [N_USED, n]
};

static std::vector<float> scaled(const std::vector<float> & v, float s, size_t first = 0, size_t count = SIZE_MAX) {
    count = std::min(count, v.size() - first);
    std::vector<float> r(v.begin() + first, v.begin() + first + count);
    for (float & f : r) { f *= s; }
    return r;
}

static std::vector<float> as_float(const std::vector<int32_t> & v) {
    return std::vector<float>(v.begin(), v.end());
}

struct test_case {
    const char * name;
    int          taken;       // products that take a kept copy in the shared graph with the switch on
    int          taken_again; // the same in a second run of the shared graph with z as its first input, -1: no second run
    std::function<void(graph &, const model &, const data &, bool)> build;
};

static ggml_tensor * mm(graph & g, const model & m, int w, ggml_tensor * x) {
    return ggml_mul_mat(g.c, m.w[w], x);
}

// a computed tensor with the values of an input. The scheduler gives the device a copy of a graph input, so only a computed
// tensor can be changed in place between two products that read it.
static ggml_tensor * var(graph & g, const std::vector<float> & data, int64_t ne0, int64_t ne1) {
    return ggml_dup(g.c, g.in(data, ne0, ne1));
}

// the input of the routed gate and up products: one activation per token, as build_moe_ffn() makes it once for both
static ggml_tensor * per_token(graph & g, ggml_tensor * x) {
    return ggml_reshape_3d(g.c, x, x->ne[0], 1, x->ne[1]);
}

static ggml_tensor * mm_id(graph & g, const model & m, int w, ggml_tensor * x3, ggml_tensor * ids) {
    return ggml_mul_mat_id(g.c, m.w[w], x3, ids);
}

static const test_case cases[] = {
    { "q, k, v of one graph input", 2, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = g.in(d.x, N, d.n);
        g.out(mm(g, m, W_Q, x));
        g.out(mm(g, m, W_K, shared ? x : g.in(d.x, N, d.n)));
        g.out(mm(g, m, W_V, shared ? x : g.in(d.x, N, d.n)));
    } },
    { "q, k, v of one computed tensor", 2, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = var(g, d.x, N, d.n);
        g.out(mm(g, m, W_Q, x));
        g.out(mm(g, m, W_K, shared ? x : var(g, d.x, N, d.n)));
        g.out(mm(g, m, W_V, shared ? x : var(g, d.x, N, d.n)));
    } },
    { "between: nodes that do not read the input, a product of another weight that reads it", 1, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = var(g, d.x, N, d.n);
        ggml_tensor * q = mm(g, m, W_Q, x);
        g.out(ggml_silu(g.c, q));
        g.out(mm(g, m, W_OUT, shared ? x : var(g, d.x, N, d.n)));
        g.out(ggml_add(g.c, q, q));
        g.out(mm(g, m, W_K, shared ? x : var(g, d.x, N, d.n)));
    } },
    { "between: a node that is not a product reads the input (the kept copy is dropped)", 0, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = var(g, d.x, N, d.n);
        g.out(mm(g, m, W_Q, x));
        g.out(ggml_sqr(g.c, shared ? x : var(g, d.x, N, d.n)));
        g.out(mm(g, m, W_K, shared ? x : var(g, d.x, N, d.n)));
    } },
    { "input scaled in place between", 1, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = var(g, d.x, N, d.n);
        g.out(mm(g, m, W_Q, x));
        if (shared) {
            g.step(ggml_scale_inplace(g.c, x, 0.5f));
            g.out(mm(g, m, W_K, x));
            g.out(mm(g, m, W_V, x));
        } else {
            g.out(mm(g, m, W_K, var(g, scaled(d.x, 0.5f), N, d.n)));
            g.out(mm(g, m, W_V, var(g, scaled(d.x, 0.5f), N, d.n)));
        }
    } },
    { "another tensor copied into the input between", 0, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = var(g, d.x, N, d.n);
        g.out(mm(g, m, W_Q, x));
        if (shared) {
            g.step(ggml_cpy(g.c, var(g, d.z, N, d.n), x));
            g.out(mm(g, m, W_K, x));
        } else {
            g.out(mm(g, m, W_K, var(g, d.z, N, d.n)));
        }
    } },
    { "another tensor copied into a graph input between", 0, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = g.in(d.x, N, d.n);
        g.out(mm(g, m, W_Q, x));
        if (shared) {
            g.step(ggml_cpy(g.c, g.in(d.z, N, d.n), x));
            g.out(mm(g, m, W_K, x));
        } else {
            g.out(mm(g, m, W_K, g.in(d.z, N, d.n)));
        }
    } },
    { "input is a view, the whole tensor scaled in place between", 0, -1, [](graph & g, const model & m, const data & d, bool shared) {
        if (shared) {
            ggml_tensor * base = var(g, d.base, N, 2*d.n);
            ggml_tensor * x    = ggml_view_2d(g.c, base, N, d.n, base->nb[1], d.n*base->nb[1]);
            g.out(mm(g, m, W_Q, x));
            g.step(ggml_scale_inplace(g.c, base, 2.0f));
            g.out(mm(g, m, W_K, x));
        } else {
            g.out(mm(g, m, W_Q, var(g, scaled(d.base, 1.0f, N*d.n), N, d.n)));
            g.out(mm(g, m, W_K, var(g, scaled(d.base, 2.0f, N*d.n), N, d.n)));
        }
    } },
    { "input is a view, the other half of the tensor scaled in place between", 1, -1, [](graph & g, const model & m, const data & d, bool shared) {
        if (shared) {
            ggml_tensor * base = var(g, d.base, N, 2*d.n);
            ggml_tensor * lo   = ggml_view_2d(g.c, base, N, d.n, base->nb[1], 0);
            ggml_tensor * x    = ggml_view_2d(g.c, base, N, d.n, base->nb[1], d.n*base->nb[1]);
            g.out(mm(g, m, W_Q, x));
            g.step(ggml_scale_inplace(g.c, lo, 2.0f));
            g.out(mm(g, m, W_K, x));
            g.out(mm(g, m, W_V, lo));
        } else {
            g.out(mm(g, m, W_Q, var(g, scaled(d.base, 1.0f, N*d.n), N, d.n)));
            g.out(mm(g, m, W_K, var(g, scaled(d.base, 1.0f, N*d.n), N, d.n)));
            g.out(mm(g, m, W_V, var(g, scaled(d.base, 2.0f, 0, N*d.n), N, d.n)));
        }
    } },
    // a second run with unchanged shapes is a captured graph unless GGML_CUDA_DISABLE_GRAPHS is set: nothing is kept there
    { "the same graph a second time with other data", 1, 1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = g.in(d.x, N, d.n);
        g.out(mm(g, m, W_Q, x));
        g.out(mm(g, m, W_K, shared ? x : g.in(d.x, N, d.n)));
    } },
    { "up and gate of the routed experts", 1, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x   = per_token(g, var(g, d.x, N, d.n));
        ggml_tensor * ids = g.in_ids(d.sel);
        g.out(mm_id(g, m, W_UP_EXPS, x, ids));
        g.out(mm_id(g, m, W_GATE_EXPS, shared ? x : per_token(g, var(g, d.x, N, d.n)), ids));
    } },
    { "up and gate of the routed experts with other ids", 0, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x = per_token(g, var(g, d.x, N, d.n));
        g.out(mm_id(g, m, W_UP_EXPS, x, g.in_ids(d.sel)));
        g.out(mm_id(g, m, W_GATE_EXPS, shared ? x : per_token(g, var(g, d.x, N, d.n)), g.in_ids(d.sel2)));
    } },
    { "ids of the routed experts changed in place between", 0, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x   = per_token(g, var(g, d.x, N, d.n));
        ggml_tensor * ids = g.in_ids(d.sel);
        g.out(mm_id(g, m, W_UP_EXPS, x, ids));
        if (shared) {
            g.step(ggml_cpy(g.c, g.in(as_float(d.sel2), N_USED, d.n), ids));
            g.out(mm_id(g, m, W_GATE_EXPS, x, ids));
        } else {
            g.out(mm_id(g, m, W_GATE_EXPS, per_token(g, var(g, d.x, N, d.n)), g.in_ids(d.sel2)));
        }
    } },
    { "input of the routed experts scaled in place between", 0, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_tensor * x   = var(g, d.x, N, d.n);
        ggml_tensor * x3  = per_token(g, x);
        ggml_tensor * ids = g.in_ids(d.sel);
        g.out(mm_id(g, m, W_UP_EXPS, x3, ids));
        if (shared) {
            g.step(ggml_scale_inplace(g.c, x, 0.5f));
            g.out(mm_id(g, m, W_GATE_EXPS, x3, ids));
        } else {
            g.out(mm_id(g, m, W_GATE_EXPS, per_token(g, var(g, scaled(d.x, 0.5f), N, d.n)), ids));
        }
    } },
    { "feed-forward of a block", 2, -1, [](graph & g, const model & m, const data & d, bool shared) {
        ggml_context * c = g.c;
        ggml_tensor *  x = var(g, d.x, N, d.n);
        auto copy = [&]() { return shared ? x : var(g, d.x, N, d.n); };
        // build_moe_ffn() with the settings of qwen35moe
        ggml_tensor * probs    = ggml_soft_max(c, mm(g, m, W_ROUTER, x));
        ggml_tensor * selected = ggml_argsort_top_k(c, probs, N_USED);
        probs = ggml_reshape_3d(c, probs, 1, N_EXP, d.n);
        ggml_tensor * weights = ggml_reshape_2d(c, ggml_get_rows(c, probs, selected), N_USED, d.n);
        weights = ggml_div(c, weights, ggml_clamp(c, ggml_sum_rows(c, weights), 6.103515625e-5, INFINITY));
        weights = ggml_reshape_3d(c, weights, 1, N_USED, d.n);
        ggml_build_forward_expand(g.gf, weights);
        ggml_tensor * x3   = per_token(g, x);
        ggml_tensor * up   = mm_id(g, m, W_UP_EXPS, x3, selected);
        ggml_tensor * gate = mm_id(g, m, W_GATE_EXPS, shared ? x3 : per_token(g, copy()), selected);
        ggml_tensor * experts = ggml_mul_mat_id(c, m.w[W_DOWN_EXPS], ggml_swiglu_split(c, gate, up), selected);
        g.out(ggml_mul(c, experts, weights));
        // build_ffn() of the shared expert
        ggml_tensor * tmp = mm(g, m, W_UP_SHEXP, copy());
        ggml_tensor * cur = mm(g, m, W_GATE_SHEXP, copy());
        g.out(mm(g, m, W_DOWN_SHEXP, ggml_swiglu_split(c, cur, tmp)));
    } },
};

int main(int argc, char ** argv) {
    check(argc == 2 || argc == 3, "usage: q35moe-hip-qshare ABS_LIBDIR [BACKEND]");
    const auto is_set = [](const char * name) { const char * p = std::getenv(name); return p && std::strcmp(p, "1") == 0; };
    const bool on_cpu = argc == 3 && std::strcmp(argv[2], "CPU") == 0;
    const bool on     = is_set("GGML_Q35MOE_QSHARE") && !on_cpu;
    const bool graphs = std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr; // a second run of a graph is captured
    check(on_cpu || is_set("GGML_Q35MOE_EXPTAIL"), "set GGML_Q35MOE_EXPTAIL=1");
    check(!on || is_set("GGML_Q35MOE_QSHARE_LOG"), "set GGML_Q35MOE_QSHARE_LOG=1 with GGML_Q35MOE_QSHARE=1");

    ggml_log_set(log_callback, nullptr);
    ggml_backend_load_all_from_path(argv[1]);
    model m;
    m.cpu = ggml_backend_init_by_name("CPU", nullptr);
    m.be  = on_cpu ? m.cpu : ggml_backend_init_by_name(argc == 3 ? argv[2] : "ROCm0", nullptr);
    check(m.be && m.cpu, "backends");

    std::mt19937_64 rng(2048);
    ggml_context * wctx = ggml_init({(N_W + 1)*ggml_tensor_overhead(), nullptr, true});
    for (int j = 0; j < N_W; ++j) {
        m.w[j] = ggml_new_tensor_3d(wctx, defs[j].type, defs[j].ne[0], defs[j].ne[1], defs[j].ne[2]);
        ggml_set_name(m.w[j], defs[j].name);
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, m.be);
    check(wbuf != nullptr, "allocation");
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int j = 0; j < N_W; ++j) {
        const std::vector<char> bytes = random_weight(defs[j].type, ggml_nbytes(m.w[j]), rng);
        ggml_backend_tensor_set(m.w[j], bytes.data(), 0, bytes.size());
    }

    std::normal_distribution<float> nd(0.0f, 1.0f);
    // the kept buffers grow from the first size to the second and are larger than needed for the third and fourth
    static const int64_t sizes[] = { 16, 512, 100, 9 };
    uint64_t hash    = 0xcbf29ce484222325ull;
    long     n_diff  = 0;
    long     n_count = 0;
    for (const int64_t n : sizes) {
        data d;
        d.n = n;
        auto fill = [&](std::vector<float> & v, size_t count) {
            v.resize(count);
            for (size_t i = 0; i < count; ++i) {
                // every token has its own size
                const size_t tok = i/N;
                v[i] = (0.2f + 0.1f*(tok % 7))*nd(rng);
            }
        };
        fill(d.x, N*n);
        fill(d.z, N*n);
        fill(d.base, 2*N*n);
        auto pick = [&](std::vector<int32_t> & v) {
            v.resize(N_USED*n);
            for (int64_t t = 0; t < n; ++t) {
                // N_USED different experts per token, few experts in all so that tiles fill up
                const int span = n >= 256 ? N_EXP : 24;
                for (int i = 0; i < N_USED; ++i) {
                    bool again = true;
                    while (again) {
                        v[t*N_USED + i] = (int32_t) (rng() % span)*(N_EXP/span);
                        again = false;
                        for (int j = 0; j < i; ++j) { again = again || v[t*N_USED + j] == v[t*N_USED + i]; }
                    }
                }
            }
        };
        pick(d.sel);
        pick(d.sel2);

        for (const test_case & tc : cases) {
            graph ref;
            tc.build(ref, m, d, false);
            g_kept = g_taken = 0;
            std::vector<float> want = run(m, ref);
            if (tc.taken_again >= 0) {
                // the reference of the second run: its own graph with z in every input
                data d2 = d;
                d2.x = d.z;
                graph ref2;
                tc.build(ref2, m, d2, false);
                const std::vector<float> want2 = run(m, ref2);
                want.insert(want.end(), want2.begin(), want2.end());
            }
            const int taken_ref = g_taken;

            graph shared;
            tc.build(shared, m, d, true);
            g_kept = g_taken = 0;
            const std::vector<float> got = run(m, shared, tc.taken_again >= 0 ? &d.z : nullptr);
            const int taken = g_taken, kept = g_kept;

            check(want.size() == got.size() && !got.empty(), "result size");
            long diff = 0, first = -1;
            for (size_t i = 0; i < got.size(); ++i) {
                check(std::isfinite(got[i]) && std::isfinite(want[i]), "finite");
                if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
                    first = first < 0 ? (long) i : first;
                    diff++;
                }
            }
            for (size_t i = 0; i < got.size()*sizeof(float); ++i) {
                hash = (hash ^ ((const unsigned char *) got.data())[i])*0x100000001b3ull;
            }
            const int  expect   = on ? tc.taken + (tc.taken_again >= 0 && !graphs ? tc.taken_again : 0) : 0;
            const bool count_ok = taken == expect && taken_ref == 0;
            std::printf("%4lld tokens, %s: %ld/%zu differ, taken %d (expected %d), kept %d, taken in the reference graph %d%s",
                (long long) n, tc.name, diff, got.size(), taken, expect, kept, taken_ref, count_ok ? "" : " COUNT");
            if (first >= 0) {
                std::printf(", first at %ld: %a reference, %a shared", first, want[first], got[first]);
            }
            std::printf("\n");
            n_diff  += diff;
            n_count += !count_ok;
        }
    }
    std::printf("hash %016llx\n", (unsigned long long) hash);
    check(n_diff == 0, "a shared graph differs from its reference");
    check(n_count == 0, "unexpected number of products that took a kept copy");
    std::printf("PASS switch %s, a second run of a graph %s\n", on ? "on" : "off", graphs ? "is captured" : "runs node by node");
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    if (m.be != m.cpu) {
        ggml_backend_free(m.be);
    }
    ggml_backend_free(m.cpu);
    return 0;
}

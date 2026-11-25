// Rollback of the recurrent state with several sequences in one context (llama_context_params::n_rs_seq).
// The test sends the batches of a server that verifies drafts: per sequence 1 token + drafts with logits for all, then llama_memory_seq_rm() of the rejected drafts.
// Each sequence first runs alone in a context with one sequence. Then all sequences run in one context and every output must be the one of the run alone.
// The state of another sequence or step moves the logits of the test model by 1 or more, another batch shape by less than 1e-3.
//
//   test-recurrent-rollback -m MODEL [-n N]     N random schedules after the two fixed ones (default 40)
//   MODEL: test-llama-archs -a "^qwen35moe$" -s 4711 -d 0.5 -o DIR  ->  DIR/qwen35moe-moe.gguf
//
// A schedule is NSEQ:OP,OP,... An op is a batch of items joined by '+' (one llama_decode, then the rollbacks in item order) or one of e, k, l, r:
//   vS.K.A   K tokens of sequence S with logits for all, A drafts accepted, the other K - 1 - A are removed
//   pS.N     N prompt tokens of sequence S, logits for the last one
//   eS       sequence S ends, a new one starts in its place
//   kS, lS   save and restore a checkpoint of S (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
//   rS.M     one more rollback of M tokens of S, the memory must answer as for the sequence alone
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static const uint32_t N_RS_SEQ = 3;
static const double   TOL      = 1e-2;

// find_slot() moved the cell of sequence 0 and left rs_max behind: its rollback was refused (the server aborts there)
static const char * SCHEDULE_REFUSED = "3:p1.8,p0.8,p2.8,v0.2.0+v1.4.3+v2.4.3";
// the stale rs_max was large enough: sequence 0 took a snapshot of sequence 2, no error
static const char * SCHEDULE_SILENT  = "3:p1.8,p0.8,p2.8,v2.4.3,v0.2.0+v1.4.3+v2.4.3,v0.2.1+v1.2.0+v2.2.0";

[[noreturn]] static void die(const std::string & what) {
    printf("FAIL: %s\n", what.c_str());
    exit(1);
}

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

struct item {
    int  seq;
    char kind; // 'v' or 'p'
    int  k;
    int  a;
};

struct op {
    char              type; // 'b' (batch), 'e', 'k', 'l', 'r'
    std::vector<item> items;
    int               seq;
    int               m;
};

static std::vector<op> parse_ops(const std::string & s, const int n_seq) {
    std::vector<op> ops;
    std::stringstream ss(s);
    std::string o;
    while (std::getline(ss, o, ',')) {
        op cur = { 'b', {}, -1, 0 };
        if (o[0] == 'e' || o[0] == 'k' || o[0] == 'l') {
            cur.type = o[0];
            cur.seq  = atoi(o.c_str() + 1);
        } else if (o[0] == 'r') {
            cur.type = 'r';
            if (sscanf(o.c_str(), "r%d.%d", &cur.seq, &cur.m) != 2 || cur.m < 1) {
                die("op " + o);
            }
        } else {
            std::stringstream si(o);
            std::string it;
            while (std::getline(si, it, '+')) {
                item x = { -1, it[0], 0, 0 };
                const bool ok = it[0] == 'v' ? sscanf(it.c_str(), "v%d.%d.%d", &x.seq, &x.k, &x.a) == 3 && x.a >= 0 && x.a < x.k :
                                it[0] == 'p' ? sscanf(it.c_str(), "p%d.%d", &x.seq, &x.k) == 2 : false;
                if (!ok || x.k < 1 || x.seq < 0 || x.seq >= n_seq) {
                    die("item " + it);
                }
                cur.items.push_back(x);
            }
        }
        if (cur.type != 'b' && (cur.seq < 0 || cur.seq >= n_seq)) {
            die("op " + o);
        }
        ops.push_back(cur);
    }
    return ops;
}

// batches of a random part of the sequences: verification steps of 1 to n_rs + 1 tokens with a random accept count, prompt batches in the same batch,
// sequences that end and start again, checkpoints, extra rollbacks. The sequences start with a prompt each, in a random order
static std::string random_ops(const uint64_t seed, const int n_seq, const int n_steps, const int n_rs) {
    uint64_t st = mix64(seed * 2654435761ull + 17);
    auto rnd = [&](int n) { st = mix64(st); return (int) (st % (uint64_t) n); };
    const int p_max = 40;
    std::vector<int>  pos(n_seq, 0);
    std::vector<bool> has_ckpt(n_seq, false);
    std::vector<int>  last_k(n_seq, 0); // tokens of the last batch of the sequence that can still be rolled back
    std::string out;
    auto add = [&](const std::string & o) { out += (out.empty() ? "" : ",") + o; };
    std::vector<int> order(n_seq);
    for (int i = 0; i < n_seq; ++i) {
        order[i] = i;
    }
    for (int i = n_seq - 1; i > 0; --i) {
        std::swap(order[i], order[rnd(i + 1)]);
    }
    for (int i = 0; i < n_seq; ++i) {
        const int n = 1 + rnd(p_max);
        add("p" + std::to_string(order[i]) + "." + std::to_string(n));
        pos[order[i]] = n;
    }
    for (int step = 0; step < n_steps; ++step) {
        const int what = rnd(100);
        const int s    = rnd(n_seq);
        if (what < 4) {
            add("e" + std::to_string(s));
            pos[s] = 0; has_ckpt[s] = false; last_k[s] = 0;
            continue;
        }
        if (what < 8 && pos[s] > 0) {
            add("k" + std::to_string(s));
            has_ckpt[s] = true;
            continue;
        }
        if (what < 11 && has_ckpt[s]) {
            add("l" + std::to_string(s));
            last_k[s] = 0;
            continue;
        }
        if (what < 13 && last_k[s] > 0) {
            add("r" + std::to_string(s) + "." + std::to_string(1 + rnd(n_rs)));
            last_k[s] = 0;
            continue;
        }
        // every sequence takes part with probability 1/2 to 1
        const int p_in = 50 + rnd(51);
        std::string b;
        for (int q = 0; q < n_seq; ++q) {
            if (rnd(100) >= p_in && !(b.empty() && q == n_seq - 1)) {
                continue;
            }
            if (!b.empty()) {
                b += "+";
            }
            if (pos[q] == 0 || rnd(100) < 12) {
                const int k = rnd(4) == 0 ? 1 + rnd(p_max) : 1 + rnd(8);
                b += "p" + std::to_string(q) + "." + std::to_string(k);
                pos[q] += k; last_k[q] = 0;
            } else {
                const int k = 1 + rnd(n_rs + 1);
                const int a = rnd(k);
                b += "v" + std::to_string(q) + "." + std::to_string(k) + "." + std::to_string(a);
                pos[q] += 1 + a; last_k[q] = a + 1 < k ? 0 : k;
            }
        }
        add(b);
    }
    // one more step of each sequence, alone in its batch
    for (int q = 0; q < n_seq; ++q) {
        if (pos[q] > 0) {
            add("v" + std::to_string(q) + ".2.1");
        }
    }
    return out;
}

// an output of a sequence: the logits of a token, or with pos -1 the answer of the memory to an r op
struct out_rec {
    int                pos;
    llama_token        tok;
    std::vector<float> logits;
};

struct seq_state {
    int                  pos   = 0; // tokens kept
    uint64_t             draws = 0; // tokens drawn so far, kept and rejected ones
    bool                 has_ckpt = false;
    int                  ckpt_pos = 0;
    std::vector<uint8_t> ckpt;
};

struct result {
    bool        refused = false;
    std::string refused_at;
    double      max_diff = 0.0;
};

struct runner {
    llama_model *        model   = nullptr;
    int                  n_vocab = 0;
    uint64_t             seed    = 1;
    llama_context_params cparams;

    llama_token draw(const int seq, seq_state & st, const bool wrong, const llama_token other = -1) const {
        const uint64_t i = st.draws++;
        llama_token t = (llama_token) (mix64(seed * 1000003ull + (uint64_t) seq * 7919ull + i * 2ull + (wrong ? 1 : 0)) % (uint64_t) n_vocab);
        if (wrong && t == other) {
            t = (t + 1) % n_vocab;
        }
        return t;
    }

    llama_context * new_ctx(const int n_seq) const {
        llama_context_params p = cparams;
        p.n_seq_max = n_seq;
        llama_context * ctx = llama_init_from_model(model, p);
        if (!ctx || llama_n_rs_seq(ctx) != N_RS_SEQ) {
            die("context with n_rs_seq");
        }
        return ctx;
    }

    // only >= 0: this sequence alone, as sequence 0 of the context, its outputs go to recs. only = -1: all sequences, their outputs are compared with recs
    result run(llama_context * ctx, const std::vector<op> & ops, const int n_seq, const int only, std::vector<std::vector<out_rec>> & recs) const {
        result res;
        llama_memory_t mem = llama_get_memory(ctx);
        std::vector<seq_state> st(n_seq);
        std::vector<size_t> n_cmp(n_seq, 0);
        llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
        auto cs = [&](int s) { return only >= 0 ? 0 : s; };
        auto in = [&](int s) { return only < 0 || s == only; };
        auto next_rec = [&](int s) -> const out_rec & {
            if (n_cmp[s] >= recs[s].size()) {
                die("more outputs than the sequence alone");
            }
            return recs[s][n_cmp[s]++];
        };
        for (size_t io = 0; io < ops.size() && !res.refused; ++io) {
            const op & o = ops[io];
            if (o.type == 'b') {
                batch.n_tokens = 0;
                for (const item & x : o.items) {
                    if (!in(x.seq)) {
                        continue;
                    }
                    seq_state & q = st[x.seq];
                    for (int i = 0; i < x.k; ++i) {
                        llama_token t = draw(x.seq, q, false);
                        if (x.kind == 'v' && i > x.a) {
                            // a rejected draft
                            t = draw(x.seq, q, true, t);
                        }
                        const int j = batch.n_tokens++;
                        batch.token[j]     = t;
                        batch.pos[j]       = q.pos + i;
                        batch.n_seq_id[j]  = 1;
                        batch.seq_id[j][0] = cs(x.seq);
                        batch.logits[j]    = x.kind == 'v' || i == x.k - 1;
                    }
                }
                if (batch.n_tokens == 0) {
                    continue;
                }
                if (llama_decode(ctx, batch) != 0) {
                    die("llama_decode");
                }
                for (int j = 0; j < batch.n_tokens; ++j) {
                    if (!batch.logits[j]) {
                        continue;
                    }
                    const int     s = only >= 0 ? only : batch.seq_id[j][0];
                    const float * v = llama_get_logits_ith(ctx, j);
                    if (only >= 0) {
                        recs[s].push_back({ batch.pos[j], batch.token[j], std::vector<float>(v, v + n_vocab) });
                        continue;
                    }
                    const out_rec & r = next_rec(s);
                    if (r.pos != batch.pos[j] || r.tok != batch.token[j]) {
                        die("the runs are not aligned");
                    }
                    for (int e = 0; e < n_vocab; ++e) {
                        const double d = fabs((double) r.logits[e] - (double) v[e]);
                        res.max_diff = std::max(res.max_diff, std::isfinite(d) ? d : 1e30);
                    }
                }
                // the server calls seq_rm after every verification step, in the order of the slots, also when nothing was rejected
                for (const item & x : o.items) {
                    if (!in(x.seq)) {
                        continue;
                    }
                    seq_state & q = st[x.seq];
                    const int keep = x.kind == 'v' ? 1 + x.a : x.k;
                    if (x.kind == 'v' && !llama_memory_seq_rm(mem, cs(x.seq), q.pos + keep, -1)) {
                        res.refused    = true;
                        res.refused_at = "op " + std::to_string(io) + ": llama_memory_seq_rm(seq " + std::to_string(x.seq) + ", p0 " +
                            std::to_string(q.pos + keep) + ", -1) refused a rollback of " + std::to_string(x.k - keep) + " tokens";
                        break;
                    }
                    q.pos += keep;
                }
            } else if (in(o.seq)) {
                seq_state & q = st[o.seq];
                if (o.type == 'e') {
                    if (!llama_memory_seq_rm(mem, cs(o.seq), -1, -1)) {
                        die("seq_rm of a whole sequence");
                    }
                    q.pos = 0; q.has_ckpt = false;
                } else if (o.type == 'k') {
                    const size_t n = llama_state_seq_get_size_ext(ctx, cs(o.seq), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    q.ckpt.resize(n);
                    if (llama_state_seq_get_data_ext(ctx, q.ckpt.data(), n, cs(o.seq), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != n) {
                        die("checkpoint save");
                    }
                    q.has_ckpt = true; q.ckpt_pos = q.pos;
                } else if (o.type == 'l') {
                    if (!q.has_ckpt || llama_state_seq_set_data_ext(ctx, q.ckpt.data(), q.ckpt.size(), cs(o.seq), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
                        die("checkpoint restore");
                    }
                    if (!llama_memory_seq_rm(mem, cs(o.seq), q.ckpt_pos, -1)) {
                        die("seq_rm after a checkpoint restore");
                    }
                    q.pos = q.ckpt_pos;
                } else {
                    // accepted or refused: the run alone says what is right
                    const bool ok = o.m <= q.pos - 1 && llama_memory_seq_rm(mem, cs(o.seq), q.pos - o.m, -1);
                    if (ok) {
                        q.pos -= o.m;
                    }
                    if (only >= 0) {
                        recs[o.seq].push_back({ -1, ok ? 1 : 0, {} });
                    } else {
                        const out_rec & r = next_rec(o.seq);
                        if (r.pos != -1) {
                            die("the runs are not aligned");
                        }
                        if (r.tok != (ok ? 1 : 0)) {
                            res.refused    = true;
                            res.refused_at = "op " + std::to_string(io) + ": the extra rollback of sequence " + std::to_string(o.seq) + " is " +
                                (ok ? "accepted" : "refused") + " with the other sequences and " + (r.tok ? "accepted" : "refused") + " alone";
                        }
                    }
                }
            }
        }
        llama_batch_free(batch);
        return res;
    }
};

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int n_random = 40;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_random = atoi(argv[++i]);
        } else {
            model_path = nullptr;
            break;
        }
    }
    if (model_path == nullptr) {
        printf("usage: %s -m MODEL [-n N]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    // on the CPU: there the same tokens in another batch differ in the last bits only
    ggml_backend_dev_t no_devices[1] = { nullptr };
    llama_model_params mparams = llama_model_default_params();
    mparams.devices      = no_devices;
    mparams.n_gpu_layers = 0;

    runner R;
    R.model = llama_model_load_from_file(model_path, mparams);
    if (!R.model) {
        die("model");
    }
    R.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(R.model));
    R.cparams = llama_context_default_params();
    R.cparams.n_ctx           = 16384;
    R.cparams.n_batch         = 2048;
    R.cparams.n_ubatch        = 512;
    R.cparams.kv_unified      = true;
    R.cparams.n_threads       = 4;
    R.cparams.n_threads_batch = 4;
    R.cparams.n_rs_seq        = N_RS_SEQ;

    struct schedule {
        std::string name;
        int         n_seq;
        std::string ops;
        uint64_t    seed;
    };
    std::vector<schedule> list;
    for (const char * s : { SCHEDULE_REFUSED, SCHEDULE_SILENT }) {
        list.push_back({ s, atoi(s), strchr(s, ':') + 1, 1 });
    }
    for (int i = 1; i <= n_random; ++i) {
        // 2 to 8 sequences
        const int n_seq = 2 + (int) (mix64((uint64_t) i * 77 + 5) % 7);
        list.push_back({ "random " + std::to_string(i), n_seq, random_ops(i, n_seq, 60, N_RS_SEQ), (uint64_t) i });
    }

    int    n_fail   = 0;
    double max_diff = 0.0;
    llama_context * ctx_one = R.new_ctx(1);
    for (const schedule & sc : list) {
        const std::vector<op> ops = parse_ops(sc.ops, sc.n_seq);
        R.seed = sc.seed;
        std::vector<std::vector<out_rec>> recs(sc.n_seq);
        for (int s = 0; s < sc.n_seq; ++s) {
            llama_memory_clear(llama_get_memory(ctx_one), true);
            const result r = R.run(ctx_one, ops, sc.n_seq, s, recs);
            if (r.refused) {
                die("sequence " + std::to_string(s) + " alone: " + r.refused_at);
            }
        }
        llama_context * ctx = R.new_ctx(sc.n_seq);
        const result r = R.run(ctx, ops, sc.n_seq, -1, recs);
        llama_free(ctx);
        max_diff = std::max(max_diff, r.max_diff);
        if (r.refused || r.max_diff > TOL) {
            n_fail++;
            printf("FAIL %s: %d sequences, largest logit difference to the sequences alone %g%s%s\n  %s\n", sc.name.c_str(), sc.n_seq, r.max_diff,
                r.refused ? ", " : "", r.refused_at.c_str(), sc.ops.c_str());
        }
    }
    llama_free(ctx_one);
    llama_model_free(R.model);
    llama_backend_free();

    printf("%zu schedules, %d failed, largest logit difference %g (limit %g)\n", list.size(), n_fail, max_diff, TOL);
    return n_fail == 0 ? 0 : 1;
}

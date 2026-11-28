#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <numeric>
#include <random>
#include <string>

static void init_model(llama_model & model, uint32_t n_layer) {
    model.hparams = {};
    model.hparams.n_layer_all = n_layer;
    model.hparams.n_embd_head_k_full = 4;
    model.hparams.n_embd_head_v_full = 4;
    model.hparams.n_head_arr.fill(1);
    model.hparams.n_head_kv_arr.fill(1);
    std::iota(model.hparams.kv_source_impl.begin(), model.hparams.kv_source_impl.end(), 0);
    model.layers.resize(n_layer);
}

// llama_kv_cells::seq_rm_pos() leaves the cells as the scan over all cells in llama_kv_cache::seq_rm() does
static void test_seq_rm_pos() {
    const uint32_t n_cells = 700;
    const int      n_seq   = 4;

    std::mt19937 rng(1234);

    for (int it = 0; it < 300; ++it) {
        llama_kv_cells a;
        a.resize(n_cells);

        // empty cells, cells without a sequence, shared cells and repeated positions
        for (uint32_t i = 0; i < n_cells; ++i) {
            if (rng() % 5 == 0) {
                continue;
            }
            a.pos_set(i, rng() % 250);
            for (int s = 0; s < n_seq; ++s) {
                if (rng() % 3 == 0) {
                    a.seq_add(i, s);
                }
            }
        }

        llama_kv_cells b = a;

        for (int k = 0; k < 6; ++k) {
            const llama_seq_id seq_id = rng() % n_seq;

            llama_pos p0 = (llama_pos) (rng() % 270) - 10;
            llama_pos p1 = rng() % 4 == 0 ? -1 : (llama_pos) (rng() % 270) - 10;

            // as in llama_kv_cache::seq_rm()
            p0 = std::max(p0, 0);
            p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;

            uint32_t head_a = n_cells;
            for (uint32_t i = 0; i < n_cells; ++i) {
                if (a.pos_in(i, p0, p1) && a.seq_has(i, seq_id) && a.seq_rm(i, seq_id) && head_a == n_cells) {
                    head_a = i;
                }
            }

            const uint32_t head_b = b.seq_rm_pos(seq_id, p0, p1);

            GGML_ASSERT(head_a == head_b);
            GGML_ASSERT(a.get_used() == b.get_used() && a.used_min() == b.used_min() && a.used_max_p1() == b.used_max_p1());

            for (uint32_t i = 0; i < n_cells; ++i) {
                GGML_ASSERT(a.is_empty(i) == b.is_empty(i));
                if (!a.is_empty(i)) {
                    GGML_ASSERT(a.pos_get(i) == b.pos_get(i) && a.seq_get_all(i) == b.seq_get_all(i));
                }
            }

            for (int s = 0; s < n_seq; ++s) {
                GGML_ASSERT(a.seq_pos_get(s) == b.seq_pos_get(s));

                std::vector<llama_kv_rows_slice> slices(1);
                slices[0].seq_ids = { s };

                std::vector<int32_t> rows_a(n_cells), rows_b(n_cells);
                a.get_seq_cells(slices, rows_a.data(), n_cells);
                b.get_seq_cells(slices, rows_b.data(), n_cells);
                GGML_ASSERT(rows_a == rows_b);
            }
        }
    }
}

// a ubatch of tokens {seq, second seq or -1, pos, y, x}, the fields that the cache reads; n_pos 4 gives M-RoPE positions (t, y, x, 0)
struct test_ubatch {
    std::vector<llama_token>    token;
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_data;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id>   seq_unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         output;

    llama_ubatch ub = {};

    test_ubatch(const std::vector<std::array<int, 5>> & toks, uint32_t n_pos) {
        const uint32_t n = toks.size();
        token.assign(n, 1);
        pos.assign(n*n_pos, 0);
        n_seq_id.resize(n);
        seq_data.assign(2*n, 0);
        seq_id.resize(n);
        output.assign(n, 1);
        seq_idx.assign(LLAMA_MAX_SEQ, -1);
        for (uint32_t i = 0; i < n; ++i) {
            pos[i] = toks[i][2];
            if (n_pos == 4) {
                pos[i + n]   = toks[i][3];
                pos[i + n*2] = toks[i][4];
            }
            seq_data[2*i]     = toks[i][0];
            seq_data[2*i + 1] = toks[i][1];
            n_seq_id[i]      = toks[i][1] >= 0 ? 2 : 1;
            seq_id[i]        = &seq_data[2*i];
            for (int k = 0; k < n_seq_id[i]; ++k) {
                if (std::find(seq_unq.begin(), seq_unq.end(), seq_id[i][k]) == seq_unq.end()) {
                    seq_unq.push_back(seq_id[i][k]);
                }
            }
        }
        std::sort(seq_unq.begin(), seq_unq.end());
        for (size_t k = 0; k < seq_unq.size(); ++k) {
            seq_idx[seq_unq[k]] = k;
        }
        ub.n_tokens     = n;
        ub.n_seq_tokens = n;
        ub.n_seqs       = 1;
        ub.n_seqs_unq   = seq_unq.size();
        ub.n_pos        = n_pos;
        ub.token        = token.data();
        ub.pos          = pos.data();
        ub.n_seq_id     = n_seq_id.data();
        ub.seq_id       = seq_id.data();
        ub.seq_id_unq   = seq_unq.data();
        ub.seq_idx      = seq_idx.data();
        ub.output       = output.data();
    }
};

// the K/V row lists and the causal KQ mask of decode ubatches (one sequence, runs of several, interleaved, a token of two sequences,
// with M-RoPE: tokens at one position with different 2D positions) against their definition, on random cells with holes and shared cells
static void test_attn_mask_rows(uint32_t n_pos) {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    init_model(*model, 1);

    const int n_seq = 4;

    std::mt19937 rng(4321 + n_pos);

    llama_kv_cache kv(*model, true, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true, 1024, n_seq, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr);

    auto add = [&](const std::vector<std::array<int, 5>> & toks) {
        test_ubatch u(toks, n_pos);
        const auto sinfo = kv.find_slot(u.ub, false);
        if (!sinfo.empty()) {
            kv.apply_ubatch(sinfo, u.ub);
        }
        return !sinfo.empty();
    };

    auto next = [&](int s) { return std::max(kv.seq_pos_max(s) + 1, 0); };

    // a text token has y = x = pos; a block of an image has one position and a grid of 2D positions, in a random order
    auto text  = [&](int s, int s2, llama_pos p) { return std::array<int, 5>{ s, s2, p, p, p }; };
    auto image = [&](std::vector<std::array<int, 5>> & t, int s, llama_pos p, int n) {
        std::vector<int> k(n);
        std::iota(k.begin(), k.end(), 0);
        std::shuffle(k.begin(), k.end(), rng);
        for (int i : k) {
            t.push_back({ s, -1, p, p + i/3, p + i%3 });
        }
    };

    int n_checked = 0;

    for (int step = 0; step < 400; ++step) {
        const int op = rng() % 10;
        if (op < 3) {
            const int s = rng() % n_seq;
            const int n = 1 + rng() % 150;
            std::vector<std::array<int, 5>> t;
            if (n_pos == 4 && rng() % 3 == 0) {
                image(t, s, next(s), 1 + n % 9);
            } else {
                for (int i = 0; i < n; ++i) {
                    t.push_back(text(s, -1, next(s) + i));
                }
            }
            add(t);
        } else if (op < 5) {
            kv.seq_rm(rng() % 5 - 1, rng() % 300, rng() % 2 ? -1 : (llama_pos) (300 + rng() % 100));
        } else if (op < 6) {
            const int a = rng() % n_seq;
            const int b = rng() % n_seq;
            if (a != b) {
                kv.seq_rm(b, -1, -1);
                kv.seq_cp(a, b, -1, -1);
            }
        } else {
            std::vector<int> ss = { (int) (rng() % n_seq) };
            const int kind = rng() % (n_pos == 4 ? 5 : 4);
            if (kind > 0 && kind < 4) {
                for (int s = 0; s < n_seq; ++s) {
                    if (s != ss[0] && rng() % 2) {
                        ss.push_back(s);
                    }
                }
            }
            const int n = 1 + rng() % 5;
            std::vector<std::array<int, 5>> t;
            if (kind == 4) {
                image(t, ss[0], next(ss[0]), 2 + rng() % 6);
            } else if (kind == 3 && ss.size() > 1) {
                t.push_back(text(ss[0], ss[1], std::max(next(ss[0]), next(ss[1]))));
            } else if (kind == 2) {
                for (int i = 0; i < n; ++i) {
                    for (int s : ss) {
                        t.push_back(text(s, -1, next(s) + i));
                    }
                }
            } else {
                for (int s : ss) {
                    for (int i = 0; i < n; ++i) {
                        t.push_back(text(s, -1, next(s) + i));
                    }
                }
            }
            if (!add(t)) {
                continue;
            }

            test_ubatch u(t, n_pos);
            const llama_ubatch & ub = u.ub;

            const auto & cells = kv.get_cells(0);

            const uint32_t n_kv     = kv.get_n_kv_rows(ub);
            const uint32_t n_slices = llama_kv_cache::get_n_kv_slices(ub);
            const uint32_t n_tps    = ub.n_tokens/n_slices;

            for (bool block_uniform : { false, true }) {
                ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
                ggml_context_ptr ctx(ggml_init(ip));

                llm_attn_mask m;
                m.rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_kv, n_slices);
                m.raw  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, n_tps, 1, n_slices);
                m.n_kv = n_kv;
                m.block_uniform = block_uniform;

                ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
                ggml_backend_buffer_clear(buf.get(), 0x7e);

                kv.set_input_attn_mask(m, &ub, true);

                const int32_t     * rows = (const int32_t *) m.rows->data;
                const ggml_fp16_t * mask = (const ggml_fp16_t *) m.raw->data;

                for (uint32_t s = 0; s < n_slices; ++s) {
                    std::vector<llama_seq_id> seqs = { ub.seq_id[s*n_tps][0] };
                    if (n_slices == 1) {
                        seqs.assign(ub.seq_id_unq, ub.seq_id_unq + ub.n_seqs_unq);
                    }

                    uint32_t c = 0;
                    for (uint32_t j = 0; j < cells.size(); ++j) {
                        bool in = false;
                        for (llama_seq_id x : seqs) {
                            in |= !cells.is_empty(j) && cells.seq_has(j, x);
                        }
                        if (in) {
                            GGML_ASSERT(c < n_kv && rows[s*n_kv + c] == (int32_t) j);
                            c++;
                        }
                    }
                    for (; c < n_kv; ++c) {
                        GGML_ASSERT(rows[s*n_kv + c] == -1);
                    }

                    for (uint32_t i = s*n_tps; i < (s + 1)*n_tps; ++i) {
                        const llama_seq_id x = ub.seq_id[i][0];

                        uint32_t iq = i;
                        if (block_uniform) {
                            for (iq = 0; ub.seq_id[iq][0] != x; ++iq) {
                            }
                        }

                        const llama_pos p1 = ub.pos[iq];

                        for (c = 0; c < n_kv; ++c) {
                            const int32_t j = rows[s*n_kv + c];

                            bool keep = j >= 0 && cells.seq_has(j, x) && cells.pos_get(j) <= p1;
                            if (keep && n_pos == 4 && cells.pos_get(j) == p1) {
                                keep = !cells.ext_get(j).is_2d_gt(ub.pos[iq + ub.n_tokens*2], ub.pos[iq + ub.n_tokens]);
                            }

                            GGML_ASSERT(ggml_fp16_to_fp32(mask[i*n_kv + c]) == (keep ? 0.0f : -INFINITY));
                        }
                    }
                }

                n_checked++;
            }

            if (rng() % 2) {
                for (int s : ss) {
                    kv.seq_rm(s, t[0][2], -1);
                }
            }
        }
    }

    GGML_ASSERT(n_checked > 300);
}

int main() {
    test_seq_rm_pos();

    llama_backend_init();

    test_attn_mask_rows(1);
    test_attn_mask_rows(4);

    {
        std::string errors;
        llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
            if (level == GGML_LOG_LEVEL_ERROR) {
                *static_cast<std::string *>(user_data) += text;
            }
        }, &errors);

        for (llm_arch arch : { LLM_ARCH_DEEPSEEK2, LLM_ARCH_DEEPSEEK32, LLM_ARCH_DEEPSEEK4,
                LLM_ARCH_DEEPSEEK41, LLM_ARCH_GLM_DSA, LLM_ARCH_GLM5_NEXT, LLM_ARCH_DOTS3NOTE,
                LLM_ARCH_MISTRAL_LARGE4, LLM_ARCH_HY_V4, LLM_ARCH_MINIMAX_M3, LLM_ARCH_QWEN4EXP,
                LLM_ARCH_DFLASH }) {
            std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
            init_model(*model, 1);
            if (arch == LLM_ARCH_DEEPSEEK2) {
                model->hparams.n_embd_head_k_mla_impl = 4;
                model->hparams.n_embd_head_v_mla_impl = 4;
            }

            for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                    if (type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16) {
                        continue;
                    }
                    auto params = llama_context_default_params();
                    params.type_k = type_k;
                    params.type_v = type_v;
                    errors.clear();
                    llama_context * ctx = llama_init_from_model(model.get(), params);
                    GGML_ASSERT(ctx == nullptr);
                    if (type_k == type_v || (!model->hparams.is_mla() && arch != LLM_ARCH_DEEPSEEK4 && arch != LLM_ARCH_DEEPSEEK41)) {
                        GGML_ASSERT(errors.find("FP8 cache is not supported for architecture") != std::string::npos);
                    } else {
                        GGML_ASSERT(errors.find("model does not support different K") != std::string::npos);
                    }
                }
            }
        }

        llama_log_set(nullptr, nullptr);
    }

    {
        std::unique_ptr<llama_model> target(llama_model_create(LLM_ARCH_GEMMA4, llama_model_default_params()));
        std::unique_ptr<llama_model> assistant(llama_model_create(LLM_ARCH_GEMMA4_ASSISTANT, llama_model_default_params()));
        init_model(*target, 3);
        init_model(*assistant, 2);

        ggml_context_ptr ctx(ggml_init({ 4096, nullptr, false }));
        target->layers[2].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        target->layers[2].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[1].k_cache_s = assistant->layers[0].k_cache_s;
        assistant->layers[1].v_cache_s = assistant->layers[0].v_cache_s;
        *static_cast<float *>(target->layers[2].k_cache_s->data) = 0.5f;
        *static_cast<float *>(target->layers[2].v_cache_s->data) = 0.25f;
        *static_cast<float *>(assistant->layers[0].k_cache_s->data) = 2.0f;
        *static_cast<float *>(assistant->layers[0].v_cache_s->data) = 4.0f;

        for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
            for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                llama_kv_cache source(*target, false, target->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr,
                        [](uint32_t il) { return il == 0 ? 2 : -1; }, nullptr);
                llama_kv_cache shared(*assistant, false, assistant->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, &source, nullptr, nullptr,
                        [](int32_t il) { return il == 0 ? 2 : 1; });

                GGML_ASSERT(source.get_k_scale(2) == (type_k == GGML_TYPE_F8_E4M3 ? target->layers[2].k_cache_s : nullptr));
                GGML_ASSERT(source.get_v_scale(2) == (type_v == GGML_TYPE_F8_E4M3 ? target->layers[2].v_cache_s : nullptr));
                GGML_ASSERT(source.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(source.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_storage(0) == source.get_k_storage(2));
                GGML_ASSERT(shared.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(shared.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_scale(1) == nullptr);
                GGML_ASSERT(shared.get_v_scale(1) == nullptr);
            }
        }
    }

    llama_backend_free();
    return 0;
}

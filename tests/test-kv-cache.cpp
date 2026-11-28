#include "llama-kv-cache.h"
#include "llama-model.h"

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

int main() {
    test_seq_rm_pos();

    llama_backend_init();

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

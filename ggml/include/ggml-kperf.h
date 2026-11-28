#pragma once

#include "ggml.h"

#include <stddef.h>

// Private fast paths of the native gfx1103 HIP build: one table for all of them.
// A bit-exact entry is on by default. A lossy entry is off by default, its variable set to 1 turns it on.
// Set a lossy variable per model with the env key of the models.ini section, not for the whole service: a lossy entry acts on every architecture that reaches its path.
//   qwen3.5-4b:                                                      env = GPU_MAX_HW_QUEUES=1 GGML_QWEN35_Q6K_MMQ=1
//   Inkling-2-Mini with BF16 router and controller weights (D5-BF16):  env = GGML_INKLING_BF16_F32ACC=1
// Each path keeps its own guards (device, architecture, shapes); an entry that is on only allows the path.
// The host family is CPU code of libllama and the server: it acts with any backend and for every architecture, as the hip family does for the HIP build.
// Upstream variables that also keep paths off: GGML_CUDA_DISABLE_FUSION, GGML_CUDA_DISABLE_GRAPHS, GGML_CUDA_GRAPH_OPT (here any value but 0, upstream only 1), GGML_CUDA_ENABLE_UNIFIED_MEMORY, GGML_CUDA_CUBLAS_COMPUTE_TYPE, ROCBLAS_USE_HIPBLASLT, ROCBLAS_TENSILE_LIBPATH, ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH, LLAMA_KV_ROWS_DISABLE.
// Fork changes with no entry are always on, also with GGML_KPERF_OFF=all:
//   mmvq.cu: the RDNA2 parameter table and the gfx11-generic launch width for gfx1103
//   mmq.cu: the rolled k01 loop of the tail16 kernels (MMQ_PRIVATE_TAIL16_ROLL)
//   fattn.cu: the kernel choice for K/V row lists on WMMA, the tile kernel for D256 with GQA 2 and for D512 (Gemma 4), WMMA from 7 columns for D <= 128 (gpt-oss)
//   GGML_KPERF_OFF=all | none | <list>     turn entries off for A/B runs
//   GGML_KPERF_TRACE=all | <list>          log lines when a path runs (for fixtures)
// <list>: names, families or legacy variables, separated by ',' or spaces, case is ignored. An unknown name aborts.
// Turning off an entry also turns off the entries that need it.
// The old variables of the bit-exact entries are not read any more. A set one is counted in "legacy variables ignored" of the start-up lines: ggml_kperf (INFO, at the first use) and the server's "kperf: arch ..." line of each model instance.
#ifdef __cplusplus
extern "C" {
#endif

// X(id, name, exact, needs, legacy variable, description); an entry that needs another one comes after it; an entry added after the table has no legacy variable ("")
#define GGML_KPERF_LIST(X) \
    X(HIP_MOE_NCOLS32,       "hip.moe_ncols32",       true,  "",                    "GGML_CUDA_MMQ_MOE_NCOLS",      "MUL_MAT_ID on MMQ: tile width chosen for 32 columns per expert") \
    X(HIP_MMQ_DENSE_ORDER,   "hip.mmq_dense_order",   true,  "",                    "",                             "dense Q8_0 products on MMQ: the row tiles run for groups of 2 or 4 column tiles, the Q8_1 copy of a group stays in L2") \
    X(HIP_MMQ_DENSE_PIPE,    "hip.mmq_dense_pipe",    true,  "",                    "",                             "dense Q8_0 products of 128 or more columns on MMQ: 8 warps per 64 x 128 tile, K in register-pipelined stages of 128 (4 tiles per WGP), int32 to float by subtraction") \
    X(INKLING_GATHER,        "inkling.gather",        true,  "",                    "GGML_INKLING_GATHER",          "relative bias GET_ROWS/RESHAPE/PERMUTE/CONT in one kernel") \
    X(INKLING_MASKEPI,       "inkling.maskepi",       true,  "",                    "GGML_INKLING_MASKEPI",         "bias gather and raw mask in F32, one rounding into the F16 mask") \
    X(INKLING_RAWMASK,       "inkling.rawmask",       true,  "",                    "GGML_INKLING_RAWMASK",         "prefill mask from the original projection") \
    X(INKLING_PFMASK,        "inkling.pfmask",        true,  "inkling.rawmask",     "GGML_INKLING_PFMASK",          "prefill mask tile flags for the full-attention layers") \
    X(INKLING_ROWCLIP,       "inkling.rowclip",       true,  "",                    "GGML_INKLING_ROWCLIP",         "one-token Q/R/K row clip chain") \
    X(INKLING_SITU,          "inkling.situ",          true,  "",                    "GGML_INKLING_SITU",            "expert SITU activation on the packed gate/up") \
    X(INKLING_CTRL_TAIL,     "inkling.ctrl_tail",     true,  "",                    "GGML_INKLING_CTRL_TAIL",       "TANH/MUL/ADD tails of the loop controllers") \
    X(INKLING_GATE_TAIL16,   "inkling.gate_tail16",   true,  "hip.moe_ncols32",     "GGML_INKLING_GATE_TAIL16",     "expert gate/up: partial 32-column tiles with the 16-column body") \
    X(INKLING_DOWN_TAIL16,   "inkling.down_tail16",   true,  "hip.moe_ncols32",     "GGML_INKLING_DOWN_TAIL16",     "expert down: partial 32-column tiles with the 16-column body") \
    X(INKLING_HOTPAIR_GATE,  "inkling.hotpair_gate",  true,  "inkling.gate_tail16", "GGML_INKLING_HOTPAIR_GATE",    "expert gate/up: two full tiles share one weight unpack") \
    X(INKLING_HOTPAIR_DOWN,  "inkling.hotpair_down",  true,  "inkling.down_tail16", "GGML_INKLING_HOTPAIR_DOWN",    "expert down: two full tiles share one weight unpack") \
    X(INKLING_CONCAT4,       "inkling.concat4",       true,  "",                    "GGML_INKLING_CONCAT4",         "prefill depth mix: 4 states in one pack") \
    X(INKLING_DEPTHMIX_RAW,  "inkling.depthmix_raw",  true,  "",                    "GGML_INKLING_DEPTHMIX_RAW_V2", "prefill depth mix without the transpose") \
    X(INKLING_VIEWCOL,       "inkling.viewcol",       true,  "",                    "GGML_INKLING_VIEWCOL",         "columns of the mixed states as views") \
    X(INKLING_CONV1,         "inkling.conv1",         true,  "",                    "GGML_INKLING_CONV1",           "one-token short conv, residual and state in one op") \
    X(INKLING_BIAS1,         "inkling.bias1",         true,  "",                    "GGML_INKLING_BIAS1",           "one-token relative bias table in one op") \
    X(INKLING_MVSTACK,       "inkling.mvstack",       true,  "",                    "GGML_INKLING_MVSTACK",         "one-token Q8_0 products of one input in one op") \
    X(INKLING_CTRL1,         "inkling.ctrl1",         true,  "",                    "GGML_INKLING_CTRL1",           "one-token loop controller in one op") \
    X(INKLING_MIX1,          "inkling.mix1",          true,  "",                    "GGML_INKLING_MIX1",            "one-token depth mix in one op") \
    X(INKLING_FAGQA,         "inkling.fagqa",         true,  "",                    "GGML_INKLING_FAGQA",           "decode attention, one block takes the 3 query heads of a KV head") \
    X(INKLING_PPSKIP,        "inkling.ppskip",        true,  "",                    "GGML_INKLING_PPSKIP",          "prefill attention skips fully masked prefix tiles (finite K/V)") \
    X(INKLING_PFATTN,        "inkling.pfattn",        true,  "",                    "GGML_INKLING_PFATTN",          "prefill attention kernel for 256 queries") \
    X(INKLING_PFMM,          "inkling.pfmm",          true,  "",                    "GGML_INKLING_PFMM",            "prefill router/controller float products") \
    X(INKLING_BF16_F32ACC,   "inkling.bf16_f32acc",   false, "",                    "GGML_INKLING_BF16_F32ACC",     "BF16 weight with F32 accumulation takes the F32 kernels (MMVF, SGEMM); MMF and the BF16 GEMM round src1") \
    X(QWEN35_MVSHARE,        "qwen35.mvshare",        true,  "",                    "GGML_QWEN35_MVSHARE",          "one-token products that share one input") \
    X(QWEN35_PREP1,          "qwen35.prep1",          true,  "",                    "GGML_QWEN35_PREP1",            "one-token conv, SiLU, norm, beta and gate in one op") \
    X(QWEN35_GDN1,           "qwen35.gdn1",           true,  "",                    "GGML_QWEN35_GDN1",             "one-token delta net in one op") \
    X(QWEN35_POST1,          "qwen35.post1",          true,  "",                    "GGML_QWEN35_POST1",            "one-token gated norm in one op") \
    X(QWEN35_CONVPF,         "qwen35.convpf",         true,  "",                    "GGML_QWEN35_CONVPF",           "prefill conv without the concat tensor") \
    X(QWEN35_FAGQA,          "qwen35.fagqa",          true,  "",                    "GGML_QWEN35_FAGQA",            "one-token attention, D256, 4 query heads per KV head") \
    X(QWEN35_PFATTN,         "qwen35.pfattn",         true,  "",                    "GGML_QWEN35_PFATTN",           "prefill attention, D256, 4 query heads per KV head") \
    X(QWEN35_PFLOAD,         "qwen35.pfload",         true,  "qwen35.pfattn",       "",                             "inside pfattn: the row numbers of a warp in one 16-byte load, the second K half tile loaded before the QK products") \
    X(QWEN35_PFQK,           "qwen35.pfqk",           true,  "qwen35.pfload",       "",                             "inside pfload: K in two stages of 128 elements, one Q fragment read for the QK products of both row slices of a warp") \
    X(QWEN35_PFBAR,          "qwen35.pfbar",          true,  "qwen35.pfqk",         "",                             "inside pfqk: the barriers of the K/V loop only order shared memory, the global loads of the next half tile stay in flight") \
    X(QWEN35_Q6K_MMQ,        "qwen35.q6k_mmq",        false, "",                    "GGML_QWEN35_Q6K_MMQ",          "Q6_K takes MMQ at any batch size (other rounding than dequantization + hipBLAS)") \
    X(QWEN35MOE_MVSHARE,     "qwen35moe.mvshare",     true,  "",                    "GGML_Q35MOE_MVSHARE",          "one-token products that share one input") \
    X(QWEN35MOE_BA1,         "qwen35moe.ba1",         true,  "",                    "GGML_Q35MOE_BA1",              "one-token F32 beta/alpha products in one launch") \
    X(QWEN35MOE_PREP1,       "qwen35moe.prep1",       true,  "",                    "GGML_Q35MOE_PREP1",            "one-token conv, SiLU, norm, beta and gate in one op") \
    X(QWEN35MOE_GDN1,        "qwen35moe.gdn1",        true,  "",                    "GGML_Q35MOE_GDN1",             "one-token delta net in one op") \
    X(QWEN35MOE_POST1,       "qwen35moe.post1",       true,  "",                    "GGML_Q35MOE_POST1",            "one-token gated norm in one op") \
    X(QWEN35MOE_RS1,         "qwen35moe.rs1",         true,  "",                    "GGML_Q35MOE_RS1",              "the one-token ops also with rollback snapshots") \
    X(QWEN35MOE_PREPN,       "qwen35moe.prepn",       true,  "",                    "GGML_Q35MOE_PREPN",            "prep op for 2 to 5 tokens with snapshots") \
    X(QWEN35MOE_GDNN,        "qwen35moe.gdnn",        true,  "",                    "GGML_Q35MOE_GDNN",             "delta net op for 2 to 5 tokens with snapshots") \
    X(QWEN35MOE_POSTN,       "qwen35moe.postn",       true,  "",                    "GGML_Q35MOE_POSTN",            "gated norm op for 2 to 5 tokens") \
    X(QWEN35MOE_CONVPF,      "qwen35moe.convpf",      true,  "",                    "GGML_Q35MOE_CONVPF",           "prefill conv without the concat tensor") \
    X(QWEN35MOE_CONVPFRS,    "qwen35moe.convpfrs",    true,  "qwen35moe.convpf",    "GGML_Q35MOE_CONVPFRS",         "prefill conv also with rollback snapshots") \
    X(QWEN35MOE_GDNPF,       "qwen35moe.gdnpf",       true,  "",                    "GGML_Q35MOE_GDNPF",            "delta net prompt kernel") \
    X(QWEN35MOE_GDNPF_ORDER, "qwen35moe.gdnpf_order", true,  "qwen35moe.gdnpf",     "",                             "inside gdnpf: the blocks of 4 q/k heads and their 8 value heads run together, the q and k rows stay in L2") \
    X(QWEN35MOE_FFN1,        "qwen35moe.ffn1",        true,  "",                    "GGML_Q35MOE_FFN1",             "one-token MoE feed-forward in one op") \
    X(QWEN35MOE_FFNM,        "qwen35moe.ffnm",        true,  "",                    "GGML_Q35MOE_FFNM",             "MoE feed-forward of 2 to 4 tokens in one op") \
    X(QWEN35MOE_EXPONCE,     "qwen35moe.exponce",     true,  "qwen35moe.ffnm",      "GGML_Q35MOE_EXPONCE",          "inside ffnm: the rows of an expert are read once per batch") \
    X(QWEN35MOE_FFNM_TOPK,   "qwen35moe.ffnm_topk",   true,  "qwen35moe.ffnm",      "",                             "inside ffnm: the top-k block runs first with one warp per token, the shared expert gate/up in blocks of 4 warps") \
    X(QWEN35MOE_FFNM_LIST,   "qwen35moe.ffnm_list",   true,  "qwen35moe.ffnm",      "",                             "inside ffnm: the expert products take one warp per block for each expert of the batch from a list of the owners, 4 rows of down per warp") \
    X(QWEN35MOE_FAGQA,       "qwen35moe.fagqa",       true,  "",                    "GGML_Q35MOE_FAGQA",            "one-token attention, D256, 8 query heads per KV head") \
    X(QWEN35MOE_FAGQA4,      "qwen35moe.fagqa4",      true,  "",                    "GGML_Q35MOE_FAGQA4",           "attention of 2 to 4 tokens, D256, 8 query heads per KV head") \
    X(QWEN35MOE_FAGQA4_BURST, "qwen35moe.fagqa4_burst", true, "qwen35moe.fagqa4",   "",                             "inside fagqa4: 3 and 4 tokens with Q8_0 K/V and 16384 or more rows load 2 parts of the K row per burst, a K/V tile waits 2 times for K instead of 4") \
    X(QWEN35MOE_PFATTN,      "qwen35moe.pfattn",      true,  "",                    "GGML_Q35MOE_PFATTN",           "prefill attention, D256, 8 query heads per KV head") \
    X(QWEN35MOE_PFLOAD,      "qwen35moe.pfload",      true,  "qwen35moe.pfattn",    "",                             "inside pfattn: the row numbers of a warp in one 16-byte load, the second K half tile loaded before the QK products") \
    X(QWEN35MOE_PFQK,        "qwen35moe.pfqk",        true,  "qwen35moe.pfload",    "",                             "inside pfload: K in two stages of 128 elements, one Q fragment read for the QK products of both row slices of a warp") \
    X(QWEN35MOE_PFBAR,       "qwen35moe.pfbar",       true,  "qwen35moe.pfqk",      "",                             "inside pfqk: the barriers of the K/V loop only order shared memory, the global loads of the next half tile stay in flight") \
    X(QWEN35MOE_PFOFFSET,    "qwen35moe.pfoffset",    true,  "",                    "",                             "the F16 copies of quantized K/V for head size 256 with 8 query heads per KV head start 128 bytes past a 256-byte boundary (the K stage loads of pfqk span two memory channels)") \
    X(QWEN35MOE_EXPTAIL,     "qwen35moe.exptail",     true,  "",                    "GGML_Q35MOE_EXPTAIL",          "expert products on MMQ, 16 to 64 columns per tile") \
    X(QWEN35MOE_EXPTAIL_PIPE, "qwen35moe.exptail_pipe", true, "qwen35moe.exptail",  "",                             "inside exptail: 2 row tiles per block, the weights and activations of the next k step are loaded during the dot products of the current one, the barriers only wait for shared memory") \
    X(QWEN35MOE_QSHARE,      "qwen35moe.qshare",      true,  "",                    "GGML_Q35MOE_QSHARE",           "MMQ products of one input share one Q8_1 copy") \
    X(QWEN35MOE_PFMMN,       "qwen35moe.pfmmn",       true,  "",                    "GGML_Q35MOE_PFMMN",            "F32 router/beta/alpha products of 128 to 2048 tokens") \
    X(QWEN35MOE_MTPSTORE,    "qwen35moe.mtpstore",    true,  "",                    "GGML_Q35MOE_MTPSTORE",         "MTP graphs without outputs only store K and V") \
    X(HOST_LAZY_CLONE,       "host.lazy_clone",       true,  "",                    "",                             "server: a draft verification clones the sampler only when it can restore the checkpoint") \
    X(HOST_SEQRM_POS,        "host.seqrm_pos",        true,  "",                    "",                             "KV cache: seq_rm of one sequence finds its cells by position, not by a scan of all cells") \
    X(HOST_MASK_ROWS,        "host.mask_rows",        true,  "",                    "",                             "KV cache: K/V row lists write full words of 64 cells as runs; the causal KQ mask over the rows of one sequence per slice only compares positions")

enum ggml_kperf {
#define GGML_KPERF_ENUM(id, name, exact, needs, legacy, desc) GGML_KPERF_##id,
    GGML_KPERF_LIST(GGML_KPERF_ENUM)
#undef GGML_KPERF_ENUM
    GGML_KPERF_COUNT,
};

// true if the path of the entry may run; the environment is read at the first call
GGML_API bool ggml_kperf_on(enum ggml_kperf k);
// true if GGML_KPERF_TRACE names the entry
GGML_API bool ggml_kperf_trace(enum ggml_kperf k);
GGML_API const char * ggml_kperf_name(enum ggml_kperf k);

// the HIP backend calls this when it has a gfx1103 device with warp size 32 (for the log only)
GGML_API void ggml_kperf_set_device_ok(bool ok);

// one line for a model architecture: the entries of its family that are on, the ones turned off, and the lossy entries of any family that are on
// returns the length as snprintf does
GGML_API size_t ggml_kperf_describe(const char * arch, char * buf, size_t size);

#ifdef __cplusplus
}
#endif

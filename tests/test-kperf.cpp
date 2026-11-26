// Tests of the kperf table (ggml-kperf.h): defaults, GGML_KPERF_OFF, GGML_KPERF_TRACE, lossy entries, needs, unknown names.
// The table reads the environment once per process, so every case runs in a child process.
#include "ggml-kperf.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct test_case {
    const char *                                    name;
    std::vector<std::pair<const char *, const char *>> env;
    bool                                            aborts;
    // entries that must be on and off after the case
    std::vector<ggml_kperf>                         on;
    std::vector<ggml_kperf>                         off;
    std::vector<ggml_kperf>                         trace;
    const char *                                    describe_arch;
    const char *                                    describe_part; // must appear in the line of describe_arch
};

extern char ** environ;

// in the child: 0 if the table matches the case
static int run_case(const test_case & tc) {
    // no variable of the table or of the old switches from the caller
    std::vector<std::string> names;
    for (char ** e = environ; *e; ++e) {
        const std::string kv = *e;
        for (const char * prefix : { "GGML_KPERF_", "GGML_INKLING_", "GGML_QWEN35_", "GGML_Q35MOE_", "GGML_CUDA_MMQ_MOE_NCOLS" }) {
            if (kv.rfind(prefix, 0) == 0) {
                names.push_back(kv.substr(0, kv.find('=')));
            }
        }
    }
    for (const std::string & n : names) {
        unsetenv(n.c_str());
    }
    setenv("GGML_NO_BACKTRACE", "1", 1);
    for (const auto & e : tc.env) {
        setenv(e.first, e.second, 1);
    }
    int bad = 0;
    for (ggml_kperf k : tc.on) {
        if (!ggml_kperf_on(k)) {
            fprintf(stderr, "%s: %s should be on\n", tc.name, ggml_kperf_name(k));
            bad++;
        }
    }
    for (ggml_kperf k : tc.off) {
        if (ggml_kperf_on(k)) {
            fprintf(stderr, "%s: %s should be off\n", tc.name, ggml_kperf_name(k));
            bad++;
        }
    }
    for (int k = 0; k < GGML_KPERF_COUNT; ++k) {
        bool want = false;
        for (ggml_kperf t : tc.trace) {
            want |= t == k;
        }
        if (ggml_kperf_trace((ggml_kperf) k) != want) {
            fprintf(stderr, "%s: trace of %s should be %d\n", tc.name, ggml_kperf_name((ggml_kperf) k), int(want));
            bad++;
        }
    }
    if (tc.describe_arch) {
        char line[4096];
        ggml_kperf_describe(tc.describe_arch, line, sizeof(line));
        if (!strstr(line, tc.describe_part)) {
            fprintf(stderr, "%s: describe(%s) = '%s' has no '%s'\n", tc.name, tc.describe_arch, line, tc.describe_part);
            bad++;
        }
    }
    return bad == 0 ? 0 : 1;
}

int main() {
    const std::vector<test_case> cases = {
        { "defaults", {}, false,
            { GGML_KPERF_HIP_MOE_NCOLS32, GGML_KPERF_INKLING_FAGQA, GGML_KPERF_QWEN35_GDN1, GGML_KPERF_QWEN35MOE_MTPSTORE },
            { GGML_KPERF_QWEN35_Q6K_MMQ, GGML_KPERF_INKLING_BF16_F32ACC }, {},
            "qwen35", "on 9: hip.moe_ncols32 qwen35.mvshare" },
        { "defaults of qwen35moe", {}, false,
            { GGML_KPERF_QWEN35MOE_PFATTN, GGML_KPERF_QWEN35MOE_PFLOAD }, {}, {},
            "qwen35moe", "on 24: hip.moe_ncols32 qwen35moe.mvshare" },
        { "an entry without a legacy variable can be turned off", { { "GGML_KPERF_OFF", "qwen35moe.pfload" } }, false,
            { GGML_KPERF_QWEN35MOE_PFATTN }, { GGML_KPERF_QWEN35MOE_PFLOAD }, {},
            "qwen35moe", "on 23: hip.moe_ncols32 qwen35moe.mvshare" },
        { "lossy entries need their variable", { { "GGML_QWEN35_Q6K_MMQ", "1" }, { "GGML_INKLING_BF16_F32ACC", "2" } }, false,
            { GGML_KPERF_QWEN35_Q6K_MMQ }, { GGML_KPERF_INKLING_BF16_F32ACC }, {},
            "qwen35", "lossy on: qwen35.q6k_mmq" },
        { "lossy entry of inkling", { { "GGML_INKLING_BF16_F32ACC", "1" } }, false,
            { GGML_KPERF_INKLING_BF16_F32ACC }, { GGML_KPERF_QWEN35_Q6K_MMQ }, {},
            "inkling2_loop", "lossy on: inkling.bf16_f32acc" },
        { "lossy entry of another family", { { "GGML_QWEN35_Q6K_MMQ", "1" } }, false,
            { GGML_KPERF_QWEN35_Q6K_MMQ }, {}, {},
            "qwen35moe", "lossy on: qwen35.q6k_mmq;" },
        { "lossy entries of two families", { { "GGML_QWEN35_Q6K_MMQ", "1" }, { "GGML_INKLING_BF16_F32ACC", "1" } }, false,
            { GGML_KPERF_QWEN35_Q6K_MMQ, GGML_KPERF_INKLING_BF16_F32ACC }, {}, {},
            "qwen35moe", "; off: none; lossy on: inkling.bf16_f32acc qwen35.q6k_mmq; legacy variables ignored: 0" },
        { "long off list", { { "GGML_KPERF_OFF", "qwen35moe" }, { "GGML_QWEN35_Q6K_MMQ", "1" } }, false,
            { GGML_KPERF_QWEN35_Q6K_MMQ }, { GGML_KPERF_QWEN35MOE_MTPSTORE }, {},
            "qwen35moe", "qwen35moe.pfmmn qwen35moe.mtpstore; lossy on: qwen35.q6k_mmq; legacy variables ignored: 0" },
        { "old variables are ignored and counted", { { "GGML_Q35MOE_GDN1", "0" }, { "GGML_INKLING_MASKSKIP", "1" }, { "GGML_CUDA_MMQ_MOE_NCOLS", "16" },
            { "GGML_INKLING_PFZERO", "1" } }, false,
            { GGML_KPERF_QWEN35MOE_GDN1, GGML_KPERF_HIP_MOE_NCOLS32, GGML_KPERF_INKLING_PFATTN }, {}, {},
            "qwen35moe", "legacy variables ignored: 4" },
        { "all", { { "GGML_KPERF_OFF", "all" }, { "GGML_QWEN35_Q6K_MMQ", "1" }, { "GGML_INKLING_BF16_F32ACC", "1" } }, false,
            {}, { GGML_KPERF_HIP_MOE_NCOLS32, GGML_KPERF_INKLING_PFMM, GGML_KPERF_INKLING_BF16_F32ACC, GGML_KPERF_QWEN35_Q6K_MMQ, GGML_KPERF_QWEN35MOE_PFMMN }, {},
            "inkling2", "on 0:" },
        { "none and empty", { { "GGML_KPERF_OFF", " none, " } }, false,
            { GGML_KPERF_HIP_MOE_NCOLS32, GGML_KPERF_QWEN35MOE_FAGQA4 }, {}, {},
            "qwen35moe", "off: none" },
        { "family", { { "GGML_KPERF_OFF", "Qwen35" } }, false,
            { GGML_KPERF_QWEN35MOE_GDN1, GGML_KPERF_INKLING_MIX1 }, { GGML_KPERF_QWEN35_GDN1, GGML_KPERF_QWEN35_PFATTN, GGML_KPERF_QWEN35_PFLOAD }, {},
            "qwen35", "on 1: hip.moe_ncols32;" },
        { "pfload needs pfattn", { { "GGML_KPERF_OFF", "qwen35.pfattn" } }, false,
            { GGML_KPERF_QWEN35_GDN1, GGML_KPERF_QWEN35MOE_PFLOAD }, { GGML_KPERF_QWEN35_PFATTN, GGML_KPERF_QWEN35_PFLOAD }, {},
            "qwen35", "on 7: hip.moe_ncols32 qwen35.mvshare qwen35.prep1 qwen35.gdn1 qwen35.post1 qwen35.convpf qwen35.fagqa; off: qwen35.pfattn qwen35.pfload;" },
        { "entry and legacy name", { { "GGML_KPERF_OFF", "qwen35moe.fagqa4,GGML_Q35MOE_ffn1" } }, false,
            { GGML_KPERF_QWEN35MOE_FAGQA, GGML_KPERF_QWEN35MOE_FFNM }, { GGML_KPERF_QWEN35MOE_FAGQA4, GGML_KPERF_QWEN35MOE_FFN1 }, {},
            "qwen35moe", "off: qwen35moe.ffn1 qwen35moe.fagqa4" },
        { "needs", { { "GGML_KPERF_OFF", "hip.moe_ncols32 qwen35moe.ffnm inkling.rawmask qwen35moe.convpf qwen35moe.pfattn" } }, false,
            { GGML_KPERF_INKLING_MASKEPI },
            { GGML_KPERF_INKLING_GATE_TAIL16, GGML_KPERF_INKLING_DOWN_TAIL16, GGML_KPERF_INKLING_HOTPAIR_GATE, GGML_KPERF_INKLING_HOTPAIR_DOWN,
              GGML_KPERF_QWEN35MOE_EXPONCE, GGML_KPERF_INKLING_PFMASK, GGML_KPERF_QWEN35MOE_CONVPFRS, GGML_KPERF_QWEN35MOE_PFLOAD }, {},
            "qwen35moe", "off: hip.moe_ncols32 qwen35moe.convpf qwen35moe.convpfrs qwen35moe.ffnm qwen35moe.exponce qwen35moe.pfattn qwen35moe.pfload;" },
        { "trace", { { "GGML_KPERF_TRACE", "inkling.rowclip qwen35moe" } }, false,
            { GGML_KPERF_INKLING_ROWCLIP }, {},
            { GGML_KPERF_INKLING_ROWCLIP, GGML_KPERF_QWEN35MOE_MVSHARE, GGML_KPERF_QWEN35MOE_BA1, GGML_KPERF_QWEN35MOE_PREP1, GGML_KPERF_QWEN35MOE_GDN1,
              GGML_KPERF_QWEN35MOE_POST1, GGML_KPERF_QWEN35MOE_RS1, GGML_KPERF_QWEN35MOE_PREPN, GGML_KPERF_QWEN35MOE_GDNN, GGML_KPERF_QWEN35MOE_POSTN,
              GGML_KPERF_QWEN35MOE_CONVPF, GGML_KPERF_QWEN35MOE_CONVPFRS, GGML_KPERF_QWEN35MOE_GDNPF, GGML_KPERF_QWEN35MOE_FFN1, GGML_KPERF_QWEN35MOE_FFNM,
              GGML_KPERF_QWEN35MOE_EXPONCE, GGML_KPERF_QWEN35MOE_FAGQA, GGML_KPERF_QWEN35MOE_FAGQA4, GGML_KPERF_QWEN35MOE_PFATTN,
              GGML_KPERF_QWEN35MOE_PFLOAD, GGML_KPERF_QWEN35MOE_EXPTAIL, GGML_KPERF_QWEN35MOE_QSHARE, GGML_KPERF_QWEN35MOE_PFMMN,
              GGML_KPERF_QWEN35MOE_MTPSTORE },
            nullptr, nullptr },
        { "trace off entry", { { "GGML_KPERF_TRACE", "inkling.ppskip" }, { "GGML_KPERF_OFF", "inkling.ppskip" } }, false,
            {}, { GGML_KPERF_INKLING_PPSKIP }, { GGML_KPERF_INKLING_PPSKIP },
            nullptr, nullptr },
        { "unknown name in OFF aborts", { { "GGML_KPERF_OFF", "qwen35moe.fagqa,qwen35moe.fagqa5" } }, true, {}, {}, {}, nullptr, nullptr },
        { "removed switch aborts", { { "GGML_KPERF_OFF", "GGML_INKLING_MASKSKIP" } }, true, {}, {}, {}, nullptr, nullptr },
        { "removed lossy switch aborts", { { "GGML_KPERF_OFF", "inkling.pfzero" } }, true, {}, {}, {}, nullptr, nullptr },
        { "removed entry aborts", { { "GGML_KPERF_OFF", "qwen35moe.pfmm" } }, true, {}, {}, {}, nullptr, nullptr },
        { "unknown name in TRACE aborts", { { "GGML_KPERF_TRACE", "inkling.bogus" } }, true, {}, {}, {}, nullptr, nullptr },
    };

    int n_fail = 0;
    for (const test_case & tc : cases) {
        fflush(stdout);
        fflush(stderr);
        const pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            if (tc.aborts) {
                // the abort message goes to stderr, keep the test output short
                if (!freopen("/dev/null", "w", stderr)) {
                    _exit(3);
                }
            }
            _exit(run_case(tc));
        }
        int status = 0;
        waitpid(pid, &status, 0);
        const bool aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
        const bool passed  = tc.aborts ? aborted : WIFEXITED(status) && WEXITSTATUS(status) == 0;
        printf("%s: %s\n", passed ? "OK  " : "FAIL", tc.name);
        n_fail += !passed;
    }
    printf("%s\n", n_fail == 0 ? "all cases passed" : "some cases failed");
    return n_fail == 0 ? 0 : 1;
}

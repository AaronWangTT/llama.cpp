#include "ggml-kperf.h"
#include "ggml-impl.h"

#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

namespace {

struct entry {
    const char * name;
    bool         exact;
    const char * needs;
    const char * legacy;
    const char * desc;
};

const entry entries[GGML_KPERF_COUNT] = {
#define GGML_KPERF_ENTRY(id, name, exact, needs, legacy, desc) { name, exact, needs, legacy, desc },
    GGML_KPERF_LIST(GGML_KPERF_ENTRY)
#undef GGML_KPERF_ENTRY
};

const char * const families[] = { "hip", "inkling", "qwen35", "qwen35moe" };

// variables of the deleted switches and of the old trace lines, only counted for the log
const char * const removed[] = {
    "GGML_INKLING_MASKSKIP", "GGML_INKLING_HEADORDER", "GGML_INKLING_DEPTHMIX_CONT", "GGML_INKLING_VSKIP",
    "GGML_INKLING_FAGQA_Q8_HEADS", "GGML_INKLING_MASKSKIP_GEOMETRY", "GGML_QWEN35_GLUFUSE", "GGML_QWEN35_FAGQA_BURST",
    "GGML_QWEN35_FAGQA_CHECK", "GGML_Q35MOE_FAGQA_CHECK", "GGML_Q35MOE_FAGQA4_CHECK", "GGML_Q35MOE_FFNM5",
    "GGML_INKLING_GATHER_TRACE", "GGML_INKLING_MASKEPI_TRACE", "GGML_INKLING_MASKSKIP_TRACE", "GGML_INKLING_HEADORDER_TRACE",
    "GGML_INKLING_FAGQA_TRACE", "GGML_INKLING_PPSKIP_TRACE", "GGML_INKLING_PFATTN_TRACE", "GGML_INKLING_ROWCLIP_TRACE",
    "GGML_INKLING_SITU_TRACE", "GGML_INKLING_CTRL_TAIL_TRACE", "GGML_INKLING_RAWMASK_TRACE", "GGML_INKLING_GATE_TAIL16_TRACE",
    "GGML_INKLING_DOWN_TAIL16_TRACE", "GGML_INKLING_PFMM_TRACE", "GGML_Q35MOE_PFMMN_TRACE", "GGML_Q35MOE_QSHARE_LOG",
    "GGML_INKLING_PFZERO",
};

struct state {
    bool        on[GGML_KPERF_COUNT]    = {};
    bool        off[GGML_KPERF_COUNT]   = {}; // turned off by GGML_KPERF_OFF, also through the entry it needs
    bool        trace[GGML_KPERF_COUNT] = {};
    int         needs[GGML_KPERF_COUNT] = {};
    int         n_ignored = 0;
    std::string off_list; // the names in GGML_KPERF_OFF, "none" when it names none
};

state             g_state;
std::once_flag    g_once;
std::atomic<bool> g_device_ok { false };

std::string family_of(const char * name) {
    return std::string(name, std::strchr(name, '.') - name);
}

int find_entry(const char * name) {
    for (int k = 0; k < GGML_KPERF_COUNT; ++k) {
        if (std::strcmp(entries[k].name, name) == 0) {
            return k;
        }
    }
    return -1;
}

std::string lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

bool legacy_set_to_1(const char * var) {
    const char * p = std::getenv(var);
    return p && std::strcmp(p, "1") == 0;
}

// marks the entries that a list names; aborts at an unknown name. Returns the names of the list in lower case, separated by spaces
std::string parse_list(const char * var, const char * value, bool * mark) {
    std::string names_read;
    std::string list = value ? value : "";
    for (char & c : list) {
        if (c == ',') {
            c = ' ';
        }
    }
    std::istringstream in(list);
    std::string word;
    while (in >> word) {
        const std::string token = lower(word);
        if (token == "none") {
            continue;
        }
        names_read += (names_read.empty() ? "" : " ") + token;
        bool known = false;
        for (int k = 0; k < GGML_KPERF_COUNT; ++k) {
            const entry & e = entries[k];
            if (token == "all" || token == e.name || token == family_of(e.name) || token == lower(e.legacy)) {
                mark[k] = true;
                known   = true;
            }
        }
        if (!known) {
            std::string names;
            for (const char * f : families) {
                names += std::string(" ") + f;
            }
            for (const entry & e : entries) {
                names += std::string(" ") + e.name;
            }
            GGML_ABORT("%s: unknown name '%s'; known: all none%s (or the legacy variable of an entry)", var, token.c_str(), names.c_str());
        }
    }
    return names_read;
}

void init() {
    state & s = g_state;
    for (int k = 0; k < GGML_KPERF_COUNT; ++k) {
        s.needs[k] = entries[k].needs[0] ? find_entry(entries[k].needs) : -1;
        GGML_ASSERT(s.needs[k] < k && (entries[k].needs[0] == 0 || s.needs[k] >= 0));
    }

    const char * off   = std::getenv("GGML_KPERF_OFF");
    const char * trace = std::getenv("GGML_KPERF_TRACE");
    s.off_list = parse_list("GGML_KPERF_OFF", off, s.off);
    parse_list("GGML_KPERF_TRACE", trace, s.trace);
    if (s.off_list.empty()) {
        s.off_list = "none";
    }

    for (int k = 0; k < GGML_KPERF_COUNT; ++k) {
        const entry & e = entries[k];
        if (s.needs[k] >= 0 && s.off[s.needs[k]]) {
            s.off[k] = true;
        }
        const bool wanted = e.exact || legacy_set_to_1(e.legacy);
        s.on[k] = wanted && !s.off[k] && (s.needs[k] < 0 || s.on[s.needs[k]]);
        if (e.exact && std::getenv(e.legacy) != nullptr) {
            s.n_ignored++;
        }
    }
    for (const char * var : removed) {
        if (std::getenv(var) != nullptr) {
            s.n_ignored++;
        }
    }

    std::string lossy;
    int n_on = 0;
    for (int k = 0; k < GGML_KPERF_COUNT; ++k) {
        n_on += s.on[k];
        if (s.on[k] && !entries[k].exact) {
            lossy += std::string(lossy.empty() ? "" : " ") + entries[k].name;
        }
    }
    GGML_LOG_INFO("ggml_kperf: %d of %d entries on, off: %s, lossy on: %s, legacy variables ignored: %d\n",
        n_on, (int) GGML_KPERF_COUNT, s.off_list.c_str(), lossy.empty() ? "none" : lossy.c_str(), s.n_ignored);
}

const state & get() {
    std::call_once(g_once, init);
    return g_state;
}

void append(std::string & out, const char * fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    out += buf;
}

} // namespace

bool ggml_kperf_on(enum ggml_kperf k) {
    return get().on[k];
}

bool ggml_kperf_trace(enum ggml_kperf k) {
    return get().trace[k];
}

const char * ggml_kperf_name(enum ggml_kperf k) {
    return entries[k].name;
}

void ggml_kperf_set_device_ok(bool ok) {
    g_device_ok = ok;
}

size_t ggml_kperf_describe(const char * arch, char * buf, size_t size) {
    const state & s = get();

    std::string family;
    if (arch && (std::strcmp(arch, "inkling2") == 0 || std::strcmp(arch, "inkling2_loop") == 0)) {
        family = "inkling";
    } else if (arch && (std::strcmp(arch, "qwen35") == 0 || std::strcmp(arch, "qwen35moe") == 0)) {
        family = arch;
    }

    std::string on, off, lossy;
    int n_on = 0;
    for (int k = 0; k < GGML_KPERF_COUNT; ++k) {
        const char * name = entries[k].name;
        // a lossy entry acts on any architecture that reaches its path, so it is listed for every family
        if (s.on[k] && !entries[k].exact) {
            lossy += std::string(lossy.empty() ? "" : " ") + name;
        }
        const std::string f = family_of(name);
        if (f != "hip" && f != family) {
            continue;
        }
        if (s.on[k]) {
            n_on++;
            on += std::string(" ") + name;
        } else if (s.off[k]) {
            off += std::string(off.empty() ? "" : " ") + name;
        }
    }

    std::string line;
    append(line, "arch %s, gfx1103 %s, on %d:", arch && arch[0] ? arch : "?", g_device_ok ? "yes" : "no", n_on);
    line += on;
    append(line, "; off: %s; lossy on: %s; legacy variables ignored: %d",
        off.empty() ? "none" : off.c_str(), lossy.empty() ? "none" : lossy.c_str(), s.n_ignored);
    return (size_t) std::snprintf(buf, size, "%s", line.c_str());
}

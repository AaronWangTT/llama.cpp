// the row list of the draft-mtp head (common/speculative-shortlist.h), no model and no backend
// without arguments: synthetic cases. With arguments: token lists of real sessions, one begin() per prompt file
//   test-speculative-shortlist <n_vocab> <n_rows> <rank file> <marks file> <prompt file>...   (files: token ids, white space between them)

#include "speculative-shortlist.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            exit(1); \
        } \
    } while (0)

struct sl_checker {
    common_speculative_shortlist sl;
    std::vector<int32_t> marks;
    std::vector<int32_t> rows;
    bool was_cut = false;

    void init(int32_t n_rows, int32_t n_vocab, int32_t n_seq, const std::vector<int32_t> & file_ids, const std::vector<int32_t> & marks_in) {
        marks = marks_in;
        sl.init(n_rows, n_vocab, n_seq, file_ids, marks);
        check("init");
    }

    void begin(int32_t seq_id, const std::vector<int32_t> & prompt) {
        sl.begin(seq_id, prompt.data(), prompt.size());
        check("begin");
    }

    // the token must have a row right after the sequence has seen it
    void add(int32_t seq_id, int32_t id) {
        sl.add(seq_id, id);
        CHECK(sl.in[id] != common_speculative_shortlist::ROW_NONE, "token %d has no row after add()", id);
    }

    // what must hold after every operation
    void check(const char * what) {
        const int32_t n_vocab = (int32_t) sl.in.size();

        sl.rows(rows);
        CHECK((int32_t) rows.size() == sl.n, "%s: %zu rows, want %d", what, rows.size(), sl.n);

        for (const int32_t id : marks) {
            CHECK(sl.in[id] == common_speculative_shortlist::ROW_MARK, "%s: marker %d has no row", what, id);
        }
        for (int32_t r = 0; r < sl.n/4; ++r) {
            CHECK(sl.in[sl.rank[r]] != common_speculative_shortlist::ROW_NONE, "%s: file token of rank %d has no row", what, r);
        }

        // the seen tokens with a row of their own are the most recent ones
        int32_t  n_seen     = 0;
        int32_t  n_seen_all = 0;
        uint64_t oldest_in  = UINT64_MAX;
        uint64_t newest_out = 0;
        for (int32_t id = 0; id < n_vocab; ++id) {
            if (sl.mark[id] || !sl.seen_any(id)) {
                CHECK(sl.in[id] != common_speculative_shortlist::ROW_SEEN, "%s: token %d has a row of a seen token", what, id);
                continue;
            }
            n_seen_all++;
            if (sl.in[id] == common_speculative_shortlist::ROW_SEEN) {
                n_seen++;
                oldest_in = std::min(oldest_in, sl.stamp[id]);
            } else {
                newest_out = std::max(newest_out, sl.stamp[id]);
            }
        }
        CHECK(n_seen == sl.n_seen, "%s: %d rows of seen tokens, counter %d", what, n_seen, sl.n_seen);
        CHECK(n_seen + sl.n_mark <= sl.n_keep(), "%s: %d rows of seen tokens and markers, limit %d", what, n_seen + sl.n_mark, sl.n_keep());
        CHECK(n_seen == 0 || newest_out < oldest_in, "%s: a seen token without a row is newer than one with a row", what);
        if (n_seen < n_seen_all) {
            was_cut = true;
        }

        // not cut: the list is the markers, all seen tokens and the first other tokens of the file order
        if (n_seen == n_seen_all) {
            int32_t n_cur = sl.n_mark + n_seen;
            for (int32_t r = 0; r < n_vocab; ++r) {
                const int32_t id = sl.rank[r];
                if (sl.mark[id] || sl.seen_any(id)) {
                    continue;
                }
                const bool want = n_cur < sl.n;
                CHECK((sl.in[id] != common_speculative_shortlist::ROW_NONE) == want, "%s: file token of rank %d: row %d, want %d", what, r, (int) sl.in[id], (int) want);
                n_cur += want;
            }
        }
    }
};

static std::vector<int32_t> iota_ids(int32_t first, int32_t count) {
    std::vector<int32_t> ids(count);
    for (int32_t i = 0; i < count; ++i) {
        ids[i] = first + i;
    }
    return ids;
}

static void test_synthetic() {
    const int32_t n_vocab = 20000;
    const int32_t n_rows  = 1024;

    // the markers are the highest ids, as in the Qwen vocabulary, and the file does not have them
    const std::vector<int32_t> marks = iota_ids(n_vocab - 33, 33);
    std::vector<int32_t> file_ids = iota_ids(0, 6000);
    std::mt19937 rng(1234);
    std::shuffle(file_ids.begin(), file_ids.end(), rng);

    // fewer seen tokens than rows for them: nothing is dropped
    {
        sl_checker c;
        c.init(n_rows, n_vocab, 1, file_ids, marks);
        c.begin(0, iota_ids(7000, 500));
        CHECK(!c.sl.cut && !c.was_cut, "500 seen tokens were cut");
        for (int32_t i = 0; i < 200; ++i) {
            c.add(0, 9000 + i);
            c.check("add, not cut");
        }
        CHECK(!c.was_cut, "700 seen tokens were cut");
    }

    // more distinct seen tokens than rows: the newest ones and the markers stay
    {
        sl_checker c;
        c.init(n_rows, n_vocab, 1, file_ids, marks);
        c.begin(0, iota_ids(7000, 3000));
        CHECK(c.sl.cut, "3000 seen tokens were not cut");
        CHECK(c.sl.in[9999] == common_speculative_shortlist::ROW_SEEN, "the last prompt token has no row");
        CHECK(c.sl.in[7000] != common_speculative_shortlist::ROW_SEEN, "the first prompt token kept its row");

        // a cut followed by new tokens: each one gets a row, the list is rebuilt once per n/8 tokens at most
        const uint64_t n_rebuild0 = c.sl.n_rebuild;
        const int32_t  n_new      = 10*n_rows;
        for (int32_t i = 0; i < n_new; ++i) {
            c.add(0, (int32_t) (rng() % (n_vocab - 33)));
            if (i % 97 == 0) {
                c.check("add after a cut");
            }
        }
        c.check("after the new tokens");
        CHECK(c.sl.n_rebuild - n_rebuild0 <= (uint64_t) (n_new/(n_rows/8) + 1), "%d rebuilds for %d tokens", (int) (c.sl.n_rebuild - n_rebuild0), n_new);
    }

    // four sequences at once: together more seen tokens than rows
    {
        sl_checker c;
        c.init(n_rows, n_vocab, 4, file_ids, marks);
        for (int32_t s = 0; s < 4; ++s) {
            c.begin(s, iota_ids(6000 + 300*s, 400));
        }
        CHECK(c.was_cut, "four sequences were not cut");
        for (int32_t i = 0; i < 20000; ++i) {
            const int32_t s = (int32_t) (rng() % 4);
            if (rng() % 2000 == 0) {
                // a sequence starts over with a new prompt
                std::vector<int32_t> prompt(100 + rng() % 1500);
                for (auto & id : prompt) {
                    id = (int32_t) (rng() % n_vocab);
                }
                c.begin(s, prompt);
                continue;
            }
            c.add(s, (int32_t) (rng() % (rng() % 2 ? 3000 : n_vocab)));
            if (i % 61 == 0) {
                c.check("four sequences");
            }
        }
        c.check("four sequences, end");
    }

    // small lists and a list of the whole vocabulary
    for (const int32_t n : { 1, 2, 7, 64, n_vocab }) {
        sl_checker c;
        c.init(n, n_vocab, 2, file_ids, n >= 64 ? iota_ids(n_vocab - 8, 8) : std::vector<int32_t>());
        for (int32_t i = 0; i < 3000; ++i) {
            c.add((int32_t) (rng() % 2), (int32_t) (rng() % n_vocab));
            if (i % 13 == 0) {
                c.check("small list");
            }
        }
        c.begin(0, iota_ids(100, 5000));
        c.begin(1, {});
    }

    printf("synthetic cases: OK\n");
}

static std::vector<int32_t> read_ints(const char * path) {
    std::vector<int32_t> ids;
    std::ifstream f(path);
    CHECK(f.good(), "cannot read %s", path);
    std::string word;
    while (f >> word) {
        if (word[0] == '#') {
            std::getline(f, word);
            continue;
        }
        ids.push_back((int32_t) std::strtol(word.c_str(), nullptr, 10));
    }
    return ids;
}

static void test_files(int argc, char ** argv) {
    const int32_t n_vocab = atoi(argv[1]);
    const int32_t n_rows  = atoi(argv[2]);
    const std::vector<int32_t> file_ids = read_ints(argv[3]);
    const std::vector<int32_t> marks    = read_ints(argv[4]);
    const int32_t n_seq = argc - 5;

    sl_checker c;
    c.init(n_rows, n_vocab, n_seq, file_ids, marks);

    auto report = [&](const char * what) {
        int32_t n_marks_in = 0;
        for (const int32_t id : marks) {
            n_marks_in += c.sl.in[id] != common_speculative_shortlist::ROW_NONE;
        }
        int32_t n_seen_all = 0;
        for (int32_t id = 0; id < n_vocab; ++id) {
            n_seen_all += !c.sl.mark[id] && c.sl.seen_any(id);
        }
        printf("N %d %-28s rows %zu, cut %d, seen tokens %d (with a row of their own %d), markers in the list %d of %zu, rebuilds %d\n",
                c.sl.n, what, c.rows.size(), (int) c.sl.cut, n_seen_all, c.sl.n_seen, n_marks_in, marks.size(), (int) c.sl.n_rebuild);
    };
    report("after init");

    std::vector<std::vector<int32_t>> prompts;
    for (int32_t s = 0; s < n_seq; ++s) {
        prompts.push_back(read_ints(argv[5 + s]));
        c.begin(s, prompts.back());
        const char * name = strrchr(argv[5 + s], '/');
        report(name ? name + 1 : argv[5 + s]);
    }

    // then every sequence sees tokens it has not seen: the tokens of the next prompt, in turns
    size_t n_added = 0;
    for (size_t i = 0; i < 20000; ++i) {
        for (int32_t s = 0; s < n_seq; ++s) {
            const auto & other = prompts[(s + 1) % n_seq];
            if (i < other.size()) {
                c.add(s, other[other.size() - 1 - i]);
                n_added++;
            }
        }
        if (i % 499 == 0) {
            c.check("new tokens");
        }
    }
    c.check("after the new tokens");
    char what[64];
    snprintf(what, sizeof(what), "after %zu more tokens", n_added);
    report(what);
}

int main(int argc, char ** argv) {
    if (argc >= 6) {
        test_files(argc, argv);
    } else {
        test_synthetic();
    }
    return 0;
}

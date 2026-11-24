#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

// row list of the draft-mtp head (llama_set_logits_rows()): always n distinct token ids
// who gets a row, in this order: the marker tokens, the tokens the sequences have seen (most recent first), the file order
struct common_speculative_shortlist {
    enum : uint8_t { ROW_NONE = 0, ROW_FILE = 1, ROW_SEEN = 2, ROW_MARK = 3 };

    int32_t n = 0;                             // rows, 0 - unused
    std::vector<int32_t>  rank;                // all token ids, the ones of the file first
    std::vector<uint8_t>  in;                  // [n_vocab] ROW_NONE, or why the token has a row
    std::vector<uint8_t>  mark;                // [n_vocab] the token is a marker, it always has a row
    std::vector<std::vector<uint8_t>> seen;    // [n_seq][n_vocab] the sequence has seen the token
    std::vector<uint64_t> stamp;               // [n_vocab] clock of the last time a sequence saw the token
    uint64_t clock  = 0;
    int32_t  n_mark = 0;
    int32_t  n_seen = 0;                       // rows of seen tokens
    int32_t  n_seen_all = 0;                   // seen tokens at the last rebuild(), markers not counted
    int32_t  victim = -1;                      // rank index of the next file token to give up its row
    bool     dirty  = false;                   // the rows changed since the last rows()
    bool     cut    = false;                   // the last rebuild() had more seen tokens than rows for them
    uint64_t n_rebuild = 0;

    // the markers and the seen tokens take at most this many rows, the rest goes to the file order
    int32_t n_keep() const {
        return n - n/4;
    }

    bool seen_any(int32_t id) const {
        for (const auto & s : seen) {
            if (s[id]) {
                return true;
            }
        }
        return false;
    }

    // file_ids: distinct token ids, most frequent first; marks are used only if they take at most n_rows/8 rows
    void init(int32_t n_rows, int32_t n_vocab, int32_t n_seq, const std::vector<int32_t> & file_ids, const std::vector<int32_t> & marks) {
        n = std::min(n_rows, n_vocab);

        std::vector<uint8_t> ranked(n_vocab, 0);
        rank.clear();
        for (const int32_t id : file_ids) {
            if (id >= 0 && id < n_vocab && !ranked[id]) {
                ranked[id] = 1;
                rank.push_back(id);
            }
        }
        for (int32_t id = 0; id < n_vocab; ++id) {
            if (!ranked[id]) {
                rank.push_back(id);
            }
        }

        in.assign(n_vocab, ROW_NONE);
        mark.assign(n_vocab, 0);
        seen.assign(n_seq, std::vector<uint8_t>(n_vocab, 0));
        stamp.assign(n_vocab, 0);
        clock  = 0;
        n_mark = 0;

        if ((int32_t) marks.size() <= n/8) {
            for (const int32_t id : marks) {
                if (id >= 0 && id < n_vocab && !mark[id]) {
                    mark[id] = 1;
                    n_mark++;
                }
            }
        }

        rebuild();
    }

    // the list from scratch
    void rebuild() {
        const int32_t n_vocab = (int32_t) in.size();

        std::fill(in.begin(), in.end(), (uint8_t) ROW_NONE);

        std::vector<int32_t> ids;
        for (int32_t id = 0; id < n_vocab; ++id) {
            if (mark[id]) {
                in[id] = ROW_MARK;
            } else if (seen_any(id)) {
                ids.push_back(id);
            }
        }

        n_seen_all = (int32_t) ids.size();

        int32_t room = n_keep() - n_mark;
        cut = n_seen_all > room;
        if (cut) {
            // the most recent ones stay; n/8 rows stay free for new tokens, so the next cut is not one token away
            room -= n/8;
            std::nth_element(ids.begin(), ids.begin() + room, ids.end(), [&](int32_t a, int32_t b) { return stamp[a] > stamp[b]; });
            ids.resize(room);
        }
        for (const int32_t id : ids) {
            in[id] = ROW_SEEN;
        }
        n_seen = (int32_t) ids.size();

        int32_t n_cur = n_mark + n_seen;
        victim = -1;
        for (int32_t r = 0; r < n_vocab && n_cur < n; ++r) {
            if (in[rank[r]] == ROW_NONE) {
                in[rank[r]] = ROW_FILE;
                n_cur++;
            }
            victim = r;
        }

        dirty = true;
        n_rebuild++;
    }

    // the tokens of seq_id start over with the prompt
    void begin(int32_t seq_id, const int32_t * prompt, size_t n_prompt) {
        std::fill(seen[seq_id].begin(), seen[seq_id].end(), 0);
        for (size_t i = 0; i < n_prompt; ++i) {
            const int32_t id = prompt[i];
            if (id >= 0 && id < (int32_t) in.size()) {
                seen[seq_id][id] = 1;
                stamp[id] = ++clock;
            }
        }
        rebuild();
    }

    // seq_id has seen the token: it takes the row of the least frequent file token
    void add(int32_t seq_id, int32_t id) {
        if (n == 0 || id < 0 || id >= (int32_t) in.size()) {
            return;
        }
        seen[seq_id][id] = 1;
        stamp[id] = ++clock;
        if (in[id] >= ROW_SEEN) {
            return;
        }
        if (in[id] == ROW_NONE) {
            // the first n/4 tokens of the file keep their rows
            while (victim >= n/4 && in[rank[victim]] != ROW_FILE) {
                victim--;
            }
            if (victim < n/4) {
                // no file row left to give up, the oldest seen tokens go
                rebuild();
                return;
            }
            in[rank[victim--]] = ROW_NONE;
            dirty = true;
        }
        in[id] = ROW_SEEN;
        if (++n_seen > n_keep() - n_mark) {
            rebuild();
        }
    }

    // the token ids of the list in ascending order
    void rows(std::vector<int32_t> & out) {
        out.clear();
        for (int32_t id = 0; id < (int32_t) in.size(); ++id) {
            if (in[id] != ROW_NONE) {
                out.push_back(id);
            }
        }
        dirty = false;
    }
};

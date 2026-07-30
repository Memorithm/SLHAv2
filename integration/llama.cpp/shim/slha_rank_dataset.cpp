#include "slha_rank_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

namespace slha_rank_dataset {

namespace {

constexpr uint32_t kMagic = 0x534C4841;   // "SLHA"

struct LayerData {
    // one entry per sampled row
    std::vector<int32_t> head, gqa_group;
    std::vector<int64_t> token;
    std::vector<int32_t> n_visible;
    std::vector<int32_t> chunk;
    std::vector<float> q;        // rows * q_dim
    std::vector<float> b;        // concatenated, n_visible per row
    std::vector<float> s;        // concatenated, n_visible per row
    size_t q_dim = 0;

    // key matrix in KV position order, one entry per chunk
    std::vector<std::vector<float>> keys;      // keys[chunk]
    std::vector<uint64_t> key_rows;            // key_rows[chunk]
    size_t key_dim = 0;

    // diagnostics
    uint64_t rows = 0, active_keys = 0;
    uint64_t baseline_tie_pairs = 0, slha_tie_pairs = 0, undefined_rows = 0;
    uint64_t rejected_nonfinite = 0;
};

struct State {
    std::mutex mu;
    bool on = false;
    std::string dir;
    Sampling samp;
    int chunk = 0;
    std::map<int32_t, LayerData> layers;
};

State & st() {
    static State s;
    return s;
}

void put_u32(std::ofstream & o, uint32_t v) {
    o.write(reinterpret_cast<const char *>(&v), 4);
}

void put_u64(std::ofstream & o, uint64_t v) {
    o.write(reinterpret_cast<const char *>(&v), 8);
}

}  // namespace

bool enabled() {
    return st().on;
}

const Sampling & sampling() {
    return st().samp;
}

void enable(const char * dir) {
    State & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!dir || !*dir) {
        s.on = false;
        return;
    }
    s.dir = dir;
    s.on = true;
    if (const char * v = std::getenv("SLHA_RANK_TOKEN_STRIDE")) {
        const int n = std::atoi(v);
        if (n > 0) s.samp.token_stride = n;
    }
    if (const char * v = std::getenv("SLHA_RANK_MAX_HEADS")) {
        const int n = std::atoi(v);
        if (n > 0) s.samp.max_heads = n;
    }
}

void begin_chunk() {
    State & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.on) ++s.chunk;
}

int current_chunk() {
    return st().chunk;
}

bool wanted(int64_t t, int64_t h) {
    const State & s = st();
    if (!s.on) return false;
    return (t % s.samp.token_stride) == 0 && h < s.samp.max_heads;
}

void add_row(int32_t layer, int32_t head, int32_t gqa_group, int64_t t,
             const float * q_extended, size_t q_dim,
             const float * b, const float * s_scores, size_t n_visible) {
    if (!st().on || n_visible == 0 || q_dim == 0) return;

    // Reject the whole row if any value is non-finite. A ranking objective on a
    // partially non-finite row is meaningless, and silently dropping individual
    // keys would change the visible key set.
    for (size_t i = 0; i < q_dim; ++i) {
        if (!std::isfinite(q_extended[i])) {
            std::lock_guard<std::mutex> lock(st().mu);
            ++st().layers[layer].rejected_nonfinite;
            return;
        }
    }
    for (size_t i = 0; i < n_visible; ++i) {
        if (!std::isfinite(b[i]) || !std::isfinite(s_scores[i])) {
            std::lock_guard<std::mutex> lock(st().mu);
            ++st().layers[layer].rejected_nonfinite;
            return;
        }
    }

    // Exact-equality tie taxonomy over the visible prefix, adjacent pairs of the
    // sorted order. Counted here so the training set carries the same tie
    // accounting the oracle experiment used.
    uint64_t b_ties = 0, s_ties = 0;
    {
        std::vector<float> tmp(b, b + n_visible);
        std::sort(tmp.begin(), tmp.end());
        for (size_t i = 1; i < tmp.size(); ++i) if (tmp[i] == tmp[i - 1]) ++b_ties;
        tmp.assign(s_scores, s_scores + n_visible);
        std::sort(tmp.begin(), tmp.end());
        for (size_t i = 1; i < tmp.size(); ++i) if (tmp[i] == tmp[i - 1]) ++s_ties;
    }
    const bool undefined = (n_visible < 2) || (b_ties + 1 == n_visible);

    std::lock_guard<std::mutex> lock(st().mu);
    LayerData & L = st().layers[layer];
    if (L.q_dim == 0) L.q_dim = q_dim;
    if (L.q_dim != q_dim) return;          // dimension change: refuse silently-bad data
    L.head.push_back(head);
    L.gqa_group.push_back(gqa_group);
    L.token.push_back(t);
    L.n_visible.push_back(static_cast<int32_t>(n_visible));
    L.chunk.push_back(st().chunk);
    L.q.insert(L.q.end(), q_extended, q_extended + q_dim);
    L.b.insert(L.b.end(), b, b + n_visible);
    L.s.insert(L.s.end(), s_scores, s_scores + n_visible);
    ++L.rows;
    L.active_keys += n_visible;
    L.baseline_tie_pairs += b_ties;
    L.slha_tie_pairs += s_ties;
    if (undefined) ++L.undefined_rows;
}

void add_keys(int32_t layer, const float * rows, size_t n_rows, size_t dim) {
    if (!st().on || !rows || n_rows == 0 || dim == 0) return;
    std::lock_guard<std::mutex> lock(st().mu);
    State & s = st();
    LayerData & L = s.layers[layer];
    const size_t c = static_cast<size_t>(s.chunk);
    if (L.keys.size() <= c) {
        L.keys.resize(c + 1);
        L.key_rows.resize(c + 1, 0);
    }
    L.keys[c].assign(rows, rows + n_rows * dim);
    L.key_rows[c] = n_rows;
    L.key_dim = dim;
}

bool flush(std::string * error) {
    State & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.on) return true;

    const std::string cmd = "mkdir -p '" + s.dir + "'";
    if (std::system(cmd.c_str()) != 0) {
        if (error) *error = "cannot create " + s.dir;
        return false;
    }

    std::ostringstream man;
    man << "{\n  \"format_version\": 1,\n"
        << "  \"kind\": \"slha-rank-training-dataset\",\n"
        << "  \"active_domain\": \"written tile, current stream, causally visible to the query, "
           "finite, not padding, not masked\",\n"
        << "  \"baseline_reconstruction\": \"B_j = <q_extended, k_j>; the extended query zero-pads "
           "every GQA slot except the head's own\",\n"
        << "  \"sampling\": {\"token_stride\": " << s.samp.token_stride
        << ", \"max_heads\": " << s.samp.max_heads
        << ", \"predicate\": \"(t % token_stride) == 0 && head < max_heads\"},\n"
        << "  \"layers\": [";

    bool first = true;
    uint64_t tot_rows = 0, tot_keys = 0, tot_bt = 0, tot_st = 0, tot_undef = 0, tot_rej = 0;
    bool ok = true;

    for (auto & kv : s.layers) {
        const int32_t layer = kv.first;
        LayerData & L = kv.second;
        if (L.rows == 0) continue;

        char name[64];
        std::snprintf(name, sizeof(name), "rank-layer-%03d.bin", layer);
        const std::string path = s.dir + "/" + name;
        std::ofstream o(path, std::ios::binary);
        if (!o) {
            if (error) *error = "cannot write " + path;
            ok = false;
            break;
        }
        // header: magic, version, layer, q_dim, rows, n_chunks, key_dim
        const uint64_t n_chunks = L.keys.size();
        put_u32(o, kMagic);
        put_u32(o, 2);
        put_u32(o, static_cast<uint32_t>(layer));
        put_u32(o, static_cast<uint32_t>(L.q_dim));
        put_u64(o, L.rows);
        put_u64(o, n_chunks);
        put_u32(o, static_cast<uint32_t>(L.key_dim));
        put_u32(o, 0);                                  // reserved, keeps 8-byte alignment
        // key rows per chunk, then per-row metadata
        o.write(reinterpret_cast<const char *>(L.key_rows.data()),
                static_cast<std::streamsize>(L.key_rows.size() * 8));
        o.write(reinterpret_cast<const char *>(L.head.data()),
                static_cast<std::streamsize>(L.head.size() * 4));
        o.write(reinterpret_cast<const char *>(L.gqa_group.data()),
                static_cast<std::streamsize>(L.gqa_group.size() * 4));
        o.write(reinterpret_cast<const char *>(L.token.data()),
                static_cast<std::streamsize>(L.token.size() * 8));
        o.write(reinterpret_cast<const char *>(L.n_visible.data()),
                static_cast<std::streamsize>(L.n_visible.size() * 4));
        o.write(reinterpret_cast<const char *>(L.chunk.data()),
                static_cast<std::streamsize>(L.chunk.size() * 4));
        // payloads
        o.write(reinterpret_cast<const char *>(L.q.data()),
                static_cast<std::streamsize>(L.q.size() * 4));
        o.write(reinterpret_cast<const char *>(L.b.data()),
                static_cast<std::streamsize>(L.b.size() * 4));
        o.write(reinterpret_cast<const char *>(L.s.data()),
                static_cast<std::streamsize>(L.s.size() * 4));
        for (const auto & ck : L.keys) {
            o.write(reinterpret_cast<const char *>(ck.data()),
                    static_cast<std::streamsize>(ck.size() * 4));
        }
        o.flush();
        if (!o) {
            if (error) *error = "short write on " + path;
            ok = false;
            break;
        }
        o.close();

        if (!first) man << ",";
        first = false;
        man << "\n    {\"layer\": " << layer
            << ", \"file\": \"" << name << "\""
            << ", \"rows\": " << L.rows
            << ", \"q_dim\": " << L.q_dim
            << ", \"n_chunks\": " << L.keys.size()
            << ", \"key_rows_per_chunk\": [";
        for (size_t i = 0; i < L.key_rows.size(); ++i) {
            man << (i ? "," : "") << L.key_rows[i];
        }
        man << "]"
            << ", \"key_dim\": " << L.key_dim
            << ", \"active_keys\": " << L.active_keys
            << ", \"baseline_tie_pairs\": " << L.baseline_tie_pairs
            << ", \"slha_tie_pairs\": " << L.slha_tie_pairs
            << ", \"undefined_rows\": " << L.undefined_rows
            << ", \"rejected_nonfinite_rows\": " << L.rejected_nonfinite << "}";
        tot_rows += L.rows;
        tot_keys += L.active_keys;
        tot_bt += L.baseline_tie_pairs;
        tot_st += L.slha_tie_pairs;
        tot_undef += L.undefined_rows;
        tot_rej += L.rejected_nonfinite;
    }

    man << "\n  ],\n"
        << "  \"num_layers\": " << s.layers.size() << ",\n"
        << "  \"total_rows\": " << tot_rows << ",\n"
        << "  \"total_active_keys\": " << tot_keys << ",\n"
        << "  \"total_baseline_tie_pairs\": " << tot_bt << ",\n"
        << "  \"total_slha_tie_pairs\": " << tot_st << ",\n"
        << "  \"total_undefined_rows\": " << tot_undef << ",\n"
        << "  \"total_rejected_nonfinite_rows\": " << tot_rej << ",\n"
        << "  \"valid\": " << ((ok && tot_rows > 0) ? "true" : "false") << "\n}\n";

    const std::string mpath = s.dir + "/rank_dataset_manifest.json";
    std::ofstream mo(mpath);
    if (!mo) {
        if (error) *error = "cannot write " + mpath;
        return false;
    }
    mo << man.str();
    mo.flush();
    if (!mo) {
        if (error) *error = "short write on " + mpath;
        return false;
    }
    return ok;
}

}  // namespace slha_rank_dataset

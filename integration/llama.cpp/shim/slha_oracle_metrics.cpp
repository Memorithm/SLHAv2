// Implementation of the SLHA active-key ranking and tie statistics.
// See slha_oracle_metrics.hpp.
#include "slha_oracle_metrics.hpp"
#include "slha_score_oracle.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <sstream>
#include <vector>

namespace slha_oracle_metrics {

const int kTopK[kNumTopK] = {1, 2, 4, 8, 16};

void LayerMetrics::merge(const LayerMetrics & o) {
    rows += o.rows;
    active_keys += o.active_keys;
    for (int i = 0; i < kNumTopK; ++i) {
        topk_overlap[i] += o.topk_overlap[i];
        topk_rows[i] += o.topk_rows[i];
        boundary_ties[i] += o.boundary_ties[i];
    }
    sum_spearman += o.sum_spearman;   n_spearman += o.n_spearman;
    sum_kendall_b += o.sum_kendall_b; n_kendall += o.n_kendall;
    undefined_rows += o.undefined_rows;
    b_tie_pairs += o.b_tie_pairs;     s_tie_pairs += o.s_tie_pairs;
    b_rows_with_tie += o.b_rows_with_tie; s_rows_with_tie += o.s_rows_with_tie;
    b_max_block = std::max(b_max_block, o.b_max_block);
    s_max_block = std::max(s_max_block, o.s_max_block);
    for (int i = 0; i < kBlockBins; ++i) {
        b_block_hist[i] += o.b_block_hist[i];
        s_block_hist[i] += o.s_block_hist[i];
    }
    for (int i = 0; i < kMaxHeads; ++i) {
        head_rows[i] += o.head_rows[i];
        head_b_ties[i] += o.head_b_ties[i];
    }
}

namespace {
std::mutex g_mutex;
std::atomic<bool> g_enabled{false};
LayerMetrics g_layers[kMaxLayers];
bool g_seen[kMaxLayers] = {false};

// Fractional ranks (1-based, average over tied groups) for a Spearman that is
// well defined in the presence of ties.
void fractional_ranks(const float * v, const std::vector<int32_t> & perm,
                      size_t n, std::vector<double> & rank) {
    rank.assign(n, 0.0);
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j + 1 < n && v[perm[j + 1]] == v[perm[i]]) {
            ++j;
        }
        const double avg = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
        for (size_t r = i; r <= j; ++r) {
            rank[perm[r]] = avg;
        }
        i = j + 1;
    }
}

// Tied-block sizes along a sorted order; returns adjacent-equal pair count.
uint64_t block_stats(const float * v, const std::vector<int32_t> & perm, size_t n,
                     uint64_t * hist, uint64_t * max_block) {
    uint64_t pairs = 0;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j + 1 < n && v[perm[j + 1]] == v[perm[i]]) {
            ++j;
        }
        const uint64_t sz = static_cast<uint64_t>(j - i + 1);
        pairs += sz - 1;
        if (sz > 1) {
            int bin = static_cast<int>(sz);
            if (bin >= kBlockBins) bin = kBlockBins - 1;
            hist[bin] += 1;
            if (sz > *max_block) *max_block = sz;
        }
        i = j + 1;
    }
    return pairs;
}
}  // namespace

void add_row(int32_t layer_id, int head, const float * b, const float * s, size_t n) {
    if (!g_enabled.load(std::memory_order_acquire)) return;
    if (layer_id < 0 || layer_id >= kMaxLayers || n == 0) return;

    thread_local std::vector<int32_t> pb, ps;
    thread_local std::vector<double> rb, rs;
    slha_oracle::rank_permutation(b, n, pb, nullptr);
    slha_oracle::rank_permutation(s, n, ps, nullptr);

    LayerMetrics m;
    m.rows = 1;
    m.active_keys = n;
    if (head >= 0 && head < kMaxHeads) {
        m.head_rows[head] = 1;
    }

    // --- top-k set overlap over active keys ---
    for (int t = 0; t < kNumTopK; ++t) {
        const size_t k = static_cast<size_t>(kTopK[t]);
        if (k > n) continue;
        m.topk_rows[t] = 1;
        size_t hits = 0;
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                if (pb[i] == ps[j]) { ++hits; break; }
            }
        }
        m.topk_overlap[t] = static_cast<double>(hits);
        // boundary tie: the k-th and (k+1)-th baseline scores are equal, so
        // membership of the top-k is decided by the index tiebreak alone
        if (k < n && b[pb[k - 1]] == b[pb[k]]) {
            m.boundary_ties[t] = 1;
        }
    }

    // --- tie taxonomy (exact equality) ---
    m.b_tie_pairs = block_stats(b, pb, n, m.b_block_hist, &m.b_max_block);
    m.s_tie_pairs = block_stats(s, ps, n, m.s_block_hist, &m.s_max_block);
    if (m.b_tie_pairs > 0) {
        m.b_rows_with_tie = 1;
        if (head >= 0 && head < kMaxHeads) m.head_b_ties[head] = m.b_tie_pairs;
    }
    if (m.s_tie_pairs > 0) m.s_rows_with_tie = 1;

    // --- Spearman and Kendall tau-b, both tie-aware ---
    if (n < 2) {
        m.undefined_rows = 1;
    } else {
        fractional_ranks(b, pb, n, rb);
        fractional_ranks(s, ps, n, rs);
        double mb = 0, ms = 0;
        for (size_t i = 0; i < n; ++i) { mb += rb[i]; ms += rs[i]; }
        mb /= static_cast<double>(n); ms /= static_cast<double>(n);
        double num = 0, db = 0, ds = 0;
        for (size_t i = 0; i < n; ++i) {
            const double x = rb[i] - mb, y = rs[i] - ms;
            num += x * y; db += x * x; ds += y * y;
        }
        if (db > 0.0 && ds > 0.0) {
            m.sum_spearman = num / std::sqrt(db * ds);
            m.n_spearman = 1;
        } else {
            m.undefined_rows = 1;   // one side entirely tied: correlation undefined
        }
        // Kendall tau-b: (C - D) / sqrt((C+D+Tb)(C+D+Ts))
        // O(n^2); the caller samples rows, so this stays bounded.
        double C = 0, D = 0, Tb = 0, Ts = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double bi = b[i], bj = b[j], si = s[i], sj = s[j];
                const bool bt = (bi == bj), st = (si == sj);
                if (bt && st) continue;          // tied in both: counted in neither
                if (bt) { Tb += 1; continue; }
                if (st) { Ts += 1; continue; }
                const double prod = (bi - bj) * (si - sj);
                if (prod > 0) C += 1; else if (prod < 0) D += 1;
            }
        }
        const double denom = std::sqrt((C + D + Tb) * (C + D + Ts));
        if (denom > 0.0) {
            m.sum_kendall_b = (C - D) / denom;
            m.n_kendall = 1;
        }
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    g_layers[layer_id].merge(m);
    g_seen[layer_id] = true;
}

void reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (int i = 0; i < kMaxLayers; ++i) { g_layers[i] = LayerMetrics(); g_seen[i] = false; }
}
void enable(bool on) { g_enabled.store(on, std::memory_order_release); }
bool enabled() { return g_enabled.load(std::memory_order_acquire); }

static double pctile_from_hist(const uint64_t * hist, double q) {
    uint64_t tot = 0;
    for (int i = 0; i < kBlockBins; ++i) tot += hist[i];
    if (tot == 0) return 0.0;
    const uint64_t target = static_cast<uint64_t>(q * static_cast<double>(tot));
    uint64_t cum = 0;
    for (int i = 0; i < kBlockBins; ++i) {
        cum += hist[i];
        if (cum > target) return static_cast<double>(i);
    }
    return static_cast<double>(kBlockBins - 1);
}

static double mean_block(const uint64_t * hist) {
    uint64_t tot = 0; double sum = 0;
    for (int i = 0; i < kBlockBins; ++i) { tot += hist[i]; sum += static_cast<double>(i) * hist[i]; }
    return tot ? sum / static_cast<double>(tot) : 0.0;
}

std::string dump_json() {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::ostringstream os;
    os.precision(9);
    os << "{\n  \"schema\": \"slha_oracle_active_key_metrics_v1\",\n";
    os << "  \"note\": \"all statistics restricted to ACTIVE keys: written tile, "
          "causally visible, finite, single stream. Padded and masked positions are "
          "excluded before any statistic is formed.\",\n";
    os << "  \"tie_definition\": \"exact floating-point equality\",\n";
    os << "  \"topk\": [1,2,4,8,16],\n";
    os << "  \"layers\": {\n";
    bool first = true;
    for (int i = 0; i < kMaxLayers; ++i) {
        if (!g_seen[i] || g_layers[i].rows == 0) continue;
        const LayerMetrics & m = g_layers[i];
        if (!first) os << ",\n";
        first = false;
        os << "    \"" << i << "\": {\"rows\": " << m.rows
           << ", \"active_keys\": " << m.active_keys;
        os << ", \"mean_active_len\": "
           << (m.rows ? static_cast<double>(m.active_keys) / static_cast<double>(m.rows) : 0.0);
        os << ", \"topk_agreement\": [";
        for (int t = 0; t < kNumTopK; ++t) {
            if (t) os << ",";
            const double denom = static_cast<double>(m.topk_rows[t]) * kTopK[t];
            os << (denom > 0 ? m.topk_overlap[t] / denom : 0.0);
        }
        os << "]";
        os << ", \"topk_rows\": [";
        for (int t = 0; t < kNumTopK; ++t) { if (t) os << ","; os << m.topk_rows[t]; }
        os << "]";
        os << ", \"boundary_ties\": [";
        for (int t = 0; t < kNumTopK; ++t) { if (t) os << ","; os << m.boundary_ties[t]; }
        os << "]";
        os << ", \"boundary_tie_fraction\": [";
        for (int t = 0; t < kNumTopK; ++t) {
            if (t) os << ",";
            os << (m.topk_rows[t] ? static_cast<double>(m.boundary_ties[t]) /
                                    static_cast<double>(m.topk_rows[t]) : 0.0);
        }
        os << "]";
        os << ", \"spearman\": " << (m.n_spearman ? m.sum_spearman / static_cast<double>(m.n_spearman) : 0.0);
        os << ", \"spearman_rows\": " << m.n_spearman;
        os << ", \"kendall_tau_b\": " << (m.n_kendall ? m.sum_kendall_b / static_cast<double>(m.n_kendall) : 0.0);
        os << ", \"kendall_rows\": " << m.n_kendall;
        os << ", \"undefined_rows\": " << m.undefined_rows;
        os << ", \"baseline_tie_pairs\": " << m.b_tie_pairs;
        os << ", \"slha_tie_pairs\": " << m.s_tie_pairs;
        os << ", \"baseline_rows_with_tie\": " << m.b_rows_with_tie;
        os << ", \"slha_rows_with_tie\": " << m.s_rows_with_tie;
        os << ", \"baseline_max_block\": " << m.b_max_block;
        os << ", \"slha_max_block\": " << m.s_max_block;
        os << ", \"baseline_mean_block\": " << mean_block(m.b_block_hist);
        os << ", \"slha_mean_block\": " << mean_block(m.s_block_hist);
        os << ", \"baseline_block_p50\": " << pctile_from_hist(m.b_block_hist, 0.50);
        os << ", \"baseline_block_p90\": " << pctile_from_hist(m.b_block_hist, 0.90);
        os << ", \"baseline_block_p99\": " << pctile_from_hist(m.b_block_hist, 0.99);
        os << ", \"slha_block_p50\": " << pctile_from_hist(m.s_block_hist, 0.50);
        os << ", \"slha_block_p90\": " << pctile_from_hist(m.s_block_hist, 0.90);
        os << ", \"slha_block_p99\": " << pctile_from_hist(m.s_block_hist, 0.99);
        os << ", \"head_rows\": [";
        for (int hh = 0; hh < kMaxHeads; ++hh) {
            if (m.head_rows[hh] == 0) continue;
            os << m.head_rows[hh] << ",";
        }
        os << "0]";
        os << ", \"head_baseline_ties\": [";
        for (int hh = 0; hh < kMaxHeads; ++hh) {
            if (m.head_rows[hh] == 0) continue;
            os << m.head_b_ties[hh] << ",";
        }
        os << "0]";
        os << "}";
    }
    os << "\n  }\n}\n";
    return os.str();
}

}  // namespace slha_oracle_metrics

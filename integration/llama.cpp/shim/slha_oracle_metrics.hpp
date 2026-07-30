// SLHA active-key ranking and tie statistics (diagnostic).
//
// Every metric here is computed over the ACTIVE key set only: positions that
// carry a written tile, are causally visible to the query, are finite, and
// belong to the single active stream. Padded and masked positions are excluded
// before any statistic is formed. This matters because the top-1/top-5 figures
// quoted in PR #60 were computed over the full padded row and are therefore not
// attention-relevant rank agreements; nothing here reuses them.
//
// Statistics recorded per layer (and per head where noted):
//   * top-k agreement between the baseline and SLHA rankings, k in {1,2,4,8,16}
//   * Spearman rank correlation and Kendall tau-b (tie-aware)
//   * exact-tie taxonomy: adjacent equal pairs, rows containing a tie, tied
//     block sizes (max / mean / p50 / p90 / p99 via a bounded histogram)
//   * ties crossing each top-k boundary, which is what makes a top-k oracle
//     depend on the deterministic index tiebreak
//
// Exact floating-point equality defines a tie. No near-tie threshold is mixed
// into these counts.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace slha_oracle_metrics {

constexpr int kMaxLayers = 128;
constexpr int kMaxHeads = 32;
constexpr int kBlockBins = 64;      // tied-block size histogram, last bin saturates
constexpr int kNumTopK = 5;         // k = 1, 2, 4, 8, 16
extern const int kTopK[kNumTopK];

struct LayerMetrics {
    uint64_t rows = 0;
    uint64_t active_keys = 0;

    // top-k set overlap, summed over rows; divide by rows*k for the mean fraction
    double topk_overlap[kNumTopK] = {0};
    uint64_t topk_rows[kNumTopK] = {0};   // rows where k <= active length

    double sum_spearman = 0;
    uint64_t n_spearman = 0;              // rows where the statistic is defined
    double sum_kendall_b = 0;
    uint64_t n_kendall = 0;
    uint64_t undefined_rows = 0;          // one active key, or all values tied

    // exact-tie taxonomy, tracked separately for the two source rows
    uint64_t b_tie_pairs = 0, s_tie_pairs = 0;
    uint64_t b_rows_with_tie = 0, s_rows_with_tie = 0;
    uint64_t b_max_block = 0, s_max_block = 0;
    uint64_t b_block_hist[kBlockBins] = {0};
    uint64_t s_block_hist[kBlockBins] = {0};

    // ties straddling a top-k boundary: the k-th and (k+1)-th baseline scores
    // are equal, so which key is "in" the top k is decided by the index rule
    uint64_t boundary_ties[kNumTopK] = {0};

    uint64_t head_rows[kMaxHeads] = {0};
    uint64_t head_b_ties[kMaxHeads] = {0};

    // physical-to-active reconciliation, accumulated per sampled row:
    //   physical == included + padding + causally_masked + inactive_stream + nonfinite
    uint64_t acct_rows = 0;
    uint64_t acct_physical = 0;
    uint64_t acct_included = 0;
    uint64_t acct_padding = 0;
    uint64_t acct_masked = 0;
    uint64_t acct_inactive_stream = 0;
    uint64_t acct_nonfinite = 0;
    uint64_t acct_failures = 0;      // rows where the identity did not hold

    void merge(const LayerMetrics & o);
};

// Accumulate one active row pair. `b` and `s` must be the active prefix only.
void add_row(int32_t layer_id, int head, const float * b, const float * s, size_t n);

// Record the position accounting for one sampled row. The categories must be
// disjoint and sum exactly to `physical`; a row that fails the identity is
// counted in acct_failures and invalidates the run's metrics.
void add_accounting(int32_t layer_id, uint64_t physical, uint64_t included,
                    uint64_t padding, uint64_t masked, uint64_t inactive_stream,
                    uint64_t nonfinite);

void reset();
void enable(bool on);
bool enabled();
std::string dump_json();

}  // namespace slha_oracle_metrics

// SLHA per-layer score-scale fitting accumulator (shadow-mode diagnostic).
//
// Streams the sufficient statistics needed to fit a per-layer multiplicative
// scale a_layer such that baseline_score ~= a_layer * slha_score over active,
// causally-unmasked positions:
//   * OLS scale through the origin:  a = sum(b*s) / sum(s*s)
//   * robust scale: median of the per-scalar magnitude ratios |b|/|s|,
//     estimated from a bounded log10-ratio histogram.
//   * variance-matching scale: a = std(b) / std(s).
// No score vectors are retained; only running sums and a fixed histogram per
// layer. All reported error metrics are computed EXACTLY from the sufficient
// statistics (no per-sample data is needed):
//   sum (b - a s)^2 = sum_b2 - 2 a sum_bs + a^2 sum_s2.
// Fixed-size, bounds-checked arrays (no per-head/per-bucket structures) to keep
// it crash-safe. Active only when SLHA_SCALE_FIT_JSON is set.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace slha_scale_fit {

constexpr int kMaxLayers = 128;
constexpr int kHistBins = 600;      // log10 ratio in [-3, 3], 0.01 per bin
constexpr double kHistLo = -3.0, kHistHi = 3.0;

struct LayerAcc {
    double n = 0;
    double sum_bs = 0;   // sum(b*s)
    double sum_s2 = 0;   // sum(s*s)
    double sum_b2 = 0;   // sum(b*b)
    double sum_b = 0;
    double sum_s = 0;
    double sum_abs_err = 0;      // sum |b - s|      (before scaling)
    double sum_sq_err = 0;       // sum (b - s)^2    (before scaling)
    uint64_t hist[kHistBins] = {0};   // log10(|b|/|s|) histogram for robust median
    uint64_t hist_n = 0;

    // Accumulate one (baseline, slha) score pair. Non-finite pairs are skipped.
    void add(double b, double s);
    void merge(const LayerAcc & o);

    // --- scale estimators ---
    double ols_scale() const;         // sum_bs / sum_s2  (least squares, through origin)
    double robust_scale() const;      // 10^median(log10|b/s|)
    double variance_scale() const;    // std(b) / std(s)

    // --- fit quality (all exact from sufficient statistics) ---
    double mean_b() const;
    double mean_s() const;
    double std_b() const;
    double std_s() const;
    double pearson() const;           // corr(b, s): magnitude-invariant alignment
    double var_ratio() const;         // var(s) / var(b)
    double mae_before() const;        // mean |b - s|
    double rms_before() const;        // sqrt(mean (b - s)^2)
    double rms_after(double a) const; // sqrt(mean (b - a s)^2)
    double rel_l2_before() const;     // ||b - s|| / ||b||
    double rel_l2_after(double a) const; // ||b - a s|| / ||b||
};

// Process-wide registry (one accumulator per layer id). Thread-safe merge.
void reset();
void enable(bool on);
bool enabled();
// Merge a thread-local partial for one layer into the registry.
void merge_layer(int32_t layer_id, const LayerAcc & partial);
// Serialize all non-empty layers to a pretty-printed JSON string.
std::string dump_json();

}  // namespace slha_scale_fit

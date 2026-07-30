// SLHA per-layer score-scale fitting accumulator (shadow-mode diagnostic).
//
// Streams the sufficient statistics needed to fit a per-layer multiplicative
// scale a_layer such that baseline_score ~= a_layer * slha_score over active,
// causally-unmasked positions:
//   * OLS scale through the origin:  a = sum(b*s) / sum(s*s)
//   * robust scale: median of the per-scalar magnitude ratios |b|/|s|,
//     estimated from a bounded log10-ratio histogram. Pairs whose magnitudes
//     fall at or below kRobustEps are EXCLUDED (unstable ratio) and counted.
//   * variance-matching scale: a = std(b) / std(s).
//
// It also records the diagnostics needed to show the fit is not dominated by a
// degenerate subpopulation:
//   * per-head sufficient statistics (bounded, kMaxHeads)  -> per-head OLS scale
//   * query-position bucket statistics (bounded, kPosBuckets)
//   * a log10|s| magnitude histogram carrying BOTH the sample count and the
//     sum of s^2 per decade — since OLS weights by s^2, this shows directly
//     how much of the estimator's denominator near-zero scores contribute.
//   * counts of non-finite and near-zero-denominator exclusions.
//
// All error metrics are computed EXACTLY from the sufficient statistics:
//   sum (b - a s)^2 = sum_b2 - 2 a sum_bs + a^2 sum_s2.
// No score vectors are retained. Fixed-size, bounds-checked arrays only (no
// dynamic per-head/per-bucket structures) to keep it crash-safe. Active only
// when SLHA_SCALE_FIT_JSON is set.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace slha_scale_fit {

constexpr int kMaxLayers  = 128;
constexpr int kHistBins   = 600;    // log10 ratio in [-3, 3], 0.01 per bin
constexpr double kHistLo  = -3.0, kHistHi = 3.0;
constexpr int kMaxHeads   = 32;     // bound; Qwen2.5-1.5B has 12 query heads
constexpr int kPosBuckets = 8;      // query-position buckets within a chunk
constexpr int kMagBins    = 100;    // log10|s| in [-5, 5], 0.1 per bin
constexpr double kMagLo   = -5.0, kMagHi = 5.0;

// Magnitude at or below which a ratio |b|/|s| is treated as numerically
// unstable and excluded from the ROBUST estimator only (OLS and variance use
// every finite pair). Documented and reported per layer.
constexpr double kRobustEps = 1e-6;

struct LayerAcc {
    double n = 0;                // finite score pairs accumulated
    double sum_bs = 0;           // sum(b*s)
    double sum_s2 = 0;           // sum(s*s)
    double sum_b2 = 0;           // sum(b*b)
    double sum_b = 0;
    double sum_s = 0;
    double sum_abs_err = 0;      // sum |b - s|      (before scaling)
    double sum_sq_err = 0;       // sum (b - s)^2    (before scaling) == SSE(1)

    uint64_t hist[kHistBins] = {0};   // log10(|b|/|s|) histogram for robust median
    uint64_t hist_n = 0;

    // --- provenance / domination diagnostics ---
    uint64_t n_nonfinite = 0;         // pairs rejected as non-finite
    uint64_t n_robust_excluded = 0;   // pairs excluded from the robust estimator
    uint64_t n_robust_near_zero = 0;  // ...because |b| or |s| <= kRobustEps
    uint64_t n_robust_clipped = 0;    // ...ratio outside [kHistLo,kHistHi], folded into an edge bin
    uint64_t n_head_overflow = 0;     // samples whose head id exceeded kMaxHeads
    uint64_t n_vectors = 0;           // query vectors (t,h) contributing
    uint64_t n_callbacks = 0;         // op invocations contributing

    uint64_t head_n[kMaxHeads] = {0}; // per-head sample count
    double head_bs[kMaxHeads] = {0};  // per-head sum(b*s)
    double head_s2[kMaxHeads] = {0};  // per-head sum(s*s)

    uint64_t pos_n[kPosBuckets] = {0};
    double pos_bs[kPosBuckets] = {0};
    double pos_s2[kPosBuckets] = {0};

    uint64_t mag_n[kMagBins] = {0};   // count of samples by log10|s| decade
    double mag_s2[kMagBins] = {0};    // sum(s*s) by decade -> OLS denominator weight

    // Accumulate one (baseline, slha) score pair for query head `head` at query
    // position bucket derived from `t_frac` in [0,1). Non-finite pairs are
    // counted and skipped.
    void add(double b, double s, int head, double t_frac);
    // Convenience for callers with no head/position context (tests).
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
    double pearson() const;              // corr(b, s): magnitude-invariant alignment
    double var_ratio() const;            // var(s) / var(b)
    double mae_before() const;           // mean |b - s|
    double sse_before() const;           // sum (b - s)^2
    double sse_after(double a) const;    // sum (b - a s)^2
    // SSE(1) - SSE(a_ols) computed in the cancellation-free closed form
    // sum_s2 * (1 - a_ols)^2, which is exact and avoids differencing two large
    // nearly-equal sums when the fitted scale is close to 1.
    double sse_reduction_at_ols() const;
    double rms_before() const;           // sqrt(mean (b - s)^2)
    double rms_after(double a) const;    // sqrt(mean (b - a s)^2)
    double rel_l2_before() const;        // ||b - s|| / ||b||
    double rel_l2_after(double a) const; // ||b - a s|| / ||b||

    // Per-head OLS scale (returns 1.0 for an empty/out-of-range head).
    double head_scale(int head) const;
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

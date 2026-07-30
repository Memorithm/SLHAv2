// Implementation of the SLHA per-layer score-scale fitting accumulator.
// See slha_scale_fit.hpp for the design rationale.
#include "slha_scale_fit.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <sstream>

namespace slha_scale_fit {

namespace {
// Map a value into [0, nbins) over [lo, hi], clamping. Non-finite -> nbins (invalid).
inline int bin_of(double v, double lo, double hi, int nbins) {
    if (!std::isfinite(v)) {
        return nbins;  // caller must treat as invalid; never used as an index
    }
    const double w = (hi - lo) / static_cast<double>(nbins);
    const double x = (v - lo) / w;
    if (x <= 0.0) {
        return 0;
    }
    if (x >= static_cast<double>(nbins - 1)) {
        return nbins - 1;
    }
    return static_cast<int>(x);
}
}  // namespace

void LayerAcc::add(double b, double s, int head, double t_frac) {
    // Skip non-finite pairs so the running sums stay clean.
    if (!std::isfinite(b) || !std::isfinite(s)) {
        ++n_nonfinite;
        return;
    }
    n += 1.0;
    sum_bs += b * s;
    sum_s2 += s * s;
    sum_b2 += b * b;
    sum_b += b;
    sum_s += s;
    const double e = b - s;
    sum_abs_err += std::fabs(e);
    sum_sq_err += e * e;

    // Per-head sufficient statistics (bounded).
    if (head >= 0 && head < kMaxHeads) {
        head_n[head] += 1;
        head_bs[head] += b * s;
        head_s2[head] += s * s;
    } else if (head >= kMaxHeads) {
        ++n_head_overflow;   // per-head stats would be incomplete; make it visible
    }

    // Query-position bucket (bounded). t_frac in [0,1); a negative value is the
    // "no position context" sentinel and is excluded rather than clamped into
    // bucket 0. The index is clamped in double space BEFORE the cast so the
    // conversion can never be out of range (float-cast-overflow is UB).
    if (std::isfinite(t_frac) && t_frac >= 0.0) {
        const int pb = bin_of(t_frac, 0.0, 1.0, kPosBuckets);
        if (pb >= 0 && pb < kPosBuckets) {
            pos_n[pb] += 1;
            pos_bs[pb] += b * s;
            pos_s2[pb] += s * s;
        }
    }

    // Magnitude distribution of the OLS denominator: which |s| decades carry
    // the sample count, and which carry the s^2 weight.
    const double as = std::fabs(s);
    if (as > 0.0) {
        const int mb = bin_of(std::log10(as), kMagLo, kMagHi, kMagBins);
        if (mb >= 0 && mb < kMagBins) {
            mag_n[mb] += 1;
            mag_s2[mb] += s * s;
        }
    } else {
        mag_n[0] += 1;  // exact zero -> lowest decade, contributes 0 to s^2
    }

    // Robust magnitude-ratio histogram (log10 |b|/|s|). Ratios are unstable when
    // either magnitude is at/below kRobustEps, so those pairs are excluded here
    // (and only here — OLS and variance still use them).
    if (std::fabs(b) > kRobustEps && as > kRobustEps) {
        const double r = std::log10(std::fabs(b) / as);
        const int bin = bin_of(r, kHistLo, kHistHi, kHistBins);
        if (bin >= 0 && bin < kHistBins) {
            hist[bin] += 1;
            hist_n += 1;
            // Record ratios that saturated an edge bin: the median is then a
            // lower/upper bound rather than an estimate.
            if (r <= kHistLo || r >= kHistHi) {
                ++n_robust_clipped;
            }
        } else {
            ++n_robust_excluded;  // non-finite ratio (defensive)
        }
    } else {
        ++n_robust_near_zero;
        ++n_robust_excluded;
    }
}

void LayerAcc::add(double b, double s) { add(b, s, -1, -1.0); }

void LayerAcc::merge(const LayerAcc & o) {
    n += o.n;
    sum_bs += o.sum_bs;
    sum_s2 += o.sum_s2;
    sum_b2 += o.sum_b2;
    sum_b += o.sum_b;
    sum_s += o.sum_s;
    sum_abs_err += o.sum_abs_err;
    sum_sq_err += o.sum_sq_err;
    for (int i = 0; i < kHistBins; ++i) {
        hist[i] += o.hist[i];
    }
    hist_n += o.hist_n;
    n_nonfinite += o.n_nonfinite;
    n_robust_excluded += o.n_robust_excluded;
    n_robust_near_zero += o.n_robust_near_zero;
    n_robust_clipped += o.n_robust_clipped;
    n_head_overflow += o.n_head_overflow;
    n_vectors += o.n_vectors;
    n_callbacks += o.n_callbacks;
    for (int i = 0; i < kMaxHeads; ++i) {
        head_n[i] += o.head_n[i];
        head_bs[i] += o.head_bs[i];
        head_s2[i] += o.head_s2[i];
    }
    for (int i = 0; i < kPosBuckets; ++i) {
        pos_n[i] += o.pos_n[i];
        pos_bs[i] += o.pos_bs[i];
        pos_s2[i] += o.pos_s2[i];
    }
    for (int i = 0; i < kMagBins; ++i) {
        mag_n[i] += o.mag_n[i];
        mag_s2[i] += o.mag_s2[i];
    }
}

double LayerAcc::ols_scale() const {
    return sum_s2 > 0.0 ? sum_bs / sum_s2 : 1.0;
}

double LayerAcc::robust_scale() const {
    if (hist_n == 0) {
        return 1.0;
    }
    // Lower-median bin: first bin whose cumulative count exceeds hist_n/2.
    const uint64_t target = hist_n / 2;
    uint64_t cum = 0;
    int bin = kHistBins - 1;
    for (int i = 0; i < kHistBins; ++i) {
        cum += hist[i];
        if (cum > target) {
            bin = i;
            break;
        }
    }
    const double binw = (kHistHi - kHistLo) / static_cast<double>(kHistBins);
    const double log_r = kHistLo + (static_cast<double>(bin) + 0.5) * binw;
    return std::pow(10.0, log_r);
}

double LayerAcc::variance_scale() const {
    const double sb = std_b();
    const double ss = std_s();
    return ss > 0.0 ? sb / ss : 1.0;
}

double LayerAcc::mean_b() const { return n > 0.0 ? sum_b / n : 0.0; }
double LayerAcc::mean_s() const { return n > 0.0 ? sum_s / n : 0.0; }

double LayerAcc::std_b() const {
    if (n <= 1.0) {
        return 0.0;
    }
    const double mb = sum_b / n;
    const double var = sum_b2 / n - mb * mb;
    return var > 0.0 ? std::sqrt(var) : 0.0;
}

double LayerAcc::std_s() const {
    if (n <= 1.0) {
        return 0.0;
    }
    const double ms = sum_s / n;
    const double var = sum_s2 / n - ms * ms;
    return var > 0.0 ? std::sqrt(var) : 0.0;
}

double LayerAcc::pearson() const {
    if (n <= 1.0) {
        return 0.0;
    }
    const double mb = sum_b / n;
    const double ms = sum_s / n;
    const double cov = sum_bs / n - mb * ms;
    const double vb = sum_b2 / n - mb * mb;
    const double vs = sum_s2 / n - ms * ms;
    const double denom = std::sqrt(vb * vs);
    return denom > 0.0 ? cov / denom : 0.0;
}

double LayerAcc::var_ratio() const {
    if (n <= 1.0) {
        return 0.0;
    }
    const double mb = sum_b / n;
    const double ms = sum_s / n;
    const double vb = sum_b2 / n - mb * mb;
    const double vs = sum_s2 / n - ms * ms;
    return vb > 0.0 ? vs / vb : 0.0;
}

double LayerAcc::mae_before() const { return n > 0.0 ? sum_abs_err / n : 0.0; }

double LayerAcc::sse_before() const { return sum_sq_err; }

double LayerAcc::sse_after(double a) const {
    // sum (b - a s)^2 = sum_b2 - 2 a sum_bs + a^2 sum_s2
    return std::max(0.0, sum_b2 - 2.0 * a * sum_bs + a * a * sum_s2);
}

double LayerAcc::sse_reduction_at_ols() const {
    // SSE(1) - SSE(a*) = sum_s2 * (1 - a*)^2  with a* = sum_bs / sum_s2.
    if (sum_s2 <= 0.0) {
        return 0.0;
    }
    const double a = sum_bs / sum_s2;
    const double d = 1.0 - a;
    return sum_s2 * d * d;
}

double LayerAcc::rms_before() const {
    return n > 0.0 ? std::sqrt(std::max(0.0, sum_sq_err / n)) : 0.0;
}

double LayerAcc::rms_after(double a) const {
    return n > 0.0 ? std::sqrt(sse_after(a) / n) : 0.0;
}

double LayerAcc::rel_l2_before() const {
    return sum_b2 > 0.0 ? std::sqrt(std::max(0.0, sum_sq_err) / sum_b2) : 0.0;
}

double LayerAcc::rel_l2_after(double a) const {
    return sum_b2 > 0.0 ? std::sqrt(sse_after(a) / sum_b2) : 0.0;
}

double LayerAcc::head_scale(int head) const {
    if (head < 0 || head >= kMaxHeads || head_s2[head] <= 0.0) {
        return 1.0;
    }
    return head_bs[head] / head_s2[head];
}

// --------------------------------------------------------------------------
// Process-wide registry
// --------------------------------------------------------------------------
namespace {
std::mutex g_mutex;
std::atomic<bool> g_enabled{false};
LayerAcc g_layers[kMaxLayers];
bool g_seen[kMaxLayers] = {false};
}  // namespace

void reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (int i = 0; i < kMaxLayers; ++i) {
        g_layers[i] = LayerAcc();
        g_seen[i] = false;
    }
}

void enable(bool on) { g_enabled.store(on, std::memory_order_release); }

bool enabled() { return g_enabled.load(std::memory_order_acquire); }

void merge_layer(int32_t layer_id, const LayerAcc & partial) {
    if (layer_id < 0 || layer_id >= kMaxLayers) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_layers[layer_id].merge(partial);
    g_seen[layer_id] = true;
}

static void emit_num(std::ostringstream & os, double v) {
    if (!std::isfinite(v)) {
        os << "null";  // JSON has no NaN/Inf; emit null so json.tool accepts it
    } else {
        os << v;
    }
}

std::string dump_json() {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::ostringstream os;
    os.precision(9);
    os << "{\n";
    os << "  \"schema\": \"slha_score_scale_fit_v2\",\n";
    os << "  \"description\": \"Per-layer multiplicative score-scale fit "
          "(baseline ~= a * slha) over causally-unmasked shadow positions, with "
          "per-head / position / magnitude domination diagnostics.\",\n";
    os << "  \"robust_eps\": " << kRobustEps << ",\n";
    os << "  \"robust_eps_note\": \"pairs with |b|<=eps or |s|<=eps are excluded "
          "from the ROBUST median-ratio estimator only; OLS and variance use every finite pair\",\n";
    os << "  \"pos_buckets\": " << kPosBuckets << ",\n";
    os << "  \"layers\": {\n";
    bool first = true;
    for (int i = 0; i < kMaxLayers; ++i) {
        if (!g_seen[i] || g_layers[i].n <= 0.0) {
            continue;
        }
        const LayerAcc & a = g_layers[i];
        const double ols = a.ols_scale();
        const double rob = a.robust_scale();
        const double var = a.variance_scale();
        if (!first) {
            os << ",\n";
        }
        first = false;
        os << "    \"" << i << "\": {";
        os << "\"n\": " << static_cast<uint64_t>(a.n);
        os << ", \"n_vectors\": " << a.n_vectors;
        os << ", \"n_callbacks\": " << a.n_callbacks;
        os << ", \"n_nonfinite\": " << a.n_nonfinite;
        os << ", \"n_robust_excluded\": " << a.n_robust_excluded;
        os << ", \"n_robust_near_zero\": " << a.n_robust_near_zero;
        os << ", \"n_robust_clipped\": " << a.n_robust_clipped;
        os << ", \"n_head_overflow\": " << a.n_head_overflow;
        os << ", \"n_robust_used\": " << a.hist_n;
        os << ", \"ols_scale\": ";        emit_num(os, ols);
        os << ", \"robust_scale\": ";     emit_num(os, rob);
        os << ", \"variance_scale\": ";   emit_num(os, var);
        os << ", \"pearson\": ";          emit_num(os, a.pearson());
        os << ", \"var_ratio\": ";        emit_num(os, a.var_ratio());
        os << ", \"mean_b\": ";           emit_num(os, a.mean_b());
        os << ", \"mean_s\": ";           emit_num(os, a.mean_s());
        os << ", \"std_b\": ";            emit_num(os, a.std_b());
        os << ", \"std_s\": ";            emit_num(os, a.std_s());
        os << ", \"mae_before\": ";       emit_num(os, a.mae_before());
        os << ", \"sse_before\": ";       emit_num(os, a.sse_before());
        os << ", \"sse_after_ols\": ";    emit_num(os, a.sse_after(ols));
        os << ", \"sse_reduction_exact\": "; emit_num(os, a.sse_reduction_at_ols());
        os << ", \"sse_after_robust\": "; emit_num(os, a.sse_after(rob));
        os << ", \"sum_b2\": ";           emit_num(os, a.sum_b2);
        os << ", \"sum_s2\": ";           emit_num(os, a.sum_s2);
        os << ", \"sum_bs\": ";           emit_num(os, a.sum_bs);
        os << ", \"rms_before\": ";       emit_num(os, a.rms_before());
        os << ", \"rms_after_ols\": ";    emit_num(os, a.rms_after(ols));
        os << ", \"rel_l2_before\": ";    emit_num(os, a.rel_l2_before());
        os << ", \"rel_l2_after_ols\": "; emit_num(os, a.rel_l2_after(ols));
        os << ", \"rel_l2_after_robust\": "; emit_num(os, a.rel_l2_after(rob));
        os << ", \"rel_l2_after_variance\": "; emit_num(os, a.rel_l2_after(var));
        // per-head OLS scales and counts (only heads that were observed)
        os << ", \"head_ids\": [";
        {
            bool fh = true;
            for (int h = 0; h < kMaxHeads; ++h) {
                if (a.head_n[h] == 0) continue;
                if (!fh) os << ",";
                fh = false;
                os << h;
            }
        }
        os << "]";
        os << ", \"head_n\": [";
        {
            bool fh = true;
            for (int h = 0; h < kMaxHeads; ++h) {
                if (a.head_n[h] == 0) continue;
                if (!fh) os << ",";
                fh = false;
                os << a.head_n[h];
            }
        }
        os << "]";
        os << ", \"head_scale\": [";
        {
            bool fh = true;
            for (int h = 0; h < kMaxHeads; ++h) {
                if (a.head_n[h] == 0) continue;
                if (!fh) os << ",";
                fh = false;
                emit_num(os, a.head_scale(h));
            }
        }
        os << "]";
        os << ", \"pos_n\": [";
        for (int p = 0; p < kPosBuckets; ++p) {
            if (p) os << ",";
            os << a.pos_n[p];
        }
        os << "]";
        os << ", \"pos_scale\": [";
        for (int p = 0; p < kPosBuckets; ++p) {
            if (p) os << ",";
            emit_num(os, a.pos_s2[p] > 0.0 ? a.pos_bs[p] / a.pos_s2[p] : 1.0);
        }
        os << "]";
        // magnitude decades: count and s^2 weight (OLS denominator contribution)
        os << ", \"mag_lo\": " << kMagLo << ", \"mag_hi\": " << kMagHi
           << ", \"mag_bins\": " << kMagBins;
        os << ", \"mag_n\": [";
        for (int m = 0; m < kMagBins; ++m) {
            if (m) os << ",";
            os << a.mag_n[m];
        }
        os << "]";
        os << ", \"mag_s2\": [";
        for (int m = 0; m < kMagBins; ++m) {
            if (m) os << ",";
            emit_num(os, a.mag_s2[m]);
        }
        os << "]";
        os << "}";
    }
    os << "\n  }\n";
    os << "}\n";
    return os.str();
}

}  // namespace slha_scale_fit

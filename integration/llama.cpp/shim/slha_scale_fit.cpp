// Implementation of the SLHA per-layer score-scale fitting accumulator.
// See slha_scale_fit.hpp for the design rationale.
#include "slha_scale_fit.hpp"

#include <atomic>
#include <cmath>
#include <mutex>
#include <sstream>

namespace slha_scale_fit {

void LayerAcc::add(double b, double s) {
    // Skip non-finite pairs so the running sums stay clean.
    if (!std::isfinite(b) || !std::isfinite(s)) {
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

    // Robust magnitude-ratio histogram (log10 |b|/|s|). Only defined when both
    // magnitudes are nonzero; near-zero scalars carry no ratio information.
    if (b != 0.0 && s != 0.0) {
        const double r = std::log10(std::fabs(b) / std::fabs(s));
        const double binw = (kHistHi - kHistLo) / static_cast<double>(kHistBins);
        int bin = static_cast<int>(std::floor((r - kHistLo) / binw));
        if (bin < 0) {
            bin = 0;
        }
        if (bin >= kHistBins) {
            bin = kHistBins - 1;
        }
        hist[bin] += 1;
        hist_n += 1;
    }
}

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

double LayerAcc::rms_before() const {
    return n > 0.0 ? std::sqrt(std::max(0.0, sum_sq_err / n)) : 0.0;
}

double LayerAcc::rms_after(double a) const {
    if (n <= 0.0) {
        return 0.0;
    }
    // sum (b - a s)^2 = sum_b2 - 2 a sum_bs + a^2 sum_s2
    const double num = sum_b2 - 2.0 * a * sum_bs + a * a * sum_s2;
    return std::sqrt(std::max(0.0, num) / n);
}

double LayerAcc::rel_l2_before() const {
    return sum_b2 > 0.0 ? std::sqrt(std::max(0.0, sum_sq_err) / sum_b2) : 0.0;
}

double LayerAcc::rel_l2_after(double a) const {
    if (sum_b2 <= 0.0) {
        return 0.0;
    }
    const double num = sum_b2 - 2.0 * a * sum_bs + a * a * sum_s2;
    return std::sqrt(std::max(0.0, num) / sum_b2);
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
    os << "  \"schema\": \"slha_score_scale_fit_v1\",\n";
    os << "  \"description\": \"Per-layer multiplicative score-scale fit "
          "(baseline ~= a * slha) over causal shadow positions.\",\n";
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
        os << ", \"rms_before\": ";       emit_num(os, a.rms_before());
        os << ", \"rms_after_ols\": ";    emit_num(os, a.rms_after(ols));
        os << ", \"rms_after_robust\": "; emit_num(os, a.rms_after(rob));
        os << ", \"rel_l2_before\": ";    emit_num(os, a.rel_l2_before());
        os << ", \"rel_l2_after_ols\": "; emit_num(os, a.rel_l2_after(ols));
        os << ", \"rel_l2_after_robust\": "; emit_num(os, a.rel_l2_after(rob));
        os << ", \"rel_l2_after_variance\": "; emit_num(os, a.rel_l2_after(var));
        os << "}";
    }
    os << "\n  }\n";
    os << "}\n";
    return os.str();
}

}  // namespace slha_scale_fit

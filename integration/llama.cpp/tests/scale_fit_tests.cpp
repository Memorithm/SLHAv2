// Production-linked tests for the SLHA score-scale fitting math.
// Links the real accumulator (../shim/slha_scale_fit.cpp); no logic is copied.
#include "slha_scale_fit.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

using slha_scale_fit::LayerAcc;

static int g_failures = 0;
#define TEST(name) do { std::printf("  test: %s ... ", name); } while (0)
#define CHECK(c) do { if (!(c)) { std::printf("FAIL(line %d) ", __LINE__); ++g_failures; } } while (0)
#define DONE() do { std::printf("ok\n"); } while (0)

// Deterministic pseudo-random in [-1, 1] (no <random>, no Math.random analogue).
static double prand(uint64_t & st) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint64_t x = (st >> 11);
    return (static_cast<double>(x % 2000000) / 1000000.0) - 1.0;
}

int main() {
    std::printf("=== SLHA scale-fit math tests ===\n");

    // 1. OLS recovers the exact scale for perfect b = a*s
    TEST("ols recovers exact scale (noiseless)");
    { LayerAcc acc; uint64_t st = 12345;
      const double a_true = 1.37;
      for (int i = 0; i < 5000; ++i) { double s = 2.0 * prand(st); double b = a_true * s; acc.add(b, s); }
      CHECK(std::fabs(acc.ols_scale() - a_true) < 1e-9); } DONE();

    // 2. rel_l2 and rms are exactly zero after the OLS scale for perfect data
    TEST("perfect fit -> zero residual after scaling");
    { LayerAcc acc; uint64_t st = 999;
      const double a_true = 0.62;
      for (int i = 0; i < 4000; ++i) { double s = prand(st) + 1.5; double b = a_true * s; acc.add(b, s); }
      const double a = acc.ols_scale();
      CHECK(std::fabs(a - a_true) < 1e-9);
      CHECK(acc.rel_l2_after(a) < 1e-9);
      CHECK(acc.rms_after(a) < 1e-9);
      CHECK(acc.rel_l2_before() > 0.1);   // large error before correction
    } DONE();

    // 3. pearson correlation is 1.0 for perfectly linear data
    TEST("pearson == 1 for linear data");
    { LayerAcc acc; uint64_t st = 7;
      for (int i = 0; i < 3000; ++i) { double s = prand(st) * 3.0; acc.add(0.8 * s, s); }
      CHECK(std::fabs(acc.pearson() - 1.0) < 1e-9); } DONE();

    // 4. OLS is the least-squares optimum: no other scale gives smaller residual
    TEST("ols minimizes residual (noisy)");
    { LayerAcc acc; uint64_t st = 4242;
      const double a_true = 1.15;
      for (int i = 0; i < 20000; ++i) {
          double s = prand(st) * 2.0 + 2.0;
          double b = a_true * s + 0.15 * prand(st);   // additive noise
          acc.add(b, s);
      }
      const double a = acc.ols_scale();
      const double r_opt = acc.rms_after(a);
      CHECK(r_opt <= acc.rms_after(a - 0.05) + 1e-12);
      CHECK(r_opt <= acc.rms_after(a + 0.05) + 1e-12);
      CHECK(std::fabs(a - a_true) < 0.02);   // close to truth under small noise
    } DONE();

    // 5. robust (median-ratio) scale recovers a for strictly-positive b = a*s
    TEST("robust scale recovers median ratio");
    { LayerAcc acc; uint64_t st = 555;
      const double a_true = 0.75;
      for (int i = 0; i < 8000; ++i) {
          double s = 0.5 + (prand(st) + 1.0);       // strictly positive s in (0.5, 2.5)
          acc.add(a_true * s, s);
      }
      // Histogram bin width is 0.01 in log10; 10^0.01 ~= 1.023 tolerance.
      CHECK(std::fabs(acc.robust_scale() - a_true) < 0.03); } DONE();

    // 6. variance-matching scale equals |a| for b = a*s
    TEST("variance scale matches |a|");
    { LayerAcc acc; uint64_t st = 22;
      const double a_true = 1.9;
      for (int i = 0; i < 4000; ++i) { double s = prand(st) * 2.0; acc.add(a_true * s, s); }
      CHECK(std::fabs(acc.variance_scale() - a_true) < 1e-6); } DONE();

    // 7. merge is additive: two halves merged == one accumulator over all data
    TEST("merge is additive");
    { LayerAcc whole, h1, h2; uint64_t st = 314159;
      const double a_true = 1.23;
      std::vector<std::pair<double,double>> data;
      for (int i = 0; i < 6000; ++i) { double s = prand(st) * 2.0 + 1.0; double b = a_true * s + 0.1 * prand(st); data.push_back({b, s}); }
      for (size_t i = 0; i < data.size(); ++i) {
          whole.add(data[i].first, data[i].second);
          if (i % 2 == 0) h1.add(data[i].first, data[i].second);
          else            h2.add(data[i].first, data[i].second);
      }
      LayerAcc merged; merged.merge(h1); merged.merge(h2);
      CHECK(std::fabs(merged.n - whole.n) < 1e-9);
      CHECK(std::fabs(merged.ols_scale() - whole.ols_scale()) < 1e-9);
      CHECK(std::fabs(merged.rel_l2_before() - whole.rel_l2_before()) < 1e-9);
      CHECK(merged.hist_n == whole.hist_n); } DONE();

    // 8. scaling reduces residual when there is a genuine magnitude mismatch
    TEST("scaling reduces residual vs identity");
    { LayerAcc acc; uint64_t st = 88;
      const double a_true = 2.5;   // slha scores are 2.5x too small
      for (int i = 0; i < 5000; ++i) { double s = prand(st) + 1.2; double b = a_true * s + 0.05 * prand(st); acc.add(b, s); }
      const double a = acc.ols_scale();
      CHECK(acc.rel_l2_after(a) < acc.rel_l2_before());
      CHECK(acc.rel_l2_after(1.0) > acc.rel_l2_after(a)); } DONE();

    // 9. non-finite pairs are skipped (do not poison the sums)
    TEST("non-finite pairs skipped");
    { LayerAcc acc;
      acc.add(1.0, 1.0); acc.add(2.0, 2.0);
      const double inf = std::numeric_limits<double>::infinity();
      const double nan = std::numeric_limits<double>::quiet_NaN();
      acc.add(inf, 1.0); acc.add(1.0, nan); acc.add(-inf, -inf);
      CHECK(acc.n == 2.0);
      CHECK(std::isfinite(acc.ols_scale()));
      CHECK(std::fabs(acc.ols_scale() - 1.0) < 1e-12); } DONE();

    // 10. mae_before and empty-accumulator safety
    TEST("mae_before and empty safety");
    { LayerAcc acc;
      CHECK(acc.ols_scale() == 1.0);       // empty -> neutral
      CHECK(acc.robust_scale() == 1.0);
      CHECK(acc.mae_before() == 0.0);
      acc.add(3.0, 1.0); acc.add(5.0, 1.0);   // |b-s| = 2, 4 -> mae 3
      CHECK(std::fabs(acc.mae_before() - 3.0) < 1e-12); } DONE();

    std::printf("=== scale-fit math tests complete: %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

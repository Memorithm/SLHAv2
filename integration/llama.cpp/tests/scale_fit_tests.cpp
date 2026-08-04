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

    // 11. SSE identity: sum_sq_err == sse_after(1.0) == sum_b2 - 2*sum_bs + sum_s2
    TEST("sse sufficient-statistic identity");
    { LayerAcc acc; uint64_t st = 606;
      for (int i = 0; i < 4000; ++i) { double s = prand(st) * 3.0; double b = 1.4 * s + 0.2 * prand(st); acc.add(b, s); }
      const double direct = acc.sse_before();
      const double viastats = acc.sse_after(1.0);
      CHECK(std::fabs(direct - viastats) / std::max(1.0, direct) < 1e-9);
      // and SSE is minimized exactly at the OLS scale
      const double a = acc.ols_scale();
      CHECK(acc.sse_after(a) <= acc.sse_after(1.0) + 1e-9); } DONE();

    // 12. near-zero denominators are excluded from the ROBUST estimator only
    TEST("near-zero pairs excluded from robust estimator");
    { LayerAcc acc;
      // 100 clean pairs with ratio 2.0
      for (int i = 0; i < 100; ++i) acc.add(2.0, 1.0);
      const uint64_t used_clean = acc.hist_n;
      CHECK(used_clean == 100);
      CHECK(acc.n_robust_excluded == 0);
      // pairs at/below kRobustEps must not enter the ratio histogram
      acc.add(1e-9, 1e-9); acc.add(0.0, 0.0); acc.add(5.0, 1e-12);
      CHECK(acc.hist_n == used_clean);          // histogram unchanged
      CHECK(acc.n_robust_excluded == 3);
      CHECK(acc.n == 103.0);                    // but OLS still counted them
      CHECK(std::fabs(acc.robust_scale() - 2.0) < 0.05); } DONE();

    // 13. non-finite pairs counted separately and excluded everywhere
    TEST("non-finite pairs counted");
    { LayerAcc acc;
      const double inf = std::numeric_limits<double>::infinity();
      const double nan = std::numeric_limits<double>::quiet_NaN();
      acc.add(1.0, 1.0);
      acc.add(inf, 1.0); acc.add(1.0, nan); acc.add(nan, inf);
      CHECK(acc.n == 1.0);
      CHECK(acc.n_nonfinite == 3);
      CHECK(std::isfinite(acc.ols_scale())); } DONE();

    // 14. per-head statistics are isolated and recover per-head scales
    TEST("per-head scales isolated");
    { LayerAcc acc; uint64_t st = 71;
      // head 0 has scale 2.0, head 3 has scale 0.5
      for (int i = 0; i < 500; ++i) { double s = prand(st) + 2.0; acc.add(2.0 * s, s, 0, 0.5); }
      for (int i = 0; i < 500; ++i) { double s = prand(st) + 2.0; acc.add(0.5 * s, s, 3, 0.5); }
      CHECK(std::fabs(acc.head_scale(0) - 2.0) < 1e-9);
      CHECK(std::fabs(acc.head_scale(3) - 0.5) < 1e-9);
      CHECK(acc.head_scale(1) == 1.0);          // unobserved head -> neutral
      CHECK(acc.head_scale(-1) == 1.0);         // out-of-range is safe
      CHECK(acc.head_scale(9999) == 1.0);
      CHECK(acc.head_n[0] == 500 && acc.head_n[3] == 500); } DONE();

    // 15. position buckets are bounded and out-of-range t_frac cannot escape
    TEST("position buckets bounded");
    { LayerAcc acc;
      acc.add(1.0, 1.0, 0, 0.0);      // first bucket
      acc.add(1.0, 1.0, 0, 0.999);    // last bucket
      acc.add(1.0, 1.0, 0, -5.0);     // negative == "no position" sentinel -> excluded
      acc.add(1.0, 1.0, 0, 5.0);      // out of range high -> clamped into last bucket
      uint64_t tot = 0;
      for (int p = 0; p < slha_scale_fit::kPosBuckets; ++p) tot += acc.pos_n[p];
      CHECK(tot == 3);                // the sentinel contributed no bucket sample
      CHECK(acc.pos_n[0] == 1);
      CHECK(acc.pos_n[slha_scale_fit::kPosBuckets - 1] == 2);
      // every accumulated pair still counts toward the aggregate statistics
      CHECK(acc.n == 4.0); } DONE();

    // 16. magnitude histogram carries the OLS denominator weight
    TEST("magnitude weight distribution");
    { LayerAcc acc;
      for (int i = 0; i < 1000; ++i) acc.add(1e-4, 1e-4);   // tiny scores
      for (int i = 0; i < 10; ++i)   acc.add(1e2, 1e2);     // large scores
      double tot_n = 0, tot_w = 0, big_w = 0;
      for (int m = 0; m < slha_scale_fit::kMagBins; ++m) { tot_n += acc.mag_n[m]; tot_w += acc.mag_s2[m]; }
      // the 10 large samples must dominate the s^2 weight despite being 1% of count
      for (int m = 0; m < slha_scale_fit::kMagBins; ++m) {
          double lo = -5.0 + 0.1 * m;
          if (lo >= 1.0) big_w += acc.mag_s2[m];
      }
      CHECK(tot_n == 1010);
      CHECK(std::fabs(tot_w - acc.sum_s2) / acc.sum_s2 < 1e-9);
      CHECK(big_w / tot_w > 0.99); } DONE();

    // 17. merge carries every new diagnostic array
    TEST("merge carries diagnostics");
    { LayerAcc h1, h2;
      h1.add(2.0, 1.0, 0, 0.1); h1.n_vectors = 3; h1.n_callbacks = 1;
      h2.add(2.0, 1.0, 1, 0.9); h2.n_vectors = 5; h2.n_callbacks = 1;
      h2.add(std::numeric_limits<double>::quiet_NaN(), 1.0);
      LayerAcc m; m.merge(h1); m.merge(h2);
      CHECK(m.n == 2.0);
      CHECK(m.n_vectors == 8 && m.n_callbacks == 2);
      CHECK(m.n_nonfinite == 1);
      CHECK(m.head_n[0] == 1 && m.head_n[1] == 1);
      CHECK(m.pos_n[0] == 1 && m.pos_n[slha_scale_fit::kPosBuckets - 1] == 1);
      CHECK(m.hist_n == h1.hist_n + h2.hist_n); } DONE();

    // 18. positive scaling preserves argmax / top-k / cosine (the central claim)
    TEST("positive scaling preserves ranking and cosine");
    { uint64_t st = 1234;
      for (int trial = 0; trial < 200; ++trial) {
          const int N = 64;
          std::vector<double> s(N), b(N);
          for (int i = 0; i < N; ++i) { s[i] = prand(st) * 100.0; b[i] = prand(st) * 100.0; }
          for (double a : {0.4, 0.9, 1.0, 1.3, 2.0}) {
              std::vector<double> sa(N);
              for (int i = 0; i < N; ++i) sa[i] = static_cast<float>(a * s[i]);  // f32 round-trip
              // argmax preserved
              int am = 0, ama = 0;
              for (int i = 1; i < N; ++i) { if (s[i] > s[am]) am = i; if (sa[i] > sa[ama]) ama = i; }
              CHECK(am == ama);
              // cosine with a fixed reference preserved
              auto cos = [&](const std::vector<double>& x){
                  double d=0,nx=0,nb=0; for(int i=0;i<N;++i){d+=x[i]*b[i];nx+=x[i]*x[i];nb+=b[i]*b[i];}
                  return d/std::sqrt(nx*nb); };
              CHECK(std::fabs(cos(s) - cos(sa)) < 1e-6);
          }
      } } DONE();

    // 19. exact SSE-reduction identity: SSE(1)-SSE(a*) == sum_s2*(1-a*)^2
    TEST("exact sse reduction identity");
    { LayerAcc acc; uint64_t st = 2027;
      for (int i = 0; i < 8000; ++i) { double s = prand(st) * 5.0 + 6.0; double b = 1.02 * s + 0.3 * prand(st); acc.add(b, s); }
      const double a = acc.ols_scale();
      const double diff = acc.sse_before() - acc.sse_after(a);
      const double exact = acc.sse_reduction_at_ols();
      CHECK(std::fabs(diff - exact) / std::max(1.0, exact) < 1e-6);
      CHECK(exact >= 0.0);
      // near-identity fit: reduction must be tiny relative to total SSE
      LayerAcc id; uint64_t st2 = 11;
      for (int i = 0; i < 8000; ++i) { double s = prand(st2) * 5.0; id.add(s + 0.5 * prand(st2), s); }
      CHECK(id.sse_reduction_at_ols() / id.sse_before() < 0.05); } DONE();

    // 20. near-zero exclusions are split and head overflow is visible
    TEST("exclusion counters split");
    { LayerAcc acc;
      acc.add(2.0, 1.0, 0, 0.5);
      acc.add(1e-9, 1e-9, 0, 0.5);          // near-zero -> near_zero + excluded
      acc.add(1.0, 1.0, 999, 0.5);          // head beyond kMaxHeads -> overflow
      CHECK(acc.n_robust_near_zero == 1);
      CHECK(acc.n_robust_excluded == 1);
      CHECK(acc.n_head_overflow == 1);
      // negative t_frac is the "no position" sentinel: excluded, not bucket 0
      LayerAcc q; q.add(1.0, 1.0);
      uint64_t tot = 0;
      for (int p2 = 0; p2 < slha_scale_fit::kPosBuckets; ++p2) tot += q.pos_n[p2];
      CHECK(tot == 0); } DONE();

    std::printf("=== scale-fit math tests complete: %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

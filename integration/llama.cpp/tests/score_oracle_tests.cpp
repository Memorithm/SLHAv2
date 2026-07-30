// Production-linked tests for the SLHA diagnostic score oracles.
// Links the real implementation (../shim/slha_score_oracle.cpp); the oracle
// transformations are never re-implemented here.
#include "slha_score_oracle.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace slha_oracle;

static int g_failures = 0;
#define TEST(name) do { std::printf("  test: %s ... ", name); } while (0)
#define CHECK(c) do { if (!(c)) { std::printf("FAIL(line %d) ", __LINE__); ++g_failures; } } while (0)
#define DONE() do { std::printf("ok\n"); } while (0)

static Workspace g_ws;
static std::vector<float> g_sx, g_sy;

// Apply through the real production entry point.
static ApplyStatus run(Mode m, const std::vector<float> & b, const std::vector<float> & s,
                       std::vector<float> & out, int k = 0, uint64_t * ties = nullptr) {
    out.assign(b.size(), 0.0f);
    uint64_t local = 0;
    return apply(m, k, b.data(), s.data(), b.size(), out.data(),
                 ties ? ties : &local, g_ws);
}

static bool bit_equal(const std::vector<float> & x, const std::vector<float> & y) {
    if (x.size() != y.size()) return false;
    return std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

static bool ranking_equal(const std::vector<float> & x, const std::vector<float> & y) {
    return same_ranking(x.data(), y.data(), x.size(), g_ws);
}

static bool multiset_equal(const std::vector<float> & x, const std::vector<float> & y) {
    return same_value_multiset(x.data(), y.data(), x.size(), g_sx, g_sy);
}

int main() {
    std::printf("=== SLHA score-oracle tests ===\n");

    const std::vector<float> B = {5.0f, 1.0f, 3.0f, 9.0f, 2.0f};
    const std::vector<float> S = {0.5f, 8.0f, 2.5f, 1.5f, 7.0f};
    std::vector<float> out;

    // 1. baseline identity
    TEST("baseline identity reproduces B bitwise");
    { CHECK(run(Mode::BaselineIdentity, B, S, out) == ApplyStatus::Ok);
      CHECK(bit_equal(out, B)); } DONE();

    // 2. SLHA identity
    TEST("slha identity reproduces S bitwise");
    { CHECK(run(Mode::SlhaIdentity, B, S, out) == ApplyStatus::Ok);
      CHECK(bit_equal(out, S)); } DONE();

    // 3. Oracle A: baseline ranking with SLHA values
    TEST("oracle A: baseline ranking + slha value multiset");
    { CHECK(run(Mode::BaselineRankSlhaValues, B, S, out) == ApplyStatus::Ok);
      CHECK(ranking_equal(out, B));      // ordering is baseline's
      CHECK(multiset_equal(out, S));     // values are SLHA's
      CHECK(!bit_equal(out, S)); } DONE();

    // 4. Oracle B: SLHA ranking with baseline values
    TEST("oracle B: slha ranking + baseline value multiset");
    { CHECK(run(Mode::SlhaRankBaselineValues, B, S, out) == ApplyStatus::Ok);
      CHECK(ranking_equal(out, S));
      CHECK(multiset_equal(out, B));
      CHECK(!bit_equal(out, B)); } DONE();

    // 5. already-identical vectors are a fixed point
    TEST("identical inputs are a fixed point");
    { CHECK(run(Mode::BaselineRankSlhaValues, B, B, out) == ApplyStatus::Ok);
      CHECK(bit_equal(out, B));          // self-sort + inverse permutation
      CHECK(run(Mode::SlhaRankBaselineValues, S, S, out) == ApplyStatus::Ok);
      CHECK(bit_equal(out, S)); } DONE();

    // 6. fully reversed rankings
    TEST("fully reversed rankings");
    { std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
      std::vector<float> r = {4.0f, 3.0f, 2.0f, 1.0f};
      CHECK(run(Mode::BaselineRankSlhaValues, a, r, out) == ApplyStatus::Ok);
      CHECK(ranking_equal(out, a));
      CHECK(multiset_equal(out, r));
      // a ascending, r's values descending -> out must equal r reversed = a
      CHECK(bit_equal(out, a)); } DONE();

    // 7. exact ties preserved as a multiset
    TEST("exact ties");
    { std::vector<float> a = {2.0f, 2.0f, 1.0f, 2.0f};
      std::vector<float> s2 = {5.0f, 5.0f, 5.0f, 9.0f};
      uint64_t ties = 0;
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, out, 0, &ties) == ApplyStatus::Ok);
      CHECK(multiset_equal(out, s2));
      CHECK(ties > 0); } DONE();

    // 8. deterministic tie resolution: repeated runs are bit-identical, and the
    //    tiebreak is by ascending original index
    TEST("deterministic tie resolution");
    { std::vector<float> a = {1.0f, 1.0f, 1.0f, 1.0f};
      std::vector<float> s2 = {4.0f, 3.0f, 2.0f, 1.0f};
      std::vector<float> o1, o2;
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, o1) == ApplyStatus::Ok);
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, o2) == ApplyStatus::Ok);
      CHECK(bit_equal(o1, o2));
      // all-equal baseline -> rank order is index order -> descending S values
      std::vector<float> want = {4.0f, 3.0f, 2.0f, 1.0f};
      CHECK(bit_equal(o1, want));
      // and the permutation helper agrees
      std::vector<int32_t> perm; uint64_t t = 0;
      rank_permutation(a.data(), a.size(), perm, &t);
      CHECK(perm[0] == 0 && perm[1] == 1 && perm[2] == 2 && perm[3] == 3);
      CHECK(t == 3); } DONE();

    // 9. negative scores
    TEST("negative scores");
    { std::vector<float> a = {-1.0f, -5.0f, -3.0f};
      std::vector<float> s2 = {-9.0f, -2.0f, -6.0f};
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, out) == ApplyStatus::Ok);
      CHECK(ranking_equal(out, a));
      CHECK(multiset_equal(out, s2)); } DONE();

    // 10. mixed positive and negative
    TEST("mixed sign scores");
    { std::vector<float> a = {-2.0f, 7.0f, 0.0f, -9.0f};
      std::vector<float> s2 = {3.0f, -4.0f, 8.0f, -1.0f};
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, out) == ApplyStatus::Ok);
      CHECK(ranking_equal(out, a));
      CHECK(multiset_equal(out, s2)); } DONE();

    // 11. infinities rejected
    TEST("infinities rejected");
    { const float inf = std::numeric_limits<float>::infinity();
      std::vector<float> a = {1.0f, inf, 2.0f};
      std::vector<float> s2 = {1.0f, 2.0f, 3.0f};
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, out) == ApplyStatus::NonFiniteInput);
      std::vector<float> a2 = {1.0f, 2.0f, 3.0f};
      std::vector<float> s3 = {1.0f, -inf, 3.0f};
      CHECK(run(Mode::BaselineRankSlhaValues, a2, s3, out) == ApplyStatus::NonFiniteInput); } DONE();

    // 12. NaN rejected
    TEST("NaN rejected");
    { const float nan = std::numeric_limits<float>::quiet_NaN();
      std::vector<float> a = {1.0f, nan, 2.0f};
      std::vector<float> s2 = {1.0f, 2.0f, 3.0f};
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, out) == ApplyStatus::NonFiniteInput);
      std::vector<float> a2 = {1.0f, 2.0f, 3.0f};
      std::vector<float> s3 = {nan, 2.0f, 3.0f};
      CHECK(run(Mode::SlhaRankBaselineValues, a2, s3, out) == ApplyStatus::NonFiniteInput); } DONE();

    // 13. mismatched lengths are a caller contract: the API takes one length,
    //     so a short destination is prevented structurally. Verify the length
    //     actually honoured is the one passed.
    TEST("length is honoured exactly");
    { std::vector<float> a = {3.0f, 1.0f, 2.0f, 9.0f};
      std::vector<float> s2 = {1.0f, 2.0f, 3.0f, 4.0f};
      std::vector<float> o(4, -77.0f);
      uint64_t t = 0;
      // apply over only the first 3 entries; the 4th must remain untouched
      CHECK(apply(Mode::BaselineRankSlhaValues, 0, a.data(), s2.data(), 3,
                  o.data(), &t, g_ws) == ApplyStatus::Ok);
      CHECK(o[3] == -77.0f); } DONE();

    // 14. empty active vector rejected
    TEST("empty row rejected");
    { std::vector<float> e;
      uint64_t t = 0;
      float dummy = 0.0f;
      CHECK(apply(Mode::BaselineRankSlhaValues, 0, &dummy, &dummy, 0, &dummy, &t, g_ws)
            == ApplyStatus::EmptyRow); } DONE();

    // 15. one-element row
    TEST("single element row");
    { std::vector<float> a = {4.0f}, s2 = {-2.0f};
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, out) == ApplyStatus::Ok);
      CHECK(out.size() == 1 && out[0] == -2.0f);
      CHECK(run(Mode::SlhaRankBaselineValues, a, s2, out) == ApplyStatus::Ok);
      CHECK(out[0] == 4.0f); } DONE();

    // 16. no partial destination write after a rejected row
    TEST("no partial write on failure");
    { const float nan = std::numeric_limits<float>::quiet_NaN();
      std::vector<float> a = {1.0f, nan, 3.0f, 4.0f};
      std::vector<float> s2 = {9.0f, 8.0f, 7.0f, 6.0f};
      std::vector<float> o(4, -42.0f);
      uint64_t t = 0;
      CHECK(apply(Mode::BaselineRankSlhaValues, 0, a.data(), s2.data(), 4,
                  o.data(), &t, g_ws) == ApplyStatus::NonFiniteInput);
      for (float v : o) CHECK(v == -42.0f);   // destination untouched
    } DONE();

    // 17. value multiset preservation across a large pseudo-random row
    TEST("value multiset preserved at scale");
    { uint64_t st = 991; std::vector<float> a(257), s2(257);
      auto nxt = [&st]() { st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                           return static_cast<float>(static_cast<int64_t>(st >> 33) % 20000) / 100.0f - 100.0f; };
      for (size_t i = 0; i < a.size(); ++i) { a[i] = nxt(); s2[i] = nxt(); }
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, out) == ApplyStatus::Ok);
      CHECK(multiset_equal(out, s2));
      CHECK(run(Mode::SlhaRankBaselineValues, a, s2, out) == ApplyStatus::Ok);
      CHECK(multiset_equal(out, a)); } DONE();

    // 18. rank order preservation at scale
    TEST("rank order preserved at scale");
    { uint64_t st = 4242; std::vector<float> a(129), s2(129);
      auto nxt = [&st]() { st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                           return static_cast<float>(static_cast<int64_t>(st >> 33) % 30000) / 100.0f - 150.0f; };
      for (size_t i = 0; i < a.size(); ++i) { a[i] = nxt(); s2[i] = nxt(); }
      CHECK(run(Mode::BaselineRankSlhaValues, a, s2, out) == ApplyStatus::Ok);
      CHECK(ranking_equal(out, a));
      CHECK(run(Mode::SlhaRankBaselineValues, a, s2, out) == ApplyStatus::Ok);
      CHECK(ranking_equal(out, s2)); } DONE();

    // 19. deterministic repeated execution
    TEST("deterministic repeated execution");
    { std::vector<float> o1, o2, o3;
      CHECK(run(Mode::BaselineRankSlhaValues, B, S, o1) == ApplyStatus::Ok);
      CHECK(run(Mode::BaselineRankSlhaValues, B, S, o2) == ApplyStatus::Ok);
      CHECK(run(Mode::BaselineTopKRank, B, S, o3, 2) == ApplyStatus::Ok);
      std::vector<float> o4;
      CHECK(run(Mode::BaselineTopKRank, B, S, o4, 2) == ApplyStatus::Ok);
      CHECK(bit_equal(o1, o2));
      CHECK(bit_equal(o3, o4)); } DONE();

    // 20. top-k: baseline top-k keys promoted, SLHA multiset kept
    TEST("top-k promotes baseline top-k keys");
    { uint64_t t = 0;
      CHECK(run(Mode::BaselineTopKRank, B, S, out, 1, &t) == ApplyStatus::Ok);
      CHECK(multiset_equal(out, S));
      // B's argmax is key 3 (value 9); it must now hold S's largest value (8)
      std::vector<int32_t> pb, po;
      rank_permutation(B.data(), B.size(), pb, nullptr);
      rank_permutation(out.data(), out.size(), po, nullptr);
      CHECK(po[0] == pb[0]);
      CHECK(out[pb[0]] == 8.0f); } DONE();

    // 21. top-k with k >= n is exactly Oracle A
    TEST("top-k saturates to full baseline ranking");
    { std::vector<float> oa, ok;
      CHECK(run(Mode::BaselineRankSlhaValues, B, S, oa) == ApplyStatus::Ok);
      CHECK(run(Mode::BaselineTopKRank, B, S, ok, static_cast<int>(B.size())) == ApplyStatus::Ok);
      CHECK(bit_equal(oa, ok));
      std::vector<float> ok2;
      CHECK(run(Mode::BaselineTopKRank, B, S, ok2, 99) == ApplyStatus::Ok);
      CHECK(bit_equal(oa, ok2)); } DONE();

    // 22. top-k keeps the relative SLHA order of the untouched tail
    TEST("top-k preserves tail slha order");
    { CHECK(run(Mode::BaselineTopKRank, B, S, out, 1) == ApplyStatus::Ok);
      std::vector<int32_t> pb, ps, po;
      rank_permutation(B.data(), B.size(), pb, nullptr);
      rank_permutation(S.data(), S.size(), ps, nullptr);
      rank_permutation(out.data(), out.size(), po, nullptr);
      // tail of the output ranking, in order, must equal S's ranking with the
      // promoted baseline key removed
      std::vector<int32_t> want;
      for (int32_t key : ps) if (key != pb[0]) want.push_back(key);
      std::vector<int32_t> got(po.begin() + 1, po.end());
      CHECK(got == want); } DONE();

    // 23. top-k rejects a non-positive k
    TEST("top-k rejects invalid k");
    { std::vector<float> o(B.size(), 0.0f); uint64_t t = 0;
      CHECK(apply(Mode::BaselineTopKRank, 0, B.data(), S.data(), B.size(), o.data(), &t, g_ws)
            == ApplyStatus::InvalidPermutation);
      CHECK(apply(Mode::BaselineTopKRank, -3, B.data(), S.data(), B.size(), o.data(), &t, g_ws)
            == ApplyStatus::InvalidPermutation); } DONE();

    // 24. specification parsing: known modes accepted, unknown rejected
    TEST("oracle mode parsing");
    { Config c;
      CHECK(parse_oracle("off", c) && !c.active && c.mode == Mode::Off);
      CHECK(parse_oracle("", c) && !c.active);
      CHECK(parse_oracle("baseline-identity", c) && c.active && c.mode == Mode::BaselineIdentity);
      CHECK(parse_oracle("slha-identity", c) && c.mode == Mode::SlhaIdentity);
      CHECK(parse_oracle("baseline-rank-slha-values", c) && c.mode == Mode::BaselineRankSlhaValues);
      CHECK(parse_oracle("slha-rank-baseline-values", c) && c.mode == Mode::SlhaRankBaselineValues);
      CHECK(parse_oracle("baseline-topk:8", c) && c.mode == Mode::BaselineTopKRank && c.topk == 8);
      // unknown / malformed are rejected, never silently defaulted
      CHECK(!parse_oracle("nonsense", c) && !c.valid);
      CHECK(!parse_oracle("baseline-topk:", c) && !c.valid);
      CHECK(!parse_oracle("baseline-topk:0", c) && !c.valid);
      CHECK(!parse_oracle("baseline-topk:-2", c) && !c.valid);
      CHECK(!parse_oracle("baseline-topk:abc", c) && !c.valid);
      CHECK(!parse_oracle("BASELINE-IDENTITY", c) && !c.valid);
      // config hash is deterministic and mode-specific
      Config c1, c2, c3;
      parse_oracle("baseline-topk:4", c1);
      parse_oracle("baseline-topk:4", c2);
      parse_oracle("baseline-topk:5", c3);
      CHECK(c1.config_sha256 == c2.config_sha256);
      CHECK(c1.config_sha256 != c3.config_sha256);
      CHECK(!c1.config_sha256.empty()); } DONE();

    // 25. invariant helpers behave (used for runtime sampling)
    TEST("invariant helper semantics");
    { std::vector<float> x = {1.0f, 2.0f, 3.0f};
      std::vector<float> y = {10.0f, 20.0f, 30.0f};   // same ranking, other values
      CHECK(ranking_equal(x, y));
      CHECK(!multiset_equal(x, y));
      std::vector<float> z = {3.0f, 2.0f, 1.0f};      // same multiset, other ranking
      CHECK(multiset_equal(x, z));
      CHECK(!ranking_equal(x, z));
      // -0.0 and +0.0 are distinguished by the multiset check
      std::vector<float> p = {0.0f}, m = {-0.0f};
      CHECK(!multiset_equal(p, m)); } DONE();

    // 26. tie-aware ordering invariant: with tied transplanted values the strict
    //     permutation check is the WRONG predicate; respects_ranking is correct.
    TEST("tie-aware ordering invariant");
    { std::vector<float> b = {1.0f, 9.0f};
      std::vector<float> s2 = {5.0f, 5.0f};      // S ties -> order unobservable
      CHECK(run(Mode::BaselineRankSlhaValues, b, s2, out) == ApplyStatus::Ok);
      CHECK(multiset_equal(out, s2));
      // strict permutation equality legitimately fails here ...
      CHECK(!ranking_equal(out, b));
      // ... but the tie-aware invariant holds, which is what the runtime checks
      CHECK(respects_ranking(out.data(), b.data(), b.size(), g_ws));
      // with DISTINCT values both predicates must agree
      std::vector<float> s3 = {5.0f, 6.0f};
      CHECK(run(Mode::BaselineRankSlhaValues, b, s3, out) == ApplyStatus::Ok);
      CHECK(ranking_equal(out, b));
      CHECK(respects_ranking(out.data(), b.data(), b.size(), g_ws)); } DONE();

    // 27. respects_ranking actually rejects a genuinely mis-ordered row
    TEST("tie-aware invariant rejects real violations");
    { std::vector<float> ref = {1.0f, 9.0f, 5.0f};
      std::vector<float> bad = {9.0f, 1.0f, 5.0f};   // ref ranks key1 first, bad ranks it last
      CHECK(!respects_ranking(bad.data(), ref.data(), ref.size(), g_ws));
      std::vector<float> good = {1.0f, 9.0f, 5.0f};
      CHECK(respects_ranking(good.data(), ref.data(), ref.size(), g_ws));
      // ties in the OUTPUT are accepted, a strict inversion is not
      std::vector<float> flat = {4.0f, 4.0f, 4.0f};
      CHECK(respects_ranking(flat.data(), ref.data(), ref.size(), g_ws)); } DONE();

    // 28. tie-heavy rows round-trip through every mode without violating invariants
    TEST("tie-heavy rows preserve invariants in all modes");
    { std::vector<float> b(64), s2(64);
      for (size_t i = 0; i < b.size(); ++i) {
          b[i]  = static_cast<float>(i % 4);      // many exact ties
          s2[i] = static_cast<float>(i % 3);      // many exact ties
      }
      for (Mode m : {Mode::BaselineRankSlhaValues, Mode::SlhaRankBaselineValues}) {
          CHECK(run(m, b, s2, out) == ApplyStatus::Ok);
          const std::vector<float> & ref = (m == Mode::BaselineRankSlhaValues) ? b : s2;
          const std::vector<float> & val = (m == Mode::BaselineRankSlhaValues) ? s2 : b;
          CHECK(respects_ranking(out.data(), ref.data(), out.size(), g_ws));
          CHECK(multiset_equal(out, val));
      }
      for (int k : {1, 4, 16}) {
          CHECK(run(Mode::BaselineTopKRank, b, s2, out, k) == ApplyStatus::Ok);
          CHECK(multiset_equal(out, s2));
      } } DONE();

    std::printf("=== score-oracle tests complete: %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

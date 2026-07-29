// Production-linked tests for SLHA experimental score scaling.
// Links the real parser (../shim/slha_score_scale.cpp); no logic is copied.
#include "slha_score_scale.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace slha_scale;

static int g_failures = 0;
#define TEST(name) do { std::printf("  test: %s ... ", name); } while (0)
#define CHECK(c) do { if (!(c)) { std::printf("FAIL(line %d) ", __LINE__); ++g_failures; } } while (0)
#define DONE() do { std::printf("ok\n"); } while (0)

static const int N = 28;

static ScaleMap penv(const std::string & s, int n = N) { ScaleMap m; parse_score_scale(s, "", n, m); return m; }
static ScaleMap pfile(const std::string & j, int n = N) { ScaleMap m; parse_score_scale("", j, n, m); return m; }

int main() {
    std::printf("=== SLHA score-scale tests ===\n");

    // 1. default scale 1.0
    TEST("default scale 1.0");
    { ScaleMap m; CHECK(parse_score_scale("", "", N, m)); CHECK(m.valid); CHECK(m.mode == Mode::Global);
      CHECK(m.get(0) == 1.0 && m.get(27) == 1.0); CHECK(!m.manifest_sha256.empty()); } DONE();

    // 2. valid global scale
    TEST("valid global scale");
    { auto m = penv("0.75"); CHECK(m.valid); CHECK(m.mode == Mode::Global);
      CHECK(std::fabs(m.get(3) - 0.75) < 1e-12); } DONE();

    // 3. valid per-layer scales
    TEST("valid per-layer scales");
    { auto m = penv("layer:0=0.91,5=0.72"); CHECK(m.valid); CHECK(m.mode == Mode::PerLayer);
      CHECK(std::fabs(m.get(0) - 0.91) < 1e-12); CHECK(std::fabs(m.get(5) - 0.72) < 1e-12);
      CHECK(m.get(1) == 1.0);  // unlisted -> 1.0
      CHECK(m.has_override(0) && !m.has_override(1)); } DONE();

    // 4. valid scale file
    TEST("valid scale file");
    { auto m = pfile("{\"global\":1.0,\"layers\":{\"0\":0.9,\"12\":1.1}}");
      CHECK(m.valid); CHECK(m.mode == Mode::PerLayer);
      CHECK(std::fabs(m.get(0)-0.9)<1e-12); CHECK(std::fabs(m.get(12)-1.1)<1e-12); } DONE();

    // 5. zero rejected
    TEST("zero rejected");
    { CHECK(!penv("0").valid); CHECK(!penv("0.0").valid); CHECK(!penv("layer:0=0").valid); } DONE();

    // 6. negative rejected
    TEST("negative rejected");
    { CHECK(!penv("-1").valid); CHECK(!penv("-0.5").valid); CHECK(!penv("layer:0=-0.3").valid); } DONE();

    // 7. NaN rejected
    TEST("NaN rejected");
    { CHECK(!penv("nan").valid); CHECK(!penv("NaN").valid); CHECK(!penv("layer:0=nan").valid);
      CHECK(!pfile("{\"layers\":{\"0\":nan}}").valid); } DONE();

    // 8. infinity rejected
    TEST("infinity rejected");
    { CHECK(!penv("inf").valid); CHECK(!penv("Infinity").valid); CHECK(!penv("layer:0=inf").valid); } DONE();

    // 9. malformed number rejected
    TEST("malformed number rejected");
    { CHECK(!penv("0.5x").valid); CHECK(!penv("abc").valid); CHECK(!penv("layer:0=1.2.3").valid);
      CHECK(!penv("layer:0=").valid); } DONE();

    // 10. duplicate layer rejected
    TEST("duplicate layer rejected");
    { CHECK(!penv("layer:0=0.9,0=1.1").valid);
      CHECK(!pfile("{\"layers\":{\"3\":0.9,\"3\":1.0}}").valid); } DONE();

    // 11. out-of-range layer rejected
    TEST("out-of-range layer rejected");
    { CHECK(!penv("layer:28=0.9").valid); CHECK(!penv("layer:99=0.9").valid);
      CHECK(penv("layer:27=0.9").valid); } DONE();

    // 12. missing selected-layer scale rejected
    TEST("missing selected-layer scale rejected");
    { auto m = penv("layer:0=0.9,5=0.72");
      std::vector<int32_t> sel = {0, 5, 12};  // 12 has no scale
      CHECK(!resolve_against_selected(m, sel)); CHECK(!m.error.empty());
      auto m2 = penv("layer:0=0.9,5=0.72,12=1.0");
      CHECK(resolve_against_selected(m2, sel));
      auto g = penv("0.8"); CHECK(resolve_against_selected(g, sel)); }  // global covers all
    DONE();

    // 13. deterministic parsing
    TEST("deterministic parsing");
    { auto a = penv("layer:5=0.72,0=0.91,12=1.1"); auto b = penv("layer:5=0.72,0=0.91,12=1.1");
      CHECK(a.valid && b.valid); CHECK(a.per_layer == b.per_layer);
      CHECK(a.manifest_sha256 == b.manifest_sha256); CHECK(a.canonical() == b.canonical()); } DONE();

    // 14. canonical form is order-independent (deterministic manifest)
    TEST("manifest independent of spec order");
    { auto a = penv("layer:0=0.91,5=0.72"); auto b = penv("layer:5=0.72,0=0.91");
      CHECK(a.manifest_sha256 == b.manifest_sha256); } DONE();

    // 15. sha256 known-answer (self-contained hash)
    TEST("sha256 known-answer");
    { CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
      CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"); } DONE();

    // 16. global covers all layers; per-layer leaves others at 1.0
    TEST("scale coverage semantics");
    { auto g = penv("1.25"); for (int i = 0; i < N; ++i) CHECK(std::fabs(g.get(i)-1.25)<1e-12);
      auto p = penv("layer:5=0.5"); CHECK(std::fabs(p.get(5)-0.5)<1e-12);
      for (int i = 0; i < N; ++i) if (i != 5) CHECK(p.get(i) == 1.0); } DONE();

    // 17. file with only global
    TEST("scale file global-only");
    { auto m = pfile("{\"global\":0.85}"); CHECK(m.valid); CHECK(m.mode == Mode::Global);
      CHECK(std::fabs(m.get(0)-0.85)<1e-12); } DONE();

    // 18. empty per-layer spec / empty layers rejected
    TEST("empty per-layer spec rejected");
    { CHECK(!penv("layer:").valid); CHECK(!pfile("{\"layers\":{}}").valid); } DONE();

    // 19. stray comma / empty token rejected
    TEST("stray comma rejected");
    { CHECK(!penv("layer:0=0.9,").valid); CHECK(!penv("layer:,0=0.9").valid);
      CHECK(!penv("layer:0=0.9,,5=0.7").valid); } DONE();

    // 20. file out-of-range and invalid values rejected
    TEST("scale file validation");
    { CHECK(!pfile("{\"layers\":{\"28\":0.9}}").valid);
      CHECK(!pfile("{\"layers\":{\"0\":-1}}").valid);
      CHECK(!pfile("{\"global\":0,\"layers\":{\"0\":0.9}}").valid);
      CHECK(pfile("{\"layers\":{\"0\":0.9,\"27\":1.1}}").valid); } DONE();

    std::printf("=== score-scale tests complete: %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

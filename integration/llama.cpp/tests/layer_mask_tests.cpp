// Production-linked tests for the SLHA score-replacement layer mask.
// Links the real parser (../shim/slha_layer_mask.cpp); no logic is copied.
#include "slha_layer_mask.hpp"

#include <cstdio>
#include <string>

using namespace slha_mask;

static int g_failures = 0;
#define TEST(name) do { std::printf("  test: %s ... ", name); } while (0)
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL(line %d) ", __LINE__); ++g_failures; } } while (0)
#define DONE() do { std::printf("ok\n"); } while (0)

static const int N = 28;  // model layer count for validation

static ParsedMask parse(const std::string & s, int n = N) {
    ParsedMask m;
    parse_layer_mask(s, n, m);
    return m;
}

int main() {
    std::printf("=== SLHA layer-mask tests ===\n");

    // 1. all
    TEST("all selects every layer");
    { auto m = parse("all"); CHECK(m.valid); CHECK(m.kind == MaskKind::All);
      CHECK(m.contains(0)); CHECK(m.contains(27)); CHECK(m.contains(13)); } DONE();

    // 2. none
    TEST("none selects no layer");
    { auto m = parse("none"); CHECK(m.valid); CHECK(m.kind == MaskKind::None);
      CHECK(!m.contains(0)); CHECK(!m.contains(27)); } DONE();

    // 3. single valid layer
    TEST("single valid layer");
    { auto m = parse("7"); CHECK(m.valid); CHECK(m.kind == MaskKind::Explicit);
      CHECK(m.ids.size() == 1); CHECK(m.ids[0] == 7);
      CHECK(m.contains(7)); CHECK(!m.contains(6)); CHECK(!m.contains(8)); } DONE();

    // 4. several comma-separated
    TEST("several comma-separated layers");
    { auto m = parse("3,7,12"); CHECK(m.valid); CHECK(m.ids.size() == 3);
      CHECK(m.ids[0] == 3 && m.ids[1] == 7 && m.ids[2] == 12);
      CHECK(m.contains(3) && m.contains(7) && m.contains(12) && !m.contains(4)); } DONE();

    // 5. inclusive range
    TEST("inclusive range");
    { auto m = parse("0-6"); CHECK(m.valid); CHECK(m.ids.size() == 7);
      CHECK(m.ids.front() == 0 && m.ids.back() == 6);
      CHECK(m.contains(0) && m.contains(6) && !m.contains(7)); } DONE();

    // 6. mixed ranges and individuals
    TEST("mixed ranges and individual layers");
    { auto m = parse("0-3,7,12-14"); CHECK(m.valid);
      // {0,1,2,3,7,12,13,14}
      CHECK(m.ids.size() == 8);
      CHECK(m.contains(2) && m.contains(7) && m.contains(13) && !m.contains(5) && !m.contains(11)); } DONE();

    // 7. duplicate ids deduplicated deterministically
    TEST("duplicate ids deduplicated");
    { auto m = parse("5,5,3,0-2,1"); CHECK(m.valid);
      // unique sorted {0,1,2,3,5}
      CHECK(m.ids.size() == 5);
      CHECK(m.ids[0] == 0 && m.ids[1] == 1 && m.ids[2] == 2 && m.ids[3] == 3 && m.ids[4] == 5); } DONE();

    // 8. negative layer rejected
    TEST("negative layer rejected");
    { auto m = parse("-1"); CHECK(!m.valid); CHECK(!m.error.empty()); } DONE();

    // 9. out-of-range layer rejected
    TEST("out-of-range layer rejected");
    { auto m = parse("28"); CHECK(!m.valid);
      auto m2 = parse("30"); CHECK(!m2.valid);
      auto ok = parse("27"); CHECK(ok.valid); } DONE();

    // 10. malformed range rejected
    TEST("malformed range rejected");
    { CHECK(!parse("6-3").valid);       // inverted
      CHECK(!parse("3-").valid);        // missing hi
      CHECK(!parse("-").valid);         // empty both
      CHECK(!parse("1-2-3").valid);     // extra dash
      CHECK(!parse("a-5").valid); }     // non-numeric
    DONE();

    // 11. empty value rejected (only 'none' selects nothing)
    TEST("empty value rejected");
    { CHECK(!parse("").valid);
      CHECK(!parse(",").valid);         // stray comma
      CHECK(!parse("3,,7").valid); }    // empty token
    DONE();

    // 12. resolved id list for display
    TEST("resolved ids for display");
    { auto all = parse("all"); CHECK(all.resolved_ids(28).size() == 28);
      CHECK(all.ids_to_string(4) == "0,1,2,3");
      auto none = parse("none"); CHECK(none.resolved_ids(28).empty());
      auto ex = parse("2,5"); CHECK(ex.ids_to_string(28) == "2,5"); } DONE();

    // 13. unselected-layer semantics via contains()
    TEST("unselected layers report not-contained");
    { auto m = parse("14-27"); CHECK(m.valid);
      for (int i = 0; i < 14; ++i) CHECK(!m.contains(i));
      for (int i = 14; i < 28; ++i) CHECK(m.contains(i)); } DONE();

    // 14. invalid mask carries a non-empty diagnostic (prevents measurement)
    TEST("invalid mask carries diagnostic");
    { auto m = parse("0-999"); CHECK(!m.valid); CHECK(!m.error.empty());
      auto m2 = parse("garbage"); CHECK(!m2.valid); CHECK(!m2.error.empty()); } DONE();

    // 15. repeated parsing is deterministic
    TEST("repeated parsing deterministic");
    { auto a = parse("12-14,0,0,7"); auto b = parse("12-14,0,0,7");
      CHECK(a.valid && b.valid); CHECK(a.ids == b.ids);
      CHECK(a.ids.size() == 5);  // {0,7,12,13,14}
      CHECK(a.ids[0] == 0 && a.ids[1] == 7 && a.ids[2] == 12 && a.ids[3] == 13 && a.ids[4] == 14); } DONE();

    // Deferred validation (num_layers<=0): syntax errors still caught, range not.
    TEST("deferred validation catches syntax not range");
    { ParsedMask m; CHECK(parse_layer_mask("0-999", 0, m));   // range OK when deferred
      ParsedMask n; CHECK(!parse_layer_mask("-5", 0, n));     // negative still rejected
      ParsedMask e; CHECK(!parse_layer_mask("", 0, e)); }     // empty still rejected
    DONE();

    std::printf("=== layer-mask tests complete: %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

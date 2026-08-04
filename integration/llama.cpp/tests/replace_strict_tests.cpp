#include "slha_replace_counters.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

static int g_failures = 0;

#define TEST(name) do { \
    std::printf("  test: %s ... ", name); \
    g_failures = 0; \
} while(0)

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::printf("FAIL (line %d)\n", __LINE__); \
        ++g_failures; \
        return; \
    } \
} while(0)

#define PASS() do { \
    if (g_failures == 0) std::printf("ok\n"); \
    else std::printf("FAIL with %d failures\n", g_failures); \
} while(0)

// ================================================================
// Test 1: empty run is invalid (zero callbacks)
// ================================================================
void test_empty_run_invalid() {
    TEST("empty run invalid (zero callbacks)");
    g_slha_replace_counters.reset();
    CHECK(g_slha_replace_counters.n_callbacks.load() == 0);
    CHECK(g_slha_replace_counters.is_valid() == false);
    PASS();
}

// ================================================================
// Test 1b: zero expected vectors is invalid
// ================================================================
void test_zero_expected_vectors_invalid() {
    TEST("zero expected vectors invalid");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(0);
    g_slha_replace_counters.n_expected_logits.fetch_add(100);
    CHECK(g_slha_replace_counters.is_valid() == false);
    PASS();
}

// ================================================================
// Test 1c: zero expected logits is invalid
// ================================================================
void test_zero_expected_logits_invalid() {
    TEST("zero expected logits invalid");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(100);
    g_slha_replace_counters.n_expected_logits.fetch_add(0);
    CHECK(g_slha_replace_counters.is_valid() == false);
    PASS();
}

// ================================================================
// Test 2: complete active run is valid
// ================================================================
void test_complete_active_run_valid() {
    TEST("complete active run valid");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(100);
    g_slha_replace_counters.n_expected_logits.fetch_add(10000);
    g_slha_replace_counters.active_expected_vectors.fetch_add(100);
    g_slha_replace_counters.active_replaced_vectors.fetch_add(100);
    g_slha_replace_counters.active_expected_logits.fetch_add(8000);
    g_slha_replace_counters.active_replaced_logits.fetch_add(8000);
    CHECK(g_slha_replace_counters.is_valid() == true);
    PASS();
}

// ================================================================
// Test 3: failed active vector makes run invalid
// ================================================================
void test_failed_active_vector_invalid() {
    TEST("failed active vector invalidates run");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(100);
    g_slha_replace_counters.n_expected_logits.fetch_add(10000);
    g_slha_replace_counters.active_expected_vectors.fetch_add(100);
    g_slha_replace_counters.active_replaced_vectors.fetch_add(99);
    g_slha_replace_counters.active_expected_logits.fetch_add(8000);
    g_slha_replace_counters.active_replaced_logits.fetch_add(7920);
    g_slha_replace_counters.n_failed_vectors.fetch_add(1);
    CHECK(g_slha_replace_counters.is_valid() == false);
    PASS();
}

// ================================================================
// Test 4: fallback vector makes run invalid
// ================================================================
void test_fallback_vector_invalid() {
    TEST("fallback vector invalidates run");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(100);
    g_slha_replace_counters.n_expected_logits.fetch_add(10000);
    g_slha_replace_counters.active_expected_vectors.fetch_add(95);
    g_slha_replace_counters.active_replaced_vectors.fetch_add(95);
    g_slha_replace_counters.active_expected_logits.fetch_add(7600);
    g_slha_replace_counters.active_replaced_logits.fetch_add(7600);
    g_slha_replace_counters.n_fallback_vectors.fetch_add(5);
    CHECK(g_slha_replace_counters.is_valid() == false);
    PASS();
}

// ================================================================
// Test 5: padding does not inflate active coverage
// ================================================================
void test_padding_does_not_inflate_coverage() {
    TEST("padding does not inflate active coverage");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(10);
    g_slha_replace_counters.n_expected_logits.fetch_add(10 * 512);
    // 10 active vectors, each with n_check=256 out of n_kv=512
    g_slha_replace_counters.active_expected_vectors.fetch_add(10);
    g_slha_replace_counters.active_replaced_vectors.fetch_add(10);
    g_slha_replace_counters.active_expected_logits.fetch_add(10 * 256);
    g_slha_replace_counters.active_replaced_logits.fetch_add(10 * 256);
    // padding: 10 * (512 - 256) = 2560
    g_slha_replace_counters.padding_vectors.fetch_add(10);
    g_slha_replace_counters.padding_logits.fetch_add(10 * 256);

    // Active coverage is 100% (2560/2560), padding is separate
    CHECK(g_slha_replace_counters.is_valid() == true);

    // Without padding separation, run would fail because replaced_logits != expected_logits
    const auto ae_log = g_slha_replace_counters.active_expected_logits.load();
    const auto ar_log = g_slha_replace_counters.active_replaced_logits.load();
    CHECK(ae_log == ar_log);
    const auto pad_log = g_slha_replace_counters.padding_logits.load();
    CHECK(pad_log == 10 * 256);
    PASS();
}

// ================================================================
// Test 6: inactive stream rejected
// ================================================================
void test_inactive_stream_rejected() {
    TEST("inactive stream rejected");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(100);
    g_slha_replace_counters.n_expected_logits.fetch_add(51200);
    // All vectors are active-expected (s=0), but inactive ones are tracked separately
    g_slha_replace_counters.active_expected_vectors.fetch_add(50);
    g_slha_replace_counters.active_expected_logits.fetch_add(50 * 512);
    g_slha_replace_counters.inactive_stream_vectors.fetch_add(50);
    g_slha_replace_counters.inactive_stream_logits.fetch_add(50 * 512);

    // If none were replaced, is_valid should be false
    CHECK(g_slha_replace_counters.is_valid() == false);
    PASS();
}

// ================================================================
// Test 7: active logits and padding logits reported separately
// ================================================================
void test_active_and_padding_separate() {
    TEST("active and padding logits separate");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(20);
    g_slha_replace_counters.n_expected_logits.fetch_add(20 * 512);
    // 20 active vectors
    g_slha_replace_counters.active_expected_vectors.fetch_add(20);
    g_slha_replace_counters.active_replaced_vectors.fetch_add(20);
    // n_check = 384 for each, n_kv = 512, pad = 128 each
    g_slha_replace_counters.active_expected_logits.fetch_add(20 * 384);
    g_slha_replace_counters.active_replaced_logits.fetch_add(20 * 384);
    g_slha_replace_counters.padding_vectors.fetch_add(20);
    g_slha_replace_counters.padding_logits.fetch_add(20 * 128);

    // Verify separation
    CHECK(g_slha_replace_counters.active_expected_logits.load() == 20 * 384);
    CHECK(g_slha_replace_counters.active_replaced_logits.load() == 20 * 384);
    CHECK(g_slha_replace_counters.padding_logits.load() == 20 * 128);

    // Active coverage = 100% despite padding
    CHECK(g_slha_replace_counters.is_valid() == true);
    PASS();
}

// ================================================================
// Test 8: deterministic reset
// ================================================================
void test_deterministic_reset() {
    TEST("deterministic reset");
    for (int run = 0; run < 3; ++run) {
        g_slha_replace_counters.reset();
        g_slha_replace_counters.n_callbacks.fetch_add(2);
        g_slha_replace_counters.n_expected_vectors.fetch_add(2 * 64 * 12 * 8);
        g_slha_replace_counters.n_expected_logits.fetch_add(2 * 64 * 12 * 8 * 512);
        g_slha_replace_counters.active_expected_vectors.fetch_add(2 * 64 * 12 * 8);
        g_slha_replace_counters.active_replaced_vectors.fetch_add(2 * 64 * 12 * 8);
        g_slha_replace_counters.active_expected_logits.fetch_add(2 * 64 * 12 * 8 * 256);
        g_slha_replace_counters.active_replaced_logits.fetch_add(2 * 64 * 12 * 8 * 256);

        CHECK(g_slha_replace_counters.n_callbacks.load() == 2);
        CHECK(g_slha_replace_counters.active_expected_vectors.load() == 2 * 64 * 12 * 8);
        CHECK(g_slha_replace_counters.active_replaced_vectors.load() == 2 * 64 * 12 * 8);
        CHECK(g_slha_replace_counters.is_valid() == true);

        g_slha_replace_counters.reset();
        CHECK(g_slha_replace_counters.n_callbacks.load() == 0);
        CHECK(g_slha_replace_counters.active_expected_vectors.load() == 0);
        CHECK(g_slha_replace_counters.active_replaced_vectors.load() == 0);
        CHECK(g_slha_replace_counters.is_valid() == false);
    }
    PASS();
}

// ================================================================
// Test 9: per-layer counters
// ================================================================
void test_per_layer_counters() {
    TEST("per-layer counters");
    slha_replace_reset_per_layer();
    CHECK(slha_replace_get_layer_success(0) == 0);
    CHECK(slha_replace_get_layer_fail(0) == 0);
    CHECK(slha_replace_get_layer_success(5) == 0);
    CHECK(slha_replace_get_layer_fail(5) == 0);
    CHECK(slha_replace_get_layer_success(200) == 0);

    slha_replace_inc_layer_success(0);
    slha_replace_inc_layer_success(0);
    slha_replace_inc_layer_success(5);
    slha_replace_inc_layer_fail(5);
    slha_replace_inc_layer_fail(200);

    CHECK(slha_replace_get_layer_success(0) == 2);
    CHECK(slha_replace_get_layer_fail(0) == 0);
    CHECK(slha_replace_get_layer_success(5) == 1);
    CHECK(slha_replace_get_layer_fail(5) == 1);
    CHECK(slha_replace_get_layer_success(200) == 0);
    CHECK(slha_replace_get_layer_fail(200) == 0);

    slha_replace_reset_per_layer();
    CHECK(slha_replace_get_layer_success(0) == 0);
    CHECK(slha_replace_get_layer_fail(0) == 0);
    CHECK(slha_replace_get_layer_success(5) == 0);
    CHECK(slha_replace_get_layer_fail(5) == 0);
    PASS();
}

// ================================================================
// Test 10: summary output contains all required counters
// ================================================================
void test_summary_contains_all_counters() {
    TEST("summary contains all required counters");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(2);
    g_slha_replace_counters.n_expected_vectors.fetch_add(500);
    g_slha_replace_counters.n_expected_logits.fetch_add(50000);
    g_slha_replace_counters.active_expected_vectors.fetch_add(400);
    g_slha_replace_counters.active_replaced_vectors.fetch_add(400);
    g_slha_replace_counters.active_expected_logits.fetch_add(32000);
    g_slha_replace_counters.active_replaced_logits.fetch_add(32000);
    g_slha_replace_counters.padding_vectors.fetch_add(100);
    g_slha_replace_counters.padding_logits.fetch_add(12800);
    g_slha_replace_counters.inactive_stream_vectors.fetch_add(100);
    g_slha_replace_counters.inactive_stream_logits.fetch_add(5000);
    slha_replace_inc_layer_success(0);
    slha_replace_inc_layer_success(0);
    slha_replace_inc_layer_success(1);

    std::printf("  --- begin summary capture ---\n");
    g_slha_replace_counters.print_summary();
    std::printf("  --- end summary capture ---\n");
    PASS();
}

// ================================================================
// Test 11: error code invalidates run
// ================================================================
void test_error_code_invalid() {
    TEST("error code invalidates run");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(100);
    g_slha_replace_counters.n_expected_logits.fetch_add(10000);
    g_slha_replace_counters.active_expected_vectors.fetch_add(100);
    g_slha_replace_counters.active_replaced_vectors.fetch_add(100);
    g_slha_replace_counters.active_expected_logits.fetch_add(8000);
    g_slha_replace_counters.active_replaced_logits.fetch_add(8000);
    g_slha_replace_counters.error_code.store(1);
    CHECK(g_slha_replace_counters.is_valid() == false);
    PASS();
}

// ================================================================
// Test 12: active replaced != expected makes run invalid
// ================================================================
void test_active_mismatch_invalid() {
    TEST("active replaced != expected invalid");
    g_slha_replace_counters.reset();
    g_slha_replace_counters.n_callbacks.fetch_add(1);
    g_slha_replace_counters.n_expected_vectors.fetch_add(100);
    g_slha_replace_counters.n_expected_logits.fetch_add(10000);
    g_slha_replace_counters.active_expected_vectors.fetch_add(100);
    g_slha_replace_counters.active_replaced_vectors.fetch_add(99);
    g_slha_replace_counters.active_expected_logits.fetch_add(8000);
    g_slha_replace_counters.active_replaced_logits.fetch_add(7920);
    CHECK(g_slha_replace_counters.is_valid() == false);
    PASS();
}

int main() {
    std::printf("=== replace_strict_counters tests (production-linked) ===\n\n");

    test_empty_run_invalid();
    test_zero_expected_vectors_invalid();
    test_zero_expected_logits_invalid();
    test_complete_active_run_valid();
    test_failed_active_vector_invalid();
    test_fallback_vector_invalid();
    test_padding_does_not_inflate_coverage();
    test_inactive_stream_rejected();
    test_active_and_padding_separate();
    test_deterministic_reset();
    test_per_layer_counters();
    test_summary_contains_all_counters();
    test_error_code_invalid();
    test_active_mismatch_invalid();

    std::printf("\n=== all counter tests complete ===\n");
    return 0;
}

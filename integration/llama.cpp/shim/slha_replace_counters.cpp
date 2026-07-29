#include "slha_replace_counters.hpp"

#include <cstdio>
#include <iostream>

namespace {
    std::atomic<size_t> g_per_layer_success[SLHA_MAX_LAYERS];
    std::atomic<size_t> g_per_layer_fail[SLHA_MAX_LAYERS];
}

slha_replace_counters g_slha_replace_counters;

void slha_replace_reset_per_layer() {
    for (size_t i = 0; i < SLHA_MAX_LAYERS; ++i) {
        g_per_layer_success[i].store(0, std::memory_order_relaxed);
        g_per_layer_fail[i].store(0, std::memory_order_relaxed);
    }
}

void slha_replace_counters::reset() {
    n_callbacks.store(0, std::memory_order_relaxed);
    n_expected_vectors.store(0, std::memory_order_relaxed);
    n_expected_logits.store(0, std::memory_order_relaxed);
    active_expected_vectors.store(0, std::memory_order_relaxed);
    active_replaced_vectors.store(0, std::memory_order_relaxed);
    active_expected_logits.store(0, std::memory_order_relaxed);
    active_replaced_logits.store(0, std::memory_order_relaxed);
    padding_vectors.store(0, std::memory_order_relaxed);
    padding_logits.store(0, std::memory_order_relaxed);
    inactive_stream_vectors.store(0, std::memory_order_relaxed);
    inactive_stream_logits.store(0, std::memory_order_relaxed);
    n_failed_vectors.store(0, std::memory_order_relaxed);
    n_fallback_vectors.store(0, std::memory_order_relaxed);
    n_missing_tile.store(0, std::memory_order_relaxed);
    n_query_prep_fail.store(0, std::memory_order_relaxed);
    n_score_fail.store(0, std::memory_order_relaxed);
    n_nonfinite_score.store(0, std::memory_order_relaxed);
    n_unsupported_shape.store(0, std::memory_order_relaxed);
    n_unsupported_stride.store(0, std::memory_order_relaxed);
    n_stream.store(0, std::memory_order_relaxed);
    error_code.store(0, std::memory_order_relaxed);
    slha_replace_reset_per_layer();
}

bool slha_replace_counters::is_valid() const {
    const auto n_cb = n_callbacks.load(std::memory_order_acquire);
    const auto n_ev = n_expected_vectors.load(std::memory_order_acquire);
    const auto n_el = n_expected_logits.load(std::memory_order_acquire);

    if (n_cb == 0 || n_ev == 0 || n_el == 0) {
        return false;
    }

    return active_replaced_vectors.load(std::memory_order_acquire)
            == active_expected_vectors.load(std::memory_order_acquire)
        && active_replaced_logits.load(std::memory_order_acquire)
            == active_expected_logits.load(std::memory_order_acquire)
        && n_failed_vectors.load(std::memory_order_acquire) == 0
        && n_fallback_vectors.load(std::memory_order_acquire) == 0
        && error_code.load(std::memory_order_acquire) == 0;
}

void slha_replace_counters::print_summary() const {
    const auto ae_vec = active_expected_vectors.load(std::memory_order_acquire);
    const auto ar_vec = active_replaced_vectors.load(std::memory_order_acquire);
    const auto ae_log = active_expected_logits.load(std::memory_order_acquire);
    const auto ar_log = active_replaced_logits.load(std::memory_order_acquire);
    const double active_coverage = ae_vec > 0
        ? static_cast<double>(ar_vec) / static_cast<double>(ae_vec)
        : 0.0;
    const bool valid = is_valid();

    std::cout << "SLHA_REPLACE_SUMMARY\n";
    std::cout << "callbacks="              << n_callbacks.load(std::memory_order_relaxed) << "\n";
    std::cout << "active_expected_vectors=" << ae_vec << "\n";
    std::cout << "active_replaced_vectors=" << ar_vec << "\n";
    std::cout << "active_expected_logits="  << ae_log << "\n";
    std::cout << "active_replaced_logits="  << ar_log << "\n";
    std::cout << "padding_vectors="        << padding_vectors.load(std::memory_order_relaxed) << "\n";
    std::cout << "padding_logits="         << padding_logits.load(std::memory_order_relaxed) << "\n";
    std::cout << "inactive_stream_vectors=" << inactive_stream_vectors.load(std::memory_order_relaxed) << "\n";
    std::cout << "inactive_stream_logits=" << inactive_stream_logits.load(std::memory_order_relaxed) << "\n";
    std::cout << "failed_vectors="         << n_failed_vectors.load(std::memory_order_relaxed) << "\n";
    std::cout << "fallback_vectors="       << n_fallback_vectors.load(std::memory_order_relaxed) << "\n";
    std::cout << "missing_tile="           << n_missing_tile.load(std::memory_order_relaxed) << "\n";
    std::cout << "query_prep_fail="        << n_query_prep_fail.load(std::memory_order_relaxed) << "\n";
    std::cout << "score_fail="             << n_score_fail.load(std::memory_order_relaxed) << "\n";
    std::cout << "nonfinite_score="        << n_nonfinite_score.load(std::memory_order_relaxed) << "\n";
    std::cout << "unsupported_shape="      << n_unsupported_shape.load(std::memory_order_relaxed) << "\n";
    std::cout << "unsupported_stride="     << n_unsupported_stride.load(std::memory_order_relaxed) << "\n";
    std::cout << "error_code="             << error_code.load(std::memory_order_relaxed) << "\n";
    std::cout << "n_stream="               << n_stream.load(std::memory_order_relaxed) << "\n";
    std::cout << "active_coverage="        << active_coverage << "\n";
    std::cout << "valid="                  << (valid ? "true" : "false") << "\n";

    for (size_t i = 0; i < SLHA_MAX_LAYERS; ++i) {
        const size_t s = slha_replace_get_layer_success(static_cast<int32_t>(i));
        const size_t f = slha_replace_get_layer_fail(static_cast<int32_t>(i));
        if (s > 0 || f > 0) {
            std::cout << "layer_" << i << "_success=" << s
                      << " layer_" << i << "_fail=" << f << "\n";
        }
    }
}

void slha_replace_inc_layer_success(int32_t layer_id) {
    if (layer_id >= 0 && layer_id < SLHA_MAX_LAYERS) {
        g_per_layer_success[layer_id].fetch_add(1, std::memory_order_relaxed);
    }
}

void slha_replace_inc_layer_fail(int32_t layer_id) {
    if (layer_id >= 0 && layer_id < SLHA_MAX_LAYERS) {
        g_per_layer_fail[layer_id].fetch_add(1, std::memory_order_relaxed);
    }
}

size_t slha_replace_get_layer_success(int32_t layer_id) {
    if (layer_id >= 0 && layer_id < SLHA_MAX_LAYERS) {
        return g_per_layer_success[layer_id].load(std::memory_order_relaxed);
    }
    return 0;
}

size_t slha_replace_get_layer_fail(int32_t layer_id) {
    if (layer_id >= 0 && layer_id < SLHA_MAX_LAYERS) {
        return g_per_layer_fail[layer_id].load(std::memory_order_relaxed);
    }
    return 0;
}

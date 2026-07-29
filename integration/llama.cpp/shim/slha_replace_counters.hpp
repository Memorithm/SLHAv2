#ifndef SLHA_REPLACE_COUNTERS_HPP
#define SLHA_REPLACE_COUNTERS_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>

#define SLHA_MAX_LAYERS 128

struct slha_replace_counters {
    std::atomic<size_t> n_callbacks{0};
    std::atomic<size_t> n_expected_vectors{0};
    std::atomic<size_t> n_expected_logits{0};

    std::atomic<size_t> active_expected_vectors{0};
    std::atomic<size_t> active_replaced_vectors{0};
    std::atomic<size_t> active_expected_logits{0};
    std::atomic<size_t> active_replaced_logits{0};

    std::atomic<size_t> padding_vectors{0};
    std::atomic<size_t> padding_logits{0};

    std::atomic<size_t> inactive_stream_vectors{0};
    std::atomic<size_t> inactive_stream_logits{0};

    std::atomic<size_t> n_failed_vectors{0};
    std::atomic<size_t> n_fallback_vectors{0};
    std::atomic<size_t> n_missing_tile{0};
    std::atomic<size_t> n_query_prep_fail{0};
    std::atomic<size_t> n_score_fail{0};
    std::atomic<size_t> n_nonfinite_score{0};
    std::atomic<size_t> n_unsupported_shape{0};
    std::atomic<size_t> n_unsupported_stride{0};

    std::atomic<size_t> n_stream{0};

    std::atomic<int> error_code{0};

    void reset();
    bool is_valid() const;
    void print_summary() const;
};

extern slha_replace_counters g_slha_replace_counters;

void slha_replace_reset_per_layer();
void slha_replace_inc_layer_success(int32_t layer_id);
void slha_replace_inc_layer_fail(int32_t layer_id);
size_t slha_replace_get_layer_success(int32_t layer_id);
size_t slha_replace_get_layer_fail(int32_t layer_id);

#endif

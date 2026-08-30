#include "slha_external_k.hpp"

#include "slha_llama.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>

namespace {

using steady_clock = std::chrono::steady_clock;

bool env_is(const char * name, const char * value) {
    const char * actual = std::getenv(name);
    return actual && std::strcmp(actual, value) == 0;
}

bool env_is_unset_or(const char * name, const char * value) {
    const char * actual = std::getenv(name);
    return !actual || actual[0] == '\0' || std::strcmp(actual, value) == 0;
}

bool env_has_value(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0';
}

bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "off") == 0) {
        return false;
    }
    return true;
}

bool env_flag_is_valid(const char * name) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return true;
    }
    return std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
           std::strcmp(value, "off") == 0 || std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 || std::strcmp(value, "on") == 0;
}

bool checked_mul(size_t lhs, size_t rhs, size_t * out) {
    if (!out) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool checked_add(size_t lhs, size_t rhs, size_t * out) {
    if (!out || rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool parse_positive_size_env(const char * name, size_t * out, std::string * error) {
    const char * text = std::getenv(name);
    if (!text || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || !end || *end != '\0' || value == 0 ||
        value > std::numeric_limits<size_t>::max()) {
        if (error) {
            *error = std::string(name) + " must be a positive base-10 integer";
        }
        return false;
    }
    if (out) {
        *out = static_cast<size_t>(value);
    }
    return true;
}

bool parse_positive_float_env(const char * name, float * out, std::string * error) {
    const char * text = std::getenv(name);
    if (!text || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const float value = std::strtof(text, &end);
    if (errno != 0 || end == text || !end || *end != '\0' || !std::isfinite(value) || value <= 0.0f) {
        if (error) {
            *error = std::string(name) + " must be finite and strictly positive";
        }
        return false;
    }
    if (out) {
        *out = value;
    }
    return true;
}

std::mutex g_ccos_mutex;
SlhaElasticKvCache * g_ccos_cache = nullptr;
size_t g_ccos_n_layers = 0;
size_t g_ccos_capacity = 0;
size_t g_ccos_budget_bytes = 0;
float g_ccos_importance_temperature = 1.0f;

std::atomic<size_t> g_peak_resident_bytes{0};
std::atomic<size_t> g_peak_offloaded_bytes{0};
std::atomic<size_t> g_peak_hot_slots{0};
std::atomic<size_t> g_peak_warm_slots{0};
std::atomic<size_t> g_peak_cold_slots{0};
std::atomic<uint64_t> g_write_calls{0};
std::atomic<uint64_t> g_score_calls{0};
std::atomic<uint64_t> g_score_tiles{0};
std::atomic<uint64_t> g_observe_calls{0};
std::atomic<uint64_t> g_budget_enforcements{0};
std::atomic<uint64_t> g_budget_failures{0};
std::atomic<uint64_t> g_cache_hits{0};
std::atomic<uint64_t> g_cache_misses{0};
std::atomic<uint64_t> g_compression_ns{0};
std::atomic<uint64_t> g_score_ns{0};
std::atomic<uint64_t> g_budget_ns{0};

void atomic_max(std::atomic<size_t> & target, size_t value) {
    size_t current = target.load(std::memory_order_relaxed);
    while (value > current &&
           !target.compare_exchange_weak(
               current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

void reset_runtime_counters() {
    g_peak_resident_bytes.store(0, std::memory_order_relaxed);
    g_peak_offloaded_bytes.store(0, std::memory_order_relaxed);
    g_peak_hot_slots.store(0, std::memory_order_relaxed);
    g_peak_warm_slots.store(0, std::memory_order_relaxed);
    g_peak_cold_slots.store(0, std::memory_order_relaxed);
    g_write_calls.store(0, std::memory_order_relaxed);
    g_score_calls.store(0, std::memory_order_relaxed);
    g_score_tiles.store(0, std::memory_order_relaxed);
    g_observe_calls.store(0, std::memory_order_relaxed);
    g_budget_enforcements.store(0, std::memory_order_relaxed);
    g_budget_failures.store(0, std::memory_order_relaxed);
    g_cache_hits.store(0, std::memory_order_relaxed);
    g_cache_misses.store(0, std::memory_order_relaxed);
    g_compression_ns.store(0, std::memory_order_relaxed);
    g_score_ns.store(0, std::memory_order_relaxed);
    g_budget_ns.store(0, std::memory_order_relaxed);
}

void record_cache_snapshot(const SlhaElasticKvCacheStats & stats) {
    atomic_max(g_peak_resident_bytes, stats.resident_bytes);
    atomic_max(g_peak_offloaded_bytes, stats.offloaded_bytes);
    atomic_max(g_peak_hot_slots, stats.hot_slots);
    atomic_max(g_peak_warm_slots, stats.warm_slots);
    atomic_max(g_peak_cold_slots, stats.cold_slots);
}

bool ccos_geometry_slot(int32_t layer_id, size_t position, size_t * slot) {
    if (!slot || layer_id < 0 || static_cast<size_t>(layer_id) >= g_ccos_n_layers ||
        position >= g_ccos_capacity) {
        return false;
    }
    size_t base = 0;
    return checked_mul(position, g_ccos_n_layers, &base) &&
           checked_add(base, static_cast<size_t>(layer_id), slot);
}

bool ccos_range_start(int32_t layer_id, size_t start_position, size_t count, size_t * slot) {
    if (count > g_ccos_capacity || start_position > g_ccos_capacity - count) {
        return false;
    }
    return ccos_geometry_slot(layer_id, start_position, slot);
}

bool snapshot_ccos(SlhaElasticKvCacheStats * out) {
    if (!out || !g_ccos_cache) {
        return false;
    }
    return slha_elastic_cache_stats(g_ccos_cache, out) == SLHA_OK;
}

bool prepare_ccos_geometry(size_t n_layers, size_t runtime_capacity, size_t tile_bytes) {
    std::lock_guard<std::mutex> store_lock(g_slha_tile_store.mutex);
    g_slha_tile_store.n_layers = n_layers;
    g_slha_tile_store.capacity = runtime_capacity;
    g_slha_tile_store.tile_bytes = tile_bytes;
    g_slha_tile_store.tile_base_offset = 0;
    g_slha_tile_store.tiles.clear();
    g_slha_tile_store.valid.clear();
    return true;
}

} // namespace

bool slha_external_k_enabled() {
    return env_flag_enabled("SLHA_EXTERNAL_K");
}

bool slha_external_k_ccos_enabled() {
    return slha_external_k_enabled() && env_flag_enabled("SLHA_CCOS");
}

bool slha_external_k_validate_environment(std::string * error) {
    auto fail = [&](const std::string & message) {
        if (error) {
            *error = message;
        }
        return false;
    };

    if (!slha_external_k_enabled()) {
        return true;
    }

    if (!(env_is("SLHA_EXTERNAL_K", "1") || env_is("SLHA_EXTERNAL_K", "true") ||
          env_is("SLHA_EXTERNAL_K", "on"))) {
        return fail("SLHA_EXTERNAL_K must be one of 1,true,on when enabled");
    }
    if (!env_is("SLHA_KV_MODE", "tilestore")) {
        return fail("external K requires SLHA_KV_MODE=tilestore");
    }
    if (!env_is("SLHA_SCORE_MODE", "replace")) {
        return fail("external K requires SLHA_SCORE_MODE=replace");
    }
    if (!env_is_unset_or("SLHA_SCORE_LAYERS", "all")) {
        return fail("external K requires SLHA_SCORE_LAYERS=all; partial replacement needs baseline K");
    }
    if (env_has_value("SLHA_SCORE_ORACLE")) {
        return fail("external K forbids SLHA_SCORE_ORACLE because the oracle consumes paired baseline logits");
    }
    if (env_has_value("SLHA_ORACLE_METRICS_JSON")) {
        return fail("external K forbids SLHA_ORACLE_METRICS_JSON because paired baseline logits are absent");
    }
    if (env_has_value("SLHA_SCALE_FIT_JSON")) {
        return fail("external K forbids SLHA_SCALE_FIT_JSON because fitting consumes paired baseline logits");
    }
    if (env_has_value("SLHA_RANK_DATASET_DIR")) {
        return fail("external K forbids SLHA_RANK_DATASET_DIR because ranking labels require baseline logits");
    }

    if (!env_flag_is_valid("SLHA_CCOS")) {
        return fail("SLHA_CCOS must be one of 0,false,off,1,true,on");
    }
    if (!slha_external_k_ccos_enabled()) {
        if (env_has_value("SLHA_CCOS_BUDGET_BYTES") ||
            env_has_value("SLHA_CCOS_IMPORTANCE_TEMPERATURE")) {
            return fail("CCOS budget/temperature knobs require SLHA_CCOS=1");
        }
        return true;
    }

    if (env_has_value("SLHA_CCOS_BUDGET_BYTES")) {
        size_t ignored = 0;
        std::string parse_error;
        if (!parse_positive_size_env("SLHA_CCOS_BUDGET_BYTES", &ignored, &parse_error)) {
            return fail(parse_error);
        }
    }
    if (env_has_value("SLHA_CCOS_IMPORTANCE_TEMPERATURE")) {
        float ignored = 0.0f;
        std::string parse_error;
        if (!parse_positive_float_env(
                "SLHA_CCOS_IMPORTANCE_TEMPERATURE", &ignored, &parse_error)) {
            return fail(parse_error);
        }
    }

    return true;
}

bool slha_external_k_prepare_store(size_t runtime_capacity) {
    if (!slha_external_k_enabled() || runtime_capacity == 0) {
        return false;
    }

    const int n_layers_signed = slha_get_num_layers();
    const size_t tile_bytes = slha_tile_size();
    if (n_layers_signed <= 0 || tile_bytes != 128u) {
        return false;
    }
    const size_t n_layers = static_cast<size_t>(n_layers_signed);

    reset_runtime_counters();

    if (!slha_external_k_ccos_enabled()) {
        std::lock_guard<std::mutex> lock(g_ccos_mutex);
        if (g_ccos_cache) {
            (void) slha_elastic_cache_free(g_ccos_cache);
            g_ccos_cache = nullptr;
        }
        g_ccos_n_layers = 0;
        g_ccos_capacity = 0;
        g_ccos_budget_bytes = 0;
        return g_slha_tile_store.init(n_layers, runtime_capacity, tile_bytes);
    }

    size_t logical_slots = 0;
    size_t logical_hot_bytes = 0;
    if (!checked_mul(n_layers, runtime_capacity, &logical_slots) ||
        !checked_mul(logical_slots, slha_elastic_hot_resident_bytes(), &logical_hot_bytes) ||
        logical_hot_bytes == 0) {
        return false;
    }

    size_t budget = logical_hot_bytes;
    if (env_has_value("SLHA_CCOS_BUDGET_BYTES")) {
        std::string parse_error;
        if (!parse_positive_size_env("SLHA_CCOS_BUDGET_BYTES", &budget, &parse_error)) {
            std::cerr << "[SLHA] " << parse_error << "\n";
            return false;
        }
    }

    float temperature = 1.0f;
    if (env_has_value("SLHA_CCOS_IMPORTANCE_TEMPERATURE")) {
        std::string parse_error;
        if (!parse_positive_float_env(
                "SLHA_CCOS_IMPORTANCE_TEMPERATURE", &temperature, &parse_error)) {
            std::cerr << "[SLHA] " << parse_error << "\n";
            return false;
        }
    }

    std::lock_guard<std::mutex> lock(g_ccos_mutex);
    if (g_ccos_cache) {
        if (slha_elastic_cache_free(g_ccos_cache) != SLHA_OK) {
            return false;
        }
        g_ccos_cache = nullptr;
    }
    g_ccos_cache = slha_elastic_cache_new(budget);
    if (!g_ccos_cache) {
        return false;
    }
    g_ccos_n_layers = n_layers;
    g_ccos_capacity = runtime_capacity;
    g_ccos_budget_bytes = budget;
    g_ccos_importance_temperature = temperature;

    if (!prepare_ccos_geometry(n_layers, runtime_capacity, tile_bytes)) {
        (void) slha_elastic_cache_free(g_ccos_cache);
        g_ccos_cache = nullptr;
        return false;
    }
    return true;
}

bool slha_external_k_write_tile(
    int32_t layer_id,
    size_t position,
    const SciRustSlhaTile * tile
) {
    g_write_calls.fetch_add(1, std::memory_order_relaxed);

    if (!slha_external_k_ccos_enabled()) {
        const bool ok = g_slha_tile_store.write(layer_id, position, tile);
        if (ok) {
            g_cache_hits.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_cache_misses.fetch_add(1, std::memory_order_relaxed);
        }
        return ok;
    }

    if (!tile || !g_ccos_cache) {
        g_cache_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    size_t slot = 0;
    if (!ccos_geometry_slot(layer_id, position, &slot)) {
        g_cache_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    SlhaElasticKvCacheStats before{};
    if (!snapshot_ccos(&before)) {
        g_cache_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    record_cache_snapshot(before);

    size_t active_after = before.hot_slots + before.warm_slots + before.cold_slots + before.pinned_slots;
    if (slha_elastic_cache_tier(g_ccos_cache, slot) == SLHA_ELASTIC_TIER_ABSENT) {
        if (active_after == std::numeric_limits<size_t>::max()) {
            g_budget_failures.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        ++active_after;
    }

    size_t minimum_dense_resident = 0;
    if (!checked_mul(active_after, slha_elastic_warm_resident_bytes(), &minimum_dense_resident) ||
        g_ccos_budget_bytes < minimum_dense_resident) {
        // Dense attention needs every active K on every token. Creating COLD
        // active keys here would immediately make the next score unavailable,
        // so reject the write before mutating cache state.
        g_budget_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const int32_t prior_tier = slha_elastic_cache_tier(g_ccos_cache, slot);
    size_t prior_slot_resident = 0;
    if (prior_tier == SLHA_ELASTIC_TIER_HOT || prior_tier == SLHA_ELASTIC_TIER_PINNED) {
        prior_slot_resident = slha_elastic_hot_resident_bytes();
    } else if (prior_tier == SLHA_ELASTIC_TIER_WARM) {
        prior_slot_resident = slha_elastic_warm_resident_bytes();
    }
    if (before.resident_bytes < prior_slot_resident) {
        g_budget_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    size_t hot_after_rewrite = 0;
    const size_t resident_without_slot = before.resident_bytes - prior_slot_resident;
    if (!checked_add(
            resident_without_slot, slha_elastic_hot_resident_bytes(), &hot_after_rewrite)) {
        g_budget_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const bool budget_action = hot_after_rewrite > g_ccos_budget_bytes;
    if (budget_action) {
        g_budget_enforcements.fetch_add(1, std::memory_order_relaxed);
    }

    const auto budget_start = steady_clock::now();
    const int32_t write_rc = slha_elastic_cache_write_dense_budget(
        g_ccos_cache, slot, tile, g_ccos_budget_bytes);
    const auto budget_end = steady_clock::now();
    if (budget_action) {
        // This is wall time for the complete budget-aware admission call. It
        // includes the direct HOT/WARM write because admission and mutation are
        // intentionally one atomic operation; it is not presented as a pure
        // codec-only timing.
        g_budget_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                budget_end - budget_start).count()),
            std::memory_order_relaxed);
    }
    if (write_rc != SLHA_OK) {
        g_budget_failures.fetch_add(1, std::memory_order_relaxed);
        g_cache_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    SlhaElasticKvCacheStats final_stats{};
    if (!snapshot_ccos(&final_stats)) {
        g_cache_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    record_cache_snapshot(final_stats);
    if (final_stats.cold_slots != 0 || final_stats.resident_bytes > g_ccos_budget_bytes) {
        g_budget_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    g_cache_hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

int32_t slha_external_k_score_tiles(
    SlhaModel * model,
    int32_t layer_id,
    size_t start_position,
    size_t count,
    const float * q_coarse,
    const uint64_t * q_sign,
    float * scores_out
) {
    g_score_calls.fetch_add(1, std::memory_order_relaxed);
    g_score_tiles.fetch_add(static_cast<uint64_t>(count), std::memory_order_relaxed);

    const auto score_start = steady_clock::now();
    int32_t rc = SLHA_OK;

    if (!slha_external_k_ccos_enabled()) {
        const SciRustSlhaTile * tiles = count > 0
            ? static_cast<const SciRustSlhaTile *>(
                g_slha_tile_store.read_range(layer_id, start_position, count))
            : nullptr;
        if (count > 0 && !tiles) {
            rc = SLHA_ERR_NOT_RESIDENT;
        } else {
            rc = slha_score_tiles(model, tiles, count, q_coarse, q_sign, scores_out);
        }
    } else {
        if (!g_ccos_cache) {
            rc = SLHA_ERR_INVALID_HANDLE;
        } else {
            size_t start_slot = 0;
            if (!ccos_range_start(layer_id, start_position, count, &start_slot)) {
                rc = SLHA_ERR_DIMENSION;
            } else {
                rc = slha_elastic_cache_score_strided(
                    g_ccos_cache,
                    start_slot,
                    g_ccos_n_layers,
                    count,
                    q_coarse,
                    q_sign,
                    scores_out);
                if (rc == SLHA_OK && count > 0) {
                    g_observe_calls.fetch_add(1, std::memory_order_relaxed);
                    const int32_t observe_rc = slha_elastic_cache_observe_scores_strided(
                        g_ccos_cache,
                        start_slot,
                        g_ccos_n_layers,
                        scores_out,
                        count,
                        g_ccos_importance_temperature);
                    if (observe_rc != SLHA_OK) {
                        rc = observe_rc;
                    }
                }
            }
        }
    }

    const auto score_end = steady_clock::now();
    g_score_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            score_end - score_start).count()),
        std::memory_order_relaxed);

    if (rc == SLHA_OK) {
        g_cache_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_cache_misses.fetch_add(1, std::memory_order_relaxed);
    }
    return rc;
}

void slha_external_k_reset_store() {
    if (!slha_external_k_ccos_enabled()) {
        if (g_slha_tile_store.n_layers > 0) {
            g_slha_tile_store.reset();
        }
        return;
    }
    if (g_ccos_cache) {
        (void) slha_elastic_cache_clear(g_ccos_cache);
    }
}

void slha_external_k_release_store() {
    std::lock_guard<std::mutex> lock(g_ccos_mutex);
    if (g_ccos_cache) {
        (void) slha_elastic_cache_free(g_ccos_cache);
        g_ccos_cache = nullptr;
    }
    g_ccos_n_layers = 0;
    g_ccos_capacity = 0;
    g_ccos_budget_bytes = 0;
}

void slha_external_k_record_compression_ns(uint64_t elapsed_ns) {
    g_compression_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
}

bool slha_external_k_store_stats_snapshot(slha_external_k_store_stats * out) {
    if (!out) {
        return false;
    }

    slha_external_k_store_stats stats;
    stats.ccos_enabled = slha_external_k_ccos_enabled();

    if (stats.ccos_enabled) {
        stats.n_layers = g_ccos_n_layers;
        stats.capacity = g_ccos_capacity;
        stats.tile_bytes = slha_tile_size();
        size_t logical_slots = 0;
        if (!checked_mul(stats.n_layers, stats.capacity, &logical_slots) ||
            !checked_mul(logical_slots, stats.tile_bytes, &stats.logical_tile_bytes)) {
            return false;
        }
        SlhaElasticKvCacheStats cache_stats{};
        if (!snapshot_ccos(&cache_stats)) {
            return false;
        }
        stats.resident_bytes = cache_stats.resident_bytes;
        stats.offloaded_bytes = cache_stats.offloaded_bytes;
        stats.hard_budget_bytes = cache_stats.hard_budget_bytes;
        stats.hot_slots = cache_stats.hot_slots;
        stats.warm_slots = cache_stats.warm_slots;
        stats.cold_slots = cache_stats.cold_slots;
        stats.pinned_slots = cache_stats.pinned_slots;
        stats.evictions = cache_stats.evictions;
        record_cache_snapshot(cache_stats);
    } else {
        std::lock_guard<std::mutex> lock(g_slha_tile_store.mutex);
        stats.n_layers = g_slha_tile_store.n_layers;
        stats.capacity = g_slha_tile_store.capacity;
        stats.tile_bytes = g_slha_tile_store.tile_bytes;
        size_t slots = 0;
        if (!checked_mul(stats.n_layers, stats.capacity, &slots) ||
            !checked_mul(slots, stats.tile_bytes, &stats.logical_tile_bytes)) {
            return false;
        }
        stats.tile_backing_capacity_bytes = g_slha_tile_store.tiles.capacity();
        stats.validity_backing_capacity_bytes = g_slha_tile_store.valid.capacity();
    }

    stats.peak_resident_bytes = g_peak_resident_bytes.load(std::memory_order_relaxed);
    stats.peak_offloaded_bytes = g_peak_offloaded_bytes.load(std::memory_order_relaxed);
    stats.peak_hot_slots = g_peak_hot_slots.load(std::memory_order_relaxed);
    stats.peak_warm_slots = g_peak_warm_slots.load(std::memory_order_relaxed);
    stats.peak_cold_slots = g_peak_cold_slots.load(std::memory_order_relaxed);
    stats.write_calls = g_write_calls.load(std::memory_order_relaxed);
    stats.score_calls = g_score_calls.load(std::memory_order_relaxed);
    stats.score_tiles = g_score_tiles.load(std::memory_order_relaxed);
    stats.observe_calls = g_observe_calls.load(std::memory_order_relaxed);
    stats.budget_enforcements = g_budget_enforcements.load(std::memory_order_relaxed);
    stats.budget_failures = g_budget_failures.load(std::memory_order_relaxed);
    stats.cache_hits = g_cache_hits.load(std::memory_order_relaxed);
    stats.cache_misses = g_cache_misses.load(std::memory_order_relaxed);
    stats.compression_ns = g_compression_ns.load(std::memory_order_relaxed);
    stats.score_ns = g_score_ns.load(std::memory_order_relaxed);
    stats.budget_ns = g_budget_ns.load(std::memory_order_relaxed);

    *out = stats;
    return true;
}

void slha_external_k_print_store_summary() {
    slha_external_k_store_stats stats;
    if (!slha_external_k_store_stats_snapshot(&stats)) {
        std::cerr << "SLHA_EXTERNAL_K_STORE valid=false\n";
        return;
    }

    std::cerr << "SLHA_EXTERNAL_K_STORE"
              << " valid=true"
              << " backend=" << (stats.ccos_enabled ? "ccos_elastic" : "vector")
              << " layers=" << stats.n_layers
              << " capacity=" << stats.capacity
              << " tile_bytes=" << stats.tile_bytes
              << " logical_tile_bytes=" << stats.logical_tile_bytes
              << " tile_backing_capacity_bytes=" << stats.tile_backing_capacity_bytes
              << " validity_backing_capacity_bytes=" << stats.validity_backing_capacity_bytes
              << " resident_bytes=" << stats.resident_bytes
              << " offloaded_bytes=" << stats.offloaded_bytes
              << " hard_budget_bytes=" << stats.hard_budget_bytes
              << " hot_slots=" << stats.hot_slots
              << " warm_slots=" << stats.warm_slots
              << " cold_slots=" << stats.cold_slots
              << " pinned_slots=" << stats.pinned_slots
              << " evictions=" << stats.evictions
              << " peak_resident_bytes=" << stats.peak_resident_bytes
              << " peak_offloaded_bytes=" << stats.peak_offloaded_bytes
              << " peak_hot_slots=" << stats.peak_hot_slots
              << " peak_warm_slots=" << stats.peak_warm_slots
              << " peak_cold_slots=" << stats.peak_cold_slots
              << " write_calls=" << stats.write_calls
              << " score_calls=" << stats.score_calls
              << " score_tiles=" << stats.score_tiles
              << " observe_calls=" << stats.observe_calls
              << " budget_enforcements=" << stats.budget_enforcements
              << " budget_failures=" << stats.budget_failures
              << " cache_hits=" << stats.cache_hits
              << " cache_misses=" << stats.cache_misses
              << " compression_ns=" << stats.compression_ns
              << " score_ns=" << stats.score_ns
              << " budget_ns=" << stats.budget_ns
              << "\n";
}

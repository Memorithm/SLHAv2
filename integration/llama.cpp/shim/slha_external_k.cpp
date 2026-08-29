#include "slha_external_k.hpp"

#include "slha.h"
#include "slha_llama.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

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

} // namespace

bool slha_external_k_enabled() {
    const char * value = std::getenv("SLHA_EXTERNAL_K");
    if (!value || value[0] == '\0') {
        return false;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "off") == 0) {
        return false;
    }
    return true;
}

bool slha_external_k_validate_environment(std::string * error) {
    auto fail = [&](const char * message) {
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

    return true;
}

bool slha_external_k_prepare_store(size_t runtime_capacity) {
    if (!slha_external_k_enabled() || runtime_capacity == 0) {
        return false;
    }

    const int n_layers = slha_get_num_layers();
    const size_t tile_bytes = slha_tile_size();
    if (n_layers <= 0 || tile_bytes != 128u) {
        return false;
    }

    if (g_slha_tile_store.capacity == runtime_capacity &&
        g_slha_tile_store.n_layers == static_cast<size_t>(n_layers) &&
        g_slha_tile_store.tile_bytes == tile_bytes) {
        return true;
    }

    return g_slha_tile_store.init(
        static_cast<size_t>(n_layers), runtime_capacity, tile_bytes);
}

bool slha_external_k_store_stats_snapshot(slha_external_k_store_stats * out) {
    if (!out) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_slha_tile_store.mutex);

    slha_external_k_store_stats stats;
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
              << " layers=" << stats.n_layers
              << " capacity=" << stats.capacity
              << " tile_bytes=" << stats.tile_bytes
              << " logical_tile_bytes=" << stats.logical_tile_bytes
              << " tile_backing_capacity_bytes=" << stats.tile_backing_capacity_bytes
              << " validity_backing_capacity_bytes=" << stats.validity_backing_capacity_bytes
              << "\n";
}

#include "slha_external_k.hpp"
#include "slha_llama.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

slha_tile_store g_slha_tile_store;

int slha_get_num_layers() {
    return 3;
}

bool slha_tile_store::init(size_t n_layers_, size_t capacity_, size_t tile_bytes_) {
    n_layers = n_layers_;
    capacity = capacity_;
    tile_bytes = tile_bytes_;
    tiles.assign(n_layers * capacity * tile_bytes + TILE_ALIGN - 1, 0u);
    valid.assign(n_layers * capacity, 0u);
    return n_layers > 0 && capacity > 0 && tile_bytes == 128u;
}

void slha_tile_store::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    std::fill(tiles.begin(), tiles.end(), 0u);
    std::fill(valid.begin(), valid.end(), 0u);
}

bool slha_tile_store::write(int32_t layer_id, size_t position, const void * tile) {
    if (!tile || layer_id < 0 || static_cast<size_t>(layer_id) >= n_layers || position >= capacity) {
        return false;
    }
    const size_t index = static_cast<size_t>(layer_id) * capacity + position;
    const size_t offset = index * tile_bytes;
    if (offset + tile_bytes > tiles.size() || index >= valid.size()) {
        return false;
    }
    std::memcpy(tiles.data() + offset, tile, tile_bytes);
    valid[index] = 1u;
    return true;
}

const void * slha_tile_store::read_range(int32_t layer_id, size_t start_position, size_t count) const {
    static thread_local std::vector<uint8_t> snapshot;
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= n_layers || count == 0 ||
        start_position > capacity || count > capacity - start_position) {
        return nullptr;
    }
    const size_t layer = static_cast<size_t>(layer_id);
    for (size_t i = 0; i < count; ++i) {
        if (valid[layer * capacity + start_position + i] == 0u) {
            return nullptr;
        }
    }
    snapshot.resize(count * tile_bytes);
    for (size_t i = 0; i < count; ++i) {
        const size_t index = layer * capacity + start_position + i;
        std::memcpy(snapshot.data() + i * tile_bytes, tiles.data() + index * tile_bytes, tile_bytes);
    }
    return snapshot.data();
}

namespace {

void unset_contract_env() {
    unsetenv("SLHA_EXTERNAL_K");
    unsetenv("SLHA_KV_MODE");
    unsetenv("SLHA_SCORE_MODE");
    unsetenv("SLHA_SCORE_LAYERS");
    unsetenv("SLHA_SCORE_ORACLE");
    unsetenv("SLHA_ORACLE_METRICS_JSON");
    unsetenv("SLHA_SCALE_FIT_JSON");
    unsetenv("SLHA_RANK_DATASET_DIR");
    unsetenv("SLHA_CCOS");
    unsetenv("SLHA_CCOS_BUDGET_BYTES");
    unsetenv("SLHA_CCOS_IMPORTANCE_TEMPERATURE");
}

void set_valid_external_env() {
    unset_contract_env();
    setenv("SLHA_EXTERNAL_K", "1", 1);
    setenv("SLHA_KV_MODE", "tilestore", 1);
    setenv("SLHA_SCORE_MODE", "replace", 1);
    setenv("SLHA_SCORE_LAYERS", "all", 1);
}

void set_valid_ccos_env(const char * budget = nullptr) {
    set_valid_external_env();
    setenv("SLHA_CCOS", "1", 1);
    if (budget) {
        setenv("SLHA_CCOS_BUDGET_BYTES", budget, 1);
    }
}

void expect_invalid(const char * needle) {
    std::string error;
    assert(slha_external_k_enabled());
    assert(!slha_external_k_validate_environment(&error));
    assert(error.find(needle) != std::string::npos);
}

SciRustSlhaTile make_tile() {
    SciRustSlhaTile tile{};
    std::memset(tile.latent_kv, 0x88, sizeof(tile.latent_kv));
    tile.scale = 1.0f;
    tile.dynamic_lambda = 0.25f;
    for (auto & scale : tile.group_scales) {
        scale = 255u;
    }
    return tile;
}

} // namespace

int main() {
    unset_contract_env();
    assert(!slha_external_k_enabled());
    assert(!slha_external_k_ccos_enabled());
    std::string error;
    assert(slha_external_k_validate_environment(&error));

    // Legacy vector backend remains the default external-K behavior.
    set_valid_external_env();
    error.clear();
    assert(slha_external_k_enabled());
    assert(!slha_external_k_ccos_enabled());
    assert(slha_external_k_validate_environment(&error));
    assert(error.empty());
    assert(slha_external_k_prepare_store(4096));
    assert(g_slha_tile_store.n_layers == 3);
    assert(g_slha_tile_store.capacity == 4096);
    assert(g_slha_tile_store.tile_bytes == 128u);

    slha_external_k_store_stats stats;
    assert(!slha_external_k_store_stats_snapshot(nullptr));
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(!stats.ccos_enabled);
    assert(stats.n_layers == 3);
    assert(stats.capacity == 4096);
    assert(stats.tile_bytes == 128u);
    assert(stats.logical_tile_bytes == 3u * 4096u * 128u);
    assert(stats.tile_backing_capacity_bytes >= stats.logical_tile_bytes);
    assert(stats.validity_backing_capacity_bytes >= 3u * 4096u);

    // CCOS with no explicit pressure owns the physical tiles in Rust and does
    // not allocate the context-sized C++ tile/validity vectors.
    set_valid_ccos_env();
    assert(slha_external_k_ccos_enabled());
    assert(slha_external_k_validate_environment(&error));
    assert(slha_external_k_prepare_store(8));
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.ccos_enabled);
    assert(stats.logical_tile_bytes == 3u * 8u * 128u);
    assert(stats.tile_backing_capacity_bytes == 0u);
    assert(stats.validity_backing_capacity_bytes == 0u);
    assert(stats.hard_budget_bytes == stats.logical_tile_bytes);

    const SciRustSlhaTile tile = make_tile();
    assert(slha_external_k_write_tile(1, 0, &tile));
    float q_coarse[SLHA_D_C] = {};
    uint64_t q_sign[SLHA_RESIDUAL_WORDS] = {};
    float score = -1.0f;
    assert(slha_external_k_score_tiles(nullptr, 1, 0, 1, q_coarse, q_sign, &score) == SLHA_OK);
    assert(std::isfinite(score));
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 1u);
    assert(stats.warm_slots == 0u);
    assert(stats.cold_slots == 0u);
    assert(stats.write_calls == 1u);
    assert(stats.score_calls == 1u);
    assert(stats.observe_calls == 1u);

    // A 96-byte budget forces the first active key HOT -> WARM. A second active
    // key would require at least 192 resident bytes, so it must fail BEFORE the
    // write instead of creating a COLD key that dense attention cannot score.
    set_valid_ccos_env("96");
    assert(slha_external_k_validate_environment(&error));
    assert(slha_external_k_prepare_store(8));
    assert(slha_external_k_write_tile(0, 0, &tile));
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 0u);
    assert(stats.warm_slots == 1u);
    assert(stats.cold_slots == 0u);
    assert(stats.resident_bytes == 96u);
    assert(stats.offloaded_bytes == 32u);
    assert(stats.peak_resident_bytes == 96u);
    assert(stats.peak_offloaded_bytes == 32u);
    assert(stats.budget_enforcements == 1u);
    assert(!slha_external_k_write_tile(1, 0, &tile));
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.warm_slots == 1u);
    assert(stats.cold_slots == 0u);
    assert(stats.budget_failures == 1u);

    // COLD is legal only while the dense-attention context is quiescent.
    // Offload all active keys, verify that they become unscoreable COLD state,
    // then restore the exact HOT representation before scoring resumes.
    set_valid_ccos_env("384");
    assert(slha_external_k_prepare_store(8));
    assert(slha_external_k_write_tile(0, 0, &tile));
    assert(slha_external_k_write_tile(1, 0, &tile));
    assert(slha_external_k_write_tile(2, 0, &tile));
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 3u);
    assert(stats.resident_bytes == 384u);
    assert(slha_external_k_ccos_offload_quiescent());
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 0u);
    assert(stats.warm_slots == 0u);
    assert(stats.cold_slots == 3u);
    assert(stats.resident_bytes == 0u);
    assert(stats.evictions == 3u);
    assert(stats.quiescent_offload_calls == 1u);
    assert(slha_external_k_score_tiles(nullptr, 0, 0, 1, q_coarse, q_sign, &score) == SLHA_ERR_NOT_RESIDENT);
    assert(slha_external_k_ccos_restore_quiescent());
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 3u);
    assert(stats.warm_slots == 0u);
    assert(stats.cold_slots == 0u);
    assert(stats.resident_bytes == 384u);
    assert(stats.quiescent_restore_calls == 1u);
    assert(stats.quiescent_restored_slots == 3u);
    assert(slha_external_k_score_tiles(nullptr, 0, 0, 1, q_coarse, q_sign, &score) == SLHA_OK);

    set_valid_external_env();
    setenv("SLHA_SCORE_MODE", "shadow", 1);
    expect_invalid("replace");

    set_valid_external_env();
    setenv("SLHA_SCORE_LAYERS", "0-3", 1);
    expect_invalid("all");

    set_valid_external_env();
    setenv("SLHA_SCORE_ORACLE", "rank-transplant", 1);
    expect_invalid("ORACLE");

    set_valid_external_env();
    setenv("SLHA_ORACLE_METRICS_JSON", "/tmp/metrics.json", 1);
    expect_invalid("ORACLE_METRICS");

    set_valid_external_env();
    setenv("SLHA_SCALE_FIT_JSON", "/tmp/fit.json", 1);
    expect_invalid("SCALE_FIT");

    set_valid_external_env();
    setenv("SLHA_RANK_DATASET_DIR", "/tmp/rank", 1);
    expect_invalid("RANK_DATASET");

    unset_contract_env();
    setenv("SLHA_EXTERNAL_K", "banana", 1);
    expect_invalid("1,true,on");

    set_valid_external_env();
    setenv("SLHA_CCOS", "banana", 1);
    expect_invalid("SLHA_CCOS");

    set_valid_external_env();
    setenv("SLHA_CCOS_BUDGET_BYTES", "96", 1);
    expect_invalid("require SLHA_CCOS=1");

    set_valid_ccos_env("0");
    expect_invalid("positive base-10 integer");

    set_valid_ccos_env();
    setenv("SLHA_CCOS_IMPORTANCE_TEMPERATURE", "nan", 1);
    expect_invalid("finite and strictly positive");

    slha_external_k_release_store();
    unset_contract_env();
    std::cout << "external_k_contract_tests: ok\n";
    return 0;
}

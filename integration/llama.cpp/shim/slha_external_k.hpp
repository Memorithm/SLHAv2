#ifndef SLHA_EXTERNAL_K_HPP
#define SLHA_EXTERNAL_K_HPP

#include "slha.h"

#include <cstddef>
#include <cstdint>
#include <string>

/**
 * Opt-in physical-K experiment.
 *
 * This flag is intentionally orthogonal to SLHA_KV_MODE so the existing
 * tilestore/scorediag experiments retain their historical semantics. Enable
 * with SLHA_EXTERNAL_K=1 together with SLHA_KV_MODE=tilestore and
 * SLHA_SCORE_MODE=replace.
 */
bool slha_external_k_enabled();

/** True only when the Rust ElasticKvCache backend is explicitly requested. */
bool slha_external_k_ccos_enabled();

/**
 * Validate that the environment is compatible with a baseline-free SLHA score
 * path. Returns false and fills error on any combination that would require
 * the ordinary llama.cpp K tensor or paired baseline Q*K logits.
 */
bool slha_external_k_validate_environment(std::string * error);

/**
 * Prepare physical external-K storage from llama.cpp's real KV capacity.
 *
 * Legacy external-K allocates the existing C++ tile vector. With SLHA_CCOS=1,
 * only geometry metadata remains in C++ and physical tiles are owned by the
 * Rust ElasticKvCache. The default CCOS hard budget is the full HOT logical
 * capacity; an explicit SLHA_CCOS_BUDGET_BYTES may tighten it.
 */
bool slha_external_k_prepare_store(size_t runtime_capacity);

/** Store one encoded tile at the runtime's stable layer/position address. */
bool slha_external_k_write_tile(
    int32_t layer_id,
    size_t position,
    const SciRustSlhaTile * tile
);

/**
 * Score a contiguous position range for one layer.
 *
 * The legacy backend snapshots C++ tiles and calls slha_score_tiles. The CCOS
 * backend uses the strided Rust cache ABI directly so interleaved layer slots
 * never need to be copied into a temporary tile array.
 */
int32_t slha_external_k_score_tiles(
    SlhaModel * model,
    int32_t layer_id,
    size_t start_position,
    size_t count,
    const float * q_coarse,
    const uint64_t * q_sign,
    float * scores_out
);

/** Reset logical live tiles while retaining configured backend state. */
void slha_external_k_reset_store();

/** Release optional CCOS storage at process shutdown. */
void slha_external_k_release_store();

/** Record measured K compression time from the llama.cpp K-transform seam. */
void slha_external_k_record_compression_ns(uint64_t elapsed_ns);

struct slha_external_k_store_stats {
    size_t n_layers = 0;
    size_t capacity = 0;
    size_t tile_bytes = 0;
    size_t logical_tile_bytes = 0;
    size_t tile_backing_capacity_bytes = 0;
    size_t validity_backing_capacity_bytes = 0;

    bool ccos_enabled = false;
    size_t resident_bytes = 0;
    size_t offloaded_bytes = 0;
    size_t hard_budget_bytes = 0;
    size_t hot_slots = 0;
    size_t warm_slots = 0;
    size_t cold_slots = 0;
    size_t pinned_slots = 0;
    uint64_t evictions = 0;

    size_t peak_resident_bytes = 0;
    size_t peak_offloaded_bytes = 0;
    size_t peak_hot_slots = 0;
    size_t peak_warm_slots = 0;
    size_t peak_cold_slots = 0;

    uint64_t write_calls = 0;
    uint64_t score_calls = 0;
    uint64_t score_tiles = 0;
    uint64_t observe_calls = 0;
    uint64_t budget_enforcements = 0;
    uint64_t budget_failures = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t compression_ns = 0;
    uint64_t score_ns = 0;
    uint64_t budget_ns = 0;
};

/** Snapshot owned external-K storage and cumulative runtime counters. */
bool slha_external_k_store_stats_snapshot(slha_external_k_store_stats * out);

/** Emit a machine-parseable one-line snapshot for real-model reports. */
void slha_external_k_print_store_summary();

#endif

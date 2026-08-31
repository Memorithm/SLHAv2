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
 * Validate the llama.cpp runtime geometry supported by physical external-K.
 *
 * The current integration intentionally supports one KV stream and one logical
 * sequence. A unified KV cache can otherwise expose multiple logical sequences
 * through one physical stream; accepting that shape would silently overstate
 * lifecycle support. Keep this policy in the shim so standalone contract tests
 * exercise the same predicate used by the engine patch.
 */
inline bool slha_external_k_validate_runtime(
    size_t n_stream,
    size_t n_seq_max,
    std::string * error
) {
    if (!slha_external_k_enabled()) {
        return true;
    }
    if (n_stream != 1u) {
        if (error) {
            *error = "SLHA external-K currently requires exactly one KV stream";
        }
        return false;
    }
    if (n_seq_max != 1u) {
        if (error) {
            *error = "SLHA external-K currently requires exactly one logical sequence";
        }
        return false;
    }
    if (error) {
        error->clear();
    }
    return true;
}

/**
 * State persistence is unsupported until the physical external-K payload and
 * its liveness/tier metadata are part of llama.cpp's state format.
 *
 * Returning false here is deliberate: serializing only llama.cpp's constant-K
 * sentinel plus V cache would create a state file that appears valid but cannot
 * reconstruct the SLHA attention state.
 */
inline bool slha_external_k_state_serialization_supported() {
    return !slha_external_k_enabled();
}

/**
 * Sparse sequence mutation is unsupported until llama cell liveness is
 * synchronized with the external store and scorer.
 *
 * seq_rm/seq_keep/seq_cp can create holes or new logical references without
 * rewriting K. The current external scorer intentionally assumes a dense live
 * prefix, so these operations must fail closed rather than leave stale or
 * missing physical tiles behind llama.cpp metadata.
 */
inline bool slha_external_k_sparse_sequence_mutation_supported() {
    return !slha_external_k_enabled();
}

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

/**
 * Remove a fully populated dense suffix [new_high_water, old_high_water) from
 * every external-K layer.
 *
 * This is a quiescent lifecycle operation. The complete suffix is preflighted
 * before mutation, so a missing tile rejects the trim instead of intentionally
 * creating a hole. The vector backend invalidates/zeroes its slots; the CCOS
 * backend releases the corresponding fixed slots. The caller is responsible
 * for lowering llama-side high-water metadata only after this returns true.
 */
bool slha_external_k_trim_suffix(
    size_t new_high_water,
    size_t old_high_water
);

/**
 * Move every currently active CCOS key to COLD backing storage.
 *
 * This is a quiescent lifecycle operation: callers must invoke it only after a
 * synchronous llama_decode has returned and before another decode starts. Dense
 * attention cannot score COLD keys. The operation is fail-closed and leaves the
 * cache unchanged on an offload failure.
 */
bool slha_external_k_ccos_offload_quiescent();

/**
 * Restore a cache previously offloaded with
 * slha_external_k_ccos_offload_quiescent(). HOT/WARM representations are
 * restored exactly before the next decode is allowed to run.
 */
bool slha_external_k_ccos_restore_quiescent();

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
    uint64_t quiescent_offload_calls = 0;
    uint64_t quiescent_restore_calls = 0;
    uint64_t quiescent_restored_slots = 0;
    uint64_t quiescent_offload_ns = 0;
    uint64_t quiescent_restore_ns = 0;
    uint64_t compression_ns = 0;
    uint64_t score_ns = 0;
    uint64_t budget_ns = 0;
};

/** Snapshot owned external-K storage and cumulative runtime counters. */
bool slha_external_k_store_stats_snapshot(slha_external_k_store_stats * out);

/** Emit a machine-parseable one-line snapshot for real-model reports. */
void slha_external_k_print_store_summary();

#endif
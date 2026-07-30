#ifndef SLHA_LLAMA_HPP
#define SLHA_LLAMA_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <string>

#include "slha_replace_counters.hpp"

struct ggml_context;
struct ggml_tensor;

enum slha_kv_mode : int {
    SLHA_KV_OFF = 0,
    SLHA_KV_PASSTHROUGH = 1,
    SLHA_KV_ROUNDTRIP = 2,
    SLHA_KV_COLLECT = 3,
    SLHA_KV_TILESTORE = 4,
};

enum slha_score_mode : int {
    SLHA_SCORE_OFF = 0,
    SLHA_SCORE_SHADOW = 1,
    SLHA_SCORE_REPLACE = 2,
};

struct slha_layer_state;

/** Experimental CPU-side compressed K tile store.
 *
 * One tile per (layer, cache position). Capacity is fixed per layer and
 * position indices are bounds-checked. No unbounded growth. */
struct slha_tile_store {
    size_t n_layers = 0;
    size_t capacity = 0;
    size_t tile_bytes = 0;
    std::vector<std::byte> tiles;
    std::vector<uint8_t> valid;
    mutable std::mutex mutex;

    bool init(size_t n_layers, size_t capacity, size_t tile_bytes);
    void reset();
    void clear_layer(int32_t layer_id);
    bool write(int32_t layer_id, size_t position, const void * tile);
    const void * read(int32_t layer_id, size_t position) const;
    bool check_capacity(int32_t layer_id, size_t position) const;
};

/** Shadow-mode metrics accumulated per layer and per head. */
struct slha_shadow_metrics {
    mutable std::mutex mutex;
    size_t n_samples = 0;
    size_t n_vectors = 0;
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    double sum_rel_l2 = 0.0;
    double sum_cosine = 0.0;
    double sum_pearson = 0.0;
    double sum_spearman = 0.0;
    double sum_top1_overlap = 0.0;
    double sum_top5_overlap = 0.0;
    size_t n_finite = 0;
    size_t n_mismatch = 0;

    void add_sample(double baseline, double slha);
    void add_vector(const std::vector<double> & baseline, const std::vector<double> & slha);
    void print(int32_t layer_id) const;
};

struct slha_layer_state {
    int32_t layer_id;
    int64_t n_embd_gqa;
    slha_kv_mode kv_mode;
    slha_score_mode score_mode;
    void * model_handle;
    void * scratch;

    // For collect mode
    std::vector<float> collected_k;
    std::unique_ptr<std::mutex> collect_mutex;

    // For shadow/replace modes
    std::unique_ptr<slha_shadow_metrics> shadow_metrics;

    slha_layer_state() : layer_id(0), n_embd_gqa(0), kv_mode(SLHA_KV_OFF),
                         score_mode(SLHA_SCORE_OFF),
                         model_handle(nullptr), scratch(nullptr),
                         collect_mutex(std::make_unique<std::mutex>()) {}

    // Non-copyable due to mutex
    slha_layer_state(const slha_layer_state&) = delete;
    slha_layer_state& operator=(const slha_layer_state&) = delete;

    // Movable
    slha_layer_state(slha_layer_state&&) = default;
    slha_layer_state& operator=(slha_layer_state&&) = default;
};

extern slha_tile_store g_slha_tile_store;

slha_kv_mode slha_kv_mode_from_env();
slha_score_mode slha_score_mode_from_env();

int slha_global_init(const char * weights_dir, slha_kv_mode mode);

void slha_global_shutdown();

int slha_get_num_layers();

slha_layer_state * slha_get_layer_state(int32_t il);

/** K-cache write callback: encode tiles (tilestore mode) or round-trip.
 *
 * userdata is a slha_layer_state*. In tilestore mode this also receives the
 * KV cache indices via the second input. */
void slha_k_transform(
    ggml_tensor * dst,
    const ggml_tensor * a,
    int ith,
    int nth,
    void * userdata
);

void slha_k_transform_with_idxs(
    ggml_tensor * dst,
    const ggml_tensor * a,
    const ggml_tensor * b,
    int ith,
    int nth,
    void * userdata
);

/** Shadow-mode callback: compare baseline logits with SLHA direct scores.
 *
 * Inputs: baseline kq logits and q. Output is a copy of kq (attention
 * unchanged). userdata is a slha_layer_state*. */
void slha_shadow_score(
    ggml_tensor * dst,
    const struct ggml_tensor * kq,
    const struct ggml_tensor * q,
    int ith,
    int nth,
    void * userdata
);

/** Per-layer count of tiles written to the tilestore.
 *  Incremented in slha_k_transform_with_idxs for each successful tile write.
 *  Accessed in slha_shadow_score to limit tile-presence checking to positions
 *  that have actually been written (because n_kv from kq->ne[0] is padded to
 *  at least 256 by llama_kv_cache::get_n_kv). */
extern std::atomic<size_t> g_slha_tiles_written[SLHA_MAX_LAYERS];

void slha_print_replace_summary();

/** Print the score-replacement layer-mask summary: requested spec, resolved
 *  layer ids, executed layer ids, unselected pass-through counts, and the
 *  fail-closed mask_valid flag. Called after the replace summary. */
void slha_print_score_mask_summary();

/** Print the experimental score-scale summary: active flag, source, mode,
 *  requested/canonical spec, manifest hash, per-executed-layer resolved scale,
 *  scaled_vectors/logits, invalid_scale, and the scale_manifest_valid flag.
 *  Only prints when a scale spec was provided (SLHA_SCORE_SCALE / _FILE). */
void slha_print_score_scale_summary();

/** Print the diagnostic score-oracle summary: mode, canonical spec, config
 *  hash, oracle vector/logit/permutation/tie counts, rejection counters, the
 *  runtime invariant-sampling result, and oracle_mode_valid. Only prints when
 *  an oracle spec was provided (SLHA_SCORE_ORACLE). */
void slha_print_score_oracle_summary();

void slha_flush_collected_activations(const char * output_dir);
void slha_print_shadow_metrics();
void slha_print_shadow_metrics_unlocked();

/** Clear the tile store and reset tile-written counters.
 *  Called from llama_kv_cache::clear() to avoid stale tile data between chunks. */
void slha_k_clear_all();

#endif

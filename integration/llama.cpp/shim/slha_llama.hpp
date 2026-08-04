#ifndef SLHA_LLAMA_HPP
#define SLHA_LLAMA_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <string>

struct ggml_context;
struct ggml_tensor;

#include "slha.h"

enum slha_kv_mode : int {
    SLHA_KV_OFF = 0,
    SLHA_KV_PASSTHROUGH = 1,
    SLHA_KV_ROUNDTRIP = 2,
    SLHA_KV_COLLECT = 3,
    SLHA_KV_SCORE_DIAG = 4,
    SLHA_KV_FUSED = 5,
};

struct slha_layer_state {
    int32_t layer_id;
    int64_t n_embd_gqa;
    slha_kv_mode mode;
    void * model_handle;
    void * scratch;

    // For collect mode
    std::vector<float> collected_k;
    std::unique_ptr<std::mutex> collect_mutex;

    // For score_diag mode: ring buffer of tiles, indexed by KV cache slot
    std::vector<char> tile_buffer;
    size_t tile_capacity;    // max tiles in buffer
    int32_t codec;

    slha_layer_state() : layer_id(0), n_embd_gqa(0), mode(SLHA_KV_OFF),
                         model_handle(nullptr), scratch(nullptr),
                         collect_mutex(std::make_unique<std::mutex>()),
                         tile_capacity(0), codec(SLHA_CODEC_MIXED) {}

    // Non-copyable due to mutex
    slha_layer_state(const slha_layer_state&) = delete;
    slha_layer_state& operator=(const slha_layer_state&) = delete;

    // Movable
    slha_layer_state(slha_layer_state&&) = default;
    slha_layer_state& operator=(slha_layer_state&&) = default;
};

slha_kv_mode slha_kv_mode_from_env();

int slha_global_init(const char * weights_dir, slha_kv_mode mode);

void slha_global_shutdown();

int slha_get_num_layers();

slha_layer_state * slha_get_layer_state(int32_t il);

void slha_k_transform(
    ggml_tensor * dst,
    const ggml_tensor * a,
    int ith,
    int nth,
    void * userdata
);

void slha_flush_collected_activations(const char * output_dir);

// Score-diagnostic callback for ggml_map_custom2.
// Called from llama-graph.cpp after ggml_mul_mat(k, q).
// In SCORE_DIAG mode: copies the true QK^T scores through and logs cos/KL
//   statistics vs SLHA scores computed from tiles (side branch, no effect on
//   attention).
// In FUSED mode: REPLACES the QK^T scores in dst with SLHA scores computed
//   from the encoded tiles (the attention output then uses SLHA scores).
//   Slots without a tile fall back to the true score so attention is never
//   broken during warmup.
// `a` = kq scores tensor, `b` = q tensor, userdata = slha_layer_state.
void slha_score_diag_callback(
    ggml_tensor * dst,
    const ggml_tensor * a,
    const ggml_tensor * b,
    int ith,
    int nth,
    void * userdata
);

// Build the fused-QK node and attach it to the graph.
// In SCORE_DIAG mode: returns the original kq tensor unchanged (diagnostic is
//   a side-branch). In FUSED mode: returns the kq tensor whose scores have
//   been replaced by SLHA scores, so the caller must use the return value as
//   the attention input.
// Called from build_attn_mha in llama-graph.cpp.
ggml_tensor * slha_build_fused_qk(
    ggml_context * ctx,
    ggml_tensor * kq,
    ggml_tensor * k,
    ggml_tensor * q,
    int il,
    struct ggml_cgraph * gf
);

#endif

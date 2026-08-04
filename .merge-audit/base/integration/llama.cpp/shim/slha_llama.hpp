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

enum slha_kv_mode : int {
    SLHA_KV_OFF = 0,
    SLHA_KV_PASSTHROUGH = 1,
    SLHA_KV_ROUNDTRIP = 2,
    SLHA_KV_COLLECT = 3,
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
    
    slha_layer_state() : layer_id(0), n_embd_gqa(0), mode(SLHA_KV_OFF), 
                         model_handle(nullptr), scratch(nullptr),
                         collect_mutex(std::make_unique<std::mutex>()) {}
    
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

#endif

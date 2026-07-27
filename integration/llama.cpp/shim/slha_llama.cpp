#include "slha_llama.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "ggml.h"

namespace {

struct GlobalState {
    slha_kv_mode mode = SLHA_KV_OFF;
    std::vector<slha_layer_state> layers;
    std::mutex mutex;
    bool initialized = false;
    int32_t num_layers = 0;
};

GlobalState & get_global_state() {
    static GlobalState state;
    return state;
}

}

slha_kv_mode slha_kv_mode_from_env() {
    const char * env = std::getenv("SLHA_KV_MODE");
    if (!env) {
        return SLHA_KV_OFF;
    }
    std::string mode_str(env);
    if (mode_str == "off") return SLHA_KV_OFF;
    if (mode_str == "passthrough") return SLHA_KV_PASSTHROUGH;
    if (mode_str == "roundtrip") return SLHA_KV_ROUNDTRIP;
    if (mode_str == "collect") return SLHA_KV_COLLECT;
    std::cerr << "[SLHA] unknown SLHA_KV_MODE='" << mode_str
              << "', falling back to off\n";
    return SLHA_KV_OFF;
}

int slha_global_init(const char * weights_dir, slha_kv_mode mode) {
    auto & state = get_global_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (state.initialized) {
        std::cerr << "[SLHA] already initialized\n";
        return -1;
    }

    state.mode = mode;

    if (mode == SLHA_KV_OFF) {
        state.initialized = true;
        return 0;
    }

    if (mode == SLHA_KV_PASSTHROUGH) {
        std::cout << "[SLHA] passthrough mode enabled (no compression)\n";
        state.initialized = true;
        return 0;
    }

    if (mode == SLHA_KV_ROUNDTRIP || mode == SLHA_KV_COLLECT) {
        if (!weights_dir) {
            std::cerr << "[SLHA] roundtrip/collect mode requires weights_dir\n";
            return -1;
        }
        
        // Create layer states for up to 128 layers (will be initialized on first use)
        state.num_layers = 128;
        state.layers.resize(state.num_layers);
        for (int32_t i = 0; i < state.num_layers; ++i) {
            state.layers[i].layer_id = i;
            state.layers[i].n_embd_gqa = 0;
            state.layers[i].mode = mode;
            state.layers[i].model_handle = nullptr;
            state.layers[i].scratch = nullptr;
        }
        
        std::cout << "[SLHA] " << (mode == SLHA_KV_ROUNDTRIP ? "roundtrip" : "collect")
                  << " mode enabled, weights_dir=" << weights_dir << "\n";
        state.initialized = true;
        return 0;
    }

    return 0;
}

void slha_global_shutdown() {
    auto & state = get_global_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    // Flush collected activations if in collect mode
    if (state.mode == SLHA_KV_COLLECT) {
        const char * output_dir = std::getenv("SLHA_WEIGHTS_DIR");
        if (output_dir) {
            // Unlock before calling flush (it will re-lock per layer)
            state.mutex.unlock();
            slha_flush_collected_activations(output_dir);
            state.mutex.lock();
        }
    }

    for (auto & layer : state.layers) {
        if (layer.scratch) {
            std::free(layer.scratch);
            layer.scratch = nullptr;
        }
        if (layer.model_handle) {
            // slha_weights_free(layer.model_handle);
            layer.model_handle = nullptr;
        }
        layer.collected_k.clear();
    }
    state.layers.clear();
    state.initialized = false;
}

int slha_get_num_layers() {
    auto & state = get_global_state();
    return static_cast<int>(state.layers.size());
}

slha_layer_state * slha_get_layer_state(int32_t il) {
    auto & state = get_global_state();
    if (il < 0 || il >= static_cast<int32_t>(state.layers.size())) {
        return nullptr;
    }
    return &state.layers[il];
}

void slha_k_transform(
    ggml_tensor * dst,
    const ggml_tensor * a,
    int ith,
    int nth,
    void * userdata
) {
    auto * layer = static_cast<slha_layer_state *>(userdata);
    if (!layer) {
        return;
    }

    if (layer->mode == SLHA_KV_OFF) {
        return;
    }

    // Tensor layout after view_2d:
    // ne[0] = n_embd_gqa (columns, K vector dimension)
    // ne[1] = n_tokens (rows)
    // Data is row-major: each row is one token's K vector [n_embd_gqa]
    
    const int64_t n_embd_gqa = a->ne[0];
    const int64_t n_tokens = a->ne[1];
    
    // All threads must participate in both passthrough and collection.
    // Divide tokens among threads.
    const int64_t tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int64_t token_start = ith * tokens_per_thread;
    const int64_t token_end = std::min(token_start + tokens_per_thread, n_tokens);

    if (token_start >= n_tokens) {
        return;  // This thread has no work
    }

    if (layer->mode == SLHA_KV_PASSTHROUGH) {
        // Passthrough: copy input to output unchanged.
        const size_t elem_size = ggml_type_size(a->type);
        const size_t block_size = ggml_blck_size(a->type);
        const size_t row_bytes = (n_embd_gqa / block_size) * elem_size;

        const uint8_t * src = static_cast<const uint8_t *>(a->data);
        uint8_t * dst_data = static_cast<uint8_t *>(dst->data);

        for (int64_t t = token_start; t < token_end; ++t) {
            const uint8_t * src_row = src + t * row_bytes;
            uint8_t * dst_row = dst_data + t * row_bytes;
            std::memcpy(dst_row, src_row, row_bytes);
        }
        return;
    }

    if (layer->mode == SLHA_KV_COLLECT) {
        // Collect mode: gather K vectors for training.
        // All threads collect their assigned tokens with synchronization.
        
        // Update dimension on first call (any thread can do this).
        if (layer->n_embd_gqa == 0) {
            std::lock_guard<std::mutex> lock(layer->collect_mutex);
            if (layer->n_embd_gqa == 0) {  // Double-check after acquiring lock
                layer->n_embd_gqa = n_embd_gqa;
                std::cout << "[SLHA] layer " << layer->layer_id 
                          << " collecting K vectors, dim=" << n_embd_gqa 
                          << ", n_tokens=" << n_tokens << "\n";
            }
        }

        // For now, only support GGML_TYPE_F32.
        // Other types (F16, BF16, quantized) would need conversion.
        if (a->type != GGML_TYPE_F32) {
            // Pass through without collecting.
            const size_t elem_size = ggml_type_size(a->type);
            const size_t block_size = ggml_blck_size(a->type);
            const size_t row_bytes = (n_embd_gqa / block_size) * elem_size;
            
            const uint8_t * src = static_cast<const uint8_t *>(a->data);
            uint8_t * dst_data = static_cast<uint8_t *>(dst->data);
            
            for (int64_t t = token_start; t < token_end; ++t) {
                const uint8_t * src_row = src + t * row_bytes;
                uint8_t * dst_row = dst_data + t * row_bytes;
                std::memcpy(dst_row, src_row, row_bytes);
            }
            
            if (ith == 0) {
                std::cerr << "[SLHA] WARNING: layer " << layer->layer_id 
                          << " tensor type is not F32, skipping collection\n";
            }
            return;
        }

        const float * src_data = static_cast<const float *>(a->data);
        
        // Each thread collects its assigned tokens with mutex protection.
        {
            std::lock_guard<std::mutex> lock(layer->collect_mutex);
            for (int64_t t = token_start; t < token_end; ++t) {
                const float * src_row = src_data + t * n_embd_gqa;
                layer->collected_k.insert(
                    layer->collected_k.end(),
                    src_row,
                    src_row + n_embd_gqa
                );
            }
        }

        // Pass through: copy to output.
        const size_t row_bytes = n_embd_gqa * sizeof(float);
        uint8_t * dst_data = static_cast<uint8_t *>(dst->data);
        
        for (int64_t t = token_start; t < token_end; ++t) {
            const uint8_t * src_row = static_cast<const uint8_t *>(a->data) + t * row_bytes;
            uint8_t * dst_row = dst_data + t * row_bytes;
            std::memcpy(dst_row, src_row, row_bytes);
        }
        
        return;
    }

    // SLHA_KV_ROUNDTRIP is not yet implemented.
}

void slha_flush_collected_activations(const char * output_dir) {
    auto & state = get_global_state();
    
    if (!output_dir) {
        std::cerr << "[SLHA] flush: output_dir is NULL\n";
        return;
    }

    std::cout << "[SLHA] flushing collected activations to " << output_dir << "\n";

    // Create output directory if needed.
    std::string cmd = std::string("mkdir -p ") + output_dir;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "[SLHA] failed to create output directory: " << output_dir << "\n";
        return;
    }

    for (auto & layer : state.layers) {
        std::vector<float> data_copy;
        int64_t n_embd_gqa_copy = 0;
        
        // Copy data under lock, then release before I/O.
        {
            std::lock_guard<std::mutex> lock(layer.collect_mutex);
            if (layer.collected_k.empty()) {
                continue;
            }
            data_copy = std::move(layer.collected_k);
            n_embd_gqa_copy = layer.n_embd_gqa;
            layer.collected_k.clear();
        }

        // Write to temporary file first (atomic write).
        std::string k_path_tmp = std::string(output_dir) + "/layer-" + 
                                 std::to_string(layer.layer_id) + "-k.bin.tmp";
        std::string k_path = std::string(output_dir) + "/layer-" + 
                            std::to_string(layer.layer_id) + "-k.bin";
        
        std::ofstream out(k_path_tmp, std::ios::binary);
        if (!out) {
            std::cerr << "[SLHA] failed to open " << k_path_tmp << "\n";
            continue;
        }

        // Write header: magic, rows, cols.
        const uint32_t magic = 0x534C4841;  // "SLHA"
        const uint32_t rows = static_cast<uint32_t>(data_copy.size() / n_embd_gqa_copy);
        const uint32_t cols = static_cast<uint32_t>(n_embd_gqa_copy);

        // Validate: ensure we have complete rows.
        if (data_copy.size() % n_embd_gqa_copy != 0) {
            std::cerr << "[SLHA] layer " << layer.layer_id 
                      << ": collected data size (" << data_copy.size() 
                      << ") is not a multiple of dimension (" << n_embd_gqa_copy << ")\n";
            out.close();
            std::remove(k_path_tmp.c_str());
            continue;
        }

        out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char *>(data_copy.data()),
                  data_copy.size() * sizeof(float));

        if (!out) {
            std::cerr << "[SLHA] layer " << layer.layer_id 
                      << ": write failed, discarding partial file\n";
            out.close();
            std::remove(k_path_tmp.c_str());
            continue;
        }

        out.close();
        
        // Atomically rename temporary file to final name.
        if (std::rename(k_path_tmp.c_str(), k_path.c_str()) != 0) {
            std::cerr << "[SLHA] layer " << layer.layer_id 
                      << ": failed to rename " << k_path_tmp << " to " << k_path << "\n";
            std::remove(k_path_tmp.c_str());
            continue;
        }

        std::cout << "[SLHA] layer " << layer.layer_id << ": wrote " 
                  << rows << " tokens × " << cols << " dims to " << k_path << "\n";
    }
}

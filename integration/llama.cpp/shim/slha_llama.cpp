#include "slha_llama.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <string>
#include <iostream>

#include "ggml.h"

namespace {

struct GlobalState {
    slha_kv_mode mode = SLHA_KV_OFF;
    std::vector<slha_layer_state> layers;
    std::mutex mutex;
    bool initialized = false;
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

    for (auto & layer : state.layers) {
        if (layer.scratch) {
            std::free(layer.scratch);
            layer.scratch = nullptr;
        }
        if (layer.model_handle) {
            // slha_weights_free(layer.model_handle);
            layer.model_handle = nullptr;
        }
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

    if (layer->mode == SLHA_KV_PASSTHROUGH) {
        // Passthrough: copy input to output unchanged.
        // Tensor layout: [n_embd_gqa, n_tokens] after view_2d.
        // Each thread handles a subset of tokens (rows).

        const int64_t n_embd_gqa = a->ne[0];
        const int64_t n_tokens = a->ne[1];
        const size_t elem_size = ggml_type_size(a->type);
        const size_t block_size = ggml_blck_size(a->type);
        const size_t row_bytes = (n_embd_gqa / block_size) * elem_size;

        // Divide tokens among threads.
        const int64_t tokens_per_thread = (n_tokens + nth - 1) / nth;
        const int64_t token_start = ith * tokens_per_thread;
        const int64_t token_end = (token_start + tokens_per_thread < n_tokens)
                                  ? token_start + tokens_per_thread
                                  : n_tokens;

        if (token_start >= n_tokens) {
            return;
        }

        const uint8_t * src = static_cast<const uint8_t *>(a->data);
        uint8_t * dst_data = static_cast<uint8_t *>(dst->data);

        for (int64_t t = token_start; t < token_end; ++t) {
            const uint8_t * src_row = src + t * row_bytes;
            uint8_t * dst_row = dst_data + t * row_bytes;
            std::memcpy(dst_row, src_row, row_bytes);
        }

        return;
    }

    // SLHA_KV_ROUNDTRIP and SLHA_KV_COLLECT are not yet implemented.
    // They will be added in subsequent milestones.
}

#include "slha_llama.hpp"

#include "ggml.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <mutex>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>

namespace {

struct GlobalState {
    slha_kv_mode mode = SLHA_KV_OFF;
    std::string weights_dir;
    std::vector<slha_layer_state> layers;
    std::mutex mutex;
    bool initialized = false;
    int32_t num_layers = 0;
};

GlobalState & get_global_state() {
    static GlobalState state;
    return state;
}

int32_t slha_codec_from_env() {
    const char * env = std::getenv("SLHA_CODEC");
    if (!env) {
        return SLHA_CODEC_MIXED;
    }
    std::string codec_str(env);
    if (codec_str == "mixed")  return SLHA_CODEC_MIXED;
    if (codec_str == "mix3")   return SLHA_CODEC_MIX3;
    if (codec_str == "grouped") return SLHA_CODEC_INT4_GROUPED;
    if (codec_str == "nf4")    return SLHA_CODEC_NF4;
    if (codec_str == "tq3")    return SLHA_CODEC_TQ3;
    std::cerr << "[SLHA] unknown SLHA_CODEC='" << codec_str << "', falling back to mixed\n";
    return SLHA_CODEC_MIXED;
}

std::string slha_layer_path(const std::string & weights_dir, int32_t layer_id) {
    char name[64];
    std::snprintf(name, sizeof(name), "layer-%03d.slhw", layer_id);
    return weights_dir + "/" + name;
}

bool slha_load_layer_model(slha_layer_state * layer, const std::string & weights_dir) {
    if (layer->model_handle) {
        return true;
    }
    std::string path = slha_layer_path(weights_dir, layer->layer_id);
    SlhaModel * model = slha_weights_load(path.c_str());
    if (!model) {
        std::cerr << "[SLHA] layer " << layer->layer_id
                  << ": failed to load weights from " << path
                  << ": " << slha_last_error_message() << "\n";
        layer->mode = SLHA_KV_OFF;
        return false;
    }
    size_t expected_dim = static_cast<size_t>(layer->n_embd_gqa);
    size_t model_dim = slha_model_dim(model);
    if (model_dim != expected_dim) {
        std::cerr << "[SLHA] layer " << layer->layer_id
                  << ": dimension mismatch: model d=" << model_dim
                  << ", expected " << expected_dim << "\n";
        slha_weights_free(model);
        layer->mode = SLHA_KV_OFF;
        return false;
    }
    layer->model_handle = model;
    return true;
}

static void softmax_inplace(float * data, int64_t n, float scale) {
    float mx = data[0];
    for (int64_t i = 1; i < n; ++i) if (data[i] > mx) mx = data[i];
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        data[i] = std::exp((data[i] - mx) * scale);
        sum += data[i];
    }
    if (sum > 0.0) { double inv = 1.0 / sum; for (int64_t i = 0; i < n; ++i) data[i] = (float)(data[i] * inv); }
}

static double kl_divergence(const float * p, const float * q, int64_t n) {
    double kl = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        if (p[i] > 1e-9f && q[i] > 1e-9f) kl += p[i] * std::log(p[i] / q[i]);
    }
    return kl;
}

} // namespace

slha_kv_mode slha_kv_mode_from_env() {
    const char * env = std::getenv("SLHA_KV_MODE");
    if (!env) return SLHA_KV_OFF;
    std::string mode_str(env);
    if (mode_str == "off") return SLHA_KV_OFF;
    if (mode_str == "passthrough") return SLHA_KV_PASSTHROUGH;
    if (mode_str == "roundtrip") return SLHA_KV_ROUNDTRIP;
    if (mode_str == "collect") return SLHA_KV_COLLECT;
    if (mode_str == "scorediag") return SLHA_KV_SCORE_DIAG;
    std::cerr << "[SLHA] unknown SLHA_KV_MODE='" << mode_str << "', falling back to off\n";
    return SLHA_KV_OFF;
}

int slha_global_init(const char * weights_dir, slha_kv_mode mode) {
    auto & state = get_global_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.initialized) { std::cerr << "[SLHA] already initialized\n"; return -1; }
    state.mode = mode;
    state.weights_dir = weights_dir ? weights_dir : "";
    if (mode == SLHA_KV_OFF) { state.initialized = true; return 0; }
    if (mode == SLHA_KV_PASSTHROUGH) {
        std::cout << "[SLHA] passthrough mode enabled (no compression)\n";
        state.initialized = true; std::atexit(slha_global_shutdown); return 0;
    }
    if (mode == SLHA_KV_ROUNDTRIP || mode == SLHA_KV_COLLECT || mode == SLHA_KV_SCORE_DIAG) {
        if (state.weights_dir.empty()) {
            std::cerr << "[SLHA] roundtrip/collect/scorediag mode requires weights_dir\n"; return -1;
        }
        state.num_layers = 128;
        state.layers.resize(state.num_layers);
        for (int32_t i = 0; i < state.num_layers; ++i) {
            state.layers[i].layer_id = i;
            state.layers[i].n_embd_gqa = 0;
            state.layers[i].mode = mode;
            state.layers[i].model_handle = nullptr;
            state.layers[i].scratch = nullptr;
            state.layers[i].codec = slha_codec_from_env();
        }
        std::string mode_name = (mode == SLHA_KV_ROUNDTRIP) ? "roundtrip" :
                                (mode == SLHA_KV_COLLECT) ? "collect" : "scorediag";
        std::cout << "[SLHA] " << mode_name << " mode enabled, weights_dir=" << state.weights_dir << "\n";
        state.initialized = true; std::atexit(slha_global_shutdown); return 0;
    }
    return 0;
}

void slha_global_shutdown() {
    auto & state = get_global_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.mode == SLHA_KV_COLLECT) {
        if (!state.weights_dir.empty()) {
            state.mutex.unlock();
            slha_flush_collected_activations(state.weights_dir.c_str());
            state.mutex.lock();
        }
    }
    for (auto & layer : state.layers) {
        if (layer.scratch) { std::free(layer.scratch); layer.scratch = nullptr; }
        if (layer.model_handle) {
            slha_weights_free(static_cast<SlhaModel *>(layer.model_handle));
            layer.model_handle = nullptr;
        }
        layer.collected_k.clear();
        layer.tile_buffer.clear();
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
    if (il < 0 || il >= static_cast<int32_t>(state.layers.size())) return nullptr;
    return &state.layers[il];
}

void slha_k_transform(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    auto * layer = static_cast<slha_layer_state *>(userdata);
    if (!layer) return;
    if (layer->mode == SLHA_KV_OFF || layer->mode == SLHA_KV_PASSTHROUGH) return;

    const int64_t n_embd_gqa = a->ne[0];
    const int64_t n_tokens = a->ne[1];
    const int64_t tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int64_t token_start = ith * tokens_per_thread;
    const int64_t token_end = std::min(token_start + tokens_per_thread, n_tokens);
    if (token_start >= n_tokens) return;

    // Lazy dim init for all modes
    if (layer->n_embd_gqa == 0) {
        std::lock_guard<std::mutex> lock(*layer->collect_mutex);
        if (layer->n_embd_gqa == 0) {
            layer->n_embd_gqa = n_embd_gqa;
            if (layer->mode == SLHA_KV_SCORE_DIAG || layer->mode == SLHA_KV_COLLECT) {
                std::cout << "[SLHA] layer " << layer->layer_id
                          << " dim=" << n_embd_gqa << " mode=" << (int)layer->mode << "\n";
            }
            if (layer->mode == SLHA_KV_SCORE_DIAG) {
                layer->tile_capacity = 8192;
                layer->tile_buffer.resize(layer->tile_capacity * slha_tile_size());
            }
            if ((layer->mode == SLHA_KV_ROUNDTRIP || layer->mode == SLHA_KV_SCORE_DIAG) && !layer->model_handle) {
                auto & state = get_global_state();
                slha_load_layer_model(layer, state.weights_dir);
            }
        }
    }

    if (layer->mode == SLHA_KV_OFF) {
        // Model loading failed; pass through
        const size_t row_bytes = ggml_row_size(a->type, n_embd_gqa);
        const uint8_t * src = static_cast<const uint8_t *>(a->data);
        uint8_t * dst_data = static_cast<uint8_t *>(dst->data);
        for (int64_t t = token_start; t < token_end; ++t)
            std::memcpy(dst_data + t * dst->nb[1], src + t * a->nb[1], row_bytes);
        return;
    }

    // Non-F32 input: pass through
    if (a->type != GGML_TYPE_F32) {
        const size_t row_bytes = ggml_row_size(a->type, n_embd_gqa);
        const uint8_t * src = static_cast<const uint8_t *>(a->data);
        uint8_t * dst_data = static_cast<uint8_t *>(dst->data);
        for (int64_t t = token_start; t < token_end; ++t)
            std::memcpy(dst_data + t * dst->nb[1], src + t * a->nb[1], row_bytes);
        if (ith == 0)
            std::cerr << "[SLHA] WARNING: layer " << layer->layer_id << " not F32, passing through\n";
        return;
    }

    const size_t row_bytes = n_embd_gqa * sizeof(float);
    const uint8_t * src_base = static_cast<const uint8_t *>(a->data);
    uint8_t * dst_base = static_cast<uint8_t *>(dst->data);

    // MODE: SCORE_DIAG — encode K to tiles, pass K through
    if (layer->mode == SLHA_KV_SCORE_DIAG && layer->model_handle) {
        auto * model = static_cast<SlhaModel *>(layer->model_handle);
        const size_t tile_sz = slha_tile_size();
        for (int64_t t = token_start; t < token_end; ++t) {
            const float * src_ptr = reinterpret_cast<const float *>(src_base + t * a->nb[1]);
            size_t slot = static_cast<size_t>(token_start + t);
            if (slot >= layer->tile_capacity) {
                std::lock_guard<std::mutex> lock(*layer->collect_mutex);
                layer->tile_capacity = slot + 1024;
                layer->tile_buffer.resize(layer->tile_capacity * tile_sz);
            }
            auto * tile = reinterpret_cast<SciRustSlhaTile *>(layer->tile_buffer.data() + slot * tile_sz);
            int rc = slha_encode_key(model, src_ptr, static_cast<size_t>(n_embd_gqa),
                                     static_cast<uint32_t>(slot), layer->codec, tile);
            if (rc != SLHA_OK && ith == 0)
                std::cerr << "[SLHA] layer " << layer->layer_id << " token " << (token_start + t)
                          << " encode: " << slha_last_error_message() << "\n";
        }
        // Pass K through unchanged
        for (int64_t t = token_start; t < token_end; ++t)
            std::memcpy(dst_base + t * dst->nb[1], src_base + t * a->nb[1], row_bytes);
        return;
    }

    // MODE: COLLECT — gather K vectors
    if (layer->mode == SLHA_KV_COLLECT) {
        const float * src_data = static_cast<const float *>(a->data);
        {
            std::lock_guard<std::mutex> lock(*layer->collect_mutex);
            for (int64_t t = token_start; t < token_end; ++t) {
                const float * src_row = src_data + t * n_embd_gqa;
                layer->collected_k.insert(layer->collected_k.end(), src_row, src_row + n_embd_gqa);
            }
        }
        for (int64_t t = token_start; t < token_end; ++t)
            std::memcpy(dst_base + t * dst->nb[1], src_base + t * a->nb[1], row_bytes);
        return;
    }

    // MODE: ROUNDTRIP — encode then decode K
    if (layer->mode == SLHA_KV_ROUNDTRIP && layer->model_handle) {
        auto * model = static_cast<SlhaModel *>(layer->model_handle);
        const int32_t codec = layer->codec;
        const size_t d = static_cast<size_t>(n_embd_gqa);
        thread_local std::vector<float> in_row;
        in_row.resize(d);
        for (int64_t t = token_start; t < token_end; ++t) {
            const float * src_ptr = reinterpret_cast<const float *>(src_base + t * a->nb[1]);
            float * dst_ptr = reinterpret_cast<float *>(dst_base + t * dst->nb[1]);
            SciRustSlhaTile tile;
            int rc = slha_encode_key(model, src_ptr, d, static_cast<uint32_t>(t), codec, &tile);
            if (rc != SLHA_OK) {
                std::memcpy(dst_ptr, src_ptr, row_bytes);
                continue;
            }
            rc = slha_decode_key(model, &tile, dst_ptr, d);
            if (rc != SLHA_OK) {
                std::memcpy(dst_ptr, src_ptr, row_bytes);
                continue;
            }
        }
        return;
    }

    // Fallback: pass through
    for (int64_t t = token_start; t < token_end; ++t)
        std::memcpy(dst_base + t * dst->nb[1], src_base + t * a->nb[1], row_bytes);
}

// Score diagnostic callback — called by ggml during graph computation.
// a = kq tensor (true QK^T scores)
// b = q tensor (after permute)
// dst = output (unused downstream, but must be filled)
// userdata = slha_layer_state *
void slha_score_diag_callback(
    ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
    int ith, int nth, void * userdata)
{
    if (ith != 0) return; // only thread 0 does work

    auto * layer = static_cast<slha_layer_state *>(userdata);
    if (!layer || !layer->model_handle || ggml_nbytes(a) == 0) {
        std::memcpy(dst->data, a->data, ggml_nbytes(a));
        return;
    }

    // Copy true scores through to output (diagnostic does not modify attention)
    std::memcpy(dst->data, a->data, ggml_nbytes(a));

    const int64_t n_kv          = a->ne[0];
    const int64_t n_tokens_mul  = a->ne[1];
    const int64_t n_head        = a->ne[2];
    const int64_t n_stream      = a->ne[3] > 0 ? a->ne[3] : 1;
    const int64_t n_embd_head_k = b->ne[0] > 0 ? b->ne[0] : 1;
    const int64_t n_head_q      = b->ne[2] > 0 ? b->ne[2] : 1;
    const int64_t n_tokens_q    = b->ne[1] > 0 ? b->ne[1] : 1;

    // Strides for indexing into kq data
    const int64_t kq_stride_t = n_kv;
    const int64_t kq_stride_h = kq_stride_t * n_tokens_mul;
    const int64_t kq_stride_s = kq_stride_h * n_head;

    // Strides for indexing into q data
    const int64_t q_stride_t = n_embd_head_k;
    const int64_t q_stride_h = q_stride_t * n_tokens_q;
    const int64_t q_stride_s = q_stride_h * n_head_q;

    const float * kq_data = static_cast<const float *>(a->data);
    const float * q_data  = static_cast<const float *>(b->data);

    auto * model = static_cast<SlhaModel *>(layer->model_handle);
    const size_t d = static_cast<size_t>(layer->n_embd_gqa);
    if (d == 0) return;

    const size_t tile_sz = slha_tile_size();
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(n_embd_head_k));

    std::vector<float> true_scores(static_cast<size_t>(n_kv));
    std::vector<float> slha_scores(static_cast<size_t>(n_kv));
    std::vector<float> q_coarse(SLHA_D_C);
    std::vector<uint64_t> q_sign(SLHA_RESIDUAL_WORDS);
    // Full query vector of dimension d = n_embd_gqa
    std::vector<float> full_q(d, 0.0f);

    double sum_cos = 0.0, sum_kl = 0.0;
    int64_t n_comparisons = 0;

    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t h = 0; h < n_head; ++h) {
            // Compute the collapsed full query vector of dimension n_embd_gqa.
            // The Q tensor after permute has shape [n_embd_head_k, n_tokens_mul, n_head_q, n_stream].
            // For GQA, n_head_q >= n_head (each KV head has multiple query heads).
            // The collapsed Q takes one query head per KV head.
            const int64_t qh_base = h * (n_head_q / n_head);

            for (int64_t t = 0; t < n_tokens_mul; ++t) {
                // Build full_q: for each of n_head positions in the KV head space,
                // copy n_embd_head_k values from the corresponding query head.
                // n_head * n_embd_head_k should equal d (or be close to it).
                for (int64_t g = 0; g < n_head; ++g) {
                    const int64_t src_head = qh_base + g;
                    if (src_head >= n_head_q) continue;
                    const float * q_src = q_data + src_head * q_stride_h + t * q_stride_t + s * q_stride_s;
                    float * q_dst = &full_q[static_cast<size_t>(g * n_embd_head_k)];
                    for (int64_t k = 0; k < n_embd_head_k && (g * n_embd_head_k + k) < static_cast<int64_t>(d); ++k) {
                        q_dst[k] = q_src[k];
                    }
                }

                // Prepare query via SLHA
                int rc = slha_prepare_query(model, full_q.data(), d, q_coarse.data(), q_sign.data());
                if (rc != SLHA_OK) continue;

                // Score all tiles
                size_t n_scored = 0;
                for (int64_t kv = 0; kv < n_kv; ++kv) {
                    if (kv >= static_cast<int64_t>(layer->tile_capacity)) {
                        slha_scores[static_cast<size_t>(kv)] = 0.0f;
                        continue;
                    }
                    auto * tile = reinterpret_cast<const SciRustSlhaTile *>(
                        layer->tile_buffer.data() + static_cast<size_t>(kv) * tile_sz);
                    rc = slha_process_tile(tile, q_coarse.data(), q_sign.data(), &slha_scores[static_cast<size_t>(kv)]);
                    if (rc != SLHA_OK) slha_scores[static_cast<size_t>(kv)] = 0.0f;
                    else n_scored++;
                }
                if (n_scored == 0) continue;

                // Read true scores from kq tensor
                // kq layout: [n_kv, n_tokens_mul, n_head, n_stream]
                for (int64_t kv = 0; kv < n_kv; ++kv) {
                    true_scores[static_cast<size_t>(kv)] = kq_data[
                        kv * kq_stride_t + t * kq_stride_t + h * kq_stride_h + s * kq_stride_s
                        // Wait: access pattern:
                        // kq_data[0]..kq_data[n_kv-1] = scores for all keys at (t=0, h=0, s=0)
                        // Then stride is: element at (kv, t, h, s) = kq_data[kv + t*n_kv + h*n_kv*n_tokens_mul + s*n_kv*n_tokens_mul*n_head]
                    ];
                    // Correct indexing for row-major tensor with ne=[n_kv, n_tokens_mul, n_head, n_stream]:
                    true_scores[static_cast<size_t>(kv)] = kq_data[
                        kv
                        + t * n_kv
                        + h * n_kv * n_tokens_mul
                        + s * n_kv * n_tokens_mul * n_head
                    ];
                }

                // Cosine similarity
                double dot_tt = 0.0, dot_ss = 0.0, dot_ts = 0.0;
                for (int64_t kv = 0; kv < n_kv; ++kv) {
                    double tv = (double)true_scores[static_cast<size_t>(kv)];
                    double sv = (double)slha_scores[static_cast<size_t>(kv)];
                    dot_tt += tv * tv;
                    dot_ss += sv * sv;
                    dot_ts += tv * sv;
                }
                double cos_sim = (dot_tt > 0.0 && dot_ss > 0.0)
                    ? dot_ts / (std::sqrt(dot_tt) * std::sqrt(dot_ss)) : 0.0;

                // KL divergence of softmax distributions
                std::vector<float> p(true_scores);
                std::vector<float> q_sm(slha_scores);
                softmax_inplace(p.data(), n_kv, kq_scale);
                softmax_inplace(q_sm.data(), n_kv, kq_scale);
                double kl = kl_divergence(p.data(), q_sm.data(), n_kv);

                sum_cos += cos_sim;
                sum_kl += kl;
                n_comparisons++;
            }
        }
    }

    if (n_comparisons > 0) {
        double avg_cos = sum_cos / n_comparisons;
        double avg_kl = sum_kl / n_comparisons;
        static std::mutex log_mutex;
        static int64_t total_calls = 0;
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            total_calls++;
            if (total_calls <= 10 || (total_calls % 100 == 0)) {
                std::cout << "[SLHA] score_diag layer=" << layer->layer_id
                          << " call=" << total_calls
                          << " cos=" << avg_cos
                          << " KL=" << avg_kl
                          << " n_comparisons=" << n_comparisons
                          << std::endl;
            }
        }
    }
}

ggml_tensor * slha_build_score_diag(
    ggml_context * ctx, ggml_tensor * kq, ggml_tensor * k, ggml_tensor * q,
    int il, struct ggml_graph * gf)
{
    (void)k; // unused — needed for graph dependency
    auto * layer = slha_get_layer_state(il);
    if (!layer || layer->mode != SLHA_KV_SCORE_DIAG) return kq;

    ggml_tensor * diag = ggml_map_custom2(ctx, kq, q, slha_score_diag_callback, 1, layer);
    ggml_set_name(diag, "kq_slha_diag");
    ggml_build_forward_expand(gf, diag);
    return kq;
}

void slha_flush_collected_activations(const char * output_dir) {
    auto & state = get_global_state();
    if (!output_dir) { std::cerr << "[SLHA] flush: output_dir is NULL\n"; return; }
    std::cout << "[SLHA] flushing collected activations to " << output_dir << "\n";
    std::string cmd = std::string("mkdir -p ") + output_dir;
    if (std::system(cmd.c_str()) != 0) { std::cerr << "[SLHA] mkdir failed\n"; return; }
    for (auto & layer : state.layers) {
        std::vector<float> data_copy;
        int64_t n_embd_gqa_copy = 0;
        {
            std::lock_guard<std::mutex> lock(*layer.collect_mutex);
            if (layer.collected_k.empty()) continue;
            data_copy = std::move(layer.collected_k);
            n_embd_gqa_copy = layer.n_embd_gqa;
            layer.collected_k.clear();
        }
        std::string k_path_tmp = std::string(output_dir) + "/layer-" +
            std::to_string(layer.layer_id) + "-k.bin.tmp";
        std::string k_path = std::string(output_dir) + "/layer-" +
            std::to_string(layer.layer_id) + "-k.bin";
        std::ofstream out(k_path_tmp, std::ios::binary);
        if (!out) { std::cerr << "[SLHA] failed to open " << k_path_tmp << "\n"; continue; }
        uint32_t magic = 0x534C4841;
        uint32_t rows = static_cast<uint32_t>(data_copy.size() / n_embd_gqa_copy);
        uint32_t cols = static_cast<uint32_t>(n_embd_gqa_copy);
        if (data_copy.size() % n_embd_gqa_copy != 0) {
            std::cerr << "[SLHA] layer " << layer.layer_id << ": size mismatch\n";
            out.close(); std::remove(k_path_tmp.c_str()); continue;
        }
        out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char *>(data_copy.data()), data_copy.size() * sizeof(float));
        if (!out) { out.close(); std::remove(k_path_tmp.c_str()); continue; }
        out.close();
        if (std::rename(k_path_tmp.c_str(), k_path.c_str()) != 0) {
            std::remove(k_path_tmp.c_str());
            continue;
        }
        std::cout << "[SLHA] layer " << layer.layer_id << ": wrote "
                  << rows << " tokens x " << cols << " dims\n";
    }
}

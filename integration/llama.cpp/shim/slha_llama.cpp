#include "slha_llama.hpp"

#include "slha.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <mutex>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "ggml.h"

slha_tile_store g_slha_tile_store;

namespace {

struct GlobalState {
    slha_kv_mode kv_mode = SLHA_KV_OFF;
    slha_score_mode score_mode = SLHA_SCORE_OFF;
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

} // namespace

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
    if (mode_str == "tilestore") return SLHA_KV_TILESTORE;
    std::cerr << "[SLHA] unknown SLHA_KV_MODE='" << mode_str
              << "', falling back to off\n";
    return SLHA_KV_OFF;
}

slha_score_mode slha_score_mode_from_env() {
    const char * env = std::getenv("SLHA_SCORE_MODE");
    if (!env) {
        return SLHA_SCORE_OFF;
    }
    std::string mode_str(env);
    if (mode_str == "off") return SLHA_SCORE_OFF;
    if (mode_str == "shadow") return SLHA_SCORE_SHADOW;
    if (mode_str == "replace") return SLHA_SCORE_REPLACE;
    std::cerr << "[SLHA] unknown SLHA_SCORE_MODE='" << mode_str
              << "', falling back to off\n";
    return SLHA_SCORE_OFF;
}

int slha_global_init(const char * weights_dir, slha_kv_mode mode) {
    auto & state = get_global_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (state.initialized) {
        std::cerr << "[SLHA] already initialized\n";
        return -1;
    }

    state.kv_mode = mode;
    state.score_mode = slha_score_mode_from_env();

    state.weights_dir = weights_dir ? weights_dir : "";

    if (mode == SLHA_KV_OFF && state.score_mode == SLHA_SCORE_OFF) {
        state.initialized = true;
        return 0;
    }

    if (mode == SLHA_KV_PASSTHROUGH) {
        std::cout << "[SLHA] passthrough mode enabled (no compression)\n";
        state.initialized = true;
        std::atexit(slha_global_shutdown);
        return 0;
    }

    if (mode == SLHA_KV_ROUNDTRIP || mode == SLHA_KV_COLLECT || mode == SLHA_KV_TILESTORE) {
        if (state.weights_dir.empty()) {
            std::cerr << "[SLHA] roundtrip/collect/tilestore mode requires weights_dir\n";
            return -1;
        }

        // Create layer states for up to 128 layers (will be initialized on first use)
        state.num_layers = 128;
        state.layers.resize(state.num_layers);
        for (int32_t i = 0; i < state.num_layers; ++i) {
            state.layers[i].layer_id = i;
            state.layers[i].n_embd_gqa = 0;
            state.layers[i].kv_mode = mode;
            state.layers[i].score_mode = state.score_mode;
            state.layers[i].model_handle = nullptr;
            state.layers[i].scratch = nullptr;
        }

        const char * label = "tilestore";
        if (mode == SLHA_KV_ROUNDTRIP) label = "roundtrip";
        else if (mode == SLHA_KV_COLLECT) label = "collect";
        std::cout << "[SLHA] " << label
                  << " mode enabled, weights_dir=" << state.weights_dir;
        if (state.score_mode == SLHA_SCORE_SHADOW) {
            std::cout << ", score_mode=shadow";
        } else if (state.score_mode == SLHA_SCORE_REPLACE) {
            std::cout << ", score_mode=replace";
        }
        std::cout << "\n";

        // Initialize the compressed K side store.
        if (mode == SLHA_KV_TILESTORE || state.score_mode != SLHA_SCORE_OFF) {
            const size_t n_layers = 128;
            const size_t capacity = 4096; // initial max positions per layer
            if (!g_slha_tile_store.init(n_layers, capacity, slha_tile_size())) {
                std::cerr << "[SLHA] failed to initialize tile store\n";
                return -1;
            }
        }

        state.initialized = true;
        std::atexit(slha_global_shutdown);
        return 0;
    }

    return 0;
}

bool slha_tile_store::init(size_t n_layers_, size_t capacity_, size_t tile_bytes_) {
    std::lock_guard<std::mutex> lock(mutex);
    n_layers = n_layers_;
    capacity = capacity_;
    tile_bytes = tile_bytes_;
    if (tile_bytes == 0) {
        return false;
    }
    tiles.assign(n_layers * capacity * tile_bytes, std::byte{0});
    valid.assign(n_layers * capacity, 0);
    return true;
}

void slha_tile_store::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    tiles.assign(tiles.size(), std::byte{0});
    std::fill(valid.begin(), valid.end(), 0);
}

bool slha_tile_store::write(int32_t layer_id, size_t position, const void * tile) {
    if (!tile || layer_id < 0) return false;
    const size_t layer = static_cast<size_t>(layer_id);
    std::lock_guard<std::mutex> lock(mutex);
    if (layer >= n_layers || position >= capacity || tile_bytes == 0) {
        return false;
    }
    const size_t idx = (layer * capacity + position) * tile_bytes;
    std::memcpy(tiles.data() + idx, tile, tile_bytes);
    valid[layer * capacity + position] = 1;
    return true;
}

const void * slha_tile_store::read(int32_t layer_id, size_t position) const {
    if (layer_id < 0) return nullptr;
    const size_t layer = static_cast<size_t>(layer_id);
    std::lock_guard<std::mutex> lock(mutex);
    if (layer >= n_layers || position >= capacity || tile_bytes == 0) {
        return nullptr;
    }
    if (!valid[layer * capacity + position]) {
        return nullptr;
    }
    return tiles.data() + (layer * capacity + position) * tile_bytes;
}

bool slha_tile_store::check_capacity(int32_t layer_id, size_t position) const {
    if (layer_id < 0) return false;
    const size_t layer = static_cast<size_t>(layer_id);
    std::lock_guard<std::mutex> lock(mutex);
    return layer < n_layers && position < capacity;
}

void slha_shadow_metrics::add_sample(double baseline, double slha) {
    if (!std::isfinite(baseline) || !std::isfinite(slha)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    ++n_samples;
    const double err = baseline - slha;
    const double abs_err = std::abs(err);
    max_abs_error = std::max(max_abs_error, abs_err);
    sum_abs_error += abs_err;
    const double denom = std::max(std::abs(baseline), 1e-12);
    sum_rel_l2 += (abs_err / denom);
    if (baseline != 0.0 && std::abs(slha - baseline) > 1e-3 * denom) {
        ++n_mismatch;
    }
}

void slha_shadow_metrics::add_vector(const std::vector<double> & baseline, const std::vector<double> & slha) {
    const size_t n = baseline.size();
    if (n == 0 || n != slha.size()) {
        return;
    }

    double dot = 0.0;
    double norm_b = 0.0;
    double norm_s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(baseline[i]) || !std::isfinite(slha[i])) {
            return;
        }
        dot += baseline[i] * slha[i];
        norm_b += baseline[i] * baseline[i];
        norm_s += slha[i] * slha[i];
    }
    const double denom = std::sqrt(norm_b * norm_s);
    const double cosine = denom > 1e-18 ? dot / denom : 0.0;

    std::vector<size_t> idx_b(n), idx_s(n);
    for (size_t i = 0; i < n; ++i) {
        idx_b[i] = i;
        idx_s[i] = i;
    }
    std::sort(idx_b.begin(), idx_b.end(), [&](size_t a, size_t b) { return baseline[a] > baseline[b]; });
    std::sort(idx_s.begin(), idx_s.end(), [&](size_t a, size_t b) { return slha[a] > slha[b]; });

    const auto top_k_overlap = [&](size_t k) -> double {
        k = std::min(k, n);
        size_t overlap = 0;
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                if (idx_b[i] == idx_s[j]) {
                    ++overlap;
                    break;
                }
            }
        }
        return static_cast<double>(overlap) / static_cast<double>(k);
    };

    const double top1 = n >= 1 && idx_b[0] == idx_s[0] ? 1.0 : 0.0;
    const double top5 = n >= 5 ? top_k_overlap(5) : top1;

    std::lock_guard<std::mutex> lock(mutex);
    ++n_vectors;
    sum_cosine += cosine;
    sum_top1_overlap += top1;
    sum_top5_overlap += top5;
}

static double slha_pearson(const std::vector<double> & x, const std::vector<double> & y) {
    const size_t n = x.size();
    if (n < 2 || n != y.size()) return 0.0;
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double num = 0.0, dx2 = 0.0, dy2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    const double den = std::sqrt(dx2 * dy2);
    return den > 1e-18 ? num / den : 0.0;
}

static double slha_spearman(const std::vector<double> & x, const std::vector<double> & y) {
    const size_t n = x.size();
    if (n < 2 || n != y.size()) return 0.0;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return x[a] < x[b]; });
    std::vector<double> rx(n), ry(n);
    for (size_t i = 0; i < n; ) {
        size_t j = i;
        while (j < n && x[idx[j]] == x[idx[i]]) ++j;
        const double rank = (i + 1 + j) / 2.0;
        for (size_t k = i; k < j; ++k) rx[idx[k]] = rank;
        i = j;
    }
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return y[a] < y[b]; });
    for (size_t i = 0; i < n; ) {
        size_t j = i;
        while (j < n && y[idx[j]] == y[idx[i]]) ++j;
        const double rank = (i + 1 + j) / 2.0;
        for (size_t k = i; k < j; ++k) ry[idx[k]] = rank;
        i = j;
    }
    return slha_pearson(rx, ry);
}

void slha_shadow_metrics::print(int32_t layer_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (n_samples == 0 && n_vectors == 0) return;
    const double mae = n_samples ? sum_abs_error / static_cast<double>(n_samples) : 0.0;
    const double rel_l2 = n_samples ? sum_rel_l2 / static_cast<double>(n_samples) : 0.0;
    const double cosine = n_vectors ? sum_cosine / static_cast<double>(n_vectors) : 0.0;
    const double top1 = n_vectors ? sum_top1_overlap / static_cast<double>(n_vectors) : 0.0;
    const double top5 = n_vectors ? sum_top5_overlap / static_cast<double>(n_vectors) : 0.0;
    std::cout << "[SLHA] layer " << layer_id
              << " shadow n=" << n_samples
              << " max_abs_err=" << max_abs_error
              << " mae=" << mae
              << " rel_l2=" << rel_l2
              << " mismatches=" << n_mismatch
              << " vectors=" << n_vectors
              << " cosine=" << cosine
              << " top1=" << top1
              << " top5=" << top5
              << "\n";
}

void slha_print_shadow_metrics_unlocked() {
    auto & state = get_global_state();
    for (auto & layer : state.layers) {
        if (layer.shadow_metrics) {
            layer.shadow_metrics->print(layer.layer_id);
        }
    }
}

void slha_print_shadow_metrics() {
    auto & state = get_global_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    slha_print_shadow_metrics_unlocked();
}

void slha_global_shutdown() {
    auto & state = get_global_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    // Flush collected activations if in collect mode
    if (state.kv_mode == SLHA_KV_COLLECT) {
        if (!state.weights_dir.empty()) {
            // Unlock before calling flush (it will re-lock per layer)
            state.mutex.unlock();
            slha_flush_collected_activations(state.weights_dir.c_str());
            state.mutex.lock();
        }
    }

    slha_print_shadow_metrics_unlocked();

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
        layer.shadow_metrics.reset();
    }
    state.layers.clear();
    state.initialized = false;
    g_slha_tile_store.reset();
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

namespace {

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
    std::cerr << "[SLHA] unknown SLHA_CODEC='" << codec_str
              << "', falling back to mixed\n";
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
        layer->kv_mode = SLHA_KV_OFF;
        return false;
    }

    size_t expected_dim = static_cast<size_t>(layer->n_embd_gqa);
    size_t model_dim = slha_model_dim(model);
    if (model_dim != expected_dim) {
        std::cerr << "[SLHA] layer " << layer->layer_id
                  << ": dimension mismatch: model d=" << model_dim
                  << ", expected " << expected_dim << "\n";
        slha_weights_free(model);
        layer->kv_mode = SLHA_KV_OFF;
        return false;
    }

    layer->model_handle = model;
    return true;
}

} // namespace

void slha_k_transform(
    ggml_tensor * dst,
    const ggml_tensor * a,
    int ith,
    int nth,
    void * userdata
) {
    slha_k_transform_with_idxs(dst, a, nullptr, ith, nth, userdata);
}

void slha_shadow_score(
    ggml_tensor * dst,
    const ggml_tensor * kq,
    const ggml_tensor * q,
    int ith,
    int nth,
    void * userdata
) {
    auto * layer = static_cast<slha_layer_state *>(userdata);
    if (!layer || layer->score_mode == SLHA_SCORE_OFF) {
        return;
    }

    if (kq->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return;
    }

    // Copy baseline logits through element-wise so the attention graph is
    // unchanged regardless of tensor strides or shape.
    {
        const int64_t n = ggml_nelements(dst);
        const int64_t i0 = (n * ith) / nth;
        const int64_t i1 = (n * (ith + 1)) / nth;
        const float * src = static_cast<const float *>(kq->data);
        float * out = static_cast<float *>(dst->data);
        for (int64_t i = i0; i < i1; ++i) {
            out[i] = src[i];
        }
    }

    // This callback is inserted after build_attn_mha permutes q and k with
    // permute(0,2,1,3). After that, q is [head_dim, n_tokens, n_head, ...]
    // and kq is [n_kv, n_tokens, n_head, ...].
    const int64_t n_kv     = kq->ne[0];
    const int64_t n_token  = kq->ne[1];
    const int64_t n_head   = kq->ne[2];
    const int64_t head_dim = q->ne[0];

    if (q->ne[1] != n_token || q->ne[2] != n_head || q->ne[3] != kq->ne[3]) {
        // Only complain if the layer has tiles; otherwise this is just a graph
        // warmup/build call with dummy shapes.
        if (ith == 0 && g_slha_tile_store.read(layer->layer_id, 0)) {
            std::cerr << "[SLHA] shadow score: unsupported shape (q="
                      << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3]
                      << " kq=" << kq->ne[0] << "," << kq->ne[1] << "," << kq->ne[2] << "," << kq->ne[3]
                      << ")\n";
        }
        return;
    }

    const int64_t n_stream = kq->ne[3];

    if (layer->n_embd_gqa == 0 || layer->n_embd_gqa % head_dim != 0) {
        if (ith == 0 && g_slha_tile_store.read(layer->layer_id, 0)) {
            std::cerr << "[SLHA] shadow score: unsupported dimensions: n_embd_gqa="
                      << layer->n_embd_gqa << " head_dim=" << head_dim << "\n";
        }
        return;
    }

    const int64_t n_kv_head = layer->n_embd_gqa / head_dim;
    if (n_kv_head == 0 || n_head % n_kv_head != 0) {
        if (ith == 0 && g_slha_tile_store.read(layer->layer_id, 0)) {
            std::cerr << "[SLHA] shadow score: bad GQA geometry n_head=" << n_head
                      << " n_kv_head=" << n_kv_head << "\n";
        }
        return;
    }
    const int64_t gqa_factor = n_head / n_kv_head;

    {
        std::lock_guard<std::mutex> lock(*layer->collect_mutex);
        if (!layer->model_handle) {
            auto & state = get_global_state();
            slha_load_layer_model(layer, state.weights_dir);
        }
        if (layer->score_mode != SLHA_SCORE_OFF && !layer->shadow_metrics) {
            layer->shadow_metrics = std::make_unique<slha_shadow_metrics>();
        }
    }

    if (!layer->model_handle || !layer->shadow_metrics) {
        return;
    }

    const size_t d = static_cast<size_t>(layer->n_embd_gqa);
    SlhaModel * model = static_cast<SlhaModel *>(layer->model_handle);

    thread_local std::vector<float> q_extended;
    thread_local std::vector<float> q_coarse;
    thread_local std::vector<uint64_t> q_sign;
    thread_local std::vector<float> scores;
    q_coarse.resize(SLHA_D_C);
    q_sign.resize(SLHA_RESIDUAL_WORDS);

    const float * kq_data = static_cast<const float *>(kq->data);
    const float * q_data  = static_cast<const float *>(q->data);

    const size_t q_token_stride = q->nb[1] / sizeof(float);
    const size_t q_head_stride  = q->nb[2] / sizeof(float);
    const size_t q_stream_stride = q->nb[3] / sizeof(float);
    const size_t kq_token_stride = kq->nb[1] / sizeof(float);
    const size_t kq_head_stride  = kq->nb[2] / sizeof(float);
    const size_t kq_stream_stride = kq->nb[3] / sizeof(float);

    const int64_t total = n_stream * n_token * n_head;
    const int64_t per_thread = (total + nth - 1) / nth;
    const int64_t start = ith * per_thread;
    const int64_t end = std::min(start + per_thread, total);

    for (int64_t idx = start; idx < end; ++idx) {
        const int64_t s = idx / (n_token * n_head);
        const int64_t r = idx % (n_token * n_head);
        const int64_t t = r / n_head;
        const int64_t h = r % n_head;
        const int64_t kv_head = h / gqa_factor;
        q_extended.assign(d, 0.0f);
        const float * q_head = q_data + s * q_stream_stride + t * q_token_stride + h * q_head_stride;
        const size_t slot_start = static_cast<size_t>(kv_head * head_dim);
        for (int64_t i = 0; i < head_dim; ++i) {
            q_extended[slot_start + i] = q_head[i];
        }

        int rc = slha_prepare_query(
            model,
            q_extended.data(),
            d,
            q_coarse.data(),
            q_sign.data()
        );
        if (rc != SLHA_OK) {
            std::cerr << "[SLHA] shadow score prepare_query failed: "
                      << slha_last_error_message() << "\n";
            continue;
        }

        // Assume positions are contiguous per stream in the global tile store.
        const size_t pos_offset = static_cast<size_t>(s * n_kv);
        const SciRustSlhaTile * tiles =
            static_cast<const SciRustSlhaTile *>(g_slha_tile_store.read(layer->layer_id, pos_offset));
        if (!tiles) {
            continue;
        }

        // Verify contiguous range is fully populated.
        bool contiguous = true;
        for (int64_t k = 0; k < n_kv; ++k) {
            if (!g_slha_tile_store.check_capacity(layer->layer_id, pos_offset + static_cast<size_t>(k)) ||
                !g_slha_tile_store.read(layer->layer_id, pos_offset + static_cast<size_t>(k))) {
                contiguous = false;
                break;
            }
        }
        if (!contiguous) {
            continue;
        }

        scores.resize(static_cast<size_t>(n_kv));
        rc = slha_score_tiles(
            model,
            tiles,
            static_cast<size_t>(n_kv),
            q_coarse.data(),
            q_sign.data(),
            scores.data()
        );
        if (rc != SLHA_OK) {
            std::cerr << "[SLHA] shadow score score_tiles failed: "
                      << slha_last_error_message() << "\n";
            continue;
        }

        const float * kq_head = kq_data + s * kq_stream_stride + t * kq_token_stride + h * kq_head_stride;
        std::vector<double> baseline_vec(static_cast<size_t>(n_kv));
        std::vector<double> slha_vec(static_cast<size_t>(n_kv));
        for (int64_t k = 0; k < n_kv; ++k) {
            baseline_vec[k] = static_cast<double>(kq_head[k]);
            slha_vec[k] = static_cast<double>(scores[k]);
            layer->shadow_metrics->add_sample(kq_head[k], scores[k]);
        }
        layer->shadow_metrics->add_vector(baseline_vec, slha_vec);
    }
}

void slha_k_transform_with_idxs(
    ggml_tensor * dst,
    const ggml_tensor * a,
    const ggml_tensor * b,
    int ith,
    int nth,
    void * userdata
) {
    auto * layer = static_cast<slha_layer_state *>(userdata);
    if (!layer) {
        return;
    }

    if (layer->kv_mode == SLHA_KV_OFF) {
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

    if (layer->kv_mode == SLHA_KV_PASSTHROUGH) {
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

    if (layer->kv_mode == SLHA_KV_COLLECT) {
        // Collect mode: gather K vectors for training.
        // All threads collect their assigned tokens with synchronization.
        
        // Update dimension on first call (any thread can do this).
        if (layer->n_embd_gqa == 0) {
            std::lock_guard<std::mutex> lock(*layer->collect_mutex);
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
            std::lock_guard<std::mutex> lock(*layer->collect_mutex);
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

    if (layer->kv_mode == SLHA_KV_TILESTORE) {
        // Copy K through unchanged; side-store the compressed tile for each token.
        const size_t row_bytes = ggml_row_size(a->type, n_embd_gqa);
        const uint8_t * src_base = static_cast<const uint8_t *>(a->data);
        uint8_t * dst_base = static_cast<uint8_t *>(dst->data);
        for (int64_t t = token_start; t < token_end; ++t) {
            std::memcpy(dst_base + t * dst->nb[1], src_base + t * a->nb[1], row_bytes);
        }

        if (a->type != GGML_TYPE_F32) {
            if (ith == 0) {
                std::cerr << "[SLHA] WARNING: layer " << layer->layer_id
                          << " tilestore requires F32 input, got type " << a->type
                          << "; skipping tile store\n";
            }
            return;
        }

        if (!b || b->type != GGML_TYPE_I64 || b->ne[0] < n_tokens) {
            if (ith == 0) {
                std::cerr << "[SLHA] WARNING: layer " << layer->layer_id
                          << " tilestore missing/invalid k_idxs\n";
            }
            return;
        }

        {
            std::lock_guard<std::mutex> lock(*layer->collect_mutex);
            if (layer->n_embd_gqa == 0) {
                layer->n_embd_gqa = n_embd_gqa;
                std::cout << "[SLHA] layer " << layer->layer_id
                          << " tilestore dim=" << n_embd_gqa << "\n";
            }
            if (!layer->model_handle) {
                auto & state = get_global_state();
                slha_load_layer_model(layer, state.weights_dir);
            }
            if (layer->score_mode != SLHA_SCORE_OFF && !layer->shadow_metrics) {
                layer->shadow_metrics = std::make_unique<slha_shadow_metrics>();
            }
        }

        if (layer->kv_mode == SLHA_KV_OFF || !layer->model_handle) {
            return;
        }

        SlhaModel * model = static_cast<SlhaModel *>(layer->model_handle);
        const int32_t codec = slha_codec_from_env();
        const size_t d = static_cast<size_t>(n_embd_gqa);
        const int64_t * idxs = static_cast<const int64_t *>(b->data);

        for (int64_t t = token_start; t < token_end; ++t) {
            const float * src_row = reinterpret_cast<const float *>(src_base + t * a->nb[1]);
            const int64_t pos = idxs[t];
            if (pos < 0) {
                std::cerr << "[SLHA] layer " << layer->layer_id
                          << " token " << t << " invalid position " << pos << "\n";
                continue;
            }
            alignas(SLHA_TILE_ALIGN) SciRustSlhaTile tile;
            int rc = slha_encode_key(model, src_row, d, static_cast<uint32_t>(pos), codec, &tile);
            if (rc != SLHA_OK) {
                std::cerr << "[SLHA] layer " << layer->layer_id
                          << " token " << t << " encode failed: "
                          << slha_last_error_message() << "\n";
                continue;
            }
            if (!g_slha_tile_store.write(layer->layer_id, static_cast<size_t>(pos), &tile)) {
                std::cerr << "[SLHA] layer " << layer->layer_id
                          << " position " << pos << " tile store overflow (capacity="
                          << g_slha_tile_store.capacity << ")\n";
            }
        }
        return;
    }

    if (layer->kv_mode == SLHA_KV_ROUNDTRIP) {
        // First call: set the expected dimension and load the per-layer projection.
        // Hold the per-layer mutex so that other threads do not observe a set
        // dimension before the model handle is installed.
        {
            std::lock_guard<std::mutex> lock(*layer->collect_mutex);
            if (layer->n_embd_gqa == 0) {
                layer->n_embd_gqa = n_embd_gqa;
                std::cout << "[SLHA] layer " << layer->layer_id
                          << " roundtrip dim=" << n_embd_gqa << "\n";
            }
            if (!layer->model_handle) {
                auto & state = get_global_state();
                slha_load_layer_model(layer, state.weights_dir);
            }
        }

        if (layer->kv_mode == SLHA_KV_OFF) {
            // Model loading failed: copy through unchanged.
            const size_t row_bytes = ggml_row_size(a->type, n_embd_gqa);
            const uint8_t * src = static_cast<const uint8_t *>(a->data);
            uint8_t * dst_data = static_cast<uint8_t *>(dst->data);
            for (int64_t t = token_start; t < token_end; ++t) {
                std::memcpy(dst_data + t * dst->nb[1], src + t * a->nb[1], row_bytes);
            }
            return;
        }

        if (a->type != GGML_TYPE_F32) {
            const size_t row_bytes = ggml_row_size(a->type, n_embd_gqa);
            const uint8_t * src = static_cast<const uint8_t *>(a->data);
            uint8_t * dst_data = static_cast<uint8_t *>(dst->data);
            for (int64_t t = token_start; t < token_end; ++t) {
                std::memcpy(dst_data + t * dst->nb[1], src + t * a->nb[1], row_bytes);
            }
            if (ith == 0) {
                std::cerr << "[SLHA] WARNING: layer " << layer->layer_id
                          << " roundtrip requires F32 input, got type " << a->type
                          << "; passing through\n";
            }
            return;
        }

        SlhaModel * model = static_cast<SlhaModel *>(layer->model_handle);
        if (!model) {
            const size_t row_bytes = ggml_row_size(a->type, n_embd_gqa);
            const uint8_t * src = static_cast<const uint8_t *>(a->data);
            uint8_t * dst_data = static_cast<uint8_t *>(dst->data);
            for (int64_t t = token_start; t < token_end; ++t) {
                std::memcpy(dst_data + t * dst->nb[1], src + t * a->nb[1], row_bytes);
            }
            if (ith == 0) {
                std::cerr << "[SLHA] WARNING: layer " << layer->layer_id
                          << " has no model; passing through\n";
            }
            return;
        }

        const int32_t codec = slha_codec_from_env();
        const size_t d = static_cast<size_t>(n_embd_gqa);
        const size_t row_bytes = ggml_row_size(a->type, n_embd_gqa);
        const size_t src_stride = a->nb[1];
        const size_t dst_stride = dst->nb[1];

        thread_local std::vector<float> in_row;
        thread_local std::vector<float> out_row;
        in_row.resize(d);
        out_row.resize(d);

        const uint8_t * src_base = static_cast<const uint8_t *>(a->data);
        uint8_t * dst_base = static_cast<uint8_t *>(dst->data);

        for (int64_t t = token_start; t < token_end; ++t) {
            const float * src_ptr;
            float * dst_ptr;
            if (src_stride == row_bytes && dst_stride == row_bytes) {
                src_ptr = reinterpret_cast<const float *>(src_base + t * src_stride);
                dst_ptr = reinterpret_cast<float *>(dst_base + t * dst_stride);
            } else {
                const uint8_t * src_row = src_base + t * src_stride;
                for (size_t i = 0; i < d; ++i) {
                    in_row[i] = reinterpret_cast<const float *>(src_row)[i];
                }
                src_ptr = in_row.data();
                dst_ptr = out_row.data();
            }

            alignas(SLHA_TILE_ALIGN) SciRustSlhaTile tile;
            int rc = slha_encode_key(model, src_ptr, d, static_cast<uint32_t>(t), codec, &tile);
            if (rc != SLHA_OK) {
                std::cerr << "[SLHA] layer " << layer->layer_id
                          << " token " << t << " encode failed: "
                          << slha_last_error_message() << "; passing through\n";
                std::memcpy(dst_ptr, src_ptr, row_bytes);
                continue;
            }

            rc = slha_decode_key(model, &tile, dst_ptr, d);
            if (rc != SLHA_OK) {
                std::cerr << "[SLHA] layer " << layer->layer_id
                          << " token " << t << " decode failed: "
                          << slha_last_error_message() << "; passing through\n";
                std::memcpy(dst_ptr, src_ptr, row_bytes);
                continue;
            }

            if (dst_stride != row_bytes) {
                uint8_t * dst_row = dst_base + t * dst_stride;
                for (size_t i = 0; i < d; ++i) {
                    reinterpret_cast<float *>(dst_row)[i] = out_row[i];
                }
            }
        }
        return;
    }
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
            std::lock_guard<std::mutex> lock(*layer.collect_mutex);
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

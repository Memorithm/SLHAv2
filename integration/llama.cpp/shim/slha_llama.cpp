#include "slha_llama.hpp"
#include "slha_replace_counters.hpp"
#include "slha_layer_mask.hpp"
#include "slha_score_scale.hpp"
#include "slha_scale_fit.hpp"
#include "slha_score_oracle.hpp"
#include "slha_oracle_metrics.hpp"
#include "slha_rank_dataset.hpp"

#include "slha.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "ggml.h"

slha_tile_store g_slha_tile_store;
std::atomic<size_t> g_slha_tiles_written[SLHA_MAX_LAYERS];

// --------------------------------------------------------------------------
// Score-replacement layer mask (SLHA_SCORE_LAYERS). Selected layers replace
// their attention logits with SLHA scores; unselected layers pass baseline
// Q*K through unchanged. Resolution is deferred to the first op execution,
// once the model's layer count is known (observed during graph build).
// --------------------------------------------------------------------------
static slha_mask::ParsedMask g_score_mask;      // syntactic parse at init
static bool g_score_mask_syntax_ok = true;      // false on a malformed spec
static bool g_score_mask_resolved = false;      // resolved against layer count
static bool g_score_mask_error = false;         // fail-closed flag
static std::mutex g_score_mask_mutex;
static std::atomic<int32_t> g_observed_num_layers{0};

// Unselected pass-through accounting (kept separate from the strict replace
// counters so selected-layer coverage is reported on its own).
static std::atomic<size_t> g_unselected_callbacks{0};
static std::atomic<size_t> g_unselected_vectors{0};
static std::atomic<size_t> g_unselected_logits{0};

// --------------------------------------------------------------------------
// Experimental per-layer score scaling (SLHA_SCORE_SCALE / _FILE). A positive
// scalar a_layer multiplies each SLHA score at the raw-Q*K replacement seam,
// which is exactly a per-layer softmax-temperature correction. Default 1.0
// (unchanged production behaviour). Fail-closed: an invalid spec marks the
// replace run invalid. See slha_score_scale.hpp. Kept separate from the strict
// replace counters (like the layer mask) so those guarantees are not perturbed.
// --------------------------------------------------------------------------
static slha_scale::ScaleMap g_score_scale;    // resolved scale map (default global 1.0)
static bool g_score_scale_active = false;     // an explicit scale spec was provided
static bool g_score_scale_resolved = false;   // resolved against the selected set
static std::string g_score_scale_env_spec;    // raw SLHA_SCORE_SCALE (re-parsed at resolve)
static std::string g_score_scale_file_json;   // raw SLHA_SCORE_SCALE_FILE contents
static std::atomic<size_t> g_scaled_vectors{0};   // vectors where a != 1.0 was applied
static std::atomic<size_t> g_scaled_logits{0};    // logits multiplied by a != 1.0
static std::atomic<size_t> g_scale_invalid{0};    // vectors that went non-finite after scaling

// --------------------------------------------------------------------------
// Experimental diagnostic score oracles (SLHA_SCORE_ORACLE). These rewrite an
// attention row from the PAIRED baseline and SLHA rows to separate ranking
// errors from order-preserving score-shape errors. They require the baseline
// Q*K row and are therefore diagnostic instruments, never deployable.
// --------------------------------------------------------------------------
static slha_oracle::Config g_oracle;
static std::atomic<size_t> g_oracle_vectors{0};
static std::atomic<size_t> g_oracle_logits{0};
static std::atomic<size_t> g_oracle_permutations{0};
static std::atomic<size_t> g_oracle_ties{0};
static std::atomic<size_t> g_oracle_nonfinite_input{0};
static std::atomic<size_t> g_oracle_invalid_permutation{0};
static std::atomic<size_t> g_oracle_partial_write{0};
// Runtime invariant sampling: a bounded number of rows are re-checked against
// the mode's declared rank / multiset guarantees.
static std::atomic<size_t> g_oracle_invariant_checked{0};
static std::atomic<size_t> g_oracle_invariant_failed{0};

static void slha_score_oracle_reset() {
    g_oracle_vectors.store(0, std::memory_order_relaxed);
    g_oracle_logits.store(0, std::memory_order_relaxed);
    g_oracle_permutations.store(0, std::memory_order_relaxed);
    g_oracle_ties.store(0, std::memory_order_relaxed);
    g_oracle_nonfinite_input.store(0, std::memory_order_relaxed);
    g_oracle_invalid_permutation.store(0, std::memory_order_relaxed);
    g_oracle_partial_write.store(0, std::memory_order_relaxed);
    g_oracle_invariant_checked.store(0, std::memory_order_relaxed);
    g_oracle_invariant_failed.store(0, std::memory_order_relaxed);
}

static void slha_score_scale_reset() {
    g_score_scale_resolved = false;
    g_scaled_vectors.store(0, std::memory_order_relaxed);
    g_scaled_logits.store(0, std::memory_order_relaxed);
    g_scale_invalid.store(0, std::memory_order_relaxed);
}

static void slha_score_mask_reset() {
    std::lock_guard<std::mutex> lock(g_score_mask_mutex);
    g_score_mask_resolved = false;
    g_score_mask_error = !g_score_mask_syntax_ok;
    g_unselected_callbacks.store(0, std::memory_order_relaxed);
    g_unselected_vectors.store(0, std::memory_order_relaxed);
    g_unselected_logits.store(0, std::memory_order_relaxed);
    slha_score_scale_reset();
    slha_score_oracle_reset();
}

// Resolve the mask against the observed model layer count, once. Sets the
// fail-closed error flag on any invalid specification.
static void slha_resolve_score_mask_once() {
    std::lock_guard<std::mutex> lock(g_score_mask_mutex);
    if (g_score_mask_resolved) return;
    g_score_mask_resolved = true;
    const int n = g_observed_num_layers.load(std::memory_order_acquire);
    if (!g_score_mask_syntax_ok) {
        g_score_mask_error = true;
    } else if (n > 0) {
        slha_mask::ParsedMask resolved;
        if (!slha_mask::parse_layer_mask(g_score_mask.spec, n, resolved)) {
            g_score_mask_error = true;
            std::cerr << "[SLHA] score layer mask invalid against " << n
                      << " layers: " << resolved.error << "\n";
        } else {
            g_score_mask = resolved;
        }
    }
    // Resolve the experimental score scale: re-parse against the now-known layer
    // count (catches out-of-range layer ids) and require that a PerLayer scale
    // covers every selected layer (no silent 1.0 fallback).
    if (!g_score_scale_resolved) {
        g_score_scale_resolved = true;
        if (g_score_scale_active && !g_score_mask_error && n > 0) {
            bool ok = slha_scale::parse_score_scale(
                g_score_scale_env_spec, g_score_scale_file_json, n, g_score_scale);
            if (ok) {
                const std::vector<int32_t> selected = g_score_mask.resolved_ids(n);
                ok = slha_scale::resolve_against_selected(g_score_scale, selected);
            }
            if (!ok) {
                std::cerr << "[SLHA] score scale invalid against " << n
                          << " layers / selected set: " << g_score_scale.error
                          << " (run will be marked invalid)\n";
                g_score_mask_error = true;  // reuse the fail-closed path
            }
        } else if (g_score_scale_active && !g_score_scale.valid) {
            g_score_mask_error = true;      // unparseable spec -> fail closed
        }
    }

    if (g_oracle.active && !g_oracle.valid) {
        g_score_mask_error = true;          // unknown oracle mode -> fail closed
    }

    if (g_score_mask_error) {
        // Fail closed: mark the strict replace run invalid.
        g_slha_replace_counters.error_code.store(1, std::memory_order_release);
    }
}

// Whether a given layer participates in direct score replacement.
static bool slha_layer_selected(int32_t layer_id) {
    if (g_score_mask_error) return false;
    return g_score_mask.contains(layer_id);
}

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

void slha_print_replace_summary() {
    g_slha_replace_counters.print_summary();
    g_slha_replace_counters.reset();
}

void slha_print_score_mask_summary() {
    const int n = g_observed_num_layers.load(std::memory_order_acquire);

    // Requested mask (verbatim spec), resolved id set, and the ids where
    // replacement actually executed (per-layer replaced-vector count > 0).
    std::vector<int32_t> executed;
    for (int32_t id = 0; id < n && id < SLHA_MAX_LAYERS; ++id) {
        if (slha_replace_get_layer_success(id) > 0) executed.push_back(id);
    }
    std::vector<int32_t> resolved = g_score_mask_error
        ? std::vector<int32_t>{}
        : g_score_mask.resolved_ids(n);

    auto join = [](const std::vector<int32_t> & v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += std::to_string(v[i]); }
        return s.empty() ? std::string("(none)") : s;
    };

    const bool mask_valid = !g_score_mask_error && (executed == resolved);

    std::cout << "\nSLHA_SCORE_MASK_SUMMARY\n";
    std::cout << "requested_spec=" << (g_score_mask.spec.empty() ? "(empty)" : g_score_mask.spec) << "\n";
    std::cout << "observed_num_layers=" << n << "\n";
    std::cout << "resolved_layers=" << join(resolved) << "\n";
    std::cout << "resolved_count=" << resolved.size() << "\n";
    std::cout << "executed_layers=" << join(executed) << "\n";
    std::cout << "executed_count=" << executed.size() << "\n";
    std::cout << "unselected_callbacks=" << g_unselected_callbacks.load() << "\n";
    std::cout << "unselected_vectors=" << g_unselected_vectors.load() << "\n";
    std::cout << "unselected_logits=" << g_unselected_logits.load() << "\n";
    std::cout << "mask_error=" << (g_score_mask_error ? "true" : "false") << "\n";
    std::cout << "mask_valid=" << (mask_valid ? "true" : "false") << "\n";
    // Per-executed-layer replaced-vector coverage.
    for (int32_t id : executed) {
        std::cout << "layer_" << id << "_replaced_vectors=" << slha_replace_get_layer_success(id)
                  << " layer_" << id << "_failed_vectors=" << slha_replace_get_layer_fail(id) << "\n";
    }
    std::cout << "END_SLHA_SCORE_MASK_SUMMARY\n";
    std::cout.flush();
}

void slha_print_score_scale_summary() {
    if (!g_score_scale_active) {
        return;  // default 1.0 everywhere: nothing to report
    }
    const int n = g_observed_num_layers.load(std::memory_order_acquire);
    // scale_manifest_valid: the scale parsed/resolved cleanly and the run was
    // not otherwise failed closed.
    const bool scale_valid = g_score_scale.valid && !g_score_mask_error;

    std::cout << "\nSLHA_SCORE_SCALE_SUMMARY\n";
    std::cout << "active=true\n";
    std::cout << "source=" << (g_score_scale.source.empty() ? "(none)" : g_score_scale.source) << "\n";
    std::cout << "mode=" << (g_score_scale.mode == slha_scale::Mode::Global ? "global" : "per_layer") << "\n";
    std::cout << "requested_spec=" << (g_score_scale.spec.empty() ? "(empty)" : g_score_scale.spec) << "\n";
    std::cout << "canonical=" << g_score_scale.canonical() << "\n";
    std::cout << "manifest_sha256=" << g_score_scale.manifest_sha256 << "\n";
    if (g_score_scale.mode == slha_scale::Mode::Global) {
        std::cout << "global_scale=" << g_score_scale.global << "\n";
    }
    // Resolved scale on the layers where replacement actually executed.
    for (int32_t id = 0; id < n && id < SLHA_MAX_LAYERS; ++id) {
        if (slha_replace_get_layer_success(id) > 0) {
            std::cout << "layer_" << id << "_scale=" << g_score_scale.get(id) << "\n";
        }
    }
    std::cout << "scaled_vectors=" << g_scaled_vectors.load() << "\n";
    std::cout << "scaled_logits=" << g_scaled_logits.load() << "\n";
    std::cout << "invalid_scale=" << g_scale_invalid.load() << "\n";
    std::cout << "scale_manifest_valid=" << (scale_valid ? "true" : "false") << "\n";
    std::cout << "END_SLHA_SCORE_SCALE_SUMMARY\n";
    std::cout.flush();
}

void slha_print_score_oracle_summary() {
    if (!g_oracle.active) {
        return;   // default: no oracle, nothing to report
    }
    const bool mode_valid = g_oracle.valid && !g_score_mask_error;
    std::cout << "\nSLHA_SCORE_ORACLE_SUMMARY\n";
    std::cout << "active=true\n";
    std::cout << "requested_spec=" << (g_oracle.spec.empty() ? "(empty)" : g_oracle.spec) << "\n";
    std::cout << "mode=" << slha_oracle::mode_name(g_oracle.mode) << "\n";
    std::cout << "canonical=" << g_oracle.canonical << "\n";
    std::cout << "topk=" << g_oracle.topk << "\n";
    std::cout << "config_sha256=" << g_oracle.config_sha256 << "\n";
    std::cout << "oracle_vectors=" << g_oracle_vectors.load() << "\n";
    std::cout << "oracle_logits=" << g_oracle_logits.load() << "\n";
    std::cout << "oracle_permutations=" << g_oracle_permutations.load() << "\n";
    std::cout << "oracle_ties=" << g_oracle_ties.load() << "\n";
    std::cout << "oracle_nonfinite_input=" << g_oracle_nonfinite_input.load() << "\n";
    std::cout << "oracle_invalid_permutation=" << g_oracle_invalid_permutation.load() << "\n";
    std::cout << "oracle_partial_write=" << g_oracle_partial_write.load() << "\n";
    std::cout << "oracle_invariant_checked=" << g_oracle_invariant_checked.load() << "\n";
    std::cout << "oracle_invariant_failed=" << g_oracle_invariant_failed.load() << "\n";
    std::cout << "oracle_mode_valid=" << (mode_valid ? "true" : "false") << "\n";
    std::cout << "END_SLHA_SCORE_ORACLE_SUMMARY\n";
    std::cout.flush();
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
        // Re-initializing with a new context: reset tile store and counters
        for (size_t i = 0; i < SLHA_MAX_LAYERS; ++i) {
            g_slha_tiles_written[i].store(0, std::memory_order_relaxed);
        }
        if (g_slha_tile_store.n_layers > 0) {
            g_slha_tile_store.reset();
        }
        if (state.score_mode == SLHA_SCORE_REPLACE) {
            g_slha_replace_counters.reset();
        }
        slha_score_mask_reset();
        return 0;
    }

    state.kv_mode = mode;
    state.score_mode = slha_score_mode_from_env();

    // Parse the score-replacement layer mask (default "all"). Syntax errors are
    // caught now; range validation is deferred until the layer count is known.
    {
        const char * lm = std::getenv("SLHA_SCORE_LAYERS");
        const std::string spec = lm ? std::string(lm) : std::string("all");
        g_score_mask_syntax_ok = slha_mask::parse_layer_mask(spec, 0, g_score_mask);
        if (!g_score_mask_syntax_ok && state.score_mode == SLHA_SCORE_REPLACE) {
            std::cerr << "[SLHA] invalid SLHA_SCORE_LAYERS='" << spec
                      << "': " << g_score_mask.error << " (run will be marked invalid)\n";
        }
        slha_score_mask_reset();
    }

    // Parse the experimental per-layer score scale (default global 1.0). A file
    // spec (SLHA_SCORE_SCALE_FILE) takes precedence over the inline env spec
    // (SLHA_SCORE_SCALE). Syntax/positivity is validated now; layer-id range and
    // selected-set coverage are (re)validated once the layer count is known.
    {
        g_score_scale_env_spec.clear();
        g_score_scale_file_json.clear();
        const char * sc = std::getenv("SLHA_SCORE_SCALE");
        const char * scf = std::getenv("SLHA_SCORE_SCALE_FILE");
        if (scf && scf[0]) {
            std::ifstream f(scf);
            if (!f) {
                std::cerr << "[SLHA] cannot open SLHA_SCORE_SCALE_FILE='" << scf << "'\n";
                g_score_scale_active = true;              // fail closed: active but unparseable
                g_score_scale.valid = false;
                g_score_scale.error = "cannot open scale file";
                g_score_scale.source = "file";
            } else {
                std::stringstream ss;
                ss << f.rdbuf();
                g_score_scale_file_json = ss.str();
                g_score_scale_active = true;
            }
        } else if (sc && sc[0]) {
            g_score_scale_env_spec = sc;
            g_score_scale_active = true;
        }

        if (g_score_scale_active && (!g_score_scale_file_json.empty() || !g_score_scale_env_spec.empty())) {
            // Early syntactic parse (num_layers=0 defers the id-range check).
            slha_scale::parse_score_scale(g_score_scale_env_spec, g_score_scale_file_json,
                                          0, g_score_scale);
        }
        if (g_score_scale_active && !g_score_scale.valid && state.score_mode == SLHA_SCORE_REPLACE) {
            std::cerr << "[SLHA] invalid score scale: " << g_score_scale.error
                      << " (run will be marked invalid)\n";
        }
        slha_score_scale_reset();
    }

    // Experimental diagnostic score oracle (default off). Strict parsing: an
    // unknown mode is rejected and marks the run invalid rather than silently
    // falling back.
    {
        const char * orc = std::getenv("SLHA_SCORE_ORACLE");
        const std::string spec = orc ? std::string(orc) : std::string();
        slha_oracle::parse_oracle(spec, g_oracle);
        if (g_oracle.active && !g_oracle.valid && state.score_mode == SLHA_SCORE_REPLACE) {
            std::cerr << "[SLHA] invalid SLHA_SCORE_ORACLE='" << spec << "': "
                      << g_oracle.error << " (run will be marked invalid)\n";
        }
        if (g_oracle.active && g_oracle.valid) {
            std::cout << "[SLHA] score oracle enabled: " << g_oracle.canonical << "\n";
        }
        slha_score_oracle_reset();
    }

    // Active-key ranking / tie statistics (diagnostic). Active only when
    // SLHA_ORACLE_METRICS_JSON names an output path.
    {
        const char * mp = std::getenv("SLHA_ORACLE_METRICS_JSON");
        if (mp && mp[0]) {
            slha_oracle_metrics::reset();
            slha_oracle_metrics::enable(true);
            std::cout << "[SLHA] active-key rank metrics enabled -> " << mp << "\n";
        } else {
            slha_oracle_metrics::enable(false);
        }
    }

    // Ranking-training dataset collection. OFFLINE ONLY: enabling this makes the
    // run record exact baseline logits as training labels, so it is never part
    // of a deployable measurement. Inert unless SLHA_RANK_DATASET_DIR is set.
    {
        const char * rd = std::getenv("SLHA_RANK_DATASET_DIR");
        if (rd && rd[0]) {
            slha_rank_dataset::enable(rd);
            std::cout << "[SLHA] rank-training dataset collection enabled -> " << rd
                      << " (stride=" << slha_rank_dataset::sampling().token_stride
                      << ", heads<" << slha_rank_dataset::sampling().max_heads << ")\n";
        } else {
            slha_rank_dataset::enable(nullptr);
        }
    }

    // Offline per-layer score-scale fitting (shadow diagnostic). Active only when
    // SLHA_SCALE_FIT_JSON names an output path; the JSON is written at shutdown.
    {
        const char * fit = std::getenv("SLHA_SCALE_FIT_JSON");
        if (fit && fit[0]) {
            slha_scale_fit::reset();
            slha_scale_fit::enable(true);
            std::cout << "[SLHA] score-scale fit enabled -> " << fit << "\n";
        } else {
            slha_scale_fit::enable(false);
        }
    }

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

        // Reset replace counters and tile-write tracking
        if (state.score_mode == SLHA_SCORE_REPLACE) {
            g_slha_replace_counters.reset();
        }
        for (size_t i = 0; i < SLHA_MAX_LAYERS; ++i) {
            g_slha_tiles_written[i].store(0, std::memory_order_relaxed);
        }

        // Initialize the compressed K side store.
        if (mode == SLHA_KV_TILESTORE || state.score_mode != SLHA_SCORE_OFF) {
            const size_t n_layers = 128;
            const size_t capacity = 16384; // max positions per layer (handles 12×512 chunks + margin)
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

void slha_tile_store::clear_layer(int32_t layer_id) {
    if (layer_id < 0) return;
    const size_t layer = static_cast<size_t>(layer_id);
    std::lock_guard<std::mutex> lock(mutex);
    if (layer >= n_layers || capacity == 0 || tile_bytes == 0) return;
    const size_t valid_base = layer * capacity;
    std::memset(&valid[valid_base], 0, capacity);
    const size_t tile_byte_offset = layer * capacity * tile_bytes;
    std::memset(&tiles[tile_byte_offset], 0, capacity * tile_bytes);
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

// Diagnostic counters (defined before slha_k_clear_all which references them)
static size_t g_diag_call_count = 0;
static std::mutex g_diag_mutex;

void slha_k_clear_all() {
    // The KV cache is about to be cleared and positions restart at 0, so the raw
    // key matrix collected for the ranking dataset must be sealed into the
    // current chunk BEFORE it is overwritten. Rows recorded after this point
    // belong to the next chunk.
    if (slha_rank_dataset::enabled()) {
        auto & state = get_global_state();
        for (auto & layer : state.layers) {
            std::lock_guard<std::mutex> lock(*layer.collect_mutex);
            if (layer.raw_k_max_pos > 0 && layer.n_embd_gqa > 0) {
                slha_rank_dataset::add_keys(layer.layer_id, layer.raw_k_by_pos.data(),
                                            layer.raw_k_max_pos,
                                            static_cast<size_t>(layer.n_embd_gqa));
            }
            layer.raw_k_max_pos = 0;
            std::fill(layer.raw_k_by_pos.begin(), layer.raw_k_by_pos.end(), 0.0f);
        }
        slha_rank_dataset::begin_chunk();
    }
    for (size_t i = 0; i < SLHA_MAX_LAYERS; ++i) {
        g_slha_tiles_written[i].store(0, std::memory_order_relaxed);
    }
    if (g_slha_tile_store.n_layers > 0) {
        g_slha_tile_store.reset();
    }
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

    if (state.score_mode == SLHA_SCORE_REPLACE) {
        // Mask summary first: it reads per-layer counters that the replace
        // summary clears on reset. The scale summary reads the same per-layer
        // success counts, so it must also print before the replace summary.
        slha_print_score_mask_summary();
        slha_print_score_scale_summary();
        slha_print_score_oracle_summary();
        slha_print_replace_summary();
    }

    // Write the active-key ranking / tie statistics if requested.
    if (slha_oracle_metrics::enabled()) {
        const char * mp = std::getenv("SLHA_ORACLE_METRICS_JSON");
        if (mp && mp[0]) {
            std::ofstream out(mp);
            if (out) {
                out << slha_oracle_metrics::dump_json();
                std::cout << "[SLHA] wrote active-key rank metrics to " << mp << "\n";
            } else {
                std::cerr << "[SLHA] failed to write SLHA_ORACLE_METRICS_JSON='" << mp << "'\n";
            }
        }
    }

    // Write the ranking-training dataset if collection was enabled.
    if (slha_rank_dataset::enabled()) {
        for (auto & layer : state.layers) {
            std::lock_guard<std::mutex> lock(*layer.collect_mutex);
            if (layer.raw_k_max_pos > 0 && layer.n_embd_gqa > 0) {
                slha_rank_dataset::add_keys(layer.layer_id, layer.raw_k_by_pos.data(),
                                            layer.raw_k_max_pos,
                                            static_cast<size_t>(layer.n_embd_gqa));
            }
        }
        std::string rd_err;
        if (slha_rank_dataset::flush(&rd_err)) {
            std::cout << "[SLHA] wrote rank-training dataset\n";
        } else {
            std::cerr << "[SLHA] rank-training dataset flush FAILED: " << rd_err << "\n";
        }
    }

    // Write the offline score-scale fit (shadow diagnostic) if requested.
    if (slha_scale_fit::enabled()) {
        const char * fit = std::getenv("SLHA_SCALE_FIT_JSON");
        if (fit && fit[0]) {
            std::ofstream out(fit);
            if (out) {
                out << slha_scale_fit::dump_json();
                std::cout << "[SLHA] wrote score-scale fit to " << fit << "\n";
            } else {
                std::cerr << "[SLHA] failed to write SLHA_SCALE_FIT_JSON='" << fit << "'\n";
            }
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
    // Track the model's true layer count from graph-build calls (which all
    // precede op execution) so the score mask can validate ranges.
    int32_t prev = g_observed_num_layers.load(std::memory_order_relaxed);
    while (il + 1 > prev &&
           !g_observed_num_layers.compare_exchange_weak(prev, il + 1,
                                                         std::memory_order_release,
                                                         std::memory_order_relaxed)) {
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

static void slha_diag_score(
    const float * dst_data,
    int64_t n_kv,
    int64_t n_token,
    int64_t n_head,
    int64_t n_stream,
    int32_t layer_id,
    size_t n_written
);


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

    // Common: check tensor types
    if (kq->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        if (layer->score_mode == SLHA_SCORE_REPLACE) {
            if (ith == 0) {
                g_slha_replace_counters.n_unsupported_shape.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                std::cerr << "[SLHA] replace: unsupported tensor type\n";
            }
        }
        return;
    }

    // Common: verify nb[0] == sizeof(float) for all tensors (unit K stride)
    if (kq->nb[0] != sizeof(float) || q->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        if (layer->score_mode == SLHA_SCORE_REPLACE) {
            if (ith == 0) {
                g_slha_replace_counters.n_unsupported_stride.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                std::cerr << "[SLHA] replace: unsupported nb[0] stride (kq="
                          << kq->nb[0] << " q=" << q->nb[0] << " dst=" << dst->nb[0]
                          << "); require sizeof(float)=" << sizeof(float) << "\n";
            }
        }
        return;
    }

    // This callback is inserted after build_attn_mha permutes q and k with
    // permute(0,2,1,3). After that, q is [head_dim, n_tokens, n_head, ...]
    // and kq is [n_kv, n_tokens, n_head, ...].
    const int64_t n_kv     = kq->ne[0];
    const int64_t n_token  = kq->ne[1];
    const int64_t n_head   = kq->ne[2];
    const int64_t n_stream = kq->ne[3];
    const int64_t head_dim = q->ne[0];

    // Validate q dimensions match kq
    if (q->ne[1] != n_token || q->ne[2] != n_head || q->ne[3] != n_stream) {
        if (layer->score_mode == SLHA_SCORE_REPLACE) {
            if (ith == 0) {
                g_slha_replace_counters.n_unsupported_shape.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                std::cerr << "[SLHA] replace: unsupported shape (q="
                          << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3]
                          << " kq=" << kq->ne[0] << "," << kq->ne[1] << "," << kq->ne[2] << "," << kq->ne[3]
                          << ")\n";
            }
        }
        return;
    }

    // Validate GQA dimensions
    if (layer->n_embd_gqa == 0 || layer->n_embd_gqa % head_dim != 0) {
        if (layer->score_mode == SLHA_SCORE_REPLACE) {
            if (ith == 0) {
                g_slha_replace_counters.n_unsupported_shape.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                std::cerr << "[SLHA] replace: unsupported dimensions n_embd_gqa="
                          << layer->n_embd_gqa << " head_dim=" << head_dim << "\n";
            }
        }
        return;
    }

    const int64_t n_kv_head = layer->n_embd_gqa / head_dim;
    if (n_kv_head == 0 || n_head % n_kv_head != 0) {
        if (layer->score_mode == SLHA_SCORE_REPLACE) {
            if (ith == 0) {
                g_slha_replace_counters.n_unsupported_shape.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                std::cerr << "[SLHA] replace: bad GQA geometry n_head=" << n_head
                          << " n_kv_head=" << n_kv_head << "\n";
            }
        }
        return;
    }
    const int64_t gqa_factor = n_head / n_kv_head;

    // ==================================================================
    // LAYER MASK (REPLACE mode) — unselected layers pass baseline through
    // ==================================================================
    if (layer->score_mode == SLHA_SCORE_REPLACE) {
        // Resolve the mask against the now-known layer count (once, first op).
        if (!g_score_mask_resolved) {
            slha_resolve_score_mask_once();
        }
        if (!slha_layer_selected(layer->layer_id)) {
            // Pass-through: copy baseline logits unchanged; count separately so
            // active coverage reflects only selected layers.
            const int64_t n = ggml_nelements(dst);
            const int64_t i0 = (n * ith) / nth;
            const int64_t i1 = (n * (ith + 1)) / nth;
            const float * src = static_cast<const float *>(kq->data);
            float * out = static_cast<float *>(dst->data);
            for (int64_t i = i0; i < i1; ++i) out[i] = src[i];
            if (ith == 0) {
                g_unselected_callbacks.fetch_add(1, std::memory_order_relaxed);
                g_unselected_vectors.fetch_add(
                    static_cast<size_t>(n_stream * n_token * n_head), std::memory_order_relaxed);
                g_unselected_logits.fetch_add(
                    static_cast<size_t>(n_stream * n_token * n_head) * static_cast<size_t>(n_kv),
                    std::memory_order_relaxed);
            }
            return;
        }
    }

    // ==================================================================
    // SHADOW MODE
    // ==================================================================
    if (layer->score_mode == SLHA_SCORE_SHADOW) {
        // Copy baseline logits element-wise so the attention graph is unchanged
        // regardless of tensor strides or shape.
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

        // Load model and init shadow metrics
        {
            std::lock_guard<std::mutex> lock(*layer->collect_mutex);
            if (!layer->model_handle) {
                auto & state = get_global_state();
                slha_load_layer_model(layer, state.weights_dir);
            }
            if (!layer->shadow_metrics) {
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

        // Offline score-scale fit (SLHA_SCALE_FIT_JSON). Accumulate this thread's
        // (baseline, slha) pairs over causally-unmasked positions into one local
        // accumulator, merged into the per-layer registry after the loop.
        const bool do_scale_fit = slha_scale_fit::enabled();
        slha_scale_fit::LayerAcc fit_acc;

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

            int rc = slha_prepare_query(model, q_extended.data(), d, q_coarse.data(), q_sign.data());
            if (rc != SLHA_OK) {
                std::cerr << "[SLHA] shadow score prepare_query failed: "
                          << slha_last_error_message() << "\n";
                continue;
            }

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
            rc = slha_score_tiles(model, tiles, static_cast<size_t>(n_kv), q_coarse.data(), q_sign.data(), scores.data());
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

            // Score-scale fit: accumulate (baseline, slha) over the causal region
            // only. Query token t (fresh-context chunk, position t) attends keys
            // 0..t; using k <= t is a strict subset of the truly-unmasked keys,
            // so no softmax-masked position ever enters the magnitude fit. It is
            // also clamped to n_written so no padding position (no tile data) can
            // contribute. Head id and query-position bucket are recorded so the
            // fit can be shown not to be dominated by one head or by short rows.
            if (do_scale_fit) {
                const size_t n_written_fit =
                    g_slha_tiles_written[layer->layer_id].load(std::memory_order_acquire);
                int64_t kmax = std::min<int64_t>(t + 1, n_kv);
                if (n_written_fit > 0 && static_cast<int64_t>(n_written_fit) < kmax) {
                    kmax = static_cast<int64_t>(n_written_fit);
                }
                const double t_frac = n_token > 0
                    ? static_cast<double>(t) / static_cast<double>(n_token)
                    : 0.0;
                for (int64_t k = 0; k < kmax; ++k) {
                    fit_acc.add(baseline_vec[k], slha_vec[k],
                                static_cast<int>(h), t_frac);
                }
                if (kmax > 0) {
                    fit_acc.n_vectors += 1;
                }
            }

            // Padding baseline audit: measure baseline values at positions
            // k >= n_written where no tile data exists.
            const size_t n_written_shadow = g_slha_tiles_written[layer->layer_id].load(std::memory_order_acquire);
            if (n_written_shadow > 0 && static_cast<size_t>(n_kv) > n_written_shadow) {
                double max_abs_pad = 0.0;
                size_t n_nonzero = 0;
                size_t n_nonfinite = 0;
                for (int64_t k = static_cast<int64_t>(n_written_shadow); k < n_kv; ++k) {
                    const double v = static_cast<double>(kq_head[k]);
                    if (!std::isfinite(v)) {
                        ++n_nonfinite;
                    } else if (std::abs(v) > 1e-12) {
                        ++n_nonzero;
                        max_abs_pad = std::max(max_abs_pad, std::abs(v));
                    }
                }
                if (n_nonzero > 0 || n_nonfinite > 0) {
                    std::cerr << "[SLHA] shadow padding audit: layer=" << layer->layer_id
                              << " n_kv=" << n_kv << " n_written=" << n_written_shadow
                              << " max_abs_pad=" << max_abs_pad
                              << " nonzero=" << n_nonzero
                              << " nonfinite=" << n_nonfinite << "\n";
                }
            }
        }
        // Merge even when no pair was accumulated, so the exclusion counters
        // (non-finite / near-zero) are never silently dropped. n_callbacks is
        // counted once per op invocation, not once per worker thread.
        if (do_scale_fit && (fit_acc.n > 0.0 || fit_acc.n_nonfinite > 0 ||
                             fit_acc.n_robust_excluded > 0)) {
            fit_acc.n_callbacks = (ith == 0) ? 1 : 0;
            slha_scale_fit::merge_layer(layer->layer_id, fit_acc);
        }
        return;
    }

    // ==================================================================
    // REPLACE MODE — fail-closed, no baseline fallback
    // ==================================================================

    g_slha_replace_counters.n_callbacks.fetch_add(1, std::memory_order_relaxed);

    // Record n_stream and enforce single-stream for current experiment
    if (ith == 0) {
        g_slha_replace_counters.n_stream.store(
            static_cast<size_t>(n_stream), std::memory_order_relaxed);
    }

    if (n_stream != 1) {
        if (ith == 0) {
            g_slha_replace_counters.n_unsupported_shape.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.error_code.store(1, std::memory_order_release);
            std::cerr << "[SLHA] replace: unsupported n_stream=" << n_stream
                      << "; only n_stream=1 is supported\n";
        }
        return;
    }

    // Record expected counts (thread 0 only)
    if (ith == 0) {
        const int64_t total_vectors = n_stream * n_token * n_head;
        g_slha_replace_counters.n_expected_vectors.fetch_add(
            static_cast<size_t>(total_vectors), std::memory_order_relaxed);
        g_slha_replace_counters.n_expected_logits.fetch_add(
            static_cast<size_t>(total_vectors) * static_cast<size_t>(n_kv), std::memory_order_relaxed);
    }

    // Load model
    {
        std::lock_guard<std::mutex> lock(*layer->collect_mutex);
        if (!layer->model_handle) {
            auto & state = get_global_state();
            slha_load_layer_model(layer, state.weights_dir);
        }
    }

    if (!layer->model_handle) {
        if (ith == 0) {
            g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.error_code.store(1, std::memory_order_release);
            std::cerr << "[SLHA] replace: missing layer model for layer "
                      << layer->layer_id << "\n";
        }
        return;
    }

    const size_t d = static_cast<size_t>(layer->n_embd_gqa);
    SlhaModel * model = static_cast<SlhaModel *>(layer->model_handle);

    // Thread-local working storage (never directly written to dst)
    thread_local std::vector<float> q_extended;
    thread_local std::vector<float> q_coarse;
    thread_local std::vector<uint64_t> q_sign;
    thread_local std::vector<float> temp_scores;
    q_extended.resize(d, 0.0f);
    q_coarse.resize(SLHA_D_C);
    q_sign.resize(SLHA_RESIDUAL_WORDS);
    temp_scores.resize(static_cast<size_t>(n_kv));

    const float * q_data  = static_cast<const float *>(q->data);
    // The diagnostic oracles need the PAIRED baseline row, which is exactly the
    // kq input this op replaces.
    const float * kq_data = static_cast<const float *>(kq->data);
    const size_t kq_token_stride  = kq->nb[1] / sizeof(float);
    const size_t kq_head_stride   = kq->nb[2] / sizeof(float);
    const size_t kq_stream_stride = kq->nb[3] / sizeof(float);

    const size_t q_token_stride  = q->nb[1] / sizeof(float);
    const size_t q_head_stride   = q->nb[2] / sizeof(float);
    const size_t q_stream_stride = q->nb[3] / sizeof(float);
    const size_t dst_token_stride  = dst->nb[1] / sizeof(float);
    const size_t dst_head_stride   = dst->nb[2] / sizeof(float);
    const size_t dst_stream_stride = dst->nb[3] / sizeof(float);

    const int64_t total = n_stream * n_token * n_head;
    const int64_t per_thread = (total + nth - 1) / nth;
    const int64_t start = ith * per_thread;
    const int64_t end = std::min(start + per_thread, total);

    // Load the number of tiles written for this layer. n_kv from kq->ne[0] is
    // padded to at least 256 by llama_kv_cache::get_n_kv, but only n_written
    // tiles actually exist. We only validate and score positions with tiles.
    const size_t n_written = g_slha_tiles_written[layer->layer_id].load(std::memory_order_acquire);

    for (int64_t idx = start; idx < end; ++idx) {
        const int64_t s = idx / (n_token * n_head);
        const int64_t r = idx % (n_token * n_head);
        const int64_t t = r / n_head;
        const int64_t h = r % n_head;
        const int64_t kv_head = h / gqa_factor;

        // Safety-initialize this vector's dst row to zero (defined memory).
        // Zeroing is done per vector, inside the same partition that writes the
        // scores, because the vector enumeration here (t-major, h-minor) does not
        // match dst's memory order (h-major). Zeroing by a flat element range
        // instead would let one thread blank rows another thread is concurrently
        // writing — the vector partition tiles dst exactly, so this is race-free
        // and still leaves padding positions [n_check, n_kv) at zero.
        float * dst_head = static_cast<float *>(dst->data)
            + s * static_cast<ptrdiff_t>(dst_stream_stride)
            + t * static_cast<ptrdiff_t>(dst_token_stride)
            + h * static_cast<ptrdiff_t>(dst_head_stride);
        std::fill(dst_head, dst_head + n_kv, 0.0f);

        // n_stream == 1 is enforced above, so s must be 0.
        // Defensively handle s > 0 as inactive stream (counted, not replaced).
        if (s > 0) {
            g_slha_replace_counters.inactive_stream_vectors.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.inactive_stream_logits.fetch_add(
                static_cast<size_t>(n_kv), std::memory_order_relaxed);
            continue;
        }

        // Build extended Q vector for this (stream, token, head)
        std::fill(q_extended.begin(), q_extended.end(), 0.0f);
        const float * q_head_ptr = q_data
            + s * static_cast<ptrdiff_t>(q_stream_stride)
            + t * static_cast<ptrdiff_t>(q_token_stride)
            + h * static_cast<ptrdiff_t>(q_head_stride);
        const size_t slot_start = static_cast<size_t>(kv_head * head_dim);
        for (int64_t i = 0; i < head_dim; ++i) {
            q_extended[slot_start + i] = q_head_ptr[i];
        }

        // Prepare query — failure is fatal for this vector
        int rc = slha_prepare_query(model, q_extended.data(), d, q_coarse.data(), q_sign.data());
        if (rc != SLHA_OK) {
            g_slha_replace_counters.n_query_prep_fail.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.error_code.store(1, std::memory_order_release);
            if (ith == 0) {
                std::cerr << "[SLHA] replace layer " << layer->layer_id
                          << " query prepare failed: " << slha_last_error_message() << "\n";
            }
            continue;
        }

        // Only check tile positions that have actually been written.
        // n_kv is padded (e.g. 256 for a batch of 2 tokens); n_written is the
        // real count.  Positions beyond n_written have no KV data (K=0).
        const size_t n_check = n_written < static_cast<size_t>(n_kv)
            ? n_written : static_cast<size_t>(n_kv);

        // Active vector: n_check > 0 means at least some positions are scored
        g_slha_replace_counters.active_expected_vectors.fetch_add(1, std::memory_order_relaxed);
        g_slha_replace_counters.active_expected_logits.fetch_add(
            n_check, std::memory_order_relaxed);

        // Padding: positions beyond n_written
        if (static_cast<size_t>(n_kv) > n_written) {
            g_slha_replace_counters.padding_vectors.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.padding_logits.fetch_add(
                static_cast<size_t>(n_kv) - n_written, std::memory_order_relaxed);
        }

        // Read tile pointer base (position 0) and verify all needed positions
        const SciRustSlhaTile * tiles = n_check > 0
            ? static_cast<const SciRustSlhaTile *>(g_slha_tile_store.read(layer->layer_id, 0))
            : nullptr;
        size_t first_missing = SIZE_MAX;
        if (n_check > 0 && !tiles) {
            first_missing = 0;
        } else {
            for (size_t k = 1; k < n_check; ++k) {
                if (!g_slha_tile_store.read(layer->layer_id, k)) {
                    first_missing = k;
                    break;
                }
            }
        }
        if (first_missing != SIZE_MAX) {
            g_slha_replace_counters.n_missing_tile.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.error_code.store(1, std::memory_order_release);
            if (ith == 0 && first_missing < 5) {
                static thread_local size_t diag_count = 0;
                if (diag_count < 3) {
                    ++diag_count;
                    std::cerr << "[SLHA] replace diag: layer=" << layer->layer_id
                              << " n_kv=" << n_kv << " n_token=" << n_token
                              << " n_written=" << n_written << " n_check=" << n_check
                              << " s=" << s << " t=" << t << " h=" << h
                              << " first_missing=" << first_missing
                              << " tiles_ptr=" << (void*)tiles << "\n";
                }
            }
            continue;
        }

        if (n_check > 0) {
            rc = slha_score_tiles(model, tiles, n_check, q_coarse.data(), q_sign.data(), temp_scores.data());
            if (rc != SLHA_OK) {
                g_slha_replace_counters.n_score_fail.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                if (ith == 0) {
                    std::cerr << "[SLHA] replace layer " << layer->layer_id
                              << " score_tiles failed: " << slha_last_error_message() << "\n";
                }
                continue;
            }

            bool all_finite = true;
            for (size_t k = 0; k < n_check; ++k) {
                if (!std::isfinite(temp_scores[k])) {
                    all_finite = false;
                    break;
                }
            }
            if (!all_finite) {
                g_slha_replace_counters.n_nonfinite_score.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);
                g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                continue;
            }

            // Experimental temperature correction: multiply the raw SLHA scores
            // by the per-layer scale a BEFORE llama.cpp applies the fixed
            // 1/sqrt(head_dim) softmax scale. a == 1.0 is a no-op (default). Any
            // non-finite result fails closed (no partial write).
            if (g_score_scale_active) {
                const double a = g_score_scale.get(layer->layer_id);
                if (a != 1.0) {
                    bool scaled_finite = true;
                    for (size_t k = 0; k < n_check; ++k) {
                        const double v = static_cast<double>(temp_scores[k]) * a;
                        if (!std::isfinite(v)) { scaled_finite = false; break; }
                        temp_scores[k] = static_cast<float>(v);
                    }
                    if (!scaled_finite) {
                        g_scale_invalid.fetch_add(1, std::memory_order_relaxed);
                        g_slha_replace_counters.n_nonfinite_score.fetch_add(1, std::memory_order_relaxed);
                        g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);
                        g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                        continue;
                    }
                    g_scaled_vectors.fetch_add(1, std::memory_order_relaxed);
                    g_scaled_logits.fetch_add(n_check, std::memory_order_relaxed);
                }
            }

            // CAUSALLY VISIBLE prefix. Query token t attends keys 0..t only;
            // keys in [n_visible, n_check) are written tiles that soft_max_ext
            // masks to -INF for this query, and [n_check, n_kv) is padding.
            // Both oracle construction and the ranking statistics must be
            // restricted to the visible set: a transplant that spans masked keys
            // spends value-multiset slots on keys softmax never sees, which
            // dilutes the transplant instead of measuring it.
            const size_t n_visible =
                std::min<size_t>(n_check, static_cast<size_t>(t) + 1);

            // Active-key ranking / tie statistics on the UNTRANSFORMED pair,
            // over the visible prefix only. Deterministic (t,h) sampling keeps
            // the O(n^2) Kendall tau-b bounded and thread-independent.
            if (slha_oracle_metrics::enabled() && (t % 16) == 0 && h < 2 && n_visible > 0) {
                const float * mb_row = kq_data
                    + s * static_cast<ptrdiff_t>(kq_stream_stride)
                    + t * static_cast<ptrdiff_t>(kq_token_stride)
                    + h * static_cast<ptrdiff_t>(kq_head_stride);
                slha_oracle_metrics::add_row(layer->layer_id, static_cast<int>(h),
                                             mb_row, temp_scores.data(), n_visible);
                slha_oracle_metrics::add_accounting(
                    layer->layer_id,
                    static_cast<uint64_t>(n_kv),                 // physical
                    static_cast<uint64_t>(n_visible),            // included
                    static_cast<uint64_t>(n_kv) - static_cast<uint64_t>(n_check),  // padding
                    static_cast<uint64_t>(n_check - n_visible),  // causally masked
                    0,                                           // inactive stream (n_stream==1)
                    0);                                          // non-finite rejected
            }

            // Ranking-training dataset. OFFLINE ONLY: this records the exact
            // baseline row as a training LABEL, which is precisely what the
            // deployable path may never do at inference. It is enabled solely by
            // SLHA_RANK_DATASET_DIR and is inert otherwise.
            if (slha_rank_dataset::enabled() && slha_rank_dataset::wanted(t, h)
                && n_visible > 0) {
                const float * mb_row = kq_data
                    + s * static_cast<ptrdiff_t>(kq_stream_stride)
                    + t * static_cast<ptrdiff_t>(kq_token_stride)
                    + h * static_cast<ptrdiff_t>(kq_head_stride);
                slha_rank_dataset::add_row(
                    layer->layer_id, static_cast<int32_t>(h),
                    static_cast<int32_t>(kv_head), t,
                    q_extended.data(), static_cast<size_t>(d),
                    mb_row, temp_scores.data(), n_visible);
            }

            // Diagnostic score oracle. Rewrites this row from the PAIRED
            // baseline row B and the SLHA row S so that ranking errors can be
            // separated from order-preserving score-shape errors. Applied to
            // the active prefix [0, n_check) only, so padded and causally
            // masked positions never enter an oracle construction. Fails
            // closed: on any rejection the row is left as the zeroed padding
            // and the vector is counted as failed, never partially written.
            if (g_oracle.active && g_oracle.valid && g_oracle.mode != slha_oracle::Mode::Off
                && n_visible > 0) {
                thread_local slha_oracle::Workspace ows;
                thread_local std::vector<float> oracle_out;
                oracle_out.resize(n_visible);
                const float * b_row = kq_data
                    + s * static_cast<ptrdiff_t>(kq_stream_stride)
                    + t * static_cast<ptrdiff_t>(kq_token_stride)
                    + h * static_cast<ptrdiff_t>(kq_head_stride);
                uint64_t row_ties = 0;
                const slha_oracle::ApplyStatus st = slha_oracle::apply(
                    g_oracle.mode, g_oracle.topk, b_row, temp_scores.data(),
                    n_visible, oracle_out.data(), &row_ties, ows);
                if (st != slha_oracle::ApplyStatus::Ok) {
                    if (st == slha_oracle::ApplyStatus::NonFiniteInput) {
                        g_oracle_nonfinite_input.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        g_oracle_invalid_permutation.fetch_add(1, std::memory_order_relaxed);
                    }
                    g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);
                    g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                    continue;
                }
                // Bounded runtime invariant sampling: re-verify the mode's
                // declared rank / value-multiset guarantees on a few rows.
                if (g_oracle_invariant_checked.load(std::memory_order_relaxed) < 64) {
                    g_oracle_invariant_checked.fetch_add(1, std::memory_order_relaxed);
                    thread_local std::vector<float> vx, vy;
                    bool ok = true;
                    switch (g_oracle.mode) {
                        case slha_oracle::Mode::BaselineRankSlhaValues:
                            // respects_ranking, not same_ranking: when the
                            // transplanted values tie, the relative order of the
                            // equal-valued keys is unobservable in the output.
                            ok = slha_oracle::respects_ranking(oracle_out.data(), b_row, n_visible, ows)
                              && slha_oracle::same_value_multiset(oracle_out.data(),
                                     temp_scores.data(), n_visible, vx, vy);
                            break;
                        case slha_oracle::Mode::SlhaRankBaselineValues:
                            ok = slha_oracle::respects_ranking(oracle_out.data(),
                                     temp_scores.data(), n_visible, ows)
                              && slha_oracle::same_value_multiset(oracle_out.data(),
                                     b_row, n_visible, vx, vy);
                            break;
                        case slha_oracle::Mode::BaselineTopKRank:
                            ok = slha_oracle::same_value_multiset(oracle_out.data(),
                                     temp_scores.data(), n_visible, vx, vy);
                            break;
                        default:
                            break;
                    }
                    if (!ok) {
                        g_oracle_invariant_failed.fetch_add(1, std::memory_order_relaxed);
                        g_slha_replace_counters.error_code.store(1, std::memory_order_release);
                    }
                }
                std::memcpy(temp_scores.data(), oracle_out.data(), n_visible * sizeof(float));
                g_oracle_vectors.fetch_add(1, std::memory_order_relaxed);
                g_oracle_logits.fetch_add(n_visible, std::memory_order_relaxed);
                g_oracle_permutations.fetch_add(1, std::memory_order_relaxed);
                g_oracle_ties.fetch_add(static_cast<size_t>(row_ties), std::memory_order_relaxed);
            }

            // Write validated scores to dst (first n_check positions; the rest of
            // the row was zeroed above and stays zero as padding).
            std::memcpy(dst_head, temp_scores.data(), n_check * sizeof(float));
            if (ith == 0 && idx == start && g_diag_call_count < 100) {
                slha_diag_score(dst_head, static_cast<int64_t>(n_kv),
                                n_token, n_head, n_stream, layer->layer_id,
                                n_written);
            }

            g_slha_replace_counters.active_replaced_vectors.fetch_add(1, std::memory_order_relaxed);
            g_slha_replace_counters.active_replaced_logits.fetch_add(
                n_check, std::memory_order_relaxed);
            slha_replace_inc_layer_success(layer->layer_id);
        }
    }
}

void slha_diag_score(
    const float * dst_data,
    int64_t n_kv,
    int64_t n_token,
    int64_t n_head,
    int64_t n_stream,
    int32_t layer_id,
    size_t n_written
) {
    std::lock_guard<std::mutex> lock(g_diag_mutex);
    if (g_diag_call_count >= 100) return;
    ++g_diag_call_count;
    std::cerr << "[SLHA] diag cb#" << g_diag_call_count
              << " layer=" << layer_id
              << " n_kv=" << n_kv << " n_token=" << n_token
              << " n_head=" << n_head << " n_stream=" << n_stream
              << " n_written=" << n_written;
    std::cerr << " s[0]=" << dst_data[0]
              << " s[1]=" << dst_data[1];
    std::cerr << "\n";
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
            if (layer->score_mode == SLHA_SCORE_SHADOW && !layer->shadow_metrics) {
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

        // Track the maximum tile position written across ALL micro-batches
        // within a chunk using CAS-based max-tracking.  This lets the callback
        // always check up to n_kv (padded) — any position with a valid tile is
        // scored.  Between chunks slha_k_clear_all() resets the counters via
        // the llama_memory_clear hook.
        size_t my_max_pos = 0;
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
            // Offline training data only: keep the raw K row addressed by KV
            // position so a candidate projection can be re-scored without
            // re-running the model. Never read by the deployable path.
            if (slha_rank_dataset::enabled()) {
                std::lock_guard<std::mutex> lock(*layer->collect_mutex);
                const size_t need = (static_cast<size_t>(pos) + 1) * d;
                if (layer->raw_k_by_pos.size() < need) {
                    layer->raw_k_by_pos.resize(need, 0.0f);
                }
                std::memcpy(layer->raw_k_by_pos.data() + static_cast<size_t>(pos) * d,
                            src_row, d * sizeof(float));
                layer->raw_k_max_pos =
                    std::max(layer->raw_k_max_pos, static_cast<size_t>(pos) + 1);
            }
            if (!g_slha_tile_store.write(layer->layer_id, static_cast<size_t>(pos), &tile)) {
                std::cerr << "[SLHA] layer " << layer->layer_id
                          << " position " << pos << " tile store overflow (capacity="
                          << g_slha_tile_store.capacity << ")\n";
            } else {
                const size_t pos_plus_1 = static_cast<size_t>(pos) + 1;
                if (pos_plus_1 > my_max_pos) {
                    my_max_pos = pos_plus_1;
                }
            }
        }
        if (my_max_pos > 0) {
            size_t old = g_slha_tiles_written[layer->layer_id].load(std::memory_order_relaxed);
            while (my_max_pos > old
                   && !g_slha_tiles_written[layer->layer_id].compare_exchange_weak(
                       old, my_max_pos, std::memory_order_release, std::memory_order_relaxed)) {
                // CAS failed because another thread updated to a higher value; retry
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

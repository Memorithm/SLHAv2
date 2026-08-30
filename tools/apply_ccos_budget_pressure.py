from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, got {count}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# C++ external-K shim: explicit quiescent COLD lifecycle + telemetry.
# ---------------------------------------------------------------------------
path = "integration/llama.cpp/shim/slha_external_k.hpp"
replace_once(
    path,
    """/** Reset logical live tiles while retaining configured backend state. */\nvoid slha_external_k_reset_store();\n""",
    """/**\n * Move every currently active CCOS key to COLD backing storage.\n *\n * This is a quiescent lifecycle operation: callers must invoke it only after a\n * synchronous llama_decode has returned and before another decode starts. Dense\n * attention cannot score COLD keys. The operation is fail-closed and leaves the\n * cache unchanged on an offload failure.\n */\nbool slha_external_k_ccos_offload_quiescent();\n\n/**\n * Restore a cache previously offloaded with\n * slha_external_k_ccos_offload_quiescent(). HOT/WARM representations are\n * restored exactly before the next decode is allowed to run.\n */\nbool slha_external_k_ccos_restore_quiescent();\n\n/** Reset logical live tiles while retaining configured backend state. */\nvoid slha_external_k_reset_store();\n""",
)
replace_once(
    path,
    """    uint64_t cache_misses = 0;\n    uint64_t compression_ns = 0;\n    uint64_t score_ns = 0;\n    uint64_t budget_ns = 0;\n""",
    """    uint64_t cache_misses = 0;\n    uint64_t quiescent_offload_calls = 0;\n    uint64_t quiescent_restore_calls = 0;\n    uint64_t quiescent_restored_slots = 0;\n    uint64_t quiescent_offload_ns = 0;\n    uint64_t quiescent_restore_ns = 0;\n    uint64_t compression_ns = 0;\n    uint64_t score_ns = 0;\n    uint64_t budget_ns = 0;\n""",
)

path = "integration/llama.cpp/shim/slha_external_k.cpp"
replace_once(
    path,
    """std::atomic<uint64_t> g_cache_hits{0};\nstd::atomic<uint64_t> g_cache_misses{0};\nstd::atomic<uint64_t> g_compression_ns{0};\n""",
    """std::atomic<uint64_t> g_cache_hits{0};\nstd::atomic<uint64_t> g_cache_misses{0};\nstd::atomic<uint64_t> g_quiescent_offload_calls{0};\nstd::atomic<uint64_t> g_quiescent_restore_calls{0};\nstd::atomic<uint64_t> g_quiescent_restored_slots{0};\nstd::atomic<uint64_t> g_quiescent_offload_ns{0};\nstd::atomic<uint64_t> g_quiescent_restore_ns{0};\nsize_t g_quiescent_pre_offload_resident = 0;\nsize_t g_quiescent_pre_offload_hot = 0;\nsize_t g_quiescent_pre_offload_warm = 0;\nstd::atomic<uint64_t> g_compression_ns{0};\n""",
)
replace_once(
    path,
    """    g_cache_hits.store(0, std::memory_order_relaxed);\n    g_cache_misses.store(0, std::memory_order_relaxed);\n    g_compression_ns.store(0, std::memory_order_relaxed);\n""",
    """    g_cache_hits.store(0, std::memory_order_relaxed);\n    g_cache_misses.store(0, std::memory_order_relaxed);\n    g_quiescent_offload_calls.store(0, std::memory_order_relaxed);\n    g_quiescent_restore_calls.store(0, std::memory_order_relaxed);\n    g_quiescent_restored_slots.store(0, std::memory_order_relaxed);\n    g_quiescent_offload_ns.store(0, std::memory_order_relaxed);\n    g_quiescent_restore_ns.store(0, std::memory_order_relaxed);\n    g_quiescent_pre_offload_resident = 0;\n    g_quiescent_pre_offload_hot = 0;\n    g_quiescent_pre_offload_warm = 0;\n    g_compression_ns.store(0, std::memory_order_relaxed);\n""",
)
marker = """void slha_external_k_reset_store() {\n"""
insert = r'''bool slha_external_k_ccos_offload_quiescent() {
    if (!slha_external_k_ccos_enabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_ccos_mutex);
    if (!g_ccos_cache) {
        return false;
    }

    SlhaElasticKvCacheStats before{};
    if (!snapshot_ccos(&before)) {
        return false;
    }
    const size_t active = before.hot_slots + before.warm_slots + before.cold_slots + before.pinned_slots;
    if (active == 0 || before.cold_slots != 0 || before.pinned_slots != 0) {
        return false;
    }

    const auto start = steady_clock::now();
    const int32_t rc = slha_elastic_cache_offload_to(g_ccos_cache, 0);
    const auto end = steady_clock::now();
    if (rc != SLHA_OK) {
        return false;
    }

    SlhaElasticKvCacheStats cold{};
    if (!snapshot_ccos(&cold) || cold.resident_bytes != 0 || cold.cold_slots != active) {
        return false;
    }
    record_cache_snapshot(cold);
    g_quiescent_pre_offload_resident = before.resident_bytes;
    g_quiescent_pre_offload_hot = before.hot_slots;
    g_quiescent_pre_offload_warm = before.warm_slots;
    g_quiescent_offload_calls.fetch_add(1, std::memory_order_relaxed);
    g_quiescent_offload_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
        std::memory_order_relaxed);
    return true;
}

bool slha_external_k_ccos_restore_quiescent() {
    if (!slha_external_k_ccos_enabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_ccos_mutex);
    if (!g_ccos_cache || g_quiescent_pre_offload_resident == 0) {
        return false;
    }

    size_t logical_slots = 0;
    if (!checked_mul(g_ccos_n_layers, g_ccos_capacity, &logical_slots)) {
        return false;
    }

    const auto start = steady_clock::now();
    size_t restored = 0;
    for (size_t slot = 0; slot < logical_slots; ++slot) {
        if (slha_elastic_cache_tier(g_ccos_cache, slot) != SLHA_ELASTIC_TIER_COLD) {
            continue;
        }
        if (slha_elastic_cache_restore_slot(g_ccos_cache, slot) != SLHA_OK) {
            // Return to a coherent all-COLD state rather than exposing a
            // partially restored context to a future dense-attention decode.
            (void) slha_elastic_cache_offload_to(g_ccos_cache, 0);
            return false;
        }
        ++restored;
    }
    const auto end = steady_clock::now();

    SlhaElasticKvCacheStats after{};
    if (!snapshot_ccos(&after) || after.cold_slots != 0 ||
        after.resident_bytes != g_quiescent_pre_offload_resident ||
        after.hot_slots != g_quiescent_pre_offload_hot ||
        after.warm_slots != g_quiescent_pre_offload_warm) {
        (void) slha_elastic_cache_offload_to(g_ccos_cache, 0);
        return false;
    }
    record_cache_snapshot(after);
    g_quiescent_restore_calls.fetch_add(1, std::memory_order_relaxed);
    g_quiescent_restored_slots.fetch_add(restored, std::memory_order_relaxed);
    g_quiescent_restore_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
        std::memory_order_relaxed);
    g_quiescent_pre_offload_resident = 0;
    g_quiescent_pre_offload_hot = 0;
    g_quiescent_pre_offload_warm = 0;
    return true;
}

'''
p = Path(path)
text = p.read_text()
if text.count(marker) != 1:
    raise RuntimeError("missing reset-store marker")
p.write_text(text.replace(marker, insert + marker, 1))

replace_once(
    path,
    """    stats.cache_hits = g_cache_hits.load(std::memory_order_relaxed);\n    stats.cache_misses = g_cache_misses.load(std::memory_order_relaxed);\n    stats.compression_ns = g_compression_ns.load(std::memory_order_relaxed);\n""",
    """    stats.cache_hits = g_cache_hits.load(std::memory_order_relaxed);\n    stats.cache_misses = g_cache_misses.load(std::memory_order_relaxed);\n    stats.quiescent_offload_calls = g_quiescent_offload_calls.load(std::memory_order_relaxed);\n    stats.quiescent_restore_calls = g_quiescent_restore_calls.load(std::memory_order_relaxed);\n    stats.quiescent_restored_slots = g_quiescent_restored_slots.load(std::memory_order_relaxed);\n    stats.quiescent_offload_ns = g_quiescent_offload_ns.load(std::memory_order_relaxed);\n    stats.quiescent_restore_ns = g_quiescent_restore_ns.load(std::memory_order_relaxed);\n    stats.compression_ns = g_compression_ns.load(std::memory_order_relaxed);\n""",
)
replace_once(
    path,
    """              << \" cache_hits=\" << stats.cache_hits\n              << \" cache_misses=\" << stats.cache_misses\n              << \" compression_ns=\" << stats.compression_ns\n""",
    """              << \" cache_hits=\" << stats.cache_hits\n              << \" cache_misses=\" << stats.cache_misses\n              << \" quiescent_offload_calls=\" << stats.quiescent_offload_calls\n              << \" quiescent_restore_calls=\" << stats.quiescent_restore_calls\n              << \" quiescent_restored_slots=\" << stats.quiescent_restored_slots\n              << \" quiescent_offload_ns=\" << stats.quiescent_offload_ns\n              << \" quiescent_restore_ns=\" << stats.quiescent_restore_ns\n              << \" compression_ns=\" << stats.compression_ns\n""",
)

# ---------------------------------------------------------------------------
# C++ contract test: real HOT -> COLD -> HOT quiescent roundtrip.
# ---------------------------------------------------------------------------
path = "integration/llama.cpp/tests/external_k_contract_tests.cpp"
marker = """    set_valid_external_env();\n    setenv(\"SLHA_SCORE_MODE\", \"shadow\", 1);\n"""
insert = r'''    // COLD is legal only while the dense-attention context is quiescent.
    // Offload all active keys, verify that they become unscoreable COLD state,
    // then restore the exact HOT representation before scoring resumes.
    set_valid_ccos_env("384");
    assert(slha_external_k_prepare_store(8));
    assert(slha_external_k_write_tile(0, 0, &tile));
    assert(slha_external_k_write_tile(1, 0, &tile));
    assert(slha_external_k_write_tile(2, 0, &tile));
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 3u);
    assert(stats.resident_bytes == 384u);
    assert(slha_external_k_ccos_offload_quiescent());
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 0u);
    assert(stats.warm_slots == 0u);
    assert(stats.cold_slots == 3u);
    assert(stats.resident_bytes == 0u);
    assert(stats.evictions == 3u);
    assert(stats.quiescent_offload_calls == 1u);
    assert(slha_external_k_score_tiles(nullptr, 0, 0, 1, q_coarse, q_sign, &score) == SLHA_ERR_NOT_RESIDENT);
    assert(slha_external_k_ccos_restore_quiescent());
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 3u);
    assert(stats.warm_slots == 0u);
    assert(stats.cold_slots == 0u);
    assert(stats.resident_bytes == 384u);
    assert(stats.quiescent_restore_calls == 1u);
    assert(stats.quiescent_restored_slots == 3u);
    assert(slha_external_k_score_tiles(nullptr, 0, 0, 1, q_coarse, q_sign, &score) == SLHA_OK);

'''
p = Path(path)
text = p.read_text()
if text.count(marker) != 1:
    raise RuntimeError("missing external contract insertion marker")
p.write_text(text.replace(marker, insert + marker, 1))

# ---------------------------------------------------------------------------
# Real evaluator: optional quiescent cold cycle between synchronous decodes.
# ---------------------------------------------------------------------------
path = "integration/llama.cpp/tools/slha_real_eval.cpp"
replace_once(path, '#include "llama.h"\n', '#include "llama.h"\n#include "slha_external_k.hpp"\n')
replace_once(
    path,
    """    ggml_type cache_type_k = GGML_TYPE_F16;\n    ggml_type cache_type_v = GGML_TYPE_F16;\n};\n""",
    """    ggml_type cache_type_k = GGML_TYPE_F16;\n    ggml_type cache_type_v = GGML_TYPE_F16;\n    int32_t ccos_cold_cycle_step = -1;\n};\n""",
)
replace_once(
    path,
    '        << "[--gpu-layers N] [--cache-type-k f16|f32|bf16] [--cache-type-v f16|f32|bf16]\\n";\n',
    '        << "[--gpu-layers N] [--cache-type-k f16|f32|bf16] [--cache-type-v f16|f32|bf16] "\n        << "[--ccos-cold-cycle-step N]\\n";\n',
)
replace_once(
    path,
    """        else if (arg == \"--cache-type-k\") opts.cache_type_k = parse_cache_type(next());\n        else if (arg == \"--cache-type-v\") opts.cache_type_v = parse_cache_type(next());\n        else usage(argv[0], \"unknown option \" + arg);\n""",
    """        else if (arg == \"--cache-type-k\") opts.cache_type_k = parse_cache_type(next());\n        else if (arg == \"--cache-type-v\") opts.cache_type_v = parse_cache_type(next());\n        else if (arg == \"--ccos-cold-cycle-step\") opts.ccos_cold_cycle_step = std::stoi(next());\n        else usage(argv[0], \"unknown option \" + arg);\n""",
)
replace_once(
    path,
    """    if (opts.threads <= 0) usage(argv[0], \"--threads must be positive\");\n    return opts;\n}\n""",
    """    if (opts.threads <= 0) usage(argv[0], \"--threads must be positive\");\n    if (opts.ccos_cold_cycle_step < -1 || opts.ccos_cold_cycle_step >= opts.max_tokens - 1) {\n        usage(argv[0], \"--ccos-cold-cycle-step must be -1 or leave at least one decode step after the cycle\");\n    }\n    return opts;\n}\n""",
)
marker = """void write_double_array(std::ostream & out, const std::vector<double> & values) {\n"""
p = Path(path)
text = p.read_text()
idx = text.find(marker)
if idx < 0:
    raise RuntimeError("missing write_double_array marker")
end_marker = """}\n\n} // namespace\n"""
end_idx = text.find(end_marker, idx)
if end_idx < 0:
    raise RuntimeError("missing namespace end marker")
helper = r'''
void write_store_snapshot(std::ostream & out, const slha_external_k_store_stats & stats) {
    out << "{"
        << "\"resident_bytes\":" << stats.resident_bytes << ','
        << "\"offloaded_bytes\":" << stats.offloaded_bytes << ','
        << "\"hot_slots\":" << stats.hot_slots << ','
        << "\"warm_slots\":" << stats.warm_slots << ','
        << "\"cold_slots\":" << stats.cold_slots << ','
        << "\"pinned_slots\":" << stats.pinned_slots << ','
        << "\"evictions\":" << stats.evictions
        << "}";
}

'''
text = text[:end_idx+2] + "\n" + helper + text[end_idx+2:]
p.write_text(text)

replace_once(
    path,
    """        double prefill_ms = 0.0;\n        double ttft_ms = 0.0;\n\n        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);\n""",
    """        double prefill_ms = 0.0;\n        double ttft_ms = 0.0;\n        bool ccos_lifecycle_executed = false;\n        slha_external_k_store_stats ccos_before{};\n        slha_external_k_store_stats ccos_cold{};\n        slha_external_k_store_stats ccos_restored{};\n\n        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);\n""",
)
replace_once(
    path,
    """            generated_text += token_piece(vocab, token);\n            batch = llama_batch_get_one(&generated_tokens.back(), 1);\n        }\n""",
    r'''            generated_text += token_piece(vocab, token);
            batch = llama_batch_get_one(&generated_tokens.back(), 1);

            if (opts.ccos_cold_cycle_step == step) {
                if (!slha_external_k_store_stats_snapshot(&ccos_before)) {
                    throw std::runtime_error("cannot snapshot CCOS state before quiescent COLD cycle");
                }
                const size_t active_before = ccos_before.hot_slots + ccos_before.warm_slots +
                    ccos_before.cold_slots + ccos_before.pinned_slots;
                if (active_before == 0 || ccos_before.cold_slots != 0) {
                    throw std::runtime_error("invalid active CCOS state before quiescent COLD cycle");
                }
                if (!slha_external_k_ccos_offload_quiescent() ||
                    !slha_external_k_store_stats_snapshot(&ccos_cold)) {
                    throw std::runtime_error("CCOS quiescent offload failed");
                }
                if (ccos_cold.resident_bytes != 0 || ccos_cold.cold_slots != active_before) {
                    throw std::runtime_error("CCOS quiescent offload did not produce complete COLD state");
                }
                if (!slha_external_k_ccos_restore_quiescent() ||
                    !slha_external_k_store_stats_snapshot(&ccos_restored)) {
                    throw std::runtime_error("CCOS quiescent restore failed");
                }
                if (ccos_restored.cold_slots != 0 ||
                    ccos_restored.resident_bytes != ccos_before.resident_bytes ||
                    ccos_restored.hot_slots != ccos_before.hot_slots ||
                    ccos_restored.warm_slots != ccos_before.warm_slots) {
                    throw std::runtime_error("CCOS restore did not recover pre-offload residency exactly");
                }
                ccos_lifecycle_executed = true;
            }
        }
''',
)
replace_once(
    path,
    """        report << \"    \\\"decode_step_ms\\\": \";\n        write_double_array(report, decode_step_ms);\n        report << \"\\n  }\\n\";\n        report << \"}\\n\";\n""",
    r'''        report << "    \"decode_step_ms\": ";
        write_double_array(report, decode_step_ms);
        report << "\n  },\n";
        report << "  \"ccos_lifecycle\": {\n";
        report << "    \"requested\": " << (opts.ccos_cold_cycle_step >= 0 ? "true" : "false") << ",\n";
        report << "    \"executed\": " << (ccos_lifecycle_executed ? "true" : "false") << ",\n";
        report << "    \"step\": " << opts.ccos_cold_cycle_step << ",\n";
        report << "    \"before\": ";
        if (ccos_lifecycle_executed) write_store_snapshot(report, ccos_before); else report << "null";
        report << ",\n    \"cold\": ";
        if (ccos_lifecycle_executed) write_store_snapshot(report, ccos_cold); else report << "null";
        report << ",\n    \"restored\": ";
        if (ccos_lifecycle_executed) write_store_snapshot(report, ccos_restored); else report << "null";
        report << "\n  }\n";
        report << "}\n";
''',
)

# ---------------------------------------------------------------------------
# Pair runner: pass lifecycle control only to the external CCOS arm.
# ---------------------------------------------------------------------------
path = "integration/llama.cpp/run_real_pair.sh"
replace_once(
    path,
    """CCOS_IMPORTANCE_TEMPERATURE=\"\"\n\nwhile [[ $# -gt 0 ]]; do\n""",
    """CCOS_IMPORTANCE_TEMPERATURE=\"\"\nCCOS_COLD_CYCLE_STEP=\"\"\n\nwhile [[ $# -gt 0 ]]; do\n""",
)
replace_once(
    path,
    """        --ccos-importance-temperature) CCOS_IMPORTANCE_TEMPERATURE=\"${2:?missing value for --ccos-importance-temperature}\"; shift 2 ;;\n        *) echo \"ERROR: unknown option: $1\" >&2; exit 2 ;;\n""",
    """        --ccos-importance-temperature) CCOS_IMPORTANCE_TEMPERATURE=\"${2:?missing value for --ccos-importance-temperature}\"; shift 2 ;;\n        --ccos-cold-cycle-step) CCOS_COLD_CYCLE_STEP=\"${2:?missing value for --ccos-cold-cycle-step}\"; shift 2 ;;\n        *) echo \"ERROR: unknown option: $1\" >&2; exit 2 ;;\n""",
)
replace_once(
    path,
    """if [[ \"$CCOS\" -ne 1 && ( -n \"$CCOS_BUDGET_BYTES\" || -n \"$CCOS_IMPORTANCE_TEMPERATURE\" ) ]]; then\n    echo \"ERROR: CCOS budget/temperature options require --ccos\" >&2\n    exit 2\nfi\n""",
    """if [[ \"$CCOS\" -ne 1 && ( -n \"$CCOS_BUDGET_BYTES\" || -n \"$CCOS_IMPORTANCE_TEMPERATURE\" || -n \"$CCOS_COLD_CYCLE_STEP\" ) ]]; then\n    echo \"ERROR: CCOS budget/temperature/lifecycle options require --ccos\" >&2\n    exit 2\nfi\n""",
)
replace_once(
    path,
    """    -I\"$LLAMA_DIR/include\" \\\n    -I\"$LLAMA_DIR/ggml/include\" \\\n""",
    """    -I\"$LLAMA_DIR/include\" \\\n    -I\"$LLAMA_DIR/ggml/include\" \\\n    -I\"$LLAMA_DIR/src\" \\\n""",
)
old = r'''        "$EVAL_BIN" \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --output-json "$json" \
        --logits-bin "$logits" \
        --max-tokens "$MAX_TOKENS" \
        --context-size "$CTX_SIZE" \
        --threads "$THREADS" \
        --gpu-layers "$GPU_LAYERS" \
        --cache-type-k "$CACHE_TYPE_K" \
        --cache-type-v "$CACHE_TYPE_V" \
        2>&1 | tee "$log"
'''
new = r'''        local eval_args=(
            --model "$MODEL"
            --prompt "$PROMPT"
            --output-json "$json"
            --logits-bin "$logits"
            --max-tokens "$MAX_TOKENS"
            --context-size "$CTX_SIZE"
            --threads "$THREADS"
            --gpu-layers "$GPU_LAYERS"
            --cache-type-k "$CACHE_TYPE_K"
            --cache-type-v "$CACHE_TYPE_V"
        )
        if [[ "$mode" == "external" && -n "$CCOS_COLD_CYCLE_STEP" ]]; then
            eval_args+=(--ccos-cold-cycle-step "$CCOS_COLD_CYCLE_STEP")
        fi
        "$EVAL_BIN" "${eval_args[@]}" 2>&1 | tee "$log"
'''
replace_once(path, old, new)

# Propagate lifecycle evidence into the paired JSON without changing quality rules.
path = "integration/llama.cpp/scripts/compare_real_eval.py"
replace_once(
    path,
    """        \"host\": {\n            \"platform\": platform.platform(),\n            \"machine\": platform.machine(),\n            \"logical_cpus\": os.cpu_count(),\n        },\n        \"quality\": {\n""",
    """        \"host\": {\n            \"platform\": platform.platform(),\n            \"machine\": platform.machine(),\n            \"logical_cpus\": os.cpu_count(),\n        },\n        \"ccos_lifecycle\": external.get(\"ccos_lifecycle\"),\n        \"quality\": {\n""",
)

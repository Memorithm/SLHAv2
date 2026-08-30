from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {count}")
    p.write_text(text.replace(old, new, 1))


path = "integration/llama.cpp/shim/slha_external_k.cpp"
replace_once(
    path,
    """void slha_external_k_reset_store() {
    if (!slha_external_k_ccos_enabled()) {
        if (g_slha_tile_store.n_layers > 0) {
            g_slha_tile_store.reset();
        }
        return;
    }
    if (g_ccos_cache) {
        (void) slha_elastic_cache_clear(g_ccos_cache);
    }
}
""",
    """void slha_external_k_reset_store() {
    if (!slha_external_k_ccos_enabled()) {
        if (g_slha_tile_store.n_layers > 0) {
            g_slha_tile_store.reset();
        }
        return;
    }
    std::lock_guard<std::mutex> lock(g_ccos_mutex);
    // A context reset invalidates any pending quiescent restore intent. Keeping
    // the pre-offload snapshot across reset would allow stale lifecycle state
    // to be applied to a new logical context.
    g_quiescent_pre_offload_resident = 0;
    g_quiescent_pre_offload_hot = 0;
    g_quiescent_pre_offload_warm = 0;
    if (g_ccos_cache) {
        (void) slha_elastic_cache_clear(g_ccos_cache);
    }
}
""",
)
replace_once(
    path,
    """    g_ccos_n_layers = 0;
    g_ccos_capacity = 0;
    g_ccos_budget_bytes = 0;
}

void slha_external_k_record_compression_ns(uint64_t elapsed_ns) {
""",
    """    g_ccos_n_layers = 0;
    g_ccos_capacity = 0;
    g_ccos_budget_bytes = 0;
    g_quiescent_pre_offload_resident = 0;
    g_quiescent_pre_offload_hot = 0;
    g_quiescent_pre_offload_warm = 0;
}

void slha_external_k_record_compression_ns(uint64_t elapsed_ns) {
""",
)

path = "integration/llama.cpp/tests/external_k_contract_tests.cpp"
replace_once(
    path,
    """    assert(stats.quiescent_offload_calls == 1u);
    assert(stats.quiescent_restore_calls == 1u);
    assert(stats.quiescent_restored_slots == 1u);
    assert(slha_external_k_score_tiles(nullptr, 0, 0, 1, q_coarse, q_sign, &score) == SLHA_OK);

    set_valid_external_env();
    setenv(\"SLHA_SCORE_MODE\", \"shadow\", 1);
""",
    """    assert(stats.quiescent_offload_calls == 1u);
    assert(stats.quiescent_restore_calls == 1u);
    assert(stats.quiescent_restored_slots == 1u);
    assert(slha_external_k_score_tiles(nullptr, 0, 0, 1, q_coarse, q_sign, &score) == SLHA_OK);

    // Reset is a hard lifecycle boundary: an all-COLD snapshot from the old
    // logical context must not remain restorable after the cache is cleared.
    set_valid_ccos_env(\"128\");
    assert(slha_external_k_prepare_store(8));
    assert(slha_external_k_write_tile(0, 0, &tile));
    assert(slha_external_k_ccos_offload_quiescent());
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.cold_slots == 1u);
    slha_external_k_reset_store();
    assert(slha_external_k_store_stats_snapshot(&stats));
    assert(stats.hot_slots == 0u);
    assert(stats.warm_slots == 0u);
    assert(stats.cold_slots == 0u);
    assert(stats.resident_bytes == 0u);
    assert(!slha_external_k_ccos_restore_quiescent());
    assert(slha_external_k_write_tile(0, 0, &tile));
    assert(slha_external_k_score_tiles(nullptr, 0, 0, 1, q_coarse, q_sign, &score) == SLHA_OK);

    set_valid_external_env();
    setenv(\"SLHA_SCORE_MODE\", \"shadow\", 1);
""",
)

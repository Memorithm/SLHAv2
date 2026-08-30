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

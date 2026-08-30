from pathlib import Path

p = Path('integration/llama.cpp/shim/slha_llama.cpp')
s = p.read_text()

s = s.replace(
    '#include "slha_llama.hpp"\n',
    '#include "slha_llama.hpp"\n#include "slha_external_k.hpp"\n',
    1,
)
s = s.replace(
    '#include <algorithm>\n',
    '#include <algorithm>\n#include <chrono>\n',
    1,
)

old = '''    if (g_slha_tile_store.n_layers > 0) {\n        g_slha_tile_store.reset();\n    }\n}\n\nvoid slha_global_shutdown() {\n'''
new = '''    slha_external_k_reset_store();\n}\n\nvoid slha_global_shutdown() {\n'''
if old not in s:
    raise RuntimeError('missing slha_k_clear_all reset block')
s = s.replace(old, new, 1)

old = '''    state.layers.clear();\n    state.initialized = false;\n    g_slha_tile_store.reset();\n}\n'''
new = '''    // Emit the final external-K/CCOS physical-store accounting before\n    // releasing the optional Rust cache. Peak counters survive logical clears\n    // and therefore remain meaningful even if llama.cpp reset the KV lifecycle.\n    if (slha_external_k_enabled()) {\n        slha_external_k_print_store_summary();\n    }\n    slha_external_k_release_store();\n\n    state.layers.clear();\n    state.initialized = false;\n    g_slha_tile_store.reset();\n}\n'''
if old not in s:
    raise RuntimeError('missing shutdown tail')
s = s.replace(old, new, 1)

old = '''            alignas(SLHA_TILE_ALIGN) SciRustSlhaTile tile;\n            int rc = slha_encode_key(model, src_row, d, static_cast<uint32_t>(pos), codec, &tile);\n            if (rc != SLHA_OK) {\n'''
new = '''            alignas(SLHA_TILE_ALIGN) SciRustSlhaTile tile;\n            const auto encode_start = std::chrono::steady_clock::now();\n            int rc = slha_encode_key(model, src_row, d, static_cast<uint32_t>(pos), codec, &tile);\n            const auto encode_end = std::chrono::steady_clock::now();\n            slha_external_k_record_compression_ns(static_cast<uint64_t>(\n                std::chrono::duration_cast<std::chrono::nanoseconds>(\n                    encode_end - encode_start).count()));\n            if (rc != SLHA_OK) {\n'''
if old not in s:
    raise RuntimeError('missing tilestore encode seam')
s = s.replace(old, new, 1)

old = '''            if (!g_slha_tile_store.write(layer->layer_id, static_cast<size_t>(pos), &tile)) {\n                std::cerr << "[SLHA] layer " << layer->layer_id\n                          << " position " << pos << " tile store overflow (capacity="\n                          << g_slha_tile_store.capacity << ")\\n";\n            } else {\n'''
new = '''            if (!slha_external_k_write_tile(\n                    layer->layer_id, static_cast<size_t>(pos), &tile)) {\n                std::cerr << "[SLHA] layer " << layer->layer_id\n                          << " position " << pos\n                          << " external-K store write/budget enforcement failed (capacity="\n                          << g_slha_tile_store.capacity << ")\\n";\n                if (slha_external_k_enabled()) {\n                    g_slha_replace_counters.error_code.store(1, std::memory_order_release);\n                }\n            } else {\n'''
if old not in s:
    raise RuntimeError('missing tilestore write seam')
s = s.replace(old, new, 1)

old = '''        // Materialize exactly one immutable contiguous snapshot.  Do not call\n        // read()/read_range() again before slha_score_tiles(): another copy-out\n        // call on this thread invalidates the snapshot lifetime contract.\n        const SciRustSlhaTile * tiles = n_check > 0\n            ? static_cast<const SciRustSlhaTile *>(\n                g_slha_tile_store.read_range(layer->layer_id, 0, n_check))\n            : nullptr;\n        const size_t first_missing =\n            (n_check > 0 && !tiles) ? 0 : SIZE_MAX;\n        if (first_missing != SIZE_MAX) {\n            g_slha_replace_counters.n_missing_tile.fetch_add(1, std::memory_order_relaxed);\n            g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);\n            g_slha_replace_counters.error_code.store(1, std::memory_order_release);\n            if (ith == 0 && first_missing < 5) {\n                static thread_local size_t diag_count = 0;\n                if (diag_count < 3) {\n                    ++diag_count;\n                    std::cerr << "[SLHA] replace diag: layer=" << layer->layer_id\n                              << " n_kv=" << n_kv << " n_token=" << n_token\n                              << " n_written=" << n_written << " n_check=" << n_check\n                              << " s=" << s << " t=" << t << " h=" << h\n                              << " first_missing=" << first_missing\n                              << " tiles_ptr=" << (void*)tiles << "\\n";\n                }\n            }\n            continue;\n        }\n\n        if (n_check > 0) {\n            rc = slha_score_tiles(model, tiles, n_check, q_coarse.data(), q_sign.data(), temp_scores.data());\n            if (rc != SLHA_OK) {\n                g_slha_replace_counters.n_score_fail.fetch_add(1, std::memory_order_relaxed);\n                g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);\n                g_slha_replace_counters.error_code.store(1, std::memory_order_release);\n                if (ith == 0) {\n                    std::cerr << "[SLHA] replace layer " << layer->layer_id\n                              << " score_tiles failed: " << slha_last_error_message() << "\\n";\n                }\n                continue;\n            }\n'''
new = '''        if (n_check > 0) {\n            rc = slha_external_k_score_tiles(\n                model, layer->layer_id, 0, n_check,\n                q_coarse.data(), q_sign.data(), temp_scores.data());\n            if (rc != SLHA_OK) {\n                if (rc == SLHA_ERR_NOT_RESIDENT) {\n                    g_slha_replace_counters.n_missing_tile.fetch_add(1, std::memory_order_relaxed);\n                } else {\n                    g_slha_replace_counters.n_score_fail.fetch_add(1, std::memory_order_relaxed);\n                }\n                g_slha_replace_counters.n_failed_vectors.fetch_add(1, std::memory_order_relaxed);\n                g_slha_replace_counters.error_code.store(1, std::memory_order_release);\n                if (ith == 0) {\n                    std::cerr << "[SLHA] replace layer " << layer->layer_id\n                              << " external score failed rc=" << rc\n                              << ": " << slha_last_error_message() << "\\n";\n                }\n                continue;\n            }\n'''
if old not in s:
    raise RuntimeError('missing replace scoring snapshot block')
s = s.replace(old, new, 1)
p.write_text(s)

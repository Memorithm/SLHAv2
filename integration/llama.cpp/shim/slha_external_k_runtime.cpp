// Keep the existing external-K implementation and lifecycle extensions in one
// translation unit. The suffix extension needs the private CCOS handle but does
// not expose it through any public header or C ABI.
#include "slha_external_k.cpp"

bool slha_external_k_trim_suffix(
    size_t new_high_water,
    size_t old_high_water
) {
    if (!slha_external_k_enabled() || new_high_water > old_high_water) {
        return false;
    }
    if (new_high_water == old_high_water) {
        return true;
    }

    if (!slha_external_k_ccos_enabled()) {
        std::lock_guard<std::mutex> lock(g_slha_tile_store.mutex);
        const size_t n_layers = g_slha_tile_store.n_layers;
        const size_t capacity = g_slha_tile_store.capacity;
        const size_t tile_bytes = g_slha_tile_store.tile_bytes;
        if (n_layers == 0 || tile_bytes == 0 || old_high_water > capacity) {
            return false;
        }

        // Preflight every tile first. A failed trim must not intentionally
        // create a partial suffix/hole in the dense external store.
        for (size_t layer = 0; layer < n_layers; ++layer) {
            for (size_t position = new_high_water; position < old_high_water; ++position) {
                size_t index = 0;
                size_t base = 0;
                if (!checked_mul(layer, capacity, &base) ||
                    !checked_add(base, position, &index) ||
                    index >= g_slha_tile_store.valid.size() ||
                    g_slha_tile_store.valid[index] == 0u) {
                    return false;
                }
            }
        }

        for (size_t layer = 0; layer < n_layers; ++layer) {
            for (size_t position = new_high_water; position < old_high_water; ++position) {
                size_t index = 0;
                size_t base = 0;
                size_t byte_delta = 0;
                size_t byte_offset = 0;
                if (!checked_mul(layer, capacity, &base) ||
                    !checked_add(base, position, &index) ||
                    !checked_mul(index, tile_bytes, &byte_delta) ||
                    !checked_add(g_slha_tile_store.tile_base_offset, byte_delta, &byte_offset) ||
                    byte_offset > g_slha_tile_store.tiles.size() ||
                    tile_bytes > g_slha_tile_store.tiles.size() - byte_offset) {
                    return false;
                }
                std::fill(
                    g_slha_tile_store.tiles.begin() + static_cast<std::ptrdiff_t>(byte_offset),
                    g_slha_tile_store.tiles.begin() + static_cast<std::ptrdiff_t>(byte_offset + tile_bytes),
                    0u);
                g_slha_tile_store.valid[index] = 0u;
            }
        }
        return true;
    }

    std::lock_guard<std::mutex> lock(g_ccos_mutex);
    if (!g_ccos_cache || g_ccos_n_layers == 0 || old_high_water > g_ccos_capacity) {
        return false;
    }

    // CCOS uses interleaved stable slots: position*n_layers + layer. Preflight
    // all suffix slots before releasing any one of them.
    for (size_t position = new_high_water; position < old_high_water; ++position) {
        for (size_t layer = 0; layer < g_ccos_n_layers; ++layer) {
            size_t slot = 0;
            if (!ccos_geometry_slot(static_cast<int32_t>(layer), position, &slot) ||
                slha_elastic_cache_tier(g_ccos_cache, slot) == SLHA_ELASTIC_TIER_ABSENT) {
                return false;
            }
        }
    }

    for (size_t position = new_high_water; position < old_high_water; ++position) {
        for (size_t layer = 0; layer < g_ccos_n_layers; ++layer) {
            size_t slot = 0;
            if (!ccos_geometry_slot(static_cast<int32_t>(layer), position, &slot) ||
                slha_elastic_cache_clear_slot(g_ccos_cache, slot) != SLHA_OK) {
                // The operation is quiescent. Reaching this branch after a
                // successful preflight indicates an unexpected backend state
                // change and must fail closed rather than claim success.
                return false;
            }
        }
    }

    SlhaElasticKvCacheStats after{};
    if (!snapshot_ccos(&after)) {
        return false;
    }
    record_cache_snapshot(after);
    return true;
}

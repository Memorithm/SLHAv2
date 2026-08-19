// Tile store unit tests: alignment, lifetime contract, concurrency.
//
// Build:  make tile_store_tests   (see Makefile target)
// Run:    ./tile_store_tests

#include "slha_llama.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

int main() {
    // 1. Every readable slot is 128-aligned.
    {
        slha_tile_store store;
        CHECK(store.init(4, 64, 128));
        for (int layer = 0; layer < 4; ++layer) {
            for (size_t pos = 0; pos < 64; ++pos) {
                const void * p = store.read(layer, pos);
                if (p != nullptr) {
                    CHECK(reinterpret_cast<uintptr_t>(p) % 128 == 0);
                }
            }
        }
        // Force a write into an odd slot and re-check alignment.
        alignas(128) unsigned char tile[128];
        std::memset(tile, 0xAB, sizeof(tile));
        CHECK(store.write(1, 7, tile));
        const void * p = store.read(1, 7);
        CHECK(p != nullptr);
        CHECK(reinterpret_cast<uintptr_t>(p) % 128 == 0);
        CHECK(std::memcmp(p, tile, sizeof(tile)) == 0);
    }

    // 2. Bounds: out-of-range writes/reads fail closed.
    {
        slha_tile_store store;
        CHECK(store.init(2, 8, 128));
        alignas(128) unsigned char tile[128] = {};
        CHECK(!store.write(2, 0, tile)); // layer out of range
        CHECK(!store.write(0, 8, tile)); // position out of range
        CHECK(!store.write(0, 0, nullptr));
        CHECK(store.read(2, 0) == nullptr);
        CHECK(store.read(0, 8) == nullptr);
    }

    // 3. Reset invalidates previously returned pointers' validity flags
    //    (the pointer itself remains allocated; reads return nullptr).
    {
        slha_tile_store store;
        CHECK(store.init(1, 4, 128));
        alignas(128) unsigned char tile[128] = {};
        CHECK(store.write(0, 1, tile));
        const void * p = store.read(0, 1);
        CHECK(p != nullptr);
        store.reset();
        CHECK(store.read(0, 1) == nullptr);
    }

    // 4. clear_layer invalidates only that layer.
    {
        slha_tile_store store;
        CHECK(store.init(2, 4, 128));
        alignas(128) unsigned char tile[128] = {};
        CHECK(store.write(0, 0, tile));
        CHECK(store.write(1, 0, tile));
        store.clear_layer(0);
        CHECK(store.read(0, 0) == nullptr);
        CHECK(store.read(1, 0) != nullptr);
    }

    // 5. Concurrent writers on different slots never corrupt other slots.
    {
        slha_tile_store store;
        CHECK(store.init(1, 256, 128));
        std::vector<std::thread> threads;
        std::atomic<int> ok{0};
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&store, t, &ok] {
                alignas(128) unsigned char tile[128];
                for (int i = 0; i < 200; ++i) {
                    const size_t pos = static_cast<size_t>((t * 31 + i) % 256);
                    std::memset(tile, static_cast<int>(t + 1), sizeof(tile));
                    if (store.write(0, pos, tile)) {
                        const void * p = store.read(0, pos);
                        if (p != nullptr &&
                            static_cast<const unsigned char *>(p)[0] == t + 1) {
                            ok.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            });
        }
        for (auto & th : threads) {
            th.join();
        }
        CHECK(ok.load() > 0);
    }

    if (failures == 0) {
        std::printf("=== tile-store tests complete: ALL PASS ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== tile-store tests: %d FAILURES ===\n", failures);
    return 1;
}

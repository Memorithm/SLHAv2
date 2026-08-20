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
        CHECK(store.read_range(0, 0, 0) == nullptr);
        CHECK(store.read_range(0, 7, 2) == nullptr);
    }

    // 3. read_range returns one aligned contiguous immutable snapshot.
    {
        slha_tile_store store;
        CHECK(store.init(1, 8, 128));
        alignas(128) unsigned char a[128];
        alignas(128) unsigned char b[128];
        alignas(128) unsigned char c[128];
        alignas(128) unsigned char replacement[128];
        std::memset(a, 0x11, sizeof(a));
        std::memset(b, 0x22, sizeof(b));
        std::memset(c, 0x33, sizeof(c));
        std::memset(replacement, 0x44, sizeof(replacement));
        CHECK(store.write(0, 2, a));
        CHECK(store.write(0, 3, b));
        CHECK(store.write(0, 4, c));

        const auto * range = static_cast<const unsigned char *>(store.read_range(0, 2, 3));
        CHECK(range != nullptr);
        if (range != nullptr) {
            CHECK(reinterpret_cast<uintptr_t>(range) % 128 == 0);
            CHECK(range[0] == 0x11);
            CHECK(range[128] == 0x22);
            CHECK(range[256] == 0x33);

            // The returned bytes are a snapshot, not an unlocked pointer into
            // mutable store storage. A later write must not mutate this range.
            CHECK(store.write(0, 3, replacement));
            CHECK(range[128] == 0x22);
        }

        const auto * refreshed = static_cast<const unsigned char *>(store.read_range(0, 2, 3));
        CHECK(refreshed != nullptr);
        if (refreshed != nullptr) {
            CHECK(refreshed[0] == 0x11);
            CHECK(refreshed[128] == 0x44);
            CHECK(refreshed[256] == 0x33);
        }

        // A range containing any invalid slot fails closed as a whole.
        CHECK(store.read_range(0, 1, 4) == nullptr);
    }

    // 4. Reset invalidates previously returned pointers' validity flags
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

    // 5. clear_layer invalidates only that layer.
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

    // 6. Concurrent writers on different slots never corrupt other slots.
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

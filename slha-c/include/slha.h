#ifndef SLHA_H
#define SLHA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SLHA_D_C 128
#define SLHA_D_S 256
#define SLHA_LATENT_BYTES 64
#define SLHA_RESIDUAL_WORDS 4
#define SLHA_N_GROUPS 8

/**
 * SLHA v2 Tile structure.
 *
 * Total size: exactly 128 bytes. Alignment must match the Rust build:
 *   - 64 bytes by default (all x86_64 and AArch64/Neoverse parts we target),
 *   - 128 bytes when the Rust crate was built with `cfg(cache_line_128)`
 *     (hosts with genuine 128-byte data cache lines, e.g. Apple Silicon).
 * To pair with a 128-byte Rust build, compile the C side with
 * `-DSLHA_CACHE_LINE_128=1` (or `#define SLHA_CACHE_LINE_128 1` before
 * including this header). The size is 128 either way (a multiple of both
 * alignments), so the field layout and offsets are identical regardless.
 */
#if defined(SLHA_CACHE_LINE_128) && SLHA_CACHE_LINE_128
#  define SLHA_TILE_ALIGN 128
#else
#  define SLHA_TILE_ALIGN 64
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define SLHA_ALIGNED(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#  define SLHA_ALIGNED(x) __declspec(align(x))
#else
#  define SLHA_ALIGNED(x)
#endif

typedef struct SLHA_ALIGNED(SLHA_TILE_ALIGN) {
    uint8_t latent_kv[SLHA_LATENT_BYTES];
    uint64_t residual_bitmap[SLHA_RESIDUAL_WORDS];
    float scale;
    float dynamic_lambda;
    float residual_sigma;
    uint32_t token_id;
    uint32_t position;
    uint16_t head_id;
    uint16_t flags;
    uint8_t group_scales[SLHA_N_GROUPS];
} SciRustSlhaTile;

typedef struct SlhaContext SlhaContext;

/**
 * Initialize the SLHA environment.
 */
SlhaContext* slha_init();

/**
 * Process a single tile and compute the score.
 *
 * Returns 0 on success, negative values on error.
 */
int32_t slha_process_tile(
    const SciRustSlhaTile* tile,
    const float* q_coarse,
    const uint64_t* q_sign,
    float* score_out
);

/**
 * Run the self-audit and return a JSON string.
 * The caller must free the string using slha_free_string.
 */
char* slha_audit();

/**
 * Free a string allocated by the library.
 */
void slha_free_string(char* s);

/* ------------------------------------------------------------------------- *
 * Codec path — load a learned projection, encode a d-dim key into a tile,
 * decode a tile's latent back to d-dim space. See docs/INTEGRATION.md.
 * ------------------------------------------------------------------------- */

/** Opaque handle to a loaded SLHA projection model (.slhw). */
typedef struct SlhaModel SlhaModel;

/**
 * Load a projection model from a .slhw file. Returns NULL on error.
 * Release with slha_weights_free.
 */
SlhaModel* slha_weights_load(const char* path);

/** The projection input dimension d (key/query length). 0 if model is NULL. */
size_t slha_model_dim(const SlhaModel* model);

/**
 * Encode a d-dim key into a 128-byte tile.
 * codec: 0=int4-single 1=int4-grouped 2=nf4 3=mixed 4=tq3 5=mix3.
 * Returns 0 on success; -1 null arg, -2 panic, -3 dim mismatch, -4 bad codec.
 */
int32_t slha_encode_key(
    const SlhaModel* model,
    const float* key,
    size_t d,
    uint32_t pos,
    int32_t codec,
    SciRustSlhaTile* out_tile);

/**
 * Decode a tile's latent back to the original d-dim space (reconstruct).
 * out receives d floats. Returns 0 on success; -1 null, -2 panic, -3 dim.
 */
int32_t slha_decode_latent(
    const SlhaModel* model,
    const SciRustSlhaTile* tile,
    float* out,
    size_t d);

/** Release a model from slha_weights_load. NULL is a no-op. */
void slha_weights_free(SlhaModel* model);

#ifdef __cplusplus
}
#endif

#endif /* SLHA_H */

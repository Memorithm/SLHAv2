#ifndef SLHA_H
#define SLHA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SLHA_ABI_VERSION 1u

#define SLHA_D_C 128u
#define SLHA_D_S 256u
#define SLHA_LATENT_BYTES 64u
#define SLHA_RESIDUAL_WORDS 4u
#define SLHA_N_GROUPS 8u

/* Stable status codes. */
#define SLHA_OK 0
#define SLHA_ERR_NULL (-1)
#define SLHA_ERR_PANIC (-2)
#define SLHA_ERR_DIMENSION (-3)
#define SLHA_ERR_CODEC (-4)
#define SLHA_ERR_NONFINITE (-5)
#define SLHA_ERR_INVALID_TILE (-6)
#define SLHA_ERR_INVALID_HANDLE (-7)
#define SLHA_ERR_IO (-8)
#define SLHA_ERR_UTF8 (-9)

/* Codec identifiers accepted by slha_encode_key. */
#define SLHA_CODEC_INT4_SINGLE 0
#define SLHA_CODEC_INT4_GROUPED 1
#define SLHA_CODEC_NF4 2
#define SLHA_CODEC_MIXED 3
#define SLHA_CODEC_TQ3 4
#define SLHA_CODEC_MIX3 5

/* Tile flags. */
#define SLHA_FLAG_HOT 0u
#define SLHA_FLAG_WARM (1u << 0)
#define SLHA_FLAG_NF4 (1u << 1)
#define SLHA_FLAG_MIXED (1u << 2)
#define SLHA_FLAG_TQ3 (1u << 3)
#define SLHA_FLAG_TQ3_NOCORR (1u << 4)
#define SLHA_FLAG_MIX3 (1u << 5)

/*
 * The Rust build selects 64- or 128-byte tile alignment. Define
 * SLHA_CACHE_LINE_128=1 when linking against a Rust library whose
 * slha_tile_align() returns 128.
 */
#if defined(SLHA_CACHE_LINE_128) && SLHA_CACHE_LINE_128
#  define SLHA_TILE_ALIGN 128u
#else
#  define SLHA_TILE_ALIGN 64u
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define SLHA_ALIGNED(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#  define SLHA_ALIGNED(x) __declspec(align(x))
#else
#  define SLHA_ALIGNED(x)
#endif

#if defined(__cplusplus)
#  define SLHA_STATIC_ASSERT(condition, message) static_assert((condition), message)
#  define SLHA_ALIGNOF(type) alignof(type)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define SLHA_STATIC_ASSERT(condition, message) _Static_assert((condition), message)
#  define SLHA_ALIGNOF(type) _Alignof(type)
#elif defined(_MSC_VER)
#  define SLHA_STATIC_ASSERT(condition, message) \
    typedef char slha_static_assertion_##__LINE__[(condition) ? 1 : -1]
#  define SLHA_ALIGNOF(type) __alignof(type)
#else
#  define SLHA_STATIC_ASSERT(condition, message)
#endif

/**
 * SLHA v2 tile.
 *
 * The byte layout is stable. Check slha_tile_size() and slha_tile_align() at
 * startup before exchanging tiles with a dynamically loaded library.
 */
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

SLHA_STATIC_ASSERT(sizeof(SciRustSlhaTile) == 128u, "SciRustSlhaTile must be 128 bytes");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, latent_kv) == 0u, "latent_kv offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, residual_bitmap) == 64u, "residual offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, scale) == 96u, "scale offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, dynamic_lambda) == 100u, "lambda offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, residual_sigma) == 104u, "sigma offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, token_id) == 108u, "token_id offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, position) == 112u, "position offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, head_id) == 116u, "head_id offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, flags) == 118u, "flags offset mismatch");
SLHA_STATIC_ASSERT(offsetof(SciRustSlhaTile, group_scales) == 120u, "group_scales offset mismatch");

#if defined(SLHA_ALIGNOF)
SLHA_STATIC_ASSERT(
    SLHA_ALIGNOF(SciRustSlhaTile) == SLHA_TILE_ALIGN,
    "SciRustSlhaTile alignment mismatch"
);
#endif

typedef struct SlhaContext SlhaContext;
typedef struct SlhaModel SlhaModel;

/* ABI and layout introspection. */
uint32_t slha_abi_version(void);
size_t slha_tile_size(void);
size_t slha_tile_align(void);

/**
 * Return the current thread's last error message.
 *
 * The pointer is owned by the library, must not be freed, and remains valid
 * until another SLHA call updates the error state on the same thread.
 */
const char* slha_last_error_message(void);
void slha_clear_error(void);

/**
 * Return the process-wide stateless context handle.
 * The pointer is stable for the lifetime of the loaded library.
 */
SlhaContext* slha_init(void);

/**
 * Validate a context returned by slha_init.
 * The current stateless implementation owns no per-context resources, so this
 * is a no-op for the correct handle. NULL is accepted.
 */
int32_t slha_shutdown(SlhaContext* context);

/**
 * Score one tile. Unaligned input/output storage is accepted.
 * score_out is not modified on failure.
 */
int32_t slha_process_tile(
    const SciRustSlhaTile* tile,
    const float* q_coarse,
    const uint64_t* q_sign,
    float* score_out
);

/**
 * Run the self-audit and return a JSON string.
 * Release the returned pointer exactly once with slha_free_string.
 */
char* slha_audit(void);

/** Release a string returned by slha_audit. NULL is a no-op. */
void slha_free_string(char* string);

/**
 * Load a projection model from a .slhw file.
 * Returns NULL on error; call slha_last_error_message for details.
 */
SlhaModel* slha_weights_load(const char* path);

/**
 * Return the model's input dimension.
 * Returns 0 on error; call slha_last_error_message for details.
 */
size_t slha_model_dim(const SlhaModel* model);

/**
 * Encode a d-dimensional key into one tile.
 *
 * codec must be one of SLHA_CODEC_*. out_tile is not modified on failure.
 * Unaligned key and output storage are accepted.
 */
int32_t slha_encode_key(
    const SlhaModel* model,
    const float* key,
    size_t d,
    uint32_t position,
    int32_t codec,
    SciRustSlhaTile* out_tile
);

/**
 * Decode and reconstruct a tile into d floats.
 *
 * out is not modified on failure. Unaligned tile and output storage are
 * accepted.
 */
int32_t slha_decode_latent(
    const SlhaModel* model,
    const SciRustSlhaTile* tile,
    float* out,
    size_t d
);

/**
 * Decode and reconstruct a tile's full key vector (latent + residual sign
 * sketch) into d floats.
 *
 * out is not modified on failure. Unaligned tile and output storage are
 * accepted.
 */
int32_t slha_decode_key(
    const SlhaModel* model,
    const SciRustSlhaTile* tile,
    float* out,
    size_t d
);

/**
 * Release a model and return an explicit status. NULL is a no-op.
 */
int32_t slha_weights_release(SlhaModel* model);

/**
 * Compatibility wrapper for existing callers.
 * Prefer slha_weights_release when an explicit status is needed.
 */
void slha_weights_free(SlhaModel* model);

#ifdef __cplusplus
}
#endif

#endif /* SLHA_H */

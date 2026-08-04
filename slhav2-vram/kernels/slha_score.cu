// SLHAv2 tile scoring kernel — exact scirust semantics.
// One thread scores one tile. 128 bytes per tile.

#define D_C 128
#define RESIDUAL_WORDS 4
#define D_S 256
#define LATENT_BYTES 64
#define TILE_BYTES 128
#define N_GROUPS 8
#define GROUP_DIM 16

#define FLAG_WARM      (1u << 0)
#define FLAG_NF4       (1u << 1)
#define FLAG_MIXED     (1u << 2)
#define FLAG_TQ3       (1u << 3)
#define FLAG_TQ3_NOCORR (1u << 4)
#define FLAG_MIX3      (1u << 5)

#define MIXED_HI_DIMS 8
#define MIXED_DIMS 120
#define MIXED_LO_DIMS 112
#define MIXED_LO_GROUPS 7
#define TQ3_CODE_BYTES 48
#define TQ3_CORR_BYTES 16
#define TQ3_HALF_RANGE 3.5f
#define TQ3_CORRECTION 0.25f
#define MIX3_CODES_OFF 8
#define MIX3_CORR_OFF 50

// NF4 codebook matching scirust exactly, in __constant__ memory (a single
// 64-byte read-only bank shared by all threads instead of a per-thread local
// array).
__constant__ float NF4_CODEBOOK[16] = {
    -1.0f, -0.7075f, -0.5421f, -0.4165f,
    -0.3108f, -0.2158f, -0.1272f, -0.0421f,
    0.0421f, 0.1272f, 0.2158f, 0.3108f,
    0.4165f, 0.5421f, 0.7075f, 1.0f
};

// NOTE: the `gs[]` array passed here already contains the *effective* group
// scale `scale * group_scales[g] / 255` (hoisted before the loop). The
// dequant functions must multiply by `gs[...]` only — multiplying by `scale`
// as well would apply the global scale twice and produce wrong scores for
// any tile with `scale != 1`.

static __device__ __forceinline__ int flg(unsigned short flags, unsigned short mask) {
    return (flags & mask) == mask;
}

static __device__ __forceinline__ float dequant_int4_at(
    const unsigned char* latent, int d, const float* gs
) {
    int byte_idx = d >> 1;
    unsigned char byte = latent[byte_idx];
    int nib = (d & 1) ? (byte >> 4) : (byte & 0x0F);
    float level = (float)(nib - 8);
    return level * gs[d / GROUP_DIM];
}

static __device__ __forceinline__ float dequant_nf4_at(
    const unsigned char* latent, int d, const float* gs
) {
    int byte_idx = d >> 1;
    unsigned char byte = latent[byte_idx];
    int nib = (d & 1) ? (byte >> 4) : (byte & 0x0F);
    return NF4_CODEBOOK[nib] * gs[d / GROUP_DIM];
}

// Mixed-precision: 8-bit head (zero-point 128, scale gs[0]) + 4-bit body
// (gs[1..]) + dropped tail. Mirrors scirust's dequant_at_mixed.
static __device__ __forceinline__ float dequant_mixed_at(
    const unsigned char* latent, int d, const float* gs
) {
    if (d < MIXED_HI_DIMS) {
        float level = (float)((int)latent[d] - 128);
        return level * gs[0];
    } else if (d < MIXED_DIMS) {
        int ld = d - MIXED_HI_DIMS;
        int byte = latent[MIXED_HI_DIMS + (ld >> 1)];
        int nib = (ld & 1) ? (byte >> 4) : (byte & 0x0F);
        return (float)(nib - 8) * gs[1 + ld / GROUP_DIM];
    }
    return 0.0f;
}

// TQ3: 3-bit code + 1-bit correction plane. Mirrors scirust's dequant_at_tq3.
static __device__ __forceinline__ float dequant_tq3_at(
    const unsigned char* latent, int d, const float* gs, unsigned short flags
) {
    int bit = 3 * d;
    int byte = bit >> 3;
    int shift = bit & 7;
    int lo = latent[byte];
    int hi = (byte + 1 < TQ3_CODE_BYTES) ? latent[byte + 1] << 8 : 0;
    int code = ((lo | hi) >> shift) & 0x7;
    float level = (float)code - TQ3_HALF_RANGE;
    if (!flg(flags, FLAG_TQ3_NOCORR)) {
        int corr = (latent[TQ3_CODE_BYTES + (d >> 3)] >> (d & 7)) & 1;
        level += (corr == 1) ? TQ3_CORRECTION : -TQ3_CORRECTION;
    }
    return level * gs[d / GROUP_DIM];
}

// MIX3: mixed head + TQ3 body. Mirrors scirust's dequant_at_mix3.
static __device__ __forceinline__ float dequant_mix3_at(
    const unsigned char* latent, int d, const float* gs, unsigned short flags
) {
    if (d < MIXED_HI_DIMS) {
        float level = (float)((int)latent[d] - 128);
        return level * gs[0];
    }
    if (d >= MIXED_DIMS) return 0.0f;
    int ld = d - MIXED_HI_DIMS;
    int bit = 3 * ld;
    int byte = MIX3_CODES_OFF + (bit >> 3);
    int shift = bit & 7;
    int lo = latent[byte];
    int hi = (byte + 1 < MIX3_CORR_OFF) ? latent[byte + 1] << 8 : 0;
    int code = ((lo | hi) >> shift) & 0x7;
    float level = (float)code - TQ3_HALF_RANGE;
    if (!flg(flags, FLAG_TQ3_NOCORR)) {
        int corr = (latent[MIX3_CORR_OFF + (ld >> 3)] >> (ld & 7)) & 1;
        level += (corr == 1) ? TQ3_CORRECTION : -TQ3_CORRECTION;
    }
    return level * gs[1 + ld / GROUP_DIM];
}

extern "C" __global__ void slha_score_kernel(
    const float* __restrict__ q_coarse,
    const unsigned long long* __restrict__ q_sign,
    const unsigned char* __restrict__ tiles_serialized,
    float* __restrict__ scores_out,
    int num_tiles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tiles) return;

    const unsigned char* tile_start = tiles_serialized + (size_t)tid * TILE_BYTES;

    const unsigned char* latent = tile_start;
    const unsigned long long* residual = (const unsigned long long*)(tile_start + LATENT_BYTES);
    float scale = *(const float*)(tile_start + 96);
    float dynamic_lambda = *(const float*)(tile_start + 100);
    unsigned short flags = *(const unsigned short*)(tile_start + 118);
    const unsigned char* group_scales = tile_start + 120;

    // Hoist the per-group effective scales into registers (8 values): removes
    // the per-dim integer divide and the (1/255) rescale from the hot loop.
    float gs[N_GROUPS];
    const float inv255 = 1.0f / 255.0f;
    #pragma unroll
    for (int g = 0; g < N_GROUPS; ++g) {
        gs[g] = scale * (float)group_scales[g] * inv255;
    }

    // Coarse dot product, codec-dispatched outside the per-dim loop.
    float coarse = 0.0f;
    if (flg(flags, FLAG_MIXED)) {
        #pragma unroll 8
        for (int d = 0; d < D_C; ++d) {
            coarse += q_coarse[d] * dequant_mixed_at(latent, d, gs);
        }
    } else if (flg(flags, FLAG_TQ3)) {
        #pragma unroll 8
        for (int d = 0; d < D_C; ++d) {
            coarse += q_coarse[d] * dequant_tq3_at(latent, d, gs, flags);
        }
    } else if (flg(flags, FLAG_MIX3)) {
        #pragma unroll 8
        for (int d = 0; d < D_C; ++d) {
            coarse += q_coarse[d] * dequant_mix3_at(latent, d, gs, flags);
        }
    } else if (flg(flags, FLAG_NF4)) {
        #pragma unroll 8
        for (int d = 0; d < D_C; ++d) {
            coarse += q_coarse[d] * dequant_nf4_at(latent, d, gs);
        }
    } else {
        #pragma unroll 8
        for (int d = 0; d < D_C; ++d) {
            coarse += q_coarse[d] * dequant_int4_at(latent, d, gs);
        }
    }

    if (flg(flags, FLAG_WARM)) {
        scores_out[tid] = coarse;
        return;
    }

    // Hamming distance on residual
    unsigned int ham = 0;
    #pragma unroll
    for (int w = 0; w < RESIDUAL_WORDS; ++w) {
        ham += __popcll(q_sign[w] ^ residual[w]);
    }

    float score = coarse + dynamic_lambda * ((float)D_S - 2.0f * (float)ham);
    scores_out[tid] = score;
}

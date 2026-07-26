// SLHAv2 tile scoring kernel — exact scirust semantics.
// One thread scores one tile. 128 bytes per tile.

#define D_C 128
#define RESIDUAL_WORDS 4
#define D_S 256
#define LATENT_KV_WORDS 64
#define TILE_BYTES 128

#define FLAG_WARM  (1u << 0)
#define FLAG_NF4   (1u << 1)
#define FLAG_MIXED (1u << 2)
#define FLAG_TQ3   (1u << 3)
#define FLAG_MIX3  (1u << 4)
#define FLAG_TQ3_NOCORR (1u << 5)

#define MIXED_HI_DIMS 32
#define TQ3_HALF_RANGE 4

__constant__ float nf4_codebook[16] = {
    -1.0f, -0.7075f, -0.5421f, -0.4165f,
    -0.3108f, -0.2160f, -0.0685f, 0.0f,
    0.0685f, 0.2160f, 0.3108f, 0.4165f,
    0.5421f, 0.7075f, 1.0f, 1.0f
};

static __device__ __forceinline__ int flg(const unsigned short* flags, unsigned short mask) {
    return (*flags & mask) == mask;
}

static __device__ __forceinline__ void decode_int4_nibble(unsigned char nib, float* level) {
    *level = (float)((int)nib - 8);
}

static __device__ __forceinline__ void ext_nibble(unsigned char byte, int hi, int* nib) {
    *nib = hi ? (byte >> 4) : (byte & 0x0F);
}

// Uniform INT4 zero-point decode (dimension index-based, scirust order)
static __device__ float dequant_int4_at(const unsigned char* latent, int d, float scale,
                                         const unsigned char* gscales) {
    int byte_idx = d >> 1;
    int hi = d & 1;
    unsigned char byte = latent[byte_idx];
    int nib = hi ? (byte >> 4) : (byte & 0x0F);
    float level = (float)(nib - 8);
    int group = d / 16;
    float gs = (float)gscales[group] * (1.0f / 255.0f);
    return level * scale * gs;
}

static __device__ float dequant_nf4_at(const unsigned char* latent, int d, float scale,
                                        const unsigned char* gscales) {
    int byte_idx = d >> 1;
    int hi = d & 1;
    unsigned char byte = latent[byte_idx];
    int nib = hi ? (byte >> 4) : (byte & 0x0F);
    float level = nf4_codebook[nib];
    int group = d / 16;
    float gs = (float)gscales[group] * (1.0f / 255.0f);
    return level * scale * gs;
}

static __device__ float dequant_mixed_at(const unsigned char* latent, int d, float scale,
                                          const unsigned char* gscales) {
    int group = d / 16;
    float gs = (float)gscales[group] * (1.0f / 255.0f);
    if (d < MIXED_HI_DIMS) {
        float level = (float)((int)latent[d] - 128);
        return level * scale * gs;
    } else {
        int adj = d - MIXED_HI_DIMS;
        int byte_idx = MIXED_HI_DIMS + (adj >> 1);
        int hi = adj & 1;
        unsigned char byte = latent[byte_idx];
        int nib = hi ? (byte >> 4) : (byte & 0x0F);
        float level = (float)(nib - 8);
        return level * scale * gs;
    }
}

// 3-bit little-endian read helper (cross-byte)
static __device__ int tq3_read_bits(const unsigned char* base, int d) {
    int total_bits = d * 3;
    int byte_off = total_bits / 8;
    int bit_off = total_bits % 8;
    int needed = bit_off + 3;
    int n_read = (needed + 7) / 8;
    unsigned int val = 0;
    for (int i = 0; i < n_read; ++i) {
        int idx = byte_off + i;
        unsigned int b = 0;
        if (idx < LATENT_KV_WORDS) {
            b = (unsigned int)base[idx];
        }
        val |= b << (i * 8);
    }
    return (int)((val >> bit_off) & 0x7u);
}

static __device__ int tq3_read_correction(const unsigned char* latent, int d) {
    int corr_byte = d >> 3;
    int corr_bit = d & 7;
    unsigned char cb = latent[LATENT_KV_WORDS + corr_byte];
    return (int)((cb >> corr_bit) & 1u);
}

static __device__ float tq3_decode_value(int code, int correction) {
    if (correction) {
        if (code >= TQ3_HALF_RANGE) {
            return (float)(code + 1 - TQ3_HALF_RANGE);
        } else {
            return (float)(code - 1 + TQ3_HALF_RANGE);
        }
    } else {
        return (float)(code - TQ3_HALF_RANGE);
    }
}

static __device__ float dequant_tq3_at(const unsigned char* latent, int d, float scale,
                                        const unsigned char* gscales, unsigned short flags) {
    int code = tq3_read_bits(latent, d);
    int has_corr = !flg(&flags, FLAG_TQ3_NOCORR);
    int correction = has_corr ? tq3_read_correction(latent, d) : 0;
    float level = tq3_decode_value(code, correction);
    int group = d / 16;
    float gs = (float)gscales[group] * (1.0f / 255.0f);
    return level * scale * gs;
}

static __device__ float dequant_mix3_at(const unsigned char* latent, int d, float scale,
                                         const unsigned char* gscales) {
    int group = d / 16;
    float gs = (float)gscales[group] * (1.0f / 255.0f);
    if (d < MIXED_HI_DIMS) {
        float level = (float)((int)latent[d] - 128);
        return level * scale * gs;
    } else {
        int adj = d - MIXED_HI_DIMS;
        int code = tq3_read_bits(latent + MIXED_HI_DIMS, adj);
        int corr_byte = adj >> 3;
        int corr_bit = adj & 7;
        unsigned char cb = latent[LATENT_KV_WORDS + corr_byte];
        int correction = (int)((cb >> corr_bit) & 1u);
        float level = tq3_decode_value(code, correction);
        return level * scale * gs;
    }
}

static __device__ float dequant_at(const unsigned char* latent, int d, float scale,
                                    const unsigned char* gscales, unsigned short flags) {
    if (flg(&flags, FLAG_NF4)) {
        return dequant_nf4_at(latent, d, scale, gscales);
    } else if (flg(&flags, FLAG_MIX3)) {
        return dequant_mix3_at(latent, d, scale, gscales);
    } else if (flg(&flags, FLAG_TQ3)) {
        return dequant_tq3_at(latent, d, scale, gscales, flags);
    } else if (flg(&flags, FLAG_MIXED)) {
        return dequant_mixed_at(latent, d, scale, gscales);
    } else {
        return dequant_int4_at(latent, d, scale, gscales);
    }
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

    const unsigned char* tile = tiles_serialized + (__int128)(tid) * TILE_BYTES;

    // Parse tile header
    const unsigned char* latent = tile;
    const unsigned long long* residual = (const unsigned long long*)(tile + 64);
    float scale = *(const float*)(tile + 96);
    float dynamic_lambda = *(const float*)(tile + 100);
    unsigned short flags = *(const unsigned short*)(tile + 118);
    const unsigned char* group_scales = tile + 120;

    int warm = flg(&flags, FLAG_WARM);

    // Coarse dot product
    float coarse = 0.0f;
    for (int d = 0; d < D_C; ++d) {
        float val = dequant_at(latent, d, scale, group_scales, flags);
        coarse += q_coarse[d] * val;
    }

    if (warm) {
        scores_out[tid] = coarse;
        return;
    }

    // Hamming distance on residual
    unsigned int ham = 0;
    for (int w = 0; w < RESIDUAL_WORDS; ++w) {
        ham += __popcll(q_sign[w] ^ residual[w]);
    }

    float score = coarse + dynamic_lambda * ((float)D_S - 2.0f * (float)ham);
    scores_out[tid] = score;
}

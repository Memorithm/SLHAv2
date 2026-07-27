// SLHAv2 tile scoring kernel — exact scirust semantics.
// One thread scores one tile. 128 bytes per tile.

#define D_C 128
#define RESIDUAL_WORDS 4
#define D_S 256
#define LATENT_BYTES 64
#define TILE_BYTES 128
#define N_GROUPS 8
#define GROUP_DIM 16

#define FLAG_WARM (1u << 0)
#define FLAG_NF4  (1u << 1)

// NF4 codebook matching scirust exactly
static __device__ float nf4_lookup(int idx) {
    const float cb[16] = {
        -1.0f, -0.7075f, -0.5421f, -0.4165f,
        -0.3108f, -0.2158f, -0.1272f, -0.0421f,
        0.0421f, 0.1272f, 0.2158f, 0.3108f,
        0.4165f, 0.5421f, 0.7075f, 1.0f
    };
    return cb[idx];
}

static __device__ __forceinline__ int flg(unsigned short flags, unsigned short mask) {
    return (flags & mask) == mask;
}

static __device__ float dequant_int4_at(
    const unsigned char* latent, int d, float scale, const unsigned char* gscales
) {
    int byte_idx = d >> 1;
    unsigned char byte = latent[byte_idx];
    int nib = (d & 1) ? (byte >> 4) : (byte & 0x0F);
    float level = (float)(nib - 8);
    int group = d / GROUP_DIM;
    float gs = (float)gscales[group] * (1.0f / 255.0f);
    return level * scale * gs;
}

static __device__ float dequant_nf4_at(
    const unsigned char* latent, int d, float scale, const unsigned char* gscales
) {
    int byte_idx = d >> 1;
    unsigned char byte = latent[byte_idx];
    int nib = (d & 1) ? (byte >> 4) : (byte & 0x0F);
    float level = nf4_lookup(nib);
    int group = d / GROUP_DIM;
    float gs = (float)gscales[group] * (1.0f / 255.0f);
    return level * scale * gs;
}

static __device__ float dequant_at(
    const unsigned char* latent, int d, float scale,
    const unsigned char* gscales, unsigned short flags
) {
    if (flg(flags, FLAG_NF4)) {
        return dequant_nf4_at(latent, d, scale, gscales);
    }
    return dequant_int4_at(latent, d, scale, gscales);
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

    // Coarse dot product
    float coarse = 0.0f;
    for (int d = 0; d < D_C; ++d) {
        float val = dequant_at(latent, d, scale, group_scales, flags);
        coarse += q_coarse[d] * val;
    }

    if (flg(flags, FLAG_WARM)) {
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

/*
 * lowrank_turboquant.cu
 *
 * Low-rank matrix-multiply with on-the-fly TurboQuant INT4 dequantization.
 * Targets sm_89 (Ada Lovelace, RTX 4060).
 *
 * TurboQuant format (matching SLHAv2):
 *   - Weights stored as packed INT4 (two 4-bit values per byte).
 *   - Per-group symmetric scaling: GROUP_SIZE elements share one float32 scale.
 *   - Dequantized value = (int4_nibble - 8) * group_scale.
 *
 * Kernel:  Y[M, N] = dequant(W_q[N, K]) · X[M, K]
 */

#include <stdint.h>

#define BM 128
#define BN 128
#define BK 32
#define GROUP_SIZE 16
#define THREADS_PER_BLOCK 256

/*
 * Each thread computes (BM / 16) * (BN / 16) = 8 * 8 = 64 output elements.
 */
#define ROWS_PER_THR  (BM / 16)
#define COLS_PER_THR  (BN / 16)

__forceinline__ __device__ float dequant_int4(uint8_t packed, int nibble_idx, float scale) {
    int nibble = (nibble_idx == 0)
        ? (packed & 0x0F)
        : ((packed >> 4) & 0x0F);
    int signed_val = nibble - ((nibble & 0x08) ? 16 : 0);
    return (float)signed_val * scale;
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
lowrank_turboquant_matmul(
    const float * __restrict__ input,
    const uint8_t * __restrict__ weights_q,
    const float * __restrict__ scales,
    float * __restrict__ output,
    int M,
    int N,
    int K
) {
    __shared__ float smem_input[BM * BK];
    __shared__ uint8_t smem_weight[BN * BK / 2];
    __shared__ float smem_scale[(BN * BK) / GROUP_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row_base = bx * BM;
    int col_base = by * BN;
    int tid = threadIdx.x;

    float acc[ROWS_PER_THR * COLS_PER_THR];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THR * COLS_PER_THR; i++) {
        acc[i] = 0.0f;
    }

    int thr_row = tid / 16;
    int thr_col = tid % 16;

    for (int k_base = 0; k_base < K; k_base += BK) {
        /* --- Load input tile BM x BK into smem_input --- */
        for (int i = tid; i < BM * BK; i += THREADS_PER_BLOCK) {
            int r = i / BK;
            int c = i % BK;
            int row = row_base + r;
            int col = k_base + c;
            smem_input[i] = (row < M && col < K) ? input[row * K + col] : 0.0f;
        }

        /* --- Load weight tile BN x BK (packed INT4) into smem_weight --- */
        int packed_per_row = BK / 2;
        for (int i = tid; i < BN * packed_per_row; i += THREADS_PER_BLOCK) {
            int r = i / packed_per_row;
            int c = i % packed_per_row;
            int row = col_base + r;
            int col_byte = k_base / 2 + c;
            smem_weight[i] = (row < N && (col_byte * 2) < K)
                ? weights_q[row * K / 2 + col_byte]
                : 0;
        }

        /* --- Load scales into smem_scale --- */
        int groups_per_row = BK / GROUP_SIZE;
        for (int i = tid; i < BN * groups_per_row; i += THREADS_PER_BLOCK) {
            int r = i / groups_per_row;
            int c = i % groups_per_row;
            int row = col_base + r;
            smem_scale[r * groups_per_row + c] = (row < N)
                ? scales[row * (K / GROUP_SIZE) + (k_base / GROUP_SIZE) + c]
                : 0.0f;
        }

        __syncthreads();

        /* --- Compute partial products --- */
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THR; r++) {
            int out_row = thr_row * ROWS_PER_THR + r;
            int abs_row = row_base + out_row;
            if (abs_row >= M) continue;

            #pragma unroll
            for (int c = 0; c < COLS_PER_THR; c++) {
                int out_col = thr_col * COLS_PER_THR + c;
                int abs_col = col_base + out_col;
                if (abs_col >= N) continue;

                float sum = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk++) {
                    float x_val = smem_input[out_row * BK + kk];
                    uint8_t packed = smem_weight[out_col * (BK / 2) + kk / 2];
                    float s = smem_scale[out_col * (BK / GROUP_SIZE) + kk / GROUP_SIZE];
                    sum += x_val * dequant_int4(packed, kk % 2, s);
                }
                acc[r * COLS_PER_THR + c] += sum;
            }
        }

        __syncthreads();
    }

    /* --- Write results --- */
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THR; r++) {
        int out_row = thr_row * ROWS_PER_THR + r;
        int abs_row = row_base + out_row;
        if (abs_row >= M) continue;

        #pragma unroll
        for (int c = 0; c < COLS_PER_THR; c++) {
            int out_col = thr_col * COLS_PER_THR + c;
            int abs_col = col_base + out_col;
            if (abs_col >= N) continue;
            output[abs_row * N + abs_col] = acc[r * COLS_PER_THR + c];
        }
    }
}

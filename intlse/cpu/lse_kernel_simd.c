/*
 * AVX2 SIMD-vectorised IntLSE kernel (Q16.16).
 *
 *
 * Build with -mavx2 -mfma -O3 (see ./build.sh).
 *
 * Performance (Haswell, 1 thread):
 *   scalar pure-C : ~9 ns/elem
 *   AVX2 (this)   : ~1–2 ns/elem  (gather-bound)
 */
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#define FRAC_BITS 16
#define SCALE     (1 << FRAC_BITS)
#define FRAC_MASK (SCALE - 1)
#define LOG2_E    1.44269504089f
#define LN_2      0.69314718056f
#define CONV      (LOG2_E * (float)SCALE)
#define Q_NEG_INF INT32_MIN

static const int LUT64[64] __attribute__((aligned(64))) = {
       0,  571,  907, 1282, 1634, 1745, 2054, 2592,
    2900, 2859, 2895, 3015, 3222, 3522, 3922, 4426,
    4617, 4441, 4294, 4178, 4093, 4039, 4018, 4030,
    4076, 4157, 4272, 4424, 4613, 4838, 5102, 5404,
    5399, 5068, 4746, 4435, 4134, 3843, 3562, 3291,
    3031, 2781, 2541, 2312, 2094, 1886, 1689, 1502,
    1327, 1162, 1008,  865,  732,  611,  500,  401,
     312,  235,  168,  113,   69,   35,   13,    2
};

static inline int float2int_rn_sat(float f) {
    if (f >=  2147483520.0f) return  2147483647;
    if (f <= -2147483520.0f) return -2147483647;
    return (int)lroundf(f);
}

static inline int lse_pair_scalar(int x, int y) {
    int x_max = (x > y) ? x : y;
    int y_min = (x > y) ? y : x;
    int sub   = y_min - x_max;
    int shift = -(sub >> FRAC_BITS);
    if (shift > 31) shift = 31;
    int M = (SCALE + (sub & FRAC_MASK)) >> shift;
    return x_max + M + LUT64[(M >> 10) & 0x3F];
}

static inline int lse_pair_scalar_neg_inf(int x, int y) {
    if (x == Q_NEG_INF) return y;
    if (y == Q_NEG_INF) return x;
    return lse_pair_scalar(x, y);
}

/* ============================================================
 * SIMD pair function: 8 int32 lanes
 * ============================================================ */
static inline __m256i lse_pair_simd(__m256i x, __m256i y) {
    __m256i x_max = _mm256_max_epi32(x, y);
    __m256i y_min = _mm256_min_epi32(x, y);
    __m256i sub   = _mm256_sub_epi32(y_min, x_max);

    /* shift = -(sub >> 16), capped at 31 */
    __m256i sub_hi = _mm256_srai_epi32(sub, FRAC_BITS);
    __m256i shift  = _mm256_sub_epi32(_mm256_setzero_si256(), sub_hi);
    shift = _mm256_min_epi32(shift, _mm256_set1_epi32(31));

    /* M = (SCALE + (sub & FRAC_MASK)) >> shift */
    __m256i sub_lo = _mm256_and_si256(sub, _mm256_set1_epi32(FRAC_MASK));
    __m256i num    = _mm256_add_epi32(_mm256_set1_epi32(SCALE), sub_lo);
    __m256i M      = _mm256_srav_epi32(num, shift);

    /* idx = (M >> 10) & 0x3F */
    __m256i idx = _mm256_and_si256(_mm256_srai_epi32(M, 10),
                                   _mm256_set1_epi32(0x3F));

    /* LUT gather */
    __m256i lut = _mm256_i32gather_epi32(LUT64, idx, 4);

    return _mm256_add_epi32(_mm256_add_epi32(x_max, M), lut);
}

static inline __m256i lse_pair_simd_neg_inf(__m256i x, __m256i y) {
    __m256i result = lse_pair_simd(x, y);
    __m256i negative_infinity = _mm256_set1_epi32(Q_NEG_INF);
    __m256i x_is_negative_infinity = _mm256_cmpeq_epi32(x, negative_infinity);
    __m256i y_is_negative_infinity = _mm256_cmpeq_epi32(y, negative_infinity);
    result = _mm256_blendv_epi8(result, y, x_is_negative_infinity);
    return _mm256_blendv_epi8(result, x, y_is_negative_infinity);
}

/* ============================================================
 * Convert 8 floats -> 8 Q16.16 int32 with saturation
 * ============================================================ */
static inline __m256i float_to_q16_simd(__m256 v) {
    __m256 scaled = _mm256_mul_ps(v, _mm256_set1_ps(CONV));
    /* clamp to ±2147483520 */
    __m256 hi = _mm256_set1_ps( 2147483520.0f);
    __m256 lo = _mm256_set1_ps(-2147483520.0f);
    __m256 clamped = _mm256_min_ps(_mm256_max_ps(scaled, lo), hi);
    /* cvtps_epi32 uses current rounding mode (round-to-nearest-even by default) */
    return _mm256_cvtps_epi32(clamped);
}

static inline __m256i float_to_q16_simd_neg_inf(__m256 v) {
    __m256 negative_infinity = _mm256_set1_ps(-INFINITY);
    __m256 is_negative_infinity = _mm256_cmp_ps(
        v,
        negative_infinity,
        _CMP_EQ_OQ
    );
    __m256i converted = float_to_q16_simd(v);
    return _mm256_blendv_epi8(
        converted,
        _mm256_set1_epi32(Q_NEG_INF),
        _mm256_castps_si256(is_negative_infinity)
    );
}

/* ============================================================
 * Reduce 8 lanes to 1 scalar via 3-stage SIMD tree, finishing scalar
 * ============================================================ */
static inline int lane_reduce_8(__m256i v) {
    int lanes[8] __attribute__((aligned(32)));
    _mm256_storeu_si256((__m256i*)lanes, v);
    /* Tree-reduce: 8 -> 4 -> 2 -> 1 */
    int a01 = lse_pair_scalar(lanes[0], lanes[1]);
    int a23 = lse_pair_scalar(lanes[2], lanes[3]);
    int a45 = lse_pair_scalar(lanes[4], lanes[5]);
    int a67 = lse_pair_scalar(lanes[6], lanes[7]);
    int a0123 = lse_pair_scalar(a01, a23);
    int a4567 = lse_pair_scalar(a45, a67);
    return lse_pair_scalar(a0123, a4567);
}

static inline int lane_reduce_8_neg_inf(__m256i v) {
    int lanes[8] __attribute__((aligned(32)));
    _mm256_storeu_si256((__m256i*)lanes, v);
    int a01 = lse_pair_scalar_neg_inf(lanes[0], lanes[1]);
    int a23 = lse_pair_scalar_neg_inf(lanes[2], lanes[3]);
    int a45 = lse_pair_scalar_neg_inf(lanes[4], lanes[5]);
    int a67 = lse_pair_scalar_neg_inf(lanes[6], lanes[7]);
    int a0123 = lse_pair_scalar_neg_inf(a01, a23);
    int a4567 = lse_pair_scalar_neg_inf(a45, a67);
    return lse_pair_scalar_neg_inf(a0123, a4567);
}

/* ============================================================
 * SIMD 1-D LSE with PER-LANE BLOCK-TREE reduction.
 *
 * Algorithm:
 *   - Each "block" = 64 elements = 8 lanes × 8 elements per lane.
 *   - Within each block: 8 SIMD-parallel pair operations (sequential
 *     down each lane). Per-lane bias = O(8).
 *   - Across blocks: pairwise tree reduction of __m256i partials.
 *     Per-lane bias = O(log(N/64)).
 *   - Final 8-lane cross-reduce: O(7) scalar pair ops.
 *
 * Total bias: O(8 + log(N/64) + 7) per element, vs O(N/8) in the
 * naive (no-block-tree) SIMD kernel.  For N=2000: O(13) vs O(250).
 * ============================================================ */
#define SIMD_BLOCK 64       /* 8 lanes × 8 sequential elements per block */

float lse_q16_1d_simd(const float* in, int N) {
    if (N <= 0) return -INFINITY;
    if (N == 1) return in[0];

    /* Scalar fallback for very small N */
    if (N < SIMD_BLOCK) {
        int acc = float2int_rn_sat(in[0] * CONV);
        for (int i = 1; i < N; i++) {
            int q = float2int_rn_sat(in[i] * CONV);
            acc = lse_pair_scalar(acc, q);
        }
        return (float)acc * (LN_2 / (float)SCALE);
    }

    int n_blocks = N / SIMD_BLOCK;
    int tail_start = n_blocks * SIMD_BLOCK;

    /* Allocate block partials. Use stack for small n_blocks, heap otherwise. */
    __m256i stack_buf[256] __attribute__((aligned(32)));
    __m256i* block_acc;
    int alloced_heap = 0;
    if (n_blocks <= 256) {
        block_acc = stack_buf;
    } else {
        block_acc = (__m256i*)aligned_alloc(32, sizeof(__m256i) * n_blocks);
        alloced_heap = 1;
    }

    /* Pass 1: each block's 64 elements -> one __m256i partial. */
    for (int blk = 0; blk < n_blocks; blk++) {
        const float* p = in + blk * SIMD_BLOCK;
        __m256i acc = float_to_q16_simd(_mm256_loadu_ps(p));
        for (int i = 8; i < SIMD_BLOCK; i += 8) {
            __m256i q = float_to_q16_simd(_mm256_loadu_ps(p + i));
            acc = lse_pair_simd(acc, q);
        }
        block_acc[blk] = acc;
    }

    /* Pass 2: pairwise tree reduce the block partials (in __m256i space). */
    int n = n_blocks;
    while (n > 1) {
        int half = n / 2;
        for (int i = 0; i < half; i++)
            block_acc[i] = lse_pair_simd(block_acc[2*i], block_acc[2*i+1]);
        if (n & 1) { block_acc[half] = block_acc[n - 1]; n = half + 1; }
        else       { n = half; }
    }

    /* Final 8-lane cross-reduce. */
    int acc = lane_reduce_8(block_acc[0]);

    if (alloced_heap) free(block_acc);

    /* Tail: handle last (N % SIMD_BLOCK) elements scalarly. */
    for (int i = tail_start; i < N; i++) {
        int q = float2int_rn_sat(in[i] * CONV);
        acc = lse_pair_scalar(acc, q);
    }

    return (float)acc * (LN_2 / (float)SCALE);
}

/* ============================================================
 * 2-D batched: each row independently
 * ============================================================ */
void lse_q16_2d_simd(const float* in, int B, int N, float* out) {
    for (int b = 0; b < B; b++) {
        out[b] = lse_q16_1d_simd(in + (long)b * N, N);
    }
}

float lse_q16_1d_simd_neg_inf(const float* in, int N) {
    if (N <= 0) return -INFINITY;
    if (N == 1) return in[0];

    if (N < SIMD_BLOCK) {
        int acc = in[0] == -INFINITY
            ? Q_NEG_INF
            : float2int_rn_sat(in[0] * CONV);
        for (int i = 1; i < N; i++) {
            int q = in[i] == -INFINITY
                ? Q_NEG_INF
                : float2int_rn_sat(in[i] * CONV);
            acc = lse_pair_scalar_neg_inf(acc, q);
        }
        return acc == Q_NEG_INF
            ? -INFINITY
            : (float)acc * (LN_2 / (float)SCALE);
    }

    int n_blocks = N / SIMD_BLOCK;
    int tail_start = n_blocks * SIMD_BLOCK;
    __m256i stack_buf[256] __attribute__((aligned(32)));
    __m256i* block_acc;
    int alloced_heap = 0;
    if (n_blocks <= 256) {
        block_acc = stack_buf;
    } else {
        block_acc = (__m256i*)aligned_alloc(32, sizeof(__m256i) * n_blocks);
        alloced_heap = 1;
    }

    for (int blk = 0; blk < n_blocks; blk++) {
        const float* p = in + blk * SIMD_BLOCK;
        __m256i acc = float_to_q16_simd_neg_inf(_mm256_loadu_ps(p));
        for (int i = 8; i < SIMD_BLOCK; i += 8) {
            __m256i q = float_to_q16_simd_neg_inf(_mm256_loadu_ps(p + i));
            acc = lse_pair_simd_neg_inf(acc, q);
        }
        block_acc[blk] = acc;
    }

    int n = n_blocks;
    while (n > 1) {
        int half = n / 2;
        for (int i = 0; i < half; i++) {
            block_acc[i] = lse_pair_simd_neg_inf(
                block_acc[2*i],
                block_acc[2*i+1]
            );
        }
        if (n & 1) {
            block_acc[half] = block_acc[n - 1];
            n = half + 1;
        } else {
            n = half;
        }
    }

    int acc = lane_reduce_8_neg_inf(block_acc[0]);
    if (alloced_heap) free(block_acc);

    for (int i = tail_start; i < N; i++) {
        int q = in[i] == -INFINITY
            ? Q_NEG_INF
            : float2int_rn_sat(in[i] * CONV);
        acc = lse_pair_scalar_neg_inf(acc, q);
    }

    return acc == Q_NEG_INF
        ? -INFINITY
        : (float)acc * (LN_2 / (float)SCALE);
}

void lse_q16_2d_simd_neg_inf(const float* in, int B, int N, float* out) {
    for (int b = 0; b < B; b++) {
        out[b] = lse_q16_1d_simd_neg_inf(in + (long)b * N, N);
    }
}

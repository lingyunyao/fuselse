/*
 * Q10.6 fixed-point batched logsumexp CUDA kernel (int16 storage).
 *
 * Reduced-bit twin of lse_kernel.cu (Q16.16, int32)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

/* ---- Fixed-point parameters ---- */
#define FRAC_BITS 6
#define SCALE     (1 << FRAC_BITS)
#define FRAC_MASK (SCALE - 1)

#define LOG2_E 1.44269504089f
#define LN_2   0.69314718056f

/* 64-entry correction LUT (Mitchell error per bin of M), scaled to Q10.6.
 * Derived by rounding the Q16.16 LUT divided by 2^10 = 1024, preserving
 * the same per-bin correction shape at the lower fractional resolution.
 */
__device__ __constant__ int LUT64[64] = {
    0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4,
    5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1,
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* ---- Core pair-wise LSE in Q10.6 base-2 ---- */
__device__ __forceinline__ int compute_lse_pair(int x, int y) {
    int x_max = (x > y) ? x : y;
    int y_min = (x > y) ? y : x;
    int sub   = y_min - x_max;
    int shift = -(sub >> FRAC_BITS);
    if (shift > 31) shift = 31;
    int M     = (SCALE + (sub & FRAC_MASK)) >> shift;
    return x_max + M + LUT64[M & FRAC_MASK];
}

/* ================================================================
 * Block-level Q10.6 logsumexp kernel
 *
 *   Input:  (B, N) int16  (Q10.6 base-2)
 *   Output: (B,)   float32 base-e
 *
 *   One block per row, 256 threads, vectorized 8-byte (4xint16) loads.
 * ================================================================ */
#define THREADS 256

__global__ void block_logsumexp_q10_6_kernel(
    const int16_t* __restrict__ input,
    float*         __restrict__ output,
    int N, int B_total)
{
    int row = blockIdx.x;
    if (row >= B_total) return;

    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid / 32;
    int n_warps = THREADS / 32;

    extern __shared__ int smem[];

    const int16_t* row_in = input + (int64_t)row * N;

    const int NEG_INF = -0x3fffffff;
    bool has_any = false;
    int  acc     = NEG_INF;

    /* ---- Alignment prefix to reach 8-byte (4-int16) boundary ---- */
    uintptr_t addr = reinterpret_cast<uintptr_t>(row_in);
    int prefix = 0;
    if (addr & 7) {
        prefix = (8 - (addr & 7)) / 2;   /* in int16 elements */
        if (prefix > N) prefix = N;
    }

    for (int i = tid; i < prefix; i += THREADS) {
        int q = (int)row_in[i];   /* int16 -> int32 sign-extend */
        if (!has_any) { acc = q; has_any = true; }
        else          { acc = compute_lse_pair(acc, q); }
    }

    /* ---- Vectorized: 4 int16 = 8 bytes per load via short4 ---- */
    int aligned_start = prefix;
    int remaining = N - aligned_start;
    int N4 = remaining / 4;
    const short4* row_in4 = reinterpret_cast<const short4*>(row_in + aligned_start);

    for (int i = tid; i < N4; i += THREADS) {
        short4 v = row_in4[i];

        int q = (int)v.x;
        if (!has_any) { acc = q; has_any = true; }
        else          { acc = compute_lse_pair(acc, q); }

        acc = compute_lse_pair(acc, (int)v.y);
        acc = compute_lse_pair(acc, (int)v.z);
        acc = compute_lse_pair(acc, (int)v.w);
    }

    /* ---- Suffix ---- */
    int suffix_start = aligned_start + N4 * 4;
    for (int i = suffix_start + tid; i < N; i += THREADS) {
        int q = (int)row_in[i];
        if (!has_any) { acc = q; has_any = true; }
        else          { acc = compute_lse_pair(acc, q); }
    }

    if (!has_any) acc = NEG_INF;

    /* ---- Warp-level reduction ---- */
    for (int off = 16; off > 0; off >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, acc, off);
        acc = compute_lse_pair(acc, other);
    }

    /* ---- Inter-warp reduction via shared memory ---- */
    if (lane == 0) smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int v = (lane < n_warps) ? smem[lane] : NEG_INF;
        for (int off = 16; off > 0; off >>= 1) {
            int other = __shfl_xor_sync(0xffffffff, v, off);
            v = compute_lse_pair(v, other);
        }
        if (lane == 0) {
            /* Q10.6 base-2 -> float32 base-e */
            output[row] = (float)v * (LN_2 / (float)SCALE);
        }
    }
}

/* ================================================================
 * Host entry point
 * ================================================================ */
torch::Tensor batched_logsumexp_q10_6(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Expected 2-D tensor (B, N)");
    TORCH_CHECK(input.is_cuda(),  "Input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt16,
                "Input must be int16 (Q10.6 base-2 fixed-point)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int B = input.size(0);
    int N = input.size(1);
    auto output = torch::empty({B}, input.options().dtype(torch::kFloat32));
    if (B == 0) return output;

    const c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    int n_warps = THREADS / 32;
    int smem = n_warps * sizeof(int);

    block_logsumexp_q10_6_kernel<<<B, THREADS, smem, stream>>>(
        input.data_ptr<int16_t>(), output.data_ptr<float>(), N, B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_logsumexp_q10_6", &batched_logsumexp_q10_6,
          "Batched Q10.6 IntLSE: (B,N) int16 base-2 -> (B,) float32 base-e");
}

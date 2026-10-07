/*
 * Fixed-point Q16.16 batched logsumexp CUDA kernel.
 *
 * Target: torch.logsumexp(logits, dim=-1) where logits is (B, N) float32.
 *
 * Design:
 *   - One thread-block per row  
 *   - 256 threads, float4 vectorized loads for high bandwidth
 *   - Single-pass Q16.16 fixed-point LSE reduction (no separate max pass)
 *   - Shared memory for inter-warp reduction
 *   - Converts back to float32 base-e at the end
 *
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

/* ---- Fixed-point parameters ---- */
#define FRAC_BITS 16
#define SCALE     (1 << FRAC_BITS)
#define FRAC_MASK (SCALE - 1)

#define LOG2_E 1.44269504089f
#define LN_2   0.69314718056f
#define CONV   (LOG2_E * (float)SCALE)   /* float->Q16.16 base-2 */

/* ---- 64-entry correction LUT (Q16.16), indexed by the six leading fractional
 *      bits of the Mitchell term M; each entry approximates the Mitchell error
 *      log2(1 + 2^z) - M for that bin and is added to the pair result. ---- */
__device__ __constant__ int LUT64[64] = {
       0,  571,  907, 1282, 1634, 1745, 2054, 2592,
    2900, 2859, 2895, 3015, 3222, 3522, 3922, 4426,
    4617, 4441, 4294, 4178, 4093, 4039, 4018, 4030,
    4076, 4157, 4272, 4424, 4613, 4838, 5102, 5404,
    5399, 5068, 4746, 4435, 4134, 3843, 3562, 3291,
    3031, 2781, 2541, 2312, 2094, 1886, 1689, 1502,
    1327, 1162, 1008,  865,  732,  611,  500,  401,
     312,  235,  168,  113,   69,   35,   13,    2
};

/* ---- Core pair-wise LSE in Q16.16 base-2 ---- */
template <bool USE_LUT>
__device__ __forceinline__ int compute_lse_pair(int x, int y) {
    int x_max = (x > y) ? x : y;
    int y_min = (x > y) ? y : x;
    int sub   = y_min - x_max;
    int M     = (SCALE + (sub & FRAC_MASK)) >> -(sub >> FRAC_BITS);
    if constexpr (USE_LUT)
        return x_max + M + LUT64[(M >> 10) & 0x3F];
    else
        return x_max + M;
}

/* ================================================================
 * Block-level logsumexp kernel
 *
 *   Input:  (B, N) float32 base-e
 *   Output: (B,)   float32 base-e logsumexp
 *
 *   One block per row, 256 threads, float4 vectorized loads
 * ================================================================ */
template <bool USE_LUT>
__global__ void block_logsumexp_fp_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int N, int B_total)
{
    int row = blockIdx.x;
    if (row >= B_total) return;

    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid / 32;
    int n_warps = blockDim.x / 32;

    extern __shared__ int smem[];   /* n_warps ints for warp results */

    const float* row_in = input + (int64_t)row * N;

    const int NEG_INF = -0x3fffffff;
    bool has_any = false;
    int  acc     = NEG_INF;

    /* ---- Check alignment for float4 loads ---- */
    uintptr_t addr = reinterpret_cast<uintptr_t>(row_in);
    int prefix = 0;  /* number of scalar elements before 16-byte alignment */
    if (addr & 15) {
        prefix = (16 - (addr & 15)) / 4;
        if (prefix > N) prefix = N;
    }

    /* Handle prefix (scalar loads to reach alignment) */
    for (int i = tid; i < prefix; i += blockDim.x) {
        int q = __float2int_rn(row_in[i] * CONV);
        if (!has_any) { acc = q; has_any = true; }
        else          { acc = compute_lse_pair<USE_LUT>(acc, q); }
    }

    /* ---- Vectorized load with float4 (aligned) ---- */
    int aligned_start = prefix;
    int remaining = N - aligned_start;
    int N4 = remaining / 4;
    const float4* row_in4 = reinterpret_cast<const float4*>(row_in + aligned_start);

    for (int i = tid; i < N4; i += blockDim.x) {
        float4 v = row_in4[i];
        int q;

        q = __float2int_rn(v.x * CONV);
        if (!has_any) { acc = q; has_any = true; }
        else          { acc = compute_lse_pair<USE_LUT>(acc, q); }

        q = __float2int_rn(v.y * CONV);
        acc = compute_lse_pair<USE_LUT>(acc, q);

        q = __float2int_rn(v.z * CONV);
        acc = compute_lse_pair<USE_LUT>(acc, q);

        q = __float2int_rn(v.w * CONV);
        acc = compute_lse_pair<USE_LUT>(acc, q);
    }

    /* Handle suffix (remaining elements after float4 region) */
    int suffix_start = aligned_start + N4 * 4;
    for (int i = suffix_start + tid; i < N; i += blockDim.x) {
        int q = __float2int_rn(row_in[i] * CONV);
        if (!has_any) { acc = q; has_any = true; }
        else          { acc = compute_lse_pair<USE_LUT>(acc, q); }
    }

    if (!has_any) acc = NEG_INF;

    /* ---- Warp-level reduction ---- */
    for (int off = 16; off > 0; off >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, acc, off);
        acc = compute_lse_pair<USE_LUT>(acc, other);
    }

    /* ---- Inter-warp reduction via shared memory ---- */
    if (lane == 0) smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int v = (lane < n_warps) ? smem[lane] : NEG_INF;
        for (int off = 16; off > 0; off >>= 1) {
            int other = __shfl_xor_sync(0xffffffff, v, off);
            v = compute_lse_pair<USE_LUT>(v, other);
        }
        if (lane == 0) {
            /* Convert Q16.16 base-2 -> float32 base-e */
            output[row] = (float)v * (LN_2 / (float)SCALE);
        }
    }
}

/* ================================================================
 * Host entry points
 * ================================================================ */
static void launch_kernel(torch::Tensor input, torch::Tensor output, bool use_lut) {
    int B = input.size(0);
    int N = input.size(1);
    if (B == 0) return;
    const c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();

    const int THREADS = 256;
    int n_warps = THREADS / 32;
    int smem = n_warps * sizeof(int);

    if (use_lut) {
        block_logsumexp_fp_kernel<true><<<B, THREADS, smem, stream>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), N, B);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        block_logsumexp_fp_kernel<false><<<B, THREADS, smem, stream>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), N, B);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

torch::Tensor batched_logsumexp(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Expected 2-D tensor (B, N)");
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32);
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty({input.size(0)}, input.options());
    launch_kernel(input, output, /*use_lut=*/true);
    return output;
}

torch::Tensor batched_logsumexp_nolut(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Expected 2-D tensor (B, N)");
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32);
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty({input.size(0)}, input.options());
    launch_kernel(input, output, /*use_lut=*/false);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_logsumexp", &batched_logsumexp,
          "Batched fixed-point logsumexp: (B,N) float32 base-e -> (B,) float32 base-e");
    m.def("batched_logsumexp_nolut", &batched_logsumexp_nolut,
          "Batched fixed-point logsumexp without LUT correction (ablation)");
}

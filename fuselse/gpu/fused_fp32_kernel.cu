/*
 * Fused single-pass float32 logsumexp CUDA kernel.
 *
 * Structurally IDENTICAL to the Q16.16 kernel (lse_kernel.cu):
 *   - same 256 threads per block, one block per row
 *   - same float4 vectorized loads
 *   - same warp shuffle + shared memory reduction
 *   - same single-pass design (one read of global memory)
 *
 * The ONLY difference: the pair function uses float transcendentals
 *   float:   max(x,y) + log(1 + exp(min(x,y) - max(x,y)))
 *   Q16.16:  integer shift + add + 64-entry LUT
 *
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#define THREADS 256

/* ---- Float pair-wise LSE (exp/log transcendentals via SFU) ---- */
__device__ __forceinline__ float compute_lse_pair(float x, float y) {
    float x_max = (x > y) ? x : y;
    float x_min = (x > y) ? y : x;
    return x_max + __logf(1.0f + __expf(x_min - x_max));
}


__device__ __forceinline__ float compute_lse_pair_robust(float x, float y) {
    if (isnan(x) || isnan(y)) return CUDART_NAN_F;
    if (x == CUDART_INF_F || y == CUDART_INF_F) return CUDART_INF_F;
    if (x == -CUDART_INF_F) return y;
    if (y == -CUDART_INF_F) return x;
    return compute_lse_pair(x, y);
}

template <bool ROBUST>
__device__ __forceinline__ float compute_lse_pair_mode(float x, float y) {
    if constexpr (ROBUST)
        return compute_lse_pair_robust(x, y);
    else
        return compute_lse_pair(x, y);
}

template <bool ROBUST>
__global__ void block_logsumexp_float_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int N, int B_total)
{
    int row = blockIdx.x;
    if (row >= B_total) return;

    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid / 32;
    int n_warps = THREADS / 32;

    extern __shared__ float smem[];

    const float* row_in = input + (int64_t)row * N;

    const float NEG_INF = ROBUST ? -CUDART_INF_F : -1.0e30f;
    bool has_any = false;
    float acc    = NEG_INF;

    /* ---- Alignment prefix for float4 loads ---- */
    uintptr_t addr = reinterpret_cast<uintptr_t>(row_in);
    int prefix = 0;
    if (addr & 15) {
        prefix = (16 - (addr & 15)) / 4;
        if (prefix > N) prefix = N;
    }

    for (int i = tid; i < prefix; i += THREADS) {
        float v = row_in[i];
        if (!has_any) { acc = v; has_any = true; }
        else          { acc = compute_lse_pair_mode<ROBUST>(acc, v); }
    }

    /* ---- Vectorized float4 loads ---- */
    int aligned_start = prefix;
    int remaining = N - aligned_start;
    int N4 = remaining / 4;
    const float4* row_in4 = reinterpret_cast<const float4*>(row_in + aligned_start);

    for (int i = tid; i < N4; i += THREADS) {
        float4 v = row_in4[i];

        if (!has_any) { acc = v.x; has_any = true; }
        else          { acc = compute_lse_pair_mode<ROBUST>(acc, v.x); }

        acc = compute_lse_pair_mode<ROBUST>(acc, v.y);
        acc = compute_lse_pair_mode<ROBUST>(acc, v.z);
        acc = compute_lse_pair_mode<ROBUST>(acc, v.w);
    }

    /* ---- Suffix ---- */
    int suffix_start = aligned_start + N4 * 4;
    for (int i = suffix_start + tid; i < N; i += THREADS) {
        float v = row_in[i];
        if (!has_any) { acc = v; has_any = true; }
        else          { acc = compute_lse_pair_mode<ROBUST>(acc, v); }
    }

    if (!has_any) acc = NEG_INF;

    /* ---- Warp-level shuffle reduction ---- */
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, acc, off);
        acc = compute_lse_pair_mode<ROBUST>(acc, other);
    }

    /* ---- Inter-warp reduction via shared memory ---- */
    if (lane == 0) smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < n_warps) ? smem[lane] : NEG_INF;
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, off);
            v = compute_lse_pair_mode<ROBUST>(v, other);
        }
        if (lane == 0) {
            output[row] = v;
        }
    }
}

__global__ void logsumexp_float_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ lse_output,
    float* __restrict__ grad_input,
    int N,
    int64_t total)
{
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; index < total; index += stride) {
        int row = (int)(index / N);
        grad_input[index] = grad_output[row]
            * __expf(input[index] - lse_output[row]);
    }
}


/* ---- Host launcher ---- */
torch::Tensor onepass_lse_float(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (B, N)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int B = x.size(0);
    int N = x.size(1);
    auto out = torch::empty({B}, x.options());
    if (B == 0) return out;
    const c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int n_warps = THREADS / 32;
    int smem = n_warps * sizeof(float);
    block_logsumexp_float_kernel<false><<<B, THREADS, smem, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor onepass_lse_float_robust(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (B, N)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int B = x.size(0);
    int N = x.size(1);
    auto out = torch::empty({B}, x.options());
    if (B == 0) return out;
    const c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int n_warps = THREADS / 32;
    int smem = n_warps * sizeof(float);
    block_logsumexp_float_kernel<true><<<B, THREADS, smem, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor onepass_lse_float_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor lse_output)
{
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(lse_output.is_cuda(), "lse_output must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(
        grad_output.dtype() == torch::kFloat32,
        "grad_output must be float32"
    );
    TORCH_CHECK(
        lse_output.dtype() == torch::kFloat32,
        "lse_output must be float32"
    );
    TORCH_CHECK(x.dim() == 2, "x must be 2D (B, N)");
    TORCH_CHECK(grad_output.dim() == 1, "grad_output must be 1D (B)");
    TORCH_CHECK(lse_output.dim() == 1, "lse_output must be 1D (B)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(
        grad_output.is_contiguous(),
        "grad_output must be contiguous"
    );
    TORCH_CHECK(
        lse_output.is_contiguous(),
        "lse_output must be contiguous"
    );
    TORCH_CHECK(
        grad_output.size(0) == x.size(0),
        "grad_output batch dimension must match x"
    );
    TORCH_CHECK(
        lse_output.size(0) == x.size(0),
        "lse_output batch dimension must match x"
    );

    int B = x.size(0);
    int N = x.size(1);
    int64_t total = (int64_t)B * N;
    auto grad_input = torch::empty_like(x);
    if (total == 0) return grad_input;
    const c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int blocks = (int)((total + THREADS - 1) / THREADS);
    logsumexp_float_backward_kernel<<<blocks, THREADS, 0, stream>>>(
        grad_output.data_ptr<float>(),
        x.data_ptr<float>(),
        lse_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        N,
        total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("onepass_lse_float", &onepass_lse_float,
          "Fused float32 single-pass LSE — same structure as Q16.16, but with exp/log pair");
    m.def("onepass_lse_float_robust", &onepass_lse_float_robust,
          "Fused float32 single-pass LSE with IEEE NaN/Inf handling");
    m.def("onepass_lse_float_backward", &onepass_lse_float_backward,
          "Fused float32 LSE backward using saved forward output");
}

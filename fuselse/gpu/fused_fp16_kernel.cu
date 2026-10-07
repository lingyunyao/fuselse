/*
 * Fused single-pass FP16 logsumexp CUDA kernel (FuseLSE16).
 *
 * Same structure as fused_fp32_kernel.cu but reads fp16 input and uses fp16/fp32 hybrid arithmetic
 *
 * Output is fp32 base-e (matching IntLSE's output).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#define THREADS 256

__device__ __forceinline__ float compute_lse_pair_f(float x, float y) {
    float x_max = (x > y) ? x : y;
    float x_min = (x > y) ? y : x;
    return x_max + __logf(1.0f + __expf(x_min - x_max));
}

__global__ void block_logsumexp_fp16_kernel(
    const __half* __restrict__ input,
    float*        __restrict__ output,
    int N, int B_total)
{
    int row = blockIdx.x;
    if (row >= B_total) return;

    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid / 32;
    int n_warps = THREADS / 32;

    extern __shared__ float smem[];

    const __half* row_in = input + (int64_t)row * N;

    const float NEG_INF = -1.0e30f;
    bool has_any = false;
    float acc = NEG_INF;

    /* Alignment prefix to reach 8-byte (4-half) boundary */
    uintptr_t addr = reinterpret_cast<uintptr_t>(row_in);
    int prefix = 0;
    if (addr & 7) {
        prefix = (8 - (addr & 7)) / 2;
        if (prefix > N) prefix = N;
    }

    for (int i = tid; i < prefix; i += THREADS) {
        float v = __half2float(row_in[i]);
        if (!has_any) { acc = v; has_any = true; }
        else          { acc = compute_lse_pair_f(acc, v); }
    }

    /* Vectorized: 4 halves = 8 bytes per load, reinterpret as int2 */
    int aligned_start = prefix;
    int remaining = N - aligned_start;
    int N4 = remaining / 4;
    const int2* row_in4 = reinterpret_cast<const int2*>(row_in + aligned_start);

    for (int i = tid; i < N4; i += THREADS) {
        int2 raw = row_in4[i];
        __half2 lo = *reinterpret_cast<__half2*>(&raw.x);
        __half2 hi = *reinterpret_cast<__half2*>(&raw.y);
        float a = __half2float(__low2half(lo));
        float b = __half2float(__high2half(lo));
        float c = __half2float(__low2half(hi));
        float d = __half2float(__high2half(hi));

        if (!has_any) { acc = a; has_any = true; }
        else          { acc = compute_lse_pair_f(acc, a); }
        acc = compute_lse_pair_f(acc, b);
        acc = compute_lse_pair_f(acc, c);
        acc = compute_lse_pair_f(acc, d);
    }

    /* Suffix */
    int suffix_start = aligned_start + N4 * 4;
    for (int i = suffix_start + tid; i < N; i += THREADS) {
        float v = __half2float(row_in[i]);
        if (!has_any) { acc = v; has_any = true; }
        else          { acc = compute_lse_pair_f(acc, v); }
    }

    if (!has_any) acc = NEG_INF;

    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, acc, off);
        acc = compute_lse_pair_f(acc, other);
    }

    if (lane == 0) smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < n_warps) ? smem[lane] : NEG_INF;
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, off);
            v = compute_lse_pair_f(v, other);
        }
        if (lane == 0) {
            output[row] = v;
        }
    }
}

torch::Tensor batched_logsumexp_fp16(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Expected 2-D tensor (B, N)");
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat16,
                "Input must be CUDA fp16");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    int B = input.size(0);
    int N = input.size(1);
    auto output = torch::empty({B}, input.options().dtype(torch::kFloat32));
    if (B == 0) return output;

    const c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    int n_warps = THREADS / 32;
    int smem = n_warps * sizeof(float);

    block_logsumexp_fp16_kernel<<<B, THREADS, smem, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        output.data_ptr<float>(), N, B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_logsumexp_fp16", &batched_logsumexp_fp16,
          "Batched fused fp16 logsumexp: (B,N) fp16 -> (B,) fp32");
}

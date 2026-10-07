/*
 * Fused float32 sparse forward kernel for pyjuice-style probabilistic circuits.
 *
 * Structurally IDENTICAL to sparse_q16_kernel.cu:
 *   - same thread layout (EDGE_THREADS x BATCH_THREADS)
 *   - same shared memory layout for cids + log-weights
 *   - same cooperative load pattern
 *   - same warp-shuffle edge reduction
 *
 * The ONLY difference: the pair function uses float transcendentals
 *   float:   max(x,y) + log(1 + exp(min(x,y) - max(x,y)))
 *   Q16.16:  integer shift + add + 64-entry LUT
 *
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

/* ================================================================
 * Templated kernel — EDGE_THREADS is a compile-time constant
 * so the compiler can unroll the shuffle reduction and optimise
 * the strided edge loop.
 *
 * Grid : (cdiv(batch_size, BATCH_THREADS), num_nblocks * block_size)
 * Block: (256,)
 * ================================================================ */
template <int EDGE_THREADS>
__global__ void sparse_forward_float_kernel(
    float*       __restrict__ node_mars,
    const float* __restrict__ element_mars,
    const float* __restrict__ params,
    const long*  __restrict__ nids,
    const long*  __restrict__ cids,
    const long*  __restrict__ pids,
    int batch_size,
    int num_edges,
    int block_size)
{
    constexpr int BATCH_THREADS = THREADS / EDGE_THREADS;

    int tid         = threadIdx.x;
    int edge_tid    = tid % EDGE_THREADS;
    int local_batch = tid / EDGE_THREADS;

    int b         = blockIdx.x * BATCH_THREADS + local_batch;
    int nblock_id = blockIdx.y / block_size;
    int parent    = blockIdx.y % block_size;

    /* ---- Shared memory layout ----
     *   s_cids  [num_edges]  long
     *   s_log_w [num_edges]  float
     * (No LUT needed for float version)
     */
    extern __shared__ char smem[];
    long*  s_cids  = (long*)smem;
    float* s_log_w = (float*)(s_cids + num_edges);

    /* Cooperative load: cids + log-weights for this parent */
    for (int e = tid; e < num_edges; e += THREADS) {
        int base = nblock_id * num_edges + e;
        s_cids[e] = cids[base];
        float w = params[pids[base] + parent];
        s_log_w[e] = __logf(fmaxf(w, 1e-45f));
    }
    __syncthreads();

    /* ---- Edge reduction (strided by EDGE_THREADS) ---- */
    const float NEG_INF = -1.0e30f;
    float acc = NEG_INF;

    if (b < batch_size) {
        for (int e = edge_tid; e < num_edges; e += EDGE_THREADS) {
            float child_val = element_mars[s_cids[e] * batch_size + b];
            float logit = s_log_w[e] + child_val;
            acc = compute_lse_pair(acc, logit);
        }
    }

    /* ---- Warp-shuffle reduction within each EDGE_THREADS group ----
     * All threads participate (invalid threads hold NEG_INF = identity). */
    #pragma unroll
    for (int off = EDGE_THREADS / 2; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, acc, off);
        acc = compute_lse_pair(acc, other);
    }

    /* Only the first thread in each edge group writes the result */
    if (b < batch_size && edge_tid == 0) {
        long node_start = nids[nblock_id];
        node_mars[(node_start + parent) * batch_size + b] = acc;
    }
}

/* ================================================================
 * Host entry point — picks EDGE_THREADS to target ~8 edges/thread
 * ================================================================ */
void sparse_forward_float(
    torch::Tensor node_mars,
    torch::Tensor element_mars,
    torch::Tensor params,
    torch::Tensor nids,
    torch::Tensor cids,
    torch::Tensor pids,
    int block_size)
{
    TORCH_CHECK(node_mars.is_cuda(),     "node_mars must be CUDA");
    TORCH_CHECK(element_mars.is_cuda(),  "element_mars must be CUDA");
    TORCH_CHECK(params.is_cuda(),        "params must be CUDA");

    int batch_size  = node_mars.size(1);
    int num_nblocks = nids.size(0);
    int num_edges   = cids.size(1);

    int edge_threads;
    if      (num_edges <= 8)  edge_threads = 1;
    else if (num_edges <= 16) edge_threads = 2;
    else if (num_edges <= 32) edge_threads = 4;
    else                      edge_threads = 8;

    int batch_threads = THREADS / edge_threads;

    dim3 grid(
        (batch_size + batch_threads - 1) / batch_threads,
        num_nblocks * block_size
    );

    /* Shared: cids (num_edges*8) + log_w (num_edges*4) — no LUT */
    int smem_bytes = num_edges * (sizeof(long) + sizeof(float));

    if (batch_size == 0 || num_nblocks == 0 || block_size == 0) return;
    const c10::cuda::CUDAGuard device_guard(node_mars.device());
    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(node_mars.get_device()).stream();

    #define LAUNCH(ET) \
        sparse_forward_float_kernel<ET><<<grid, THREADS, smem_bytes, stream>>>( \
            node_mars.data_ptr<float>(),     \
            element_mars.data_ptr<float>(),  \
            params.data_ptr<float>(),        \
            nids.data_ptr<long>(),           \
            cids.data_ptr<long>(),           \
            pids.data_ptr<long>(),           \
            batch_size, num_edges, block_size)

    switch (edge_threads) {
        case 1:  LAUNCH(1);  break;
        case 2:  LAUNCH(2);  break;
        case 4:  LAUNCH(4);  break;
        case 8:  LAUNCH(8);  break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    #undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_forward_float", &sparse_forward_float,
          "Fused float32 sparse forward for pyjuice-style PCs (SFU pair function)");
}

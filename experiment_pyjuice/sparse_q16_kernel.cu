/*
 * Fused Q16.16 sparse forward kernel for pyjuice-style probabilistic circuits.
 *
 *
 * Each block handles one (nblock, parent) pair.
 *
 * Thread layout — 2D within a flat block of 256 threads:
 *   edge_tid   = tid % EDGE_THREADS   — which slice of edges this thread reduces
 *   local_batch = tid / EDGE_THREADS  — which batch element in the block
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

/* ---- Fixed-point parameters (matches lse_kernel.cu) ---- */
#define FRAC_BITS 16
#define SCALE     (1 << FRAC_BITS)
#define FRAC_MASK (SCALE - 1)
#define LOG2_E    1.44269504089f
#define LN_2      0.69314718056f
#define CONV      (LOG2_E * (float)SCALE)
#define NEG_INF_Q16 (-0x3fffffff)
#define THREADS   256

/* ---- LUT source (copied to shared memory at kernel start) ---- */
__device__ __constant__ int LUT64_CONST[64] = {
       0,  571,  907, 1282, 1634, 1745, 2054, 2592,
    2900, 2859, 2895, 3015, 3222, 3522, 3922, 4426,
    4617, 4441, 4294, 4178, 4093, 4039, 4018, 4030,
    4076, 4157, 4272, 4424, 4613, 4838, 5102, 5404,
    5399, 5068, 4746, 4435, 4134, 3843, 3562, 3291,
    3031, 2781, 2541, 2312, 2094, 1886, 1689, 1502,
    1327, 1162, 1008,  865,  732,  611,  500,  401,
     312,  235,  168,  113,   69,   35,   13,    2
};

/* Core pair-wise LSE in Q16.16 base-2. */
__device__ __forceinline__ int compute_lse_pair(int x, int y,
                                                const int* __restrict__ lut) {
    int x_max = (x > y) ? x : y;
    int y_min = (x > y) ? y : x;
    int sub   = y_min - x_max;
    int shift = min(-(sub >> FRAC_BITS), 31);
    int M     = (SCALE + (sub & FRAC_MASK)) >> shift;
    return x_max + M + lut[(M >> 10) & 0x3F];
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
__global__ void sparse_forward_q16_kernel(
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
     *   s_lut   [64]         int
     *   s_cids  [num_edges]  long
     *   s_log_w [num_edges]  float
     */
    extern __shared__ char smem[];
    int*   s_lut   = (int*)smem;
    long*  s_cids  = (long*)(s_lut + 64);
    float* s_log_w = (float*)(s_cids + num_edges);

    /* Load LUT into shared memory (first 64 threads) */
    if (tid < 64)
        s_lut[tid] = LUT64_CONST[tid];

    /* Cooperative load: cids + log-weights for this parent */
    for (int e = tid; e < num_edges; e += THREADS) {
        int base = nblock_id * num_edges + e;
        s_cids[e] = cids[base];
        float w = params[pids[base] + parent];
        s_log_w[e] = __logf(fmaxf(w, 1e-45f));
    }
    __syncthreads();

    /* ---- Edge reduction (strided by EDGE_THREADS) ---- */
    int acc = NEG_INF_Q16;

    if (b < batch_size) {
        for (int e = edge_tid; e < num_edges; e += EDGE_THREADS) {
            float child_val = element_mars[s_cids[e] * batch_size + b];
            float logit = s_log_w[e] + child_val;
            int   q = __float2int_rn(fmaxf(logit, -20000.0f) * CONV);
            acc = compute_lse_pair(acc, q, s_lut);
        }
    }

    /* ---- Warp-shuffle reduction within each EDGE_THREADS group ----
     * All threads participate (invalid threads hold NEG_INF = identity). */
    #pragma unroll
    for (int off = EDGE_THREADS / 2; off > 0; off >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, acc, off);
        acc = compute_lse_pair(acc, other, s_lut);
    }

    /* Only the first thread in each edge group writes the result */
    if (b < batch_size && edge_tid == 0) {
        long node_start = nids[nblock_id];
        float result = (acc > NEG_INF_Q16)
            ? (float)acc * (LN_2 / (float)SCALE)
            : -INFINITY;
        node_mars[(node_start + parent) * batch_size + b] = result;
    }
}

/* ================================================================
 * Host entry point — picks EDGE_THREADS to target ~8 edges/thread
 * ================================================================ */
void sparse_forward_q16(
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

    /* Select EDGE_THREADS: balance edge parallelism vs batch parallelism.
     * Tree reduction gives O(log ET) accuracy regardless of ET, so pick
     * ET to minimise total block count while keeping the serial chain short.
     * With large num_nblocks * block_size on gridDim.y, keep ET small
     * to avoid exploding gridDim.x. */
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

    /* Shared: LUT (256 B) + cids (num_edges*8) + log_w (num_edges*4) */
    int smem_bytes = 64 * sizeof(int)
                   + num_edges * (sizeof(long) + sizeof(float));

    if (batch_size == 0 || num_nblocks == 0 || block_size == 0) return;
    const c10::cuda::CUDAGuard device_guard(node_mars.device());
    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(node_mars.get_device()).stream();

    #define LAUNCH(ET) \
        sparse_forward_q16_kernel<ET><<<grid, THREADS, smem_bytes, stream>>>( \
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
    m.def("sparse_forward_q16", &sparse_forward_q16,
          "Fused Q16.16 sparse forward for pyjuice-style PCs");
}

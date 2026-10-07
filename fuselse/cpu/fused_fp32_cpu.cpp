/*
 * FuseLSE CPU kernel (fp32) — single-pass pair-reduce with std::log1p / std::exp.
 *
 * Algorithm (same as fuselse/fused_fp32_kernel.cu but for CPU):
 *   acc = max(acc, x_i) + log1p(exp(-|acc - x_i|))
 *
 * Sequential per-row reduction. Used for the CPU panel of the
 * microbenchmark figure as the float counterpart to IntLSE.
 *
 *
 * The companion intlse/cpu/lse_kernel_simd.c does use hand AVX2 because 
 * (a) its inner loop is integer-only, 
 * (b) integer ops are 1-cycle so the dependency chain becomes the bottleneck,
 * (c) no vectorised transcendental is needed.
 *
 * Build: torch.utils.cpp_extension.load(..., extra_cflags=["-O3", "-march=native", "-fopenmp"])
 */
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>

static inline float lse_pair_float(float x, float y) {
    float xm = (x > y) ? x : y;
    float ym = (x > y) ? y : x;
    return xm + std::log1p(std::exp(ym - xm));
}

torch::Tensor cpu_lse_float(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2 && input.dtype() == torch::kFloat32);
    TORCH_CHECK(!input.is_cuda());
    int64_t B = input.size(0), N = input.size(1);
    auto output = torch::empty({B}, input.options());
    const float* in_ptr  = input.data_ptr<float>();
    float*       out_ptr = output.data_ptr<float>();
    at::parallel_for(0, B, /*grain_size=*/1,
        [&](int64_t b_start, int64_t b_end) {
            for (int64_t b = b_start; b < b_end; b++) {
                const float* row = in_ptr + b * N;
                float acc = row[0];
                for (int64_t i = 1; i < N; i++)
                    acc = lse_pair_float(acc, row[i]);
                out_ptr[b] = acc;
            }
        });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_lse_float", &cpu_lse_float,
          "Fused float32 LSE on CPU (single-pass pair reduction)");
}

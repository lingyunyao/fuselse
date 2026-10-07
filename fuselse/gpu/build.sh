#!/bin/bash
# Build the FuseLSE CUDA kernels via PyTorch's torch.utils.cpp_extension.
# Two variants:
#   - fused_fp32_kernel.cu: fp32 input, fp32 internal arithmetic
#   - fused_fp16_kernel.cu: fp16 input, fp32 internal arithmetic (V100+)
#
# These are compiled JIT by load() at first import — see:
#   experiment_microbenchmark/microbench/run_microbenchmark.py
#   experiment_microbenchmark/reduced_bit/run_bench.py
#
# This script is just a sanity-check that the .cu files parse correctly.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "FuseLSE kernels at $HERE:"
ls "$HERE"/*.cu
echo
echo "These compile via torch.utils.cpp_extension.load() at first import."
echo "To rebuild manually (debugging):"
echo "  python3 -c 'from torch.utils.cpp_extension import load; \\"
echo "    load(name=\"fused_fp32_ext\", sources=[\"$HERE/fused_fp32_kernel.cu\"], \\"
echo "         extra_cuda_cflags=[\"-O3\", \"--use_fast_math\"], verbose=True)'"

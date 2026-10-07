#!/bin/bash
# Build the IntLSE AVX2 SIMD kernel as a shared library.
# Targets x86_64 with AVX2 + FMA (Haswell or newer).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
gcc -O3 -fPIC -shared \
    -mavx2 -mfma -mtune=haswell \
    "$HERE/lse_kernel_simd.c" \
    -o "$HERE/lse_kernel_simd.so" \
    -lm
echo "Built: $HERE/lse_kernel_simd.so"

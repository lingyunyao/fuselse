#!/usr/bin/env python3
"""
ncu-friendly kernel runner.

"""

import os, sys, shutil, ctypes

local_bin = os.path.expanduser("~/.local/bin")
if local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = local_bin + ":" + os.environ.get("PATH", "")
if not os.environ.get("CUDA_HOME"):
    nvcc = shutil.which("nvcc")
    if nvcc:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc))

import torch
import torch.cuda.nvtx as nvtx
from torch.utils.cpp_extension import load

HERE  = os.path.dirname(os.path.abspath(__file__))
MICRO = os.path.dirname(HERE)   # cross_gpu_breakdown lives inside experiment_microbenchmark/
DEVICE = torch.device("cuda:0")

major, minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
arch_tag = f"sm{major}{minor}"

print(f"GPU: {torch.cuda.get_device_name()}  (cc {major}.{minor})", flush=True)

# --- Compile kernels with arch-tagged names so each cluster builds its own binary ---
print("Compiling Q16.16 kernel...", flush=True)
q16_ext = load(name=f"q16_ncu_{arch_tag}",
               sources=[os.path.join(MICRO, "..", "intlse", "gpu", "lse_kernel.cu")],
               extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
print("Compiling fused-float kernel...", flush=True)
flt_ext = load(name=f"flt_ncu_{arch_tag}",
               sources=[os.path.join(MICRO, "..", "fuselse", "gpu", "fused_fp32_kernel.cu")],
               extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
print("Kernels compiled.\n", flush=True)


B       = 512
N_grid  = [256, 1024, 4096, 16384, 65536, 128256]
WARMUP  = 50
N_LAUNCHES_PER_PROFILE = 10   # ncu averages within each launch; more = lower noise

KERNELS = {
    "torch":       lambda x: torch.logsumexp(x, dim=-1),
    "fused_float": lambda x: flt_ext.onepass_lse_float(x),
    "q16":         lambda x: q16_ext.batched_logsumexp(x),
}

# --- WARMUP outside the profiler region ---
print(f"Warmup ({WARMUP} iters per (kernel, N))...", flush=True)
for N in N_grid:
    x = torch.rand(B, N, device=DEVICE, dtype=torch.float32) * (-100.0)
    for kname, fn in KERNELS.items():
        for _ in range(WARMUP):
            fn(x)
torch.cuda.synchronize()
print("Warmup done.\n", flush=True)

# --- MEASURED region (ncu profiles only what's between Start/Stop) ---
torch.cuda.profiler.start()

for N in N_grid:
    x = torch.rand(B, N, device=DEVICE, dtype=torch.float32) * (-100.0)
    for kname, fn in KERNELS.items():
        nvtx.range_push(f"{kname}_N={N}")
        for _ in range(N_LAUNCHES_PER_PROFILE):
            fn(x)
        torch.cuda.synchronize()
        nvtx.range_pop()

torch.cuda.profiler.stop()
print("Measured region complete.")

#!/usr/bin/env python3
"""
LinearLSE kernel microbenchmark: speedup and accuracy.

"""
import os
import sys
import json
import time
import shutil

local_bin = os.path.expanduser("~/.local/bin")
if local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = local_bin + ":" + os.environ.get("PATH", "")
if not os.environ.get("CUDA_HOME"):
    nvcc = shutil.which("nvcc")
    if nvcc:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc))

import torch
import numpy as np
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
# CPU IntLSE kernel lives in ../../intlse/cpu/ (built via ../../intlse/cpu/build.sh)
sys.path.insert(0, os.path.join(HERE, "..", "..", "intlse", "cpu"))

DEVICE = torch.device("cuda:0")
major, minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}")
print(f"PyTorch: {torch.__version__}")

# Read CPU model
cpu_name = "unknown"
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if "model name" in line:
                cpu_name = line.split(":")[1].strip()
                break
except Exception:
    pass
print(f"CPU: {cpu_name}")
omp_threads = os.environ.get("OMP_NUM_THREADS", "auto")
print(f"OMP_NUM_THREADS: {omp_threads}\n")


# ============================================================
# Compile all kernels
# ============================================================
print("Compiling Q16.16 CUDA kernel...", flush=True)
q16_ext = load(
    name="lse_kernel_ext",
    sources=[os.path.join(HERE, "..", "..", "intlse", "gpu", "lse_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

print("Compiling fused float CUDA kernel (GPU FuseLSE)...", flush=True)
float_ext = load(
    name="fused_float_ext",
    sources=[os.path.join(HERE, "..", "..", "fuselse", "gpu", "fused_fp32_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

print("Compiling fused float CPU kernel (CPU FuseLSE)...", flush=True)
float_cpu_ext = load(
    name="fused_float_cpu_ext",
    sources=[os.path.join(HERE, "..", "..", "fuselse", "cpu", "fused_fp32_cpu.cpp")],
    extra_cflags=["-O3", "-march=native", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)

# CPU IntLSE: AVX2 SIMD kernel from ../intlse/
print("Loading IntLSE-SIMD CPU kernel from intlse/...", flush=True)
from lse_ctypes_wrapper import lse_intlse_2d_simd as intlse_cpu_2d
print("Done.\n", flush=True)


# ============================================================
# Config
# ============================================================
B_FIXED = 512
n_steps = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
           2048, 4096, 8192, 16384, 32000, 50257, 65536, 128256]

GPU_ITERS = 200
GPU_WARMUP = 20
CPU_WARMUP = 5
NUM_SEEDS = 20


def make_data(B, N, seed=42, device="cuda", lo=-100.0, hi=0.0):
    """Generate uniform (B, N) float32 tensor in U(lo, hi)."""
    torch.manual_seed(seed)
    if device == "cuda":
        u = torch.rand(B, N, device=DEVICE, dtype=torch.float32)
    else:
        u = torch.rand(B, N, dtype=torch.float32)
    return u * (hi - lo) + lo



RANGES = [(-1.0, 0.0), (-100.0, 0.0), (-500.0, 0.0)]


def range_label(lo, hi):
    """Compact filename-safe label, e.g. 'U(-100,0)'."""
    return f"U({lo:g},{hi:g})"


def cpu_iters(N):
    """Fewer iterations for large N on CPU to keep runtime manageable."""
    if N > 50000:
        return 10
    elif N > 16384:
        return 20
    elif N > 4096:
        return 30
    else:
        return 50


ACC_KERNELS_GPU = {
    "fused_float": lambda x: float_ext.onepass_lse_float(x),
    "q16":         lambda x: q16_ext.batched_logsumexp(x),
    "q16_nolut":   lambda x: q16_ext.batched_logsumexp_nolut(x),
}

def cpu_intlse_call(t):
    """Bridge: torch.Tensor -> numpy -> intlse_cpu_2d -> torch.Tensor."""
    arr = t.detach().contiguous().numpy()
    out = intlse_cpu_2d(arr)
    return torch.from_numpy(out)

def fp64_ref(x_cpu):
    """Ground-truth LSE in fp64. Compute on CPU regardless of input device."""
    x = x_cpu.detach().to(dtype=torch.float64)
    m = x.max(dim=-1, keepdim=True).values
    return (m + torch.log(torch.sum(torch.exp(x - m), dim=-1, keepdim=True))
            ).squeeze(-1).to(dtype=torch.float32)


def time_gpu(fn, data, warmup=GPU_WARMUP, iters=GPU_ITERS):
    for _ in range(warmup):
        fn(data)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(data)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def time_cpu(fn, data, warmup=CPU_WARMUP, iters=50):
    for _ in range(warmup):
        fn(data)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(data)
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000  # ms


def run_one_range(lo, hi):
    """Run accuracy + GPU timing + CPU timing for one input range.
    Returns the per-range result dict for embedding into the master JSON."""
    label = range_label(lo, hi)
    print("\n" + "=" * 90)
    print(f"RANGE {label}")
    print("=" * 90)

    
    method_names = ["torch", "fuselse_gpu", "fuselse_cpu",
                    "intlse_gpu", "intlse_cpu", "intlse_nolut"]
    accuracy = {k: {"median_rel_err_pct": [], "mean_abs_err": []}
                for k in method_names}
    for N in n_steps:
        per_k = {k: {"rel": [], "abs": []} for k in method_names}
        for seed in range(NUM_SEEDS):
            data = make_data(B_FIXED, N, seed=seed, lo=lo, hi=hi)
            data_cpu = data.cpu().contiguous()
            ref = fp64_ref(data_cpu).to(DEVICE)

            outputs = {
                "torch":         torch.logsumexp(data, dim=-1),
                "fuselse_gpu":   float_ext.onepass_lse_float(data),
                "fuselse_cpu":   float_cpu_ext.cpu_lse_float(data_cpu).to(DEVICE),
                "intlse_gpu":    q16_ext.batched_logsumexp(data),
                "intlse_cpu":    cpu_intlse_call(data_cpu).to(DEVICE),
                "intlse_nolut":  q16_ext.batched_logsumexp_nolut(data),
            }
            for kname, ours in outputs.items():
                ae = (ref - ours).abs()
                re = ae / ref.abs().clamp(min=1e-8) * 100
                per_k[kname]["abs"].append(ae.mean().item())
                per_k[kname]["rel"].append(re.median().item())
        for kname in method_names:
            accuracy[kname]["median_rel_err_pct"].append(
                float(np.median(per_k[kname]["rel"])))
            accuracy[kname]["mean_abs_err"].append(
                float(np.mean(per_k[kname]["abs"])))

    # ---------- GPU timing ----------
    # Symmetric 3-line layout: torch + FuseLSE (GPU) + IntLSE (GPU)
    gpu_timing = {"N": [], "torch_ms": [], "fuselse_ms": [], "intlse_ms": []}
    print(f"\n[{label}] GPU timing")
    print(f"{'N':>8s}  {'torch':>10s}  {'FuseLSE':>10s}  {'IntLSE':>10s}")
    print("-" * 50)
    for N in n_steps:
        data = make_data(B_FIXED, N, seed=42, lo=lo, hi=hi)
        t_torch = time_gpu(lambda x: torch.logsumexp(x, dim=-1), data)
        t_f     = time_gpu(lambda x: float_ext.onepass_lse_float(x), data)
        t_q     = time_gpu(lambda x: q16_ext.batched_logsumexp(x), data)
        gpu_timing["N"].append(N)
        gpu_timing["torch_ms"].append(t_torch)
        gpu_timing["fuselse_ms"].append(t_f)
        gpu_timing["intlse_ms"].append(t_q)
        print(f"{N:>8,d}  {t_torch:>8.4f}ms  {t_f:>8.4f}ms  {t_q:>8.4f}ms")

    # ---------- CPU timing ----------
    # Symmetric 3-line layout: torch + FuseLSE (CPU) + IntLSE (CPU SIMD)
    cpu_timing = {"N": [], "torch_ms": [], "fuselse_ms": [], "intlse_ms": []}
    print(f"\n[{label}] CPU timing")
    print(f"{'N':>8s}  {'torch':>10s}  {'FuseLSE':>10s}  {'IntLSE':>10s}")
    print("-" * 50)
    for N in n_steps:
        data_cpu = make_data(B_FIXED, N, seed=42, device="cpu", lo=lo, hi=hi)
        iters = cpu_iters(N)
        t_torch = time_cpu(lambda x: torch.logsumexp(x, dim=-1), data_cpu, iters=iters)
        t_f     = time_cpu(lambda x: float_cpu_ext.cpu_lse_float(x), data_cpu, iters=iters)
        t_int   = time_cpu(lambda x: cpu_intlse_call(x), data_cpu, iters=iters)
        cpu_timing["N"].append(N)
        cpu_timing["torch_ms"].append(t_torch)
        cpu_timing["fuselse_ms"].append(t_f)
        cpu_timing["intlse_ms"].append(t_int)
        print(f"{N:>8,d}  {t_torch:>8.4f}ms  {t_f:>8.4f}ms  {t_int:>8.4f}ms")

    return {
        "range_label": label,
        "range_lo": lo, "range_hi": hi,
        "accuracy": accuracy,
        "gpu_timing": gpu_timing,
        "cpu_timing": cpu_timing,
    }


# ============================================================
# Sweep over input ranges
# ============================================================
all_ranges = [run_one_range(lo, hi) for (lo, hi) in RANGES]


# ============================================================
# Save results
# ============================================================
save_data = {
    "gpu_name": gpu_name,
    "cpu_name": cpu_name,
    "omp_threads": omp_threads,
    "B": B_FIXED,
    "N": list(n_steps),
    "ranges": all_ranges,
}
out_dir = os.path.join(HERE, "outputs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "benchmark_results.json")
with open(out_path, "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to {out_path}")
print("Done.", flush=True)

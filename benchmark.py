#!/usr/bin/env python3
"""Quick speed comparison of torch.logsumexp against FuseLSE (GPU) and IntLSE (CPU).

    python benchmark.py                # GPU and CPU
    python benchmark.py --gpu-only     # skip the CPU comparison
    python benchmark.py --compile      # also time torch.compile(torch.logsumexp) on GPU

Inputs are drawn from U(-100, 0), as in the paper.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def gpu_time_ms(fn, x, warmup=20, iters=100):
    for _ in range(warmup):
        fn(x)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cpu_time_ms(fn, x, warmup=2, iters=5):
    for _ in range(warmup):
        fn(x)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples))


def bench_gpu(sizes, batch, use_compile):
    from torch.utils.cpp_extension import load

    sys.path.insert(0, str(ROOT / "fuselse" / "gpu"))
    from autograd import make_fuselse_logsumexp

    print("Compiling FuseLSE (first run only)...", flush=True)
    ext = load(name="fuselse", sources=[str(ROOT / "fuselse" / "gpu" / "fused_fp32_kernel.cu")],
               extra_cuda_cflags=["-O3", "--use_fast_math"])
    fuselse = make_fuselse_logsumexp(ext, robust=True)
    reference = lambda t: torch.logsumexp(t, dim=-1)
    compiled = torch.compile(reference, dynamic=False) if use_compile else None

    print(f"\nGPU: {torch.cuda.get_device_name()}, B={batch}")
    header = f"{'N':>8} | {'torch (ms)':>10} | "
    header += f"{'compile (ms)':>12} | " if use_compile else ""
    header += f"{'FuseLSE (ms)':>12} | {'speedup':>7} | {'max |diff|':>10}"
    print(header)
    print("-" * len(header))
    for n in sizes:
        x = torch.rand(batch, n, device="cuda") * 100 - 100
        t_ref = gpu_time_ms(reference, x)
        t_fuse = gpu_time_ms(lambda t: fuselse(t, dim=-1), x)
        diff = (fuselse(x, dim=-1) - reference(x)).abs().max().item()
        row = f"{n:>8} | {t_ref:>10.4f} | "
        if use_compile:
            row += f"{gpu_time_ms(compiled, x):>12.4f} | "
        row += f"{t_fuse:>12.4f} | {t_ref / t_fuse:>6.1f}x | {diff:>10.2e}"
        print(row)


def bench_cpu(sizes, batch):
    sys.path.insert(0, str(ROOT / "intlse" / "cpu"))
    if not (ROOT / "intlse" / "cpu" / "lse_kernel_simd.so").exists():
        print("\nCPU: skipped, build IntLSE first with: bash intlse/cpu/build.sh")
        return
    from torch_wrapper import logsumexp_intlse_torch

    torch.set_num_threads(1)
    cpu_name = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    print(f"\nCPU: {cpu_name}, one thread for both methods, B={batch}")
    header = f"{'N':>8} | {'torch (ms)':>10} | {'IntLSE (ms)':>11} | {'speedup':>7} | {'max |diff|':>10}"
    print(header)
    print("-" * len(header))
    for n in sizes:
        x = torch.rand(batch, n) * 100 - 100
        reference = lambda t: torch.logsumexp(t, dim=-1)
        intlse = lambda t: logsumexp_intlse_torch(t, dim=-1)
        t_ref, t_int = cpu_time_ms(reference, x), cpu_time_ms(intlse, x)
        diff = (intlse(x) - reference(x)).abs().max().item()
        print(f"{n:>8} | {t_ref:>10.2f} | {t_int:>11.2f} | {t_ref / t_int:>6.1f}x | {diff:>10.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--compile", action="store_true", help="also time torch.compile on GPU")
    parser.add_argument("--gpu-batch", type=int, default=512)
    parser.add_argument("--cpu-batch", type=int, default=32)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 128256],
                        help="row lengths N (default: small rows and an LLM vocabulary)")
    args = parser.parse_args()
    torch.manual_seed(0)
    if not args.cpu_only:
        if torch.cuda.is_available():
            bench_gpu(args.sizes, args.gpu_batch, args.compile)
        else:
            print("GPU: skipped, no CUDA device found")
    if not args.gpu_only:
        bench_cpu(args.sizes, args.cpu_batch)


if __name__ == "__main__":
    main()

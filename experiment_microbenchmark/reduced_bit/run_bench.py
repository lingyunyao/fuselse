#!/usr/bin/env python3
"""
Reduced-bit LSE microbenchmark.


"""
import os, sys, json, math, time, shutil

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
DEVICE = torch.device("cuda:0")
major, minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

print(f"GPU:      {torch.cuda.get_device_name()}  (cc {major}.{minor})", flush=True)
print(f"PyTorch:  {torch.__version__}", flush=True)


# ---------------- Compile kernels from the central fuselse/gpu/ folder ----------------
FUSELSE_GPU = os.path.join(HERE, "..", "..", "fuselse", "gpu")
print("Compiling FuseLSE-fp32 from fuselse/gpu/...", flush=True)
fp32_ext = load(
    name="fuselse_fp32",
    sources=[os.path.join(FUSELSE_GPU, "fused_fp32_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
print("Compiling FuseLSE-fp16 from fuselse/gpu/...", flush=True)
fp16_ext = load(
    name="fuselse_fp16",
    sources=[os.path.join(FUSELSE_GPU, "fused_fp16_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
INTLSE_GPU = os.path.join(HERE, "..", "..", "intlse", "gpu")
print("Compiling IntLSE-Q10.6 from intlse/gpu/...", flush=True)
q106_ext = load(
    name="intlse_q10_6",
    sources=[os.path.join(INTLSE_GPU, "lse_kernel_q10_6.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
print("Done.\n", flush=True)

# float -> Q10.6 base-2 conversion factor (log2(e) * 2^6).
CONV_Q106 = math.log2(math.e) * 64.0
Q106_MIN  = -32768
Q106_MAX  =  32767


# ---------------- Config ----------------
B = 512
# log grid N=2..128256 (matching microbench)
N_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
          2048, 4096, 8192, 16384, 32000, 50257, 65536, 128256]
LO, HI = -100.0, 0.0
WARMUP = 100
ITERS  = 1000
NUM_SEEDS_ACC = 20


def make_data(B, N, seed=42, lo=LO, hi=HI):
    torch.manual_seed(seed)
    u = torch.rand(B, N, device=DEVICE, dtype=torch.float32)
    return u * (hi - lo) + lo


def fp64_ref(x_fp32):
    """fp64 ground truth on CPU; returns fp32 tensor on GPU for compare."""
    x = x_fp32.detach().cpu().to(torch.float64)
    m = x.max(dim=-1, keepdim=True).values
    out = m + torch.log(torch.sum(torch.exp(x - m), dim=-1, keepdim=True))
    return out.squeeze(-1).to(torch.float32).to(DEVICE)


def time_gpu(fn, data, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup): fn(data)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn(data)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters     # ms



def call_torch(x):           return torch.logsumexp(x, dim=-1)
def call_fuselse_fp32(x):    return fp32_ext.onepass_lse_float(x)
def call_fuselse_fp16(x):    return fp16_ext.batched_logsumexp_fp16(x)
def call_intlse_q106(x):     return q106_ext.batched_logsumexp_q10_6(x)


def to_q10_6(data_fp32):
    """fp32 base-e -> int16 Q10.6 base-2 (round-to-nearest, saturating)."""
    q = (data_fp32 * CONV_Q106).round().clamp(Q106_MIN, Q106_MAX).to(torch.int16)
    return q.contiguous()


# ---------------- Sweep ----------------
print(f"B={B}  N sweep ({len(N_LIST)} points)  range U({LO},{HI})  "
      f"warmup={WARMUP}  iters={ITERS}  acc seeds={NUM_SEEDS_ACC}\n",
      flush=True)
print(f"{'N':>8s}  {'torch (ms)':>11s}  {'FuseLSE (ms)':>13s}  {'FuseLSE16 (ms)':>15s}  "
      f"{'IntLSE-Q10.6 (ms)':>18s}  "
      f"{'fuse/torch':>11s}  {'fuse16/torch':>13s}  {'q106/torch':>11s}  "
      f"{'fuse_abs':>10s}  {'fuse16_abs':>11s}  {'q106_abs':>10s}", flush=True)
print("-" * 170, flush=True)

results = []
for N in N_LIST:
    # ---------- accuracy (averaged over multiple seeds) ----------
    abs_torch, abs_fp32, abs_fp16, abs_q106 = [], [], [], []
    rel_torch, rel_fp32, rel_fp16, rel_q106 = [], [], [], []
    for seed in range(NUM_SEEDS_ACC):
        data      = make_data(B, N, seed=seed)
        data_fp16 = data.to(torch.float16).contiguous()
        data_q106 = to_q10_6(data)
        ref  = fp64_ref(data)
        out_t  = call_torch(data)
        out_f  = call_fuselse_fp32(data)
        out_h  = call_fuselse_fp16(data_fp16)
        out_q  = call_intlse_q106(data_q106)
        for lst_abs, lst_rel, out in [
            (abs_torch, rel_torch, out_t),
            (abs_fp32,  rel_fp32,  out_f),
            (abs_fp16,  rel_fp16,  out_h),
            (abs_q106,  rel_q106,  out_q),
        ]:
            ae = (ref - out).abs()
            re = ae / ref.abs().clamp(min=1e-8) * 100
            lst_abs.append(ae.mean().item())
            lst_rel.append(re.median().item())

    # ---------- timing (inputs pre-converted to native dtype) ----------
    data      = make_data(B, N, seed=42)
    data_fp16 = data.to(torch.float16).contiguous()
    data_q106 = to_q10_6(data)
    t_torch = time_gpu(call_torch,        data)
    t_fp32  = time_gpu(call_fuselse_fp32, data)
    t_fp16  = time_gpu(call_fuselse_fp16, data_fp16)
    t_q106  = time_gpu(call_intlse_q106,  data_q106)

    sp_fp32 = t_torch / t_fp32
    sp_fp16 = t_torch / t_fp16
    sp_q106 = t_torch / t_q106

    e_fp32 = float(np.mean(abs_fp32))
    e_fp16 = float(np.mean(abs_fp16))
    e_q106 = float(np.mean(abs_q106))

    print(f"{N:>8,d}  {t_torch:>9.4f}     {t_fp32:>10.4f}     {t_fp16:>12.4f}     "
          f"{t_q106:>15.4f}     "
          f"{sp_fp32:>9.2f}x  {sp_fp16:>11.2f}x  {sp_q106:>9.2f}x  "
          f"{e_fp32:>10.2e}  {e_fp16:>11.2e}  {e_q106:>10.2e}",
          flush=True)

    results.append({
        "N": N,
        "torch_ms": t_torch,
        "fuselse_fp32_ms": t_fp32,
        "fuselse_fp16_ms": t_fp16,
        "intlse_q106_ms":  t_q106,
        "speedup_fp32": sp_fp32,
        "speedup_fp16": sp_fp16,
        "speedup_q106": sp_q106,
        "abs_err_torch": float(np.mean(abs_torch)),
        "abs_err_fp32":  e_fp32,
        "abs_err_fp16":  e_fp16,
        "abs_err_q106":  e_q106,
        "rel_err_torch": float(np.median(rel_torch)),
        "rel_err_fp32":  float(np.median(rel_fp32)),
        "rel_err_fp16":  float(np.median(rel_fp16)),
        "rel_err_q106":  float(np.median(rel_q106)),
    })


OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "bench_results.json"), "w") as f:
    json.dump({
        "config": {
            "gpu": torch.cuda.get_device_name(),
            "B": B, "N_list": N_LIST,
            "range": [LO, HI],
            "warmup": WARMUP, "iters": ITERS,
            "num_seeds_acc": NUM_SEEDS_ACC,
            "reference": "fp64 inline (CPU)"},
        "results": results}, f, indent=2)
print("\nDone.", flush=True)

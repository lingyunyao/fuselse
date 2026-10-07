#!/usr/bin/env python3
"""
PAPER EXPERIMENT — particles BPF on CPU, single library, four primitives.

"""
import os, sys, json, time

import numpy as np
from scipy import special
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# `particles` (Chopin) must be cloned into ./particles/ — see top-level README.
sys.path.insert(0, os.path.join(HERE, "particles"))
# IntLSE CPU kernel lives at project root (../intlse/cpu/).
sys.path.insert(0, os.path.join(HERE, "..", "intlse", "cpu"))

import particles
from particles import state_space_models as ssm
from particles.core import SMC
from particles.resampling import Weights

from lse_ctypes_wrapper import lse_intlse_simd, HAS_SIMD


CPU_NAME = "unknown"
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if "model name" in line:
                CPU_NAME = line.split(":")[1].strip(); break
except Exception: pass
print(f"CPU: {CPU_NAME}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'auto')}")
print(f"SIMD IntLSE loaded: {HAS_SIMD}")
assert HAS_SIMD, "Need SIMD .so to run paper experiment"


# ============================================================
# Reference and primitives
# ============================================================
def fp64_lse_reference(x_fp32):
    """Ground-truth LSE in fp64. ~10^-15 absolute error, treated as exact."""
    x = np.asarray(x_fp32, dtype=np.float64)
    m = x.max()
    return float(m + np.log(np.sum(np.exp(x - m))))


def lse_scipy(x):
    return float(special.logsumexp(x))


def lse_numpy(x):
    """The inline pattern used inside particles.resampling.Weights."""
    m = x.max()
    return float(m + np.log(np.sum(np.exp(x - m))))


_orig_torch_lse = torch.logsumexp
def lse_torch(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    return float(_orig_torch_lse(x, dim=-1).item())


def lse_intlse(x):
    return float(lse_intlse_simd(x))


# ============================================================
# Run BPF, capture all per-step log-weight tensors
# ============================================================
T = 200
N_LIST = [50, 100, 200, 500, 1000, 2000]


def run_and_capture(N):
    np.random.seed(42)
    sv = ssm.StochVol()  # Pitt-Shephard 1998 default params
    x_true, y_obs = sv.simulate(T)
    fk = ssm.Bootstrap(ssm=sv, data=y_obs)
    pf = SMC(fk=fk, N=N, resampling="systematic", verbose=False)
    captured = []
    orig_init = Weights.__init__
    def cap_init(self, lw=None):
        if lw is not None:
            captured.append(np.array(lw, dtype=np.float32, copy=True))
        orig_init(self, lw=lw)
    Weights.__init__ = cap_init
    try:
        pf.run()
    finally:
        Weights.__init__ = orig_init
    return captured


# ============================================================
# Timing helper
# ============================================================
WARMUP = 100
ITERS  = 500

def time_call(fn, x):
    for _ in range(WARMUP): fn(x)
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn(x)
        samples.append((time.perf_counter() - t0) * 1e6)
    samples = np.array(samples)
    return float(np.median(samples)), \
           float(np.percentile(samples, 75) - np.percentile(samples, 25))


# ============================================================
# Run sweep
# ============================================================
print(f"\nWarmup={WARMUP}  Iters={ITERS}  T={T}")
print(f"\n{'='*40} TIMING (µs) {'='*60}")
print(f"{'N':>5s} {'#calls':>6s} {'sp_med':>7s} | "
      f"{'scipy':>10s} {'numpy':>10s} {'torch':>10s} {'IntLSE-SIMD':>12s}")
print("-" * 110)

results = []
for N in N_LIST:
    captured = run_and_capture(N)
    if not captured: continue
    spreads = [float(c.max() - c.min()) for c in captured]
    sp_med = float(np.median(spreads))

    # Pick representative late-step capture
    x = captured[-1]
    x_torch = torch.from_numpy(x).contiguous()

    # Time each primitive
    t_sp,    q_sp    = time_call(lse_scipy, x)
    t_np,    q_np    = time_call(lse_numpy, x)
    t_to,    q_to    = time_call(lambda v: lse_torch(v), x_torch)
    t_int,   q_int   = time_call(lse_intlse, x)

    # Accuracy vs fp64 ground truth, averaged over all captures
    errs_sp, errs_np, errs_to, errs_int = [], [], [], []
    for c in captured:
        ref = fp64_lse_reference(c)
        errs_sp.append(abs(lse_scipy(c) - ref))
        errs_np.append(abs(lse_numpy(c) - ref))
        c_t = torch.from_numpy(c).contiguous()
        errs_to.append(abs(lse_torch(c_t) - ref))
        errs_int.append(abs(lse_intlse(c) - ref))

    print(f"{N:>5d} {len(captured):>6d} {sp_med:>6.1f} | "
          f"{t_sp:>7.2f}±{q_sp:>2.1f}  "
          f"{t_np:>7.2f}±{q_np:>2.1f}  "
          f"{t_to:>7.2f}±{q_to:>2.1f}  "
          f"{t_int:>9.2f}±{q_int:>2.1f}", flush=True)

    results.append({
        "N": N, "n_captures": len(captured), "spread_med_nats": sp_med,
        "scipy":  {"time_us_med": t_sp, "time_us_iqr": q_sp,
                   "err_mean": float(np.mean(errs_sp)),
                   "err_max":  float(np.max(errs_sp))},
        "numpy":  {"time_us_med": t_np, "time_us_iqr": q_np,
                   "err_mean": float(np.mean(errs_np)),
                   "err_max":  float(np.max(errs_np))},
        "torch":  {"time_us_med": t_to, "time_us_iqr": q_to,
                   "err_mean": float(np.mean(errs_to)),
                   "err_max":  float(np.max(errs_to))},
        "intlse_simd": {"time_us_med": t_int, "time_us_iqr": q_int,
                        "err_mean": float(np.mean(errs_int)),
                        "err_max":  float(np.max(errs_int))},
    })

print(f"\n{'='*40} ACCURACY (mean abs error vs fp64) {'='*40}")
print(f"{'N':>5s} | {'scipy':>10s} {'numpy':>10s} {'torch':>10s} {'IntLSE-SIMD':>12s}")
print("-" * 110)
for r in results:
    print(f"{r['N']:>5d} | "
          f"{r['scipy']['err_mean']:>10.2e}  "
          f"{r['numpy']['err_mean']:>10.2e}  "
          f"{r['torch']['err_mean']:>10.2e}  "
          f"{r['intlse_simd']['err_mean']:>10.2e}")

print(f"\n{'='*40} SPEEDUP vs IntLSE-SIMD {'='*52}")
print(f"{'N':>5s} | {'scipy/intlse':>14s} {'numpy/intlse':>14s} {'torch/intlse':>14s}")
print("-" * 110)
for r in results:
    t_int = r['intlse_simd']['time_us_med']
    print(f"{r['N']:>5d} | "
          f"{r['scipy']['time_us_med']/t_int:>12.2f}x  "
          f"{r['numpy']['time_us_med']/t_int:>12.2f}x  "
          f"{r['torch']['time_us_med']/t_int:>12.2f}x")

OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "paper_results.json"), "w") as f:
    json.dump({"config": {"cpu_name": CPU_NAME, "T": T, "N_list": N_LIST,
                          "warmup": WARMUP, "iters": ITERS,
                          "library": "particles (Chopin)",
                          "model": "Stochastic Volatility (Pitt-Shephard 1998)",
                          "filter": "Bootstrap PF, systematic resampling"},
               "results": results}, f, indent=2)
print("\nDone.", flush=True)

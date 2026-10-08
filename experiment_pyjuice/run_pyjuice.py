#!/usr/bin/env python3
"""
Pyjuice HCLT MNIST experiment: GPU-only.



"""

import os, sys, json, time
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.cpp_extension import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Requires PyJuice cloned into this folder and installed:
#   cd experiment_pyjuice && git clone https://github.com/Tractables/pyjuice.git
#   pip install -e pyjuice/
sys.path.insert(0, os.path.join(SCRIPT_DIR, "pyjuice", "src"))

import pyjuice as juice
from pyjuice.layer.sum_layer import SumLayer

# ============================================================
# Compile kernels
# ============================================================
print("Compiling kernels...", flush=True)
major, minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

sparse_q16_ext = load(name="sparse_q16_ext",
                     sources=[os.path.join(SCRIPT_DIR, "sparse_q16_kernel.cu")],
                     extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
sparse_float_ext = load(name="sparse_float_ext",
                        sources=[os.path.join(SCRIPT_DIR, "sparse_float_kernel.cu")],
                        extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
print("Done.\n", flush=True)

# ============================================================
# Setup
# ============================================================
device = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}", flush=True)

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
valid_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True)
train_data = train_dataset.data.reshape(60000, 28 * 28)
valid_data = valid_dataset.data.reshape(10000, 28 * 28)

BATCH_SIZE = 512
valid_loader = DataLoader(TensorDataset(valid_data),
                          batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# ============================================================
# Monkey-patch forward implementations
# ============================================================
_orig_forward        = SumLayer._forward
_orig_forward_sparse = SumLayer._forward_sparse


def _make_sparse_dispatcher(ext_call):
    """Return a _forward_sparse that calls ext_call(node_mars, element_mars,
    params, nids, cids, pids, block_size) instead of pyjuice's path."""
    def _fn(self, node_mars, element_mars, params, nids, cids, pids,
            local_ids=None, partition_id=-1, propagation_alg="LL", **kwargs):
        if propagation_alg != "LL":
            return _orig_forward_sparse(self, node_mars, element_mars, params,
                                        nids, cids, pids, local_ids=local_ids,
                                        partition_id=partition_id,
                                        propagation_alg=propagation_alg, **kwargs)
        if local_ids is not None:
            nids = nids[local_ids]; cids = cids[local_ids]; pids = pids[local_ids]
        ext_call(node_mars, element_mars, params, nids, cids, pids, self.block_size)
        return None
    return _fn


def _force_sparse_forward(self, node_mars, element_mars, params, nids, cids, pids,
                          local_ids=None, partition_id=-1, mode=None,
                          force_use_bf16=False, force_use_fp32=False,
                          propagation_alg="LL", **kwargs):
    """Override _forward to always route through the sparse path."""
    self._forward_sparse(node_mars, element_mars, params, nids, cids, pids,
                         local_ids=local_ids, partition_id=partition_id,
                         propagation_alg=propagation_alg, **kwargs)


def _torch_sparse_forward(self, node_mars, element_mars, params, nids, cids, pids,
                          local_ids=None, partition_id=-1,
                          propagation_alg="LL", **kwargs):
    """Reference sparse-forward using torch.logsumexp.
    """
    if propagation_alg != "LL":
        return _orig_forward_sparse(self, node_mars, element_mars, params,
                                    nids, cids, pids, local_ids=local_ids,
                                    partition_id=partition_id,
                                    propagation_alg=propagation_alg, **kwargs)
    if local_ids is not None:
        nids = nids[local_ids]; cids = cids[local_ids]; pids = pids[local_ids]

    # Block-size expansion (mirrors pyjuice/src/pyjuice/layer/sum_layer.py:1227-1231).
    num_nblocks = nids.size(0)
    num_edges   = cids.size(1)
    bs = self.block_size
    nids = (nids[:, None].repeat(1, bs)
            + torch.arange(0, bs, device=nids.device)[None, :]
            ).reshape(num_nblocks * bs)
    cids = cids[:, None, :].repeat(1, bs, 1).reshape(num_nblocks * bs, num_edges)
    pids = (pids[:, None, :].repeat(1, bs, 1)
            + torch.arange(0, bs, device=cids.device)[None, :, None]
            ).reshape(num_nblocks * bs, num_edges)

    # Weighted logsumexp:  log sum_i w_i exp(x_i)  =  LSE(x_i + log w_i).
    ch_mars = element_mars[cids]                                    # (N, E, batch)
    log_w   = params[pids].clamp(min=1e-30).log().unsqueeze(-1)     # (N, E, 1)
    node_mars[nids] = torch.logsumexp(ch_mars + log_w, dim=1)
    return None


gpu_int_sparse   = _make_sparse_dispatcher(sparse_q16_ext.sparse_forward_q16)
gpu_fused_sparse = _make_sparse_dispatcher(sparse_float_ext.sparse_forward_float)


def install_kernel(kn):
    if kn == "pyjuice_default":
        SumLayer._forward = _orig_forward
        SumLayer._forward_sparse = _orig_forward_sparse
        return
    SumLayer._forward = _force_sparse_forward
    if kn == "pyjuice_sparse":
        SumLayer._forward_sparse = _orig_forward_sparse
    elif kn == "torch_lse":
        SumLayer._forward_sparse = _torch_sparse_forward
    elif kn == "fused":
        SumLayer._forward_sparse = gpu_fused_sparse
    elif kn == "int":
        SumLayer._forward_sparse = gpu_int_sparse
    else:
        raise ValueError(f"unknown kernel name: {kn}")


def restore_kernel():
    SumLayer._forward = _orig_forward
    SumLayer._forward_sparse = _orig_forward_sparse


# ============================================================
# Timing helpers
# ============================================================
def time_forward_gpu(pc, loader, n_batches=20, warmup=5):
    """Time forward and return (mean_time_ms, per_sample_LL_tensor).
    """
    times_ms, lls, count = [], [], 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            ll = pc(x)
            e.record()
            torch.cuda.synchronize()
            if count >= warmup:
                times_ms.append(s.elapsed_time(e))
                lls.append(ll.detach().cpu())
            count += 1
            if count >= warmup + n_batches:
                break
    per_sample_lls = torch.cat(lls) if lls else torch.empty(0)
    return float(np.mean(times_ms)), per_sample_lls


def per_sample_lls_full_val(pc, loader):
    """Run a non-timed pass over the full loader and return per-sample LLs.
    """
    lls = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            ll = pc(x)
            lls.append(ll.detach().cpu())
    return torch.cat(lls)


# ============================================================
# Main sweep
# ============================================================
NUM_LATENTS = [4, 8, 16, 32]
GPU_BATCHES, GPU_WARMUP = 20, 5

KERNEL_ORDER = ["pyjuice_default", "pyjuice_sparse", "torch_lse", "fused", "int"]

print("\n" + "=" * 110, flush=True)
print(f"{'K':>3s} | "
      + "  ".join(f"{kn[:9]:>9s}" for kn in KERNEL_ORDER)
      + f" | {'err_pj':>9s} {'err_fused':>9s} {'err_int':>9s} "
        f"{'max_int':>9s}", flush=True)
print("-" * 110, flush=True)

results = []
for K in NUM_LATENTS:
    np.random.seed(42 + K)
    torch.manual_seed(42 + K)
    torch.cuda.manual_seed_all(42 + K)
    print(f"\n[num_latents={K}] Building HCLT...", flush=True)
    ns = juice.structures.HCLT(train_data.float().to(device), num_latents=K)
    pc_gpu = juice.compile(ns); pc_gpu.to(device)

    measurements = []
    full_val_lls = {}     # kernel name -> per-sample LL tensor (full val set)

    for kn in KERNEL_ORDER:
        install_kernel(kn)

        # Timed pass: GPU_BATCHES batches after GPU_WARMUP warmups.
        t, _ = time_forward_gpu(pc_gpu, valid_loader,
                                n_batches=GPU_BATCHES, warmup=GPU_WARMUP)

        # Accuracy pass: full validation set, no timing, fresh data iter.
        per_sample_lls = per_sample_lls_full_val(pc_gpu, valid_loader)
        full_val_lls[kn] = per_sample_lls

        measurements.append({"device": "gpu", "kernel": kn, "time_ms": t})

    # Compute per-sample errors against torch_lse reference.
    ref = full_val_lls["torch_lse"]
    for m in measurements:
        kn = m["kernel"]
        diff = (full_val_lls[kn] - ref).abs()
        m["err_mean"]    = float(diff.mean())
        m["err_max"]     = float(diff.max())
        m["err_p99"]     = float(diff.quantile(0.99))
        m["mean_ll"]     = float(full_val_lls[kn].mean())
        # Backward-compatible bias metric (current paper number):
        m["abs_err_bias"] = abs(float(full_val_lls[kn].mean())
                                - float(ref.mean()))

    restore_kernel()
    del pc_gpu, ns, full_val_lls
    torch.cuda.empty_cache()

    r = {"num_latents": K, "measurements": measurements,
         "ref_ll": float(ref.mean()),
         "n_samples": int(ref.numel())}
    results.append(r)

    times = "  ".join(f"{m['time_ms']:>9.2f}" for m in measurements)
    err_pj = next(m for m in measurements if m["kernel"] == "pyjuice_default")["err_mean"]
    err_f  = next(m for m in measurements if m["kernel"] == "fused")["err_mean"]
    err_i  = next(m for m in measurements if m["kernel"] == "int")["err_mean"]
    max_i  = next(m for m in measurements if m["kernel"] == "int")["err_max"]
    print(f"{K:>3d} | {times} | "
          f"{err_pj:>9.2e} {err_f:>9.2e} {err_i:>9.2e} {max_i:>9.2e}", flush=True)

# ============================================================
# Save JSON + LaTeX table
# ============================================================
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

json_path = os.path.join(OUT_DIR, "pyjuice_results.json")
with open(json_path, "w") as f:
    json.dump({"gpu": GPU_NAME,
               "batch_size": BATCH_SIZE,
               "num_latents_grid": NUM_LATENTS,
               "results": results}, f, indent=2)
print(f"\nSaved JSON to {json_path}", flush=True)


def m(r, device, kernel, key):
    for x in r["measurements"]:
        if x["device"] == device and x["kernel"] == kernel:
            return x[key]
    return float("nan")


def fmt(v, w=7, p=2):
    if isinstance(v, float) and (v != v):  # NaN
        return f"{'N/A':>{w}s}"
    return f"{v:>{w}.{p}f}"


tex_path = os.path.join(OUT_DIR, "pyjuice_table.tex")
with open(tex_path, "w") as f:
    f.write("% =========================================================================\n")
    f.write("% PyJuice HCLT MNIST forward pass - GPU table\n")
    f.write("% =========================================================================\n")
    f.write(f"% Hardware    : {GPU_NAME}\n")
    f.write(f"% Workload    : HCLT compiled circuit, MNIST validation set, batch={BATCH_SIZE}\n")
    f.write(f"% Measurement : end-to-end forward pc(x) (ms, mean over {GPU_BATCHES} batches)\n")
    f.write("% Kernels     : pj_default  = pyjuice's actual default dispatch (mixed mode)\n")
    f.write("%               pj_sparse   = forced sparse + pyjuice's Triton kernel\n")
    f.write("%               torch_lse   = forced sparse + torch.logsumexp reference\n")
    f.write("%               fused       = forced sparse + our FuseLSE CUDA kernel\n")
    f.write("%               int         = forced sparse + our IntLSE Q16.16 CUDA kernel\n")
    f.write("% Error       : per-sample mean and max |LL_kernel - LL_torch_lse| over\n")
    f.write("%               the full MNIST validation set (torch_lse is the reference)\n")
    f.write("% Auto-generated by run_pyjuice.py - do not edit by hand.\n")
    f.write("%\n")
    f.write("% Columns: K | t_pj_default | t_pj_sparse | t_torch | t_fused | t_int  (ms)\n")
    f.write("%            | err_mean_pj_default | err_mean_fused | err_mean_int\n")
    f.write("%            | err_max_pj_default  | err_max_fused  | err_max_int\n")
    f.write("% -------------------------------------------------------------------------\n")
    for r in results:
        f.write(f"  {r['num_latents']:>3d}"
                + " & " + fmt(m(r,'gpu','pyjuice_default','time_ms'), 7, 2)
                + " & " + fmt(m(r,'gpu','pyjuice_sparse','time_ms'),  7, 2)
                + " & " + fmt(m(r,'gpu','torch_lse','time_ms'),       7, 2)
                + " & " + fmt(m(r,'gpu','fused','time_ms'),           7, 2)
                + " & " + fmt(m(r,'gpu','int','time_ms'),             7, 2)
                + " & " + fmt(m(r,'gpu','pyjuice_default','err_mean'), 9, 4)
                + " & " + fmt(m(r,'gpu','fused','err_mean'),           9, 4)
                + " & " + fmt(m(r,'gpu','int','err_mean'),             9, 4)
                + " & " + fmt(m(r,'gpu','pyjuice_default','err_max'),  9, 4)
                + " & " + fmt(m(r,'gpu','fused','err_max'),            9, 4)
                + " & " + fmt(m(r,'gpu','int','err_max'),              9, 4)
                + " \\\\\n")
print(f"Saved GPU table to {tex_path}", flush=True)
print("Done.", flush=True)

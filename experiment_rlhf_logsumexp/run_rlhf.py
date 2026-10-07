"""
RLHF log-softmax benchmark: GPU-only.

"""

import os
import sys
import json
import shutil

local_bin = os.path.expanduser("~/.local/bin")
if local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = local_bin + ":" + os.environ.get("PATH", "")
if not os.environ.get("CUDA_HOME"):
    nvcc = shutil.which("nvcc")
    if nvcc:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc))

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import trl
from trl.trainer.utils import selective_log_softmax

import rlhf_timing as rt


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

assert torch.cuda.is_available(), "CUDA required"
DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
print(f"GPU: {GPU_NAME}", flush=True)

major, minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

print("Compiling kernels...", flush=True)
# GPU kernels loaded from the central intlse/ and fuselse/ folders so all
# experiments share the same kernel binaries.
INTLSE_GPU = os.path.join(HERE, "..", "intlse", "gpu")
FUSELSE_GPU = os.path.join(HERE, "..", "fuselse", "gpu")
lse_ext = load(name="lse_rlhf_ext",
               sources=[os.path.join(INTLSE_GPU, "lse_kernel.cu")],
               extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
float_ext = load(name="lse_float_rlhf_ext",
                 sources=[os.path.join(FUSELSE_GPU, "fused_fp32_kernel.cu")],
                 extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
print("Done.\n", flush=True)

_orig_lse = torch.logsumexp
gpu_int_lse   = rt.make_kernel_lse_wrapper(lse_ext.batched_logsumexp,    _orig_lse)
gpu_fused_lse = rt.make_kernel_lse_wrapper(float_ext.onepass_lse_float,  _orig_lse)


# ============================================================
# CONFIG
# ============================================================
T = 1024
GPU_WARMUP, GPU_ITERS = 3, 50

# (display name, vocabulary size V, batch size B)
VOCAB_CONFIGS = [
    ("Llama-2",  32000, 8),
    ("GPT-2",    50257, 8),
    ("Llama-3", 128256, 4),
    ("Gemma",   256128, 2),
]
print(f"Config: T={T}, GPU iters={GPU_ITERS}", flush=True)


# ============================================================
# Data + model
# ============================================================
print("Loading GPT-2 tokenizer and backbone...", flush=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
HIDDEN = GPT2LMHeadModel.from_pretrained("gpt2").config.n_embd

ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
texts = []
for example in ds:
    texts.append(example["chosen"])
    if len(texts) >= 16:
        break

B_max = max(b for _, _, b in VOCAB_CONFIGS)
encoded = tokenizer(texts[:B_max], return_tensors="pt",
                    padding="max_length", truncation=True, max_length=T)
all_input_ids = encoded["input_ids"].to(DEVICE)
all_attention_mask = encoded["attention_mask"].to(DEVICE)


def build_logits(V, B):
    """Run GPT-2 backbone once on GPU and return (B, T-1, V) float32 logits.
    Also time the forward pass (transformer + LM head) as a reference
    for how big the LSE step is relative to producing the logits."""
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    if V != 50257:
        head = nn.Linear(HIDDEN, V, bias=False).to(DEVICE)
        head.weight.data.normal_(0, 0.02)
        model.lm_head = head
    input_ids = all_input_ids[:B]
    attention_mask = all_attention_mask[:B]

    def fwd_only():
        with torch.no_grad():
            hidden = model.transformer(
                input_ids=input_ids, attention_mask=attention_mask)
            return model.lm_head(hidden.last_hidden_state[:, :-1, :])
    # Time just the forward (transformer + LM head) for reference.
    t_fwd_ref = rt.time_call(fwd_only, warmup=GPU_WARMUP, iters=GPU_ITERS)
    logits = fwd_only().contiguous()
    targets = input_ids[:, 1:].clamp(max=V - 1).contiguous()
    del model
    torch.cuda.empty_cache()
    return logits, targets, t_fwd_ref


# ============================================================
# Benchmark sweep
# ============================================================
print("\n" + "=" * 90, flush=True)
print(f"{'Model':>10s} {'V':>8s} {'B':>3s} | "
      f"{'gpu_torch':>9s} {'gpu_fused':>9s} {'gpu_int':>9s} | "
      f"{'lse%':>5s} {'err_int':>9s}", flush=True)
print("-" * 90, flush=True)

results = []
for name, V, B in VOCAB_CONFIGS:
    logits_gpu, targets_gpu, t_fwd_ref = build_logits(V, B)

    # ---------- GPU reference (torch) ----------
    torch.logsumexp = _orig_lse
    ref_gpu = selective_log_softmax(logits_gpu, targets_gpu)

    measurements = []  # 3 GPU entries

    # ---------- GPU: torch / fused / int ----------
    gpu_kernels = [("torch", _orig_lse),
                   ("fused", gpu_fused_lse),
                   ("int",   gpu_int_lse)]
    for kn, lse_fn in gpu_kernels:
        torch.logsumexp = lse_fn
        t = rt.time_call(lambda: selective_log_softmax(logits_gpu, targets_gpu),
                         warmup=GPU_WARMUP, iters=GPU_ITERS)
        out = selective_log_softmax(logits_gpu, targets_gpu)
        err = (ref_gpu - out).abs().mean().item()
        measurements.append({"device": "gpu", "kernel": kn,
                             "time_ms": t, "abs_err": err})

    # ---------- LSE fraction (GPU torch baseline) ----------
    # SLS calls torch.logsumexp B times, once per row of `logits`,
    # at shape (T-1, V). Time one such call and multiply by B.
    torch.logsumexp = _orig_lse
    lg = logits_gpu[0]  # (T-1, V)
    t_one_lse = rt.time_call(lambda: torch.logsumexp(lg, dim=-1),
                             warmup=GPU_WARMUP, iters=GPU_ITERS)
    t_sls_torch_gpu = measurements[0]["time_ms"]
    lse_frac_pct = min(100.0, t_one_lse * B / t_sls_torch_gpu * 100)

    torch.logsumexp = _orig_lse
    del logits_gpu, targets_gpu, ref_gpu
    torch.cuda.empty_cache()

    r = {"model": name, "V": V, "B": B, "T": T,
         "t_fwd_ref_ms": t_fwd_ref,
         "lse_frac_pct": lse_frac_pct,
         "measurements": measurements}
    results.append(r)

    g_t, g_f, g_q = (m["time_ms"] for m in measurements[:3])
    err_int_gpu = measurements[2]["abs_err"]
    print(f"{name:>10s} {V:>8,d} {B:>3d} | "
          f"{g_t:>8.2f}  {g_f:>8.2f}  {g_q:>8.2f} | "
          f"{lse_frac_pct:>4.1f}% {err_int_gpu:>9.5f}", flush=True)


# ============================================================
# Save JSON
# ============================================================
json_path = os.path.join(OUT_DIR, "rlhf_results.json")
with open(json_path, "w") as f:
    json.dump({"gpu": GPU_NAME,
               "trl_version": trl.__version__,
               "T": T, "vocab_configs": VOCAB_CONFIGS,
               "results": results}, f, indent=2)
print(f"\nSaved JSON to {json_path}", flush=True)


# ============================================================
# LaTeX tables
# ============================================================
def m(r, device, kernel, key):
    for x in r["measurements"]:
        if x["device"] == device and x["kernel"] == kernel:
            return x[key]
    return float("nan")


tex_path = os.path.join(OUT_DIR, "rlhf_table.tex")
with open(tex_path, "w") as f:
    f.write("% =========================================================================\n")
    f.write("% RLHF selective_log_softmax benchmark - GPU table\n")
    f.write("% =========================================================================\n")
    f.write(f"% Hardware    : {GPU_NAME}\n")
    f.write(f"% Workload    : GPT-2 backbone + swapped LM head, sequence length T={T}\n")
    f.write(f"% Measurement : end-to-end TRL selective_log_softmax (ms, mean over {GPU_ITERS} iters)\n")
    f.write("% Error       : mean absolute error vs torch.logsumexp reference\n")
    f.write("% Auto-generated by run_rlhf.py - do not edit by hand.\n")
    f.write("%\n")
    f.write("% Columns: V (vocab) | B (batch) | LSE%% (share of SLS time in logsumexp) |\n")
    f.write("%          t_torch | t_fused | t_int  (ms)\n")
    f.write("%          err_fused | err_int  (mean abs err)\n")
    f.write("% -------------------------------------------------------------------------\n")
    for r in results:
        Vlbl = f"{r['V'] // 1000}K"
        f.write(f"  {Vlbl:>4s} & {r['B']:d} & "
                f"{r['lse_frac_pct']:>4.1f}\\% & "
                f"{m(r,'gpu','torch','time_ms'):>6.2f} & "
                f"{m(r,'gpu','fused','time_ms'):>6.2f} & "
                f"{m(r,'gpu','int',  'time_ms'):>6.2f} & "
                f"{m(r,'gpu','fused','abs_err'):>7.5f} & "
                f"{m(r,'gpu','int',  'abs_err'):>7.5f} \\\\"
                f"  % {r['model']}\n")
print(f"Saved GPU table to {tex_path}", flush=True)
print("Done.", flush=True)

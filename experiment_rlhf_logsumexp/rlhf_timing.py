"""
Timing helpers for the RLHF log-softmax benchmark.

"""

import torch


def time_call(fn, warmup=3, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def make_kernel_lse_wrapper(kernel_fn, original_logsumexp):
    def fn(input, dim=-1, keepdim=False):
        if input.is_cuda and input.dtype == torch.float32:
            d = dim if dim >= 0 else input.dim() + dim
            if d == input.dim() - 1 and input.shape[-1] >= 2:
                flat = input.reshape(-1, input.shape[-1]).contiguous()
                result = kernel_fn(flat)
                out_shape = list(input.shape[:-1])
                if keepdim:
                    out_shape.append(1)
                return result.reshape(out_shape)
        return original_logsumexp(input, dim=dim, keepdim=keepdim)
    return fn



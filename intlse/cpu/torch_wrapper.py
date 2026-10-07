"""PyTorch-facing IntLSE wrapper with exact autograd fallback."""

import numpy as np
import torch

from lse_ctypes_wrapper import logsumexp_intlse


def logsumexp_intlse_torch(input_tensor, dim=-1, keepdim=False):
    normalized_dim = dim if dim >= 0 else input_tensor.dim() + dim
    supported = (
        input_tensor.device.type == "cpu"
        and input_tensor.dtype == torch.float32
        and normalized_dim == input_tensor.dim() - 1
        and input_tensor.shape[-1] >= 2
        and not input_tensor.requires_grad
    )
    if not supported:
        return torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)

    contiguous_input = input_tensor.detach().contiguous()
    output = logsumexp_intlse(
        contiguous_input.numpy(),
        axis=-1,
        keepdims=keepdim,
    )
    if isinstance(output, np.ndarray):
        return torch.from_numpy(output)
    return torch.tensor(output, dtype=input_tensor.dtype)

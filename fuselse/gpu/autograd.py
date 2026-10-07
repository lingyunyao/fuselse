"""Autograd wrapper for the FuseLSE CUDA extension."""

import torch


def make_fuselse_logsumexp(extension, robust=True):
    forward_kernel = (
        extension.onepass_lse_float_robust
        if robust
        else extension.onepass_lse_float
    )
    backward_kernel = extension.onepass_lse_float_backward
    reference_logsumexp = torch.logsumexp

    class FuseLSEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor):
            row_size = input_tensor.shape[-1]
            flat_input = input_tensor.reshape(-1, row_size).contiguous()
            output = forward_kernel(flat_input)
            ctx.save_for_backward(flat_input, output)
            ctx.input_shape = tuple(input_tensor.shape)
            return output.reshape(input_tensor.shape[:-1])

        @staticmethod
        def backward(ctx, grad_output):
            flat_input, output = ctx.saved_tensors
            flat_grad_output = grad_output.reshape(-1).contiguous()
            grad_input = backward_kernel(
                flat_grad_output,
                flat_input,
                output,
            )
            return grad_input.reshape(ctx.input_shape)

    def logsumexp(input_tensor, dim=-1, keepdim=False):
        normalized_dim = dim if dim >= 0 else input_tensor.dim() + dim
        supported = (
            input_tensor.is_cuda
            and input_tensor.dtype == torch.float32
            and normalized_dim == input_tensor.dim() - 1
            and input_tensor.shape[-1] >= 2
        )
        if not supported:
            return reference_logsumexp(
                input_tensor,
                dim=dim,
                keepdim=keepdim,
            )
        if torch.is_grad_enabled() and input_tensor.requires_grad:
            result = FuseLSEFunction.apply(input_tensor)
        elif input_tensor.dim() == 2 and input_tensor.is_contiguous():
            # Common case: call the kernel directly, without autograd or reshapes.
            result = forward_kernel(input_tensor)
        else:
            row_size = input_tensor.shape[-1]
            flat_input = input_tensor.reshape(-1, row_size).contiguous()
            result = forward_kernel(flat_input).reshape(input_tensor.shape[:-1])
        return result.unsqueeze(-1) if keepdim else result

    return logsumexp

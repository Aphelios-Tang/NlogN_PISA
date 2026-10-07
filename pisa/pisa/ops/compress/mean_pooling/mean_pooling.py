"""Mean-pooling dispatch and recursive pyramid construction."""

import torch

from .mean_pooling_torch import torch_mean_pool


def mean_pool(keys, size, *, backend="auto", fp32_accumulation=False):
    """Average valid tokens, using an optional forward-only CUDA kernel.

    CUDA construction reduces in the input dtype. Set fp32_accumulation for
    incremental cache updates. The Torch path uses FP32 (or FP64) accumulation.
    Auto preserves Torch autograd and FP64 behavior. Explicit Triton requests
    reject inputs requiring gradients instead of silently detaching the result.
    """
    if size < 1:
        raise ValueError("pool size must be positive")
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("pooling backend must be auto, torch, or triton")
    if backend == "torch" or (backend == "auto" and keys.numel() == 0):
        return torch_mean_pool(keys, size)
    reason = None
    if not keys.is_cuda:
        reason = "Triton pooling requires CUDA tensors"
    elif keys.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        reason = "Triton pooling requires float16, bfloat16, or float32 dtype"
    elif torch.is_grad_enabled() and keys.requires_grad:
        reason = "Triton pooling does not support gradients; use backend='torch' or 'auto'"
    elif keys.numel() == 0:
        reason = "Triton pooling requires nonempty input"
    if reason is not None:
        if backend == "triton":
            raise ValueError(reason)
        return torch_mean_pool(keys, size)
    try:
        from pisa.ops.compress.mean_pooling.mean_pooling_triton import mean_pool_forward
    except ImportError as error:
        if backend == "triton":
            raise RuntimeError("Triton is required for the requested pooling backend") from error
        return torch_mean_pool(keys, size)
    return mean_pool_forward(keys, size, fp32_accumulation=fp32_accumulation)



def build_pyramid(keys, block_size, *, backend="auto", max_level=None):
    """Return leaf means first, stopping at the root or an inclusive level limit."""
    if max_level is not None and (type(max_level) is not int or max_level < 0):
        raise ValueError("max_level must be a nonnegative integer or None")
    levels = [mean_pool(keys, block_size, backend=backend)]
    while levels[-1].shape[1] > 1 and (max_level is None or len(levels) <= max_level):
        levels.append(mean_pool(levels[-1], 2, backend=backend))
    return levels


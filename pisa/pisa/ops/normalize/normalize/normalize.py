"""RMS normalization dispatch, loading the CUDA kernel only when needed."""
import importlib.util
import math

import torch

from .normalize_torch import rms_norm_torch


def rms_norm_fn(x, weight, eps=1e-5, *, num_warps=None):
    """Use automatic CUDA tuning, or pin num_warps for repeatable reductions."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be positive and finite")
    if x.ndim < 1 or x.shape[-1] < 1 or weight.shape != (x.shape[-1],):
        raise ValueError("RMSNorm input width must match its scale")
    if weight.device != x.device:
        raise ValueError("RMSNorm input and scale must share a device")
    if x.is_cuda and x.dtype in (torch.float32, torch.bfloat16) and importlib.util.find_spec("triton") is not None:
        from .normalize_triton import rms_norm
        return rms_norm(x, weight, eps, num_warps=num_warps)
    return rms_norm_torch(x, weight, eps)

"""Rotary tensor dispatch; frequency initialization belongs to the model."""
import importlib.util

import torch

from .rotary_torch import rotary_torch


def rotary_fn(x, frequency, positions, *, use_triton=True):
    frequency = frequency.flatten().to(device=x.device, dtype=torch.float32)
    if (use_triton and x.is_cuda and x.ndim == 4 and x.dtype in (torch.float32, torch.bfloat16)
            and not positions.requires_grad and importlib.util.find_spec("triton") is not None):
        from .rotary_triton import rotary_embedding
        return rotary_embedding(x, frequency, positions)
    return rotary_torch(x, frequency, positions)

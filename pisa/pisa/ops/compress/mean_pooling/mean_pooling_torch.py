"""Mean over valid tokens, including incomplete tail blocks."""

import torch
import torch.nn.functional as F


def torch_mean_pool(keys, size):
    batch, length, heads, dim = keys.shape
    blocks = (length + size - 1) // size
    dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    padded = F.pad(keys.to(dtype), (0, 0, 0, 0, 0, blocks * size - length))
    sums = padded.reshape(batch, blocks, size, heads, dim).sum(2)
    counts = (length - torch.arange(blocks, device=keys.device) * size).clamp(1, size)
    return (sums / counts[None, :, None, None]).to(keys.dtype)


"""Differentiable Torch attention over selected original K/V blocks."""

import torch


def sparse_attention_torch(q, k, v, indices, block_size, scale, positions, causal):
    batch, length, heads_q, dim = q.shape
    heads = k.shape[2]
    scale = dim**-0.5 if scale is None else scale
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    batch_ids = torch.arange(batch, device=q.device)[:, None, None, None]
    head_ids = torch.arange(heads, device=q.device)[None, None, :, None]
    offsets = torch.arange(block_size, device=q.device)
    results = []
    for start in range(0, length, 32):
        ids = indices[:, start : start + 32]
        tokens = (ids[..., None] * block_size + offsets).flatten(-2)
        valid = (tokens >= 0) & (tokens < k.shape[1])
        if causal:
            valid &= tokens <= positions[:, start : start + 32, None, None]
        tokens = tokens.clamp(0, k.shape[1] - 1)
        keys = k[batch_ids, tokens, head_ids].to(dtype)
        values = v[batch_ids, tokens, head_ids].to(dtype)
        query = q[:, start : start + 32].to(dtype).reshape(batch, -1, heads, heads_q // heads, dim)
        with torch.autocast(q.device.type, enabled=False):
            logits = torch.einsum("bthgd,bthsd->bthgs", query, keys) * scale
        masked = logits.masked_fill(~valid[..., None, :], -torch.inf)
        # Avoid NaN gradients even when a caller supplies no valid blocks.
        any_valid = valid.any(-1)[..., None, None]
        masked = torch.where(any_valid, masked, torch.zeros_like(masked))
        probability = masked.softmax(-1) * valid[..., None, :]
        with torch.autocast(q.device.type, enabled=False):
            result = torch.einsum("bthgs,bthsv->bthgv", probability, values)
        results.append(result.reshape(batch, -1, heads_q, v.shape[-1]).to(q.dtype))
    return torch.cat(results, 1)

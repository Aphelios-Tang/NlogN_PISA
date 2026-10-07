"""Shape and sequence-boundary contracts shared by the public operators."""

from itertools import pairwise

import torch


def sequence_slices(cu_seqlens, length):
    if cu_seqlens is None:
        return [(0, length)]
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be a one-dimensional integer tensor")
    boundaries = cu_seqlens.detach().cpu().tolist()
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != length:
        raise ValueError("cu_seqlens must start at zero and end at the key length")
    if any(a >= b for a, b in pairwise(boundaries)):
        raise ValueError("packed sequences must be nonempty and strictly ordered")
    return list(pairwise(boundaries))


def validate_qkv(q, k, v=None):
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, tokens, heads, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share batch size and head dimension")
    if min(*q.shape, *k.shape) <= 0 or q.shape[2] % k.shape[2]:
        raise ValueError(
            "nonempty tensors and a whole number of query heads per KV head are required"
        )
    if q.device != k.device or q.dtype != k.dtype or not q.is_floating_point():
        raise ValueError("q and k must share a floating dtype and device")
    if v is not None and (
        v.ndim != 4 or v.shape[:3] != k.shape[:3] or v.device != k.device or v.dtype != k.dtype
    ):
        raise ValueError("v must share k's batch, length, heads, dtype and device")


def query_positions(q, k, query_offset=None):
    if query_offset is None or isinstance(query_offset, int):
        offset = k.shape[1] - q.shape[1] if query_offset is None else int(query_offset)
        if offset < 0 or offset + q.shape[1] > k.shape[1]:
            raise ValueError("queries must refer to positions within their key sequence")
        # Ordinary prefill/decode bounds are known from shapes. Checking them
        # on the host avoids synchronizing two device reductions per call.
        return torch.arange(
            offset, offset + q.shape[1], device=q.device, dtype=torch.long,
        )[None].expand(q.shape[0], -1)
    offset = k.shape[1] - q.shape[1] if query_offset is None else query_offset
    offset = torch.as_tensor(offset, device=q.device, dtype=torch.long)
    if offset.ndim > 1 or offset.numel() not in (1, q.shape[0]):
        raise ValueError("query_offset must be a scalar or one offset per batch element")
    positions = offset.reshape(-1, 1) + torch.arange(q.shape[1], device=q.device)
    positions = positions.expand(q.shape[0], -1)
    if bool((positions < 0).any()) or bool((positions >= k.shape[1]).any()):
        raise ValueError("queries must refer to positions within their key sequence")
    return positions


def packed_queries(q, k, cu_seqlens, query_offset=None):
    """Yield query/key slices and a local offset; indices are document-relative."""
    if q.shape[0] != 1:
        raise ValueError("packed inputs require batch size one")
    segments = sequence_slices(cu_seqlens, k.shape[1])
    if q.shape[1] == k.shape[1]:
        if query_offset is not None and bool(torch.as_tensor(query_offset).ne(0).any()):
            raise ValueError("packed prefill has local query_offset zero")
        for start, end in segments:
            yield start, end, start, end, 0
    elif q.shape[1] == len(segments):
        offsets = None if query_offset is None else torch.as_tensor(query_offset).reshape(-1)
        if offsets is not None and offsets.numel() not in (1, len(segments)):
            raise ValueError("packed decode needs a scalar or one offset per document")
        for index, (start, end) in enumerate(segments):
            offset = (
                end - start - 1
                if offsets is None
                else int(offsets[min(index, offsets.numel() - 1)])
            )
            yield index, index + 1, start, end, offset
    else:
        raise ValueError("packed inputs support full prefill or one decode query per document")

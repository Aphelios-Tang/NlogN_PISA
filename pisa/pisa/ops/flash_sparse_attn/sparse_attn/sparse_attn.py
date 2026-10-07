"""Attention over selected original K/V blocks, with a shared softmax."""

import torch

from .sparse_attn_torch import sparse_attention_torch

from pisa.utils.validation import packed_queries, query_positions, validate_qkv


def sparse_attention(
    q,
    k,
    v,
    indices,
    block_size=64,
    scale=None,
    cu_seqlens=None,
    query_offset=None,
    backend="torch",
    causal=True,
):
    validate_qkv(q, k, v)
    if indices.ndim != 4 or indices.shape[:3] != (*q.shape[:2], k.shape[2]):
        raise ValueError("indices must have shape [B,Tq,Hkv,K]")
    if indices.dtype not in (torch.int32, torch.int64) or indices.device != q.device:
        raise ValueError("indices must be integers on the query device")
    if block_size < 1 or backend not in ("torch", "auto", "triton", "fla"):
        raise ValueError("invalid block_size or attention backend")
    if cu_seqlens is not None:
        segments = list(packed_queries(q, k, cu_seqlens, query_offset))
        if q.shape[1] == k.shape[1] and (
            backend in ("triton", "fla") or (backend == "auto" and q.is_cuda)
        ):
            from pisa.ops.flash_sparse_attn.sparse_attn.sparse_attn_fla import selected_attention

            return selected_attention(
                q, k, v, indices, block_size, scale, None, causal,
                cu_seqlens=cu_seqlens,
            )
        outputs = []
        for qs, qe, ks, ke, offset in segments:
            outputs.append(
                sparse_attention(
                    q[:, qs:qe],
                    k[:, ks:ke],
                    v[:, ks:ke],
                    indices[:, qs:qe],
                    block_size,
                    scale,
                    query_offset=offset,
                    backend=backend,
                    causal=causal,
                )
            )
        return torch.cat(outputs, 1)
    positions = query_positions(q, k, query_offset)
    if backend in ("triton", "fla") or (backend == "auto" and q.is_cuda):
        from pisa.ops.flash_sparse_attn.sparse_attn.sparse_attn_fla import selected_attention

        return selected_attention(q, k, v, indices, block_size, scale, positions, causal)
    return sparse_attention_torch(q, k, v, indices, block_size, scale, positions, causal)

"""Learned compressed and selected attention: validation and dispatch."""

from collections.abc import Callable

from torch import Tensor

from pisa.utils.validation import validate_qkv
from .nsa_cmp_select_torch import nsa_cmp_select_torch, nsa_sequence_layout


def native_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    *,
    compressed_gate: Tensor | None = None,
    selected_gate: Tensor | None = None,
    block_size: int = 64,
    topk: int = 8,
    compression_size: int = 32,
    compression_stride: int = 16,
    compression_k: Tensor | None = None,
    rope_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    rope_position_mode: str = "document",
    cu_seqlens: Tensor | None = None,
    query_offset: int | Tensor | None = None,
    scale: float | None = None,
    force_first: bool = True,
    force_recent: int = 2,
    query_chunk_size: int = 128,
    backend: str = "torch",
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Combine learned compressed attention and block-selected attention.

    Q/K are the positioned tensors used by selected attention. Pass raw K as
    ``compression_k`` and a ``rope_fn(keys, positions)`` callback to position
    compressed keys *after* pooling. ``rope_position_mode`` must match Q/K:
    document-relative positions or continuous packed-row sequence positions.
    Supported CUDA BF16 full prefill uses the public NSA compression kernels
    and FLA selected attention. Other inputs retain Torch compression, SDPA
    compressed output, and chunked Torch routing. Gates are independent.
    """
    validate_qkv(q, k, v)
    if compression_size <= 0 or compression_stride <= 0 or block_size <= 0 or topk <= 0:
        raise ValueError("window sizes, block size and topk must be positive")
    if compression_size % compression_stride or block_size % compression_stride:
        raise ValueError("compression and selection sizes must be divisible by compression stride")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if backend not in {"torch", "triton", "auto"}:
        raise ValueError("backend must be torch, triton or auto")
    if rope_position_mode not in {"document", "sequence"}:
        raise ValueError("rope_position_mode must be document or sequence")
    for gate in (compressed_gate, selected_gate):
        if gate is not None and gate.shape != q.shape[:3]:
            raise ValueError("gates must have shape [B,T,Hq]")
    raw_keys = k if compression_k is None else compression_k
    if raw_keys.shape != k.shape:
        raise ValueError("compression_k must have the same shape as k")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    layout = nsa_sequence_layout(q, k, cu_seqlens, query_offset)
    if backend != "torch":
        from pisa.ops.flash_sparse_attn.nsa_cmp_select.nsa_cmp_select_triton import prefill, prefill_supported

        if prefill_supported(q, k, v, block_size, topk, compression_size,
                             compression_stride, force_first, force_recent, layout):
            output, indices = prefill(
                q, k, v, key_weight, value_weight, compression_k=raw_keys,
                rope_fn=rope_fn, rope_position_mode=rope_position_mode,
                cu_seqlens=cu_seqlens, compressed_gate=compressed_gate,
                selected_gate=selected_gate, block_size=block_size, topk=topk,
                compression_size=compression_size, compression_stride=compression_stride,
                scale=scale, force_recent=force_recent, layout=layout,
            )
            return (output, indices) if return_indices else output
    return nsa_cmp_select_torch(
        q, k, v, key_weight, value_weight, raw_keys=raw_keys, layout=layout,
        compressed_gate=compressed_gate, selected_gate=selected_gate,
        block_size=block_size, topk=topk, compression_size=compression_size,
        compression_stride=compression_stride, rope_fn=rope_fn,
        rope_position_mode=rope_position_mode, cu_seqlens=cu_seqlens, scale=scale,
        force_first=force_first, force_recent=force_recent,
        query_chunk_size=query_chunk_size, backend=backend, return_indices=return_indices,
    )

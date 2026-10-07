"""Torch learned compression, probability routing and attention."""

import torch
from torch import Tensor
from torch.nn import functional as F

from pisa.ops.flash_sparse_attn.sparse_attn import sparse_attention
from pisa.utils.validation import packed_queries, query_positions, sequence_slices


def pooling_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32



def pool_sequence(x: Tensor, weight: Tensor, size: int, stride: int) -> Tensor:
    """Pool complete windows of a fixed-length [B,T,H,D] tensor."""
    if x.shape[1] < size:
        return x[:, :0] + weight.sum() * 0
    windows = x.unfold(1, size, stride).to(pooling_dtype(x))
    weights = weight.to(windows.dtype)[None, None, :, None, :]
    return (windows * weights).sum(-1).to(x.dtype)



def weighted_pool(
    x: Tensor,
    weight: Tensor,
    kernel_size: int = 32,
    stride: int = 16,
    cu_seqlens: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Compress complete windows without mixing packed sequences.

    ``weight`` has shape [H, kernel_size] and is not normalized. Packed input
    has shape [1,total_tokens,H,D]; its returned cumulative lengths describe
    the compressed sequences, including sequences with no complete window.
    """
    if x.ndim != 4 or weight.ndim != 2:
        raise ValueError("expected x [B,T,H,D] and weight [H,kernel_size]")
    if kernel_size <= 0 or stride <= 0:
        raise ValueError("kernel_size and stride must be positive")
    if weight.shape != (x.shape[2], kernel_size):
        raise ValueError("compression weight does not match heads/window size")
    if x.device != weight.device:
        raise ValueError("input and compression weight must share a device")
    if cu_seqlens is None:
        return pool_sequence(x, weight, kernel_size, stride), None
    if x.shape[0] != 1:
        raise ValueError("packed input requires batch size 1")
    chunks = [
        pool_sequence(x[:, start:end], weight, kernel_size, stride)
        for start, end in sequence_slices(cu_seqlens, x.shape[1])
    ]
    lengths = [chunk.shape[1] for chunk in chunks]
    compressed_cu = torch.tensor(
        [0, *lengths], device=cu_seqlens.device, dtype=cu_seqlens.dtype
    ).cumsum(0, dtype=cu_seqlens.dtype)
    return torch.cat(chunks, dim=1), compressed_cu



def compression_probabilities(
    q: Tensor,
    compressed_k: Tensor,
    query_positions: Tensor,
    compression_size: int,
    compression_stride: int,
    scale: float,
) -> Tensor:
    """Return [B,T,Hq,M] probabilities, with zero rows before any full window."""
    groups = q.shape[2] // compressed_k.shape[2]
    keys = compressed_k.repeat_interleave(groups, dim=2)
    with torch.autocast(q.device.type, enabled=False):
        logits = (
            torch.einsum("bthd,bmhd->bthm", q.to(pooling_dtype(q)), keys.to(pooling_dtype(q)))
            * scale
        )
    ends = torch.arange(compressed_k.shape[1], device=q.device)
    ends = ends * compression_stride + compression_size - 1
    visible = query_positions[..., None] >= ends
    mask = visible[:, :, None, :]
    any_visible = mask.any(-1, keepdim=True)
    logits = logits.masked_fill(~mask, -torch.inf)
    logits = torch.where(any_visible, logits, torch.zeros_like(logits))
    return logits.softmax(-1) * mask



def compression_block_scores(
    probabilities: Tensor,
    num_kv_heads: int,
    sequence_length: int,
    block_size: int = 64,
    compression_size: int = 32,
    compression_stride: int = 16,
) -> Tensor:
    """Convert compressed probabilities to [B,T,Hkv,selection_blocks].

    A compressed window contributes once per stride-sized interval shared
    with a selection block. For 32/16 compression and 64-token selection,
    interior block overlap weights are 1,2,2,2,1.
    """
    batch, length, query_heads, windows = probabilities.shape
    if query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // num_kv_heads
    grouped = probabilities.reshape(batch, length, num_kv_heads, groups, windows).sum(3)
    blocks = (sequence_length + block_size - 1) // block_size
    starts = torch.arange(windows, device=probabilities.device) * compression_stride
    first_block = starts // block_size
    scores = grouped.new_zeros(batch, length, num_kv_heads, blocks)
    # Each window overlaps only a bounded number of blocks. A dense window-to-
    # block matrix would turn this local reduction into an unnecessary GEMM.
    for displacement in range((compression_size + block_size - 1) // block_size + 1):
        target = first_block + displacement
        overlap = (
            torch.minimum(starts + compression_size, (target + 1) * block_size)
            - torch.maximum(starts, target * block_size)
        ).clamp_min(0)
        weight = overlap.to(probabilities.dtype) / compression_stride
        weight = weight * (target < blocks)
        index = target.clamp_max(blocks - 1).expand_as(grouped)
        scores.scatter_add_(-1, index, grouped * weight)
    return scores



def select_nsa_blocks(scores, positions, topk, block_size, max_blocks, force_first, force_recent):
    """Apply NSA's topk-before-causal-mask rule at the packed batch width.

    Equal scores follow torch.topk on the current device. Future zero scores
    participate in selection, then their indices become -1; do not refill them.
    """
    if topk < 1 or block_size < 1 or force_recent < 0:
        raise ValueError("positive topk/block_size and nonnegative force_recent are required")
    if scores.shape[-1] != max_blocks:
        scores = F.pad(scores, (0, max_blocks - scores.shape[-1]))
    blocks = torch.arange(max_blocks, device=scores.device)
    current = positions[..., None, None] // block_size
    forced = torch.zeros_like(scores, dtype=torch.bool)
    if force_first:
        forced |= blocks == 0
    for recent in range(force_recent):
        forced |= (current >= recent) & (blocks == current - recent)
    if bool(forced.sum(-1).gt(topk).any()):
        raise ValueError("topk is smaller than the number of required distinct blocks")
    scores = scores.masked_fill(forced, torch.inf)
    indices = scores.topk(min(topk, max_blocks), dim=-1).indices.sort(-1).values
    indices = indices.masked_fill(indices > current, -1)
    if indices.shape[-1] < topk:
        indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices



def compressed_attention(
    q: Tensor,
    compressed_k: Tensor,
    compressed_v: Tensor,
    query_positions: Tensor,
    compression_size: int = 32,
    compression_stride: int = 16,
    scale: float | None = None,
    query_chunk_size: int = 128,
) -> Tensor:
    """Causal attention to complete compressed windows using PyTorch SDPA.

    Query chunking bounds mask allocations. SDPA supplies the differentiable
    attention output; routing probabilities are computed separately without
    autograd by ``native_sparse_attention``.
    """
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if not compressed_k.shape[1]:
        zero = q.sum(-1, keepdim=True) * 0 + compressed_k.sum() * 0
        return zero.expand(*q.shape[:-1], compressed_v.shape[-1]) + compressed_v.sum() * 0
    groups = q.shape[2] // compressed_k.shape[2]
    keys = compressed_k.repeat_interleave(groups, dim=2).transpose(1, 2)
    values = compressed_v.repeat_interleave(groups, dim=2).transpose(1, 2)
    ends = torch.arange(compressed_k.shape[1], device=q.device)
    ends = ends * compression_stride + compression_size - 1
    outputs = []
    for start in range(0, q.shape[1], query_chunk_size):
        queries = q[:, start : start + query_chunk_size].transpose(1, 2)
        positions = query_positions[:, start : start + query_chunk_size]
        mask = positions[:, None, :, None] >= ends
        result = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=mask, dropout_p=0.0, scale=scale
        )
        outputs.append(result.transpose(1, 2))
    return torch.cat(outputs, dim=1)



def nsa_sequence_layout(
    q: Tensor, k: Tensor, cu_seqlens: Tensor | None, query_offset: int | Tensor | None
) -> list[tuple[int, int, int, int, int, int]]:
    """Return (batch,q_start,q_end,k_start,k_end,query_offset) per sequence."""
    if cu_seqlens is not None:
        return [(0, *entry) for entry in packed_queries(q, k, cu_seqlens, query_offset)]
    positions = query_positions(q, k, query_offset)
    return [
        (batch, 0, q.shape[1], 0, k.shape[1], int(positions[batch, 0]))
        for batch in range(q.shape[0])
    ]



def nsa_cmp_select_torch(
    q, k, v, key_weight, value_weight, *, raw_keys, layout,
    compressed_gate, selected_gate, block_size, topk, compression_size,
    compression_stride, rope_fn, rope_position_mode, cu_seqlens, scale,
    force_first, force_recent, query_chunk_size, backend, return_indices,
):
    """Portable NSA assembly for validated inputs and sequence layout."""
    output_sequences, index_sequences = [], []
    max_blocks = max((ke - ks + block_size - 1) // block_size for _, _, _, ks, ke, _ in layout)
    for batch, qs, qe, ks, ke, offset in layout:
        queries = q[batch : batch + 1, qs:qe]
        keys = k[batch : batch + 1, ks:ke]
        values = v[batch : batch + 1, ks:ke]
        pooled_k, _ = weighted_pool(
            raw_keys[batch : batch + 1, ks:ke], key_weight, compression_size, compression_stride
        )
        pooled_v, _ = weighted_pool(values, value_weight, compression_size, compression_stride)
        if rope_fn is not None and pooled_k.shape[1]:
            positions = torch.arange(pooled_k.shape[1], device=q.device)[None] * compression_stride
            if rope_position_mode == "sequence":
                positions = positions + ks
            pooled_k = rope_fn(pooled_k, positions)
        positions = torch.arange(qe - qs, device=q.device)[None] + offset
        compressed = compressed_attention(
            queries,
            pooled_k,
            pooled_v,
            positions,
            compression_size,
            compression_stride,
            scale,
            query_chunk_size,
        )
        routed = []
        with torch.no_grad():
            for start in range(0, qe - qs, query_chunk_size):
                chunk_positions = positions[:, start : start + query_chunk_size]
                chunk_q = queries[:, start : start + query_chunk_size]
                probabilities = compression_probabilities(
                    chunk_q, pooled_k, chunk_positions, compression_size, compression_stride, scale
                )
                scores = compression_block_scores(
                    probabilities,
                    k.shape[2],
                    ke - ks,
                    block_size,
                    compression_size,
                    compression_stride,
                )
                routed.append(
                    select_nsa_blocks(
                        scores,
                        chunk_positions,
                        topk,
                        block_size,
                        max_blocks,
                        force_first=force_first,
                        force_recent=force_recent,
                    )
                )
        indices = torch.cat(routed, dim=1)
        selected = sparse_attention(
            queries,
            keys,
            values,
            indices,
            block_size=block_size,
            scale=scale,
            query_offset=offset,
            backend=backend,
        )
        if selected_gate is not None:
            selected = selected * selected_gate[batch : batch + 1, qs:qe, :, None]
        if compressed_gate is not None:
            combined = torch.addcmul(
                selected, compressed, compressed_gate[batch : batch + 1, qs:qe, :, None]
            )
        else:
            combined = selected + compressed
        output_sequences.append(combined)
        index_sequences.append(indices)
    dimension = 0 if cu_seqlens is None else 1
    output = torch.cat(output_sequences, dim=dimension)
    indices = torch.cat(index_sequences, dim=dimension)
    return (output, indices) if return_indices else output


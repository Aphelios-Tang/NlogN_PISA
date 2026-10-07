"""Validated block-selection entry point and backend dispatch."""

import torch

from pisa.ops.compress.mean_pooling import build_pyramid
from pisa.utils.validation import packed_queries, query_positions, validate_qkv
from ..bsa import bsa_fn
from .top_k_pyramid_torch import pyramid_select, routing_geometry


@torch.no_grad()
def select_blocks(
    q,
    k,
    method="pisa",
    block_size=64,
    topk=8,
    cu_seqlens=None,
    query_offset=None,
    force_first=True,
    force_recent=2,
    backend="torch",
    pyramid=None,
    leaf_tile=4,
    routing_schedule="auto",
):
    """Return block ids [B,Tq,Hkv,K], padded by -1.

    Index storage is torch.int32 or torch.int64 depending on the backend path.
    Supported CUDA BSA prefill preserves native int32; general BSA decode and
    Torch routing retain int64. sparse_attention accepts both. Consumers that
    require int64 gather/scatter indices should convert at that boundary.

    Queries/keys must already contain any desired position encoding. Packed ids
    are local to each document. Grouped packed routing uses each document's
    traversal depth; fused packed routing shares the longest document's depth.
    Slot order follows the comparison network even when all
    visible blocks fit. Required blocks count toward the K-slot budget.
    Pyramid routing requires the current block (force_recent >= 1), because
    its summaries can include keys beyond the current query. Its score is
    ignored by the forced-block policy. Automatic scheduling uses fused routing
    for PISA-1. PISA-2 and LSE use grouped four-query leaf tiles for prefill and
    multi-query suffixes, and fused routing for single-query decode.
    Explicit GPU schedules share the same equations
    but can differ in their last score bits.
    """
    validate_qkv(q, k)
    if method not in ("bsa", "pisa1", "pisa2", "pisa"):
        raise ValueError("method must be bsa, pisa1, pisa2 or pisa")
    if backend not in ("torch", "triton", "auto"):
        raise ValueError("backend must be torch, triton or auto")
    if leaf_tile not in (1, 2, 4):
        raise ValueError("leaf_tile must be 1, 2 or 4")
    if routing_schedule not in ("auto", "grouped", "fused"):
        raise ValueError("routing_schedule must be auto, grouped or fused")
    if block_size < 1 or topk < 1 or force_recent < 0:
        raise ValueError("positive block_size/topk and nonnegative force_recent required")
    if method != "bsa" and force_recent == 0:
        raise ValueError("causal pyramid routing requires force_recent >= 1")
    if method == "bsa":
        return bsa_fn(q, k, block_size, topk, cu_seqlens=cu_seqlens,
                      query_offset=query_offset, force_first=force_first,
                      force_recent=force_recent, backend=backend)
    if routing_schedule == "auto":
        # Resolve the full call before slicing packed documents: singleton
        # prefill documents and one-token packed decode slices are distinct.
        grouped = method != "pisa1" and (
            q.shape[1] == k.shape[1] or (cu_seqlens is None and q.shape[1] > 1)
        )
        routing_schedule = "grouped" if grouped else "fused"
        if grouped:
            leaf_tile = 4
    if cu_seqlens is not None:
        segments = list(packed_queries(q, k, cu_seqlens, query_offset))
        blocks = max((ke - ks + block_size - 1) // block_size for _, _, ks, ke, _ in segments)
        geometry = routing_geometry(blocks, topk) if routing_schedule == "fused" else None
        pieces = []
        for qs, qe, ks, ke, offset in segments:
            pieces.append(
                _select_unpacked(
                    q[:, qs:qe], k[:, ks:ke], method, block_size, topk,
                    offset, force_first, force_recent, backend, None,
                    leaf_tile, routing_schedule, geometry,
                )
            )
        return torch.cat(pieces, 1)
    return _select_unpacked(
        q, k, method, block_size, topk, query_offset, force_first, force_recent,
        backend, pyramid, leaf_tile, routing_schedule,
    )



def _select_unpacked(
    q, k, method, block_size, topk, query_offset, force_first, force_recent,
    backend, pyramid, leaf_tile, routing_schedule, geometry=None,
):
    positions = query_positions(q, k, query_offset)
    if backend == "triton" or (backend == "auto" and q.is_cuda):
        if not q.is_cuda:
            raise ValueError("the Triton backend requires a CUDA device")
        with torch.cuda.device(q.device):
            from pisa.ops.flash_sparse_attn.top_k_pyramid.top_k_pyramid_triton import select_cuda

            return select_cuda(
                q,
                k,
                method,
                block_size,
                topk,
                positions,
                force_first,
                force_recent,
                pyramid,
                leaf_tile,
                routing_schedule,
                geometry,
            )
    levels = build_pyramid(k, block_size, backend="torch") if pyramid is None else pyramid
    return pyramid_select(
        q, k, method, block_size, topk, positions, force_first, force_recent, levels, geometry
    )

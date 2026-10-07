"""Validated BSA routing with document-local prefill and decode."""

import torch

from pisa.utils.validation import packed_queries, query_positions, validate_qkv
from .bsa_torch import bsa_select


@torch.no_grad()
def bsa_fn(q, k, block_size=64, topk=8, *, cu_seqlens=None, query_offset=None,
           force_first=True, force_recent=2, backend="auto"):
    validate_qkv(q, k)
    if backend not in ("torch", "triton", "auto"):
        raise ValueError("backend must be torch, triton or auto")
    if block_size < 1 or topk < 1 or force_recent < 0:
        raise ValueError("positive block_size/topk and nonnegative force_recent required")
    if cu_seqlens is not None:
        segments = list(packed_queries(q, k, cu_seqlens, query_offset))
        return torch.cat([
            select_unpacked(q[:, qs:qe], k[:, ks:ke], block_size, topk,
                            offset, force_first, force_recent, backend)
            for qs, qe, ks, ke, offset in segments
        ], dim=1)
    return select_unpacked(q, k, block_size, topk, query_offset,
                           force_first, force_recent, backend)


def select_unpacked(q, k, block_size, topk, query_offset,
                    force_first, force_recent, backend):
    if q.is_cuda and backend in ("triton", "auto"):
        from .bsa_fla import bsa_prefill, bsa_prefill_supported

        if bsa_prefill_supported(q, k, block_size, topk, query_offset, force_first, force_recent):
            return bsa_prefill(q, k, block_size, topk)
    positions = query_positions(q, k, query_offset)
    if backend == "triton" or (backend == "auto" and q.is_cuda):
        if not q.is_cuda:
            raise ValueError("the Triton backend requires a CUDA device")
        with torch.cuda.device(q.device):
            if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("Triton routing supports FP16, BF16 and FP32")
            if topk & (topk - 1) or block_size & (block_size - 1) or not 8 <= topk <= 64 or block_size < 16:
                raise ValueError(
                    "GPU routing requires power-of-two block_size >= 16 and topk in {8,16,32,64}"
                )
            blocks = (k.shape[1] + block_size - 1) // block_size
            if topk < min(blocks, force_recent + int(force_first)):
                raise ValueError("topk cannot hold all required blocks")
            from .bsa_triton import select_bsa

            return select_bsa(q, k, block_size, topk, positions, force_first, force_recent)
    return bsa_select(q, k, block_size, topk, positions, force_first, force_recent)

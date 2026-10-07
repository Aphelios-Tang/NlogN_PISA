"""Public FLA pooling and BSA prefill selection."""

import torch


def bsa_prefill_supported(q, k, block_size, topk, query_offset, force_first, force_recent):
    """Metadata-only dispatch after the public operator validates Q/K.

    This topk path assumes full prefill and always forces first/current/previous
    blocks. Keep other layouts and policies on the general position-aware path.
    """
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.shape[:2] == k.shape[:2]
        and q.shape[-1] in (64, 128)
        and q.shape[2] == 16 * k.shape[2]
        and type(block_size) is int and block_size == 64
        and type(topk) is int and topk in (8, 32)
        and force_first is True
        and type(force_recent) is int and force_recent == 2
        and (query_offset is None or (type(query_offset) is int and query_offset == 0))
    )



def bsa_prefill(q, k, block_size, topk):
    """Use FLA mean pooling and ranked topk for a validated supported prefill."""
    # FLA initializes device-dependent helpers at import time. First import
    # needs the same input-device scope as the calls, including multi-GPU use.
    with torch.cuda.device(q.device):
        try:
            from fla.ops.nsa.parallel import parallel_nsa_topk
            from fla.ops.utils.pooling import mean_pooling
        except ImportError as error:
            raise ImportError("install pisa-attention[cuda] to use CUDA BSA routing") from error
        summaries = mean_pooling(k.contiguous(), block_size)
        return parallel_nsa_topk(
            q.contiguous(), summaries, TK=k.shape[1], lse=None, block_counts=topk,
            block_size=block_size, scale=q.shape[-1] ** -0.5, cu_seqlens=None,
        )

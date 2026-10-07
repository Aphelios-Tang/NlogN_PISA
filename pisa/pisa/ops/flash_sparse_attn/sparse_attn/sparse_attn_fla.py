"""Public FLA attention over selected blocks."""

import torch


def prepare_prefill_indices(indices, positions, block_size):
    """Give empty rows a finite softmax; callers must mask their outputs."""
    current = positions[..., None] // block_size
    visible = (indices >= 0) & (indices <= current[..., None])
    has_tokens = visible.any(-1)
    if indices.shape[-1] == 0:
        safe = torch.full(
            (*indices.shape[:-1], 1), -1, dtype=torch.int32, device=indices.device
        )
    else:
        safe = torch.where(visible, indices, -1).to(torch.int32)
    safe[..., 0] = torch.where(has_tokens, safe[..., 0], current)
    return safe.contiguous(), has_tokens



def selected_attention(q, k, v, indices, block_size, scale, positions, causal=True, cu_seqlens=None):
    if not q.is_cuda:
        raise ValueError("the FLA/Triton backend requires a CUDA device")
    if not causal:
        raise ValueError("the CUDA backend currently implements causal attention only")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("CUDA attention supports FP16, BF16 and FP32")
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    # sparse_attention validates offsets first; equal lengths imply zero local
    # offset, including packed prefill. Keep dispatch free of device reads.
    aligned = q.shape[1] == k.shape[1]
    if not aligned:
        if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
            raise ValueError(
                "CUDA cached decoding is inference-only; use backend='torch' for its gradients"
            )
        from pisa.ops.flash_sparse_attn.decode.decode_triton import decode_attention

        return decode_attention(q, k, v, indices, positions, block_size, scale)
    group = q.shape[2] // k.shape[2]
    if group < 16 or group & (group - 1):
        raise ValueError("the FLA training backend requires a power-of-two GQA group size >= 16")
    if k.shape[-1] > 128:
        raise ValueError("the FLA training backend supports key dimensions up to 128")
    with torch.cuda.device(q.device):
        try:
            from fla.ops.nsa.parallel import ParallelNSAFunction
            if cu_seqlens is not None:
                from fla.ops.utils import prepare_position_ids
        except ImportError as error:
            raise ImportError("install pisa-attention[cuda] to use CUDA selected attention") from error
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device=q.device).contiguous()
            # Block IDs and the fallback block are local to each document.
            # Reuse FLA's cached positions, also consumed by its varlen kernels.
            positions = prepare_position_ids(cu_seqlens)[None]
        safe_indices, has_tokens = prepare_prefill_indices(indices, positions, block_size)
        output = ParallelNSAFunction.apply(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            safe_indices,
            safe_indices.shape[-1],
            block_size,
            scale,
            cu_seqlens,
        )
    # FLA saves its output for backward, so masking a NaN after the call is not enough.
    # The dummy block makes that saved output finite; this mask zeros its gradient.
    return torch.where(has_tokens.repeat_interleave(group, 2)[..., None], output, 0)


"""Next-token loss with explicit document boundaries and bounded logits memory."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def next_token_targets(labels, cu_seqlens=None, *, mask_document_targets=True):
    """Shift labels globally, optionally removing document-transition targets."""
    if not isinstance(mask_document_targets, bool):
        raise ValueError("mask_document_targets must be a boolean")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    targets = torch.full_like(labels, -100)
    targets[:, :-1] = labels[:, 1:]
    if cu_seqlens is not None:
        if labels.shape[0] != 1:
            raise ValueError("Packed inputs require batch size 1")
        boundaries = cu_seqlens.to(device=labels.device, dtype=torch.long)
        if boundaries.ndim != 1 or boundaries.numel() < 2:
            raise ValueError("cu_seqlens must contain sequence boundaries")
        if boundaries[0] != 0 or boundaries[-1] != labels.shape[1]:
            raise ValueError("cu_seqlens must span the input")
        if torch.any(boundaries[1:] <= boundaries[:-1]):
            raise ValueError("Packed documents must have positive lengths")
        if mask_document_targets:
            targets[0, boundaries[1:] - 1] = -100
    return targets


def causal_lm_loss(
    hidden_states, weight, labels, *, cu_seqlens=None, chunk_size=1024, backend="torch",
    mask_document_targets=True,
):
    if backend not in {"torch", "fla", "pisa"}:
        raise ValueError(f"Unknown loss backend: {backend}")
    targets = next_token_targets(
        labels, cu_seqlens, mask_document_targets=mask_document_targets,
    ).reshape(-1)
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    if backend == "pisa":
        from pisa import linear_cross_entropy

        return linear_cross_entropy(hidden, weight, targets)
    valid_tokens = (targets != -100).sum()
    if valid_tokens == 0:
        return hidden.sum() * 0 + weight.sum() * 0
    if backend == "fla":
        from fla.modules import FusedLinearCrossEntropyLoss

        return FusedLinearCrossEntropyLoss()(x=hidden, target=targets, weight=weight, bias=None)
    def project_and_sum(x, w, y):
        logits = F.linear(x, w).float()
        return F.cross_entropy(logits, y, ignore_index=-100, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[0], chunk_size):
        x = hidden[start : start + chunk_size]
        y = targets[start : start + chunk_size]
        if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
            # Recompute the projection in backward instead of retaining every
            # chunk's vocabulary-sized logits through the entire forward pass.
            value = checkpoint(project_and_sum, x, weight, y, use_reentrant=False)
        else:
            value = project_and_sum(x, weight, y)
        total = total + value
    return total / valid_tokens


class Loss(torch.nn.Module):
    """Public LLaMA loss boundary using the single PISA label-shift contract."""
    def forward(self, weight, hidden, labels=None, bias=None, num_logits_to_keep=0,
                fuse_linear_and_cross_entropy=True, *, ce_type="pisa", cu_seqlens=None,
                mask_document_targets=True, chunk_size=1024, return_logits=None):
        loss = None if labels is None else causal_lm_loss(
            hidden, weight, labels, cu_seqlens=cu_seqlens,
            mask_document_targets=mask_document_targets, chunk_size=chunk_size,
            backend=ce_type)
        if return_logits is None:
            return_logits = labels is None
        logits = F.linear(hidden[:, -num_logits_to_keep:] if num_logits_to_keep else hidden,
                          weight, bias) if return_logits else None
        return logits, loss

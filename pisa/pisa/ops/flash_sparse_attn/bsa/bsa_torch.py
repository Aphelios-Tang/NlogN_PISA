"""Probability-based block routing with streaming comparison-network selection."""

import torch
import torch.nn.functional as F

from pisa.ops.compress.mean_pooling import mean_pool


def bitonic_merge(scores, ids, stage, descending):
    """Merge runs through XOR partners, carrying ids without comparing them.

    ``descending=None`` alternates run direction. Equality exchanges in a
    descending run and stays in place in an ascending run, matching the
    score-only network used by the selected-attention dependency.
    """
    lanes = torch.arange(scores.shape[-1], device=scores.device)
    direction = (lanes & (1 << stage)) != 0 if descending is None else descending
    for exponent in range(stage - 1, -1, -1):
        distance = 1 << exponent
        partner = lanes ^ distance
        lower = lanes & ~distance
        upper = lower | distance
        exchange = (scores[..., lower] > scores[..., upper]) != direction
        scores = torch.where(exchange, scores[..., partner], scores)
        ids = torch.where(exchange, ids[..., partner], ids)
    return scores, ids


def streaming_topk(scores, ids, topk, tile_size):
    """Keep the best half of each tile and merge it with the retained half.

    The returned ids are in network rank order. Padding ids are -1; their
    scores must not outrank valid candidates. All-invalid tiles are ignored.
    """
    if topk < 1 or tile_size < 2 * topk or tile_size & (tile_size - 1):
        raise ValueError("tile_size must be a power of two and at least 2 * topk")
    if scores.shape != ids.shape or scores.shape[-1] < 1:
        raise ValueError("scores and ids must have the same nonempty shape")
    stages = tile_size.bit_length() - 1
    retained_scores = scores.new_full((*scores.shape[:-1], tile_size), -1)
    retained_ids = ids.new_zeros((*ids.shape[:-1], tile_size))
    for start in range(0, scores.shape[-1], tile_size):
        values = scores[..., start : start + tile_size]
        indices = ids[..., start : start + tile_size]
        active = (indices >= 0).any(-1, keepdim=True)
        padding = tile_size - values.shape[-1]
        values = F.pad(values, (0, padding), value=0)
        indices = F.pad(indices, (0, padding), value=-1)
        for stage in range(1, stages):
            values, indices = bitonic_merge(values, indices, stage, None)
        if start:
            values, indices = bitonic_merge(values, indices, stages, False)
            half = tile_size // 2
            values = torch.cat((retained_scores[..., :half], values[..., half:]), -1)
            indices = torch.cat((retained_ids[..., :half], indices[..., half:]), -1)
        values, indices = bitonic_merge(values, indices, stages, True)
        retained_scores = torch.where(active, values, retained_scores)
        retained_ids = torch.where(active, indices, retained_ids)
    return retained_ids[..., :topk]


@torch.no_grad()
def bsa_select(q, k, block_size, topk, positions, force_first, force_recent):
    """Route by summed per-head probabilities over complete mean-pooled blocks.

    Paper shapes use the block size as the comparison tile, with C >= 2K and
    power-of-two C. Tiny portable tests also support C < 2K by enlarging only
    the comparison tile; pooling and causal block boundaries remain unchanged.
    """
    summaries = mean_pool(k, block_size, backend="torch")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    batch, length, heads_q, dimension = q.shape
    heads = k.shape[2]
    groups = heads_q // heads
    width = 1 << (max(block_size, 2 * topk) - 1).bit_length()
    blocks = torch.arange(summaries.shape[1], device=q.device)
    scale = dimension**-0.5
    outputs = []
    for start in range(0, length, 128):
        query = q[:, start : start + 128]
        # Apply the FP32 scale before rounding scaled Q to its stored dtype.
        query = (query.to(dtype) * scale).to(q.dtype).to(dtype)
        query = query.reshape(batch, -1, heads, groups, dimension)
        with torch.autocast(q.device.type, enabled=False):
            logits = torch.einsum("bthgd,bmhd->bthgm", query, summaries.to(dtype))
        position = positions[:, start : start + 128, None, None]
        current = position // block_size
        complete = (position + 1) // block_size
        visible = blocks < complete[..., None]
        normalizer = logits.masked_fill(~visible, -torch.inf).logsumexp(-1, keepdim=True)
        normalizer = torch.where(complete[..., None] > 0, normalizer, 0)
        forced = torch.zeros_like(current + blocks, dtype=torch.bool)
        if force_first:
            forced |= blocks == 0
        for recent in range(force_recent):
            forced |= (current >= recent) & (blocks == current - recent)
        valid = blocks <= current
        if bool((forced & valid).sum(-1).gt(topk).any()):
            raise ValueError("topk is smaller than the number of required distinct blocks")
        candidate_logits = logits.masked_fill(~(blocks < current)[..., None, :], -torch.inf)
        probabilities = (candidate_logits - normalizer).exp()
        probabilities = torch.where(forced[..., None, :], 1, probabilities)
        scores = probabilities.sum(-2)
        ids = blocks.expand_as(scores).masked_fill(~valid, -1)
        outputs.append(streaming_topk(scores, ids, topk, width))
    return torch.cat(outputs, 1)

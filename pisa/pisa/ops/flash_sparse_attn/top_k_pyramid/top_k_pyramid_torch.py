"""Torch reference for pyramid scoring and comparison-network traversal."""

import torch


def force_mask(ids, positions, block_size, force_first=True, force_recent=2, level=0):
    leaf = positions // block_size
    result = torch.zeros_like(ids, dtype=torch.bool)
    if force_first:
        result |= ids == 0
    for recent in range(force_recent):
        target = leaf - recent
        result |= (target[..., None] >= 0) & (ids == (target[..., None] >> level))
    return result & (ids >= 0)



def choose_candidates(ids, scores, valid, forced, topk):
    if bool((forced & valid).sum(-1).gt(topk).any()):
        raise ValueError("topk is smaller than the number of required distinct blocks")
    # Input ids are ordered. Stable sorting gives the lower index priority on ties.
    ranks = scores.masked_fill(~valid, -torch.inf).masked_fill(forced & valid, torch.inf)
    order = torch.argsort(ranks, dim=-1, descending=True, stable=True)[..., :topk]
    selected = ids.gather(-1, order).masked_fill(~valid.gather(-1, order), -1)
    selected = selected.masked_fill(selected < 0, torch.iinfo(torch.long).max).sort(-1).values
    selected = selected.masked_fill(selected == torch.iinfo(torch.long).max, -1)
    if selected.shape[-1] < topk:
        selected = torch.nn.functional.pad(selected, (0, topk - selected.shape[-1]), value=-1)
    return selected



def select_from_scores(scores, topk, query_positions, block_size, force_first=True, force_recent=2):
    """Select document-relative blocks from [B,T,Hkv,M] scores."""
    if topk < 1 or block_size < 1 or force_recent < 0:
        raise ValueError("positive topk/block_size and nonnegative force_recent are required")
    ids = torch.arange(scores.shape[-1], device=scores.device).expand_as(scores)
    positions = torch.as_tensor(query_positions, device=scores.device)
    if positions.ndim == 1:
        positions = positions[None]
    positions = positions[..., None].expand(scores.shape[:-1])
    valid = ids * block_size <= positions[..., None]
    forced = force_mask(ids, positions, block_size, force_first, force_recent)
    return choose_candidates(ids, scores, valid, forced, topk)



def block_scores(query, source, ids, support, method, scale):
    """Scores for one query per batch and candidate support of size C or two."""
    batch, heads, _ = ids.shape
    groups = query.shape[1] // heads
    samples = ids[..., None] * support + torch.arange(support, device=query.device)
    valid = (samples >= 0) & (samples < source.shape[1]) & (ids[..., None] >= 0)
    batch_ids = torch.arange(batch, device=query.device)[:, None, None, None]
    head_ids = torch.arange(heads, device=query.device)[None, :, None, None]
    values = source[batch_ids, samples.clamp(0, source.shape[1] - 1), head_ids]
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    grouped = query.to(dtype).reshape(batch, heads, groups, query.shape[-1])
    # Training autocast must not lower the reference score accumulation dtype.
    with torch.autocast(query.device.type, enabled=False):
        products = torch.einsum("bhgd,bhwsd->bhgws", grouped, values.to(dtype))
    mask = valid[:, :, None]
    if method == "pisa":
        score = (products * scale).masked_fill(~mask, -torch.inf).logsumexp(-1)
    else:
        count = mask.sum(-1).clamp_min(1)
        mean = products.masked_fill(~mask, 0).sum(-1) / count
        score = scale * mean
        if method == "pisa2":
            if support == 2:
                z = products * scale
                z = z.masked_fill(~mask, 0)
                total = z.sum(-1)
                difference = z[..., 0] - z[..., 1]
                score = torch.where(count == 2, 0.5 * total + 0.125 * difference.square(), total)
            else:
                second = products.square().masked_fill(~mask, 0).sum(-1) / count
                variance = (second - mean.square()).clamp_min(0)
                score = scale * mean + 0.5 * scale * scale * variance
    return score.sum(2)



def network_topk(scores, ids, topk):
    """Descending bitonic network, including its compare/exchange tie order.

    Each pass pairs lanes at one XOR distance. Equal values exchange in a
    descending group and stay in an ascending group; ids are payload only.
    """
    width = 1 << (scores.shape[-1] - 1).bit_length()
    padding = width - scores.shape[-1]
    scores = torch.nn.functional.pad(scores, (0, padding), value=-1e9)
    ids = torch.nn.functional.pad(ids, (0, padding), value=-1)
    lane = torch.arange(width, device=scores.device)
    for exponent in range(1, width.bit_length()):
        group = 1 << exponent
        descending = (
            torch.ones_like(lane, dtype=torch.bool) if group == width else (lane & group) != 0
        )
        for shift in range(exponent - 1, -1, -1):
            distance = 1 << shift
            partner = lane ^ distance
            peer_score, peer_id = scores[..., partner], ids[..., partner]
            left = (lane & distance) == 0
            left_score = torch.where(left, scores, peer_score)
            right_score = torch.where(left, peer_score, scores)
            exchange = (left_score > right_score) != descending
            scores = torch.where(exchange, peer_score, scores)
            ids = torch.where(exchange, peer_id, ids)
    return ids[..., :topk]



def routing_masks(ids, positions, block_size, topk, force_first, force_recent, level=0):
    valid = (ids >= 0) & (ids * (block_size << level) <= positions[..., None])
    forced = force_mask(ids, positions, block_size, force_first, force_recent, level) & valid
    if bool(forced.sum(-1).gt(topk).any()):
        raise ValueError("topk is smaller than the number of required distinct blocks")
    identity = valid.sum(-1) <= topk
    return valid, forced, valid & ~forced & ~identity[..., None]



def routing_geometry(blocks, topk):
    """Common traversal depth and first network width for one packed batch."""
    top = 0
    while blocks > topk:
        parent = (blocks + 1) // 2
        if parent < topk or parent == blocks:
            break
        blocks = parent
        top += 1
    return top, max(16, 1 << (max(blocks, topk) - 1).bit_length())



def pyramid_select(
    q, k, method, block_size, topk, positions, force_first, force_recent, levels, geometry=None
):
    batch, length, _, _ = q.shape
    heads = k.shape[2]
    if hasattr(levels, "levels"):
        levels = levels.levels
    top, first_width = geometry or routing_geometry(levels[0].shape[1], topk)
    # Fused packed traversal uses the longest document's depth. A one-node local
    # pyramid remains that same node at each additional common level.
    levels = list(levels) + [levels[-1]] * max(0, top + 1 - len(levels))
    result = []
    for token in range(length):
        position = positions[:, token, None].expand(batch, heads)
        ids = torch.arange(first_width, device=q.device).expand(batch, heads, -1)
        for level in range(top, -1, -1):
            total = levels[level].shape[1]
            ids = ids.masked_fill(ids >= total, -1)
            valid, forced, scoreable = routing_masks(
                ids, position, block_size, topk, force_first, force_recent, level
            )
            if method == "pisa1":
                scoreable = valid & ~forced
                source, support, scale = levels[level], 1, 1.0
            else:
                source, support = (k, block_size) if level == 0 else (levels[level - 1], 2)
                scale = k.shape[-1] ** -0.5
            scores = torch.zeros_like(ids, dtype=torch.float32)
            if bool(scoreable.any()):
                scores = block_scores(
                    q[:, token], source, ids.masked_fill(~scoreable, -1), support, method, scale
                )
            scores = (
                scores.masked_fill(~scoreable, 0).masked_fill(~valid, -1e9).masked_fill(forced, 1e9)
            )
            ids = network_topk(scores, ids, topk)
            if level:
                children = ids[..., None] * 2 + torch.arange(2, device=q.device)
                ids = children.masked_fill(ids[..., None] < 0, -1).flatten(-2)
        result.append(ids.masked_fill((ids < 0) | (ids * block_size > position[..., None]), -1))
    return torch.stack(result, dim=1)


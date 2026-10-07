"""Bounded pyramid traversal and leaf scoring with a fixed comparison network."""

from itertools import accumulate

import torch
import triton
import triton.language as tl

from fla.ops.nsa.utils import _bitonic_merge

from pisa.ops.compress.mean_pooling.mean_pooling import build_pyramid


@triton.jit
def required(
    ids, position, C: tl.constexpr, LEVEL: tl.constexpr, FIRST: tl.constexpr, RECENT: tl.constexpr
):
    forced = tl.full(ids.shape, False, tl.int1)
    if FIRST:
        forced = ids == 0
    for r in tl.static_range(RECENT):
        leaf = position // C - r
        forced |= (leaf >= 0) & (ids == (leaf >> LEVEL))
    return forced


@triton.jit
def swap_lanes(values, WIDTH: tl.constexpr, DISTANCE: tl.constexpr):
    pairs = tl.reshape(values, (WIDTH // (2 * DISTANCE), 2, DISTANCE))
    left, right = tl.split(tl.permute(pairs, (0, 2, 1)))
    return tl.reshape(tl.permute(tl.join(right, left), (0, 2, 1)), (WIDTH,))


@triton.jit
def best_ids(scores, ids, K: tl.constexpr, WIDTH: tl.constexpr):
    """The id travels with its score; equal scores follow the network direction."""
    lane = tl.arange(0, WIDTH)
    for exponent in tl.static_range(1, tl.standard._log2(WIDTH) + 1):
        descending = (lane & (1 << exponent)) != 0
        if (1 << exponent) == WIDTH:
            descending = tl.full((WIDTH,), True, tl.int1)
        for shift in tl.static_range(exponent - 1, -1, -1):
            peer_score = swap_lanes(scores, WIDTH, 1 << shift)
            peer_id = swap_lanes(ids, WIDTH, 1 << shift)
            left = (lane & (1 << shift)) == 0
            left_score = tl.where(left, scores, peer_score)
            right_score = tl.where(left, peer_score, scores)
            exchange = (left_score > right_score) != descending
            scores = tl.where(exchange, peer_score, scores)
            ids = tl.where(exchange, peer_id, ids)
    rows = tl.reshape(ids, (WIDTH // K, K))
    return tl.sum(tl.where(tl.arange(0, WIDTH // K)[:, None] == 0, rows, 0), 0)


@triton.jit
def integer_key_best_ids(scores, ids, K: tl.constexpr, WIDTH: tl.constexpr):
    """Preserve float comparison/tie order while exchanging integer payloads."""
    tl.static_assert(scores.dtype == tl.float32)
    if tl.sum((scores != scores).to(tl.int32), 0) > 0:
        # Unordered NaNs must retain the original comparison network.
        selected = best_ids(scores, ids, K, WIDTH)
    else:
        bits = scores.to(tl.int32, bitcast=True)
        keys = bits ^ tl.where(bits < 0, 0x7fffffff, 0)
        # Both zero signs compare equal in the original float network.
        keys = tl.where((bits & 0x7fffffff) == 0, 0, keys)
        dimensions: tl.constexpr = tl.standard._log2(WIDTH)
        for stage in tl.static_range(1, dimensions + 1):
            keys, ids = _bitonic_merge(
                keys, ids, stage, 2 if stage < dimensions else True, dimensions
            )
        rows = tl.reshape(ids, (WIDTH // K, K))
        selected = tl.sum(tl.where(tl.arange(0, WIDTH // K)[:, None] == 0, rows, 0), 0)
    return selected


@triton.jit
def score_support(
    products, mask, MODE: tl.constexpr, SCALE: tl.constexpr, BINARY: tl.constexpr = False
):
    count = tl.maximum(tl.sum(mask.to(tl.float32), -1), 1)
    if MODE == 2:
        logits = tl.where(mask, products * SCALE, -float("inf"))
        maximum = tl.max(logits, -1)
        maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        total = tl.sum(
            tl.where(mask, tl.exp2((logits - tl.expand_dims(maximum, -1)) * 1.4426950408889634), 0.0),
            -1,
        )
        return maximum + tl.log2(tl.maximum(total, 1.0e-30)) / 1.4426950408889634
    if BINARY:
        logits = tl.where(mask, products * SCALE, 0.0)
        total = tl.sum(logits, -1)
        difference = tl.sum(logits * (1 - 2 * tl.arange(0, 2)), -1)
        return tl.where(count == 2, 0.5 * total + 0.125 * difference * difference, total)
    products = tl.where(mask, products, 0.0)
    mean = tl.sum(products, -1) / count
    if MODE == 1:
        second = tl.sum(products * products, -1) / count
        variance = tl.maximum(second - mean * mean, 0.0)
        return SCALE * mean + 0.5 * SCALE * SCALE * variance
    return SCALE * mean


@triton.jit
def score_masks(
    ids,
    position,
    count,
    C: tl.constexpr,
    LEVEL: tl.constexpr,
    K: tl.constexpr,
    FIRST: tl.constexpr,
    RECENT: tl.constexpr,
    MODE: tl.constexpr,
):
    valid = (ids >= 0) & (ids < count) & (ids * (C << LEVEL) <= position)
    forced = required(ids, position, C, LEVEL, FIRST, RECENT) & valid
    scoreable = valid & ~forced
    if MODE != 0:
        scoreable &= tl.sum(valid.to(tl.int32), 0) > K
    return valid, forced, scoreable


@triton.jit(do_not_specialize=["N"])
def coarse_kernel(
    Q,
    SUMMARY,
    OFFSETS,
    LENGTHS,
    POS,
    OUT,
    RAW_KEY,
    N,
    SUMMARY_BATCH,
    KEY_BATCH,
    KEY_TIME,
    T: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    TOP: tl.constexpr,
    FIRST_WIDTH: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    BD: tl.constexpr,
    W: tl.constexpr,
    MODE: tl.constexpr,
    FIRST: tl.constexpr,
    RECENT: tl.constexpr,
    FUSE_LEAF: tl.constexpr,
    INTEGER_SORT: tl.constexpr = False,
):
    token, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HK, bh % HK
    position = tl.load(POS + batch * T + token)
    g, d = tl.arange(0, BG), tl.arange(0, BD)
    query = tl.load(
        Q + ((batch * T + token) * HQ + head * G + g[:, None]) * D + d[None, :],
        (g[:, None] < G) & (d[None, :] < D),
        other=0,
    )
    selected = tl.full((K,), 0, tl.int32)
    for level in tl.static_range(TOP, 0, -1):
        if level == TOP:
            ids = tl.arange(0, FIRST_WIDTH)
        else:
            ids = tl.reshape(selected[:, None] * 2 + tl.arange(0, 2)[None, :], (W,))
        count = tl.load(LENGTHS + level)
        valid, forced, scoreable = score_masks(
            ids, position, count, C, level, K, FIRST, RECENT, MODE
        )
        scores = tl.full(ids.shape, 0.0, tl.float32)
        if tl.sum(scoreable.to(tl.int32), 0) > 0:
            if MODE == 0:
                base = tl.load(OFFSETS + level)
                keys = tl.load(
                    SUMMARY
                    + batch * SUMMARY_BATCH
                    + ((base + ids[None, :]) * HK + head) * D
                    + d[:, None],
                    scoreable[None, :] & (d[:, None] < D),
                    other=0,
                )
                products = tl.dot(query.to(tl.float32), keys.to(tl.float32))
                scores = tl.sum(tl.where(g[:, None] < G, products, 0.0), 0)
            else:
                child_count = tl.load(LENGTHS + level - 1)
                base = tl.load(OFFSETS + level - 1)
                children = tl.reshape(
                    ids[:, None] * 2 + tl.arange(0, 2)[None, :], (2 * ids.shape[0],)
                )
                child_mask = tl.reshape(
                    scoreable[:, None]
                    & (ids[:, None] * 2 + tl.arange(0, 2)[None, :] < child_count),
                    (2 * ids.shape[0],),
                )
                keys = tl.load(
                    SUMMARY
                    + batch * SUMMARY_BATCH
                    + ((base + children[None, :]) * HK + head) * D
                    + d[:, None],
                    child_mask[None, :] & (d[:, None] < D),
                    other=0,
                )
                products = tl.reshape(tl.dot(query, keys), (BG, ids.shape[0], 2))
                mask = tl.broadcast_to(tl.reshape(child_mask, (1, ids.shape[0], 2)), products.shape)
                per_head = score_support(products, mask, MODE, D**-0.5, True)
                # Reduce heads along the last axis. Older Triton releases can
                # mix adjacent candidates when lowering two consecutive
                # reductions along different axes of this dot-product layout.
                scores = tl.sum(tl.where(g[None, :] < G, tl.trans(per_head), 0.0), 1)
        scores = tl.where(scoreable, scores, 0.0)
        scores = tl.where(valid, tl.where(forced, 1.0e9, scores), -1.0e9)
        if INTEGER_SORT:
            selected = integer_key_best_ids(scores, ids, K, ids.shape[0])
        else:
            selected = best_ids(scores, ids, K, ids.shape[0])
    if TOP == 0:
        ids = tl.arange(0, W)
    else:
        ids = tl.reshape(selected[:, None] * 2 + tl.arange(0, 2)[None, :], (W,))
    leaf_count = tl.load(LENGTHS)
    valid, forced, scoreable = score_masks(ids, position, leaf_count, C, 0, K, FIRST, RECENT, MODE)
    if FUSE_LEAF:
        lane, sample = tl.arange(0, W), tl.arange(0, C)
        scores = tl.full((W,), 0.0, tl.float32)
        if MODE == 0:
            if tl.sum(scoreable.to(tl.int32), 0) > 0:
                base = tl.load(OFFSETS)
                means = tl.load(
                    SUMMARY
                    + batch * SUMMARY_BATCH
                    + ((base + ids[None, :]) * HK + head) * D
                    + d[:, None],
                    scoreable[None, :] & (d[:, None] < D),
                    other=0,
                ).to(tl.float32)
                products = tl.dot(query.to(tl.float32), means)
                scores = tl.sum(tl.where(g[:, None] < G, products, 0.0), 0)
        else:
            for slot in tl.static_range(W):
                block = tl.sum(tl.where(lane == slot, ids, 0), 0)
                active = tl.sum(tl.where(lane == slot, scoreable.to(tl.int32), 0), 0) > 0
                value = tl.full((), 0.0, tl.float32)
                if active:
                    key_position = block * C + sample
                    mask = key_position < N
                    keys = tl.load(
                        RAW_KEY
                        + batch * KEY_BATCH
                        + key_position[None, :] * KEY_TIME
                        + head * D
                        + d[:, None],
                        mask[None, :] & (d[:, None] < D),
                        other=0,
                    )
                    products = tl.dot(query, keys)
                    head_scores = score_support(
                        products, tl.broadcast_to(mask[None, :], (BG, C)), MODE, D**-0.5
                    )
                    value = tl.sum(tl.where(g < G, head_scores, 0.0), 0)
                scores = tl.where(lane == slot, value, scores)
        scores = tl.where(valid, tl.where(forced, 1.0e9, scores), -1.0e9)
        if INTEGER_SORT:
            selected = integer_key_best_ids(scores, ids, K, W)
        else:
            selected = best_ids(scores, ids, K, W)
        selected = tl.where(
            (selected >= 0) & (selected < leaf_count) & (selected * C <= position), selected, -1
        )
        tl.store(OUT + ((batch * T + token) * HK + head) * K + tl.arange(0, K), selected)
    else:
        tl.store(OUT + ((batch * T + token) * HK + head) * W + tl.arange(0, W), ids)


@triton.jit
def final_kernel(
    IDS,
    SCORES,
    POS,
    OUT,
    T: tl.constexpr,
    HK: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    W: tl.constexpr,
    FIRST: tl.constexpr,
    RECENT: tl.constexpr,
    MODE: tl.constexpr,
    INTEGER_SORT: tl.constexpr = False,
):
    item = tl.program_id(0)
    lane = tl.arange(0, W)
    ids = tl.load(IDS + item * W + lane)
    position = tl.load(POS + item // HK)
    valid, forced, scoreable = score_masks(ids, position, M, C, 0, K, FIRST, RECENT, MODE)
    scores = tl.load(SCORES + item * W + lane, scoreable, other=0)
    scores = tl.where(valid, tl.where(forced, 1.0e9, scores), -1.0e9)
    if INTEGER_SORT:
        selected = integer_key_best_ids(scores, ids, K, W)
    else:
        selected = best_ids(scores, ids, K, W)
    selected = tl.where((selected >= 0) & (selected < M) & (selected * C <= position), selected, -1)
    tl.store(OUT + item * K + tl.arange(0, K), selected)


def select_cuda(
    q,
    k,
    method,
    block_size,
    topk,
    positions,
    force_first,
    force_recent,
    pyramid=None,
    leaf_tile=4,
    routing_schedule="grouped",
    geometry=None,
):
    if not q.is_cuda:
        raise ValueError("the Triton backend requires an NVIDIA CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton routing supports FP16, BF16 and FP32")
    if topk & (topk - 1) or block_size & (block_size - 1) or not 8 <= topk <= 64 or block_size < 16:
        raise ValueError(
            "GPU routing requires power-of-two block_size >= 16 and topk in {8,16,32,64}"
        )
    blocks = triton.cdiv(k.shape[1], block_size)
    if topk < min(blocks, force_recent + int(force_first)):
        raise ValueError("topk cannot hold all required blocks")
    if method == "bsa":
        from pisa.ops.flash_sparse_attn.bsa.bsa_triton import select_bsa

        return select_bsa(q, k, block_size, topk, positions, force_first, force_recent)
    b, t, hq, d = q.shape
    hk = k.shape[2]
    from pisa.ops.flash_sparse_attn.top_k_pyramid.top_k_pyramid_torch import routing_geometry

    top, first_width = geometry or routing_geometry(blocks, topk)
    q = q.contiguous()
    if k.stride(-1) != 1 or k.stride(-2) != d:
        k = k.contiguous()
    if hasattr(pyramid, "storage"):
        summary, offsets, lengths = pyramid.storage, pyramid.offsets, pyramid.lengths
        sizes = pyramid.level_sizes
    else:
        levels = build_pyramid(k, block_size, max_level=top) if pyramid is None else pyramid
        levels = list(levels) + [levels[-1]] * max(0, top + 1 - len(levels))
        summary = torch.cat(levels, dim=1).contiguous()
        sizes = tuple(x.shape[1] for x in levels)
        starts = tuple(accumulate(sizes, initial=0))[:-1]
        lengths, offsets = torch.tensor(
            (sizes, starts), device=k.device, dtype=torch.int32
        ).unbind(0)
    group = hq // hk
    width = first_width if top == 0 else 2 * topk
    bg, bd = max(16, triton.next_power_of_2(group)), max(16, triton.next_power_of_2(d))
    mode = {"pisa1": 0, "pisa2": 1, "pisa": 2}[method]
    output = torch.empty((b, t, hk, topk), device=q.device, dtype=torch.int32)
    positions = positions.contiguous()
    fuse_leaf = routing_schedule == "fused"
    candidates = (
        output if fuse_leaf else torch.empty((b, t, hk, width), device=q.device, dtype=torch.int32)
    )
    # Torch 2.7 mutation analysis does not bind JIT default arguments.
    launch = {"INTEGER_SORT": False}
    leaf_launch, final_launch = {}, {"INTEGER_SORT": False}
    if (q.dtype == torch.bfloat16 and mode in (1, 2)
            and (block_size, d, topk, group) == (64, 64, 8, 16)):
        # Preserve the calibrated fused layout and keep the grouped one-warp
        # route within its checked network widths and four-query leaf tiles.
        if fuse_leaf:
            launch = {"num_warps": 2, "num_stages": 2, "INTEGER_SORT": True}
        elif (routing_schedule == "grouped" and leaf_tile == 4
                and first_width <= 16 and width <= 16):
            launch = {"num_warps": 1, "num_stages": 2, "INTEGER_SORT": True}
            leaf_launch = {"num_warps": 4, "num_stages": 2}
            final_launch = {"num_warps": 1, "num_stages": 2, "INTEGER_SORT": True}
    coarse_kernel[(t, b * hk)](
        q,
        summary,
        offsets,
        lengths,
        positions,
        candidates,
        k,
        k.shape[1],
        summary.stride(0),
        k.stride(0),
        k.stride(1),
        t,
        hq,
        hk,
        d,
        top,
        first_width,
        block_size,
        topk,
        group,
        bg,
        bd,
        width,
        mode,
        force_first,
        force_recent,
        fuse_leaf,
        **launch,
    )
    if not fuse_leaf:
        from pisa.ops.flash_sparse_attn.top_k_pyramid.leaf_triton import grouped_leaf_scores

        scores = grouped_leaf_scores(
            q,
            k,
            candidates,
            block_size,
            method,
            leaf_tile,
            positions=positions,
            topk=topk,
            force_first=force_first,
            force_recent=force_recent,
            leaf_means=summary[:, : sizes[0]],
            **leaf_launch,
        )
        final_kernel[(b * t * hk,)](
            candidates,
            scores,
            positions,
            output,
            t,
            hk,
            blocks,
            block_size,
            topk,
            width,
            force_first,
            force_recent,
            mode,
            **final_launch,
        )
    return output.long()

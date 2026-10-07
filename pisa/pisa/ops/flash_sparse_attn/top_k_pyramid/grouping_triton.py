"""Linear-size request buckets, built entirely on the device."""

import torch
import triton
import triton.language as tl


@triton.jit
def bucket_kernel(
    CANDIDATES,
    ACTIVE,
    COUNTS,
    OFFSETS,
    CURSORS,
    OWNERS,
    REQUESTS,
    SIZE: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr,
    TILE: tl.constexpr,
    FILTER: tl.constexpr,
    SCATTER: tl.constexpr,
    LANES: tl.constexpr,
):
    request = tl.program_id(0) * LANES + tl.arange(0, LANES)
    block = tl.load(CANDIDATES + request, request < SIZE, other=-1)
    active = (request < SIZE) & (block >= 0) & (block < BLOCKS)
    if FILTER:
        active &= tl.load(ACTIVE + request, request < SIZE, other=False)
    row = request // WIDTH
    group = ((row // (TOKENS * HEADS)) * HEADS + row % HEADS) * BLOCKS + block
    if SCATTER:
        rank = tl.atomic_add(CURSORS + group, 1, active, sem="relaxed")
        start = tl.load(OFFSETS + group, active, other=0)
        tile = start + rank // TILE
        tl.store(REQUESTS + tile * TILE + rank % TILE, request, active)
        tl.store(OWNERS + tile, group, active & (rank % TILE == 0))
    else:
        tl.atomic_add(COUNTS + group, 1, active, sem="relaxed")


def group_requests(candidates, num_blocks, tile_size, scoreable=None):
    """Return fixed-capacity tiles; an owner of -1 marks an unused tile.

    Counts determine padded bucket offsets on the GPU. The allocation uses
    the maximum possible number of tiles, so no host read of a count is needed.
    Request order within a bucket is unspecified; each score is scattered back
    to its original query slot.
    """
    batch, tokens, heads, width = candidates.shape
    candidates = candidates.contiguous()
    if scoreable is not None:
        scoreable = scoreable.contiguous()
    count = candidates.numel()
    groups = batch * heads * num_blocks
    capacity = triton.cdiv(count, tile_size) + groups
    counts = torch.zeros(groups, dtype=torch.int32, device=candidates.device)
    owners = torch.full((capacity,), -1, dtype=torch.int32, device=candidates.device)
    requests = torch.full((capacity, tile_size), -1, dtype=torch.int32, device=candidates.device)
    grid = (triton.cdiv(count, 256),)
    active = candidates if scoreable is None else scoreable
    bucket_kernel[grid](
        candidates,
        active,
        counts,
        counts,
        counts,
        owners,
        requests,
        count,
        tokens,
        heads,
        width,
        num_blocks,
        tile_size,
        scoreable is not None,
        False,
        256,
    )
    tiles = (counts + tile_size - 1) // tile_size
    offsets = tiles.cumsum(0, dtype=torch.int32) - tiles
    cursors = torch.zeros_like(counts)
    bucket_kernel[grid](
        candidates,
        active,
        counts,
        offsets,
        cursors,
        owners,
        requests,
        count,
        tokens,
        heads,
        width,
        num_blocks,
        tile_size,
        scoreable is not None,
        True,
        256,
    )
    return owners, requests

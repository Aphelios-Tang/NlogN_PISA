"""Group leaf requests so up to four queries reuse one key-block load.

Metadata uses device counts, prefix sums and scatter, included in end-to-end timing.
The kernel changes the execution schedule, not the per-query candidate set or
the scoring equations. CUDA numerical/performance tests must be run separately.
"""

import torch
import triton
import triton.language as tl

from pisa.utils.validation import validate_qkv

from .top_k_pyramid_triton import score_masks, score_support


def leaf_request_groups(candidates, num_blocks, tile_size, scoreable=None):
    from .grouping_triton import group_requests

    return group_requests(candidates, num_blocks, tile_size, scoreable)


@triton.jit
def scoreable_mask_kernel(
    CANDIDATES, POSITIONS, OUTPUT,
    ID_BATCH, ID_TIME, ID_HEAD, ID_SLOT, POS_BATCH, POS_TIME,
    T: tl.constexpr, HK: tl.constexpr, WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
    FIRST: tl.constexpr, RECENT: tl.constexpr, MODE: tl.constexpr,
    LANES: tl.constexpr,
):
    row = tl.program_id(0)
    batch, token, head = row // (T * HK), (row // HK) % T, row % HK
    lane = tl.arange(0, LANES)
    ids = tl.load(
        CANDIDATES + batch * ID_BATCH + token * ID_TIME + head * ID_HEAD + lane * ID_SLOT,
        lane < WIDTH, other=-1,
    )
    position = tl.load(POSITIONS + batch * POS_BATCH + token * POS_TIME)
    valid, forced, scoreable = score_masks(ids, position, BLOCKS, C, 0, K, FIRST, RECENT, MODE)
    tl.store(OUTPUT + row * WIDTH + lane, scoreable, lane < WIDTH)


def leaf_scoreable_mask(candidates, positions, num_blocks, block_size, method, topk, force_first, force_recent):
    """Build the original per-slot eligibility mask with one integer kernel."""
    if positions.device != candidates.device:
        raise ValueError("positions and candidates must share a device")
    batch, tokens, heads, width = candidates.shape
    positions = torch.broadcast_to(positions, (batch, tokens))
    output = torch.empty(candidates.shape, dtype=torch.bool, device=candidates.device)
    if output.numel():
        with torch.cuda.device(candidates.device):
            scoreable_mask_kernel[(batch * tokens * heads,)](
                candidates, positions, output, *candidates.stride(), *positions.stride(),
                tokens, heads, width, num_blocks, block_size,
                0 if method == "pisa1" else topk, force_first, force_recent,
                {"pisa1": 0, "pisa2": 1, "pisa": 2}[method], triton.next_power_of_2(width),
                num_warps=1, num_stages=2,
            )
    return output


@triton.jit
def grouped_leaf_kernel(
    Q,
    KEY,
    GROUPS,
    REQUESTS,
    OUTPUT,
    MEAN,
    KEY_BATCH,
    KEY_TIME,
    MEAN_BATCH,
    MEAN_TIME,
    T: tl.constexpr,
    N,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
    BLOCKS: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    BD: tl.constexpr,
    BC: tl.constexpr,
    TILE: tl.constexpr,
    MODE: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.load(GROUPS + tile)
    if group >= 0:
        block, shared = group % BLOCKS, group // BLOCKS
        batch, head = shared // HK, shared % HK
        row = tl.arange(0, TILE * BG)
        query_slot, query_head = row // BG, row % BG
        request = tl.load(REQUESTS + tile * TILE + query_slot)
        token = (request // W // HK) % T
        d, position = tl.arange(0, BD), tl.arange(0, BC)
        query = tl.load(
            Q
            + ((batch * T + token[:, None]) * HQ + head * G + query_head[:, None]) * D
            + d[None, :],
            (request[:, None] >= 0) & (query_head[:, None] < G) & (d[None, :] < D),
            other=0,
        )
        if MODE == 0:
            mean = tl.load(
                MEAN + batch * MEAN_BATCH + block * MEAN_TIME + head * D + d, d < D, other=0
            ).to(tl.float32)
            products = tl.dot(query.to(tl.float32), tl.broadcast_to(mean[:, None], (BD, 16)))
            head_score = tl.sum(tl.where(tl.arange(0, 16)[None, :] == 0, products, 0.0), 1)
        else:
            key_position = block * C + position
            valid_key = (position < C) & (key_position < N)
            key = tl.load(
                KEY + batch * KEY_BATCH + key_position[None, :] * KEY_TIME + head * D + d[:, None],
                valid_key[None, :] & (d[:, None] < D),
                other=0,
            )
            products = tl.dot(query, key)
            if MODE != 0:
                # Keep the token/head reduction layouts stable for close FP32 scores.
                cube = tl.reshape(products, (TILE, BG, BC))
                head_score = score_support(
                    cube, tl.broadcast_to(valid_key[None, None, :], (TILE, BG, BC)),
                    MODE, D**-0.5,
                )
                head_score = tl.reshape(head_score, (TILE * BG,))
            else:
                head_score = score_support(
                    products, tl.broadcast_to(valid_key[None, :], (TILE * BG, BC)), MODE, D**-0.5
                )
        head_score = tl.where(query_head < G, head_score, 0.0)
        if MODE != 0:
            score = tl.sum(tl.trans(tl.reshape(head_score, (TILE, BG))), 0)
        else:
            score = tl.sum(tl.reshape(head_score, (TILE, BG)), 1)
        slots = tl.arange(0, TILE)
        destination = tl.load(REQUESTS + tile * TILE + slots)
        tl.store(OUTPUT + destination, score, destination >= 0)


@torch.no_grad()
def grouped_leaf_scores(
    q,
    k,
    candidates,
    block_size,
    method,
    tile_size=4,
    *,
    positions=None,
    topk=None,
    force_first=True,
    force_recent=2,
    leaf_means=None,
    num_warps=None,
    num_stages=None,
):
    """Return FP32 leaf scores with the shape and order of ``candidates``.

    ``candidates`` is [B,Tq,Hkv,W]. Invalid block ids receive negative infinity.
    Tile size is 1, 2, or 4; all GQA heads retain their own score before summation.
    Keys from a partial final block are masked using its actual token count.
    Unspecified launch options retain Triton's defaults for direct callers.
    """
    validate_qkv(q, k)
    if not q.is_cuda:
        raise ValueError("grouped leaf scoring requires a CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("grouped leaf scoring supports FP16, BF16 and FP32")
    if method not in {"pisa1", "pisa2", "pisa"} or tile_size not in (1, 2, 4):
        raise ValueError("expected method pisa1/pisa2/pisa and tile_size 1/2/4")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if candidates.ndim != 4 or candidates.shape[:3] != (*q.shape[:2], k.shape[2]):
        raise ValueError("candidates must have shape [B,Tq,Hkv,W]")
    if candidates.dtype not in (torch.int32, torch.int64) or candidates.device != q.device:
        raise ValueError("candidates must use an integer dtype on the query device")
    q = q.contiguous()
    if k.stride(-1) != 1 or k.stride(-2) != k.shape[-1]:
        k = k.contiguous()
    tokens, query_heads, dimension = q.shape[1:]
    heads = k.shape[2]
    blocks = triton.cdiv(k.shape[1], block_size)
    scoreable = None
    if positions is not None:
        scoreable = leaf_scoreable_mask(
            candidates, positions, blocks, block_size, method, topk, force_first, force_recent,
        )
    if method == "pisa1" and leaf_means is None:
        from pisa.ops.compress.mean_pooling.mean_pooling import mean_pool

        leaf_means = mean_pool(k, block_size)
    means = k if leaf_means is None else leaf_means
    groups, requests = leaf_request_groups(candidates, blocks, tile_size, scoreable)
    output = torch.full(candidates.shape, -torch.inf, device=q.device, dtype=torch.float32)
    launch = {}
    if num_warps is not None:
        launch["num_warps"] = num_warps
    if num_stages is not None:
        launch["num_stages"] = num_stages
    if groups.numel():
        grouped_leaf_kernel[(groups.numel(),)](
            q,
            k,
            groups,
            requests,
            output,
            means,
            k.stride(0),
            k.stride(1),
            means.stride(0),
            means.stride(1),
            tokens,
            k.shape[1],
            query_heads,
            heads,
            dimension,
            block_size,
            candidates.shape[-1],
            blocks,
            query_heads // heads,
            max(16, triton.next_power_of_2(query_heads // heads)),
            max(16, triton.next_power_of_2(dimension)),
            max(16, triton.next_power_of_2(block_size)),
            tile_size,
            {"pisa1": 0, "pisa2": 1, "pisa": 2}[method],
            **launch,
        )
    return output

"""Flat block selection using normalized compression probabilities."""

import torch
import triton
import triton.language as tl
from fla.ops.nsa.utils import _bitonic_merge

from pisa.ops.compress.mean_pooling.mean_pooling import mean_pool

from pisa.ops.flash_sparse_attn.top_k_pyramid.top_k_pyramid_triton import required


@triton.jit(do_not_specialize=["M"])
def bsa_kernel(
    Q,
    KEY,
    POS,
    OUT,
    T: tl.constexpr,
    M,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    BG: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    STAGES: tl.constexpr,
    FIRST: tl.constexpr,
    RECENT: tl.constexpr,
):
    token, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HK, bh % HK
    position = tl.load(POS + batch * T + token)
    g, d, lane = tl.arange(0, BG), tl.arange(0, BD), tl.arange(0, BM)
    query = tl.load(
        Q + ((batch * T + token) * HQ + head * G + g[:, None]) * D + d[None, :],
        (g[:, None] < G) & (d[None, :] < D),
        other=0,
    )
    # Promote Q so a constexpr scale is not first rounded to the input dtype.
    query = (query.to(tl.float32) * (D**-0.5)).to(query.dtype)
    complete = (position + 1) // C
    current = position // C
    maximum = tl.full((BG,), -float("inf"), tl.float32)
    total = tl.zeros((BG,), tl.float32)
    for start in range(tl.cdiv(complete, BM)):
        ids = start * BM + lane
        keys = tl.load(
            KEY + ((batch * M + ids[None, :]) * HK + head) * D + d[:, None],
            (ids[None, :] < M) & (d[:, None] < D),
            other=0,
        )
        logits = tl.where((ids < complete)[None, :], tl.dot(query, keys), -float("inf"))
        new_max = tl.maximum(maximum, tl.max(logits, 1))
        total = total * tl.exp(maximum - new_max) + tl.sum(tl.exp(logits - new_max[:, None]), 1)
        maximum = new_max
    normalizer = tl.where(complete > 0, maximum + tl.log(total), 0.0)
    retained_scores = tl.full((BM,), -1.0, tl.float32)
    retained_ids = tl.full((BM,), 0, position.dtype)
    for start in range(tl.cdiv(current + 1, BM)):
        ids = start * BM + lane
        keys = tl.load(
            KEY + ((batch * M + ids[None, :]) * HK + head) * D + d[:, None],
            (ids[None, :] < M) & (d[:, None] < D),
            other=0,
        )
        logits = tl.where((ids < current)[None, :], tl.dot(query, keys), -float("inf"))
        forced = required(ids, position, C, 0, FIRST, RECENT)
        probabilities = tl.where(forced[None, :], 1.0, tl.exp(logits - normalizer[:, None]))
        scores = tl.sum(tl.where(g[:, None] < G, probabilities, 0.0), 0)
        ids = tl.where((ids <= current) & (ids < M), ids, -1)
        # Form alternating sorted runs, then merge the best half with the
        # preceding tile. Score-only comparisons preserve the network's ties.
        for stage in tl.static_range(1, STAGES):
            scores, ids = _bitonic_merge(scores, ids, stage, 2, STAGES)
        if start > 0:
            scores, ids = _bitonic_merge(scores, ids, STAGES, False, STAGES)
            scores = tl.where(lane < BM // 2, retained_scores, scores)
            ids = tl.where(lane < BM // 2, retained_ids, ids)
        retained_scores, retained_ids = _bitonic_merge(scores, ids, STAGES, True, STAGES)
    tl.store(OUT + ((batch * T + token) * HK + head) * K + lane, retained_ids, lane < K)


def select_bsa(q, k, block_size, topk, positions, force_first, force_recent):
    if block_size < 2 * topk:
        raise ValueError("flat GPU routing requires block_size >= 2 * topk")
    summaries = mean_pool(k, block_size).contiguous()
    b, t, hq, d = q.shape
    hk = k.shape[2]
    output = torch.empty((b, t, hk, topk), device=q.device, dtype=torch.int32)
    bsa_kernel[(t, b * hk)](
        q.contiguous(),
        summaries,
        positions.contiguous(),
        output,
        t,
        summaries.shape[1],
        hq,
        hk,
        d,
        hq // hk,
        block_size,
        topk,
        max(16, triton.next_power_of_2(hq // hk)),
        max(16, triton.next_power_of_2(d)),
        block_size,
        block_size.bit_length() - 1,
        force_first,
        force_recent,
    )
    return output.long()

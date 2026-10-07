"""Online-softmax attention for selected blocks at arbitrary query offsets."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "K_BATCH", "V_BATCH"])
def decode_kernel(
    Q,
    K,
    V,
    I,
    P,
    O,
    T: tl.constexpr,
    N,
    K_BATCH,
    V_BATCH,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    K_DIM: tl.constexpr,
    V_TOKEN: tl.constexpr,
    V_HEAD: tl.constexpr,
    V_DIM: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    SCALE: tl.constexpr,
):
    token, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HK, bh % HK
    g, d, v, s = tl.arange(0, BG), tl.arange(0, BD), tl.arange(0, BV), tl.arange(0, BC)
    position = tl.load(P + batch * T + token)
    q = tl.load(
        Q + ((batch * T + token) * HQ + head * G + g[:, None]) * D + d[None, :],
        (g[:, None] < G) & (d[None, :] < D),
        other=0,
    )
    maximum = tl.full((BG,), -float("inf"), tl.float32)
    denominator = tl.zeros((BG,), tl.float32)
    numerator = tl.zeros((BG, BV), tl.float32)
    for slot in range(S):
        block = tl.load(I + ((batch * T + token) * HK + head) * S + slot)
        indices = block * C + s
        visible = (block >= 0) & (s < C) & (indices < N) & (indices <= position)
        keys = tl.load(
            K + batch * K_BATCH + indices[None, :] * K_TOKEN + head * K_HEAD + d[:, None] * K_DIM,
            visible[None, :] & (d[:, None] < D),
            other=0,
        )
        values = tl.load(
            V + batch * V_BATCH + indices[:, None] * V_TOKEN + head * V_HEAD + v[None, :] * V_DIM,
            visible[:, None] & (v[None, :] < DV),
            other=0,
        )
        logits = tl.dot(q, keys) * SCALE
        logits = tl.where(visible[None, :], logits, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp2((maximum - safe_max) * 1.4426950408889634)
        probability = tl.exp2((logits - safe_max[:, None]) * 1.4426950408889634)
        numerator = numerator * alpha[:, None] + tl.dot(probability.to(values.dtype), values)
        denominator = denominator * alpha + tl.sum(probability, 1)
        maximum = new_max
    result = numerator / tl.maximum(denominator[:, None], 1.0e-30)
    tl.store(
        O + ((batch * T + token) * HQ + head * G + g[:, None]) * DV + v[None, :],
        result,
        (g[:, None] < G) & (v[None, :] < DV),
    )


def decode_attention(q, k, v, indices, positions, block_size, scale):
    b, t, hq, d = q.shape
    hk, dv = k.shape[2], v.shape[-1]
    output = torch.empty((b, t, hq, dv), device=q.device, dtype=q.dtype)
    with torch.cuda.device(q.device):
        decode_kernel[(t, b * hk)](
            q.contiguous(),
            k,
            v,
            indices.contiguous(),
            positions.contiguous(),
            output,
            t,
            k.shape[1],
            k.stride(0),
            v.stride(0),
            *k.stride()[1:],
            *v.stride()[1:],
            hq,
            hk,
            d,
            dv,
            indices.shape[-1],
            block_size,
            hq // hk,
            max(16, triton.next_power_of_2(hq // hk)),
            max(16, triton.next_power_of_2(d)),
            max(16, triton.next_power_of_2(dv)),
            max(16, triton.next_power_of_2(block_size)),
            scale,
            # FP32 K/V tiles exceed Ada's shared-memory limit with three stages.
            num_stages=1 if q.dtype == torch.float32 else 3,
        )
    return output

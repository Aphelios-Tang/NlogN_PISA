"""RMSNorm with FP32 intermediates and configurable reduction layouts.

Import this CUDA specialization lazily. Its module and parameter ownership stay
with the caller; scale gradients accumulate in FP32 across token rows.
"""

import math

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({}, num_warps=n, num_stages=2) for n in (1, 2, 4, 8, 16, 32)],
    key=["WIDTH"],
)
@triton.jit
def row_scale(X, W, Y, S, WIDTH: tl.constexpr, TILE: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    active = column < WIDTH
    value = tl.load(X + row * WIDTH + column, active, other=0).to(tl.float32)
    scale = tl.load(W + column, active, other=1).to(tl.float32)
    radius = tl.sqrt(tl.sum(value * value, 0) / WIDTH + EPS)
    normalized = value / radius
    tl.store(Y + row * WIDTH + column, normalized * scale, active)
    tl.store(S + row, radius)


@triton.autotune(
    configs=[triton.Config({}, num_warps=n, num_stages=2) for n in (1, 2, 4, 8, 16, 32)],
    key=["WIDTH"],
)
@triton.jit
def row_adjoint(X, W, G, S, DX, PARTIAL, WIDTH: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    active = column < WIDTH
    offset = row * WIDTH + column
    value = tl.load(X + offset, active, other=0).to(tl.float32)
    upstream = tl.load(G + offset, active, other=0).to(tl.float32)
    scale = tl.load(W + column, active, other=1).to(tl.float32)
    radius = tl.load(S + row)
    normalized = value / radius
    weighted = upstream * scale
    inner_product = tl.sum(normalized * weighted, 0)
    correction = (normalized * inner_product) / WIDTH
    input_adjoint = (1.0 / radius) * (weighted - correction)
    tl.store(DX + offset, input_adjoint, active)
    tl.store(PARTIAL + offset, upstream * normalized, active)


class RowRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, epsilon, num_warps):
        width = values.shape[-1]
        rows = values.numel() // width
        result = torch.empty_like(values)
        radius = torch.empty(rows, dtype=torch.float32, device=values.device)
        if rows:
            with torch.cuda.device(values.device):
                kernel = row_scale if num_warps is None else row_scale.fn
                options = {} if num_warps is None else dict(num_warps=num_warps, num_stages=2)
                kernel[(rows,)](values, weight, result, radius, width,
                                triton.next_power_of_2(width), epsilon, **options)
        ctx.num_warps = num_warps
        ctx.save_for_backward(values, weight, radius)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        values, weight, radius = ctx.saved_tensors
        width = values.shape[-1]
        rows = values.numel() // width
        input_adjoint = torch.empty_like(values)
        scale_rows = torch.empty((rows, width), dtype=torch.float32, device=values.device)
        if rows:
            with torch.cuda.device(values.device):
                kernel = row_adjoint if ctx.num_warps is None else row_adjoint.fn
                options = {} if ctx.num_warps is None else dict(num_warps=ctx.num_warps, num_stages=2)
                kernel[(rows,)](values, weight, upstream.contiguous(), radius,
                                input_adjoint, scale_rows, width, triton.next_power_of_2(width),
                                **options)
        return input_adjoint, scale_rows.sum(0), None, None


def rms_norm(values, weight, epsilon=1e-5, num_warps=None):
    """Normalize CUDA FP32/BF16 inputs with FP32 intermediate arithmetic.

    Supports first-order training gradients for both inputs and scale. Higher
    derivatives are explicitly unsupported by this CUDA specialization.
    """
    if values.device.type != "cuda" or values.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("RMSNorm kernel requires CUDA FP32/BF16 inputs")
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("RMSNorm input must have a positive feature width")
    if weight.dtype not in (torch.float32, torch.bfloat16) or weight.shape != (values.shape[-1],):
        raise ValueError("RMSNorm kernel requires one FP32/BF16 scale per feature width")
    if weight.device != values.device:
        raise ValueError("RMSNorm input and scale must share a device")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be positive and finite")
    if num_warps is not None and (type(num_warps) is not int or num_warps not in (1, 2, 4, 8, 16, 32)):
        raise ValueError("RMSNorm num_warps must be None or one of 1, 2, 4, 8, 16, 32")
    return RowRMS.apply(values.contiguous(), weight.contiguous(), epsilon, num_warps)

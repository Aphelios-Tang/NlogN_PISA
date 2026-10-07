"""Valid-token pooling with native or FP32 accumulation."""

import torch
import triton
import triton.language as tl


@triton.jit
def pool_kernel(
    KEYS, OUTPUT, BATCH_STRIDE, TIME_STRIDE, HEAD_STRIDE, DIM_STRIDE,
    N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BLOCKS: tl.constexpr,
    SIZE: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
    FP32_ACCUMULATION: tl.constexpr,
):
    block = tl.program_id(0)
    batch, head = tl.program_id(1) // H, tl.program_id(1) % H
    row = tl.arange(0, ROWS)
    column = tl.program_id(2) * COLS + tl.arange(0, COLS)
    token = block * SIZE + row
    value = tl.load(
        KEYS + batch * BATCH_STRIDE + token[:, None] * TIME_STRIDE
        + head * HEAD_STRIDE + column[None, :] * DIM_STRIDE,
        (row[:, None] < SIZE) & (token[:, None] < N) & (column[None, :] < D),
        other=0.0,
    )
    if FP32_ACCUMULATION:
        value = value.to(tl.float32)
    count = tl.minimum(SIZE, N - block * SIZE).to(tl.float32)
    mean = tl.sum(value, 0) / count
    tl.store(OUTPUT + ((batch * BLOCKS + block) * H + head) * D + column,
             mean, column < D)


def mean_pool_forward(keys, size, *, fp32_accumulation=False):
    """Pool a nonempty CUDA tensor without recording an autograd operation."""
    if not keys.is_cuda or keys.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton pooling requires CUDA float16, bfloat16, or float32 tensors")
    if torch.is_grad_enabled() and keys.requires_grad:
        raise ValueError("Triton pooling does not support gradients")
    if size < 1 or keys.ndim != 4 or not keys.numel():
        raise ValueError("Triton pooling requires a positive size and nonempty [B,N,H,D] input")
    batch, length, heads, dimension = keys.shape
    blocks = triton.cdiv(length, size)
    output = torch.empty((batch, blocks, heads, dimension), device=keys.device, dtype=keys.dtype)
    columns = min(256, triton.next_power_of_2(dimension))
    with torch.cuda.device(keys.device):
        pool_kernel[(blocks, batch * heads, triton.cdiv(dimension, columns))](
            keys, output, *keys.stride(), length, heads, dimension, blocks,
            size, triton.next_power_of_2(size), columns, fp32_accumulation, num_warps=4, num_stages=2,
        )
    return output

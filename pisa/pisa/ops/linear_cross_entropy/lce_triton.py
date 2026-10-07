"""Independent first-order linear CE with grouped FP32 normalization.

Projection and adjoint GEMMs preserve active autocast. Each dZ element has one
storage cast after target subtraction and mean normalization. Fixed four-warp,
two-stage launches define the reduction configuration; different configurations
or hardware need their own numerical validation.
"""

import math

import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_log_normalizer(LOGITS, PARTIAL, V: tl.constexpr, GROUPS: tl.constexpr,
                         WIDTH: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    column = group * WIDTH + tl.arange(0, WIDTH)
    value = tl.load(LOGITS + row * V + column, column < V, other=-float("inf")).to(tl.float32)
    maximum = tl.max(value, 0)
    normalizer = maximum + tl.log(tl.sum(tl.exp(value - maximum), 0))
    tl.store(PARTIAL + row * GROUPS + group, normalizer)


@triton.jit
def loss_and_gradient(LOGITS, PARTIAL, TARGET, LOSS, VALID_COUNT,
                      V: tl.constexpr, GROUPS: tl.constexpr,
                      GROUP_WIDTH: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    group = tl.arange(0, GROUP_WIDTH)
    partial = tl.load(PARTIAL + row * GROUPS + group, group < GROUPS, other=-float("inf"))
    maximum = tl.max(partial, 0)
    normalizer = maximum + tl.log(tl.sum(tl.exp(partial - maximum), 0))
    target = tl.load(TARGET + row)
    valid = target != -100
    target_value = tl.load(LOGITS + row * V + target, valid, other=0).to(tl.float32)
    row_loss = tl.where(valid, (normalizer - target_value) / VALID_COUNT, 0.0)
    tl.store(LOSS + row, row_loss)
    lane = tl.arange(0, WIDTH)
    for start in range(tl.cdiv(V, WIDTH)):
        column = start * WIDTH + lane
        value = tl.load(LOGITS + row * V + column, column < V, other=-float("inf")).to(tl.float32)
        probability = tl.exp(value - normalizer)
        gradient = (probability - (column == target).to(tl.float32)) / VALID_COUNT
        tl.store(LOGITS + row * V + column, tl.where(valid, gradient, 0.0), column < V)


def projection_chunk_size(rows, vocabulary, dimension):
    partitions = min(8, math.ceil(vocabulary / dimension))
    return 1 << (max(1, math.ceil(rows / partitions)) - 1).bit_length()


def valid_target_count(targets, vocabulary):
    if not targets.numel():
        return 0
    valid = targets != -100
    invalid = valid & ((targets < 0) | (targets >= vocabulary))
    # One host synchronization validates class indices and obtains the common
    # denominator for all projection chunks. Do not inspect tensor values again.
    count, invalid_present = torch.stack((valid.sum(), invalid.any().long())).tolist()
    if invalid_present:
        raise ValueError("targets must be -100 or an in-vocabulary class index")
    return count


@triton.jit
def accumulate_weight_gradient(ACCUMULATOR, CONTRIBUTION, COUNT: tl.constexpr, WIDTH: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * WIDTH + tl.arange(0, WIDTH)
    active = offset < COUNT
    previous = tl.load(ACCUMULATOR + offset, active, other=0)
    contribution = tl.load(CONTRIBUTION + offset, active, other=0)
    # Preserve contribution.to(hidden.dtype) before adding, without materializing
    # its converted tensor. A BF16 accumulator can distinguish the two roundings.
    contribution = contribution.to(previous.dtype).to(tl.float32)
    result = previous.to(tl.float32) + contribution
    tl.store(ACCUMULATOR + offset, result, active)


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets):
        rows, dimension = hidden.shape
        vocabulary = weight.shape[0]
        targets = targets.contiguous()
        denominator = valid_target_count(targets, vocabulary)
        hidden_gradient = torch.empty_like(hidden)
        weight_gradient = torch.zeros(weight.shape, device=weight.device, dtype=hidden.dtype)
        ctx.weight_dtype = weight.dtype
        if denominator == 0:
            hidden_gradient.zero_()
            ctx.save_for_backward(hidden_gradient, weight_gradient.to(weight.dtype))
            return torch.zeros((), device=hidden.device, dtype=torch.float32)
        chunk_size = projection_chunk_size(rows, vocabulary, dimension)
        width = min(8192, triton.next_power_of_2(vocabulary))
        groups = triton.cdiv(vocabulary, width)
        row_losses = torch.empty(rows, device=hidden.device, dtype=torch.float32)
        with torch.cuda.device(hidden.device):
            for start in range(0, rows, chunk_size):
                end = min(rows, start + chunk_size)
                source = hidden[start:end]
                logits = F.linear(source, weight).contiguous()
                partial = torch.empty((end - start, groups), device=hidden.device, dtype=torch.float32)
                group_log_normalizer[(end - start, groups)](
                    logits, partial, vocabulary, groups, width, num_warps=4, num_stages=2,
                )
                loss_and_gradient[(end - start,)](
                    logits, partial, targets[start:end], row_losses[start:end], denominator,
                    vocabulary, groups, triton.next_power_of_2(groups), width, num_warps=4, num_stages=2,
                )
                hidden_gradient[start:end].copy_(torch.mm(logits, weight))
                contribution = torch.mm(logits.T, source)
                accumulate_weight_gradient[(triton.cdiv(weight_gradient.numel(), 1024),)](
                    weight_gradient, contribution, weight_gradient.numel(), 1024,
                    num_warps=4, num_stages=2,
                )
        # The cross-chunk accumulator follows hidden.dtype; the saved adjoint
        # follows the weight's storage dtype before upstream scaling occurs.
        ctx.save_for_backward(hidden_gradient, weight_gradient.to(weight.dtype))
        return row_losses.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, scalar_gradient):
        hidden_gradient, weight_gradient = ctx.saved_tensors
        scale = scalar_gradient.float()
        return (hidden_gradient.float() * scale).to(hidden_gradient.dtype), \
            (weight_gradient.float() * scale).to(ctx.weight_dtype), None

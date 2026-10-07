"""FP32 planar rotations with a single activation-dtype storage cast.

This CUDA specialization treats positions and frequencies as fixed metadata.
Import it lazily so CPU model execution does not require Triton.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def rotate_features(X, FREQUENCY, POSITION, Y, COUNT: tl.constexpr,
                    HEADS: tl.constexpr, WIDTH: tl.constexpr,
                    TRANSPOSE: tl.constexpr, TILE: tl.constexpr):
    pair = tl.program_id(0) * TILE + tl.arange(0, TILE)
    active = pair < COUNT
    half = WIDTH // 2
    feature = pair % half
    row = pair // half
    offset = row * WIDTH + feature
    first = tl.load(X + offset, active, other=0).to(tl.float32)
    second = tl.load(X + offset + half, active, other=0).to(tl.float32)
    frequency = tl.load(FREQUENCY + feature, active, other=0).to(tl.float32)
    position = tl.load(POSITION + row // HEADS, active, other=0).to(tl.float32)
    if TRANSPOSE:
        frequency = -frequency
    phase = frequency * position
    cosine, sine = tl.cos(phase), tl.sin(phase)
    tl.store(Y + offset, first * cosine - second * sine, active)
    tl.store(Y + offset + half, first * sine + second * cosine, active)


class RotaryPairs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, frequencies, positions):
        ctx.save_for_backward(frequencies, positions)
        result = torch.empty_like(value)
        count = value.numel() // 2
        if count:
            with torch.cuda.device(value.device):
                rotate_features[(triton.cdiv(count, 256),)](
                    value, frequencies, positions, result, count,
                    value.shape[-2], value.shape[-1], False, 256,
                    num_warps=4, num_stages=2)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        frequencies, positions = ctx.saved_tensors
        gradient = gradient.contiguous()
        result = torch.empty_like(gradient)
        count = gradient.numel() // 2
        if count:
            with torch.cuda.device(gradient.device):
                rotate_features[(triton.cdiv(count, 256),)](
                    gradient, frequencies, positions, result, count,
                    gradient.shape[-2], gradient.shape[-1], True, 256,
                    num_warps=4, num_stages=2)
        return result, None, None


def rotary_embedding(value, frequencies, positions):
    """Rotate [B,T,H,D] split-half pairs; provide first-order input gradients.

    Positions may have shape [T], [1,T] or [B,T], including explicit document
    resets and decode offsets. Frequencies and positions cannot require grads;
    the model uses its differentiable Torch path for trainable metadata.
    """
    if value.device.type != 'cuda' or value.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Rotary kernel requires CUDA FP32/BF16 inputs')
    if value.ndim != 4 or value.shape[-1] < 2 or value.shape[-1] % 2:
        raise ValueError('Rotary kernel requires [B,T,H,D] with positive even feature width')
    if frequencies.requires_grad or positions.requires_grad:
        raise ValueError('Rotary kernel requires fixed frequency and position metadata')
    if frequencies.shape != (value.shape[-1] // 2,) or frequencies.device != value.device:
        raise ValueError('Rotary frequency shape and device must match the input')
    batch, length = value.shape[:2]
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if positions.ndim != 2 or positions.shape[1] != length or positions.shape[0] not in (1, batch):
        raise ValueError('Rotary positions must have shape [T], [1,T] or [B,T]')
    positions = positions.to(device=value.device, dtype=torch.float32).expand(batch, length).contiguous()
    return RotaryPairs.apply(value.contiguous(), frequencies.float().contiguous(), positions)

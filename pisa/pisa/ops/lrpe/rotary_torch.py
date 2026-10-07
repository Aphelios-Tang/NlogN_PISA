"""Split-half rotary encoding with eager FP32 arithmetic."""
import torch


def rotary_torch(x, frequency, positions):
    angles = positions.to(device=x.device, dtype=torch.float32)[..., None, None] * frequency
    first, second = x.float().chunk(2, -1)
    cosine, sine = angles.cos(), angles.sin()
    return torch.cat((first * cosine - second * sine, first * sine + second * cosine), -1).to(x.dtype)

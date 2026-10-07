"""RMS normalization with the paper's FP32 division and final dtype cast."""
import torch


def rms_norm_torch(x, weight, eps=1e-5):
    fp32 = x.float()
    mean_square = fp32.square().sum(-1, keepdim=True) / fp32.shape[-1]
    return (fp32 / torch.sqrt(mean_square + eps) * weight.float()).to(x.dtype)

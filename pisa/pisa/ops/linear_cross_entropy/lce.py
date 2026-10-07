"""Explicit CUDA linear cross entropy without importing GPU dependencies eagerly."""

import torch


def linear_cross_entropy(hidden, weight, targets):
    """Mean CE for hidden [N,D], weight [V,D], and already shifted targets [N].

    Dense CUDA FP32/BF16 inputs and int32/int64 targets are supported. Targets
    are class IDs or -100; no label shift, bias, smoothing, or distributed
    reduction is performed. Empty/all-ignored targets return a connected FP32
    zero. Only first derivatives are supported. CPU callers should explicitly
    use their Torch loss implementation instead of changing this operator's
    precision contract implicitly.

    GEMMs retain the caller's active CUDA autocast policy. Projection gradients
    are normalized over all valid rows and cast once to the projection dtype.
    Weight-gradient chunks accumulate in hidden.dtype and are cast to
    weight.dtype before being saved. Backward scales both saved gradients in
    FP32, then casts them back to their original input dtypes.
    """
    if not all(isinstance(value, torch.Tensor) for value in (hidden, weight, targets)):
        raise TypeError("hidden, weight, and targets must be Torch tensors")
    if hidden.ndim != 2 or weight.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected shapes hidden [N,D], weight [V,D], and already shifted targets [N]")
    if hidden.shape[0] != targets.shape[0] or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden, weight, and target shapes do not agree")
    if hidden.shape[1] < 1 or weight.shape[0] < 1:
        raise ValueError("feature and vocabulary dimensions must be positive")
    if any(value.layout != torch.strided for value in (hidden, weight, targets)):
        raise ValueError("linear cross entropy requires strided dense tensors")
    if hidden.dtype not in (torch.float32, torch.bfloat16) or weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("linear cross entropy supports FP32/BF16 hidden and weight tensors")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must use int32 or int64 with -100 as the ignore index")
    if hidden.device != weight.device or hidden.device != targets.device:
        raise ValueError("hidden, weight, and targets must be on the same device")
    if hidden.device.type != "cuda":
        raise ValueError("linear cross entropy requires CUDA tensors; use an explicit Torch loss on CPU")
    if hidden.dtype != weight.dtype and not torch.is_autocast_enabled("cuda"):
        raise ValueError("mixed hidden/weight dtypes require active CUDA autocast")
    try:
        from pisa.ops.linear_cross_entropy.lce_triton import LinearCrossEntropy
    except ImportError as error:
        raise RuntimeError("CUDA linear cross entropy requires Triton") from error
    return LinearCrossEntropy.apply(hidden, weight, targets)

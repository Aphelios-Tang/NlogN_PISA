"""CUDA linear cross entropy for already shifted targets."""
from .lce import linear_cross_entropy

linear_cross_entropy_fn = linear_cross_entropy
__all__ = ["linear_cross_entropy_fn", "linear_cross_entropy"]

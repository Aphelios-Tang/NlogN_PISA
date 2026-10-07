"""Reusable block sparse attention. GPU dependencies are loaded on demand."""

from .ops import linear_cross_entropy, select_blocks, sparse_attention

__version__ = "0.1.0"
__all__ = ["select_blocks", "sparse_attention", "linear_cross_entropy"]

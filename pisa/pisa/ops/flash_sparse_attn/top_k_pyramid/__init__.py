"""PISA first-order, second-order and log-sum-exp block selection."""
from .top_k_pyramid import select_blocks

top_k_pyramid_fn = select_blocks
__all__ = ["top_k_pyramid_fn", "select_blocks"]

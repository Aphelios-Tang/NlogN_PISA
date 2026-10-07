"""Selected-block attention with a shared softmax."""
from .sparse_attn import sparse_attention
from .sparse_attn_torch import sparse_attention_torch

sparse_attn_fn = sparse_attention
__all__ = ["sparse_attn_fn", "sparse_attention", "sparse_attention_torch"]

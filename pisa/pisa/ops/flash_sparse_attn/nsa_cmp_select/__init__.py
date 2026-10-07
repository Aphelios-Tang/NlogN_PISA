"""NSA with independent compressed and selected branches."""
from .nsa_cmp_select import native_sparse_attention

nsa_cmp_select_fn = native_sparse_attention
__all__ = ["nsa_cmp_select_fn", "native_sparse_attention"]

"""Trainable parameters for the paper's compression + selection NSA baseline."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from pisa.ops.flash_sparse_attn.nsa_cmp_select import native_sparse_attention


class NativeSparseAttention(nn.Module):
    """Learned K/V compression and independent per-query-head branch gates.

    Q/K/V projections and positional encoding belong to the surrounding model.
    Supply its hidden states for gates, positioned Q/K for selected attention,
    and unpositioned K plus ``rope_fn`` for the compressed branch. No sliding
    window branch is included in this paper baseline.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int,
        block_size: int = 64,
        topk: int = 8,
        compression_size: int = 32,
        compression_stride: int = 16,
        bias: bool = False,
        force_first: bool = True,
        force_recent: int = 2,
        query_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if min(embed_dim, num_heads, kv_heads) <= 0 or num_heads % kv_heads:
            raise ValueError("positive dimensions and num_heads divisible by kv_heads are required")
        self.block_size = block_size
        self.topk = topk
        self.compression_size = compression_size
        self.compression_stride = compression_stride
        self.force_first = force_first
        self.force_recent = force_recent
        self.query_chunk_size = query_chunk_size
        self.compress_key_weight = nn.Parameter(torch.empty(kv_heads, compression_size))
        self.compress_value_weight = nn.Parameter(torch.empty(kv_heads, compression_size))
        self.g_cmp_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        self.g_slc_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.compress_key_weight)
        nn.init.xavier_uniform_(self.compress_value_weight)
        self.g_cmp_proj.reset_parameters()
        self.g_slc_proj.reset_parameters()

    def backend_description(self, backend: str = "torch") -> dict[str, str]:
        if backend in {"triton", "auto"}:
            return {
                "compression": "public_nsa_triton_prefill_or_torch",
                "compressed_attention": "public_nsa_triton_prefill_or_torch_sdpa",
                "routing": "public_nsa_triton_prefill_or_torch_chunked",
                "selected": backend,
            }
        return {
            "compression": "torch",
            "compressed_attention": "torch_sdpa",
            "routing": "torch_chunked",
            "selected": backend,
        }

    def forward(
        self,
        hidden_states: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        compression_k: Tensor | None = None,
        rope_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        rope_position_mode: str = "document",
        cu_seqlens: Tensor | None = None,
        query_offset: int | Tensor | None = None,
        scale: float | None = None,
        backend: str = "torch",
        return_indices: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if hidden_states.shape[:2] != q.shape[:2]:
            raise ValueError("hidden states and queries must share batch and token dimensions")
        return native_sparse_attention(
            q,
            k,
            v,
            self.compress_key_weight,
            self.compress_value_weight,
            compressed_gate=self.g_cmp_proj(hidden_states).sigmoid(),
            selected_gate=self.g_slc_proj(hidden_states).sigmoid(),
            block_size=self.block_size,
            topk=self.topk,
            compression_size=self.compression_size,
            compression_stride=self.compression_stride,
            compression_k=compression_k,
            rope_fn=rope_fn,
            rope_position_mode=rope_position_mode,
            cu_seqlens=cu_seqlens,
            query_offset=query_offset,
            scale=scale,
            force_first=self.force_first,
            force_recent=self.force_recent,
            query_chunk_size=self.query_chunk_size,
            backend=backend,
            return_indices=return_indices,
        )

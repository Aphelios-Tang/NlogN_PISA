import torch
from torch import nn
import torch.nn.functional as F
from xmixers.modules.pes.lrpe import RotaryEmbedding

def token_positions(length, device, cu_seqlens=None, offset=0):
    positions = torch.arange(length, device=device)[None] + offset
    if cu_seqlens is not None:
        for begin, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            positions[:, begin:end] -= begin
    return positions

def full_attention(q, k, v, cu_seqlens=None, query_offset=0):
    """Public FlashAttention GQA, with SDPA for other dtypes and alignments."""
    from pisa.utils.validation import sequence_slices, validate_qkv

    validate_qkv(q, k, v)
    flash_attention = flash_attention_varlen = None
    if (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and 0 < q.shape[-1] <= 256
        and q.shape[-1] % 8 == 0
        and v.shape[-1] == q.shape[-1]
    ):
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except (ImportError, OSError):
            pass
        else:
            flash_attention = flash_attn_func
            flash_attention_varlen = flash_attn_varlen_func
    if cu_seqlens is not None:
        if q.shape[0] != 1 or q.shape[1] != k.shape[1] or query_offset != 0:
            raise ValueError("Packed full attention requires a complete prefill")
        segments = sequence_slices(cu_seqlens, k.shape[1])
        if flash_attention_varlen is not None:
            boundaries = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
            max_length = max(end - start for start, end in segments)
            return flash_attention_varlen(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=boundaries, cu_seqlens_k=boundaries,
                max_seqlen_q=max_length, max_seqlen_k=max_length, causal=True,
            ).unsqueeze(0)
        parts = [
            full_attention(q[:, a:b], k[:, a:b], v[:, a:b])
            for a, b in segments
        ]
        return torch.cat(parts, dim=1)
    # FlashAttention aligns unequal lengths to the bottom right. An arbitrary
    # explicit query offset needs the SDPA mask below instead.
    if flash_attention is not None and query_offset == k.shape[1] - q.shape[1]:
        return flash_attention(q, k, v, causal=True)
    keys = k.transpose(1, 2)
    values = v.transpose(1, 2)
    queries = q.transpose(1, 2)
    enable_gqa = q.shape[2] != k.shape[2]
    if query_offset == 0 and q.shape[1] == k.shape[1]:
        result = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True, enable_gqa=enable_gqa,
        )
    else:
        query_positions = torch.arange(q.shape[1], device=q.device) + query_offset
        key_positions = torch.arange(k.shape[1], device=q.device)
        causal = key_positions[None] <= query_positions[:, None]
        result = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=causal, enable_gqa=enable_gqa,
        )
    return result.transpose(1, 2)

class PyramidSparseAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=config.bias)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=config.bias)
        self.rope = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.rope_fraction
        )
        if config.attention_method == "nsa":
            from .native_sparse_attention import NativeSparseAttention

            self.nsa = NativeSparseAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                kv_heads=config.num_key_value_heads,
                block_size=config.block_size,
                topk=config.topk,
                compression_size=config.nsa_compression_size,
                compression_stride=config.nsa_compression_stride,
                bias=config.bias,
                force_first=config.force_first,
                force_recent=config.force_recent,
            )

    def forward(self, x, cu_seqlens=None, cache=None, use_cache=False, backend=None,
                attention_mask=None, past_key_values=None, **kwargs):
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("Remove evaluation padding before native attention")
        if past_key_values is not None:
            cache = past_key_values[self.layer_idx] if len(past_key_values) > self.layer_idx else None
        config = self.config
        batch, length, _ = x.shape
        q = self.q_proj(x).view(
            batch, length, config.num_attention_heads, config.head_dim
        )
        raw_k = self.k_proj(x).view(
            batch, length, config.num_key_value_heads, config.head_dim
        )
        v = self.v_proj(x).view(
            batch, length, config.num_key_value_heads, config.head_dim
        )
        offset = 0 if cache is None else cache["k"].shape[1]
        position_boundaries = cu_seqlens if config.rope_position_mode == "document" else None
        q_positions = token_positions(length, x.device, position_boundaries, offset)
        q = self.rope(q, q_positions)
        new_k = self.rope(raw_k, q_positions)
        pyramid = None
        next_cache = None
        if use_cache:
            from xmixers.utils.attention_cache import PyramidCache, TensorCache

            next_cache = {} if cache is None else cache
            if config.attention_method in {"pisa1", "pisa2", "pisa"}:
                pyramid = (
                    PyramidCache.from_keys(new_k, block_size=config.block_size)
                    if cache is None
                    else cache["pyramid"].append(new_k)
                )
                next_cache["pyramid"] = pyramid
                k = pyramid.keys
            else:
                keys = (
                    TensorCache.from_tensor(new_k)
                    if cache is None
                    else cache["key_storage"].append(new_k)
                )
                next_cache["key_storage"] = keys
                k = keys.tensor
            values = (
                TensorCache.from_tensor(v)
                if cache is None
                else cache["value_storage"].append(v)
            )
            next_cache.update(k=k, v=values.tensor, value_storage=values)
            v = values.tensor
            if config.attention_method == "nsa":
                raw_keys = (
                    TensorCache.from_tensor(raw_k)
                    if cache is None
                    else cache["raw_key_storage"].append(raw_k)
                )
                next_cache.update(raw_k=raw_keys.tensor, raw_key_storage=raw_keys)
                raw_k = raw_keys.tensor
        else:
            k = new_k if cache is None else torch.cat((cache["k"], new_k), dim=1)
            if cache is not None:
                v = torch.cat((cache["v"], v), dim=1)
                if config.attention_method == "nsa":
                    raw_k = torch.cat((cache["raw_k"], raw_k), dim=1)
        backend = backend or config.backend
        if config.attention_method == "full":
            output = full_attention(q, k, v, cu_seqlens, offset)
        elif config.attention_method == "nsa":
            output = self.nsa(
                x,
                q,
                k,
                v,
                compression_k=raw_k,
                rope_fn=self.rope.torch_forward,
                rope_position_mode=config.rope_position_mode,
                cu_seqlens=cu_seqlens,
                query_offset=offset,
                backend=backend,
            )
        else:
            from pisa.ops import select_blocks, sparse_attention

            routing = {}
            if config.attention_method in {"pisa1", "pisa2", "pisa"}:
                routing = {
                    "routing_schedule": (
                        "fused" if cache is not None and length == 1 else config.prefill_routing
                    ),
                    "leaf_tile": config.leaf_tile,
                }
            indices = select_blocks(
                q,
                k,
                method=config.attention_method,
                block_size=config.block_size,
                topk=config.topk,
                cu_seqlens=cu_seqlens,
                query_offset=offset,
                force_first=config.force_first,
                force_recent=config.force_recent,
                backend=backend,
                pyramid=pyramid,
                **routing,
            )
            output = sparse_attention(
                q,
                k,
                v,
                indices,
                block_size=config.block_size,
                cu_seqlens=cu_seqlens,
                query_offset=offset,
                backend=backend,
            )
        if past_key_values is not None and use_cache:
            while len(past_key_values) <= self.layer_idx:
                past_key_values.append(None)
            past_key_values[self.layer_idx] = next_cache
            next_cache = past_key_values
        return self.o_proj(output.reshape(batch, length, -1)), next_cache

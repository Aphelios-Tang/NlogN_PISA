"""Assembly of the public NSA compression kernels and FLA selected attention.

Imports stay lazy and scoped to the input CUDA device. The portable operator
owns validation and suffix/decode behavior; this adapter only handles prefill.
"""

import torch
import torch.nn.functional as F


def prefill_supported(q, k, v, block_size, topk, compression_size, compression_stride,
                      force_first, force_recent, layout):
    if not (
        q.is_cuda and q.dtype == torch.bfloat16
        and q.shape[:2] == k.shape[:2]
        and q.shape[2] == 32 and k.shape[2] == 2
        and q.shape[-1] in (64, 128) and v.shape[-1] == q.shape[-1]
        and block_size == 64 and compression_size == 32 and compression_stride == 16
        and type(topk) is int and topk in (8, 16, 32)
        and force_first is True and type(force_recent) is int and force_recent in (1, 2)
        and all(qe - qs == ke - ks and offset == 0 for _, qs, qe, ks, ke, offset in layout)
        and max(ke - ks for _, _, _, ks, ke, _ in layout) >= compression_size
    ):
        return False
    # The D128 launch needs this much opt-in shared memory.
    # Keep smaller devices on the portable path.
    if q.shape[-1] == 128:
        properties = torch.cuda.get_device_properties(q.device)
        if properties.shared_memory_per_block_optin < 163840:
            return False
    return True


@torch.compiler.disable
def prefill(q, k, v, key_weight, value_weight, *, compression_k, rope_fn,
            rope_position_mode, cu_seqlens, compressed_gate, selected_gate,
            block_size, topk, compression_size, compression_stride, scale,
            force_recent, layout):
    """Run each original packed kernel once, preserving its reduction order.

    Data-dependent compressed lengths remain outside Dynamo. FP32 master
    compression parameters are explicitly cast to the activation dtype;
    autograd propagates their gradients through this cast to the master leaves.
    """
    batch, length = q.shape[:2]
    for weight in (key_weight, value_weight):
        if weight.shape != (k.shape[2], compression_size):
            raise ValueError("compression weight does not match heads/window size")
        if weight.device != q.device:
            raise ValueError("input and compression weight must share a device")
    with torch.cuda.device(q.device):
        from .compressed_attention_triton import compressed_attention
        from .weighted_pool_triton import weightedpool_compress
        from pisa.ops.flash_sparse_attn.sparse_attn.sparse_attn_fla import selected_attention

        if cu_seqlens is None:
            packed_cu = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * length
        else:
            packed_cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        packed_q = q.reshape(-1, *q.shape[2:]).contiguous()
        packed_k = compression_k.reshape(-1, *compression_k.shape[2:]).contiguous()
        packed_v = v.reshape(-1, *v.shape[2:]).contiguous()
        pooled_k, compressed_cu = weightedpool_compress(
            packed_k, key_weight.to(q.dtype).contiguous(), packed_cu,
            compression_size, compression_stride, pe=None,
        )
        compressed_lengths = [max(0, (ke - ks - compression_size) // compression_stride + 1)
                              for _, _, _, ks, ke, _ in layout]
        if rope_fn is not None:
            positions = torch.cat([
                torch.arange(count, device=q.device) * compression_stride
                + (entry[3] if rope_position_mode == "sequence" else 0)
                for entry, count in zip(layout, compressed_lengths)
            ])[None]
            pooled_k = rope_fn(pooled_k[None], positions)[0]
        pooled_v, _ = weightedpool_compress(
            packed_v, value_weight.to(q.dtype).contiguous(), packed_cu,
            compression_size, compression_stride, pe=None,
        )
        compressed, indices = compressed_attention(
            packed_q, pooled_k.contiguous(), pooled_v.contiguous(),
            compression_size, compression_stride, block_size, topk,
            packed_cu, compressed_cu,
            max(ke - ks for _, _, _, ks, ke, _ in layout), max(compressed_lengths),
            scale, 1, force_recent,
        )
        indices = indices.permute(1, 0, 2).contiguous()
        if indices.shape[-1] < topk:
            indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
        indices = indices.reshape(batch, length, k.shape[2], topk)
        compressed = compressed.reshape(batch, length, *q.shape[2:])
        positions = torch.arange(length, device=q.device)[None].expand(batch, -1)
        selected = selected_attention(
            q, k, v, indices, block_size, scale, positions, cu_seqlens=cu_seqlens,
        )
        if selected_gate is not None:
            selected = selected * selected_gate[..., None]
        output = (torch.addcmul(selected, compressed, compressed_gate[..., None])
                  if compressed_gate is not None else selected + compressed)
        return output, indices

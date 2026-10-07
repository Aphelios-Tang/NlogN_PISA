"""Geometrically growing token storage and incremental pyramid summaries."""

import torch

from pisa.ops.compress.mean_pooling import mean_pool


class TensorCache:
    """Append-only, detached [B,T,H,D] storage for inference."""

    def __init__(self):
        self.buffer = None
        self.length = 0

    @classmethod
    def from_tensor(cls, tensor):
        return cls().append(tensor)

    @property
    def tensor(self):
        if self.buffer is None:
            raise ValueError("the token cache is empty")
        return self.buffer[:, : self.length]

    @torch.no_grad()
    def append(self, tensor):
        if tensor.ndim != 4 or tensor.shape[1] == 0 or not tensor.is_floating_point():
            raise ValueError("new tokens must be a nonempty [B,T,H,D] floating tensor")
        if self.buffer is not None:
            old = self.buffer
            if (old.shape[0], *old.shape[2:], old.dtype, old.device) != (
                tensor.shape[0],
                *tensor.shape[2:],
                tensor.dtype,
                tensor.device,
            ):
                raise ValueError("cache batch, heads, dimension, dtype and device cannot change")
        end = self.length + tensor.shape[1]
        if self.buffer is None or end > self.buffer.shape[1]:
            capacity = 1 << (end - 1).bit_length()
            grown = tensor.new_empty(tensor.shape[0], capacity, *tensor.shape[2:])
            if self.buffer is not None:
                grown[:, : self.length].copy_(self.tensor)
            self.buffer = grown
        self.buffer[:, self.length : end].copy_(tensor)
        self.length = end
        return self


class PyramidCache:
    """Cache means in one allocation, updating only changed leaves and ancestors.

    Layer offsets change only when the leaf capacity doubles. Valid views may
    have a batch stride larger than their length; kernels must honor strides.
    """

    def __init__(self, block_size=64):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.tokens = TensorCache()
        self.storage = None
        self.capacities = ()
        self.starts = ()
        self.sizes = []
        self.offsets = None
        self.lengths = None

    @classmethod
    def from_keys(cls, keys, block_size=64):
        return cls(block_size).append(keys)

    @property
    def length(self):
        return self.tokens.length

    @property
    def key_buffer(self):
        return self.tokens.buffer

    @property
    def keys(self):
        return self.tokens.tensor

    @property
    def level_sizes(self):
        return tuple(self.sizes)

    @property
    def buffers(self):
        if self.storage is None:
            return []
        return [
            self.storage[:, start : start + capacity]
            for start, capacity in zip(self.starts, self.capacities)
        ]

    @property
    def levels(self):
        return [buffer[:, :size] for buffer, size in zip(self.buffers, self.sizes)]

    def reserve(self, reference, leaves):
        if self.capacities and leaves <= self.capacities[0]:
            return
        capacity = 1 << (leaves - 1).bit_length()
        capacities, starts, total = [], [], 0
        while capacity:
            capacities.append(capacity)
            starts.append(total)
            total += capacity
            capacity //= 2
        storage = reference.new_empty(reference.shape[0], total, *reference.shape[2:])
        for start, previous in zip(starts, self.levels):
            storage[:, start : start + previous.shape[1]].copy_(previous)
        self.storage = storage
        self.capacities, self.starts = tuple(capacities), tuple(starts)
        self.offsets = torch.tensor(starts, device=reference.device, dtype=torch.int32)
        self.lengths = torch.empty(len(starts), device=reference.device, dtype=torch.int32)

    @torch.no_grad()
    def append(self, keys):
        previous = self.length
        self.tokens.append(keys)
        leaves = (self.length + self.block_size - 1) // self.block_size
        self.reserve(keys, leaves)
        changed = previous // self.block_size
        source, width = self.keys, self.block_size
        sizes = []
        for buffer in self.buffers:
            count = (source.shape[1] + width - 1) // width
            buffer[:, changed:count].copy_(mean_pool(
                source[:, changed * width :], width, fp32_accumulation=previous > 0
            ))
            sizes.append(count)
            source = buffer[:, :count]
            changed //= 2
            width = 2
        self.sizes = sizes
        self.lengths.copy_(torch.tensor(sizes, device=keys.device, dtype=torch.int32))
        return self

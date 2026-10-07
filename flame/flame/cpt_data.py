"""Prepared CPT tokens with explicit document boundaries and one-pass sampling.

The on-disk format is a numeric Zarr array [rows, 65536] and CSR document ends
in document-boundaries/{split}-{row-offsets,segment-ends}.npy, described by
{split}-manifest.json. Document boundaries are read from these arrays, not EOS tokens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from flame.data import DPAwareDataLoader


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_tokens(path):
    import zarr

    storage = zarr.open(str(path), mode="r")
    if isinstance(storage, zarr.Array):
        return storage
    names = list(storage.array_keys())
    if "data" in names:
        return storage["data"]
    if len(names) == 1:
        return storage[names[0]]
    raise ValueError("CPT Zarr group must contain data or a single token array")


class PackedCPTDataset(Dataset):
    def __init__(self, data_root, seq_len, vocab_size, split="train",
                 document_start=0, document_count=None):
        self.root = Path(data_root).expanduser().resolve()
        self.split = {"val": "validation", "heldout": "validation"}.get(split, split)
        if self.split not in ("train", "validation"):
            raise ValueError("CPT split must be train or validation")
        if type(seq_len) is not int or seq_len <= 0 or 65536 % seq_len:
            raise ValueError("CPT seq_len must be a positive divisor of 65536")
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("CPT vocab_size must be positive")
        self.seq_len, self.vocab_size = seq_len, vocab_size
        self.slices_per_row = 65536 // seq_len
        self.path = self.root / ("train.zarr" if self.split == "train" else "heldout.zarr")
        if not self.path.exists():
            config = json.loads((self.root / "dataset-config.json").read_text())
            self.path = (self.root / config["splits"][self.split]).resolve()
        if not self.path.resolve().is_relative_to(self.root):
            raise ValueError("CPT split path escapes data root")
        array = open_tokens(self.path)
        if len(array.shape) != 2 or array.shape[1] != 65536:
            raise ValueError("CPT token array must have shape [rows, 65536]")
        if np.dtype(array.dtype).kind not in "iuf":
            raise ValueError("CPT token storage must be numeric")
        rows = int(array.shape[0])
        if document_count is None:
            document_count = rows - document_start
        if (type(document_start) is not int or type(document_count) is not int
                or document_start < 0 or document_count <= 0 or document_start + document_count > rows):
            raise ValueError("CPT document window is outside the token array")
        self.document_start, self.document_count = document_start, document_count
        directory = self.root / "document-boundaries"
        self.offsets_path = directory / f"{self.split}-row-offsets.npy"
        self.ends_path = directory / f"{self.split}-segment-ends.npy"
        manifest_path = directory / f"{self.split}-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.offsets = np.load(self.offsets_path, mmap_mode="r", allow_pickle=False)
        self.ends = np.load(self.ends_path, mmap_mode="r", allow_pickle=False)
        expected = {"format_version": 1, "split": self.split, "rows": rows,
                    "row_tokens": 65536, "segments": len(self.ends)}
        if any(type(manifest.get(key)) is not type(value) or manifest[key] != value
               for key, value in expected.items()):
            raise ValueError("CPT boundary manifest does not match the token array")
        if self.offsets.dtype != np.dtype("<i8") or self.ends.dtype != np.dtype("<i4"):
            raise ValueError("CPT boundaries require int64 offsets and int32 ends")
        if (self.offsets.shape != (rows + 1,) or self.ends.ndim != 1
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.ends)
                or np.any(np.diff(self.offsets) <= 0)):
            raise ValueError("CPT boundary offsets must cover every nonempty row")
        metadata = self.path / array.path / ".zarray"
        if not metadata.is_file():
            raise ValueError("CPT requires prepared Zarr v2 array metadata")
        self.identity = {"root": str(self.root), "split": self.split,
                         "document_start": document_start, "document_count": document_count,
                         "seq_len": seq_len, "vocab_size": vocab_size,
                         "files": {str(path.relative_to(self.root)): file_sha(path)
                                   for path in (manifest_path, self.offsets_path, self.ends_path, metadata)}}
        self.tokens = None

    def __len__(self):
        return self.document_count * self.slices_per_row

    def __getstate__(self):
        values = self.__dict__.copy()
        values.update(tokens=None, offsets=None, ends=None)
        return values

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row, part = divmod(index, self.slices_per_row)
        row += self.document_start
        begin = part * self.seq_len
        if self.tokens is None:
            self.tokens = open_tokens(self.path)
        if self.offsets is None:
            self.offsets = np.load(self.offsets_path, mmap_mode="r", allow_pickle=False)
            self.ends = np.load(self.ends_path, mmap_mode="r", allow_pickle=False)
        boundaries = self.ends[int(self.offsets[row]):int(self.offsets[row + 1])]
        if (not len(boundaries) or boundaries[0] <= 0 or boundaries[-1] != 65536
                or np.any(np.diff(boundaries) <= 0)):
            raise ValueError(f"CPT row {row} has invalid document boundaries")
        first = np.searchsorted(boundaries, begin, side="right")
        last = np.searchsorted(boundaries, begin + self.seq_len, side="left")
        local = [0, *(boundaries[first:last].astype(np.int64) - begin).tolist(), self.seq_len]
        values = np.asarray(self.tokens[row, begin:begin + self.seq_len])
        if (values.shape != (self.seq_len,) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values >= self.vocab_size)
                or (values.dtype.kind == "f" and np.any(values != np.floor(values)))):
            raise ValueError(f"CPT row {row} contains invalid token IDs")
        return {"input_ids": torch.tensor(values, dtype=torch.long),
                "cu_seqlens": torch.tensor(local, dtype=torch.int32)}


class CPTCollator:
    def __init__(self, max_tokens):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("CPT max_tokens must be positive")
        self.max_tokens = max_tokens

    def __call__(self, examples):
        chunks, offsets = [], [0]
        remaining = self.max_tokens
        for example in examples:
            tokens, bounds = example["input_ids"], example["cu_seqlens"].tolist()
            if (tokens.ndim != 1 or len(bounds) < 2 or bounds[0] != 0
                    or bounds[-1] != tokens.numel() or any(a >= b for a, b in zip(bounds, bounds[1:]))):
                raise ValueError("CPT example boundaries must cover its 1D tokens")
            for begin, end in zip(bounds, bounds[1:]):
                length = min(end - begin, remaining)
                if length > 1:
                    chunks.append(tokens[begin:begin + length])
                    offsets.append(offsets[-1] + length)
                    remaining -= length
                if remaining < 2:
                    break
        if not chunks:
            raise ValueError("CPT batch must contain a segment longer than one token")
        ids = torch.cat(chunks).to(dtype=torch.long).unsqueeze(0)
        labels = ids.clone()
        labels[0, offsets[:-1]] = -100
        return {"input_ids": ids, "labels": labels,
                "cu_seqlens": torch.tensor(offsets, dtype=torch.int32)}


class CPTDataLoader(DPAwareDataLoader):
    def __init__(self, *, identity, **kwargs):
        self.identity = identity
        super().__init__(**kwargs)

    def state_dict(self):
        return {f"rank_{self.rank}": pickle.dumps({"identity": self.identity,
                                                 "loader": super().state_dict()})}

    def load_state_dict(self, state_dict):
        key = f"rank_{self.rank}"
        if key not in state_dict:
            raise ValueError(f"CPT checkpoint is missing dataloader {key}")
        saved = pickle.loads(state_dict[key])
        if saved["identity"] != self.identity:
            raise ValueError("CPT dataloader resume identity changed")
        super().load_state_dict(saved["loader"])


def build_cpt_dataloader(*, data_root, vocab_size, rank, world_size, batch_size,
                         seq_len, gradient_accumulation_steps, steps, split="train",
                         document_start=0, document_count=None, allow_partial_data=False,
                         seed=42, num_workers=0, pin_memory=False, persistent_workers=False,
                         prefetch_factor=2, snapshot_every_n_steps=1):
    if any(type(value) is not int or value <= 0
           for value in (world_size, batch_size, gradient_accumulation_steps, steps)):
        raise ValueError("CPT budget dimensions must be positive integers")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("CPT rank must be inside world_size")
    dataset = PackedCPTDataset(data_root, seq_len, vocab_size, split, document_start, document_count)
    planned = world_size * batch_size * gradient_accumulation_steps * steps
    if planned > len(dataset) or (planned != len(dataset) and not allow_partial_data):
        raise ValueError(f"CPT budget consumes {planned} sequences from a pool of {len(dataset)}; "
                         "using fewer sequences requires allow_partial_data")
    persistent = persistent_workers if num_workers else False
    prefetch = prefetch_factor if num_workers else None
    identity = {"format_version": 1, "data": dataset.identity, "rank": rank,
                "world_size": world_size, "batch_size": batch_size,
                "accumulation": gradient_accumulation_steps, "steps": steps,
                "seed": seed, "num_workers": num_workers,
                "persistent_workers": persistent, "prefetch_factor": prefetch,
                "ordering": "rank-strided-single-pass"}
    loader = CPTDataLoader(identity=identity, rank=rank,
        dataset=Subset(dataset, range(rank, planned, world_size)), batch_size=batch_size,
        collate_fn=CPTCollator(batch_size * seq_len), num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent, prefetch_factor=prefetch,
        snapshot_every_n_steps=snapshot_every_n_steps)
    loader.generator = torch.Generator().manual_seed(seed)
    return loader

"""Compatibility helpers; the active trainer uses registered AutoModel classes."""
import json
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM
from xmixers import LLaMAConfig


def load_model_config(path, tokenizer_vocab_size, stage="pt"):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("PISA model JSON must contain an object")
    values = dict(values.get("model", values))
    if values.pop("model_type", "llama_") not in ("llama_", "pisa"):
        raise ValueError("expected an xmixers LLaMA configuration")
    if stage not in ("pt", "cpt"):
        raise ValueError("training.stage must be pt or cpt")
    values["vocab_size"] = tokenizer_vocab_size
    padded = values.get("padded_vocab_size")
    if padded is None or padded < tokenizer_vocab_size:
        values["padded_vocab_size"] = (tokenizer_vocab_size + 255) // 256 * 256
    # The shared model performs the only label shift. PT retains transition
    # targets even with document attention; CPT masks them after that shift.
    values["mask_document_targets"] = stage == "cpt"
    return LLaMAConfig(**values)


def build_model(config, device="meta"):
    with torch.device(device):
        return AutoModelForCausalLM.from_config(config)


def validate_parallelism(parallel_dims):
    for name in ("tp", "pp", "cp"):
        if getattr(parallel_dims, name) != 1:
            raise ValueError(f"PISA Flame does not support {name.upper()} degree other than 1")

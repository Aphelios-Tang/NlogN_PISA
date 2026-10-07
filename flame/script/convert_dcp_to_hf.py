"""Convert a trusted local DCP step to Transformers format."""

import argparse
from pathlib import Path
import os
import tempfile

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
import xmixers  # noqa: F401; register model classes
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def save_pretrained(checkpoint, path, config, tokenizer, model_max_length=2048, stage="pt"):
    print(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    tokenizer.model_max_length = model_max_length
    # Preserve the saved vocabulary, padding and training stage.
    saved_config = Path(checkpoint) / "config.json"
    if saved_config.is_file():
        model_config = AutoConfig.from_pretrained(str(saved_config))
        if model_config.vocab_size != len(tokenizer):
            raise ValueError("Tokenizer vocabulary does not match the saved Flame model")
    else:
        model_config = AutoConfig.from_pretrained(config)
        model_config.vocab_size = len(tokenizer)
        if hasattr(model_config, "padded_vocab_size"):
            model_config.padded_vocab_size = max(
                model_config.padded_vocab_size, (len(tokenizer) + 255) // 256 * 256
            )
        if hasattr(model_config, "mask_document_targets"):
            model_config.mask_document_targets = stage == "cpt"
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        print(f"Reading the distributed checkpoint from {checkpoint}")
        dcp_to_torch_save(checkpoint, checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(model_config)
        model.load_state_dict(state, strict=True, assign=True)
    model.config.max_position_embeddings = model_max_length
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config = GenerationConfig.from_model_config(model.config)
    print(f"Saving the model and tokenizer to {path}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--stage", choices=("pt", "cpt"), default="pt")
    args = parser.parse_args()
    save_pretrained(args.checkpoint, args.path, args.config, args.tokenizer, args.model_max_length, args.stage)

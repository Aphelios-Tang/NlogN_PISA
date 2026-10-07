"""Architecture and operator settings for the paper's decoder models."""

from dataclasses import asdict, dataclass
from typing import ClassVar


@dataclass
class PisaConfig:
    model_type: ClassVar[str] = "pisa"
    vocab_size: int = 50257
    padded_vocab_size: int | None = None
    pad_token_id: int | None = None
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 64
    attention_method: str = "pisa"
    block_size: int = 64
    topk: int = 8
    prefill_routing: str | None = None
    leaf_tile: int = 4
    force_first: bool = True
    force_recent: int = 2
    rope_theta: float = 10000.0
    rope_fraction: float = 0.5
    rope_position_mode: str = "document"
    rms_norm_eps: float = 1e-5
    rms_norm_num_warps: int | None = None
    bias: bool = False
    tie_word_embeddings: bool = False
    init_std: float = 0.02
    attention_init_gain: float = 0.01
    zero_init_output: bool = True
    nsa_compression_size: int = 32
    nsa_compression_stride: int = 16
    mask_document_targets: bool = True
    loss_chunk_size: int = 1024
    loss_backend: str = "torch"
    gradient_checkpointing: bool = False
    backend: str = "auto"

    def __post_init__(self):
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.block_size,
            self.topk,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Model dimensions and block budgets must be positive")
        if self.padded_vocab_size is None:
            self.padded_vocab_size = (self.vocab_size + 255) // 256 * 256
        if self.padded_vocab_size < self.vocab_size:
            raise ValueError("padded_vocab_size must cover the tokenizer vocabulary")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be a tokenizer vocabulary index or None")
        if self.rope_position_mode not in {"document", "sequence"}:
            raise ValueError("rope_position_mode must be document or sequence")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        if self.head_dim % 2 or not 0 <= self.rope_fraction <= 1:
            raise ValueError(
                "RoPE needs an even head dimension and a fraction in [0, 1]"
            )
        if self.attention_method not in {
            "full",
            "bsa",
            "pisa1",
            "pisa2",
            "pisa",
            "nsa",
        }:
            raise ValueError(f"Unknown attention method: {self.attention_method}")
        if self.prefill_routing is None:
            self.prefill_routing = "fused" if self.attention_method == "pisa1" else "grouped"
        if self.prefill_routing not in {"grouped", "fused"}:
            raise ValueError("prefill_routing must be grouped or fused")
        if type(self.leaf_tile) is not int or self.leaf_tile not in {1, 2, 4}:
            raise ValueError("leaf_tile must be 1, 2 or 4")
        if not isinstance(self.mask_document_targets, bool):
            raise ValueError("mask_document_targets must be a boolean")
        if self.loss_chunk_size < 1 or self.loss_backend not in {"torch", "fla", "pisa"}:
            raise ValueError("Invalid cross-entropy configuration")
        if self.rms_norm_num_warps is not None and (
            type(self.rms_norm_num_warps) is not int
            or self.rms_norm_num_warps not in (1, 2, 4, 8, 16, 32)
        ):
            raise ValueError("rms_norm_num_warps must be None or one of 1, 2, 4, 8, 16, 32")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("RoPE base and normalization epsilon must be positive")

    @property
    def num_heads(self):
        return self.num_attention_heads

    @property
    def embed_dim(self):
        return self.hidden_size

    def to_dict(self):
        return asdict(self)

    def operator_settings(self):
        """Settings that determine attention and loss, independent of training stage."""
        names = (
            "attention_method",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "block_size",
            "topk",
            "force_first",
            "force_recent",
            "rope_theta",
            "rope_fraction",
            "rope_position_mode",
            "rms_norm_eps",
            "rms_norm_num_warps",
            "backend",
            "loss_backend",
        )
        if self.attention_method == "nsa":
            names += ("nsa_compression_size", "nsa_compression_stride")
        if self.attention_method in {"pisa1", "pisa2", "pisa"}:
            names += ("prefill_routing", "leaf_tile")
        return {name: getattr(self, name) for name in names}

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

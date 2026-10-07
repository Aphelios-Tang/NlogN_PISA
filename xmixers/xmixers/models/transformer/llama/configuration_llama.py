# coding=utf-8
""" LLaMA configuration"""

from transformers.configuration_utils import PretrainedConfig
from xmixers.pisa_config import PisaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LLaMAConfig(PretrainedConfig):
    model_type = "llama_"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "embed_dim", "intermediate_size": "mid_dim",
                     "num_attention_heads": "num_heads", "num_key_value_heads": "kv_heads",
                     "rope_theta": "base", "attention_method": "token_mixer_type"}

    def __init__(
        self,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=64000,
        use_cache=True,
        init_std=0.02,
        tie_word_embeddings=False,
        ########## model config
        ##### token mixer config
        token_mixer_type="attn",
        embed_dim=1024,
        num_heads=8,
        kv_heads=-1,
        bias=False,
        use_lrpe=True,
        lrpe_type=1,
        base=10000,
        mpa_type=0,
        mpa_activation="sigmoid",
        use_l2_norm=False,
        gate_type=0,
        head_dim=-1,
        q_rank=8,
        kv_rank=2,
        cp_activation="none",
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        window_size=-1,
        block_size=64,
        chunk_size=128,
        token_mixer_top_k=2,
        share_kv=False,
        poly_order=4,
        poly_type=1,
        window_head_dim=128,
        use_decay=False,
        use_kernel_regression=True,
        scale_type=0,
        threshold=0.99,
        token_mixer_norm_type="grouprmsnorm",
        use_qk_norm=False,
        use_beta=True,
        use_w=False,
        num_hope_groups=1,
        use_offset=False,
        share_w=False,
        hope_ratio=1,
        rope_ratio=0,
        use_w_hope=True,
        use_static_w=False,
        attn_use_decay=False,
        use_a_or_b: bool = True,
        share_ab: bool = False,
        norm_ab: bool = False,
        state_offset: int = -1,
        use_initial_state: bool = False,
        use_identity: bool = False,
        cutoff_len: int = 1024,
        rope_pos: str = "pre",
        l2_norm_pos: str = "pre",
        corpe_type: int = 1,
        decay_type: str = "di",
        neg_decay_type: int = 0,  # 0 for no neg decay, 1 for cumprod neg decay, 2 for no cumprod neg decay
        ssm_type: int = 0,
        pos_type: int = 0,
        func_type: str = "default",
        theta_type: int = 0,
        talking_head_group_size: int = -1,
        ###### channel mixer config
        channel_mixer_type="glu",
        mid_dim=1024,
        channel_mixer_activation="silu",
        use_gate_linear=True,
        # for alu and lalu
        qk_dim=1024,
        v_dim=1024,
        mem_dim=1024,
        use_scale=0,
        use_output_gate=False,
        output_gate_activation="silu",
        use_low_rank_output_gate=False,
        channel_mixer_init_type=0,
        ##### others
        max_position_embeddings=1024,
        num_hidden_layers=24,
        norm_type="rmsnorm",
        use_postnorm=False,
        use_embed_scale=False,
        ce_type="pisa",
        fuse_norm_add=False,
        num_bins=128,
        center=False,
        use_proj=True,
        share_proj=True,
        ##### init
        init_type=1,
        token_mixer_init_type=4,
        rescale_type=2,
        gain=0.01,
        pad_embed_dim=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        for key, value in locals().items():
            if key not in [
                "self",
                "kwargs",
                "__class__",
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "tie_word_embeddings",
            ]:
                setattr(self, key, value)

        options = dict(kwargs)
        aliases = {"hidden_size": embed_dim, "intermediate_size": mid_dim,
                   "num_attention_heads": num_heads,
                   "num_key_value_heads": num_heads if kv_heads == -1 else kv_heads,
                   "head_dim": embed_dim // num_heads if head_dim == -1 else head_dim,
                   "attention_method": "full" if token_mixer_type == "attn" else token_mixer_type,
                   "rope_theta": base, "rope_fraction": 1.0 if lrpe_type == 1 else .5,
                   "init_std": init_std, "attention_init_gain": gain,
                   "zero_init_output": rescale_type == 2, "bias": bias,
                   "tie_word_embeddings": tie_word_embeddings, "vocab_size": vocab_size,
                   "pad_token_id": pad_token_id, "num_hidden_layers": num_hidden_layers,
                   "block_size": block_size,
                   "loss_backend": {"naive": "torch", "pisa": "pisa", "fla_fce": "fla"}.get(ce_type, "pisa")}
        aliases.update({key: value for key, value in options.items() if key in PisaConfig.__dataclass_fields__ and key != "model_type"})
        if "block_counts" in options and "topk" not in options:
            aliases["topk"] = options["block_counts"]
        validated = PisaConfig(**aliases)
        for key, value in validated.to_dict().items():
            setattr(self, key, value)
        self.embed_dim, self.mid_dim = self.hidden_size, self.intermediate_size
        self.num_heads, self.kv_heads = self.num_attention_heads, self.num_key_value_heads
        self.base, self.token_mixer_type = self.rope_theta, self.attention_method
        self.ce_type = self.loss_backend

    @classmethod
    def from_pisa_config(cls, config):
        return cls(**config.to_dict(), use_cache=False)

    def to_dict(self):
        result = super().to_dict()
        for name in PisaConfig.__dataclass_fields__:
            if name != "model_type" and hasattr(self, name):
                result[name] = getattr(self, name)
        return result

    def operator_settings(self):
        return PisaConfig(**{key: getattr(self, key) for key in PisaConfig.__dataclass_fields__ if key != "model_type"}).operator_settings()

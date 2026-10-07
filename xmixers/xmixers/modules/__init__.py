from .normalizations.rms_norm import RMSNorm
from .channel_mixers.glu import GLU
from .token_mixers.vanilla_attention.pyramid_sparse_attention import PyramidSparseAttention

def get_token_mixer(config, layer_idx=0):
    return PyramidSparseAttention(config, layer_idx)

def get_channel_mixer(config):
    if config.channel_mixer_type != "glu":
        raise ValueError("The paper model uses the public GLU channel mixer")
    return GLU(config.embed_dim, config.mid_dim, config.channel_mixer_activation,
               bias=config.bias, use_gate_linear=config.use_gate_linear)

def get_norm_fn(norm_type):
    if norm_type != "rmsnorm":
        raise ValueError("The paper model uses RMSNorm")
    return RMSNorm

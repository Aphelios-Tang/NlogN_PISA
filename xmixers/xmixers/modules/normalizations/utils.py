from torch import nn
from pisa.ops.normalize.normalize import rms_norm_fn

class NormOp(nn.Module):
    def __init__(self, norm_type, num_warps=None):
        super().__init__()
        if norm_type != "rmsnorm":
            raise ValueError("This LLaMA package provides RMSNorm")
        self.norm_type = norm_type
        self.num_warps = num_warps

    def forward(self, x, weight=None, bias=None, residual=None, dim=1, eps=1e-5,
                use_mean=False, num_groups=1, return_residual=False, gate=None,
                gate_act="sigmoid", gate_pos="pre"):
        if gate is not None:
            raise ValueError("The paper RMSNorm has no gate")
        value = x if residual is None else x + residual
        output = rms_norm_fn(value, weight, eps, num_warps=self.num_warps)
        return (output, value) if return_residual or residual is not None else output

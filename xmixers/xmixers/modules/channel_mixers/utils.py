import torch.nn as nn
import torch.nn.functional as F
from xmixers.modules.activations import get_activation_fn
from pisa.ops.gate_linear import silu_gate_linear

class GateLinearOp(nn.Module):
    def forward(self, x1, x2, weight, bias, act, residual=None):
        output = (silu_gate_linear(x1, x2, weight, bias) if act == "silu"
                  else F.linear(get_activation_fn(act)(x1) * x2, weight, bias))
        return output if residual is None else output + residual

"""SiLU gate and down projection with recomputed first-order gradients."""
from .gate_linear_ag import SiluGateLinear, silu_gate_linear

gate_linear_fn = silu_gate_linear

__all__ = ["gate_linear_fn", "silu_gate_linear", "SiluGateLinear"]

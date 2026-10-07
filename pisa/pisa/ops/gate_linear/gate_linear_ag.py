# SPDX-License-Identifier: Apache-2.0
"""SiLU gate/down projection with recomputation in backward.

This tensor operator consumes the two linear projection outputs. Under
autocast those activations already have the compute dtype. It preserves the
first-order backward, including activation-dtype rounding, while
recomputing the gate product instead of retaining it for the weight gradient.
"""

import torch
import torch.nn.functional as F


class SiluGateLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, value, weight, bias):
        gate, value, weight = gate.contiguous(), value.contiguous(), weight.contiguous()
        bias = bias.contiguous() if bias is not None else None
        output = F.linear(F.silu(gate) * value, weight, bias)
        ctx.save_for_backward(gate, value, weight, bias)
        return output

    @staticmethod
    def backward(ctx, gradient):
        gate, value, weight, bias = ctx.saved_tensors
        flat_gate = gate.reshape(-1, gate.shape[-1]).contiguous()
        flat_value = value.reshape(-1, value.shape[-1]).contiguous()
        flat_gradient = gradient.contiguous().reshape(-1, gradient.shape[-1])
        bias_gradient = flat_gradient.sum(0) if bias is not None else None
        with torch.no_grad():
            activation = F.silu(flat_gate)
            product = activation * flat_value
            weight_gradient = flat_gradient.T @ product
            product_gradient = flat_gradient @ weight.to(activation.dtype)
            probability = flat_gate.sigmoid()
            derivative = probability * (1 + flat_gate * (1 - probability))
            # Reassociation changes BF16 rounding near the SiLU minimum.
            gate_gradient = (derivative * flat_value) * product_gradient
            value_gradient = activation * product_gradient
        return (
            gate_gradient.reshape_as(gate),
            value_gradient.reshape_as(value),
            weight_gradient,
            bias_gradient,
        )


def silu_gate_linear(gate, value, weight, bias=None):
    """Apply SiLU(gate) * value and a down projection for first-order training."""
    return SiluGateLinear.apply(gate, value, weight, bias)

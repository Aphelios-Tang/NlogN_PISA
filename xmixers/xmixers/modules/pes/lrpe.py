"""Public xmixers split-half Lrpe, adapted to PISA kernels and packed positions."""
import torch
from torch import nn
from pisa.ops.lrpe import rotary_fn

class Lrpe(nn.Module):
    def __init__(self, head_dim=128, num_heads=8, lrpe_type=1, base=10000,
                 act="none", act_dim=None, fraction=None):
        super().__init__()
        if lrpe_type not in (1, 2, 3) or act != "none":
            raise ValueError("The paper uses ordinary split-half rotary encoding")
        self.head_dim, self.num_heads = head_dim, num_heads
        self.lrpe_type_, self.base = lrpe_type, base
        self.fraction = (1.0 if lrpe_type == 1 else .5) if fraction is None else fraction
        self.register_buffer("theta", torch.empty(0), persistent=False)
        self._init_weights()

    def _init_weights(self):
        # Construct FP32 frequencies on CPU,
        # then move the fixed table rather than recomputing pow on each device.
        theta = self.base ** (-2 / self.head_dim * torch.arange(self.head_dim // 2, dtype=torch.int64, device="cpu")).float()
        count = int(self.head_dim // 2 * self.fraction)
        if self.lrpe_type_ == 2:
            theta[:theta.numel() - count] = 0
        else:
            theta[count:] = 0
        self.theta = theta.reshape(1, -1).to(self.theta.device)
        self._is_hf_initialized = True

    def forward(self, x, offset=0, *, use_triton=True):
        positions = offset if isinstance(offset, torch.Tensor) else torch.arange(x.shape[1], device=x.device)[None] + offset
        return rotary_fn(x, self.theta, positions, use_triton=use_triton)

    def torch_forward(self, x, positions):
        return self(x, positions, use_triton=False)

class RotaryEmbedding(Lrpe):
    def __init__(self, head_dim, base, fraction):
        super().__init__(head_dim=head_dim, base=base, lrpe_type=3, fraction=fraction)

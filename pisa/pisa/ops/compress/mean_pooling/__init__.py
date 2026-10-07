"""Mean pooling and pyramid summaries."""
from .mean_pooling import build_pyramid, mean_pool
from .mean_pooling_torch import torch_mean_pool

mean_pooling_fn = mean_pool
__all__ = ["mean_pooling_fn", "mean_pool", "torch_mean_pool", "build_pyramid"]

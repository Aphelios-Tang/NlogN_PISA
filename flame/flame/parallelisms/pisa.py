"""Compatibility exports for the restored public xmixers parallelism module."""
from .xmixers.parallelize_xmixers import (
    apply_ac, apply_compile, apply_ddp, apply_fsdp,
    parallelize_xmixers as parallelize_pisa,
)

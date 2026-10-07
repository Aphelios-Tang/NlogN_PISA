from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.pipelining import PipelineStage
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torchtitan.config_manager import TORCH_DTYPE_MAP, JobConfig
from torchtitan.logging import logger
if TYPE_CHECKING:
    from torchtitan.parallelisms.parallel_dims import ParallelDims
from transformers import PretrainedConfig

DeviceType = Union[int, str, torch.device]


def validate_model_parallelism(model_config, parallel_dims):
    """PISA attention adapters currently support data parallelism only."""
    if getattr(model_config, "attention_method", None) in {
        "full", "bsa", "pisa1", "pisa2", "pisa", "nsa"
    }:
        for name in ("tp", "pp", "cp"):
            if getattr(parallel_dims, name) != 1:
                raise ValueError(
                    f"PISA attention does not support {name.upper()} degree other than 1"
                )


def pipeline_model(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: PretrainedConfig,
    loss_fn: Callable[..., torch.Tensor],
    model_source: str = "xmixers",
):
    validate_model_parallelism(model_config, parallel_dims)
    if model_source == "fla":
        from .fla import pipeline_fla
        return pipeline_fla(
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )
    elif model_source == "xmixers":
        from .xmixers import pipeline_xmixers
        return pipeline_xmixers(
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )
    raise ValueError(f"Unsupported model source: {model_source}")


def parallelize_model(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    model_source: str = "xmixers",
):
    validate_model_parallelism(model.config, parallel_dims)
    if model_source == "fla":
        from .fla import parallelize_fla
        return parallelize_fla(model, world_mesh, parallel_dims, job_config)
    elif model_source == "xmixers":
        from .xmixers import parallelize_xmixers
        return parallelize_xmixers(model, world_mesh, parallel_dims, job_config)
    raise ValueError(f"Unsupported model source: {model_source}")

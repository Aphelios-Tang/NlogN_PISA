# TODO: still has bug in bwd, update this later
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


class LossParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = False,
    ):
        super().__init__()
        self.use_local_output = use_local_output
        self.input_layouts = input_layouts
        self.desired_input_layouts = (Shard(1),)
        self.output_layouts = output_layouts

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        (
            weight,
            hidden_states,
            labels,
            bias,
            num_logits_to_keep,
            fuse_linear_and_cross_entropy,
        ) = inputs
        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, (Replicate(),))
        if weight.placements != (Replicate(),):
            weight = weight.redistribute(placements=(Replicate(),), async_op=True)

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, (Replicate(),))
        if bias is not None and bias.placements != (Replicate(),):
            bias = bias.redistribute(placements=(Replicate(),), async_op=True)

        if not isinstance(hidden_states, DTensor):
            hidden_states = DTensor.from_local(
                hidden_states, device_mesh, input_layouts
            )
        if hidden_states.placements != desired_input_layouts:
            hidden_states = hidden_states.redistribute(
                placements=desired_input_layouts, async_op=True
            )

        if not isinstance(labels, DTensor):
            labels = DTensor.from_local(labels, device_mesh, (Replicate(),))
        if labels.placements != desired_input_layouts:
            labels = labels.redistribute(
                placements=desired_input_layouts, async_op=True
            )

        return (
            weight.to_local(),
            hidden_states.to_local(),
            labels.to_local(),
            bias.to_local() if bias is not None else bias,
            num_logits_to_keep,
            fuse_linear_and_cross_entropy,
        )

    def _partition_fn(self, name, module, device_mesh):
        pass

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        output1, output2 = outputs
        if output1 is not None and not isinstance(output1, DTensor):
            output1 = DTensor.from_local(output1, device_mesh, output_layouts)
        if output1 is not None and output1.placements != output_layouts:
            output1 = output1.redistribute(placements=output_layouts, async_op=True)

        return output1, output2

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class PrepareModuleWeight(ParallelStyle):
    def __init__(
        self,
        *,
        layouts: Optional[Placement] = None,
        replicate_name_list: Optional[List[str]] = None,
    ):
        super().__init__()
        self.layouts = (layouts,)
        self.replicate_name_list = replicate_name_list

    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            if (
                self.replicate_name_list is not None
                and name in self.replicate_name_list
            ):
                replicated_param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, (Replicate(),))
                )
            else:
                replicated_param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, self.layouts)
                )
            module.register_parameter(p_name, replicated_param)

    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh):
        pass

    @staticmethod
    def _prepare_output_fn(mod, outputs, device_mesh):
        pass

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            self._prepare_input_fn,
            self._prepare_output_fn,
        )


class RowwiseGateLinearParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.desired_input_layouts = (Shard(-1),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        x1, x2, weight, bias, act, residual = inputs
        if not isinstance(x1, DTensor):
            x1 = DTensor.from_local(x1, device_mesh, input_layouts, run_check=False)
        if x1.placements != desired_input_layouts:
            x1 = x1.redistribute(placements=desired_input_layouts, async_op=True)

        if not isinstance(x2, DTensor):
            x2 = DTensor.from_local(x2, device_mesh, input_layouts, run_check=False)
        if x2.placements != desired_input_layouts:
            x2 = x2.redistribute(placements=desired_input_layouts, async_op=True)

        if residual is not None and not isinstance(residual, DTensor):
            residual = DTensor.from_local(
                residual, device_mesh, (Replicate(),), run_check=False
            )
        if residual is not None and residual.placements != (Replicate(),):
            residual = residual.redistribute(placements=(Replicate(),), async_op=True)

        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, (Shard(1),))
        if weight.placements != (Shard(1),):
            weight = weight.redistribute(placements=(Shard(1),), async_op=True)

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, (Replicate(),))
        if bias is not None and bias.placements != (Replicate(),):
            bias = bias.redistribute(placements=(Replicate(),), async_op=True)

        return (
            x1,
            x2,
            weight,
            bias,
            act,
            residual,
        )

    def _partition_fn(self, name, module, device_mesh):
        pass

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class NormParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        dim: int = 1,
        use_local_output: bool = False,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(1),)
        self.output_layouts = (output_layouts or Shard(1),)
        self.desired_input_layouts = (Shard(1),)
        self.weight_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        pass

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, weight_layouts, mod, inputs, device_mesh
    ):
        (
            x,
            weight,
            bias,
            residual,
            dim,
            eps,
            use_mean,
            num_groups,
            return_residual,
        ) = inputs

        if x is not None and not isinstance(x, DTensor):
            x = DTensor.from_local(x, device_mesh, input_layouts, run_check=False)
        if x is not None and x.placements != desired_input_layouts:
            x = x.redistribute(placements=desired_input_layouts, async_op=True)

        if weight is not None and not isinstance(weight, DTensor):
            weight = DTensor.from_local(
                weight, device_mesh, weight_layouts, run_check=False
            )
        if weight is not None and weight.placements != weight_layouts:
            weight = weight.redistribute(placements=weight_layouts, async_op=True)

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, bias_layouts, run_check=False)
        if bias is not None and bias.placements != bias_layouts:
            bias = bias.redistribute(placements=bias_layouts, async_op=True)

        if residual is not None and not isinstance(residual, DTensor):
            residual = DTensor.from_local(
                residual, device_mesh, input_layouts, run_check=False
            )
        if residual is not None and residual.placements != desired_input_layouts:
            residual = residual.redistribute(
                placements=desired_input_layouts, async_op=True
            )

        return (
            x.to_local(),
            weight.to_local() if weight is not None else weight,
            bias.to_local() if bias is not None else bias,
            residual.to_local() if residual is not None else residual,
            dim,
            eps,
            use_mean,
            num_groups,
            return_residual,
        )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if outputs is not None and not isinstance(outputs, DTensor):
            outputs = DTensor.from_local(outputs, device_mesh, output_layouts)
        if outputs is not None and outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(
                self._prepare_input_fn,
                self.input_layouts,
                self.desired_input_layouts,
                self.weight_layouts,
            ),
            partial(
                self._prepare_output_fn,
                self.output_layouts,
                self.use_local_output,
            ),
        )

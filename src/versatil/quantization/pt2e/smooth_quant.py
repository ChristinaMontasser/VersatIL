"""SmoothQuant-style preparation for PT2E export."""

from collections.abc import Callable, Iterable

import torch
from torch import nn


class SmoothQuantPreparer:
    """Collect activation ranges and prepare affine modules before PT2E export."""

    def __init__(self, alpha: float = 0.5) -> None:
        """Initialize the preparer.

        Args:
            alpha: Fraction of activation range migrated into weights.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"SmoothQuant alpha must be in [0, 1], got {alpha}.")
        self.alpha = alpha

    def collect_activation_scales(
        self,
        model: nn.Module,
        calibration: Callable[[], Iterable[tuple[torch.Tensor, ...]]],
        module_filter: Callable[[str, nn.Module], bool],
    ) -> dict[str, torch.Tensor]:
        """Collect per-input-channel activation maxima from calibration data.

        Args:
            model: Float model used for calibration.
            calibration: Callable returning an iterable of positional batches.
            module_filter: Predicate selecting affine modules by name.

        Returns:
            Maximum absolute activation for each selected module input channel.
        """
        activation_scales: dict[str, torch.Tensor] = {}
        handles = []

        def collect(
            module_name: str,
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            values = inputs[0].detach().abs()
            if isinstance(_module, nn.Linear):
                values = values.reshape(-1, values.shape[-1])
                current = values.amax(dim=0)
            else:
                # If it is conv
                current = values.amax(dim=(0, 2, 3))
            previous = activation_scales.get(module_name)
            activation_scales[module_name] = (
                current if previous is None else torch.maximum(previous, current)
            )

        for module_name, module in model.named_modules():
            if not module_filter(module_name, module):
                continue
            handles.append(
                module.register_forward_pre_hook(
                    lambda current_module, inputs, name=module_name: collect(
                        name, current_module, inputs
                    )
                )
            )
        try:
            with torch.no_grad():
                for batch in calibration():
                    model(*batch)
        finally:
            for handle in handles:
                handle.remove()
        return activation_scales

    def apply(
        self,
        model: nn.Module,
        activation_scales: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply SmoothQuant scales to selected affine modules in-place.

        Args:
            model: Float model to mutate before export.
            activation_scales: Per-module activation maxima.

        Returns:
            Fully qualified module names mapped to their migration scales.
        """
        transformed: dict[str, torch.Tensor] = {}
        for module_name, activation_scale in activation_scales.items():
            module = model.get_submodule(module_name)
            # Only apply to Conv2d and Linear modules, which are the only ones that have weights to scale.
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            # Get the maximum weight range for each input channel.
            weight_scale = self._input_weight_ranges(module=module)
            # Compute the SmoothQuant migration scale for each input channel.
            scale = self._migration_scale(
                activation_scale=activation_scale,
                weight_scale=weight_scale,
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            with torch.no_grad():
                # Scale the module weights in-place to migrate activation range into weights.
                # What gets applied here is weight-only change, the reciprocal of this scale will be applied to the input in the exported graph.
                #Since weight-only change is not mathematically sufficient by itself.
                module.weight.mul_(self._weight_scale(module=module, scale=scale))
            transformed[module_name] = scale.detach()
        return transformed

    @staticmethod
    def _weight_scale(
        module: nn.Conv2d | nn.Linear,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(module, nn.Linear):
            return scale.reshape(1, -1)
        input_channels_per_group = module.in_channels // module.groups
        output_channels_per_group = module.out_channels // module.groups
        grouped_scale = scale.reshape(module.groups, input_channels_per_group)
        repeated_scale = grouped_scale.repeat_interleave(
            output_channels_per_group,
            dim=0,
        )
        return repeated_scale.reshape(
            module.out_channels,
            input_channels_per_group,
            1,
            1,
        )

    @staticmethod
    def _input_weight_ranges(module: nn.Conv2d | nn.Linear) -> torch.Tensor:
        """Return one maximum weight range for each input channel."""
        weight = module.weight.detach().abs()
        if isinstance(module, nn.Linear):
            return weight.amax(dim=0)
        input_channels_per_group = module.in_channels // module.groups
        output_channels_per_group = module.out_channels // module.groups
        grouped_weight = weight.reshape(
            module.groups,
            output_channels_per_group,
            input_channels_per_group,
            *module.kernel_size,
        )
        return grouped_weight.amax(dim=(1, 3, 4)).reshape(-1)

    def _migration_scale(
        self,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = torch.finfo(torch.float32).eps
        return (
            (activation_scale.float() + epsilon).pow(self.alpha)
            / (weight_scale.float() + epsilon).pow(1.0 - self.alpha)
        ).to(activation_scale.device)

def apply_smooth_quant_scales_to_graph(
    model: torch.fx.GraphModule,
    scales: dict[str, torch.Tensor],
) -> None:
    """Insert reciprocal SmoothQuant scales into an exported FX graph.

    The affine weights are scaled before export. This graph rewrite preserves
    the mathematically equivalent reciprocal input scaling without registering
    a wrapper module around every affine operator.

    Args:
        model: Exported float graph module containing the scaled affine ops.
        scales: Module paths mapped to their SmoothQuant migration scales.
    """
    if not scales:
        return

    buffer_names: dict[str, str] = {}
    for index, (module_path, scale) in enumerate(scales.items()):
        buffer_name = f"_smooth_quant_scale_{index}"
        #buffer is tensor state belonging to the module that isn't a trainable parameter.
        model.register_buffer(buffer_name, scale)
        buffer_names[module_path] = buffer_name #"conv1": "_smooth_quant_scale_0",

    for node in model.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target not in (
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
        ): #Only target Conv2d and Linear
            continue
        module_path = _exported_module_path(node=node, scales=scales)
        #Which module from scales does this FX node represent?
        if module_path is None:
            continue
        input_node = node.args[0]
        scale_node = model.graph.get_attr(buffer_names[module_path])
        if node.target == torch.ops.aten.conv2d.default:
            scale_node = model.graph.call_function(
                torch.ops.aten.reshape.default,
                #We need to reshape it so broadcasting works correctly.
                args=(scale_node, [1, -1, 1, 1]),
            )
        #This creates a new FX node representing the inverse scale:
        scaled_input = model.graph.call_function(
            torch.ops.aten.div.Tensor, #division is the reciprocal of multiplication, which is what we did to the weights.
            args=(input_node, scale_node),
        )
        #place the scales
        node.prepend(scaled_input)
        scaled_input.prepend(scale_node)
        if node.target == torch.ops.aten.conv2d.default:
            scale_node.prepend(scale_node.args[0])
        node.replace_input_with(input_node, scaled_input)
    #Check graph consistency
    model.graph.lint()
    model.recompile()


def remove_unused_exported_state(model: torch.fx.GraphModule) -> int:
    """Remove zero-user exported attributes from a converted graph.

    PT2E conversion replaces affine weights with frozen quantized parameters,
    but an exported GraphModule can retain the original float attributes and
    their zero-user ``get_attr`` nodes. Removing those nodes and attributes
    keeps the serialized state aligned with the executable graph.

    Args:
        model: Converted PT2E graph module to clean.

    Returns:
        Number of removed exported attributes.
    """
    removed = 0
    for node in list(model.graph.nodes):
        if node.op != "get_attr" or node.users:
            continue
        target = str(node.target)
        parent_name, _, attribute_name = target.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if not hasattr(parent, attribute_name):
            continue
        delattr(parent, attribute_name)
        model.graph.erase_node(node)
        removed += 1
    if removed:
        model.graph.lint()
        model.recompile()
    return removed


def _exported_module_path(
    node: torch.fx.Node,
    scales: dict[str, torch.Tensor],
) -> str | None:
    module_stack = node.meta.get("nn_module_stack", {})
    stack_paths = [str(value[0]) for value in module_stack.values()]
    for module_path in scales:
        if module_path in stack_paths:
            return module_path
    return None
"""SmoothQuant-style preparation for PT2E export."""

from collections.abc import Callable, Iterable

import torch
from torch import nn


class InputScaledAffine(nn.Module):
    """Apply TorchAO's reciprocal input scale before an affine module."""

    def __init__(self, module: nn.Conv2d | nn.Linear, scale: torch.Tensor) -> None:
        """Initialize an input-scaled affine module.

        Args:
            module: Conv2d or Linear module whose input channels are scaled.
            scale: Positive scale for each input channel.
        """
        super().__init__()
        self.module = module
        self.register_buffer("input_scale", scale)
        weight_scale = self._weight_scale(module=module, scale=scale)
        with torch.no_grad():
            module.weight.mul_(weight_scale)

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
        weight_scale = grouped_scale.repeat_interleave(
            output_channels_per_group,
            dim=0,
        )
        return weight_scale.reshape(module.out_channels, input_channels_per_group, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Scale affine inputs and run the rescaled module."""
        if isinstance(self.module, nn.Linear):
            scaled_inputs = inputs / self.input_scale
        else:
            scaled_inputs = inputs / self.input_scale.reshape(1, -1, 1, 1)
        return self.module(scaled_inputs)


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
    ) -> list[str]:
        """Apply SmoothQuant scales to selected affine modules in-place.

        Args:
            model: Float model to mutate before export.
            activation_scales: Per-module activation maxima.

        Returns:
            Fully qualified module names that were transformed.
        """
        transformed: list[str] = []
        for module_name, activation_scale in activation_scales.items():
            module = model.get_submodule(module_name)
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            weight_scale = self._input_weight_ranges(module=module)
            scale = self._migration_scale(
                activation_scale=activation_scale,
                weight_scale=weight_scale,
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            self._replace_module(
                model=model,
                module_name=module_name,
                wrapped=InputScaledAffine(module=module, scale=scale),
            )
            transformed.append(module_name)
        return transformed

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

    @staticmethod
    def _replace_module(
        model: nn.Module,
        module_name: str,
        wrapped: InputScaledAffine,
    ) -> None:
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrapped)
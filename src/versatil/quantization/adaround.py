"""Native PyTorch AdaRound optimization for affine weights."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class AdaRoundConfig:
    """Hyperparameters for layer-wise AdaRound optimization.

    Attributes:
        iterations: Number of optimization steps per layer.
        regularization_strength: Weight of the rounding regularizer.
        beta_start: Initial regularizer temperature.
        beta_end: Final regularizer temperature.
        warm_start: Fraction of iterations without rounding regularization.
        bitwidth: Symmetric weight quantization bitwidth.
        group_size: Number of input columns sharing one scale, or ``None`` for
            per-output-channel scaling.
        optimization_batch_size: Number of calibration rows used per optimizer
            step.
    """

    iterations: int = 10000
    regularization_strength: float = 0.01
    beta_start: float = 20.0
    beta_end: float = 2.0
    warm_start: float = 0.2
    bitwidth: int = 8
    group_size: int | None = None
    optimization_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {self.iterations}.")
        if self.regularization_strength < 0.0:
            raise ValueError(
                "regularization_strength must be non-negative, "
                f"got {self.regularization_strength}."
            )
        if self.beta_start < self.beta_end or self.beta_end <= 0.0:
            raise ValueError(
                "beta_start must be at least beta_end and beta_end must be positive."
            )
        if not 0.0 <= self.warm_start < 1.0:
            raise ValueError(
                f"warm_start must be in [0.0, 1.0), got {self.warm_start}."
            )
        if self.group_size is not None and self.group_size < 1:
            raise ValueError(f"group_size must be positive, got {self.group_size}.")
        if self.optimization_batch_size < 1:
            raise ValueError(
                "optimization_batch_size must be positive, "
                f"got {self.optimization_batch_size}."
            )
        if not 2 <= self.bitwidth <= int(torch.iinfo(torch.int32).bits):
            raise ValueError(f"bitwidth must be between 2 and 32, got {self.bitwidth}.")


@dataclass(frozen=True)
class AdaRoundResult:
    """Reconstruction metrics for one optimized layer."""

    initial_loss: float
    final_loss: float


def collect_linear_inputs(
    model: nn.Module,
    calibration: Callable[[], Iterable[tuple[torch.Tensor, ...]]],
    forward: Callable[
        [tuple[torch.Tensor, ...]], torch.Tensor | tuple[torch.Tensor, ...]
    ],
    module_names: set[str],
    storage_device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Collect inputs seen by selected linear modules.

    Args:
        model: Model containing the selected modules.
        calibration: Factory returning positional calibration batches.
        forward: Function that executes one positional batch through the model.
        module_names: Fully qualified linear module names to collect.

    Returns:
        Concatenated input tensors for each selected module.
    """
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}
    handles = []

    for name in module_names:
        module = model.get_submodule(name)

        def collect(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            module_name: str = name,
        ) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                collected[module_name].append(
                    inputs[0].detach().to(device=storage_device)
                )

        handles.append(module.register_forward_pre_hook(collect))
    try:
        with torch.inference_mode():
            for batch in calibration():
                forward(batch)
    finally:
        for handle in handles:
            handle.remove()
    return {
        name: torch.cat(inputs, dim=0)
        for name, inputs in collected.items()
        if inputs
    }


def apply_adaround(
    module: nn.Linear,
    inputs: torch.Tensor,
    config: AdaRoundConfig,
) -> AdaRoundResult:
    """Optimize and materialize AdaRound weights for one linear layer.

    Args:
        module: Floating-point linear layer to mutate in-place.
        inputs: Calibration inputs accepted by the layer.
        config: AdaRound optimization and quantization settings.

    Returns:
        Initial and final local reconstruction losses.
    """
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"AdaRound only supports nn.Linear, got {type(module).__name__}."
        )
    if inputs.ndim < 2 or inputs.shape[-1] != module.in_features:
        raise ValueError(
            f"AdaRound inputs must end in {module.in_features} features, "
            f"got shape {tuple(inputs.shape)}."
        )

    weight = module.weight.detach()
    quantization_max = (1 << (config.bitwidth - 1)) - 1
    if config.group_size is not None:
        if module.in_features % config.group_size != 0:
            raise ValueError(
                f"in_features {module.in_features} is not divisible by "
                f"AdaRound group_size {config.group_size}."
            )
        grouped_weight = weight.reshape(
            module.out_features, module.in_features // config.group_size, config.group_size
        )
        scale = grouped_weight.abs().amax(dim=2, keepdim=True)
        scale = scale.repeat_interleave(config.group_size, dim=2).reshape_as(weight)
    else:
        scale = weight.abs().amax(dim=1, keepdim=True)
    scale = scale.clamp_min(torch.finfo(weight.dtype).eps)
    scale = scale / quantization_max
    normalized = weight / scale
    floor_weight = normalized.floor()
    fractional = (normalized - floor_weight).clamp(1e-6, 1.0 - 1e-6)
    alpha = nn.Parameter(
        torch.logit(fractional).float().to(device=weight.device),
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([alpha])
    evaluation_inputs = inputs[: config.optimization_batch_size].to(
        device=weight.device
    )
    evaluation_float_output = functional.linear(
        evaluation_inputs, weight, module.bias
    ).detach()
    initial_output = _adaround_output(
        evaluation_inputs,
        module.bias,
        scale,
        floor_weight,
        alpha,
        soft=False,
        quantization_max=quantization_max,
    )
    initial_loss = functional.mse_loss(initial_output, evaluation_float_output)

    sample_count = inputs.shape[0]
    if sample_count < 1:
        raise ValueError("AdaRound requires at least one calibration input.")
    for iteration in range(config.iterations):
        optimizer.zero_grad()
        start = (iteration * config.optimization_batch_size) % sample_count
        end = min(start + config.optimization_batch_size, sample_count)
        batch_inputs = inputs[start:end].to(device=weight.device)
        batch_float_output = functional.linear(
            batch_inputs, weight, module.bias
        ).detach()
        quantized_output = _adaround_output(
            batch_inputs,
            module.bias,
            scale,
            floor_weight,
            alpha,
            soft=True,
            quantization_max=quantization_max,
        )
        reconstruction_loss = functional.mse_loss(
            quantized_output, batch_float_output
        )
        rounding_loss = _rounding_loss(alpha=alpha, config=config, iteration=iteration)
        (reconstruction_loss + rounding_loss).backward()
        optimizer.step()

    with torch.no_grad():
        rounded_output = _adaround_output(
            evaluation_inputs,
            module.bias,
            scale,
            floor_weight,
            alpha,
            soft=False,
            quantization_max=quantization_max,
        )
        module.weight.copy_(_hard_weight(scale, floor_weight, alpha, quantization_max))
    return AdaRoundResult(
        initial_loss=float(initial_loss.detach()),
        final_loss=float(
            functional.mse_loss(rounded_output, evaluation_float_output).detach()
        ),
    )


def _adaround_output(
    inputs: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    floor_weight: torch.Tensor,
    alpha: torch.Tensor,
    soft: bool,
    quantization_max: int,
) -> torch.Tensor:
    weight = _hard_weight(scale, floor_weight, alpha, quantization_max, soft=soft)
    return functional.linear(inputs, weight, bias)


def _hard_weight(
    scale: torch.Tensor,
    floor_weight: torch.Tensor,
    alpha: torch.Tensor,
    quantization_max: int,
    soft: bool = False,
) -> torch.Tensor:
    if soft:
        rounding = (
            torch.sigmoid(alpha.to(dtype=scale.dtype)) * 1.2 - 0.1
        ).clamp(0.0, 1.0)
    else:
        rounding = (alpha >= 0.0).to(dtype=scale.dtype)
    quantized = (floor_weight + rounding).clamp(-quantization_max, quantization_max)
    return quantized * scale


def _rounding_loss(
    alpha: torch.Tensor,
    config: AdaRoundConfig,
    iteration: int,
) -> torch.Tensor:
    if iteration < config.iterations * config.warm_start:
        return alpha.new_zeros(())
    progress = (iteration - config.iterations * config.warm_start) / (
        config.iterations * (1.0 - config.warm_start)
    )
    beta = config.beta_end + 0.5 * (config.beta_start - config.beta_end) * (
        1.0 + torch.cos(alpha.new_tensor(progress * torch.pi))
    )
    rounding = (torch.sigmoid(alpha) * 1.2 - 0.1).clamp(0.0, 1.0)
    regularizer = (1.0 - (2.0 * rounding - 1.0).abs().pow(beta)).sum()
    return config.regularization_strength * regularizer
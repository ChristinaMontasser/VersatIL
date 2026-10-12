"""Tests for versatil.quantization.adaround module."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from versatil.quantization.adaround import (
    AdaRoundConfig,
    apply_adaround,
    collect_linear_inputs,
)


@pytest.fixture
def linear_factory() -> Callable[..., nn.Linear]:
    """Factory for deterministic linear layers."""

    def factory(
        in_features: int = 4,
        out_features: int = 3,
    ) -> nn.Linear:
        module = nn.Linear(in_features, out_features)
        with torch.no_grad():
            module.weight.copy_(
                torch.linspace(
                    -0.75,
                    0.75,
                    steps=in_features * out_features,
                ).reshape(out_features, in_features)
            )
            module.bias.zero_()
        return module

    return factory


@pytest.mark.unit
def test_collect_linear_inputs_uses_forward_hooks(
    linear_factory: Callable[..., nn.Linear],
) -> None:
    model = nn.Sequential(linear_factory(), nn.ReLU())
    batches = [(torch.ones(2, 4),), (torch.zeros(1, 4),)]

    collected = collect_linear_inputs(
        model=model,
        calibration=lambda: iter(batches),
        forward=lambda batch: model(*batch),
        module_names={"0"},
    )

    assert collected["0"].shape == (3, 4)
    assert torch.equal(collected["0"][:2], batches[0][0])
    assert torch.equal(collected["0"][2:], batches[1][0])


@pytest.mark.unit
def test_apply_adaround_materializes_symmetric_hard_weights(
    linear_factory: Callable[..., nn.Linear],
) -> None:
    module = linear_factory()
    inputs = torch.tensor(
        [[1.0, -0.5, 0.25, 2.0], [0.5, 1.0, -1.0, 0.0]],
    )
    original_weight = module.weight.detach().clone()
    config = AdaRoundConfig(iterations=4, bitwidth=4)

    result = apply_adaround(module=module, inputs=inputs, config=config)

    quantization_max = 2 ** (config.bitwidth - 1) - 1
    scale = original_weight.abs().amax(dim=1, keepdim=True) / quantization_max
    normalized = module.weight.detach() / scale
    assert torch.allclose(normalized, normalized.round())
    assert result.initial_loss >= 0.0
    assert result.final_loss >= 0.0


@pytest.mark.unit
def test_adaround_config_rejects_invalid_temperature_schedule() -> None:
    with pytest.raises(
        ValueError,
        match="beta_start must be at least beta_end and beta_end must be positive.",
    ):
        AdaRoundConfig(beta_start=1.0, beta_end=2.0)

"""Tests for versatil.post_training_compression.preparation.equalization module."""

import pytest
import torch
from torch import nn

from versatil.post_training_compression.preparation.equalization import (
    equalize_cross_layer_pairs,
)


@pytest.mark.unit
def test_equalizes_conv_pair_and_preserves_output() -> None:
    model = nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(3, 2, kernel_size=1, bias=True),
    )
    model[0].weight.data.copy_(
        torch.tensor([[[[1.0]], [[2.0]]], [[[2.0]], [[4.0]]], [[[3.0]], [[6.0]]]])
    )
    model[0].bias.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    model[2].weight.data.copy_(
        torch.tensor([[[[4.0]], [[1.0]], [[2.0]]], [[[2.0]], [[3.0]], [[1.0]]]])
    )
    inputs = torch.tensor([[[[1.0]], [[-1.0]]]])
    expected = model(inputs)
    count = equalize_cross_layer_pairs(model=model)

    assert count == 1
    torch.testing.assert_close(model(inputs), expected)
    first_ranges = model[0].weight.detach().abs().amax(dim=(1, 2, 3))
    second_ranges = model[2].weight.detach().abs().amax(dim=(0, 2, 3))
    torch.testing.assert_close(first_ranges, second_ranges)


@pytest.mark.unit
def test_equalization_skips_incompatible_pairs() -> None:
    model = nn.Sequential(nn.Conv2d(2, 3, kernel_size=1), nn.Conv2d(4, 2, kernel_size=1))
    weights = [layer.weight.detach().clone() for layer in model]

    assert equalize_cross_layer_pairs(model=model) == 0
    for layer, weight in zip(model, weights):
        torch.testing.assert_close(layer.weight, weight)


@pytest.mark.unit
def test_equalization_handles_zero_range_channels() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    model[0].weight.data.zero_()
    model[1].weight.data.fill_(2.0)
    first_weight = model[0].weight.detach().clone()
    second_weight = model[1].weight.detach().clone()

    assert equalize_cross_layer_pairs(model=model) == 1
    torch.testing.assert_close(model[0].weight, first_weight)
    torch.testing.assert_close(model[1].weight, second_weight)


@pytest.mark.unit
def test_equalizes_depthwise_conv_pair_and_preserves_output() -> None:
    model = nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=True),
    )
    inputs = torch.ones(1, 2, 4, 4)
    expected = model(inputs)

    assert equalize_cross_layer_pairs(model=model) == 1
    torch.testing.assert_close(model(inputs), expected)

"""Tests for PT2E SmoothQuant preparation."""

import torch
from torch import nn

from versatil.quantization.pt2e.smooth_quant import InputScaledAffine
from versatil.quantization.pt2e.smooth_quant import SmoothQuantPreparer


def test_input_scaled_conv_preserves_float_output() -> None:
    convolution = nn.Conv2d(2, 3, kernel_size=1, bias=True)
    convolution.weight.data.copy_(torch.tensor(
        [
            [[[1.0]], [[2.0]]],
            [[[3.0]], [[4.0]]],
            [[[5.0]], [[6.0]]],
        ]
    ))
    convolution.bias.data.copy_(torch.tensor([0.5, -0.5, 1.0]))
    inputs = torch.tensor(
        [[[[1.0, 2.0]], [[3.0, 4.0]]]],
    )
    expected = convolution(inputs)
    prepared = InputScaledAffine(
        module=convolution,
        scale=torch.tensor([2.0, 4.0]),
    )

    actual = prepared(inputs)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        prepared.module.weight,
        torch.tensor(
            [
                [[[2.0]], [[8.0]]],
                [[[6.0]], [[16.0]]],
                [[[10.0]], [[24.0]]],
            ]
        ),
    )


def test_migration_scale_matches_activation_and_weight_ranges() -> None:
    preparer = SmoothQuantPreparer(alpha=0.5)

    scale = preparer._migration_scale(
        activation_scale=torch.tensor([4.0, 9.0]),
        weight_scale=torch.tensor([1.0, 1.0]),
    )

    torch.testing.assert_close(scale, torch.tensor([2.0, 3.0]))


def test_input_scaled_depthwise_conv_preserves_float_output() -> None:
    convolution = nn.Conv2d(2, 2, kernel_size=1, groups=2, bias=False)
    convolution.weight.data.copy_(torch.tensor(
        [[[[2.0]]], [[[3.0]]]],
    ))
    inputs = torch.tensor(
        [[[[1.0, 2.0]], [[3.0, 4.0]]]],
    )
    expected = convolution(inputs)
    prepared = InputScaledAffine(
        module=convolution,
        scale=torch.tensor([2.0, 4.0]),
    )

    actual = prepared(inputs)

    torch.testing.assert_close(actual, expected)

"""Tests for PT2E SmoothQuant preparation."""

import torch
from torch import nn

from versatil.quantization.pt2e.smooth_quant import (
    SmoothQuantPreparer,
    apply_smooth_quant_scales_to_graph,
)


def test_apply_folds_migration_scale_into_conv_weight() -> None:
    convolution = nn.Conv2d(2, 3, kernel_size=1, bias=True)
    convolution.weight.data.copy_(torch.tensor(
        [
            [[[1.0]], [[2.0]]],
            [[[3.0]], [[4.0]]],
            [[[5.0]], [[6.0]]],
        ]
    ))
    convolution.bias.data.copy_(torch.tensor([0.5, -0.5, 1.0]))
    model = nn.Module()
    model.add_module("convolution", convolution)
    preparer = SmoothQuantPreparer(alpha=0.5)
    expected_scale = preparer._migration_scale(
        activation_scale=torch.tensor([4.0, 16.0]),
        weight_scale=torch.tensor([5.0, 6.0]),
    )

    scales = preparer.apply(
        model=model,
        activation_scales={"convolution": torch.tensor([4.0, 16.0])},
    )

    torch.testing.assert_close(scales["convolution"], expected_scale)
    torch.testing.assert_close(
        model.convolution.weight,
        torch.tensor(
            [
                [[[1.0 * expected_scale[0]]], [[2.0 * expected_scale[1]]]],
                [[[3.0 * expected_scale[0]]], [[4.0 * expected_scale[1]]]],
                [[[5.0 * expected_scale[0]]], [[6.0 * expected_scale[1]]]],
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


def test_weight_scale_supports_depthwise_conv() -> None:
    convolution = nn.Conv2d(2, 2, kernel_size=1, groups=2, bias=False)
    convolution.weight.data.copy_(torch.tensor(
        [[[[2.0]]], [[[3.0]]]],
    ))
    model = nn.Module()
    model.add_module("convolution", convolution)
    preparer = SmoothQuantPreparer(alpha=0.5)
    expected_scale = preparer._migration_scale(
        activation_scale=torch.tensor([4.0, 16.0]),
        weight_scale=torch.tensor([2.0, 3.0]),
    )

    scales = preparer.apply(
        model=model,
        activation_scales={"convolution": torch.tensor([4.0, 16.0])},
    )

    torch.testing.assert_close(scales["convolution"], expected_scale)
    torch.testing.assert_close(
        model.convolution.weight,
        torch.tensor(
            [
                [[[2.0 * expected_scale[0]]]],
                [[[3.0 * expected_scale[1]]]],
            ]
        ),
    )


def test_graph_scaling_preserves_output_without_affine_wrapper_state() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convolution = nn.Conv2d(2, 3, kernel_size=1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.convolution(inputs)

    model = Model().eval()
    inputs = torch.ones(2, 2, 2, 2)
    expected = model(inputs)
    preparer = SmoothQuantPreparer(alpha=0.5)
    scales = preparer.apply(
        model=model,
        activation_scales={"convolution": torch.tensor([2.0, 3.0])},
    )
    exported = torch.export.export(model, (inputs,), strict=False).module()

    apply_smooth_quant_scales_to_graph(model=exported, scales=scales)

    torch.testing.assert_close(exported(inputs), expected)
    assert sum(
        node.target == torch.ops.aten.div.Tensor for node in exported.graph.nodes
    ) == 1
    assert not any(".module." in name for name, _ in exported.named_parameters())

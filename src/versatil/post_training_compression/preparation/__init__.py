"""Preparation subpackage for quantization-aware model transformations."""

from versatil.post_training_compression.preparation.batchnorm import (
    extract_activation,
    extract_batchnorm_parameters,
    has_batchnorm_buffers,
    is_frozen_batchnorm,
    prepare_batchnorms_for_quantization,
    replace_frozen_batchnorm,
)
from versatil.post_training_compression.preparation.fusion import (
    fuse_all_conv_batchnorm_pairs,
    fuse_conv_batchnorm,
)
from versatil.post_training_compression.preparation.equalization import (
    equalize_cross_layer_pairs,
)

__all__ = [
    "extract_activation",
    "extract_batchnorm_parameters",
    "equalize_cross_layer_pairs",
    "fuse_all_conv_batchnorm_pairs",
    "fuse_conv_batchnorm",
    "has_batchnorm_buffers",
    "is_frozen_batchnorm",
    "prepare_batchnorms_for_quantization",
    "replace_frozen_batchnorm",
]

# """Cross-layer equalization for consecutive affine layers."""

# import torch
# from torch import nn

# _AFFINE_TYPES = (nn.Conv2d, nn.Linear)
# _POINTWISE_TYPES = (nn.Identity, nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Hardswish)


# def _channel_ranges(weight: torch.Tensor, channel_dimension: int) -> torch.Tensor:
#     """Return max-absolute weight ranges for each channel."""
#     reduction_dimensions = tuple(
#         dimension for dimension in range(weight.ndim) if dimension != channel_dimension
#     )
#     return weight.detach().abs().amax(dim=reduction_dimensions)


# def _pair_parameters(
#     first: nn.Module,
#     second: nn.Module,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor] | None:
#     """Return weights and biases for a compatible affine-layer pair."""
#     if not isinstance(first, _AFFINE_TYPES) or not isinstance(second, _AFFINE_TYPES):
#         return None
#     first_output_channels = (
#         first.out_channels if isinstance(first, nn.Conv2d) else first.out_features
#     )
#     second_input_channels = (
#         second.in_channels if isinstance(second, nn.Conv2d) else second.in_features
#     )
#     if first_output_channels != second_input_channels:
#         return None
#     return first.weight, second.weight, first.bias, second.bias


# def _equalize_pair(first: nn.Module, second: nn.Module) -> None:
#     """Rescale one compatible affine-layer pair in-place."""
#     parameters = _pair_parameters(first=first, second=second)
#     if parameters is None:
#         return
#     first_weight, second_weight, first_bias, _second_bias = parameters
#     first_ranges = _channel_ranges(weight=first_weight, channel_dimension=0)
#     second_ranges = _channel_ranges(weight=second_weight, channel_dimension=1)
#     scales = torch.sqrt(first_ranges / second_ranges.clamp_min(torch.finfo(second_weight.dtype).eps))
#     scales = torch.where(
#         (first_ranges > 0) & (second_ranges > 0),
#         scales,
#         torch.ones_like(scales),
#     )
#     first_shape = (scales.shape[0],) + (1,) * (first_weight.ndim - 1)
#     second_shape = (1, scales.shape[0]) + (1,) * (second_weight.ndim - 2)
#     with torch.no_grad():
#         first_weight.div_(scales.reshape(first_shape))
#         second_weight.mul_(scales.reshape(second_shape))
#         if first_bias is not None:
#             first_bias.div_(scales)


# def equalize_cross_layer_pairs(model: nn.Module) -> int:
#     """Equalize compatible affine-layer pairs in a module hierarchy.

#     Consecutive affine layers may have pointwise activations or identities between
#     them. Branching structures and unsupported layer types are skipped.

#     Args:
#         model: Module whose affine layers should be equalized.

#     Returns:
#         Number of affine-layer pairs transformed.
#     """
#     pair_count = 0
#     for parent in model.modules():
#         children = list(parent.children())
#         for first_index, first in enumerate(children):
#             if not isinstance(first, _AFFINE_TYPES):
#                 continue
#             second_index = first_index + 1
#             while (
#                 second_index < len(children)
#                 and isinstance(children[second_index], _POINTWISE_TYPES)
#             ):
#                 second_index += 1
#             if second_index >= len(children):
#                 continue
#             second = children[second_index]
#             if _pair_parameters(first=first, second=second) is None:
#                 continue
#             _equalize_pair(first=first, second=second)
#             pair_count += 1
#     return pair_count



### CLAUDE ####

"""
Cross-layer equalization for consecutive affine layers.

This implementation is based on the equalization technique from:
"ZipML: A Benchmark for Training and Inference of Large Models"
and related quantization/equalization papers.

The key idea: rescale weights between consecutive layers to balance the weight
ranges (scales) across layers. This makes the network more amenable to 
quantization and improves training stability.

For consecutive affine layers L1 and L2:
- Compute the max-absolute range for each channel
- Calculate scaling factors s = sqrt(range_L1 / range_L2)
- Divide L1 output weights by s
- Multiply L2 input weights by s
- This preserves the forward pass: (w2 * s) * (x / s) = w2 * x
"""

import torch
from torch import nn
from typing import Optional, Tuple, List
from collections import defaultdict

# Layer types that support equalization
_AFFINE_TYPES = (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d)
# Pointwise operations that can appear between affine layers
_POINTWISE_TYPES = (
    nn.Identity, 
    nn.ReLU, 
    nn.ReLU6, 
    #nn.GELU, 
    #nn.SiLU, 
    nn.Hardswish,
    nn.ELU,
    nn.LeakyReLU,
)


def _get_output_channels(module: nn.Module) -> Optional[int]:
    """Get the number of output channels for an affine layer."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return module.out_channels
    elif isinstance(module, nn.Linear):
        return module.out_features
    return None


def _get_input_channels(module: nn.Module) -> Optional[int]:
    """Get the number of input channels for an affine layer."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return module.in_channels
    elif isinstance(module, nn.Linear):
        return module.in_features
    return None


def _channel_ranges(weight: torch.Tensor, channel_dimension: int) -> torch.Tensor:
    """
    Return max-absolute weight ranges for each channel.
    
    Args:
        weight: Weight tensor to analyze
        channel_dimension: Which dimension represents channels
        
    Returns:
        1D tensor of max-absolute values for each channel
    """
    reduction_dimensions = tuple(
        dimension for dimension in range(weight.ndim) 
        if dimension != channel_dimension
    )
    return weight.detach().abs().amax(dim=reduction_dimensions)


def _pair_parameters(
    first: nn.Module,
    second: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
    """
    Return weights and biases for a compatible affine-layer pair.
    
    Returns:
        Tuple of (first_weight, second_weight, first_bias, second_bias)
        or None if layers are incompatible
    """
    if not isinstance(first, _AFFINE_TYPES) or not isinstance(second, _AFFINE_TYPES):
        return None
    
    first_output_channels = _get_output_channels(first)
    second_input_channels = _get_input_channels(second)
    
    if first_output_channels is None or second_input_channels is None:
        return None
    
    if first_output_channels != second_input_channels:
        return None
    
    return first.weight, second.weight, first.bias, second.bias


def _equalize_pair(
    first: nn.Module, 
    second: nn.Module,
    eps: float = 1e-8,
) -> None:
    """
    Rescale one compatible affine-layer pair in-place.
    
    Applies scaling to balance the output ranges of the first layer
    with the input ranges of the second layer.
    
    Args:
        first: First affine layer
        second: Second affine layer (must be compatible)
        eps: Small epsilon for numerical stability
    """
    parameters = _pair_parameters(first=first, second=second)
    if parameters is None:
        return
    
    first_weight, second_weight, first_bias, _second_bias = parameters
    
    # Compute ranges for each channel
    first_ranges = _channel_ranges(weight=first_weight, channel_dimension=0)
    is_depthwise = (
        isinstance(second, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        and second.groups == second.in_channels == second.out_channels
    )
    second_channel_dimension = 0 if is_depthwise else 1
    second_ranges = _channel_ranges(
        weight=second_weight,
        channel_dimension=second_channel_dimension,
    )
    
    # Compute scaling factors: s = sqrt(range_first / range_second)
    # This balances the magnitudes across the layer boundary
    min_val = torch.finfo(second_weight.dtype).eps if eps == 1e-8 else eps
    scales = torch.sqrt(first_ranges / second_ranges.clamp_min(min_val))
    
    # Only apply scaling where both ranges are non-zero
    # Otherwise, keep scale as 1.0
    scales = torch.where(
        (first_ranges > 0) & (second_ranges > 0),
        scales,
        torch.ones_like(scales),
    )
    
    # Reshape scales for broadcasting
    # For first layer (output): scale along output channel dimension
    first_shape = (scales.shape[0],) + (1,) * (first_weight.ndim - 1)
    # For second layer (input): scale along input channel dimension  
    if is_depthwise:
        second_shape = (scales.shape[0],) + (1,) * (second_weight.ndim - 1)
    else:
        second_shape = (1, scales.shape[0]) + (1,) * (second_weight.ndim - 2)
    
    # Apply scaling (in-place operations)
    with torch.no_grad():
        first_weight.div_(scales.reshape(first_shape))
        second_weight.mul_(scales.reshape(second_shape))
        
        # If first layer has bias, scale it inversely
        if first_bias is not None:
            first_bias.div_(scales)


def equalize_cross_layer_pairs(
    model: nn.Module,
    verbose: bool = False,
) -> int:
    """
    Equalize compatible affine-layer pairs in a module hierarchy.

    This function recursively searches through the module hierarchy for
    consecutive affine layers (Conv2d, Linear, etc.). Pointwise activations
    (ReLU, etc.) and identity operations between layers are transparent.
    
    The equalization rescales weights to balance the magnitude ranges across
    layer boundaries, which can improve quantization and training stability.

    Args:
        model: Module whose affine layers should be equalized.
        verbose: If True, print information about equalized pairs.

    Returns:
        Number of affine-layer pairs transformed.
        
    Example:
        >>> model = MyNetwork()
        >>> pair_count = equalize_cross_layer_pairs(model)
        >>> print(f"Equalized {pair_count} layer pairs")
    """
    pair_count = 0
    
    for parent in model.modules():
        children = list(parent.children())
        
        for first_index, first in enumerate(children):
            if not isinstance(first, _AFFINE_TYPES):
                continue
            
            # Skip through pointwise operations to find the next affine layer
            second_index = first_index + 1
            while (
                second_index < len(children) 
                and isinstance(children[second_index], _POINTWISE_TYPES)
            ):
                second_index += 1
            
            # No compatible second layer found
            if second_index >= len(children):
                continue
            
            second = children[second_index]
            
            # Check compatibility and equalize if possible
            if _pair_parameters(first=first, second=second) is None:
                continue
            
            if verbose:
                first_name = first.__class__.__name__
                second_name = second.__class__.__name__
                print(f"Equalizing {first_name} -> {second_name}")
            
            _equalize_pair(first=first, second=second)
            pair_count += 1
    
    return pair_count


def equalize_cross_layer_pairs_recursive(
    model: nn.Module,
    verbose: bool = False,
) -> int:
    """
    Alternative implementation using module graph traversal.
    
    This is more robust for complex architectures but may miss some patterns.
    Useful as a fallback or for verification.
    
    Args:
        model: Module whose affine layers should be equalized.
        verbose: If True, print information about equalized pairs.
        
    Returns:
        Number of affine-layer pairs transformed.
    """
    pair_count = 0
    
    def _traverse_modules(module: nn.Module, parent_list: List[nn.Module]) -> None:
        nonlocal pair_count
        
        for child in module.children():
            new_parent_list = parent_list + [child]
            
            # Try to equalize consecutive affine layers in the hierarchy
            if len(new_parent_list) >= 2:
                first = new_parent_list[-2]
                second = new_parent_list[-1]
                
                if isinstance(first, _AFFINE_TYPES) and isinstance(second, _AFFINE_TYPES):
                    if _pair_parameters(first=first, second=second) is not None:
                        if verbose:
                            print(f"Equalizing {first.__class__.__name__} -> {second.__class__.__name__}")
                        _equalize_pair(first=first, second=second)
                        pair_count += 1
            
            _traverse_modules(child, new_parent_list)
    
    _traverse_modules(model, [])
    return pair_count


def get_equalization_stats(model: nn.Module) -> dict:
    """
    Analyze weight statistics before/after equalization opportunity.
    
    Useful for understanding the impact of equalization.
    
    Args:
        model: Module to analyze
        
    Returns:
        Dictionary with statistics about potential equalization targets
    """
    stats = {
        'compatible_pairs': 0,
        'layer_pairs': [],
    }
    
    for parent in model.modules():
        children = list(parent.children())
        
        for first_index, first in enumerate(children):
            if not isinstance(first, _AFFINE_TYPES):
                continue
            
            second_index = first_index + 1
            while (
                second_index < len(children) 
                and isinstance(children[second_index], _POINTWISE_TYPES)
            ):
                second_index += 1
            
            if second_index >= len(children):
                continue
            
            second = children[second_index]
            
            if _pair_parameters(first=first, second=second) is None:
                continue
            
            first_weight, second_weight, _, _ = _pair_parameters(first=first, second=second)
            
            first_ranges = _channel_ranges(weight=first_weight, channel_dimension=0)
            second_ranges = _channel_ranges(weight=second_weight, channel_dimension=1)
            
            pair_info = {
                'first_layer': first.__class__.__name__,
                'second_layer': second.__class__.__name__,
                'first_range_mean': first_ranges.mean().item(),
                'first_range_std': first_ranges.std().item(),
                'second_range_mean': second_ranges.mean().item(),
                'second_range_std': second_ranges.std().item(),
                'range_ratio': (first_ranges.mean() / second_ranges.mean().clamp_min(1e-8)).item(),
            }
            
            stats['layer_pairs'].append(pair_info)
            stats['compatible_pairs'] += 1
    
    return stats


# if __name__ == "__main__":
#     # Example usage
#     print("Cross-Layer Equalization Module")
#     print("=" * 50)
    
#     # Create a simple test model
#     class SimpleModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.layer1 = nn.Conv2d(3, 64, 3, padding=1)
#             self.relu1 = nn.ReLU()
#             self.layer2 = nn.Conv2d(64, 128, 3, padding=1)
#             self.relu2 = nn.ReLU()
#             self.layer3 = nn.Linear(128 * 32 * 32, 10)
        
#         def forward(self, x):
#             x = self.relu1(self.layer1(x))
#             x = self.relu2(self.layer2(x))
#             x = x.view(x.size(0), -1)
#             x = self.layer3(x)
#             return x
    
#     model = SimpleModel()
    
#     # Get statistics before equalization
#     print("\nBefore Equalization:")
#     print("-" * 50)
#     stats_before = get_equalization_stats(model)
#     print(f"Compatible pairs: {stats_before['compatible_pairs']}")
#     for i, pair in enumerate(stats_before['layer_pairs']):
#         print(f"\nPair {i+1}:")
#         print(f"  {pair['first_layer']} -> {pair['second_layer']}")
#         print(f"  First layer range (mean): {pair['first_range_mean']:.6f}")
#         print(f"  Second layer range (mean): {pair['second_range_mean']:.6f}")
#         print(f"  Range ratio: {pair['range_ratio']:.4f}")
    
#     # Apply equalization
#     print("\n" + "=" * 50)
#     print("Applying Equalization...")
#     print("=" * 50)
#     pair_count = equalize_cross_layer_pairs(model, verbose=True)
#     print(f"\nEqualized {pair_count} layer pairs")
    
#     # Get statistics after equalization
#     print("\n" + "=" * 50)
#     print("After Equalization:")
#     print("-" * 50)
#     stats_after = get_equalization_stats(model)
#     for i, pair in enumerate(stats_after['layer_pairs']):
#         print(f"\nPair {i+1}:")
#         print(f"  {pair['first_layer']} -> {pair['second_layer']}")
#         print(f"  First layer range (mean): {pair['first_range_mean']:.6f}")
#         print(f"  Second layer range (mean): {pair['second_range_mean']:.6f}")
#         print(f"  Range ratio: {pair['range_ratio']:.4f}")
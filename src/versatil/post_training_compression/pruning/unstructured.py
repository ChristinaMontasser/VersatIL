import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# class ElementwiseL2Unstructured(prune.BasePruningMethod):
#     PRUNING_TYPE = "unstructured"

#     def __init__(self, amount):
#         self.amount = amount

#     def compute_mask(self, t, default_mask):
#         # This is only used by direct apply().
#         # global_unstructured overrides this with importance_scores.
#         mask = default_mask.clone()

#         num_params = t.numel()
#         num_to_prune = prune._compute_nparams_toprune(
#             self.amount, num_params
#         )

#         if num_to_prune == 0:
#             return mask

#         scores = t.abs().flatten()

#         threshold, _ = torch.kthvalue(
#             scores,
#             num_to_prune
#         )

#         mask[scores.reshape_as(t) <= threshold] = 0
#         return mask

"""Global L1 unstructured weight pruning."""

from torch import nn
from torch.nn.utils import prune

from versatil.post_training_compression.constants import (
    PrunableLayerType,
    PruningTargetAttribute,
)
from versatil.post_training_compression.pruning.base import BasePruner


class UnstructuredPruner(BasePruner):
    """Global L1 unstructured pruning across all target layers."""

    def __init__(
        self,
        amount: float,
        layer_types: list[str] | None = None,
    ) -> None:
        """Initialize with pruning amount and target layer types.

        Args:
            amount: Fraction of weights to zero, must be in (0, 1).
            layer_types: PrunableLayerType values to target. Defaults to
                convolution and linear layers (normalization scales and
                embedding tables are usually not good pruning targets).

        Raises:
            ValueError: If amount is not in the open interval (0, 1).
        """
        if amount <= 0.0 or amount >= 1.0:
            raise ValueError(f"Pruning amount must be in (0, 1), got {amount}")
        self._amount = amount
        if layer_types is None:
            layer_types = [
                PrunableLayerType.CONV1D.value,
                PrunableLayerType.CONV2D.value,
                PrunableLayerType.LINEAR.value,
                PrunableLayerType.EMBEDDING.value, #changed by christy
            ]
        self._layer_types = tuple(
            PrunableLayerType(name).to_module_type() for name in layer_types
        )

    @property
    def amount(self) -> float:
        """Fraction of weights to prune."""
        return self._amount

    @property
    def layer_types(self) -> tuple[type[nn.Module], ...] | None:
        """Module types targeted for pruning. None means all."""
        return self._layer_types
    

    # def _build_importance_scores(self, prunable_params: List[Tuple[nn.Module, str]]) -> Dict[Tuple[nn.Module, str], torch.Tensor]:
    #     """Compute normalized magnitude importance scores per-prunable parameter.

    #     importance = abs(w) / (||w||_2 + eps)
    #     """
    #     eps = 1e-9
    #     importance: Dict[Tuple[nn.Module, str], torch.Tensor] = {}
    #     for module, name in prunable_params:
    #         w = getattr(module, name)
    #         # Use absolute values; avoid NaNs if all-zero
    #         norm = float(w.norm(p=2).item()) if w.numel() > 0 else 0.0
    #         denom = norm + eps
    #         importance[(module, name)] = w.abs() / denom
    #     return importance

    def prune(self, module: nn.Module, calibration_loader: DataLoader | None = None, policy: Policy | None = None) -> tuple[int, int]:
        """Apply global L1 unstructured pruning.

        Identifies all target layers, applies global L1 unstructured pruning,
        then removes the pruning reparametrization to make weights permanent.

        Args:
            module: Neural network module to prune.

        Returns:
            Tuple of (total_parameters, zero_parameters).
        """
        parameters_to_prune = [
            (child_module, PruningTargetAttribute.WEIGHT.value)
            for child_module in module.modules()
            if isinstance(
                getattr(child_module, PruningTargetAttribute.WEIGHT.value, None),
                nn.Parameter,
            )
            and isinstance(child_module, self._layer_types)
        ]
        if not parameters_to_prune:
            raise ValueError(
                "Unstructured pruning selected no modules; the target module "
                f"contains no {[t.__name__ for t in self._layer_types]} layers."
            )
        #Added manually, the importance score will treat make the prunning method as L2
        #importance_scores = self._build_importance_scores(parameters_to_prune)
        # Dictionary to store pre-calculated L2 importance scores
        # importance_scores = {}

        # # Compute element-wise L2 scores for each parameter
        # for module, name in parameters_to_prune:
        #     weight_tensor = getattr(module, name)
        #     # Compute exact element-wise L2 norm
        #     l2_score = torch.sqrt(torch.pow(weight_tensor, 2))
        #     importance_scores[module] = l2_score




        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self._amount,
            #importance_scores=importance_scores,
)
        for child_module, name in parameters_to_prune:
            prune.remove(child_module, name)
        return self.compute_sparsity(module)

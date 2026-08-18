"""Iterative Calibration Pruner (ICP) implementation.

This module provides `IterativeCalibrationPruner`, a `BasePruner` that
implements the Iterative Calibration Pruning algorithm described in the
paper. It preserves the public API of the original class while correcting
layer-wise calibration to use recorded activations from a dense model.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple

import logging
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from versatil.explainability.attribution.policy import (
    prepare_policy_observation_for_explanation,
)
from versatil.models.policy import Policy

from versatil.post_training_compression.pruning.base import BasePruner

logger = logging.getLogger(__name__)

class IterativeCalibrationPruner(BasePruner):
    """Iterative Calibration Pruner (ICP).

    Preserves the original constructor signature and public API while
    implementing correct layer-wise calibration using dense-model
    activations collected via forward hooks.
    """

    def __init__(
        self,
        T: int = 10,
        initial_sparsity: float = 0.1,
        final_sparsity: float = 0.3,
        calibration_steps: int = 16,
    ) -> None:
        """Create an ICP pruner.

        Args:
            model: The mutable (to-be-pruned) model instance.
            T: Number of iterative pruning steps.
            initial_sparsity: Starting sparsity fraction (s_i).
            final_sparsity: Final sparsity fraction (s_f).
        """
        # Configurable schedule parameters only. The concrete module to prune
        # is provided to `prune()` at runtime so this class can be
        # instantiated by Hydra from YAML.
        self.T = int(T) # TODO: tha paper set the stop criteria either by target sparisty metric or allowed accuracy drop 
        self.si = float(initial_sparsity)
        self.sf = float(final_sparsity)
        self.calibration_steps = int(calibration_steps)

    # Public API preserved: prune(calibration_loader)
    def prune(self, module: nn.Module, calibration_loader: DataLoader | None = None, policy: Policy | None = None) -> Tuple[int, int]:
        """Run the iterative pruning pipeline using `calibration_loader`.

        Args:
            calibration_loader: DataLoader yielding calibration batches. Each
                batch is forwarded through the dense model to collect
                activations used for layer-wise tuning.

        Returns:
            Tuple of (total_parameters, zero_parameters) for the pruned model.
        """
        prunable_map = self._build_module_map(module)
        if not prunable_map:
            raise ValueError("No matching prunable modules found in the model.")

        dense_model: nn.Module | None = None
        dense_map: OrderedDict[str, nn.Module] | None = None
        common_names = list(prunable_map.keys())

        if calibration_loader is not None:
            dense_model = deepcopy(module).eval()
            dense_map = self._build_module_map(dense_model)

            common_names = [name for name in prunable_map.keys() if name in dense_map]
            if not common_names:
                raise ValueError(
                    "No matching prunable modules found between model and dense_model."
                )

        prunable_params = [(prunable_map[n], "weight") for n in common_names]

        previous_s = 0.0

        for t in range(1, self.T + 1):
            current_s = self._sparsity_at_step(t)

            current_amount = (
                current_s - previous_s
            ) / (1.0 - previous_s)

            logger.info(
                f"Step {t}: target sparsity = {current_s:.3f}, "
                f"previous sparsity = {previous_s:.3f}, "
                f"incremental pruning amount = {current_amount:.3f} "
                f"(of {1.0 - previous_s:.3f} remaining)"
            )

            importance_scores = self._build_importance_scores(prunable_params)

            prune.global_unstructured(
                prunable_params,
                pruning_method=prune.L1Unstructured,
                importance_scores=importance_scores,
                amount=current_amount,
            )

            previous_s = current_s
            if (
                calibration_loader is not None
                and dense_model is not None
                and dense_map is not None
            ):
                logger.info(
                    "IterativeCalibrationPruner: calibration_loader provided; starting calibration pass for pruning step %s/%s.",
                    t,
                    self.T,
                )
                self._stream_and_calibrate(
                    dense_root=dense_model,
                    dense_map=dense_map,
                    prunable_map=prunable_map,
                    calibration_loader=calibration_loader,
                    policy=policy,
                    module_names=common_names,
                    max_calibration_batches=self.calibration_steps,
                )
            elif calibration_loader is None:
                logger.info(
                    "IterativeCalibrationPruner: no calibration_loader supplied; skipping calibration/tuning."
                )

        # Finalize pruning: make masks permanent
        for m, name in prunable_params:
            prune.remove(m, name)

        return self.compute_sparsity(module)

    # Helper utilities
    def _build_module_map(self, model: nn.Module) -> "OrderedDict[str, nn.Module]":
        """Return an ordered dict mapping fully-qualified module names to modules.

        Only includes modules that are instance of Conv2d or Linear and have a
        `weight` parameter.
        """
        module_map: "OrderedDict[str, nn.Module]" = OrderedDict()
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                module_map[name] = module
        return module_map

    def _sparsity_at_step(self, t: int) -> float:
        """Progressive cubic sparsity schedule: s_t = s_f + (s_i - s_f) * (1 - t/T)^3"""
        s_t =  float(self.sf + (self.si - self.sf) * ((1.0 - t / float(self.T)) ** 3))
        logger.info(f"Step {t}: target sparsity = {s_t:.3f}")  # ADD THIS
        return s_t
       

    def _build_importance_scores(self, prunable_params: List[Tuple[nn.Module, str]]) -> Dict[Tuple[nn.Module, str], torch.Tensor]:
        """Compute normalized magnitude importance scores per-prunable parameter.

        importance = abs(w) / (||w||_2 + eps)
        """
        eps = 1e-9
        importance: Dict[Tuple[nn.Module, str], torch.Tensor] = {}
        for module, name in prunable_params:
            w = getattr(module, name)
            # Use absolute values; avoid NaNs if all-zero
            norm = float(w.norm(p=2).item()) if w.numel() > 0 else 0.0
            denom = norm + eps
            importance[(module, name)] = w.abs() / denom
        return importance

    def _collect_dense_activations(self, dense_map: "OrderedDict[str, nn.Module]", calibration_loader: DataLoader) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """(Deprecated) kept for compatibility but not used.

        The class now uses `_stream_and_calibrate` for streaming per-batch
        activation collection and immediate calibration. This method is
        retained only for API completeness and will return an empty mapping.
        """
        return {name: [] for name in dense_map.keys()}

    @staticmethod
    def _normalize_tensor(x: torch.Tensor) -> torch.Tensor:
        """Flatten loader-added wrappers when a tensor already carries a batch axis."""
        if x.dim() == 5:
            return x.reshape(-1, *x.shape[2:])
        return x

    def _prepare_dense_input(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | list[torch.Tensor] | torch.Tensor,
        policy: Policy | None,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor:
        """Convert a calibration batch into the exact dense-model input once."""
        if isinstance(batch, dict):
            normalized_batch = {
                key: self._normalize_tensor(value)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            observation = normalized_batch.get("observation")
            if not isinstance(observation, dict):
                raise ValueError(
                    "Calibration batches with dict inputs must contain an 'observation' mapping."
                )
            if policy is None:
                raise ValueError(
                    "Calibration batches with observation dicts require a policy."
                )

            prepared_observation = prepare_policy_observation_for_explanation(
                policy=policy,
                observation=observation,
                preprocess_observation=False,
            )
            prepared_observation.pop("language_instruction", None)
            return prepared_observation

        if isinstance(batch, tuple):
            return tuple(
                self._normalize_tensor(value) if isinstance(value, torch.Tensor) else value
                for value in batch
            )

        if isinstance(batch, list):
            return tuple(
                self._normalize_tensor(value) if isinstance(value, torch.Tensor) else value
                for value in batch
            )

        if isinstance(batch, torch.Tensor):
            return self._normalize_tensor(batch)

        raise TypeError(f"Unsupported calibration batch type: {type(batch)!r}")
        


    def _bias_variance_correction(
        self,
        pruned_module: nn.Module,
        dense_module: nn.Module,
    ) -> None:
        """Correct sparse weights to match dense weight statistics.

        The correction is performed independently for each output channel
        (Conv2d) or output feature (Linear).

        For each output feature/channel:

            W_corr = lambda * W_sparse
                    + mean_dense
                    - lambda * mean_sparse

        where:

            lambda = std_dense / (std_sparse + eps)

        Only surviving weights are used to estimate sparse statistics.
        Pruned weights remain exactly zero.
        """

        if not hasattr(pruned_module, "weight"):
            return

        if not hasattr(dense_module, "weight"):
            return

        with torch.no_grad():
            sparse_weight = pruned_module.weight
            dense_weight = dense_module.weight.to(sparse_weight.device)

            if sparse_weight.shape != dense_weight.shape:
                raise ValueError(
                    "Dense and sparse weight shapes do not match: "
                    f"{dense_weight.shape} vs {sparse_weight.shape}"
                )

            # -------------------------------------------------------------
            # Mask
            # -------------------------------------------------------------
            if hasattr(pruned_module, "weight_mask"):
                mask = pruned_module.weight_mask.to(
                    device=sparse_weight.device,
                    dtype=torch.bool,
                )
            else:
                mask = sparse_weight != 0

            # -------------------------------------------------------------
            # Flatten every output feature/channel independently.
            #
            # Conv2d:
            #   [out_channels, in_channels, kH, kW]
            #
            # Linear:
            #   [out_features, in_features]
            #
            # Both become:
            #   [num_output_features, parameters_per_output_feature]
            # -------------------------------------------------------------
            sparse_flat = sparse_weight.reshape(
                sparse_weight.shape[0], -1
            )
            dense_flat = dense_weight.reshape(
                dense_weight.shape[0], -1
            )
            mask_flat = mask.reshape(
                mask.shape[0], -1
            )

            corrected = sparse_flat.clone()

            eps = 1e-9

            for channel_idx in range(sparse_flat.shape[0]):
                channel_mask = mask_flat[channel_idx]

                # No surviving parameters in this output channel.
                if not channel_mask.any():
                    continue

                sparse_values = sparse_flat[channel_idx][channel_mask]
                dense_values = dense_flat[channel_idx]

                # Dense statistics are computed over the complete dense
                # channel; sparse statistics only over surviving weights.
                dense_mean = dense_values.mean()
                dense_std = dense_values.std(unbiased=False)

                sparse_mean = sparse_values.mean()
                sparse_std = sparse_values.std(unbiased=False)

                if not torch.isfinite(sparse_std) or sparse_std <= eps:
                    scale = torch.ones_like(dense_std)
                else:
                    scale = dense_std / (sparse_std + eps)

                corrected_values = (
                    scale * sparse_values
                    + dense_mean
                    - scale * sparse_mean
                )

                corrected[channel_idx][channel_mask] = corrected_values

            # Explicitly preserve all pruned entries as zero.
            corrected = corrected * mask_flat.to(corrected.dtype)

            sparse_weight.copy_(
                corrected.reshape_as(sparse_weight)
            )



    def _layer_wise_tuning(self, pruned_module: nn.Module, dense_module: nn.Module, activation_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Tune a single pruned layer using recorded input/output activation pairs.

        Args:
            pruned_module: The pruned module to optimize (Conv2d or Linear).
            dense_module: Corresponding dense module producing targets.
            activation_pairs: List of (input, target_output) tensors collected from dense model.
        """
        if not activation_pairs:
            return

        # Prepare optimizer only for existing Parameter objects. When pruning
        # has run, the live `weight` attribute can be a tensor/view (non-leaf)
        # while the real parameter lives at `weight_orig`. We therefore locate
        # the Parameter objects explicitly to avoid "can't optimize a
        # non-leaf Tensor" errors.
        def _find_param(module: nn.Module, base_name: str):
            # Prefer the `*_orig` parameter created by torch.nn.utils.prune
            cand_name = f"{base_name}_orig"
            if hasattr(module, cand_name):
                p = getattr(module, cand_name)
                if isinstance(p, nn.Parameter):
                    return p
            # Fall back to direct named_parameters (non-recursive)
            for n, p in module.named_parameters(recurse=False):
                if n == base_name and isinstance(p, nn.Parameter):
                    return p
            # Finally, check the attribute directly
            p = getattr(module, base_name, None)
            if isinstance(p, nn.Parameter):
                return p
            return None

        params = []
        w_param = _find_param(pruned_module, "weight")
        # Keep masked weights frozen without registering a new backward hook
        # on every tuning call. Hook accumulation was creating unnecessary
        # overhead across calibration passes.
        mask = None
        if hasattr(pruned_module, "weight_mask"):
            mask = pruned_module.weight_mask.detach()

        b_param = _find_param(pruned_module, "bias")
        if w_param is not None:
            params.append({"params": [w_param], "lr": 1e-5})
        if b_param is not None:
            params.append({"params": [b_param], "lr": 1e-4})
        if not params:
            return

        optimizer = torch.optim.Adam(params, weight_decay=0)
        criterion = nn.MSELoss()

        pruned_module.train()
        device = None
        #But there's another important issue: I am doing the layer optimization one batch at a time:
        for inp, target in activation_pairs:
            # Move activations to the module's device once
            if device is None:
                device = next(p for p in pruned_module.parameters() if p is not None).device
            inp = inp.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            out = pruned_module(inp)
            loss = criterion(out, target)
            loss.backward()
            if mask is not None and w_param is not None and w_param.grad is not None:
                w_param.grad.mul_(mask)
            optimizer.step()

        # Leave module in eval mode to be consistent with pruning framework expectations
        pruned_module.eval()

    

    def _stream_and_calibrate(
        self,
        dense_root: nn.Module,
        dense_map: "OrderedDict[str, nn.Module]",
        prunable_map: "OrderedDict[str, nn.Module]",
        calibration_loader: DataLoader,
        policy: Policy | None,
        module_names: List[str],
        max_calibration_batches: int,
    ) -> None:
        """Run layer-wise calibration against the frozen dense model.

        Three-phase process (matches paper exactly):
        
        Phase 1: Collect all activation pairs from dense model
        ├─ Forward calibration batches through dense model
        ├─ Record (input, output) for each prunable layer
        └─ Store as Dict[layer_name, List[(input, output)]]
        
        Phase 2: Apply corrections to each layer (per-feature, aggregated stats)
        ├─ Weight bias/variance correction (Eq. 3-4)
        └─ Activation bias correction (Eq. 5)
        
        Phase 3: Fine-tune each layer (on all collected activation pairs)
        └─ MSE loss on complete dataset

        The dense model is never modified.
        """

        def make_hook(
            name: str,
            storage: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        ):
            def hook(module, inp, out):
                if not isinstance(inp, (tuple, list)) or len(inp) == 0:
                    raise RuntimeError(
                        f"Expected tensor input for layer '{name}', "
                        f"got {type(inp)!r}."
                    )

                x = inp[0]
                if not isinstance(x, torch.Tensor):
                    raise RuntimeError(
                        f"Expected tensor input for layer '{name}', "
                        f"got {type(x)!r}."
                    )

                if not isinstance(out, torch.Tensor):
                    raise RuntimeError(
                        f"Expected tensor output for layer '{name}', "
                        f"got {type(out)!r}."
                    )

                storage[name] = (
                    x.detach().clone(),
                    out.detach().clone(),
                )

            return hook

        hooks = []
        storage: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        for name in module_names:
            hooks.append(
                dense_map[name].register_forward_hook(
                    make_hook(name, storage)
                )
            )

        # =====================================================================
        # PHASE 1: Collect all activation pairs from dense model
        # =====================================================================
        logger.info(
            "IterativeCalibrationPruner: Phase 1 - collecting activations from %d batches.",
            max_calibration_batches,
        )
        
        # Dict to store all collected activations per layer
        # Key: layer_name, Value: List[(dense_input, dense_output)]
        all_activations: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {
            name: [] for name in module_names
        }

        try:
            for batch_index, batch in enumerate(calibration_loader):
                if batch_index >= max_calibration_batches:
                    break

                storage.clear()

                dense_input = self._prepare_dense_input(batch, policy)

                # The dense model is the immutable reference model.
                with torch.no_grad():
                    if isinstance(dense_input, tuple):
                        dense_root(*dense_input)
                    else:
                        dense_root(dense_input)

                # Collect activations for this batch
                for name in module_names:
                    pair = storage.get(name)
                    if pair is None:
                        logger.warning(
                            "No activation pair collected for layer '%s' in batch %d.",
                            name,
                            batch_index,
                        )
                        continue

                    all_activations[name].append(pair)

                if (batch_index + 1) % max(1, max_calibration_batches // 4) == 0:
                    logger.debug(
                        "IterativeCalibrationPruner: collected %d batches.",
                        batch_index + 1,
                    )

        finally:
            for hook in hooks:
                hook.remove()

        # =====================================================================
        # PHASE 2: Apply bias/variance corrections (aggregated, per-feature)
        # =====================================================================
        logger.info(
            "IterativeCalibrationPruner: Phase 2 - applying bias/variance corrections."
        )

        for name in module_names:
            if not all_activations[name]:
                logger.warning(
                    "No activations collected for layer '%s', skipping corrections.",
                    name,
                )
                continue

            pruned_module = prunable_map[name]
            dense_module = dense_map[name]

            # Weight bias/variance correction (Eq. 3-4 from paper)
            # This operates on weight statistics and is independent of activations
            self._bias_variance_correction(
                pruned_module=pruned_module,
                dense_module=dense_module,
            )

            # Activation bias correction (Eq. 5 from paper)
            # Aggregate statistics over all activation pairs collected for this layer
            # This ensures bias correction is stable and uses full dataset statistics
            self._activation_bias_correction_aggregated(
                pruned_module=pruned_module,
                activation_pairs=all_activations[name],
            )

        # =====================================================================
        # PHASE 3: Layer-wise fine-tuning (on all collected activations)
        # =====================================================================
        logger.info(
            "IterativeCalibrationPruner: Phase 3 - fine-tuning each layer."
        )

        for name in module_names:
            if not all_activations[name]:
                logger.warning(
                    "No activations collected for layer '%s', skipping fine-tuning.",
                    name,
                )
                continue

            pruned_module = prunable_map[name]
            dense_module = dense_map[name]

            # Layer-wise fine-tuning (Eq. 6 from paper)
            # Pass all collected activation pairs for this layer at once
            # Optimizer will accumulate gradients and perform updates on full dataset
            self._layer_wise_tuning(
                pruned_module=pruned_module,
                dense_module=dense_module,
                activation_pairs=all_activations[name],
            )


    def _activation_bias_correction_aggregated(
        self,
        pruned_module: nn.Module,
        activation_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> None:
        """Correct sparse-layer bias using aggregated activation statistics.

        This is the aggregated version of activation bias correction (Eq. 5).
        Instead of correcting per-batch, we collect all activation pairs first,
        then compute mean statistics over the entire calibration set.

        For each output feature/channel:
        
            mean_dense_all = E[Y_dense over all samples]
            mean_sparse_all = E[Y_sparse over all samples]
            b_corr = b_sparse + (mean_dense_all - mean_sparse_all)

        This matches the paper's intent: use dataset-wide statistics for bias
        correction, not per-batch statistics.
        """

        if pruned_module.bias is None:
            return

        if not activation_pairs:
            return

        device = pruned_module.weight.device
        
        # Accumulate sums and counts for each output feature
        was_training = pruned_module.training
        pruned_module.eval()

        with torch.no_grad():
            # Collect sparse outputs for all samples
            dense_outputs = []
            sparse_outputs = []

            for dense_input, dense_output in activation_pairs:
                x = dense_input.to(device)
                y_dense = dense_output.to(device)

                y_sparse = pruned_module(x)

                if y_sparse.shape != y_dense.shape:
                    raise ValueError(
                        "Dense and sparse activation shapes do not match: "
                        f"{y_dense.shape} vs {y_sparse.shape}"
                    )

                dense_outputs.append(y_dense)
                sparse_outputs.append(y_sparse)

            # Concatenate all batches
            # Conv2d: [N, C, H, W] -> concatenate over N dimension
            # Linear: [N, F] -> concatenate over N dimension
            y_dense_all = torch.cat(dense_outputs, dim=0)
            y_sparse_all = torch.cat(sparse_outputs, dim=0)

            # Determine output feature dimension and compute means
            if isinstance(pruned_module, nn.Conv2d):
                # Conv2d output: [N, C, H, W]
                # Reduce over batch, height, width -> [C]
                reduce_dims = (0, 2, 3)

            elif isinstance(pruned_module, nn.Linear):
                # Linear output: [N, F]
                # Reduce over batch -> [F]
                reduce_dims = (0,)

            else:
                # Generic fallback: assume dimension 1 is the feature axis
                if y_dense_all.dim() < 2:
                    raise ValueError(
                        "Cannot determine output-feature dimension for "
                        f"{type(pruned_module).__name__}."
                    )
                reduce_dims = tuple(
                    dim for dim in range(y_dense_all.dim())
                    if dim != 1
                )

            # Compute mean over all samples (not per-batch)
            dense_mean = y_dense_all.mean(dim=reduce_dims)
            sparse_mean = y_sparse_all.mean(dim=reduce_dims)

            bias_delta = dense_mean - sparse_mean

            if bias_delta.shape != pruned_module.bias.shape:
                raise ValueError(
                    "Activation bias correction has incompatible shape: "
                    f"delta={bias_delta.shape}, "
                    f"bias={pruned_module.bias.shape}"
                )

            pruned_module.bias.add_(bias_delta)

        if was_training:
            pruned_module.train()
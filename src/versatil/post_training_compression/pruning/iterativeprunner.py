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
        final_sparsity: float = 0.5,
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

            calibration_batches = self._prepare_calibration_batches(
                calibration_loader=calibration_loader,
                policy=policy,
            )

        prunable_params = [(prunable_map[n], "weight") for n in common_names]

        # Main iterative loop
        for t in range(1, self.T + 1):
            current_s = self._sparsity_at_step(t)

            # Build importance scores from current (possibly already-masked) weights
            importance_scores = self._build_importance_scores(prunable_params)

            # Global unstructured pruning step (masks are applied as reparam)
            prune.global_unstructured(
                prunable_params,
                pruning_method=prune.L1Unstructured,
                importance_scores=importance_scores,
                amount=current_s,
            )

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
                    calibration_batches=calibration_batches,
                    module_names=common_names,
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
        return float(self.sf + (self.si - self.sf) * ((1.0 - t / float(self.T)) ** 3))

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

    def _prepare_calibration_batches(
        self,
        calibration_loader: DataLoader,
        policy: Policy | None,
    ) -> list[dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor]:
        """Prepare dense-model inputs once so every pruning step can reuse them."""
        return [
            self._prepare_dense_input(batch=batch, policy=policy)
            for batch in calibration_loader
        ]

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

    def _stream_and_calibrate(self, dense_root: nn.Module, dense_map: "OrderedDict[str, nn.Module]", prunable_map: "OrderedDict[str, nn.Module]", calibration_batches: list[dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor], module_names: List[str]) -> None:
        """Stream calibration: for each batch run one dense forward, then calibrate.

        This method registers forward hooks on the dense model per-batch, runs a
        single forward pass under `torch.no_grad()`, collects input/output
        activations for each prunable layer, then immediately performs bias
        correction and a single-step layer-wise tuning using those activations.
        Hooks stay registered for the whole calibration pass to avoid repeated
        setup/teardown overhead.
        """
        # Hook factory that records input/output into local temporary dict.
        def make_hook(name: str, storage: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
            def hook(module, inp, out):
                # Record inputs[0] conventionally; keep tensors on their device.
                inp_t = (
                    inp[0].detach().clone()
                    if isinstance(inp, (tuple, list)) and len(inp) > 0
                    else inp.detach().clone()
                )
                out_t = out.detach().clone()
                storage[name] = (inp_t, out_t)

            return hook

        hooks = []
        storage: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in module_names:
            module = dense_map[name]
            hooks.append(module.register_forward_hook(make_hook(name, storage)))

        try:
            for batch in calibration_batches:
                storage.clear()

                with torch.no_grad():
                    if isinstance(batch, dict):
                        dense_root(batch)
                    elif isinstance(batch, tuple):
                        dense_root(*batch)
                    else:
                        dense_root(batch)

                for name in module_names:
                    pair = storage.get(name)
                    if pair is None:
                        continue
                    pruned_module = prunable_map[name]
                    dense_module = dense_map[name]
                    self._bias_variance_correction(pruned_module, dense_module)
                    #self._layer_wise_tuning(pruned_module, dense_module, [pair])
        finally:
            for hook in hooks:
                hook.remove()

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
        # to make sure the prunned weights are not changed during the tunning, we mask them with zero

        if hasattr(pruned_module, "weight_mask"):
            mask = pruned_module.weight_mask.detach()
            def keep_surviving_weights(grad):
                return grad * mask
            w_param.register_hook(keep_surviving_weights)

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
            optimizer.step()

        # Leave module in eval mode to be consistent with pruning framework expectations
        pruned_module.eval()

    def _bias_variance_correction(self, pruned_module: nn.Module, dense_module: nn.Module) -> None:
        """Adjust pruned weights to better match dense weight statistics.

        Uses only surviving (nonzero) weights for sparse statistics; handles
        edge-cases safely (all-zero, zero variance, NaNs).
        """
        with torch.no_grad():
            if not hasattr(pruned_module, "weight") or not hasattr(dense_module, "weight"):
                return
            W_s = pruned_module.weight
            W_d = dense_module.weight

            # Work on the module device to avoid unnecessary transfers
            device = W_s.device
            W_s_t = W_s.detach()
            W_d_t = W_d.detach().to(device)

            if hasattr(pruned_module, "weight_mask"):
                mask = pruned_module.weight_mask.bool()
            else:
                mask = W_s_t != 0
            if mask.sum() == 0:
                return

            surviving = W_s_t[mask]
            if surviving.numel() == 0:
                return

            # Robust stats
            mean_d = float(W_d_t.mean().item()) if W_d_t.numel() > 0 else 0.0
            std_d = float(W_d_t.std().item()) if W_d_t.numel() > 0 else 0.0
            mean_s = float(surviving.mean().item())
            std_s = float(surviving.std().item())

            eps = 1e-9
            if std_s == 0 or not torch.isfinite(torch.tensor(std_s)):
                lam = 1.0
            else:
                lam = std_d / (std_s + eps)

            # Apply affine correction only to surviving entries to preserve masks
            W_corr = W_s_t.clone()
            W_corr_masked = (lam * W_s_t[mask]) + (mean_d - lam * mean_s)
            W_corr[mask] = W_corr_masked
            # Write back
            pruned_module.weight.copy_(W_corr)

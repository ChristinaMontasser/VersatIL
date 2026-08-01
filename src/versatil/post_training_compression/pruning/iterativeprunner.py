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

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import prune
import logging
from versatil.data.tokenization import Tokenizer
from versatil.models.policy import Policy
from versatil.explainability.attribution.policy import prepare_policy_observation_for_explanation

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
        # Dense reference used as teacher for activations/outputs
        dense_model = deepcopy(module).eval()
        dense_map = self._build_module_map(dense_model)

        # Ensure we only operate on modules present in both models
        common_names = [n for n in prunable_map.keys() if n in dense_map]
        if not common_names:
            raise ValueError("No matching prunable modules found between model and dense_model.")

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

            # Stream calibration: for each batch run one dense forward, then
            # immediately calibrate every layer using that batch's recorded
            # activations. This keeps memory usage constant w.r.t dataset size.
            # Perform streaming calibration/tuning only if a calibration
            # DataLoader was supplied. This keeps the method compatible with
            # the compressor which may not provide calibration data.
            # TODO: the problem that was no calibration loader was uploaded. 
            if calibration_loader is not None:
                try:
                    n_batches = len(calibration_loader)
                except Exception:
                    n_batches = None
                logger.info(
                    "IterativeCalibrationPruner: calibration_loader provided (batches=%s); starting streaming calibration.",
                    n_batches,
                )
                self._stream_and_calibrate(
                    dense_root=dense_model,
                    dense_map=dense_map,
                    prunable_map=prunable_map,
                    calibration_loader=calibration_loader,
                    module_names=common_names,
                    policy= policy
                )
            else:
                logger.info("IterativeCalibrationPruner: no calibration_loader supplied; skipping streaming calibration/tuning.")

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

    def _stream_and_calibrate(self, dense_root: nn.Module, dense_map: "OrderedDict[str, nn.Module]", prunable_map: "OrderedDict[str, nn.Module]", calibration_loader: DataLoader, module_names: List[str], policy: Policy | None) -> None:
        """Stream calibration: for each batch run one dense forward, then calibrate.

        This method registers forward hooks on the dense model per-batch, runs a
        single forward pass under `torch.no_grad()`, collects input/output
        activations for each prunable layer, then immediately performs bias
        correction and a single-step layer-wise tuning using those activations.
        Hooks are removed after each batch to avoid accumulation.
        """
        # Hook factory that records input/output into local temporary dict
        def make_hook(name: str, storage: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
            def hook(module, inp, out):
                # Record inputs[0] conventionally; keep tensors on their device
                inp_t = inp[0].detach().clone() if isinstance(inp, (tuple, list)) and len(inp) > 0 else inp.detach().clone()
                out_t = out.detach().clone()
                storage[name] = (inp_t, out_t)

            return hook

        # Iterate over calibration batches streaming
        for batch in calibration_loader:
            # if isinstance(batch, dict):
              #  logger.info("Calibration batch keys: %s", batch.keys())
            hooks = []
            storage: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            # Register hooks for every dense prunable module
            for name in module_names:
                module = dense_map[name]
                h = module.register_forward_hook(make_hook(name, storage))
                hooks.append(h)

            # Run dense forward once per batch without gradient
            try:
                with torch.no_grad():
                    # Normalize tensor shapes that DataLoader may wrap with an
                    # extra batch-dimension. Common case: a list of 4D
                    # tensors (each already a batch) becomes a 5D tensor with
                    # shape (1, T, C, H, W) when `batch_size=1` in DataLoader.
                    def _normalize_arg(x):
                        if isinstance(x, torch.Tensor):
                            if x.dim() == 5:
                                # Collapse leading two dims (B, T, C, H, W)
                                return x.reshape(-1, *x.shape[2:])
                        return x
                    if isinstance(batch, dict):
                       # logger.info("Top-level keys: %s", batch.keys())

                        obs = batch.get("observation")

                        # if isinstance(obs, dict):
                        #    # logger.info("Observation keys: %s", obs.keys())

                        #     for k, v in obs.items():
                        #         logger.info(
                        #             "obs[%s]: type=%s shape=%s",
                        #             k,
                        #             type(v),
                        #             getattr(v, "shape", None)
                        #         )
                    if isinstance(batch, dict):
                        batch = {k: _normalize_arg(v) for k, v in batch.items()}
                        # import inspect

                        # logger.info(
                        #     "Encoder forward signature: %s",
                        #     inspect.signature(dense_root.forward)
                        # )
                        prepared_observation = prepare_policy_observation_for_explanation(policy= policy, observation = batch["observation"], preprocess_observation=False )
                        prepared_observation.pop("language_instruction", None)
                        dense_root(prepared_observation)
                    elif isinstance(batch, (list, tuple)):
                        batch = tuple(_normalize_arg(v) for v in batch)
                        # Expand as positional args
                        dense_root(*batch)
                    else:
                        batch = _normalize_arg(batch)
                        dense_root(batch)
            finally:
                # Remove hooks promptly to avoid leaks
                for h in hooks:
                    h.remove()

            # Immediately calibrate every prunable layer using this batch's activations
            for name in module_names:
                pair = storage.get(name)
                if pair is None:
                    # Layer may not have run for this input (e.g., conditional branches)
                    continue
                pruned_module = prunable_map[name]
                dense_module = dense_map[name]
                # Bias/variance correction first
                self._bias_variance_correction(pruned_module, dense_module)
                # Single-step layer-wise tuning with this batch's activation pair
                self._layer_wise_tuning(pruned_module, dense_module, [pair])

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

"""PT2E quantization workflow."""

import copy
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchao.quantization.pt2e import HistogramObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.composable_quantizer import (
    ComposableQuantizer,
)

from versatil.data.episodic_dataset import EpisodicDataset
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.post_training_compression.export import (
    _export_with_dynamic_batch,
    build_example_inputs,
    export_policy,
)
from versatil.post_training_compression.policy_loading import load_float_policy_context
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.adaround import (
    AdaRoundConfig,
    apply_adaround,
    collect_linear_inputs,
)
from versatil.quantization.constants import FXNodePattern, QuantizationMode
from versatil.quantization.module_target import PT2EQuantizationModuleTarget
from versatil.quantization.pt2e.backends.base import BasePT2EBackend
from versatil.quantization.pt2e.smooth_quant import (
    SmoothQuantPreparer,
    apply_smooth_quant_scales_to_graph,
    remove_unused_exported_state,
)
from versatil.quantization.workflows.base import (
    BaseQuantizationWorkflow,
    PolicyContext,
    QuantizedContext,
)

logger = logging.getLogger(__name__)

#---------------------------------------------------------------------------
#helper to understand the modeules' distributions  
def _analyze_pt2e_observers(prepared: nn.Module) -> None:
    """Log PT2E modules, layer types, and activation observer parameters."""

    logger.info("PT2E model: %s", prepared.__class__.__name__)

    # Print the FX graph once
    graph = getattr(prepared, "graph", None)
    if graph is not None:
        for node in graph.nodes:
            if node.op == "call_module":
                try:
                    target_module = prepared.get_submodule(node.target)
                    module_type = target_module.__class__.__name__
                except Exception:
                    module_type = "Unknown"

                logger.info(
                    "PT2E layer: name=%s type=%s category=%s target=%s",
                    node.name,
                    module_type,
                    _classify_module(module_type),
                    node.target,
                )
            else:
                logger.info(
                    "PT2E node: name=%s op=%s target=%s",
                    node.name,
                    node.op,
                    node.target,
                )

    # Analyze observers
    for name, module in prepared.named_modules():
        if not isinstance(module, HistogramObserver):
            continue

        if (
            not torch.isfinite(module.min_val).all()
            or not torch.isfinite(module.max_val).all()
        ):
            logger.warning(
                "PT2E observer [%s]: NOT CALIBRATED "
                "(min=%s, max=%s)",
                name,
                module.min_val,
                module.max_val,
            )
            continue

        optimized_min, optimized_max = module._non_linear_param_search()
        scale, zero_point = module.calculate_qparams()

        logger.info(
            "PT2E observer [%s]: category=ACTIVATION "
            "observed_range=[%.6f, %.6f], "
            "optimized_range=[%.6f, %.6f], "
            "scale=%.6f, zero_point=%d",
            name,
            module.min_val.item(),
            module.max_val.item(),
            optimized_min.item(),
            optimized_max.item(),
            scale.item(),
            zero_point.item(),
        )


def _classify_module(module_type: str) -> str:
    if "Observer" in module_type:
        return "ACTIVATION_OBSERVER"
    if module_type in {"Linear", "Conv1d", "Conv2d", "Conv3d"}:
        return "LINEAR_OR_CONV"
    if module_type in {"ReLU", "GELU", "SiLU", "Sigmoid", "Tanh"}:
        return "ACTIVATION"
    if module_type in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}:
        return "NORMALIZATION"

    return "OTHER"


def _log_silu_quantization_status(converted: nn.Module) -> None:
    """Log whether converted SiLU outputs have quantization boundaries."""
    graph = getattr(converted, "graph", None)
    if graph is None:
        return

    silu_nodes = [node for node in graph.nodes if "silu" in str(node.target)]
    quantized_nodes = []
    for node in silu_nodes:
        if any("quantize_per_tensor" in str(user.target) for user in node.users):
            quantized_nodes.append(node.name)

    logger.info(
        "SiLU quantization status: total=%d quantized_outputs=%d "
        "floating_outputs=%d quantized_nodes=%s",
        len(silu_nodes),
        len(quantized_nodes),
        len(silu_nodes) - len(quantized_nodes),
        quantized_nodes,
    )
#---------------------------------------------------------------------------


class PT2EQuantizationWorkflow(BaseQuantizationWorkflow):
    """PT2E graph quantization workflow."""

    def __init__(self, targets: list[PT2EQuantizationModuleTarget]) -> None:
        """Initialize with PT2E module targets.

        Args:
            targets: module-level PT2E quantization targets.
        """
        if not targets:
            raise ValueError("PT2EQuantizationWorkflow requires at least one target.")
        self._targets = targets
        if self.is_qat:
            raise NotImplementedError("PT2E QAT configuration is not supported yet.")

    @property
    def targets(self) -> list[PT2EQuantizationModuleTarget]:
        """Return PT2E quantization targets."""
        return self._targets

    @property
    def pt2e_backend(self) -> BasePT2EBackend:
        """Return the first target backend for runtime environment setup."""
        return self._targets[0].pt2e_backend

    @property
    def pt2e_backend_names(self) -> tuple[str, ...]:
        """Return serialized PT2E backend names used by all targets."""
        return tuple(target.pt2e_backend.name for target in self._targets)

    @property
    def quantization_mode(self) -> str:
        """Return ``pt2e`` because this workflow quantizes an exported graph."""
        return QuantizationMode.PT2E.value

    @property
    def needs_calibration(self) -> bool:
        """Static PT2E requires calibration, dynamic does not."""
        return any(target.needs_calibration for target in self._targets)

    @property
    def is_qat(self) -> bool:
        """Return whether the PT2E backend is configured for QAT."""
        return any(target.pt2e_backend.is_qat for target in self._targets)

    def load_policy_context(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
    ) -> PolicyContext:
        """Load the policy checkpoint required by PT2E quantization.

        Args:
            checkpoint_path: Directory containing the training checkpoint.
            checkpoint_name: Checkpoint filename to load from the directory.

        Returns:
            Float policy context used for PT2E export, preparation,
            calibration, and conversion.
        """
        context = load_float_policy_context(
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
        )
        context.policy.to(device=torch.device("cpu"))
        return context

    def quantize(
        self,
        context: PolicyContext,
        exportable: ExportablePolicy,
        calibration_steps: int,
    ) -> QuantizedContext:
        """Export, prepare, optionally calibrate, and convert with PT2E.

        Args:
            context: Loaded float policy context.
            exportable: Policy wrapper exposing positional tensor inputs for
                ``torch.export``.
            calibration_steps: Maximum number of training batches used for
                static PT2E calibration.

        Returns:
            Float exported model, PT2E-converted model, and example inputs.

        Raises:
            ValueError: If a target path is invalid or targets overlap.
        """
        self.validate_targets(model=context.policy)
        calibration = self._build_calibration(
            context=context,
            exportable=exportable,
            targets=self._targets,
            calibration_steps=calibration_steps,
        )
        # Export always uses synthetic batch>=2 example inputs: a raw
        # training batch with batch_size=1 would trip torch.export's 0/1
        # specialization on the dynamic batch dimension. Calibration still
        # runs on real dataloader batches.
        smooth_quant_scales = self._prepare_smooth_quant(
            context=context,
            exportable=exportable,
            calibration=calibration,
        )
        self._prepare_adaround(
            context=context,
            exportable=exportable,
            calibration=calibration,
        )
        example_inputs = build_example_inputs(
            exportable=exportable,
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
        )
        device = next(context.policy.parameters()).device
        example_inputs = self._move_to_device(example_inputs, device=device)
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)
        # This is the reciprocal of this scale
        apply_smooth_quant_scales_to_graph(
            model=exported,
            scales=smooth_quant_scales,
        )
        # prepare_pt2e/convert_pt2e mutate the exported graph in place; keep a
        # genuinely float copy so the report compares against the pre-
        # quantization model instead of the mutated graph itself.
        float_exported = copy.deepcopy(exported)
        converted = self._convert_exported_model(
            exported=exported,
            targets=self._targets,
            calibration=calibration,
        )
        if isinstance(converted, torch.fx.GraphModule):
            converted = _export_with_dynamic_batch(
                model=converted,
                example_inputs=example_inputs,
            ).module()
            removed_state = remove_unused_exported_state(model=converted)
            logger.info("Removed %d unused PT2E exported state entries.", removed_state)
            logger.info(
                "Re-exported converted PT2E graph to remove unused float state."
            )
        return QuantizedContext(
            float_model=float_exported,
            quantized_model=converted,
            example_inputs=example_inputs,
            quantization_workflow=QuantizationWorkflow.PT2E.value,
        )

    @staticmethod
    def _move_to_device(value, device: torch.device):
        """Move nested export inputs to the policy device."""
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            moved = tuple(
                PT2EQuantizationWorkflow._move_to_device(item, device)
                for item in value
            )
            if all(item is original for item, original in zip(moved, value)):
                return value
            return moved
        if isinstance(value, list):
            moved = [
                PT2EQuantizationWorkflow._move_to_device(item, device)
                for item in value
            ]
            if all(item is original for item, original in zip(moved, value)):
                return value
            return moved
        if isinstance(value, dict):
            moved = {
                key: PT2EQuantizationWorkflow._move_to_device(item, device)
                for key, item in value.items()
            }
            if all(item is value[key] for key, item in moved.items()):
                return value
            return moved
        return value

    def _prepare_smooth_quant(
        self,
        context: PolicyContext,
        exportable: ExportablePolicy,
        calibration: CalibrationDataProvider | None,
    ) -> dict[str, torch.Tensor]:
        """Apply configured SmoothQuant preparation before PT2E export."""
        smooth_targets = [target for target in self._targets if target.smooth_quant]
        if not smooth_targets:
            return {}
        if calibration is None:
            raise ValueError(
                "SmoothQuant preparation requires calibration data, but no "
                "CalibrationDataProvider was created."
            )
        transformed: dict[str, torch.Tensor] = {}
        for target in smooth_targets:
            preparer = SmoothQuantPreparer(alpha=target.smooth_quant_alpha)
            activation_scales = preparer.collect_activation_scales(
                model=exportable,
                calibration=lambda: iter(calibration),
                module_filter=lambda name, module, target=target: (
                    target.contains_module(name)
                    and isinstance(module, (nn.Conv2d, nn.Linear))
                ),
            )
            selected = preparer.apply(
                model=context.policy,
                activation_scales=activation_scales,
            )
            if not selected:
                raise ValueError(
                    f"SmoothQuant target '{target.label}' selected no supported "
                    "Conv2d or Linear modules."
                )
            logger.info(
                "SmoothQuant target %s transformed %d modules: %s",
                target.label,
                len(selected),
                list(selected),
            )
            transformed.update(selected)
        return transformed

    @staticmethod
    def _build_calibration(
        context: PolicyContext,
        exportable: ExportablePolicy,
        targets: list[PT2EQuantizationModuleTarget],
        calibration_steps: int,
    ) -> CalibrationDataProvider | None:
        """Build calibration data for static PT2E quantization.

        Args:
            context: Loaded policy context containing the training dataloader
                config.
            exportable: Policy wrapper whose observation key order determines
                calibration batch layout.
            targets: PT2E targets that determine whether calibration is needed.
            calibration_steps: Maximum number of calibration batches.

        Returns:
            Calibration provider for static targets, or ``None`` when all
            targets are dynamic.
        """
        needs_calibration = any(
            target.needs_calibration or target.smooth_quant or target.adaround
            for target in targets
        )
        if not needs_calibration:
            return None
        if calibration_steps < 1:
            raise ValueError(
                "Static PT2E quantization requires calibration_steps >= 1, "
                f"got {calibration_steps}; converting uncalibrated observers "
                "produces garbage quantization parameters."
            )
        dataset = EpisodicDataset(
            zarr_path=context.config.task.dataset_schema.zarr_path,
            action_space=context.config.task.action_space,
            observation_space=context.observation_space,
            dataloader_config=context.config.task.dataloader,
            pred_horizon=context.config.task.prediction_horizon,
            obs_horizon=context.observation_horizon,
            train=True,
            seed=context.config.experiment.seed,
            augment_images=False,
        )
        dataset.set_normalizer(context.policy.normalizer)
        dataset.set_tokenizer(context.tokenizer)
        calibration_loader = DataLoader(
            dataset=dataset,
            batch_size=context.config.task.dataloader.batch_size,
            shuffle=False,
            num_workers=0,
        )
        return CalibrationDataProvider(
            dataloader=calibration_loader,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=calibration_steps,
        )

    def _prepare_adaround(
        self,
        context: PolicyContext,
        exportable: ExportablePolicy,
        calibration: CalibrationDataProvider | None,
    ) -> None:
        """Apply native PyTorch AdaRound before PT2E export."""
        adaround_targets = [target for target in self._targets if target.adaround]
        if not adaround_targets:
            return
        if calibration is None:
            raise ValueError("AdaRound requires a calibration data provider.")
        for target in adaround_targets:
            selected = [
                name
                for name, module in context.policy.named_modules()
                if target.contains_module(name)
                and isinstance(module, nn.Linear)
            ]
            if not selected:
                raise ValueError(
                    f"AdaRound target '{target.label}' selected zero eligible "
                    "nn.Linear modules."
                )
            inputs = collect_linear_inputs(
                model=context.policy,
                calibration=lambda: iter(calibration),
                forward=lambda batch: exportable(*batch),
                module_names=set(selected),
                storage_device="cpu",
            )
            config = AdaRoundConfig(
                iterations=target.adaround_iterations,
                regularization_strength=target.adaround_regularization_strength,
                beta_start=target.adaround_beta_start,
                beta_end=target.adaround_beta_end,
                warm_start=target.adaround_warm_start,
                bitwidth=target.adaround_bitwidth,
                group_size=target.adaround_group_size,
                optimization_batch_size=target.adaround_optimization_batch_size,
            )
            for name in selected:
                if name not in inputs:
                    raise RuntimeError(
                        f"AdaRound calibration did not execute selected module '{name}'."
                    )
                result = apply_adaround(
                    module=context.policy.get_submodule(name),
                    inputs=inputs[name],
                    config=config,
                )
                logger.info(
                    "AdaRound module %s: loss %.6g -> %.6g",
                    name,
                    result.initial_loss,
                    result.final_loss,
                )

    @staticmethod
    def _convert_exported_model(
        exported: nn.Module,
        targets: list[PT2EQuantizationModuleTarget],
        calibration: CalibrationDataProvider | None,
    ) -> nn.Module:
        """Apply PT2E prepare/calibrate/convert to an exported model.

        Args:
            exported: Exported float graph module.
            targets: PT2E targets used to create backend quantizers.
            calibration: Calibration batches for static PT2E targets.

        Returns:
            Converted PT2E graph module.

        Raises:
            ValueError: If any target needs calibration but no calibration
                provider was supplied.
        """
        if not targets:
            return exported
        needs_calibration = any(target.needs_calibration for target in targets)
        if needs_calibration and calibration is None:
            raise ValueError(
                "PT2E static quantization requires calibration data "
                "but no CalibrationDataProvider was supplied."
            )
        quantizers = []
        for target in targets:
            backend = target.pt2e_backend
            quantizers.append(backend.create_quantizer(module_path=target.module_path))
            logger.info(f"PT2E target: {target.label}")

        composed = ComposableQuantizer(quantizers)
        # print("=== GRAPH BEFORE PREPARE ===")
        # print(exported.graph)

        # for node in exported.graph.nodes:
        #     if "conv" in str(node.target).lower():
        #         print(
        #             "CONV:",
        #             node,
        #             "op=", node.op,
        #             "target=", node.target,
        #         )

        # print("=== QUANTIZERS ===")
        # for target in targets:
        #     print(target.label, type(target.pt2e_backend))
        first_backend = targets[0].pt2e_backend
        with first_backend.environment_context():
            prepared = prepare_pt2e(exported, composed)
            if calibration is not None:
                logger.info("Calibrating PT2E...")
                with torch.no_grad():
                    for batch in calibration:
                        prepared(*batch)
            #_analyze_pt2e_observers(prepared) ##change
            
            converted = convert_pt2e(prepared)
        _log_silu_quantization_status(converted)
        static_op_count = str(converted.graph).count(
            FXNodePattern.QUANTIZE_PER_TENSOR.value
        )
        logger.info(f"PT2E done, static ops: {static_op_count}")
        return converted

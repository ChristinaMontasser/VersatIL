"""Hydra configuration dataclasses for quantization workflows and backends."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class BasePT2EBackendConfig:
    """Shared settings for PT2E quantization backends.

    Attributes:
        is_dynamic: Whether activations are quantized dynamically.
        is_qat: Whether the backend prepares quantization-aware training observers.
    """

    is_dynamic: bool = False
    is_qat: bool = False


@dataclass
class X86InductorQuantizationSpecConfig:
    """Hydra-safe scalar description of one PT2E quantization spec.

    Attributes:
        bit_width: Integer quantization width.
        quant_min: Minimum representable quantized weight value.
        quant_max: Maximum representable quantized weight value.
        qscheme: Torch quantization scheme name.
        ch_axis: Tensor dimension used for per-channel quantization.
        is_dynamic: Whether activation quantization is dynamic.
    """

    bit_width: int = 4
    quant_min: int = -8
    quant_max: int = 7
    qscheme: str = "per_channel_symmetric"
    ch_axis: int = 0
    is_dynamic: bool = False


@dataclass
class X86InductorWeightQuantizationConfig(X86InductorQuantizationSpecConfig):
    """Weight quantization settings for the X86 Inductor PT2E backend."""

    quantize_bias: bool = False


@dataclass
class X86InductorActivationQuantizationConfig(X86InductorQuantizationSpecConfig):
    """Activation quantization settings for the X86 Inductor PT2E backend."""

    bit_width: int = 8
    quant_min: int = -128
    quant_max: int = 127
    qscheme: str = "per_tensor_affine"
    ch_axis: int | None = None
    is_dynamic: bool | None = None


@dataclass
class X86InductorBackendConfig(BasePT2EBackendConfig):
    """X86 Inductor backend for PT2E quantized operator lowering.

    Attributes:
        _target_: Import path instantiated by Hydra.
        reduce_range: Reduce quantization range for older CPUs without VNNI.
        weight_quantization: Optional weight override. ``None`` leaves weights
            unquantized when activation settings are provided.
        activation_quantization: Optional activation override. ``None`` leaves
            activations unquantized when weight settings are provided.
    """

    _target_: str = (
        "versatil.quantization.pt2e.backends.x86_inductor.X86InductorBackend"
    )
    reduce_range: bool = False
    weight_quantization: X86InductorWeightQuantizationConfig | None = None
    activation_quantization: X86InductorActivationQuantizationConfig | None = None


@dataclass
class XNNPACKPT2EBackendConfig(BasePT2EBackendConfig):
    """XNNPACK backend for PT2E quantization and ExecuTorch deployment.

    Attributes:
        _target_: Import path instantiated by Hydra.
        is_per_channel: Use per-channel symmetric weight quantization.
    """

    _target_: str = "versatil.quantization.pt2e.backends.xnnpack.XNNPACKPT2EBackend"
    is_per_channel: bool = True


@dataclass
class Int8DynamicQuantizeConfig:
    """Dynamic int8 activation + int8 weight quantization (`quantize_` API)."""

    _target_: str = "torchao.quantization.Int8DynamicActivationInt8WeightConfig"


@dataclass
class Int4WeightOnlyQuantizeConfig:
    """Int4 weight-only quantization with groupwise scaling (`quantize_` API).

    Attributes:
        _target_: Import path instantiated by Hydra.
        group_size: Rows sharing one quantization scale.
    """

    _target_: str = "torchao.quantization.Int4WeightOnlyConfig"
    group_size: int = 128


@dataclass
class EagerQuantizationModuleTargetConfig:
    """Module target for eager torchao quantization.

    Attributes:
        _target_: Import path instantiated by Hydra.
        module_path: Dotted path to the target module, or ``""`` for root.
        quantize_config: torchao eager quantization config applied to this target.
        smooth_quant: Apply TorchAO SmoothQuant preparation before quantization.
        smooth_quant_alpha: SmoothQuant activation-to-weight migration factor.
        exclude_module_patterns: Glob patterns for child module names to leave
            unquantized.
        gptq: Apply TorchAO prototype GPTQ using calibration activations.
        gptq_percdamp: Damping factor applied to the GPTQ Hessian diagonal.
        gptq_quantize_block_size: Number of weight columns processed by one GPTQ
            block.
        adaround: Apply native PyTorch AdaRound using calibration activations.
        adaround_iterations: Optimization steps per selected linear.
        adaround_regularization_strength: AdaRound rounding regularizer weight.
        adaround_beta_start: Initial AdaRound regularizer temperature.
        adaround_beta_end: Final AdaRound regularizer temperature.
        adaround_warm_start: Fraction of steps without rounding regularization.
        adaround_bitwidth: Symmetric weight bitwidth used during AdaRound.
        adaround_group_size: Input-column group size used during AdaRound.
    """

    _target_: str = "versatil.quantization.module_target.EagerQuantizationModuleTarget"
    module_path: str = ""
    quantize_config: Any = MISSING
    smooth_quant: bool = False
    smooth_quant_alpha: float = 0.5
    exclude_module_patterns: list[str] = field(default_factory=list)
    gptq: bool = False
    gptq_percdamp: float = 0.01
    gptq_quantize_block_size: int = 256
    adaround: bool = False
    adaround_iterations: int = 10000
    adaround_regularization_strength: float = 0.01
    adaround_beta_start: float = 20.0
    adaround_beta_end: float = 2.0
    adaround_warm_start: float = 0.2
    adaround_bitwidth: int = 8
    adaround_group_size: int | None = None
    adaround_optimization_batch_size: int = 32


@dataclass
class PT2EQuantizationModuleTargetConfig:
    """Module target for PT2E quantization.

    Attributes:
        _target_: Import path instantiated by Hydra.
        module_path: Dotted path to the target module, or ``""`` for root.
        pt2e_backend: PT2E backend that creates the quantizer for this target.
        smooth_quant: Apply SmoothQuant-style float preparation before export.
        smooth_quant_alpha: Activation-to-weight migration factor.
        adaround: Apply native PyTorch AdaRound before export.
        adaround_iterations: Optimization steps per selected linear.
        adaround_regularization_strength: AdaRound rounding regularizer weight.
        adaround_beta_start: Initial AdaRound regularizer temperature.
        adaround_beta_end: Final AdaRound regularizer temperature.
        adaround_warm_start: Fraction of steps without rounding regularization.
        adaround_bitwidth: Symmetric weight bitwidth used during AdaRound.
        adaround_group_size: Input-column group size used during AdaRound.
    """

    _target_: str = "versatil.quantization.module_target.PT2EQuantizationModuleTarget"
    module_path: str = ""
    pt2e_backend: BasePT2EBackendConfig = field(
        default_factory=X86InductorBackendConfig
    )
    smooth_quant: bool = False
    smooth_quant_alpha: float = 0.5
    adaround: bool = False
    adaround_iterations: int = 10000
    adaround_regularization_strength: float = 0.01
    adaround_beta_start: float = 20.0
    adaround_beta_end: float = 2.0
    adaround_warm_start: float = 0.2
    adaround_bitwidth: int = 8
    adaround_group_size: int | None = None
    adaround_optimization_batch_size: int = 32


@dataclass
class PT2EQuantizationWorkflowConfig:
    """Graph-level quantization with operator fusion via torch.export.

    Attributes:
        _target_: Import path instantiated by Hydra.
        targets: module-level PT2E quantization targets.
    """

    _target_: str = "versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow"
    targets: list[Any] = field(
        default_factory=lambda: [PT2EQuantizationModuleTargetConfig()]
    )


@dataclass
class EagerQuantizationWorkflowConfig:
    """Eager torchao quantization via quantize_().

    Attributes:
        _target_: Import path instantiated by Hydra.
        targets: Module-level eager quantization targets.
        is_qat: Whether this workflow is used for QAT checkpoint training and
            conversion.
        auto_filter_incompatible_linears: Whether to skip linears whose ``in_features``
            are incompatible with the config group size.
    """

    _target_: str = "versatil.quantization.workflows.eager.EagerQuantizationWorkflow"
    targets: list[Any] = MISSING
    is_qat: bool = False
    auto_filter_incompatible_linears: bool = True

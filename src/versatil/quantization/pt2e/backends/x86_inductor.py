"""X86 Inductor backend for PT2E quantized operator lowering."""

import os
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch._inductor.config as inductor_config
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from torchao.quantization.pt2e.quantizer import QuantizationConfig, QuantizationSpec
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep
from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig
from versatil.configs.quantization import (
    X86InductorActivationQuantizationConfig,
    X86InductorQuantizationSpecConfig,
    X86InductorWeightQuantizationConfig,
)
from versatil.quantization.constants import PT2EBackendName
from versatil.quantization.pt2e.backends.base import BasePT2EBackend
from versatil.quantization.workflows import none

_CUDA_VISIBLE_DEVICES_KEY = "CUDA_VISIBLE_DEVICES"
_TORCHINDUCTOR_FREEZING_KEY = "TORCHINDUCTOR_FREEZING"
_SILU_TARGETS = {torch.ops.aten.silu.default, torch.ops.aten.silu_.default, torch.nn.modules.activation.SiLU}


class _SiluExcludedX86InductorQuantizer(X86InductorQuantizer):
    """X86 quantizer that keeps fused SiLU outputs in floating point."""

    def annotate(self, model: GraphModule) -> GraphModule:
        """Annotate the model while excluding quantization after SiLU nodes."""
        model = super().annotate(model)
        for node in model.graph.nodes:
            if node.target not in _SILU_TARGETS:
                continue
            annotation = node.meta.get("quantization_annotation")
            if annotation is None:
                continue
            annotation.output_qspec = None
            annotation._is_output_of_quantized_pattern = False
        return model


class _ResidualADDExcludedX86InductorQuantizer(X86InductorQuantizer):
    """X86 quantizer that keeps residual ADD outputs in floating point."""

    def annotate(self, model: GraphModule) -> GraphModule:
        model = super().annotate(model)

        for node in model.graph.nodes:
            if node.target != torch.ops.aten.add.Tensor:
                continue

            annotation = node.meta.get("quantization_annotation")
            if annotation is None:
                continue

            annotation.output_qspec = None
            annotation._is_output_of_quantized_pattern = False

        return model


class X86InductorBackend(BasePT2EBackend):
    """X86 Inductor backend for PT2E quantization and lowering."""

    @property
    def name(self) -> str:
        """Serialized PT2E backend name."""
        return PT2EBackendName.X86_INDUCTOR.value

    def __init__(
        self,
        is_dynamic: bool = False,
        is_qat: bool = False,
        reduce_range: bool = False,
        weight_quantization: X86InductorWeightQuantizationConfig | None = None,
        activation_quantization: X86InductorActivationQuantizationConfig | None = None,
    ) -> None:
        """Initialize X86 Inductor backend configuration.

        Args:
            is_dynamic: Use dynamic activation quantization.
            is_qat: Use quantization-aware training observers.
            reduce_range: Reduce quantization range for older CPUs
                without VNNI.
            weight_quantization: Optional weight-only quantization override.
                Weights are unquantized when omitted alongside activation
                settings.
            activation_quantization: Optional activation quantization override.
                Activations are unquantized when omitted alongside weight
                settings.
        """
        self._is_dynamic = is_dynamic
        self._is_qat = is_qat
        self._reduce_range = reduce_range
        self._weight_quantization = weight_quantization
        self._activation_quantization = activation_quantization

    @property
    def is_dynamic(self) -> bool:
        """Whether this backend uses dynamic activation quantization."""
        return self._is_dynamic

    @property
    def is_qat(self) -> bool:
        """Whether this backend uses QAT observer configuration."""
        return self._is_qat

    @property
    def supported_device_types(self) -> tuple[str, ...]:
        """X86 inductor only supports CPU inference."""
        return ("cpu",)

    def create_quantizer(self, module_path: str) -> Quantizer:
        """Create an X86InductorQuantizer targeting a specific module.

        Args:
            module_path: Dotted path to the target submodule.
                Empty string means global (whole model).

        Returns:
            Configured X86InductorQuantizer.
        """
        #quantizer = _SiluExcludedX86InductorQuantizer()
        quantizer = X86InductorQuantizer()
        #quantizer = _ResidualADDExcludedX86InductorQuantizer()
        # quantization_config
        # config = get_default_x86_inductor_quantization_config(
        #     is_dynamic=self._is_dynamic,
        #     is_qat=self._is_qat,
        #     reduce_range=self._reduce_range,
        # )

        default_config = get_default_x86_inductor_quantization_config(
            is_dynamic=self._is_dynamic,
            is_qat=self._is_qat,
            reduce_range=self._reduce_range,
        )
        if (
            self._weight_quantization is None
            and self._activation_quantization is None
        ):
            config = default_config
        else:
            config = self._build_custom_config(default_config)
    #     weight_quantization_spec_updated_int4 = QuantizationSpec(
    #         dtype = torch.int4,
    #         quant_min= quantization_config.weight.quant_min,
    #         quant_max= quantization_config.weight.quant_max,
    #         qscheme=quantization_config.weight.qscheme,
    #         ch_axis=quantization_config.weight.ch_axis,
    #         is_dynamic=quantization_config.weight.is_dynamic,
    #         observer_or_fake_quant_ctr=quantization_config.weight.observer_or_fake_quant_ctr,
    #     )
    #     config = QuantizationConfig( ##change
    #         None,
    #         None,
    #         quantization_config.weight,
    #         quantization_config.bias,
    #         quantization_config.is_qat,
    # )
    #     weight_quantization_spec_updated_int4 = QuantizationSpec(
    #         dtype = torch.int4,
    #         quant_min= quantization_config.weight.quant_min,
    #         quant_max= quantization_config.weight.quant_max,
    #         qscheme=quantization_config.weight.qscheme,
    #         ch_axis=quantization_config.weight.ch_axis,
    #         is_dynamic=quantization_config.weight.is_dynamic,
    #         observer_or_fake_quant_ctr=quantization_config.weight.observer_or_fake_quant_ctr,
    #     )
    #     config = QuantizationConfig( ##change
    #         None,
    #         None,
    #         quantization_config.weight,
    #         quantization_config.bias,
    #         quantization_config.is_qat,
    # )
    #     config = SmoothQuantConfig(
    #     base_config=Int8DynamicActivationInt8WeightConfig(),
    #     step=QuantizationStep.PREPARE,
    #     alpha=0.5,
    # )
      
        if module_path == "":
            quantizer.set_global(config)
        else:
            quantizer.set_module_name_qconfig(module_path, config)
        return quantizer

    def _build_custom_config(
        self,
        default_config: QuantizationConfig,
    ) -> QuantizationConfig:
        """Build a validated config from Hydra scalar quantization settings."""
        weight_spec = self._build_spec(
            self._weight_quantization,
            default_config.weight,
            is_dynamic=False,
        )
        activation_spec = self._build_spec(
            self._activation_quantization,
            default_config.input_activation,
            is_dynamic=(
                self._is_dynamic
                if self._activation_quantization is None
                or self._activation_quantization.is_dynamic is None
                else self._activation_quantization.is_dynamic
            ),
        )
        bias_spec = None
        if self._weight_quantization is not None:
            if self._weight_quantization.quantize_bias:
                bias_spec = weight_spec
        elif self._activation_quantization is not None:
            bias_spec = None

        return QuantizationConfig(
            input_activation=activation_spec,
            output_activation=activation_spec,
            weight=weight_spec,
            bias=bias_spec,
            is_qat=self._is_qat,
        )

    @staticmethod
    def _build_spec(
        custom_config: X86InductorQuantizationSpecConfig | None,
        default_spec: QuantizationSpec | None,
        is_dynamic: bool,
    ) -> QuantizationSpec | None:
        """Build one torchao spec while reusing the default observer."""
        if custom_config is None:
            return None
        if custom_config.quant_min >= custom_config.quant_max:
            raise ValueError("quant_min must be smaller than quant_max")
        if custom_config.bit_width not in (4, 8):
            raise ValueError("X86 quantization bit_width must be 4 or 8")
        if default_spec is None:
            raise ValueError("No default observer is available for this spec")
        # IMPORTANT
        # Torchao's default PT2E observer stores 4-bit values in int8 while
        # the quantization range still limits them to the signed int4 range.
        dtype = torch.int8
        qscheme = getattr(torch, custom_config.qscheme, None)
        if qscheme is None:
            raise ValueError(f"Unsupported X86 qscheme: {custom_config.qscheme}")

        return QuantizationSpec(
            dtype=dtype,
            quant_min=custom_config.quant_min,
            quant_max=custom_config.quant_max,
            qscheme=qscheme,
            ch_axis=custom_config.ch_axis,
            is_dynamic=is_dynamic,
            observer_or_fake_quant_ctr=default_spec.observer_or_fake_quant_ctr,
        )

    @contextmanager
    def environment_context(self) -> Generator[None]:
        """Context manager that sets and restores X86 Inductor env vars.

        Sets CUDA_VISIBLE_DEVICES to empty (CPU-only), enables
        TORCHINDUCTOR_FREEZING, and enables cpp_wrapper. Restores
        original values on exit.

        Yields:
            None.
        """
        saved = {
            _CUDA_VISIBLE_DEVICES_KEY: os.environ.get(_CUDA_VISIBLE_DEVICES_KEY),
            _TORCHINDUCTOR_FREEZING_KEY: os.environ.get(_TORCHINDUCTOR_FREEZING_KEY),
        }
        saved_cpp_wrapper = inductor_config.cpp_wrapper
        os.environ[_CUDA_VISIBLE_DEVICES_KEY] = ""
        os.environ[_TORCHINDUCTOR_FREEZING_KEY] = "1"
        inductor_config.cpp_wrapper = True
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            inductor_config.cpp_wrapper = saved_cpp_wrapper

    def activate_environment(self) -> None:
        """Set X86 Inductor env vars permanently."""
        os.environ[_CUDA_VISIBLE_DEVICES_KEY] = ""
        os.environ[_TORCHINDUCTOR_FREEZING_KEY] = "1"
        inductor_config.cpp_wrapper = True

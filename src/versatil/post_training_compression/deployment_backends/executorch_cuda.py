"""ExecuTorch CUDA deployment backend for .pte artifacts."""

import importlib
from types import ModuleType

import torch
import torch.nn as nn

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentArtifact,
    DeploymentBackend,
)
from versatil.post_training_compression.export import _export_with_dynamic_batch


class ExecutorchCUDABackend(DeploymentBackend):
    """Backend that lowers exported programs to ExecuTorch CUDA."""

    name = DeploymentBackendName.EXECUTORCH_CUDA.value
    artifact_format = ArtifactFormat.EXECUTORCH_PTE
    model_filename = CompressionFilename.EXECUTORCH_MODEL.value

    def __init__(self, max_batch_size: int, model_name: str = "forward") -> None:
        """Initialize CUDA deployment settings.

        Args:
            max_batch_size: Upper bound for dynamic batch execution in the
                serialized ExecuTorch program.
            model_name: Method name used to generate the CUDA compile spec.
        """
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1.")
        if not model_name:
            raise ValueError("model_name must not be empty.")
        self.max_batch_size = max_batch_size
        self.model_name = model_name

    def export(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> DeploymentArtifact:
        """Lower a PyTorch module into an ExecuTorch CUDA .pte buffer."""
        exported_program = _export_with_dynamic_batch(
            model=model,
            example_inputs=example_inputs,
            max_batch_size=self.max_batch_size,
        )
        model_bytes = self._lower_to_pte_buffer(exported_program=exported_program)
        return DeploymentArtifact(
            converted_model=None,
            example_inputs=example_inputs,
            model_filename=self.model_filename,
            artifact_format=self.artifact_format,
            backend_name=self.name,
            model_bytes=model_bytes,
        )

    def _lower_to_pte_buffer(
        self,
        exported_program: torch.export.ExportedProgram,
    ) -> bytes:
        """Lower an exported program to an ExecuTorch CUDA PTE buffer."""
        executorch_exir = importlib.import_module("executorch.exir")
        cuda_backend = importlib.import_module(
            "executorch.backends.cuda.cuda_backend"
        )
        cuda_partitioner = importlib.import_module(
            "executorch.backends.cuda.cuda_partitioner"
        )
        return _lower_exported_program(
            exported_program=exported_program,
            executorch_exir=executorch_exir,
            cuda_backend=cuda_backend,
            cuda_partitioner=cuda_partitioner,
            model_name=self.model_name,
        )


def _lower_exported_program(
    exported_program: torch.export.ExportedProgram,
    executorch_exir: ModuleType,
    cuda_backend: ModuleType,
    cuda_partitioner: ModuleType,
    model_name: str,
) -> bytes:
    """Lower an exported program using imported ExecuTorch CUDA modules."""
    edge_compile_config = executorch_exir.EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )
    compile_specs = [
        cuda_backend.CudaBackend.generate_method_name_compile_spec(model_name)
    ]
    edge_program = executorch_exir.to_edge_transform_and_lower(
        exported_program,
        partitioner=[cuda_partitioner.CudaPartitioner(compile_specs)],
        compile_config=edge_compile_config,
    )
    executorch_program = edge_program.to_executorch()
    return bytes(executorch_program.buffer)
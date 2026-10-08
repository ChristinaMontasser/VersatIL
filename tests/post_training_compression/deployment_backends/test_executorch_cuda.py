"""Tests for versatil.post_training_compression.deployment_backends.executorch_cuda module."""

from unittest.mock import MagicMock

import pytest
import torch

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.executorch_cuda import (
    ExecutorchCUDABackend,
    _lower_exported_program,
)


@pytest.mark.unit
def test_cuda_backend_stores_configuration() -> None:
    backend = ExecutorchCUDABackend(max_batch_size=8, model_name="forward")

    assert backend.max_batch_size == 8
    assert backend.model_name == "forward"


@pytest.mark.unit
def test_cuda_backend_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_batch_size must be >= 1\\."):
        ExecutorchCUDABackend(max_batch_size=0)

    with pytest.raises(ValueError, match="model_name must not be empty"):
        ExecutorchCUDABackend(max_batch_size=1, model_name="")


@pytest.mark.unit
def test_cuda_backend_export_returns_pte_artifact() -> None:
    backend = ExecutorchCUDABackend(max_batch_size=8)
    model = MagicMock()
    example_inputs = (torch.zeros(2, 4),)
    exported_program = MagicMock()

    backend._lower_to_pte_buffer = MagicMock(return_value=b"pte-bytes")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "versatil.post_training_compression.deployment_backends.executorch_cuda._export_with_dynamic_batch",
            MagicMock(return_value=exported_program),
        )
        artifact = backend.export(model=model, example_inputs=example_inputs)

    assert artifact.converted_model is None
    assert artifact.model_bytes == b"pte-bytes"
    assert artifact.model_filename == CompressionFilename.EXECUTORCH_MODEL.value
    assert artifact.artifact_format == ArtifactFormat.EXECUTORCH_PTE
    assert artifact.backend_name == DeploymentBackendName.EXECUTORCH_CUDA.value


@pytest.mark.unit
def test_lower_exported_program_uses_cuda_partitioner() -> None:
    exported_program = MagicMock()
    edge_program = MagicMock()
    executorch_program = MagicMock()
    executorch_program.buffer = b"pte"
    edge_program.to_executorch.return_value = executorch_program
    executorch_exir = MagicMock()
    executorch_exir.to_edge_transform_and_lower.return_value = edge_program
    cuda_backend = MagicMock()
    compile_spec = MagicMock()
    cuda_backend.CudaBackend.generate_method_name_compile_spec.return_value = (
        compile_spec
    )
    cuda_partitioner = MagicMock()
    partitioner = MagicMock()
    cuda_partitioner.CudaPartitioner.return_value = partitioner

    result = _lower_exported_program(
        exported_program=exported_program,
        executorch_exir=executorch_exir,
        cuda_backend=cuda_backend,
        cuda_partitioner=cuda_partitioner,
        model_name="forward",
    )

    cuda_backend.CudaBackend.generate_method_name_compile_spec.assert_called_once_with(
        "forward"
    )
    cuda_partitioner.CudaPartitioner.assert_called_once_with([compile_spec])
    executorch_exir.to_edge_transform_and_lower.assert_called_once_with(
        exported_program,
        partitioner=[partitioner],
        compile_config=executorch_exir.EdgeCompileConfig.return_value,
    )
    assert result == b"pte"
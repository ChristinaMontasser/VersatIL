"""Reports for serialized compressed policy checkpoints."""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    CompressionMetadataKey,
)
from versatil.post_training_compression.serialization import load_compression_metadata


class CompressedPolicyReport:
    """Inspect a compressed policy directory without running inference."""

    def __init__(self, checkpoint_path: str) -> None:
        """Initialize a report for a compressed checkpoint directory.

        Args:
            checkpoint_path: Directory containing compression metadata and its
                serialized policy artifact.

        Raises:
            FileNotFoundError: If the directory or metadata file is missing.
        """
        self._checkpoint_path = Path(checkpoint_path)
        if not self._checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Compressed checkpoint directory not found: {checkpoint_path}"
            )
        metadata_path = self._checkpoint_path / CompressionFilename.COMPRESSION_METADATA.value
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Compression metadata not found at {metadata_path}.")
        self._metadata = load_compression_metadata(metadata_path=str(metadata_path))
        model_filename = self._metadata.get(CompressionMetadataKey.MODEL_FILE.value)
        if not model_filename:
            raise ValueError(
                f"Compression metadata is missing '{CompressionMetadataKey.MODEL_FILE.value}'."
            )
        self._model_path = self._checkpoint_path / model_filename
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Compressed model not found at {self._model_path}.")

    def build_report(self) -> dict[str, Any]:
        """Build a JSON-serializable report for the compressed checkpoint."""
        artifact_bytes = self._model_path.stat().st_size
        checkpoint_directory_bytes = self._directory_bytes(
            directory=self._checkpoint_path
        )
        report: dict[str, Any] = {
            "checkpoint_path": str(self._checkpoint_path),
            "artifact_path": str(self._model_path),
            "artifact_format": self._metadata.get(
                CompressionMetadataKey.ARTIFACT_FORMAT.value
            ),
            "deployment_backend": self._metadata.get(
                CompressionMetadataKey.DEPLOYMENT_BACKEND.value
            ),
            "quantization_workflow": self._metadata.get(
                CompressionMetadataKey.QUANTIZATION_WORKFLOW.value
            ),
            "artifact_bytes": artifact_bytes,
            "artifact_size_mb": self._bytes_to_mb(value=artifact_bytes),
            "checkpoint_directory_bytes": checkpoint_directory_bytes,
            "checkpoint_directory_size_mb": self._bytes_to_mb(
                value=checkpoint_directory_bytes
            ),
            "input_keys": self._metadata.get(CompressionMetadataKey.INPUT_KEYS.value, []),
            "output_keys": self._metadata.get(CompressionMetadataKey.OUTPUT_KEYS.value, []),
            "model": None,
            "source_checkpoint": self._source_checkpoint_report(),
            "metadata": self._metadata,
        }
        if report["artifact_format"] == ArtifactFormat.TORCH_EXPORT_PT2.value:
            exported_program = torch.export.load(str(self._model_path))
            report["model"] = self._model_report(model=exported_program.module())
        return report

    def write(self, output_path: str) -> Path:
        """Write the report as formatted JSON and return its path."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w") as file:
            json.dump(self.build_report(), file, indent=2)
        return destination

    def _source_checkpoint_report(self) -> dict[str, Any]:
        source_path_value = self._metadata.get(
            CompressionMetadataKey.TRAINING_CHECKPOINT_PATH.value
        )
        if not source_path_value:
            return {"path": None, "directory_bytes": None, "compression_ratio": None}
        source_path = Path(source_path_value)
        if not source_path.is_dir():
            return {"path": str(source_path), "directory_bytes": None, "compression_ratio": None}
        source_bytes = self._directory_bytes(directory=source_path)
        artifact_bytes = self._model_path.stat().st_size
        return {
            "path": str(source_path),
            "directory_bytes": source_bytes,
            "directory_size_mb": self._bytes_to_mb(value=source_bytes),
            "compression_ratio": source_bytes / max(artifact_bytes, 1),
        }

    @staticmethod
    def _directory_bytes(directory: Path) -> int:
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

    @staticmethod
    def _bytes_to_mb(value: int) -> float:
        return value / (1024 * 1024)

    @classmethod
    def _model_report(cls, model: nn.Module) -> dict[str, Any]:
        parameters = list(model.parameters())
        buffers = list(model.buffers())
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
        buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
        zero_elements = sum(
            int(torch.count_nonzero(parameter).item() == 0) for parameter in parameters
        )
        parameter_elements = sum(parameter.numel() for parameter in parameters)
        modules: dict[str, dict[str, int | str]] = {}
        for name, module in model.named_modules():
            module_parameters = list(module.parameters(recurse=False))
            if not module_parameters:
                continue
            modules[name] = {
                "type": type(module).__name__,
                "parameter_count": sum(parameter.numel() for parameter in module_parameters),
                "parameter_bytes": sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in module_parameters
                ),
            }
        return {
            "parameter_count": parameter_elements,
            "parameter_bytes": parameter_bytes,
            "parameter_size_mb": cls._bytes_to_mb(value=parameter_bytes),
            "buffer_bytes": buffer_bytes,
            "buffer_size_mb": cls._bytes_to_mb(value=buffer_bytes),
            "tensor_bytes": parameter_bytes + buffer_bytes,
            "tensor_size_mb": cls._bytes_to_mb(
                value=parameter_bytes + buffer_bytes
            ),
            "zero_parameter_count": zero_elements,
            "parameter_sparsity": zero_elements / max(parameter_elements, 1),
            "modules": modules,
        }
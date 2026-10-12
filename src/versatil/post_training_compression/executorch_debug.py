"""ExecuTorch runtime debug artifact generation."""

import importlib
from pathlib import Path

import torch

from versatil.post_training_compression.constants import CompressionFilename


def generate_etdump(
    model_path: str | Path,
    inputs: tuple[torch.Tensor, ...],
    output_directory: str | Path | None = None,
    debug_buffer_size: int = 64 * 1024 * 1024,
) -> tuple[Path, Path]:
    """Execute an ExecuTorch program with ETDump tracing enabled.

    Args:
        model_path: Path to the serialized ExecuTorch ``.pte`` program.
        inputs: Positional inputs for the program's ``forward`` method.
        output_directory: Directory for the ETDump and debug buffer. Defaults
            to the directory containing ``model_path``.
        debug_buffer_size: Runtime event buffer size in bytes.

    Returns:
        Paths to the ETDump file and its associated debug buffer.

    Raises:
        ValueError: If ``debug_buffer_size`` is not positive.
    """
    if debug_buffer_size < 1:
        raise ValueError("debug_buffer_size must be positive.")
    model_path = Path(model_path)
    output_path = (
        model_path.parent if output_directory is None else Path(output_directory)
    )
    output_path.mkdir(parents=True, exist_ok=True)
    etdump_path = output_path / CompressionFilename.ETDUMP.value
    debug_path = output_path / CompressionFilename.ETDUMP_DEBUG.value

    runtime_module = importlib.import_module("executorch.runtime")
    runtime = runtime_module.Runtime()
    program = runtime.load_program(
        model_path,
        enable_etdump=True,
        debug_buffer_size=debug_buffer_size,
    )
    method = program.load_method("forward")
    if method is None:
        raise RuntimeError("ExecuTorch program does not contain a forward method.")
    method.execute(list(inputs))
    program.write_etdump_result_to_file(str(etdump_path), str(debug_path))
    return etdump_path, debug_path

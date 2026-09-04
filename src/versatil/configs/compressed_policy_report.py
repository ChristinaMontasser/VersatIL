"""Configuration for the compressed policy report endpoint."""

from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class CompressedPolicyReportConfig:
    """Settings for inspecting a compressed policy directory."""

    checkpoint_path: str = MISSING
    output_path: str | None = None
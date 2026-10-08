"""Hydra endpoint for reporting a serialized compressed policy."""

import json
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from versatil.common.logging import override_log_format
from versatil.configs.paths import get_hydra_configs_dir
from versatil.post_training_compression.compressed_report import CompressedPolicyReport

EXPERIMENTS_DIR = get_hydra_configs_dir()


@hydra.main(
    version_base=None,
    config_path=str(EXPERIMENTS_DIR),
    config_name="compressed_policy_report",
)
def main(config: DictConfig) -> None:
    """Generate and print a report for a compressed policy directory."""
    override_log_format()
    logging.info(OmegaConf.to_yaml(config))
    report_generator = CompressedPolicyReport(checkpoint_path=str(config.checkpoint_path))
    if config.output_path is not None:
        report_generator.write(output_path=str(config.output_path))
    print(json.dumps(report_generator.build_report(), indent=2))
    if config.include_quantization_report:
        print(
            report_generator.build_quantization_report(
                device=torch.device(config.device),
                compile_eager_for_report=config.compile_eager_for_report,
            )
        )


if __name__ == "__main__":
    main()
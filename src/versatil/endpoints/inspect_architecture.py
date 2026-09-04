import json
import torch
import hydra
import logging
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from versatil.common.logging import override_log_format
from versatil.configs.paths import get_hydra_configs_dir
from versatil.validation import validate_experiment

EXPERIMENTS_DIR = get_hydra_configs_dir()

def count_parameters(module):
    return sum(
        p.numel()
        for p in module.parameters()
    )


def parameter_size_mb(module):
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total / (1024**2)


def report_model(model):

    print("="*80)
    print("MODEL COMPONENT REPORT")
    print("="*80)

    for name, module in model.named_modules():

        params = count_parameters(module)

        if params == 0:
            continue

        print("\n", name)
        print("-"*50)
        print("Type:", module.__class__.__name__)
        print("Parameters:", params)
        print(
            "Size MB:",
            round(parameter_size_mb(module),3)
        )

        first_param = next(
            module.parameters(),
            None
        )

        if first_param is not None:
            print(
                "dtype:",
                first_param.dtype
            )
            print(
                "device:",
                first_param.device
            )

def export_report(model, path="reports/model_report.json"):

    report = {}

    for name, module in model.named_modules():

        params = sum(
            p.numel()
            for p in module.parameters()
        )

        if params == 0:
            continue

        report[name] = {
            "type": module.__class__.__name__,
            "parameters": params,
            "size_MB": sum(
                p.numel()*p.element_size()
                for p in module.parameters()
            ) / 1024**2
        }

    with open(path,"w") as f:
        json.dump(
            report,
            f,
            indent=4
        )


@hydra.main(version_base=None, config_path=str(EXPERIMENTS_DIR), config_name="main")
def main(cfg: DictConfig):

    override_log_format()

    logging.info("=" * 80)
    logging.info("Inspect Configuration")
    logging.info("=" * 80)
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info("=" * 80)


    logging.info("Instantiating policy model...")

    model = hydra.utils.instantiate(
        cfg.policy
    )

    model.eval()

    logging.info(
        f"Model type: {type(model)}"
    )

    export_report(model)




if __name__ == "__main__":
    main()
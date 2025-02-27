import hydra
from omegaconf import DictConfig

import os
from utils.run_experiment import run_experiment
import torch.distributed as dist


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    os.environ["WANDB_MODE"] = cfg.experiment.wandb_mode
    os.environ["WANDB_DIR"] = cfg.experiment.wandb_path
    dist.init_process_group(backend="nccl")
    print("Initialized")
    run_experiment(cfg)


if __name__ == "__main__":
    main()

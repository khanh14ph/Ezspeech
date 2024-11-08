import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    # Print the config
    task=instantiate(cfg.task,_recursive=True)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    
    # Instantiate model
    model = instantiate(cfg.model)
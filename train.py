import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./config", config_name="asr")
def train(cfg: DictConfig):
    # Print the config
    task = instantiate(cfg.task,_recursive_=False)

    callbacks=[instantiate(cfg.callbacks[i]) for i in cfg.callbacks.keys()]
    logger=instantiate(cfg.logger)
    # # Trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        **cfg.trainer,
        logger=logger
    )
    
    # # Train
    trainer.fit(task)

if __name__ == "__main__":
    train()

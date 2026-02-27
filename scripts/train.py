#!/usr/bin/env python3
"""
Enhanced training script for EzSpeech ASR models.
Supports multiple model types with improved configuration management.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger

# from ezspeech.models.ctc import ASR_ctc_training
from ezspeech.models.tdt import ASR_tdt_training
from ezspeech.utils import color

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")

@hydra.main(version_base=None, config_path="../config", config_name="tdt")
def main(config: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting EzSpeech training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")


    # Setup model
    model = ASR_tdt_training(config)
    
    # Load pretrained weights if specified
    if config.model.get("model_pretrained") is not None:

        checkpoint = torch.load(config.model.model_pretrained.path,weights_only=False)
        model.encoder.load_state_dict(checkpoint["state_dict"]["encoder"])
        logger.info("Pretrained weights loaded successfully.")


    callbacks = []
    for callback_name, callback_config in config.callbacks.items():
        try:
            callback = instantiate(callback_config)
            callbacks.append(callback)
            logger.info(f"Added callback: {callback_name}")
        except Exception as e:
            logger.error(f"Failed to instantiate callback {callback_name}: {e}")


    tb_logger = instantiate(config.loggers.tb)
    # Initialize trainer
    logger.info("Initializing PyTorch Lightning trainer...")
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=tb_logger
    )

    # Start training
    logger.info("Starting training process...")
    trainer.fit(model)

    # Save final model if training completed successfully
    if trainer.state.finished:
        save_path = Path(config.get("output_dir", "outputs")) / "final_model.ckpt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(save_path)
        logger.info(f"Final model saved to: {save_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
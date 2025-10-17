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

from ezspeech.models.ctc import ASR_ctc_training
from ezspeech.utils import color
from ezspeech.utils.training import setup_callbacks, setup_loggers, load_pretrained_weights

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")


def validate_config(config: DictConfig) -> None:
    """Validate configuration for common issues."""
    required_sections = ["dataset", "model", "trainer"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Check dataset paths exist
    if hasattr(config.dataset, 'train_ds') and hasattr(config.dataset.train_ds, 'filepaths'):
        for filepath in config.dataset.train_ds.filepaths:
            if not Path(filepath).exists():
                logger.warning(f"Training file not found: {filepath}")

    logger.info("Configuration validation completed")


def setup_model(config: DictConfig) -> ASR_ctc_training:
    """Initialize and setup the ASR model."""
    logger.info("Initializing ASR model...")
    model = ASR_ctc_training(config)

    # Load pretrained weights if specified
    if config.model.get("model_pretrained") is not None:
        load_pretrained_weights(model, config.model.model_pretrained)

    return model


@hydra.main(version_base=None, config_path="../config", config_name="ctc")
def main(config: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting EzSpeech training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Validate configuration
    validate_config(config)

    # Setup model
    model = setup_model(config)

    # Setup callbacks
    callbacks = setup_callbacks(config)

    # Setup loggers
    loggers = setup_loggers(config)

    # Initialize trainer
    logger.info("Initializing PyTorch Lightning trainer...")
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=loggers,
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
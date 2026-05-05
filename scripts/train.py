#!/usr/bin/env python3
"""
Training script for EzSpeech ASR models.
The model class is resolved from `training_module` in the YAML config,
so the same script works for any model (ctc, ctc_llm, …).
"""

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from ezspeech.utils import color


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../config", config_name="ctc_llm")
def main(config: DictConfig) -> None:
    logger.info("Starting EzSpeech training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Resolve the model class from config (e.g. ezspeech.models.ctc.ASR_ctc_training)
    ModelClass = get_class(config.training_module)
    model = ModelClass(config)


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

    trainer.fit(model, ckpt_path="/scratch/midway2/khanhnd/lightning_logs/ctc_llm/version_2/checkpoints/last.ckpt")

    # Save final model if training completed successfully
    if trainer.state.finished:
        save_path = Path(config.get("output_dir", "outputs")) / "final_model.ckpt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(save_path)
        logger.info(f"Final model saved to: {save_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
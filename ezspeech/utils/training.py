"""
Training utilities for EzSpeech.
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from ezspeech.utils import color

logger = logging.getLogger(__name__)


def setup_loggers(config: DictConfig) -> Optional[List[Logger]]:
    """Setup PyTorch Lightning loggers from configuration."""
    if not config.get("loggers"):
        return None

    loggers = []
    for logger_name, logger_config in config.loggers.items():
        try:
            logger_instance = instantiate(logger_config)
            loggers.append(logger_instance)
            logger.info(f"Added logger: {logger_name}")
        except Exception as e:
            logger.error(f"Failed to instantiate logger {logger_name}: {e}")

    return loggers if loggers else None





def get_model_summary(model) -> str:
    """Get a summary of the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = f"""
Model Summary:
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Non-trainable parameters: {total_params - trainable_params:,}
"""

    return summary


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


def setup_callbacks(config: DictConfig) -> Optional[List[Callback]]:
    """Setup PyTorch Lightning callbacks from configuration."""
    if not config.get("callbacks"):
        return None

    callbacks = []
    for callback_name, callback_config in config.callbacks.items():
        try:
            callback = instantiate(callback_config)
            callbacks.append(callback)
            logger.info(f"Added callback: {callback_name}")
        except Exception as e:
            logger.error(f"Failed to instantiate callback {callback_name}: {e}")

    return callbacks if callbacks else None


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


def load_pretrained_weights(model, pretrained_config: DictConfig) -> None:
    """Load pretrained weights into model."""
    checkpoint_filepath = pretrained_config.path

    if not Path(checkpoint_filepath).exists():
        logger.error(f"Checkpoint file not found: {checkpoint_filepath}")
        return

    logger.info(f"Loading pretrained weights from: {checkpoint_filepath}")

    try:
        checkpoint = torch.load(
            checkpoint_filepath, map_location="cpu", weights_only=False
        )
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    if "state_dict" not in checkpoint:
        logger.error("No 'state_dict' found in checkpoint")
        return

    # Load specified modules
    for attr in pretrained_config.include:
        if attr not in checkpoint["state_dict"]:
            logger.warning(f"Module {attr} not found in checkpoint")
            continue

        if not hasattr(model, attr):
            logger.warning(f"Model doesn't have attribute {attr}")
            continue

        try:
            weights = checkpoint["state_dict"][attr]
            net = getattr(model, attr)
            net.load_state_dict(weights)
            logger.info(
                f"Successfully loaded {color.GREEN}{attr}{color.RESET} from checkpoint"
            )
        except Exception as e:
            logger.error(
                f"Failed to load {color.RED}{attr}{color.RESET}: {e}"
            )


def save_training_config(config: DictConfig, output_dir: Path) -> None:
    """Save training configuration to output directory."""
    from omegaconf import OmegaConf

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "training_config.yaml"

    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    logger.info(f"Training configuration saved to: {config_path}")


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


class TrainingMonitor:
    """Monitor training progress and metrics."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.metrics_history = []

    def log_step(self, metrics: dict):
        """Log metrics for a training step."""
        self.step_count += 1
        self.metrics_history.append(metrics)

        if self.step_count % self.log_interval == 0:
            self._print_metrics(metrics)

    def _print_metrics(self, metrics: dict):
        """Print formatted metrics."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {self.step_count} | {metrics_str}")

    def get_average_metrics(self, last_n_steps: Optional[int] = None) -> dict:
        """Get average metrics over last n steps."""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-last_n_steps:] if last_n_steps else self.metrics_history

        if not recent_metrics:
            return {}

        # Calculate averages
        avg_metrics = {}
        for key in recent_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in recent_metrics) / len(recent_metrics)

        return avg_metrics
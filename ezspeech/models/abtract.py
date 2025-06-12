from omegaconf import DictConfig
from typing import Tuple, Any, Dict, Optional
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from abc import ABC, abstractmethod
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import deque
from ezspeech.utils.common import load_module
from ezspeech.utils import color
from ezspeech.optims.scheduler import NoamAnnealing
from omegaconf import OmegaConf

class SpeechModel(LightningModule, ABC):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.train_dataset = instantiate(
            self.config.dataset.train_ds, _recursive_=False
        )
        self.val_dataset = instantiate(self.config.dataset.val_ds, _recursive_=False)
        self.preprocessor = instantiate(config.model.preprocessor)

        self.spec_augment = instantiate(config.model.spec_augment)
        # Add loss tracking lists
        # Use deque with maxlen=100 to store last 100 losses
        self.window_losses = deque(maxlen=100)

        # List to store mean values for plotting
        self.mean_losses = []

        self.current_step = 0
        self.modules_map = {}
        # Create directory for loss plots if it doesn't exist
        if self.training:
            self.plot_dir = f"{config.loggers.tb.save_dir}/{config.loggers.tb.version}"
            os.makedirs(self.plot_dir, exist_ok=True)

    
        
    def train_dataloader(self) -> DataLoader:

        loaders = self.config.dataset.loaders

        train_dl = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_asr_data,
            shuffle=True,
            **loaders,
        )

        return train_dl

    def val_dataloader(self) -> DataLoader:

        loaders = self.config.dataset.loaders

        val_dl = DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate_asr_data,
            shuffle=False,
            **loaders,
        )

        return val_dl
    # def on_save_checkpoint(self, checkpoint):
    #     print(checkpoint)
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Abstract method for defining training step logic.

        Args:
            batch: Training batch data
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing loss and other metrics
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Abstract method for defining validation step logic.

        Args:
            batch: Validation batch data
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing loss and other metrics
        """
        pass

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            **self.hparams.config.optimizer,
        )
        scheduler = NoamAnnealing(
            optimizer,
            **self.hparams.config.scheduler,
        )
        return [optimizer], [scheduler]

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        steps = list(range(100, len(self.mean_losses) * 100 + 100, 100))

        plt.plot(
            steps, self.mean_losses, label="Training Loss", marker="o", color="blue"
        )
        plt.xlabel("Steps")
        plt.ylabel("Mean Loss (per 100 steps)")
        plt.title("Training Loss Over Time (100-step moving average)")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(self.plot_dir, f"mean_loss_plot.png")
        plt.savefig(plot_path)
        plt.close()

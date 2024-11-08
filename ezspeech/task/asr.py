import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from typing import Any, Dict, List, Optional, Tuple, Union
from ezspeech.data.asr import collate_asr
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
import torch
from omegaconf import DictConfig
class ASR_task(pl.LightningModule):
    def __init__(self, model_config=None,dataset_config=None):
        super().__init__()
        self.save_hyperparameters() #-> This will help us to have self.hparams with key is model_config and dataset_config
        # self.model = instantiate(model_config.model)
    
    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step logic.
        
        Args:
            batch: Current batch of data
            batch_idx: Index of current batch
            
        Returns:
            Training loss
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_accuracy(logits, y)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> None:
        """
        Validation step logic.
        
        Args:
            batch: Current batch of data
            batch_idx: Index of current batch
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.val_accuracy(logits, y)
        self.log('val_acc', self.val_accuracy, on_epoch=True, prog_bar=True)

    def test_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> None:
        """
        Test step logic.
        
        Args:
            batch: Current batch of data
            batch_idx: Index of current batch
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.test_accuracy(logits, y)
        self.log('test_acc', self.test_accuracy, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers for training.
        
        Returns:
            Configured optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # Optional: Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5, 
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }

    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader.
        
        Returns:
            Training DataLoader
        """
        # Replace with your dataset
        train_dataset = None  # YourDataset(...)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_asr
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader.
        
        Returns:
            Validation DataLoader
        """
        # Replace with your dataset
        val_dataset = None  # YourDataset(...)
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_asr
        )

    def test_dataloader(self) -> DataLoader:

        test_dataset = None  # YourDataset(...)
        return DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_asr
        )

if __name__=="__main__":
    a=ASR_task()
    print(a.hparams)
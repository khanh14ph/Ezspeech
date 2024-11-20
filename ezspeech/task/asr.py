import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from typing import Any, Dict, List, Optional, Tuple, Union
from ezspeech.data.asr import collate_asr
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR,StepLR

class ASR_ctc_task(pl.LightningModule):
    def __init__(self, model: DictConfig, dataset: DictConfig):
        super(ASR_ctc_task,self).__init__()
        self.save_hyperparameters()  # -> This will help us to have self.hparams with key is model_config and dataset_config
        self.dataset_cfg=dataset
        self.model_cfg=model
        self.encoder=instantiate(self.model_cfg.encoder)
        self.criterion=instantiate(self.model_cfg.criterion)

    def share_step(self, batch):
        x, x_len, label, label_len = batch
        x, x_len = self.encoder(x, x_len)
        loss = self.criterion(x, x_len, label, label_len)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.share_step(batch)

        # Log metrics
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.share_step(batch)

        # Log metrics
        self.log("val_loss", loss, on_step=True, prog_bar=True)


    def train_dataloader(self) -> DataLoader:
        train_dataset=instantiate(self.dataset_cfg.trainset)
        return DataLoader(
            train_dataset,
            batch_size=self.dataset_cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_asr,
        )

    def val_dataloader(self) -> DataLoader:

        # Replace with your dataset
        val_dataset = instantiate(self.dataset_cfg.valset)  # YourDataset(...)
        return DataLoader(
            val_dataset,
            batch_size=self.dataset_cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_asr,
        )
    def configure_optimizers(self):
        # Optimizer with weight decay
        optimizer = torch.optim.Adam(
            self.parameters(),
            **self.model_cfg.manager.optimizer
        )
        return optimizer

    def export_checkpoint(self,new_path):
        checkpoint={"state_dict":{"encoder":self.encoder.state_dict()},"hyper_parameters":self.hparams.model}
        torch.save(checkpoint,new_path)
        print("new checkpoint save to", new_path)

        
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from typing import Any, Dict, List, Optional, Tuple, Union
from ezspeech.data.asr import collate_asr
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf import DictConfig
from torch.optim import AdamW
from ezspeech.modules.optimizer import NoamAnnealing
import librosa

class ASR_ctc_task(pl.LightningModule):
    def __init__(self, model: DictConfig, dataset: DictConfig):
        super(ASR_ctc_task, self).__init__()
        self.save_hyperparameters()  # -> This will help us to have self.hparams with key is model_config and dataset_config
        self.dataset_cfg = dataset
        self.model_cfg = model
        self.encoder = instantiate(self.model_cfg.encoder)
        self.predictor=instantiate(self.model_cfg.predictor)
        self.jointer=instantiate(self.model_cfg.jointer)
        self.criterion = instantiate(self.model_cfg.criterion)

    def _shared_step(
        self,
        batch
    ) -> Tuple[torch.Tensor, ...]:
        inputs,input_lengths,targets,target_lengths,audio=batch
        ctc_logits,enc_outs, enc_lens = self.encoder(inputs, input_lengths)

        ys = F.pad(targets, (1, 0))

        pred_outs, __ = self.predictor(ys)


        rnnt_logits = self.jointer(enc_outs, pred_outs)

        loss, ctc_loss, rnnt_loss = self.criterion(
            ctc_logits, rnnt_logits, enc_lens, targets, target_lengths
        )

        return loss, ctc_loss, rnnt_loss

    def training_step(self, batch, batch_idx):
        
        loss, ctc_loss, rnnt_loss = self._shared_step(batch)

        # Log metrics
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_ctc_loss", ctc_loss, on_step=True, prog_bar=True)
        self.log("train_rnnt_loss", rnnt_loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ctc_loss, rnnt_loss = self._shared_step(batch)

        # Log metrics
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        self.log("val_ctc_loss", ctc_loss, on_step=True, prog_bar=True)
        self.log("val_rnnt_loss", rnnt_loss, on_step=True, prog_bar=True)
        return loss

    def train_dataloader(self) -> DataLoader:
        train_dataset = instantiate(self.dataset_cfg.trainset)
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
        optimizer = torch.optim.Adam(self.parameters(), **self.model_cfg.optimizer)
        scheduler = NoamAnnealing(optimizer, **self.hparams.model.scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def export_checkpoint(self, new_path):
        checkpoint = {
            "state_dict": {"encoder": self.encoder.state_dict()},
            "hyper_parameters": self.hparams.model,
        }
        torch.save(checkpoint, new_path)
        print("new checkpoint save to", new_path)

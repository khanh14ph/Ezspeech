from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


class AlignmentTask(LightningModule):
    def __init__(self, dataset: DictConfig, model: DictConfig):
        super(AlignmentTask, self).__init__()
        self.save_hyperparameters()

        self.audio_encoder = instantiate(model.audio_encoder)
        self.audio_encoder.requires_grad_(False)
        self.audio_pooler = instantiate(model.audio_pooler)
        self.text_encoder = instantiate(model.text_encoder)
        self.text_encoder.requires_grad_(False)
        self.text_projector = instantiate(model.text_projector)
        self.temperature = torch.nn.Parameter(torch.ones([]) * np.log(1 / 7)).to(
            self.device
        )
        self.encoder_is_frozen = True
        self.freeze_after_n_epochs = model.encoder_freeze_epoch

    def train_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.dataset.train_ds, _recursive_=False)
        loaders = self.hparams.dataset.loaders

        train_dl = DataLoader(
            dataset=dataset,
            collate_fn=collate_alignment_data,
            shuffle=True,
            **loaders,
        )

        return train_dl

    def val_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.dataset.val_ds, _recursive_=False)
        loaders = self.hparams.dataset.loaders

        val_dl = DataLoader(
            dataset=dataset,
            collate_fn=collate_alignment_data,
            shuffle=False,
            **loaders,
        )

        return val_dl

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:

        inputs, input_lengths, text_token, text_token_attention_mask = batch

        loss, logits = self._shared_step(
            inputs, input_lengths, text_token, text_token_attention_mask
        )
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:

        inputs, input_lengths, text_token, text_token_attention_mask = batch

        loss, logits = self._shared_step(
            inputs, input_lengths, text_token, text_token_attention_mask
        )

        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def _shared_step(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        tokens: torch.Tensor,
        tokens_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        xs, xs_len = self.audio_encoder(features, feature_lengths)
        audio_context_vector = self.audio_pooler(xs, xs_len)
        text_context_vector = self.text_projector(
            self.text_encoder(tokens, tokens_attention_mask)
        )[:, 0, :]
        logits = (
            audio_context_vector @ text_context_vector.T * torch.exp(self.temperature)
        )

        labels = torch.arange(audio_context_vector.size(0)).to(self.device)

        loss_I = F.cross_entropy(logits.T, labels)
        loss_T = F.cross_entropy(logits, labels)

        loss = (loss_I + loss_T) / 2.0

        return loss, logits

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            **self.hparams.model.optimizer,
        )
        scheduler = NoamAnnealing(
            optimizer,
            **self.hparams.model.scheduler,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def freeze_audio_encoder(self):
        self.encoder.requires_grad_(False)

    def freeze_text_encoder(self):
        self.text_encoder.requires_grad_(False)

    def unfreeze_audio_encoder(self):
        self.encoder.requires_grad_(True)

    def unfreeze_text_encoder(self):
        self.text_encoder.requires_grad_(True)

    # def on_train_epoch_end(self) -> None:
    #     if self.encoder_is_frozen:
    #         if self.freeze_after_n_epochs == self.current_epoch:
    #             self.encoder.requires_grad_(True)
    #             self.text_encoder.requires_grad_(True)
    #             self.backbone_is_frozen = False
    #     return super().on_train_epoch_end()

    def export(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "audio_encoder": self.encoder.state_dict(),
                "audio_pooler": self.audio_pooler.state_dict(),
                "text_encoder": self.text_encoder.state_dict(),
                "text_projector": self.text_projector.state_dict(),
            },
            "hyper_parameters": self.hparams.model,
        }
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint is saved to "{filepath}" ...')

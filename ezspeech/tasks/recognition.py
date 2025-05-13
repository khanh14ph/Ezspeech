from typing import Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import deque

from ezspeech.modules.dataset.dataset import collate_asr_data
from ezspeech.optims.scheduler import NoamAnnealing
from ezspeech.modules.metric.wer import WER


class SpeechRecognitionTask(LightningModule):
    def __init__(self, config: DictConfig):
        super(SpeechRecognitionTask, self).__init__()

        self.save_hyperparameters()

        self.preprocessor = instantiate(config.model.preprocessor)

        self.spec_augment = instantiate(config.model.spec_augment)

        self.encoder = instantiate(config.model.encoder)
        

        self.decoder = instantiate(config.model.decoder)

        self.predictor = instantiate(config.model.predictor)

        self.joint = instantiate(config.model.joint)

        self.rnnt_loss = instantiate(config.model.loss.rnnt_loss)

        self.joint.set_loss(self.rnnt_loss)

        self.ctc_loss = instantiate(config.model.loss.ctc_loss)


        # Add loss tracking lists
        # Use deque with maxlen=100 to store last 100 losses
        self.window_losses = deque(maxlen=100)
        
        # List to store mean values for plotting
        self.mean_losses = []
        
        self.current_step = 0
        
        # Create directory for loss plots if it doesn't exist
        self.plot_dir = f"{config.loggers.tb.save_dir}/{config.loggers.tb.version}"
        os.makedirs(self.plot_dir, exist_ok=True)

    def train_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.train_ds, _recursive_=False)
        loaders = self.hparams.config.dataset.loaders

        train_dl = DataLoader(
            dataset=dataset,
            collate_fn=collate_asr_data,
            shuffle=True,
            **loaders,
        )

        return train_dl

    def val_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.val_ds, _recursive_=False)
        loaders = self.hparams.config.dataset.loaders

        val_dl = DataLoader(
            dataset=dataset,
            collate_fn=collate_asr_data,
            shuffle=False,
            **loaders,
        )

        return val_dl

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        wavs, wav_lengths, targets, target_lengths = batch
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)
        features = self.spec_augment(features, feature_lengths)
        
        loss, ctc_loss, rnnt_loss = self._shared_step(
            features, feature_lengths, targets, target_lengths
        )
        
        # Add current loss to window
        self.window_losses.append(loss.item())
        
        self.current_step += 1
        
        # Calculate and store mean every 100 steps
        if self.current_step % 100 == 0:
            mean_loss = np.mean(self.window_losses)
            self.mean_losses.append(mean_loss)
            self.plot_losses()
            
            # Log mean loss
            self.log("mean_train_loss", mean_loss, sync_dist=True)
        
        self.log("train_ctc_loss", ctc_loss, sync_dist=True, prog_bar=True)
        self.log("train_rnnt_loss", rnnt_loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        wavs, wav_lengths, targets, target_lengths = batch
        features,feature_lengths=self.preprocessor(wavs, wav_lengths)
        

        loss, ctc_loss, rnnt_loss = self._shared_step(
            features, feature_lengths, targets, target_lengths
        )

        self.log("val_ctc_loss", ctc_loss, sync_dist=True)
        self.log("val_rnnt_loss", rnnt_loss, sync_dist=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def _shared_step(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

        enc_outs, enc_lens = self.encoder(inputs, input_lengths)
        ctc_logits = self.decoder(enc_outs)

        decoder_outputs, target_length, states = self.predictor(
            targets=targets, target_length=target_lengths
        )
        rnnt_loss = self.joint(
            encoder_outputs=enc_outs,
            decoder_outputs=decoder_outputs,
            encoder_lengths=enc_lens,
            transcripts=targets,
            transcript_lengths=target_lengths,
        )

        ctc_loss = self.ctc_loss(
            log_probs=ctc_logits,
            targets=targets,
            input_lengths=enc_lens,
            target_lengths=target_lengths,
        )
        loss = 0.7 * ctc_loss + 0.3 * rnnt_loss

        return loss, ctc_loss, rnnt_loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            **self.hparams.config.model.optimizer,
        )
        scheduler = NoamAnnealing(
            optimizer,
            **self.hparams.config.model.scheduler,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        steps = list(range(100, len(self.mean_losses) * 100 + 100, 100))
        
        plt.plot(steps, self.mean_losses, label='Training Loss', marker='o', color='blue')
        plt.xlabel('Steps')
        plt.ylabel('Mean Loss (per 100 steps)')
        plt.title('Training Loss Over Time (100-step moving average)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(self.plot_dir, f'mean_loss_plot.png')
        plt.savefig(plot_path)
        plt.close()

    def export_checkpoint(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "joint": self.joint.state_dict(),
            },
            "hyper_parameters": self.hparams.model,
        }
        print("checkpoint")
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint is saved to "{filepath}" ...')

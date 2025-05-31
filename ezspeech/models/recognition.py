from typing import Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import numpy as np
from ezspeech.modules.metric.wer import WER
from ezspeech.models.abtract import SpeechModel


class SpeechRecognitionTask(SpeechModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        self.save_hyperparameters()
        self.config=config
        self.preprocessor = instantiate(config.model.preprocessor)

        self.spec_augment = instantiate(config.model.spec_augment)

        self.encoder = instantiate(config.model.encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = instantiate(config.model.decoder)

        self.predictor = instantiate(config.model.predictor)

        self.joint = instantiate(config.model.joint)

        self.rnnt_loss = instantiate(config.model.loss.rnnt_loss)

        self.joint.set_loss(self.rnnt_loss)

        self.ctc_loss = instantiate(config.model.loss.ctc_loss)
        self.modules_map={"encoder":self.encoder,
                          "preprocessor":self.preprocessor}

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        if self.global_step==self.config.model.freeze_encoder_steps:
            print(f"UNFREEZE ENCODER after {str(self.config.model.freeze_encoder_steps)} steps")
            for param in self.encoder.parameters():
                param.requires_grad = True
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


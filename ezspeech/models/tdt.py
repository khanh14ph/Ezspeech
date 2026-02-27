from typing import List, Tuple

import torch
from hydra.utils import instantiate
from jiwer import wer
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ezspeech.modules.data.sampler import DynamicBatchSampler
from ezspeech.modules.data.utils.text import Tokenizer
from ezspeech.optims.scheduler import NoamAnnealing


class ASR_tdt_training(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.preprocessor = instantiate(config.model.preprocessor)
        self.spec_augment  = instantiate(config.model.spec_augment)
        self.encoder       = instantiate(config.model.encoder)

        # TDT prediction network and joint
        self.decoder = instantiate(config.model.decoder)
        # Force fuse_loss_wer=False — loss is computed explicitly below
        self.joint = instantiate(config.model.joint, fuse_loss_wer=False)

        # CTC branch
        self.ctc_decoder = instantiate(config.model.ctc_decoder)

        # Losses
        self.tdt_loss        = instantiate(config.model.loss.tdt_loss)
        self.ctc_loss_fn     = instantiate(config.model.loss.ctc_loss)
        self.ctc_loss_weight = config.model.loss.ctc_loss_weight

        # vocab_size used to split joint output into label / duration heads
        self.vocab_size = config.model.vocab_size   # excludes blank

        self.tokenizer = Tokenizer(spe_file=config.dataset.spe_file)
        self.val_predictions = []
        self.val_references  = []

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.train_ds, _recursive_=False)
        dataset.set_tokenizer(self.tokenizer)
        loader  = self.hparams.config.dataset.train_loader
        batcher = DynamicBatchSampler(
            sampler=SequentialSampler(dataset),
            max_batch_duration=loader.max_batch_duration,
            num_buckets=loader.num_bucket,
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batcher,
            collate_fn=dataset.collate_asr_data,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        dataset    = instantiate(self.hparams.config.dataset.val_ds, _recursive_=False)
        dataset.set_tokenizer(self.tokenizer)
        val_loader = self.hparams.config.dataset.val_loader
        return DataLoader(
            dataset=dataset,
            sampler=DistributedSampler(dataset),
            collate_fn=dataset.collate_asr_data,
            shuffle=False,
            **val_loader,
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _forward(self, wavs, wav_lengths, targets, target_lengths, augment=False):
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)
        # if augment:
        #     features = self.spec_augment(features)
        # enc_outs: (B, T, D),  enc_lens: (B,)
        enc_outs, enc_lens = self.encoder(features, feature_lengths)

        # --- TDT branch ---
        # decoder: (B, H, U+1)
        dec_outs, _, _ = self.decoder(targets, target_lengths)
        # joint expects encoder (B, D, T) → transpose from (B, T, D)
        # output: (B, T, U+1, vocab+1 + num_durations)
        joint_out = self.joint(enc_outs.transpose(1, 2), dec_outs)

        # split label logits (B, T, U+1, V+1) and duration logits (B, T, U+1, D)
        label_logits = joint_out[..., : self.vocab_size + 1]
        dur_logits   = joint_out[..., self.vocab_size + 1 :]

        # --- CTC branch ---
        # ConvASRDecoder transposes internally; output: (B, T, V+1)
        ctc_logits = self.ctc_decoder(enc_outs)

        return enc_outs, enc_lens, label_logits, dur_logits, ctc_logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        wavs, wav_lengths, targets, target_lengths = batch

        enc_outs, enc_lens, label_logits, dur_logits, ctc_logits = self._forward(
            wavs, wav_lengths, targets, target_lengths, augment=True
        )

        # TDT loss
        tdt_loss = self.tdt_loss(
            label_acts=label_logits,
            duration_acts=dur_logits,
            targets=targets,
            input_lengths=enc_lens,
            target_lengths=target_lengths,
        )

        # CTC loss
        ctc_loss = self.ctc_loss_fn(
            log_probs=ctc_logits,
            targets=targets,
            input_lengths=enc_lens,
            target_lengths=target_lengths,
        )

        loss = tdt_loss + self.ctc_loss_weight * ctc_loss

        self.log("loss",     loss,     sync_dist=True, prog_bar=True)
        self.log("tdt_loss", tdt_loss, sync_dist=True, prog_bar=True)
        self.log("ctc_loss", ctc_loss, sync_dist=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation  (greedy CTC decode for WER; TDT decode is TODO)
    # ------------------------------------------------------------------

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        wavs, wav_lengths, targets, target_lengths = batch

        _, enc_lens, _, _, ctc_logits = self._forward(
            wavs, wav_lengths, targets, target_lengths, augment=False
        )

        predictions = self._ctc_greedy_decode(ctc_logits, enc_lens)
        references  = self._targets_to_text(targets, target_lengths)

        self.val_predictions.extend(predictions)
        self.val_references.extend(references)

        validation_wer = wer(self.val_references, self.val_predictions)
        self.log("val_wer", validation_wer, sync_dist=True, prog_bar=True)

    def on_validation_epoch_start(self):
        self.val_predictions = []
        self.val_references  = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ctc_greedy_decode(self, logits: torch.Tensor, input_lengths: torch.Tensor) -> List[str]:
        predicted_ids = torch.argmax(logits, dim=-1)  # (B, T)
        predictions = []
        for i, pred_seq in enumerate(predicted_ids):
            seq_len      = input_lengths[i].item()
            pred_seq     = pred_seq[:seq_len]
            unique_seq   = torch.unique_consecutive(pred_seq)
            filtered_seq = unique_seq[unique_seq != self.vocab_size].cpu().numpy().tolist()
            if filtered_seq:
                text = "".join(self.tokenizer.decode(filtered_seq)).replace("_", " ").strip()
            else:
                text = ""
            predictions.append(text)
        return predictions

    def _targets_to_text(self, targets: torch.Tensor, target_lengths: torch.Tensor) -> List[str]:
        references = []
        for i, target_seq in enumerate(targets):
            seq_len = target_lengths[i].item()
            tokens  = self.tokenizer.decode(target_seq[:seq_len].cpu().numpy().tolist())
            references.append("".join(tokens).replace("_", " ").strip())
        return references

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.hparams.config.model.optimizer)
        scheduler = NoamAnnealing(optimizer, **self.hparams.config.model.scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

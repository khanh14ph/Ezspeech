from typing import List, Optional, Tuple

import torch
import torchaudio
from hydra.utils import instantiate
from jiwer import wer
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ezspeech.modules.data.sampler import DynamicBatchSampler
from ezspeech.modules.data.utils.text import Tokenizer
from ezspeech.modules.searcher.tdt import BeamTDTInfer, GreedyTDTInfer
from ezspeech.optims.scheduler import NoamAnnealing
from ezspeech.utils.common import load_module


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

    def export_checkpoint(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "preprocessor": self.preprocessor.state_dict(),
                "encoder":      self.encoder.state_dict(),
                "decoder":      self.decoder.state_dict(),
                "joint":        self.joint.state_dict(),
                "ctc_decoder":  self.ctc_decoder.state_dict(),
            },
            "hyper_parameters": self.hparams.config.model,
        }
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved to "{filepath}"')


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class ASR_tdt_inference:
    """Inference wrapper for a trained hybrid CTC-TDT model.

    Loads a checkpoint produced by ``ASR_tdt_training.export_checkpoint()``
    and exposes three decode modes:

    * ``transcribe``      — TDT greedy decode (fastest)
    * ``transcribe_beam`` — TDT beam search (``default`` or ``maes``)
    * ``transcribe_ctc``  — CTC greedy decode
    """

    def __init__(
        self,
        filepath: str,
        device: str,
        tokenizer_path: str,
        max_symbols_per_step: int = 10,
    ):
        self.device = device
        self.tokenizer = Tokenizer(spe_file=tokenizer_path)
        self.vocab_size = len(self.tokenizer.vocab)  # excludes blank

        (
            self.preprocessor,
            self.encoder,
            self.decoder,
            self.joint,
            self.ctc_decoder,
            self.durations,
        ) = self._load_checkpoint(filepath, device)

        self.greedy_searcher = GreedyTDTInfer(
            decoder_model=self.decoder,
            joint_model=self.joint,
            blank_index=self.vocab_size,
            durations=self.durations,
            max_symbols_per_step=max_symbols_per_step,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self, filepath: str, device: str):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        hparams  = checkpoint["hyper_parameters"]
        weights  = checkpoint["state_dict"]

        preprocessor = load_module(hparams["preprocessor"], weights["preprocessor"], device)
        encoder      = load_module(hparams["encoder"],      weights["encoder"],      device)
        decoder      = load_module(hparams["decoder"],      weights["decoder"],      device)
        joint        = load_module(hparams["joint"],        weights["joint"],        device)
        ctc_decoder  = load_module(hparams["ctc_decoder"],  weights["ctc_decoder"],  device)

        durations = list(hparams["loss"]["tdt_loss"]["durations"])
        return preprocessor, encoder, decoder, joint, ctc_decoder, durations

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def collate_wav(self, speeches: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        wavs       = [b[0] for b in speeches]
        wav_lengths = [torch.tensor(len(f)) for f in wavs]
        max_len    = max(wav_lengths).item()
        padded     = []
        for sig, sig_len in zip(wavs, wav_lengths):
            if sig_len < max_len:
                sig = torch.nn.functional.pad(sig, (0, max_len - sig_len))
            padded.append(sig)
        return torch.stack(padded), torch.stack(wav_lengths)

    @torch.inference_mode()
    def forward_encoder(
        self, speeches: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, x_lens = self.collate_wav(speeches)
        xs, x_lens = self.preprocessor(xs.to(self.device), x_lens.to(self.device))
        enc_outs, enc_lens = self.encoder(xs, x_lens)
        return enc_outs, enc_lens   # (B, T, D), (B,)

    def _load_audio(self, audio_lst: List[str]) -> List[torch.Tensor]:
        return [torchaudio.load(p) for p in audio_lst]

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _hyps_to_text(self, hypotheses) -> List[str]:
        results = []
        for hyp in hypotheses:
            token_ids = hyp.y_sequence.cpu().tolist()
            text = "".join(self.tokenizer.decode(token_ids)).replace("_", " ").strip()
            results.append(text)
        return results

    def _ctc_greedy(self, enc_outs: torch.Tensor, enc_lens: torch.Tensor) -> List[str]:
        logits      = self.ctc_decoder(enc_outs)          # (B, T, V+1)
        predicted   = torch.argmax(logits, dim=-1)        # (B, T)
        results     = []
        for i, seq in enumerate(predicted):
            seq      = seq[: enc_lens[i].item()]
            unique   = torch.unique_consecutive(seq)
            filtered = unique[unique != self.vocab_size].cpu().tolist()
            text     = "".join(self.tokenizer.decode(filtered)).replace("_", " ").strip()
            results.append(text)
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def transcribe(self, audio_lst: List[str]) -> List[str]:
        """TDT greedy decode."""
        audios   = self._load_audio(audio_lst)
        speeches = [a[0] for a in audios]
        enc_outs, enc_lens = self.forward_encoder(speeches)
        # encoder output is (B, T, D); searcher expects (B, D, T)
        hypotheses, = self.greedy_searcher(enc_outs.transpose(1, 2), enc_lens)
        return self._hyps_to_text(hypotheses)

    @torch.inference_mode()
    def transcribe_beam(
        self,
        audio_lst: List[str],
        beam_size: int = 5,
        search_type: str = 'default',
        ngram_lm_model: Optional[str] = None,
        ngram_lm_alpha: float = 0.3,
    ) -> List[str]:
        """TDT beam search decode (``default`` or ``maes``)."""
        audios   = self._load_audio(audio_lst)
        speeches = [a[0] for a in audios]
        enc_outs, enc_lens = self.forward_encoder(speeches)

        searcher = BeamTDTInfer(
            decoder_model=self.decoder,
            joint_model=self.joint,
            durations=self.durations,
            beam_size=beam_size,
            search_type=search_type,
            ngram_lm_model=ngram_lm_model,
            ngram_lm_alpha=ngram_lm_alpha,
        )
        hypotheses, = searcher(enc_outs.transpose(1, 2), enc_lens)
        return self._hyps_to_text(hypotheses)

    @torch.inference_mode()
    def transcribe_ctc(self, audio_lst: List[str]) -> List[str]:
        """CTC greedy decode (faster, uses the CTC branch)."""
        audios   = self._load_audio(audio_lst)
        speeches = [a[0] for a in audios]
        enc_outs, enc_lens = self.forward_encoder(speeches)
        return self._ctc_greedy(enc_outs, enc_lens)

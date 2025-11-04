from typing import Tuple

import torch
import torchaudio
from hydra.utils import instantiate
from jiwer import wer
from typing import List, Tuple
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from ezspeech.modules.data.sampler import DynamicBatchSampler
from ezspeech.modules.data.utils.text import Tokenizer
from ezspeech.optims.scheduler import NoamAnnealing
from ezspeech.utils.common import load_module
from torchaudio.models.decoder import ctc_decoder


class ASR_ctc_training(LightningModule):
    def __init__(self, config: DictConfig):
        super(ASR_ctc_training, self).__init__()

        self.save_hyperparameters()
        self.config = config
        self.preprocessor = instantiate(config.model.preprocessor)

        self.spec_augment = instantiate(config.model.spec_augment)

        self.encoder = instantiate(config.model.encoder)

        self.ctc_decoder = instantiate(config.model.ctc_decoder)

        self.ctc_loss = instantiate(config.model.loss.ctc_loss)

        # Initialize tokenizer for WER calculation
        self.tokenizer_grapheme = Tokenizer(spe_file=config.dataset.spe_file_grapheme)
        # Initialize WER accumulation for validation set
        self.val_predictions = []
        self.val_references = []

    def train_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.train_ds, _recursive_=False)
        dataset.set_tokenizer(self.tokenizer_grapheme)
        sampler = SequentialSampler(dataset)
        loader = self.hparams.config.dataset.train_loader

        dynamic_batcher = DynamicBatchSampler(
            sampler=sampler,
            max_batch_duration=loader.max_batch_duration,
            num_buckets=loader.num_bucket,
        )
        train_dl = DataLoader(
            dataset=dataset,
            batch_sampler=dynamic_batcher,
            collate_fn=dataset.collate_asr_data,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            # shuffle=True,
        )
        return train_dl

    def val_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.config.dataset.val_ds, _recursive_=False)
        dataset.set_tokenizer(self.tokenizer_grapheme)
        val_loader = self.hparams.config.dataset.val_loader
        sampler = DistributedSampler(dataset)
        val_dl = DataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=dataset.collate_asr_data,
            shuffle=False,
            **val_loader,
        )

        return val_dl

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        (
            wavs,
            wav_lengths,
            targets_grapheme,
            target_lengths_grapheme,
        ) = batch
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)
        enc_outs, enc_lens = self.encoder(features, feature_lengths)
        ctc_logits = self.ctc_decoder(enc_outs)

        loss = self.ctc_loss(
            log_probs=ctc_logits,
            targets=targets_grapheme,
            input_lengths=enc_lens,
            target_lengths=target_lengths_grapheme,
        )
        self.log("loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        wavs, wav_lengths, targets, target_lengths = batch
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)
        # Calculate WER
        # Get CTC predictions for WER calculation
        enc_outs, enc_lens = self.encoder(features, feature_lengths)
        ctc_logits = self.ctc_decoder(enc_outs)

        # Decode predictions and targets to text
        predictions = self._ctc_decode_predictions(ctc_logits, enc_lens)
        references = self._targets_to_text(targets, target_lengths)
        # Accumulate predictions and references for validation set WER
        self.val_predictions.extend(predictions)
        self.val_references.extend(references)

        # Calculate WER for entire validation set so far
        validation_wer = wer(self.val_references, self.val_predictions)
        self.log("val_wer", validation_wer, sync_dist=True, prog_bar=True)
        return None

    def on_validation_start(self):
        """
        Reset WER accumulation at the start of validation
        """
        self.val_predictions = []
        self.val_references = []

    def self_condition_step(
        self, logits_lst, enc_lens, targets: torch.Tensor, target_lengths: torch.Tensor
    ):
        loss_lst = []
        for logit in logits_lst:
            loss = self.ctc_loss(
                log_probs=logit,
                targets=targets,
                input_lengths=enc_lens,
                target_lengths=target_lengths,
            )
            loss_lst.append(loss)
        return sum(loss_lst) / len(loss_lst)

    def _ctc_decode_predictions(self, logits, input_lengths):
        """
        Decode CTC predictions to text for WER calculation
        """
        # Get predictions using argmax
        predicted_ids = torch.argmax(logits, dim=-1)  # [B, T]

        predictions = []
        for i, pred_seq in enumerate(predicted_ids):
            # Get the actual length for this sequence
            seq_len = input_lengths[i].item()
            pred_seq = pred_seq[:seq_len]

            # Remove consecutive duplicates and blank tokens
            unique_seq = torch.unique_consecutive(pred_seq)
            # Remove blank token (usually the last token in vocab)
            filtered_seq = (
                unique_seq[unique_seq != len(self.tokenizer_grapheme.vocab)]
                .cpu()
                .numpy()
                .tolist()
            )

            # Convert token IDs to text
            if len(filtered_seq) > 0:
                tokens = self.tokenizer_grapheme.decode(filtered_seq)
                text = "".join(tokens).replace("_", " ").strip()
            else:
                text = ""

            predictions.append(text)

        return predictions

    def _targets_to_text(self, targets, target_lengths):
        """
        Convert target token sequences to text
        """
        references = []
        for i, target_seq in enumerate(targets):
            # Get the actual length for this sequence
            seq_len = target_lengths[i].item()
            target_seq = target_seq[:seq_len]

            # Convert token IDs to text
            tokens = self.tokenizer_grapheme.decode(target_seq.cpu().numpy().tolist())
            text = "".join(tokens).replace("_", " ").strip()
            references.append(text)

        return references

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

    def export_checkpoint(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "preprocessor": self.preprocessor.state_dict(),
                "encoder": self.encoder.state_dict(),
                "ctc_decoder": self.ctc_decoder.state_dict(),
            },
            "hyper_parameters": self.hparams.config.model,
        }
        print("checkpoint")
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint is saved to "{filepath}" ...')


class ASR_ctc_inference:
    def __init__(
        self,
        filepath: str,
        device: str,
        tokenizer_path: str = None,
        lexicon_path=None,
        lm_path=None,
        LM_WEIGHT=1,
        WORD_SCORE=0.5,
    ):
        self.device = device
        self.tokenizer = Tokenizer(spe_file=tokenizer_path)
        self.vocab = self.tokenizer.vocab
        self.blank = len(self.vocab)
        (self.preprocessor, self.encoder, self.ctc_decoder) = self._load_checkpoint(
            filepath, device
        )

        self.beam_search_decoder = ctc_decoder(
            lexicon=lexicon_path,
            tokens=self.vocab,
            lm=lm_path,
            nbest=3,
            beam_size=1500,
            lm_weight=LM_WEIGHT,
            word_score=WORD_SCORE,
        )

    def _load_checkpoint(self, filepath: str, device: str):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        hparams = checkpoint["hyper_parameters"]
        weights = checkpoint["state_dict"]

        preprocessor = load_module(
            hparams["preprocessor"], weights["preprocessor"], device
        )
        encoder = load_module(hparams["encoder"], weights["encoder"], device)

        ctc_decoder = load_module(
            hparams["ctc_decoder"], weights["ctc_decoder"], device
        )

        return preprocessor, encoder, ctc_decoder

    @torch.inference_mode()
    def forward_encoder(
        self, speeches: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs, x_lens = self.collate_wav(speeches)
        xs, x_lens = self.preprocessor(xs.to(self.device), x_lens.to(self.device))
        enc_outs, enc_lens = self.encoder(xs, x_lens)
        return enc_outs, enc_lens

    def transcribe(self, audio_lst):
        audios = [torchaudio.load(i) for i in audio_lst]
        speeches = [i[0] for i in audios]
        sample_rates = [i[1] for i in audios]
        enc, enc_length = self.forward_encoder(speeches)
        res = self.ctc_decode(enc, enc_length)
        return res

    def transcribe_lm(self, audio_lst):
        audios = [torchaudio.load(i) for i in audio_lst]
        speeches = [i[0] for i in audios]
        sample_rates = [i[1] for i in audios]
        enc, enc_length = self.forward_encoder(speeches)
        res = self.beam_search_decode(enc, enc_length)
        return res

    @torch.inference_mode()
    def ctc_decode(self, enc_outs: torch.Tensor, enc_lens: torch.Tensor):
        logits = self.ctc_decoder(enc_outs)
        predicted_ids = torch.argmax(logits, dim=-1)

        predicted_transcripts = []
        for i, pred_seq in enumerate(predicted_ids):
            seq_len = enc_lens[i].item()
            pred_seq = pred_seq[:seq_len]

            unique_seq = torch.unique_consecutive(pred_seq)
            # Filter out blank tokens
            filtered_seq = unique_seq[unique_seq != self.blank]
            # Decode using tokenizer
            tokens = self.tokenizer.decode(filtered_seq.cpu().numpy().tolist())
            transcript = "".join(tokens).replace("_", " ").strip()
            predicted_transcripts.append(transcript)
        return predicted_transcripts
    @torch.inference_mode()
    def beam_search_decode(self, enc_outs: torch.Tensor, enc_lens: torch.Tensor):
        beam_search_result = self.beam_search_decoder(enc_outs)
        predicted_transcripts=[]
        for i in beam_search_result:
            beam_search_transcript = " ".join(i[0].words).strip()
            predicted_transcripts.append(beam_search_transcript)
        return predicted_transcripts
    @torch.inference_mode()
    def collate_wav(
        self, speeches: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        wavs = [b[0] for b in speeches]
        wav_lengths = [torch.tensor(len(f)) for f in wavs]
        max_audio_len = max(wav_lengths).item()
        new_audio_signal = []
        for sig, sig_len in zip(wavs, wav_lengths):
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            new_audio_signal.append(sig)
        new_audio_signal = torch.stack(new_audio_signal)
        audio_lengths = torch.stack(wav_lengths)
        return new_audio_signal, audio_lengths


if __name__ == "__main__":
    a = ASR_ctc_inference()

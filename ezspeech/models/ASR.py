import os
import glob
import time
import shutil
from typing import Tuple, List

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

from ezspeech.models.abtract import SpeechModel
from ezspeech.modules.dataset.utils.text import Tokenizer
from ezspeech.modules.metric.wer import WER
from ezspeech.modules.decoder.rnnt.rnnt_decoding.rnnt_decoding import RNNTDecoding
from ezspeech.utils.common import load_module
class RNNT_CTC_Training(SpeechModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.save_hyperparameters()
        self.config = config

        # set up tokenizer
        self.tokenizer = instantiate(self.config.dataset.tokenizer)
        self.train_dataset.set_tokenizer(self.tokenizer)
        self.val_dataset.set_tokenizer(self.tokenizer)

        self.encoder = instantiate(config.model.encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.ctc_decoder = instantiate(config.model.ctc_decoder)
        self.decoder = instantiate(config.model.decoder)

        self.joint = instantiate(config.model.joint)

        self.rnnt_loss = instantiate(config.loss.rnnt_loss)

        self.joint.set_loss(self.rnnt_loss)

        self.ctc_loss = instantiate(config.loss.ctc_loss)
        self.modules_map = {
            "preprocessor": self.preprocessor,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "joint": self.joint,
            "ctc_decoder": self.ctc_decoder,
        }

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        if self.global_step == self.config.model.freeze_encoder_steps:
            print(
                f"UNFREEZE ENCODER after {str(self.config.model.freeze_encoder_steps)} steps"
            )
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
        features, feature_lengths = self.preprocessor(wavs, wav_lengths)

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
        ctc_logits = self.ctc_decoder(enc_outs)

        decoder_outputs, target_length, states = self.decoder(
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

    def export_ez_checkpoint(self, export_path):
        checkpoint = {
            "state_dict": {
                "preprocessor": self.preprocessor.state_dict(),
                "encoder": self.encoder.state_dict(),
                "ctc_decoder": self.ctc_decoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "joint": self.joint.state_dict(),
            },
            "hyper_parameters": self.config.model,
        }

        config = self.config
        shutil.copy(self.config.dataset.tokenizer.spe_file, export_path)
        OmegaConf.save(config, export_path + "/config.yaml")
        torch.save(checkpoint, export_path + "/model.ckpt")
        print(" ")





class RNNT_CTC_Inference(object):
    def __init__(
        self, filepath: str, device: str, tokenizer_path: str = None, decoding_cfg=None
    ):
        self.blank = 0
        self.beam_size = 5

        self.device = device
        self.tokenizer = Tokenizer(spe_file=tokenizer_path)
        self.vocab = self.tokenizer.vocab
        (
            self.preprocessor,
            self.encoder,
            self.ctc_decoder,
            self.predictor,
            self.joint,
        ) = self._load_checkpoint(filepath, device)
        self.rnnt_decoding = RNNTDecoding(
            decoding_cfg, self.predictor, self.joint, self.vocab
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

        predictor = load_module(hparams["decoder"], weights["decoder"], device)
        joint = load_module(hparams["joint"], weights["joint"], device)

        return preprocessor, encoder, ctc_decoder, predictor, joint

    @torch.inference_mode()
    def forward_encoder(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs, x_lens = self.collate_wav(speeches, sample_rates)
        xs, x_lens = self.preprocessor(xs.to(self.device), x_lens.to(self.device))
        enc_outs, enc_lens = self.encoder(xs, x_lens)

        return enc_outs, enc_lens

    @torch.inference_mode()
    def greedy_tdt_decode(
        self, enc_outs: List[torch.Tensor], enc_lens: List[torch.Tensor]
    ):
        hypothesises = self.rnnt_decoding.rnnt_decoder_predictions_tensor(
            encoder_output=enc_outs, encoded_lengths=enc_lens
        )
        text_token = [
            self.idx_to_token(hypothesis["idx_sequence"]) for hypothesis in hypothesises
        ]
        text = ["".join(i).replace("|", " ") for i in text_token]
        text = [i.replace("▁", " ").strip() for i in text]
        return text

    @torch.inference_mode()
    def greedy_ctc_decode(
        self, enc_outs: List[torch.Tensor], enc_lens: List[torch.Tensor]
    ):
        logits = self.ctc_decoder(enc_outs)
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids = [torch.unique_consecutive(i) for i in predicted_ids]
        predicted_tokens = [self.idx_to_token(i) for i in predicted_ids]
        predicted_transcripts = ["".join(i) for i in predicted_tokens]
        predicted_transcripts = [
            i.replace("▁", " ").strip() for i in predicted_transcripts
        ]
        return predicted_transcripts

    def idx_to_token(self, lst):
        lst = [j for j in lst if j != len(self.vocab)]
        return [self.vocab[i] for i in lst]

    def collate_wav(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(speeches) == len(sample_rates), "The batch is mismatch."

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
    config = OmegaConf.load("/home4/khanhnd/Ezspeech/config/test/test.yaml")
    model = instantiate(config.model)
    import torchaudio

    audio2, sr2 = torchaudio.load(
        "/home4/tuannd/vbee-asr/self-condition-asr/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/train-clean-100/374/180298/374-180298-0004.flac"
    )
    audio1, sr1 = torchaudio.load(
        "/home4/tuannd/vbee-asr/self-condition-asr/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/train-clean-100/374/180298/374-180298-0004.flac"
    )
    audio = [audio1, audio2]
    sr = [sr1, sr2]
    enc, enc_length = model.forward_encoder(audio, sr)
    print(model.greedy_ctc_decode(enc, enc_length))
    print(model.greedy_tdt_decode(enc, enc_length))

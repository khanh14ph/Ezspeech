import os
import glob
import time
import shutil
from typing import Tuple, List
import torchaudio
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
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
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

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
        # features = self.spec_augment(features, feature_lengths)

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
            transcript_lengths=target_length,
        )
        for i,j in zip(enc_lens, target_lengths):
            if i < j:
                print("Warning: Encoder length is less than target length. This may cause issues with CTC loss calculation.")
                print(f"Encoder length: {i}, Target length: {j}")
        ctc_loss = self.ctc_loss(
            log_probs=ctc_logits,
            targets=targets,
            input_lengths=enc_lens,
            target_lengths=target_lengths,
        )
        loss =0.5* ctc_loss + 0.5* rnnt_loss

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
            # self.predictor,
            # self.joint,
        ) = self._load_checkpoint(filepath+"/model.ckpt", device)
        # self.rnnt_decoding = RNNTDecoding(
        #     decoding_cfg.rnnt, self.predictor, self.joint, self.vocab
        # )

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

        # predictor = load_module(hparams["decoder"], weights["decoder"], device).eval()
        # joint = load_module(hparams["joint"], weights["joint"], device).eval()

        # return preprocessor, encoder, ctc_decoder, predictor, joint
        return preprocessor, encoder, ctc_decoder

    @torch.inference_mode()
    def forward_encoder(
        self, speeches: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs, x_lens = self.collate_wav(speeches)
        xs, x_lens = self.preprocessor(xs.to(self.device), x_lens.to(self.device))
        enc_outs, enc_lens = self.encoder(xs, x_lens)
        return enc_outs, enc_lens

    @torch.inference_mode()
    def tdt_decode(
        self,
        enc_outs: List[torch.Tensor],
        enc_lens: List[torch.Tensor],
        previous_hypotheses=None,
    ):
        hypothesises = self.rnnt_decoding.rnnt_decoder_predictions_tensor(
            encoder_output=enc_outs,
            encoded_lengths=enc_lens,
            partial_hypotheses=previous_hypotheses,
        )
        print(hypothesises)
        text_token = [
            self.idx_to_token(hypothesis.y_sequence) for hypothesis in hypothesises
        ]
        text = ["".join(i).replace("|", " ") for i in text_token]
        text = [i.replace("▁", " ").strip() for i in text]
        return text,hypothesises
    def decode_hybrid(self,enc_outs: List[torch.Tensor],
        enc_lens: List[torch.Tensor],
        previous_hypotheses=None,
        type_decode="ctc"):
        if type_decode=="ctc":
            return self.ctc_decode(enc_outs,enc_lens)
        else:
            return self.tdt_decode(enc_outs,enc_lens,previous_hypotheses)
    def transcribe(self, audio_lst,previous_hyp=None):
        audios = [torchaudio.load(i) for i in audio_lst]
        speeches = [i[0] for i in audios]
        sample_rates = [i[1] for i in audios]
        enc, enc_length = self.forward_encoder(speeches)
        res = self.ctc_decode(enc, enc_length)
        return res

    def transcribe_chunk(self, audio_signal):
        self.pre_encode_cache_size = self.encoder.streaming_cfg.pre_encode_cache_size[1]

        audio_signal, audio_signal_len = self.collate_wav([audio_signal])

        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal.to(self.device),
            length=audio_signal_len.to(self.device),
        )
        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += self.cache_pre_encode.shape[1]
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size :]
        with torch.no_grad():
            (
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = self.stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                drop_extra_pre_encoded=None,
            )
        # step_num += 1

        return transcribed_texts[0]

    def stream_step(
        self,
        processed_signal: Tensor,
        processed_signal_length: Tensor = None,
        cache_last_channel: Tensor = None,
        cache_last_time: Tensor = None,
        cache_last_channel_len: Tensor = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: List = None,
        drop_extra_pre_encoded: int = None,
    ):
        (
            encoded,
            encoded_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
        ) = self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )
        best_hyp = self.decode_hybrid(
            enc_outs=encoded,
            enc_lens=encoded_len,
            previous_hypotheses=previous_hypotheses,
            type_decode="ctc"
        )
        result = [
            best_hyp,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
            best_hyp,
        ]
        return tuple(result)

    def transcribe_streaming(self, audio_file_path):

        ENCODER_STEP_LENGTH = 80  # Example value, adjust as needed
        lookahead_size = 1040  # Example value, adjust as needed
        chunk_size = lookahead_size + ENCODER_STEP_LENGTH
        SAMPLE_RATE = 16000

        left_context_size = self.encoder.att_context_size[0]
        self.encoder.set_default_att_context_size(
            [left_context_size, int(lookahead_size / ENCODER_STEP_LENGTH)]
        )
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = (
            self.encoder.get_initial_cache_state(batch_size=1)
        )
        self.previous_hypotheses = None
        self.pred_out_stream = None
        step_num = 0
        self.pre_encode_cache_size = self.encoder.streaming_cfg.pre_encode_cache_size[1]
        # cache-aware models require some small section of the previous processed_signal to
        # be fed in at each timestep - we initialize this to a tensor filled with zeros
        # so that we will do zero-padding for the very first chunk(s)
        num_channels = 80
        self.cache_pre_encode = torch.zeros(
            (1, num_channels, self.pre_encode_cache_size), device=self.device
        )
        import numpy as np
        import time

        # Constants

        waveform, sample_rate = torchaudio.load(audio_file_path)

        # Ensure the sample rate matches
        if sample_rate != SAMPLE_RATE:
            print(
                f"Warning: Expected sample rate {SAMPLE_RATE}, but got {sample_rate}."
            )

        print("Processing audio file...")

        # Calculate the number of chunks
        num_samples = waveform.size(1)
        chunk_samples = int(SAMPLE_RATE * chunk_size / 1000)
        num_chunks = num_samples // chunk_samples
        # Process the audio in chunks
        for i in range(num_chunks + 1):
            start_sample = i * chunk_samples
            end_sample = start_sample + chunk_samples
            signal = waveform[0, start_sample:end_sample]  # Get the chunk

            if signal.numel() == 0:
                break  # End of file
            signal = signal.unsqueeze(0)
            text = self.transcribe_chunk(signal)  # Convert to numpy for processing
            # print(f"\r{' ' * 100}", end="")  # Clear previous line
            if text.strip():  # Only print if there's actual text
                print(text, end=" ", flush=True)

            time.sleep(chunk_size / 1000)  # Simulate processing time for each chunk

        print("\nAudio file processing completed.")

    @torch.inference_mode()
    def ctc_decode(self, enc_outs: List[torch.Tensor], enc_lens: List[torch.Tensor]):
        logits = self.ctc_decoder(enc_outs)
        predicted_ids = torch.argmax(logits, dim=-1)
        # predicted_ids = [torch.unique_consecutive(i) for i in predicted_ids]
        print("predicted_ids:", predicted_ids)
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

    config = OmegaConf.load("/home4/khanhnd/Ezspeech/config/test/test.yaml")
    model = instantiate(config.model)
    import torchaudio

    audio1 = "/home4/khanhnd/vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R170.wav"
    #THE ENGLISH FORWARDED TO THE FRENCH BASKETS OF FLOWERS OF WHICH THEY HAD MADE A PLENTIFUL PROVISION TO GREET THE ARRIVAL OF THE YOUNG PRINCESS THE FRENCH IN RETURN INVITED THE ENGLISH TO A SUPPER WHICH WAS TO BE GIVEN THE NEXT DAY
    # audio = [audio1]
    # audio2="/home4/tuannd/vbee-asr/self-condition-asr/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/test-clean/6930/75918/6930-75918-0000.flac"
    # print(model.transcribe_streaming(audio1))
    tex1=model.transcribe([audio1])
    print(tex1)
    # print(model.transcribe(audio)[0]=="hear nothing thing so expezcaris flow boes theatre sus days country tele can never refer one'ssel as i have tou had little money and")

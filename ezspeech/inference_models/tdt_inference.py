from typing import Tuple, List
import time
import torch
from torch.nn.utils.rnn import pad_sequence
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from omegaconf import OmegaConf
from ezspeech.modules.decoder.rnnt.rnnt_decoding.rnnt_decoding import RNNTDecoding
from ezspeech.utils.common import load_module


class LightningASR(object):
    def __init__(
        self, filepath: str, device: str, vocab: str = None, decoding_cfg=None
    ):
        self.blank = 0
        self.beam_size = 5

        self.device = device
        self.vocab = open(vocab).read().splitlines()

        (
            self.preprocessor,
            self.encoder,
            self.decoder,
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
        decoder = load_module(hparams["decoder"], weights["decoder"], device)

        predictor = load_module(hparams["predictor"], weights["predictor"], device)
        joint = load_module(hparams["joint"], weights["joint"], device)

        return preprocessor, encoder, decoder, predictor, joint

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
        text = [i.replace("_", " ").strip() for i in text]
        return text

    @torch.inference_mode()
    def greedy_ctc_decode(
        self, enc_outs: List[torch.Tensor], enc_lens: List[torch.Tensor]
    ):
        logits = self.decoder(enc_outs)
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids = [torch.unique_consecutive(i) for i in predicted_ids]
        predicted_tokens = [self.idx_to_token(i) for i in predicted_ids]
        predicted_transcripts = ["".join(i) for i in predicted_tokens]
        predicted_transcripts = [
            i.replace("_", " ").strip() for i in predicted_transcripts
        ]
        return predicted_transcripts

    def idx_to_token(self, lst):
        lst = [j for j in lst if j != 0]
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

    audio2, sr2 = torchaudio.load("/home4/khanhnd/vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R106.wav")
    audio1,sr1=torchaudio.load("/home4/khanhnd/vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R122.wav")
    audio=[audio1,audio2]
    sr=[sr1,sr2]
    enc, enc_length = model.forward_encoder(audio, sr)
    print(model.greedy_ctc_decode(enc, enc_length))
    print(model.greedy_tdt_decode(enc, enc_length))

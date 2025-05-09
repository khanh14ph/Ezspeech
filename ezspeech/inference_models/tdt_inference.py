import re
import math
from dataclasses import field, dataclass
from typing import Tuple, List, Dict, Union, Optional
import time
import torch
from torch.nn.utils.rnn import pad_sequence
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from omegaconf import OmegaConf
from ezspeech.modules.decoder.rnnt.rnnt_decoding.rnnt_decoding import RNNTDecoding
from ezspeech.utils.common import load_module
from ezspeech.modules.dataset.utils.audio import extract_filterbank


class LightningASR(object):
    def __init__(
        self, filepath: str, device: str, vocab: str = None, decoding_cfg=None
    ):
        self.blank = 0
        self.beam_size = 5

        self.device = device
        self.vocab = open(vocab).read().splitlines()

        (
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
        encoder = load_module(hparams["encoder"], weights["encoder"], device)
        decoder = load_module(hparams["decoder"], weights["decoder"], device)

        predictor = load_module(hparams["predictor"], weights["predictor"], device)
        joint = load_module(hparams["joint"], weights["joint"], device)

        return encoder, decoder, predictor, joint

    @torch.inference_mode()
    def forward_encoder(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs, x_lens = self._preprocess(speeches, sample_rates)

        enc_outs, enc_lens = self.encoder(xs, x_lens)

        return enc_outs, enc_lens

    @torch.inference_mode()
    def transcribe(
        self,
        speeches: List[torch.Tensor],
        sample_rates: List[int],
        searcher: str = "ctc",
    ) -> Tuple[str, float]:

        enc_outs, enc_lens = self.forward_encoder(speeches, sample_rates)
        hypothesises = self.rnnt_decoding.rnnt_decoder_predictions_tensor(
            encoder_output=enc_outs, encoded_lengths=enc_lens
        )
        text_token = [
            self.idx_to_token(hypothesis["idx_sequence"]) for hypothesis in hypothesises
        ]
        text = ["".join(i).replace("|", " ") for i in text_token]
        return text

    def idx_to_token(self, lst):
        return [self.vocab[i] for i in lst]

    def _preprocess(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(speeches) == len(sample_rates), "The batch is mismatch."

        batches = [
            extract_filterbank(speech, sample_rates[i])
            for i, speech in enumerate(speeches)
        ]

        batches = [x[0] if len(x.size()) == 3 else x for x in batches]
        xs = [x.t() for x in batches]
        x_lens = [x.size(0) for x in xs]

        xs = pad_sequence(xs, batch_first=True).to(self.device)
        x_lens = torch.tensor(x_lens, device=self.device)

        return xs, x_lens


if __name__ == "__main__":
    config = OmegaConf.load("/data/khanhnd65/Ezspeech/config/test/test.yaml")
    model = instantiate(config.model)
    import torchaudio

    audio, sr = torchaudio.load(
        "/data/datbt7/dataset/speech/16k/youtube/2019/mn-20190921-other-0006-0585026-0585254.wav"
    )
    print(model.transcribe([audio], [sr]))
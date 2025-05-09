import re
import math
from dataclasses import field, dataclass
from typing import Tuple, List, Dict, Union, Optional
import time
import torch
from torch.nn.utils.rnn import pad_sequence
from pyctcdecode import build_ctcdecoder
import sentencepiece as spm

from ezspeech.modules.decoder.rnnt.rnnt_decoding.rnnt_decoding import RNNTDecoding
from ezspeech.utils.common import load_module
from ezspeech.modules.dataset.utils.audio import extract_filterbank


# from lightspeech.modules.searcher import ctc_decoder
from torchaudio.models.decoder import ctc_decoder


@dataclass
class RNNTHypothesis:
    score: float
    tokens: List[int]
    state: Union[List[torch.Tensor], torch.Tensor]
    lm_states: Optional[List[str]] = field(default_factory=list)
    timesteps: Optional[List[int]] = field(default_factory=list)
    alignment: Optional[List[int]] = field(default_factory=list)
    confidence: Optional[List[float]] = field(default_factory=list)


class LightningASR(object):
    def __init__(self, filepath: str, device: str, vocab: str = None,decoding_cfg=None):
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
        self.rnnt_decoding=RNNTDecoding(decoding_cfg,self.predictor,self.joint,self.vocab)

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
       
        enc_outs, enc_lens=self.forward_encoder(speeches,sample_rates)
        hypothesis=self.rnnt_decoding.rnnt_decoder_predictions_tensor(encoder_output=enc_outs,encoded_lengths=enc_lens)

        return hypothesis
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

    def _ctc_greedy_search(
        self, emissions: torch.Tensor, lengths: torch.Tensor
    ) -> List[Tuple[str, float]]:

        hypos = []
        for i, emission in enumerate(emissions):
            emission = emission[: lengths[i]]

            indices = torch.argmax(emission, dim=1)
            indices = torch.unique_consecutive(indices, dim=0).tolist()
            # indices = torch.masked_select(indices, indices != self.blank)

            indices = [idx for idx in indices if idx != self.blank]
            tokens = self.bpe_model.decode(indices)
            text = "".join(tokens)


            hypos.append(text)  # The alignment is None

        return hypos

    
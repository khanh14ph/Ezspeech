import re
import math
from dataclasses import field, dataclass
from typing import Tuple, List, Dict, Union, Optional
import time
import torch
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import (
    CTCHypothesis,
    # cuda_ctc_decoder,
    CUCTCHypothesis,
)
from pyctcdecode import build_ctcdecoder
import sentencepiece as spm


from lightspeech.datas.audio import extract_filterbank
from lightspeech.datas.text import (
    build_vocab,
    build_lexicon,
    tokenize,
    check_end_word,
)
from lightspeech.utils.common import load_module

from lightspeech.utils.alignment import (
    get_trellis,
    backtrack,
    merge_tokens,
    merge_words,
)

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
    def __init__(self, filepath: str, device: str, searcher: str, vocab: str = None,bpe_model=None):
        self.blank = 0
        self.beam_size = 5

        self.device = device
        self.searcher = searcher
        self.vocab = open(vocab).read().splitlines()
        self.vocab=[i.split("\t")[0] for i in self.vocab]
        self.bpe_model= spm.SentencePieceProcessor(model_file=bpe_model)
        self.lexicon = build_lexicon()

        (
            self.encoder,
            self.decoder,
            self.predictor,
            self.joint,
        ) = self._load_checkpoint(filepath, device)

        if searcher == "rnnt":
            token = torch.tensor([[self.blank]], device=device)
            _, state = self.predictor(token)
            state = torch.zeros_like(state)
            self.init_token = token
            self.init_state = state

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
    def __call__(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs, x_lens = self._preprocess(speeches, sample_rates)

        enc_outs, enc_lens = self.encoder(xs, x_lens)
        dec_outs = self.decoder(enc_outs)

        return dec_outs.cpu(), enc_lens.cpu()

    @torch.inference_mode()
    def stream(
        self,
        speeches: List[torch.Tensor],
        sample_rates: List[int],
        states: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:

        xs, x_lens = self._preprocess(speeches, sample_rates)

        enc_outs, enc_lens, states = self.encoder.infer(xs, x_lens, states)
        dec_outs = self.decoder(enc_outs)

        return dec_outs.cpu(), enc_lens.cpu(), states

    @torch.inference_mode()
    def transcribe(
        self,
        speeches: List[torch.Tensor],
        sample_rates: List[int],
        searcher: str = "ctc",
    ) -> Tuple[str, float]:
        bs = len(speeches)
        xs, x_lens = self._preprocess(speeches, sample_rates)

        if searcher == "ctc":
            dec_outs, dec_lens = self(speeches, sample_rates)
            hyps = self._ctc_greedy_search(dec_outs, dec_lens)

        if searcher == "rnnt":
            print("hello")
            enc_outs, enc_lens = self.encoder(xs, x_lens)
            enc_outs = [enc_outs[i, : enc_lens[i]] for i in range(bs)]
            hyps = [self._rnnt_beam_search(enc_out) for enc_out in enc_outs]

        return hyps

    @torch.inference_mode()
    def force_alignment(
        self,
        speeches: List[torch.Tensor],
        sample_rates: List[int],
        transcripts: List[str],
    ):

        emissions, lengths = self(speeches, sample_rates)

        alignments = []
        for i, emission in enumerate(emissions):
            emission = emission[: lengths[i]]

            length = lengths[i].item()
            duration = speeches[i].size(1) / sample_rates[i]
            tokens = tokenize(transcripts[i], self.vocab, self.lexicon)

            token_indices = [self.vocab.index(token) for token in tokens]

            trellis = get_trellis(emission, token_indices)
            path = backtrack(trellis, emission, token_indices)

            token_segments = merge_tokens(path, tokens, length, duration)
            word_segments = merge_words(token_segments, self.silence)
            alignments.append((token_segments, word_segments))

        return alignments

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

    def _rnnt_beam_search(self, enc_out: torch.Tensor) -> Tuple[str, float]:
        token = self.init_token.clone()
        memory = self._compute_memory(token, self.init_state)

        kept_hyps = [
            RNNTHypothesis(
                score=0.0,
                tokens=[self.blank],
                state=self.init_state,
            )
        ]

        for t, eout in enumerate(enc_out):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                # update decoder state and get next score
                y_rnnt, state, lm_states = self._gen_next_token(
                    eout,
                    max_hyp.tokens,
                    max_hyp.state,
                    max_hyp.lm_states,
                    memory,
                )

                # remove blank token before topk
                topk_values, topk_indices = y_rnnt[1:].topk(self.beam_size)
                topk_indices += 1  # revert indices cause of removing blank

                topk_indices = topk_indices.tolist()
                topk_values = topk_values.tolist()

                # token is blank, don't update sequence,
                # just store the current hypothesis
                blank_score = y_rnnt[self.blank].item()
                blank_prob = math.exp(blank_score)
                blank_token = ""

                kept_hyps.append(
                    RNNTHypothesis(
                        score=max_hyp.score + blank_score,
                        tokens=max_hyp.tokens[:],
                        state=max_hyp.state,
                        lm_states=max_hyp.lm_states,
                        timesteps=max_hyp.timesteps[:] + [t],
                        alignment=max_hyp.alignment[:] + [blank_token],
                        confidence=max_hyp.confidence[:] + [blank_prob],
                    )
                )

                # for each possible step
                for log_prob, k in zip(topk_values, topk_indices):
                    # non-blank token was predicted, update hypothesis
                    prob = math.exp(log_prob)
                    token = self.vocab[k]

                    new_hyp = RNNTHypothesis(
                        score=max_hyp.score + log_prob,
                        tokens=max_hyp.tokens[:] + [k],
                        state=state,
                        lm_states=lm_states,
                        timesteps=max_hyp.timesteps[:] + [t],
                        alignment=max_hyp.alignment[:] + [token],
                        confidence=max_hyp.confidence[:] + [prob],
                    )

                    hyps.append(new_hyp)

                # keep those hypothesis that have scores
                # greater than next search generation
                hyp_max = max(hyps, key=lambda x: x.score).score
                kept_most_prob = [h for h in kept_hyps if h.score > hyp_max]

                # if enough hypothesis have scores
                # greater than next search generation, stop beam search.
                if len(kept_most_prob) >= self.beam_size:
                    kept_hyps = kept_most_prob
                    break

        # sort hypothesis by normalized score
        kept_hyps = sorted(
            kept_hyps, key=lambda x: x.score / len(x.tokens), reverse=True
        )

        # post-processing
        words = kept_hyps[0].alignment

        text = "".join(words)
        text = text.replace("|", " ").strip()
        score = math.exp(kept_hyps[0].score / len(kept_hyps[0].tokens))

        return text

    def _compute_memory(
        self, inp: torch.Tensor, inp_state: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        out, out_state = self.predictor(inp, inp_state)
        memory = {"input": (inp, inp_state), "output": (out, out_state)}
        return memory

    def _gen_next_token(
        self,
        eout: torch.Tensor,
        hyps: List[int],
        state: torch.Tensor,
        lm_states: List[str] = None,
        memory: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # preprocess input
        eout = eout[None, None, :]
        token = eout.new_zeros((1, 1), dtype=torch.long).fill_(hyps[-1])

        # update prediction network
        cached_token, cached_state = memory["input"]
        if token.equal(cached_token) and torch.allclose(state, cached_state):
            pout, state = memory["output"]
        else:
            memory["input"] = token, state
            pout, state = self.predictor(token, state)
            memory["output"] = pout, state

        # compute rnnt output
        jout = self.joint(eout, pout)
        jout = jout.log_softmax(dim=-1).squeeze()

        return jout, state, lm_states


class LightningASRSelfCondition(object):
    def __init__(
        self,
        filepath: str,
        device: str,
    ):
        self.blank = 0

        self.device = device

        self.vocab = (
            open(
                "/data/khanhnd65/lightspeech_khanhnd/src/lightspeech/corpus/char_vocab_1204.txt"
            )
            .read()
            .splitlines()
        )

        self.encoder = self._load_checkpoint(filepath, device)

    def _load_checkpoint(self, filepath: str, device: str):
        checkpoint = torch.load(filepath, map_location="cpu")

        hparams = checkpoint["hyper_parameters"]
        weights = checkpoint["state_dict"]

        encoder = load_module(hparams["encoder"], weights["encoder"], device)

        return encoder

    @torch.inference_mode()
    def __call__(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs, x_lens = self._preprocess(speeches, sample_rates)

        enc_outs, enc_lens = self.encoder(xs, x_lens)

        return enc_outs.cpu(), enc_lens.cpu()

    @torch.inference_mode()
    def transcribe(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> Tuple[str, float]:
        bs = len(speeches)

        enc_outs, enc_lens = self(speeches, sample_rates)
        hyps = self._ctc_greedy_search(enc_outs, enc_lens)

        return hyps

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
            indices = torch.unique_consecutive(indices, dim=0)
            # indices = torch.masked_select(indices, indices != self.blank)

            # tokens = [self.vocab[idx] for idx in indices if idx != self.blank]
            tokens = [self.vocab[idx] for idx in indices]
            text = "".join(tokens)

            text = text.replace("<<", "").replace(">>", "")
            text = text.replace("-", "").replace("|", " ")
            text = re.sub(r"\s+", " ", text).strip()

            hypos.append(text)  # The alignment is None

        return hypos

"""Standalone TDT (Token-and-Duration Transducer) searchers.

Includes:
  - GreedyTDTInfer      : per-sample greedy decoding, also handles batches
  - BeamTDTInfer        : beam search with `default` and `maes` strategies
                          (`maes` supports kenlm n-gram LM shallow fusion)

Stripped vs. NeMo original:
  - No CUDA graphs
  - No alignment / confidence tracking
  - No MALSD / label-looping / BeamBatchedTDTInfer
  - No partial_hypotheses support
  - No NeMo type-checking or abstract neural module inheritance
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

try:
    import kenlm
    _KENLM_AVAILABLE = True
except ImportError:
    _KENLM_AVAILABLE = False

from ezspeech.modules.decoder.rnnt import (
    AbstractRNNTDecoder,
    AbstractRNNTJoint,
    Hypothesis,
    NBestHypotheses,
    label_collate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        return dec_state.to(device)
    if isinstance(dec_state, (list, tuple)):
        return tuple(_states_to_device(s, device) for s in dec_state)
    return dec_state


def is_prefix(x: List[int], pref: List[int]) -> bool:
    """Return True if `pref` is a strict prefix of `x`."""
    if len(pref) >= len(x):
        return False
    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False
    return True


def _pack_greedy(hypotheses: List[Hypothesis], logitlen: torch.Tensor) -> List[Hypothesis]:
    """Convert y_sequence to tensor and record lengths; move dec_state to CPU."""
    logitlen_cpu = logitlen.cpu()
    for idx, hyp in enumerate(hypotheses):
        hyp.y_sequence = (
            hyp.y_sequence.to(torch.long)
            if isinstance(hyp.y_sequence, torch.Tensor)
            else torch.tensor(hyp.y_sequence, dtype=torch.long)
        )
        hyp.length = logitlen_cpu[idx]
        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)
    return hypotheses


def _pack_beam(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    """Convert y_sequence to tensor; strip leading blank sentinel; move dec_state to CPU."""
    for hyp in hypotheses:
        hyp.y_sequence = torch.tensor(hyp.y_sequence, dtype=torch.long)
        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)
        # Remove the leading blank / -1 sentinel added at initialisation
        if hyp.timestamp and len(hyp.timestamp) > 0 and hyp.timestamp[0] == -1:
            hyp.timestamp = hyp.timestamp[1:]
            hyp.y_sequence = hyp.y_sequence[1:]
    return hypotheses


# ---------------------------------------------------------------------------
# Greedy TDT inference
# ---------------------------------------------------------------------------

class GreedyTDTInfer:
    """Greedy TDT decoder (single-sample loop, handles full batches).

    Args:
        decoder_model: RNNTDecoder instance.
        joint_model:   RNNTJoint instance.
        blank_index:   Index of the blank token (= vocab_size).
        durations:     List of supported frame-advance durations, e.g. [0,1,2,3,4].
        max_symbols_per_step: Max non-blank emissions per frame; None = unlimited.
    """

    def __init__(
        self,
        decoder_model: AbstractRNNTDecoder,
        joint_model: AbstractRNNTJoint,
        blank_index: int,
        durations: List[int],
        max_symbols_per_step: Optional[int] = None,
    ):
        self.decoder = decoder_model
        self.joint = joint_model
        self._blank_index = blank_index
        self._SOS = blank_index          # start-of-sequence token is blank
        self.durations = durations
        self.max_symbols = max_symbols_per_step

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        encoder_output: torch.Tensor,   # (B, D, T)
        encoded_lengths: torch.Tensor,  # (B,)
    ) -> Tuple[List[Hypothesis], ...]:
        decoder_train = self.decoder.training
        joint_train = self.joint.training

        with torch.inference_mode():
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            self.decoder.eval()
            self.joint.eval()

            hypotheses = []
            for b in range(encoder_output.size(0)):
                # inseq: (T, 1, D)  — greedy loop indexes dim-0
                inseq = encoder_output[b].unsqueeze(1)
                hyp = self._greedy_decode(inseq, encoded_lengths[b])
                hypotheses.append(hyp)

            packed = _pack_greedy(hypotheses, encoded_lengths)

        self.decoder.train(decoder_train)
        self.joint.train(joint_train)
        return (packed,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pred_step(
        self,
        label,
        hidden,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, object]:
        """Run the prediction network for one step."""
        if isinstance(label, torch.Tensor):
            if label.dtype != torch.long:
                label = label.long()
        else:
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=False, batch_size=batch_size)
            label = label_collate([[label]])
        return self.decoder.predict(label, hidden, add_sos=False)

    def _joint_step(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Run the joint network; log-normalise on CPU."""
        logits = self.joint.joint(enc, pred)
        if not logits.is_cuda:
            logits = logits.log_softmax(dim=-1)
        return logits

    @torch.no_grad()
    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor) -> Hypothesis:
        """Decode a single sequence.

        Args:
            x:       (T, 1, D)
            out_len: scalar tensor with the valid length.
        """
        hyp = Hypothesis(
            score=0.0,
            y_sequence=[],
            dec_state=None,
            timestamp=[],
            token_duration=[],
            last_token=None,
        )

        time_idx = 0
        skip = 1  # default advance; overwritten in loop

        while time_idx < out_len:
            f = x.narrow(0, time_idx, 1)  # (1, 1, D)
            symbols_added = 0
            need_loop = True

            while need_loop and (self.max_symbols is None or symbols_added < self.max_symbols):
                # Prediction network input
                if hyp.last_token is None and hyp.dec_state is None:
                    last_label = self._SOS
                else:
                    last_label = label_collate([[hyp.last_token]])

                g, hidden_prime = self._pred_step(last_label, hyp.dec_state)
                logits = self._joint_step(f, g)          # (1, 1, 1, V+1+D)

                logp = logits[0, 0, 0, : -len(self.durations)].float()
                dur_logp = torch.log_softmax(
                    logits[0, 0, 0, -len(self.durations):].float(), dim=-1
                )

                v, k = logp.max(0)
                k = k.item()
                _, d_k = dur_logp.max(0)
                skip = self.durations[d_k.item()]

                if k != self._blank_index:
                    hyp.y_sequence.append(k)
                    hyp.score += float(v)
                    hyp.timestamp.append(time_idx)
                    hyp.dec_state = hidden_prime
                    hyp.last_token = k
                    hyp.token_duration.append(skip)

                symbols_added += 1
                time_idx += skip
                # Only loop back at the same frame for a non-blank token with duration=0.
                # blank+duration=0 is degenerate; exit inner loop and let outer guard advance.
                need_loop = (k != self._blank_index) and (skip == 0)

            # If time did not advance (blank+dur=0 or max_symbols hit with dur=0),
            # force a minimum step forward to prevent an infinite outer loop.
            if skip == 0:
                time_idx += 1

            if symbols_added == self.max_symbols:
                time_idx += 1

        # Unpack batch dimension from the stored hidden state
        hyp.dec_state = self.decoder.batch_select_state(hyp.dec_state, 0)
        return hyp


# ---------------------------------------------------------------------------
# Beam TDT inference
# ---------------------------------------------------------------------------

class BeamTDTInfer:
    """Beam search for TDT models.

    Supports two strategies:
      ``default`` — standard beam search (time-synchronous).
      ``maes``    — modified Adaptive Expansion Search.

    Args:
        decoder_model:          RNNTDecoder instance.
        joint_model:            RNNTJoint instance.
        durations:              List of supported durations, e.g. [0,1,2,3,4].
        beam_size:              Beam width (>= 1).
        search_type:            ``'default'`` or ``'maes'``.
        score_norm:             Normalise final scores by sequence length.
        return_best_hypothesis: Return the single best hypothesis per sample.
        maes_num_steps:         mAES: number of adaptive expansion steps.
        maes_prefix_alpha:      mAES: max prefix-length difference for prefix search.
        maes_expansion_beta:    mAES: extra candidates beyond beam_size.
        maes_expansion_gamma:   mAES: prune-by-value threshold.
        softmax_temperature:    Scale logits before log-softmax.
    """

    def __init__(
        self,
        decoder_model: AbstractRNNTDecoder,
        joint_model: AbstractRNNTJoint,
        durations: List[int],
        beam_size: int,
        search_type: str = 'default',
        score_norm: bool = True,
        return_best_hypothesis: bool = True,
        maes_num_steps: int = 2,
        maes_prefix_alpha: int = 1,
        maes_expansion_beta: int = 2,
        maes_expansion_gamma: float = 2.3,
        softmax_temperature: float = 1.0,
        ngram_lm_model: Optional[str] = None,
        ngram_lm_alpha: float = 0.3,
    ):
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if search_type not in ('default', 'maes'):
            raise ValueError(f"search_type must be 'default' or 'maes', got '{search_type}'")
        if ngram_lm_model and search_type != 'maes':
            raise ValueError("ngram_lm_model requires search_type='maes'")

        self.decoder = decoder_model
        self.joint = joint_model
        self.durations = durations
        self.blank = decoder_model.blank_idx
        self.vocab_size = decoder_model.vocab_size

        self.beam_size = beam_size
        self.search_type = search_type
        self.score_norm = score_norm
        self.return_best_hypothesis = return_best_hypothesis
        self.softmax_temperature = softmax_temperature

        self.max_candidates = beam_size   # extended for mAES below

        # Pre-compute duration indices
        try:
            self.zero_duration_idx = self.durations.index(0)
        except ValueError:
            self.zero_duration_idx = None
        self.min_non_zero_duration_idx = int(
            np.argmin(np.ma.masked_where(np.array(self.durations) == 0, self.durations))
        )

        if search_type == 'default':
            self.search_algorithm = self.default_beam_search
        else:  # maes
            self.maes_num_steps = int(maes_num_steps)
            self.maes_prefix_alpha = int(maes_prefix_alpha)
            self.maes_expansion_beta = int(maes_expansion_beta)
            self.maes_expansion_gamma = float(maes_expansion_gamma)
            self.max_candidates += maes_expansion_beta

            if self.maes_prefix_alpha < 0:
                raise ValueError("maes_prefix_alpha must be >= 0")
            if self.maes_num_steps < 1:
                raise ValueError("maes_num_steps must be >= 1")
            if self.vocab_size < beam_size + maes_expansion_beta:
                raise ValueError(
                    f"beam_size ({beam_size}) + maes_expansion_beta ({maes_expansion_beta}) "
                    f"must be <= vocab_size ({self.vocab_size})"
                )
            self.search_algorithm = self.modified_adaptive_expansion_search

        # n-gram LM (maes only)
        if ngram_lm_model:
            if not _KENLM_AVAILABLE:
                raise ImportError("kenlm is not installed. Run: pip install pypi-kenlm")
            self.ngram_lm = kenlm.Model(ngram_lm_model)
            self.ngram_lm_alpha = ngram_lm_alpha
        else:
            self.ngram_lm = None
            self.ngram_lm_alpha = ngram_lm_alpha

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        encoder_output: torch.Tensor,   # (B, D, T)
        encoded_lengths: torch.Tensor,  # (B,)
    ) -> Tuple[List[Union[Hypothesis, NBestHypotheses]], ...]:
        decoder_train = self.decoder.training
        joint_train = self.joint.training

        with torch.inference_mode():
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            self.decoder.eval()
            self.joint.eval()

            _p = next(self.joint.parameters())
            dtype = _p.dtype

            hypotheses = []
            for b in tqdm(range(encoder_output.size(0)), desc='Beam TDT', unit='sample'):
                inseq = encoder_output[b : b + 1, : encoded_lengths[b], :]  # (1, T, D)
                if inseq.dtype != dtype:
                    inseq = inseq.to(dtype)

                nbest = self.search_algorithm(inseq, encoded_lengths[b])
                nbest = _pack_beam(nbest)

                if self.return_best_hypothesis:
                    hypotheses.append(nbest[0])
                else:
                    hypotheses.append(NBestHypotheses(n_best_hypotheses=nbest))

        self.decoder.train(decoder_train)
        self.joint.train(joint_train)
        return (hypotheses,)

    # ------------------------------------------------------------------
    # Default beam search
    # ------------------------------------------------------------------

    def default_beam_search(
        self,
        encoder_outputs: torch.Tensor,  # (1, T, D)
        encoded_lengths: torch.Tensor,  # scalar
    ) -> List[Hypothesis]:
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, self.vocab_size - 1)
        durations_beam_k = min(beam, len(self.durations))

        decoder_state = self.decoder.initialize_state(encoder_outputs)
        cache: Dict = {}

        start_hyp = Hypothesis(
            score=0.0,
            y_sequence=[self.blank],
            dec_state=decoder_state,
            timestamp=[-1],
            length=0,
            last_frame=0,
        )
        kept_hyps = [start_hyp]

        for time_idx in range(int(encoded_lengths)):
            hyps = [h for h in kept_hyps if h.last_frame == time_idx]
            kept_hyps = [h for h in kept_hyps if h.last_frame > time_idx]

            while len(hyps) > 0:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                enc_t = encoder_outputs[:, time_idx : time_idx + 1, :]  # (1, 1, D)
                dec_out, dec_state, _ = self.decoder.score_hypothesis(max_hyp, cache)
                logits = self.joint.joint(enc_t, dec_out) / self.softmax_temperature
                logp = torch.log_softmax(logits[0, 0, 0, : -len(self.durations)], dim=-1)
                dur_logp = torch.log_softmax(logits[0, 0, 0, -len(self.durations):], dim=-1)

                # Top-k tokens (excluding blank) and top-k durations
                logp_topks, logp_topk_idxs = logp[:-1].topk(beam_k, dim=-1)
                dur_topks, dur_topk_idxs = dur_logp.topk(durations_beam_k, dim=-1)

                # Top-k (token, duration) pairs by combined log-prob
                combined = torch.cartesian_prod(dur_topks, logp_topks).sum(dim=-1)
                total_topks, total_topk_idxs = combined.topk(beam_k, dim=-1)

                # Expand non-blank hypotheses
                for total_lp, flat_idx in zip(total_topks, total_topk_idxs):
                    token_idx = int(logp_topk_idxs[flat_idx % beam_k])
                    dur_idx = int(dur_topk_idxs[flat_idx // beam_k])
                    duration = self.durations[dur_idx]

                    new_hyp = Hypothesis(
                        score=float(max_hyp.score + total_lp),
                        y_sequence=max_hyp.y_sequence + [token_idx],
                        dec_state=dec_state,
                        timestamp=max_hyp.timestamp + [time_idx + duration],
                        length=encoded_lengths,
                        last_frame=max_hyp.last_frame + duration,
                    )
                    if duration == 0:
                        hyps.append(new_hyp)
                    else:
                        kept_hyps.append(new_hyp)

                # Expand blank hypotheses (blank must have non-zero duration)
                for dur_idx in dur_topk_idxs:
                    dur_idx = int(dur_idx)
                    if dur_idx == self.zero_duration_idx:
                        if dur_topk_idxs.shape[0] == 1:
                            dur_idx = self.min_non_zero_duration_idx
                        else:
                            continue
                    duration = self.durations[dur_idx]
                    new_hyp = Hypothesis(
                        score=float(max_hyp.score + logp[self.blank] + dur_logp[dur_idx]),
                        y_sequence=max_hyp.y_sequence[:],
                        dec_state=max_hyp.dec_state,
                        timestamp=max_hyp.timestamp[:],
                        length=encoded_lengths,
                        last_frame=max_hyp.last_frame + duration,
                    )
                    kept_hyps.append(new_hyp)

                kept_hyps = self._merge_duplicate_hypotheses(kept_hyps)

                if len(hyps) > 0:
                    hyps_max = float(max(hyps, key=lambda x: x.score).score)
                    kept_most_prob = sorted(
                        [h for h in kept_hyps if h.score > hyps_max],
                        key=lambda x: x.score,
                    )
                    if len(kept_most_prob) >= beam:
                        kept_hyps = kept_most_prob
                        break
                else:
                    kept_hyps = sorted(kept_hyps, key=lambda x: x.score, reverse=True)[:beam]

        return self._sort_nbest(kept_hyps)

    # ------------------------------------------------------------------
    # Modified Adaptive Expansion Search (mAES)
    # ------------------------------------------------------------------

    def modified_adaptive_expansion_search(
        self,
        encoder_outputs: torch.Tensor,  # (1, T, D)
        encoded_lengths: torch.Tensor,  # scalar
    ) -> List[Hypothesis]:
        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.initialize_state(
            torch.zeros(1, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        )
        start_hyp = Hypothesis(
            y_sequence=[self.blank],
            score=0.0,
            dec_state=self.decoder.batch_select_state(beam_state, 0),
            timestamp=[-1],
            length=0,
            last_frame=0,
        )
        cache: Dict = {}

        # Prime the decoder for the start hypothesis
        beam_dec_out, beam_state_list = self.decoder.batch_score_hypothesis([start_hyp], cache)
        start_hyp.dec_out = [beam_dec_out[0]]
        start_hyp.dec_state = beam_state_list[0]

        # Initialise LM state
        if self.ngram_lm:
            init_lm_state = kenlm.State()
            self.ngram_lm.BeginSentenceWrite(init_lm_state)
            start_hyp.ngram_lm_state = init_lm_state

        kept_hyps = [start_hyp]

        for time_idx in range(int(encoded_lengths)):
            hyps = [h for h in kept_hyps if h.last_frame == time_idx]
            kept_hyps = [h for h in kept_hyps if h.last_frame > time_idx]

            if not hyps:
                continue

            enc_t = encoder_outputs[:, time_idx : time_idx + 1]  # (1, 1, D)

            # Prefix search to update hypothesis scores
            if self.zero_duration_idx is not None:
                hyps = self._prefix_search(
                    sorted(hyps, key=lambda x: len(x.y_sequence), reverse=True),
                    enc_t,
                )

            list_b = []   # blank emissions
            list_nb = []  # non-blank emissions with non-zero duration

            for n in range(self.maes_num_steps):
                # Batched joint over all current hypotheses
                beam_dec_out = torch.stack([h.dec_out[-1] for h in hyps])  # (H, 1, D)
                beam_logits = self.joint.joint(enc_t, beam_dec_out) / self.softmax_temperature
                beam_logp = torch.log_softmax(
                    beam_logits[:, 0, 0, : -len(self.durations)], dim=-1
                )
                beam_dur_logp = torch.log_softmax(
                    beam_logits[:, 0, 0, -len(self.durations):], dim=-1
                )

                # Top-max_candidates tokens per hypothesis
                beam_logp_topks, beam_idx_topks = beam_logp.topk(self.max_candidates, dim=-1)

                # Combined (duration, token) log-probs: (H, D*max_candidates)
                beam_total = (
                    beam_dur_logp[:, :, None] + beam_logp_topks[:, None, :]
                ).view(len(hyps), -1)
                beam_total_topks, beam_total_topk_idxs = beam_total.topk(self.max_candidates, dim=-1)

                # Prune by value (gamma threshold)
                best_scores = beam_total_topks.max(dim=-1, keepdim=True).values
                masks = beam_total_topks >= best_scores - self.maes_expansion_gamma
                k_idxs = [idxs[mask] for idxs, mask in zip(beam_total_topk_idxs, masks)]

                list_exp = []     # zero-duration non-blank (expand again)
                list_nb_exp = []  # non-zero-duration non-blank

                for h_idx, hyp in enumerate(hyps):
                    for flat_idx in k_idxs[h_idx]:
                        k = int(beam_idx_topks[h_idx][flat_idx % self.max_candidates])
                        dur_idx = int(flat_idx // self.max_candidates)
                        duration = self.durations[dur_idx]
                        total_lp = float(beam_total[h_idx][flat_idx])

                        # Blank must have non-zero duration
                        if k == self.blank and duration == 0:
                            duration = self.durations[self.min_non_zero_duration_idx]

                        new_hyp = Hypothesis(
                            score=hyp.score + total_lp,
                            y_sequence=hyp.y_sequence[:],
                            dec_out=hyp.dec_out[:],
                            dec_state=hyp.dec_state,
                            timestamp=hyp.timestamp[:],
                            length=time_idx,
                            last_frame=hyp.last_frame + duration,
                            ngram_lm_state=hyp.ngram_lm_state,
                        )

                        if k == self.blank:
                            list_b.append(new_hyp)
                        else:
                            new_hyp.y_sequence.append(k)
                            new_hyp.timestamp.append(time_idx + duration)
                            if self.ngram_lm:
                                lm_score, new_hyp.ngram_lm_state = self.compute_ngram_score(
                                    hyp.ngram_lm_state, int(k)
                                )
                                new_hyp.score += self.ngram_lm_alpha * lm_score
                            if duration == 0:
                                list_exp.append(new_hyp)
                            else:
                                list_nb_exp.append(new_hyp)

                # Update decoder states for expanded hypotheses
                to_update = list_nb_exp + list_exp
                if to_update:
                    dec_outs, dec_states = self.decoder.batch_score_hypothesis(to_update, cache)
                    for i, hyp in enumerate(to_update):
                        hyp.dec_out.append(dec_outs[i])
                        hyp.dec_state = dec_states[i]

                list_nb += list_nb_exp

                if not list_exp:
                    # No zero-duration expansions: merge and prune
                    kept_hyps = kept_hyps + list_b + list_nb
                    kept_hyps = self._merge_duplicate_hypotheses(kept_hyps)
                    kept_hyps = sorted(kept_hyps, key=lambda x: x.score, reverse=True)[:beam]
                    break
                else:
                    if n < self.maes_num_steps - 1:
                        hyps = self._merge_duplicate_hypotheses(list_exp)
                    else:
                        # Last mAES step: score zero-duration expansions with blank
                        beam_dec_out2 = torch.stack([h.dec_out[-1] for h in list_exp])
                        beam_logits2 = (
                            self.joint.joint(enc_t, beam_dec_out2) / self.softmax_temperature
                        )
                        beam_logp2 = torch.log_softmax(
                            beam_logits2[:, 0, 0, : -len(self.durations)], dim=-1
                        )
                        beam_dur_logp2 = torch.log_softmax(
                            beam_logits2[:, 0, 0, -len(self.durations):], dim=-1
                        )
                        _, best_dur_idxs = beam_dur_logp2.max(dim=-1)

                        for i, hyp in enumerate(list_exp):
                            dur_idx = int(best_dur_idxs[i])
                            if dur_idx == self.zero_duration_idx:
                                dur_idx = self.min_non_zero_duration_idx
                            total_lp = float(
                                beam_logp2[i, self.blank] + beam_dur_logp2[i, dur_idx]
                            )
                            hyp.score += total_lp
                            hyp.last_frame += self.durations[dur_idx]

                        kept_hyps = kept_hyps + list_b + list_exp + list_nb
                        kept_hyps = self._merge_duplicate_hypotheses(kept_hyps)
                        kept_hyps = sorted(kept_hyps, key=lambda x: x.score, reverse=True)[:beam]

        return self._sort_nbest(kept_hyps)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _merge_duplicate_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Merge hypotheses with identical (y_sequence, last_frame) by log-sum-exp."""
        sorted_hyps = sorted(hypotheses, key=lambda x: x.score, reverse=True)
        kept: Dict = {}
        for hyp in sorted_hyps:
            key = (tuple(hyp.y_sequence), int(hyp.last_frame))
            if key in kept:
                kept[key].score = float(
                    torch.logaddexp(
                        torch.tensor(kept[key].score),
                        torch.tensor(hyp.score),
                    )
                )
            else:
                kept[key] = hyp
        return list(kept.values())

    def _prefix_search(
        self,
        hypotheses: List[Hypothesis],     # sorted longest→shortest
        encoder_output: torch.Tensor,     # (1, 1, D)
    ) -> List[Hypothesis]:
        """Update hypothesis scores using prefix relationships."""
        for curr_idx, curr_hyp in enumerate(hypotheses[:-1]):
            for pref_hyp in hypotheses[curr_idx + 1:]:
                curr_len = len(curr_hyp.y_sequence)
                pref_len = len(pref_hyp.y_sequence)

                if (
                    is_prefix(curr_hyp.y_sequence, pref_hyp.y_sequence)
                    and (curr_len - pref_len) <= self.maes_prefix_alpha
                ):
                    logits = self.joint.joint(encoder_output, pref_hyp.dec_out[-1]) / self.softmax_temperature
                    logp = torch.log_softmax(logits[0, 0, 0, : -len(self.durations)], dim=-1)
                    dur_logp = torch.log_softmax(logits[0, 0, 0, -len(self.durations):], dim=-1)

                    curr_score = pref_hyp.score + float(
                        logp[curr_hyp.y_sequence[pref_len]] + dur_logp[self.zero_duration_idx]
                    )
                    next_lm_state = pref_hyp.ngram_lm_state
                    if self.ngram_lm:
                        lm_score, next_lm_state = self.compute_ngram_score(
                            pref_hyp.ngram_lm_state, int(curr_hyp.y_sequence[pref_len])
                        )
                        curr_score += self.ngram_lm_alpha * lm_score

                    for k in range(pref_len, curr_len - 1):
                        logits = self.joint.joint(encoder_output, curr_hyp.dec_out[k]) / self.softmax_temperature
                        logp = torch.log_softmax(logits[0, 0, 0, : -len(self.durations)], dim=-1)
                        dur_logp = torch.log_softmax(logits[0, 0, 0, -len(self.durations):], dim=-1)
                        curr_score += float(
                            logp[curr_hyp.y_sequence[k + 1]] + dur_logp[self.zero_duration_idx]
                        )
                        if self.ngram_lm:
                            lm_score, next_lm_state = self.compute_ngram_score(
                                next_lm_state, int(curr_hyp.y_sequence[k + 1])
                            )
                            curr_score += self.ngram_lm_alpha * lm_score

                    curr_hyp.score = float(np.logaddexp(curr_hyp.score, curr_score))
        return hypotheses

    def compute_ngram_score(self, current_lm_state, label: int) -> Tuple[float, object]:
        """Score the next token with the kenlm n-gram LM.

        Args:
            current_lm_state: current kenlm.State.
            label: integer token id.

        Returns:
            (lm_score, next_lm_state): log-prob score and updated state.
        """
        next_state = kenlm.State()
        lm_score = self.ngram_lm.BaseScore(current_lm_state, str(label), next_state)
        lm_score *= 1.0 / np.log10(np.e)   # convert log10 → natural log
        return lm_score, next_state

    def _sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort by normalised or raw score, descending."""
        if self.score_norm:
            return sorted(hyps, key=lambda x: x.score / max(len(x.y_sequence), 1), reverse=True)
        return sorted(hyps, key=lambda x: x.score, reverse=True)


# ---------------------------------------------------------------------------
# NeMo-backed batched decoders (CUDA-graph-capable, label-looping)
# ---------------------------------------------------------------------------
# These classes require a NeMo installation and provide significantly faster
# batched inference via GPU-optimised label-looping kernels and CUDA graphs.
# They accept the same decoder/joint interfaces as the standalone classes above.

try:
    from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import (
        GreedyBatchedTDTInfer,
    )
    from nemo.collections.asr.parts.submodules.tdt_beam_decoding import (
        BeamBatchedTDTInfer,
    )
except ImportError:
    GreedyBatchedTDTInfer = None
    BeamBatchedTDTInfer = None

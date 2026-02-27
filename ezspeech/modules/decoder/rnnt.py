import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestamp: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    length: Union[int, torch.Tensor] = 0
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None
    token_duration: Optional[List[int]] = field(default_factory=list)
    last_frame: Optional[int] = None
    ngram_lm_state: Optional[Any] = None


@dataclass
class NBestHypotheses:
    """Container for N-best hypotheses."""
    n_best_hypotheses: Optional[List[Hypothesis]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_collate(labels, device=None):
    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor, not {type(labels)}")
    batch_size = len(labels)
    max_len = max(len(l) for l in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    return torch.tensor(cat_labels, dtype=torch.int64, device=device)


class LSTMDropout(torch.nn.Module):
    """LSTM with optional forget-gate bias init and output dropout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        forget_gate_bias: float = 1.0,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        if forget_gate_bias is not None:
            for name, _ in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].zero_()

        self.dropout = torch.nn.Dropout(dropout) if dropout else None

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, h = self.lstm(x, h)
        if self.dropout:
            x = self.dropout(x)
        return x, h


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class AbstractRNNTJoint(torch.nn.Module, ABC):
    @abstractmethod
    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.joint_after_projection(self.project_encoder(f), self.project_prednet(g))

    @property
    def num_classes_with_blank(self):
        raise NotImplementedError()

    @property
    def num_extra_outputs(self):
        raise NotImplementedError()


class AbstractRNNTDecoder(torch.nn.Module, ABC):
    def __init__(self, vocab_size: int, blank_idx: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        if blank_idx not in [0, vocab_size]:
            raise ValueError("`blank_idx` must be either 0 or vocab_size")

    @abstractmethod
    def predict(self, y, state=None, add_sos=False, batch_size=None):
        raise NotImplementedError()

    @abstractmethod
    def initialize_state(self, y: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def score_hypothesis(self, hypothesis: Hypothesis, cache: dict):
        raise NotImplementedError()

    def batch_score_hypothesis(self, hypotheses, cache):
        raise NotImplementedError()

    def batch_initialize_states(self, decoder_states):
        raise NotImplementedError()

    def batch_select_state(self, batch_states, idx):
        raise NotImplementedError()

    @classmethod
    def batch_aggregate_states_beam(cls, src_states, batch_size, beam_size, indices, dst_states=None):
        raise NotImplementedError()

    @classmethod
    def batch_replace_states_mask(cls, src_states, dst_states, mask, other_src_states=None):
        raise NotImplementedError()

    @classmethod
    def batch_replace_states_all(cls, src_states, dst_states, batch_size=None):
        raise NotImplementedError()

    @classmethod
    def clone_state(cls, states):
        raise NotImplementedError()

    @classmethod
    def batch_split_states(cls, batch_states):
        raise NotImplementedError()

    @classmethod
    def batch_unsplit_states(cls, batch_states, device=None, dtype=None):
        raise NotImplementedError()

    def batch_concat_states(self, batch_states):
        raise NotImplementedError()

    def batch_copy_states(self, old_states, new_states, ids, value=None):
        raise NotImplementedError()

    def mask_select_states(self, states, mask):
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# RNNTDecoder  (normalization_mode=null, blank_as_pad=true, random_state_sampling=false)
# ---------------------------------------------------------------------------

class RNNTDecoder(AbstractRNNTDecoder):
    """RNNT Prediction Network — stateful LSTM, blank-as-pad embedding.

    Args:
        prednet: dict with ``pred_hidden``, ``pred_rnn_layers``, ``dropout``,
                 and optionally ``forget_gate_bias`` (default 1.0).
        vocab_size: vocabulary size excluding blank.
    """

    def __init__(self, prednet: Dict[str, Any], vocab_size: int):
        self.pred_hidden = prednet["pred_hidden"]
        self.pred_rnn_layers = prednet["pred_rnn_layers"]
        blank_idx = vocab_size  # blank is always the last token

        super().__init__(vocab_size=vocab_size, blank_idx=blank_idx)

        dropout = prednet.get("dropout", 0.0)
        forget_gate_bias = prednet.get("forget_gate_bias", 1.0)

        # blank-as-pad embedding: vocab_size+1 tokens, padding_idx=blank_idx
        embed = torch.nn.Embedding(vocab_size + 1, self.pred_hidden, padding_idx=blank_idx)
        dec_rnn = LSTMDropout(
            input_size=self.pred_hidden,
            hidden_size=self.pred_hidden,
            num_layers=self.pred_rnn_layers,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
        )
        self.prediction = torch.nn.ModuleDict({"embed": embed, "dec_rnn": dec_rnn})

    def forward(
        self,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        y = label_collate(targets)
        g, states = self.predict(y, state=states, add_sos=True)
        return g.transpose(1, 2), target_length, states  # (B, H, U+1)

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _p = next(self.parameters())
        device, dtype = _p.device, _p.dtype

        if y is not None:
            if y.device != device:
                y = y.to(device)
            y = self.prediction["embed"](y)          # (B, U, H)
        else:
            B = batch_size if batch_size is not None else (1 if state is None else state[0].size(1))
            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        if add_sos:
            B, U, H = y.shape
            y = torch.cat([torch.zeros((B, 1, H), device=y.device, dtype=y.dtype), y], dim=1)

        g, hid = self.prediction["dec_rnn"](y.transpose(0, 1), state)
        return g.transpose(0, 1), hid              # (B, U[+1], H)

    def initialize_state(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = y.size(0)
        return (
            torch.zeros(self.pred_rnn_layers, B, self.pred_hidden, dtype=y.dtype, device=y.device),
            torch.zeros(self.pred_rnn_layers, B, self.pred_hidden, dtype=y.dtype, device=y.device),
        )

    def score_hypothesis(
        self, hypothesis: Hypothesis, cache: dict
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        device = (
            hypothesis.dec_state[0].device
            if hypothesis.dec_state is not None
            else next(self.parameters()).device
        )
        blank_state = len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx
        target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
        lm_token = target[:, -1]
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            if blank_state:
                y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)
            else:
                y, new_state = self.predict(target, state=hypothesis.dec_state, add_sos=False)
            y = y[:, -1:, :]
            cache[sequence] = (y, new_state)
        return y, new_state, lm_token

    def batch_score_hypothesis(
        self, hypotheses: List[Hypothesis], cache: dict
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        final_batch = len(hypotheses)
        if final_batch == 0:
            raise ValueError("No hypotheses provided!")
        device = next(self.parameters()).device
        tokens, to_process = [], []
        final: List[Optional[Any]] = [None] * final_batch

        for i, hyp in enumerate(hypotheses):
            seq = tuple(hyp.y_sequence)
            if seq in cache:
                final[i] = cache[seq]
            else:
                tokens.append(hyp.y_sequence[-1])
                to_process.append((seq, hyp.dec_state))

        if to_process:
            batch = len(to_process)
            tokens_t = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
            dec_states = self.batch_initialize_states([d for _, d in to_process])
            dec_out, dec_states = self.predict(tokens_t, state=dec_states, add_sos=False, batch_size=batch)

            proc = 0
            for i in range(final_batch):
                if final[i] is None:
                    new_state = self.batch_select_state(dec_states, proc)
                    final[i] = (dec_out[proc], new_state)
                    cache[to_process[proc][0]] = (dec_out[proc], new_state)
                    proc += 1

        return [item[0] for item in final], [item[1] for item in final]

    def batch_initialize_states(self, decoder_states) -> List[torch.Tensor]:
        stacked = torch.stack([torch.stack(ds) for ds in decoder_states])  # (B, 2, L, H)
        return list(stacked.permute(1, 2, 0, 3).contiguous())              # [2 x (L, B, H)]

    def batch_select_state(self, batch_states, idx: int):
        if batch_states is not None:
            return [s[:, idx] for s in batch_states]
        return None

    @classmethod
    def batch_aggregate_states_beam(cls, src_states, batch_size, beam_size, indices, dst_states=None):
        L, _, H = src_states[0].shape
        beam_shape = (L, batch_size, beam_size, H)
        flat_shape = (L, batch_size * beam_size, H)
        idx_exp = indices[None, :, :, None].expand(beam_shape)
        if dst_states is not None:
            torch.gather(src_states[0].view(beam_shape), 2, idx_exp, out=dst_states[0].view(beam_shape))
            torch.gather(src_states[1].view(beam_shape), 2, idx_exp, out=dst_states[1].view(beam_shape))
            return dst_states
        return (
            torch.gather(src_states[0].view(beam_shape), 2, idx_exp).view(flat_shape),
            torch.gather(src_states[1].view(beam_shape), 2, idx_exp).view(flat_shape),
        )

    def batch_concat_states(self, batch_states) -> List[torch.Tensor]:
        state_list = []
        for sid in range(len(batch_states[0])):
            tensors = []
            for sample in batch_states:
                t = sample[sid]
                if not isinstance(t, torch.Tensor):
                    t = torch.stack(t)
                tensors.append(t.unsqueeze(0))
            state_list.append(torch.cat(tensors, 0).transpose(1, 0))  # (L, B, H)
        return state_list

    @classmethod
    def batch_replace_states_mask(cls, src_states, dst_states, mask, other_src_states=None):
        other = other_src_states if other_src_states is not None else dst_states
        dtype = dst_states[0].dtype
        m = mask.unsqueeze(0).unsqueeze(-1)
        torch.where(m, src_states[0].to(dtype), other[0].to(dtype), out=dst_states[0])
        torch.where(m, src_states[1].to(dtype), other[1].to(dtype), out=dst_states[1])

    @classmethod
    def batch_replace_states_all(cls, src_states, dst_states, batch_size=None):
        if batch_size is None:
            dst_states[0].copy_(src_states[0])
            dst_states[1].copy_(src_states[1])
        else:
            dst_states[0][:, :batch_size].copy_(src_states[0][:, :batch_size])
            dst_states[1][:, :batch_size].copy_(src_states[1][:, :batch_size])

    @classmethod
    def clone_state(cls, state):
        return state[0].clone(), state[1].clone()

    @classmethod
    def batch_split_states(cls, batch_states):
        return [
            (s1.squeeze(1), s2.squeeze(1))
            for s1, s2 in zip(batch_states[0].split(1, dim=1), batch_states[1].split(1, dim=1))
        ]

    @classmethod
    def batch_unsplit_states(cls, batch_states, device=None, dtype=None):
        return (
            torch.stack([s[0] for s in batch_states], dim=1).to(device=device, dtype=dtype),
            torch.stack([s[1] for s in batch_states], dim=1).to(device=device, dtype=dtype),
        )

    def batch_copy_states(self, old_states, new_states, ids, value=None):
        for sid in range(len(old_states)):
            if value is None:
                old_states[sid][:, ids, :] = new_states[sid][:, ids, :]
            else:
                old_states[sid][:, ids, :] = value
        return old_states

    def mask_select_states(self, states, mask):
        return states[0][:, mask], states[1][:, mask]


# ---------------------------------------------------------------------------
# RNNTJoint  (log_softmax=null, preserve_memory=false, fuse_loss_wer=true)
# ---------------------------------------------------------------------------

class RNNTJoint(AbstractRNNTJoint):
    """RNNT Joint Network.

    Args:
        jointnet: dict with ``joint_hidden``, ``activation``, ``dropout``,
                  ``encoder_hidden``, ``pred_hidden``.
        num_classes: vocabulary size excluding blank.
        num_extra_outputs: extra output heads (e.g. TDT durations).
        log_softmax: None=auto (CPU only), True=always, False=never.
        fuse_loss_wer: enable sub-batch fused loss/WER.
        fused_batch_size: sub-batch size for fused mode.
    """

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        num_extra_outputs: int = 0,
        log_softmax: Optional[bool] = None,
        fuse_loss_wer: bool = False,
        fused_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self._vocab_size = num_classes
        self._num_extra_outputs = num_extra_outputs
        self._num_classes = num_classes + 1 + num_extra_outputs  # +1 blank

        self.log_softmax = log_softmax
        self._fuse_loss_wer = fuse_loss_wer
        self._fused_batch_size = fused_batch_size
        self._loss = None
        self._wer = None

        if fuse_loss_wer and fused_batch_size is None:
            raise ValueError("`fuse_loss_wer` requires `fused_batch_size`")

        self.encoder_hidden = jointnet["encoder_hidden"]
        self.pred_hidden = jointnet["pred_hidden"]
        self.joint_hidden = jointnet["joint_hidden"]
        dropout = jointnet.get("dropout", 0.0)
        activation = jointnet["activation"].lower()

        self.enc = torch.nn.Linear(self.encoder_hidden, self.joint_hidden)
        self.pred = torch.nn.Linear(self.pred_hidden, self.joint_hidden)

        if activation == "relu":
            act_fn = torch.nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            act_fn = torch.nn.Sigmoid()
        elif activation == "tanh":
            act_fn = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = [act_fn]
        if dropout:
            layers.append(torch.nn.Dropout(p=dropout))
        layers.append(torch.nn.Linear(self.joint_hidden, self._num_classes))
        self.joint_net = torch.nn.Sequential(*layers)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[torch.Tensor] = None,
        transcripts: Optional[torch.Tensor] = None,
        transcript_lengths: Optional[torch.Tensor] = None,
        compute_wer: bool = False,
    ):
        encoder_outputs = encoder_outputs.transpose(1, 2)   # (B, T, D)
        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        if not self._fuse_loss_wer:
            if decoder_outputs is None:
                raise ValueError("decoder_outputs cannot be None when fuse_loss_wer is False")
            return self.joint(encoder_outputs, decoder_outputs)

        # --- fused sub-batch loop ---
        if self._loss is None or self._wer is None:
            raise ValueError("`fuse_loss_wer` requires loss and wer modules to be set")
        if encoder_lengths is None or transcript_lengths is None:
            raise ValueError("`fuse_loss_wer` requires encoder_lengths and transcript_lengths")

        losses, wers, wer_nums, wer_denoms, target_lengths = [], [], [], [], []
        B = encoder_outputs.size(0)

        for begin in range(0, B, self._fused_batch_size):
            end = min(begin + self._fused_batch_size, B)

            sub_enc = encoder_outputs[begin:end]
            sub_transcripts = transcripts[begin:end]
            sub_enc_lens = encoder_lengths[begin:end]
            sub_transcript_lens = transcript_lengths[begin:end]

            max_enc_len = sub_enc_lens.max()
            max_tgt_len = sub_transcript_lens.max()

            if decoder_outputs is not None:
                sub_enc = sub_enc[:, :max_enc_len]
                sub_dec = decoder_outputs[begin:end, :max_tgt_len + 1]
                sub_joint = self.joint(sub_enc, sub_dec)
                sub_transcripts = sub_transcripts[:, :max_tgt_len]

                loss_reduction = self._loss.reduction
                self._loss.reduction = None
                losses.append(self._loss(
                    log_probs=sub_joint,
                    targets=sub_transcripts,
                    input_lengths=sub_enc_lens,
                    target_lengths=sub_transcript_lens,
                ))
                target_lengths.append(sub_transcript_lens)
                self._loss.reduction = loss_reduction

            if compute_wer:
                self._wer.update(
                    predictions=sub_enc.transpose(1, 2).detach(),
                    predictions_lengths=sub_enc_lens,
                    targets=sub_transcripts.detach(),
                    targets_lengths=sub_transcript_lens,
                )
                wer, wer_num, wer_denom = self._wer.compute()
                self._wer.reset()
                wers.append(wer); wer_nums.append(wer_num); wer_denoms.append(wer_denom)

        loss = self._loss.reduce(losses, target_lengths) if losses else None
        if compute_wer:
            return loss, sum(wers) / len(wers), sum(wer_nums), sum(wer_denoms)
        return loss, None, None, None

    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.enc(encoder_output)

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        return self.pred(prednet_output)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # f: (B, T, H), g: (B, U, H)
        inp = f.unsqueeze(2) + g.unsqueeze(1)   # (B, T, U, H)
        res = self.joint_net(inp)                # (B, T, U, V+1)
        # log_softmax=None: apply only on CPU
        if self.log_softmax is None:
            if not res.is_cuda:
                res = res.log_softmax(dim=-1)
        elif self.log_softmax:
            res = res.log_softmax(dim=-1)
        return res

    @property
    def num_classes_with_blank(self) -> int:
        return self._num_classes

    @property
    def num_extra_outputs(self) -> int:
        return self._num_extra_outputs

    @property
    def loss(self):
        return self._loss

    def set_loss(self, loss):
        self._loss = loss

    @property
    def wer(self):
        return self._wer

    def set_wer(self, wer):
        self._wer = wer

    @property
    def fuse_loss_wer(self) -> bool:
        return self._fuse_loss_wer

    def set_fuse_loss_wer(self, fuse_loss_wer: bool, loss=None, metric=None):
        self._fuse_loss_wer = fuse_loss_wer
        self._loss = loss
        self._wer = metric

    @property
    def fused_batch_size(self) -> Optional[int]:
        return self._fused_batch_size

    def set_fused_batch_size(self, fused_batch_size: int):
        self._fused_batch_size = fused_batch_size

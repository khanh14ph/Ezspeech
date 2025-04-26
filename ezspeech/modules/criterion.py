import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as F_audio

from lightspeech.utils.common import make_padding_mask
from lightspeech.modules.rnnt import RNNTLossNumba
import torch
from torch import nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

class SequenceToSequenceLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        ctc_weight: float = 1.0,
        rnnt_weight: float = 1.0,
    ):
        super(SequenceToSequenceLoss, self).__init__()
        self.blank_label = 0
        self.ctc_weight = ctc_weight
        self.rnnt_weight = rnnt_weight

    def forward(
        self,
        ctc_logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ctc_loss = F.ctc_loss(
            log_probs=ctc_logits.transpose(0, 1),
            targets=targets,
            input_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=self.blank_label,
            zero_infinity=True,
        )
        # rnnt_loss = F_audio.rnnt_loss(
        #     logits=rnnt_logits.to(torch.float32),
        #     targets=targets.int(),
        #     logit_lengths=logit_lengths.int(),
        #     target_lengths=target_lengths.int(),
        #     blank=self.blank_label,
        # )
        # rnnt_loss=ctc_loss
        # loss = self.ctc_weight * ctc_loss + self.rnnt_weight * rnnt_loss

        return ctc_loss




class SequenceToSequenceLoss_numba(nn.modules.loss._Loss):
    def __init__(
        self,
        ctc_weight: float = 1.0,
        rnnt_weight: float = 1.0,
    ):
        super(SequenceToSequenceLoss_numba, self).__init__()
        self.blank_label = 0
        self.ctc_weight = ctc_weight
        self.rnnt_weight = rnnt_weight
        self.rnnt_loss=RNNTLossNumba()
        self.ctc_loss=CTCLoss()

    def forward(
        self,
        ctc_logits: torch.Tensor,
        rnnt_logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ctc_loss = self.ctc_loss(
            log_probs=ctc_logits,
            targets=targets,
            input_lengths=logit_lengths,
            target_lengths=target_lengths,
        )

        rnnt_loss = self.rnnt_loss(
            rnnt_logits,
            targets.to(torch.int64),
            logit_lengths.to(torch.int64),
            target_lengths.to(torch.int64),
        )
        # rnnt_loss=ctc_loss
        loss = self.ctc_weight * ctc_loss + self.rnnt_weight * rnnt_loss

        return loss, ctc_loss, rnnt_loss

class CTCLoss(nn.CTCLoss):

    def __init__(self, blank=0, zero_infinity=False, reduction='mean_batch'):
        self._blank = blank
        # Don't forget to properly call base constructor
        if reduction not in ['none', 'mean', 'sum', 'mean_batch', 'mean_volume']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch, mean_volume]')

        self.config_reduction = reduction
        if reduction == 'mean_batch' or reduction == 'mean_volume':
            ctc_reduction = 'none'
            self._apply_reduction = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_reduction = False
        super().__init__(blank=self._blank, reduction=ctc_reduction, zero_infinity=zero_infinity)

    def reduce(self, losses, target_lengths):
        if self.config_reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.config_reduction == 'mean_volume':
            losses = losses.sum() / target_lengths.sum()  # same as above but longer samples weigh more

        return losses

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # override forward implementation
        # custom logic, if necessary
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        log_probs = log_probs.transpose(1, 0)
        loss = super().forward(
            log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        )
        if self._apply_reduction:
            loss = self.reduce(loss, target_lengths)
        return loss
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as F_audio

from ezspeech.utils.common import make_padding_mask

import torch
from torch import nn


class HybridRNNT_CTC(nn.modules.loss._Loss):
    def __init__(
        self,
        ctc_loss: nn.Module,
        rnnt_loss: nn.Module,
        ctc_weight: float = 1.0,
        rnnt_weight: float = 1.0,
    ):
        super(HybridRNNT_CTC, self).__init__()
        self.ctc_loss = ctc_loss
        self.rnnt_loss = rnnt_loss
        self.ctc_weight = ctc_weight
        self.rnnt_weight = rnnt_weight

    def forward(
        self,
        ctc_logits: torch.Tensor,
        rnnt_logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        loss_ctc_value = self.ctc_loss(
            log_probs=ctc_logits,
            targets=targets,
            input_lengths=logit_lengths,
            target_lengths=target_lengths,
        )
        loss_rnnt_value = self.rnnt_loss(
            acts=rnnt_logits.to(torch.float32),
            labels=targets.int(),
            act_lens=logit_lengths.int(),
            label_lens=target_lengths.int(),
        )
        loss = self.ctc_weight * loss_ctc_value + self.rnnt_weight * loss_rnnt_value

        return loss, loss_ctc_value, loss_rnnt_value

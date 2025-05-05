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
        ctc_weight: float = 1.0,
        rnnt_weight: float = 1.0,
    ):
        super(HybridRNNT_CTC, self).__init__()
        self.blank_label = 0
        self.ctc_weight = ctc_weight
        self.rnnt_weight = rnnt_weight

    def forward(
        self,
        ctc_logits: torch.Tensor,
        rnnt_logits: torch.Tensor,
        logit_lengths: toxrch.Tensor,
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
        rnnt_loss = F_audio.rnnt_loss(
            logits=rnnt_logits.to(torch.float32),
            targets=targets.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=self.blank_label,
        )
        rnnt_loss=ctc_loss
        loss = self.ctc_weight * ctc_loss + self.rnnt_weight * rnnt_loss

        return loss,ctc_loss,rnnt_loss




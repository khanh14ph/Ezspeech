import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio.transforms as T_audio
import torchaudio.functional as F_audio

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
        rnnt_logits: torch.Tensor,
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
            zero_infinity=False,
        )

        rnnt_loss = F_audio.rnnt_loss(
            logits=rnnt_logits,
            targets=targets.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=self.blank_label,
        )

        loss = self.ctc_weight * ctc_loss + self.rnnt_weight * rnnt_loss

        return loss, ctc_loss, rnnt_loss

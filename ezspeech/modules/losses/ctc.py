# ! /usr/bin/python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

__all__ = ["CTCLoss"]


class CTCLoss(nn.CTCLoss):
    def __init__(self, blank_idx, zero_infinity=False, reduction="mean_batch"):
        # Don't forget to properly call base constructor
        if reduction not in ["none", "mean", "sum", "mean_batch", "mean_volume"]:
            raise ValueError(
                "`reduction` must be one of [mean, sum, mean_batch, mean_volume]"
            )

        self.config_reduction = reduction
        if reduction == "mean_batch" or reduction == "mean_volume":
            ctc_reduction = "none"
            self._apply_reduction = True
        elif reduction in ["sum", "mean", "none"]:
            ctc_reduction = reduction
            self._apply_reduction = False
        super().__init__(
            blank=blank_idx, reduction=ctc_reduction, zero_infinity=zero_infinity
        )

    def reduce(self, losses, target_lengths):
        if self.config_reduction == "mean_batch":
            losses = losses.mean()  # global batch size average
        elif self.config_reduction == "mean_volume":
            losses = (
                losses.sum() / target_lengths.sum()
            )  # same as above but longer samples weigh more

        return losses

    def forward(self, log_probs, targets, input_lengths, target_lengths):

        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        log_probs = log_probs.transpose(1, 0)
        loss = super().forward(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )
        if self._apply_reduction:
            loss = self.reduce(loss, target_lengths)
        return loss

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

# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


from ezspeech.modules.losses.rnnt_numba.rnnt_pytorch import TDTLossNumba




class TDTLoss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean_batch', loss_name: str = "default", loss_kwargs=None):
        """
        RNN-T Loss function based on https://github.com/HawkAaron/warp-transducer.
        Optionally, can utilize a numba implementation of the same loss without having to compile the loss,
        albiet there is a small speed penalty for JIT numba compile.

        Note:
            Requires Numba 0.53.0 or later to be installed to use this loss function.

        Losses can be selected via the config, and optionally be passed keyword arguments as follows.

        Examples:
            .. code-block:: yaml

                model:  # RNNT Model config
                    ...
                    loss:
                        loss_name: "warprnnt_numba"
                        warprnnt_numba_kwargs:
                            fastemit_lambda: 0.0

        Warning:
            In the case that GPU memory is exhausted in order to compute RNNTLoss, it might cause
            a core dump at the cuda level with the following error message.

            ```
                ...
                costs = costs.to(acts.device)
            RuntimeError: CUDA error: an illegal memory access was encountered
            terminate called after throwing an instance of 'c10::Error'
            ```

            Please kill all remaining python processes after this point, and use a smaller batch size
            for train, validation and test sets so that CUDA memory is not exhausted.

        Args:
            num_classes: Number of target classes for the joint network to predict.
                In all cases (conventional RNNT, multi-blank RNNT, and TDT model), this equals the token-id
                for the standard "blank" symbol. In particular, say V is the number of non-blank tokens in
                the vocabulary, then in the case of,
                standard RNNT: num_classes = V
                multiblank RNNT: num_classes = V + number-big-blanks (since we store big-blanks before
                                 standard blank, and the standard blank is the last symbol in the vocab)
                TDT: num_classes = V. Note, V here does not include any of the "duration outputs".

            reduction: Type of reduction to perform on loss. Possible values are 
                `mean_batch`, 'mean_volume`, `mean`, `sum` or None.
                `None` will return a torch vector comprising the individual loss values of the batch.
                `mean_batch` will average the losses in the batch
                `mean` will divide each loss by the target length and then average
                `mean_volume` will add up all the losses and divide by sum of target lengths

            loss_name: String that is resolved into an RNNT loss function. Available list of losses
                is ininitialized in `RNNT_LOSS_RESOLVER` dictionary.

            loss_kwargs: Optional Dict of (str, value) pairs that are passed to the instantiated loss
                function.
        """
        super(TDTLoss, self).__init__()

        if reduction not in [None, 'mean', 'sum', 'mean_batch', 'mean_volume']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch, mean_volume]')

        self._blank = 0
        self.reduction = reduction
        self._loss = TDTLossNumba(blank_idx=self._blank, loss_kwargs=loss_kwargs)
        self._fp16_compat_checked = False

    def reduce(self, losses, target_lengths):

        if isinstance(losses, List):
            losses = torch.cat(losses, 0)
            target_lengths = torch.cat(target_lengths, 0)

        if self.reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.reduction == 'mean':
            losses = torch.div(losses, target_lengths).mean()
        elif self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean_volume':
            losses = losses.sum() / target_lengths.sum()  # same as above but longer samples weigh more

        return losses

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Cast to int 64
        targets = targets.long()
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()

        max_logit_len = input_lengths.max()
        max_targets_len = target_lengths.max()

        # Force cast joint to float32
        

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        # of the log_probs tensor, therefore we increment the input_lengths by the difference.
        # This difference is generally small.
        if log_probs.shape[1] != max_logit_len:
            log_probs = log_probs.narrow(dim=1, start=0, length=max_logit_len).contiguous()

        # Reduce transcript length to correct alignment if additional padding was applied.
        # Transcript: [B, L] -> [B, L']; If L' < L
        if not targets.is_contiguous():
            targets = targets.contiguous()

        if targets.shape[1] != max_targets_len:
            targets = targets.narrow(dim=1, start=0, length=max_targets_len).contiguous()

        # Temporarily override loss reduction
        loss_reduction = self._loss.reduction
        self._loss.reduction = None

        # Compute RNNT loss
        loss = self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)

        # Loss reduction can be dynamic, so reset it after call
        self._loss.reduction = loss_reduction

        # reduce here using our own reduction function
        if self.reduction is not None:
            loss = self.reduce(loss, target_lengths)

        # del new variables that may have been created
        del (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        return loss
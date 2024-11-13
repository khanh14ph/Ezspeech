import torch
import torch.nn.functional as F


class CTCLoss:
    def __init__(self, reduction="mean", zero_infinity=False):
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def __call__(self,x, x_lengths, target, target_lengths):

        res = F.ctc_loss(
            x.transpose(0,1),
            target,
            x_lengths,
            target_lengths,
            blank=0,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )
        return res

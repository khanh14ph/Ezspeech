import math
from typing import Tuple, Optional

import torch
import torch.nn as nn

from lightspeech.utils.common import make_padding_mask


class ConvolutionSubsampling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        factor: int,
        num_filters: int,
        kernel_size: Optional[int] = 5,
        dropout: Optional[float] = 0.1,
    ):
        super(ConvolutionSubsampling, self).__init__()
        self.factor = factor

        stride = 2
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size, stride, padding),
            nn.BatchNorm2d(num_filters),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=num_filters,
            ),
            nn.SiLU(),
        )

        self.proj = nn.Linear(
            num_filters * math.ceil(input_dim / self.factor),
            output_dim,
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs = xs[:, None, :, :]
        masks = make_padding_mask(x_lens, xs.size(2))
        masks = masks[:, None, :, None]

        masks = masks[:, :, ::2, :]
        xs = self.conv1(xs) * masks
        masks = masks[:, :, ::2, :]
        xs = self.conv2(xs) * masks

        b, c, t, f = xs.size()
        xs = xs.transpose(1, 2).contiguous().view(b, t, c * f)

        xs = self.proj(xs)
        xs = self.drop(xs)

        x_lens = torch.div(x_lens - 1, self.factor, rounding_mode="trunc")
        x_lens = (x_lens + 1).type(torch.long)

        return xs, x_lens
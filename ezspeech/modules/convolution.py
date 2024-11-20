import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import math
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


def make_padding_mask(seq_lens: torch.Tensor, max_time: int):
    bs = seq_lens.size(0)
    device = seq_lens.device
    seq_range = torch.arange(0, max_time, dtype=torch.long, device=device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)
    seq_length = seq_lens.unsqueeze(-1)
    mask = seq_range < seq_length
    return mask


class ConvolutionSubSampling(nn.Module):
    def __init__(
        self, d_input, d_ouput, subsampling_factor, num_filter, kernel_size, dropout
    ):
        super(ConvolutionSubSampling, self).__init__()
        self.subsampling_factor = subsampling_factor
        stride = 2
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_filter, kernel_size, stride, padding),
            nn.BatchNorm2d(num_filter),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filter,
                out_channels=num_filter,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=num_filter,
            ),
            nn.SiLU(),
        )
        self.proj = nn.Linear(
            num_filter * math.ceil(d_input / subsampling_factor), d_ouput
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, length):
        x = x[:, None, :, :]
        masks = make_padding_mask(length, x.size(2))
        masks = masks[:, None, :, None]
        masks = masks[:, :, ::2, :]
        x = self.conv1(x) * masks
        masks = masks[:, :, ::2, :]
        x = self.conv2(x) * masks
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.proj(x)
        x = self.drop(x)
        length = torch.div(length - 1, self.subsampling_factor, rounding_mode="trunc")
        length = (length + 1).type(torch.long)
        return x, length


class Conv2dSubSampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubSampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        side = inputs.shape[1]
        outputs = self.sequential(inputs.unsqueeze(1))

        # print("outputs",outputs)
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(
            batch_size, subsampled_lengths, channels * sumsampled_dim
        )
        new_side = outputs.shape[1]
        ratio = side / new_side
        output_lengths = torch.round(input_lengths / ratio).to(dtype=torch.long)
        return outputs, output_lengths

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn

from ezspeech.utils.common import make_padding_mask


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


class ConvSubsampling(torch.nn.Module):
    """Convolutional subsampling which supports dw-striding approach introduced in:

    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding", "dw-striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        subsampling_conv_chunking_factor (int): Input chunking factor which can be -1 (no chunking)
        1 (auto) or a power of 2. Default is 1
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
    ):
        super(ConvSubsampling, self).__init__()
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a power of 2"
            )
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        # stride
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2
        self._max_cache_len = 0

        # Layer 1

        layers.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._left_padding,
            )
        )
        in_channels = conv_channels
        layers.append(activation)

        for i in range(self._sampling_num - 1):

            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=in_channels,
                )
            )

            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            layers.append(activation)
            in_channels = conv_channels

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv2d_subsampling = True

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, lengths):
        lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        x = x.unsqueeze(1)
        # Transpose to Channel First mode

        # if subsampling_conv_chunking_factor is 1, we split only if needed
        # avoiding a bug / feature limiting indexing of tensors to 2**31
        # see https://github.com/pytorch/pytorch/issues/80020
        x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
        if torch.numel(x) > x_ceil:
            need_to_split = True
        else:
            need_to_split = False
        if need_to_split:
            x, success = self.conv_split_by_batch(x)
            if not success:  # if unable to split by batch, try by channel
                x = self.conv_split_by_channel(x)
        else:
            x = self.conv(x)
        # Flatten Channel and Frequency Axes
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode

        return x, lengths

    def reset_parameters(self):
        # initialize weights
    
        with torch.no_grad():
            # init conv
            scale = 1.0 / self._kernel_size
            dw_max = (self._kernel_size**2) ** -0.5
            pw_max = self._conv_channels**-0.5

            torch.nn.init.uniform_(self.conv[0].weight, -scale, scale)
            torch.nn.init.uniform_(self.conv[0].bias, -scale, scale)

            for idx in range(2, len(self.conv), 3):
                torch.nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                torch.nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                torch.nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                torch.nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)

            # init fc (80 * 64 = 5120 from https://github.com/kssteven418/Squeezeformer/blob/13c97d6cf92f2844d2cb3142b4c5bfa9ad1a8951/src/models/conformer_encoder.py#L487
            fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
            torch.nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
            torch.nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x):
        """Tries to split input by batch, run conv and concat results"""
        b, _, _, _ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
            print(f"using manually set chunking factor: {cf}")
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2**p
            print(f"using auto set chunking factor: {cf}")

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, False

        print(f"conv subsampling: using split batch size {new_batch_size}")
        return (
            torch.cat(
                [self.conv(chunk) for chunk in torch.split(x, new_batch_size, 0)]
            ),
            True,
        )

    def conv_split_by_channel(self, x):
        """For dw convs, tries to split input by time, run conv and concat results"""
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
                print(f"using manually set chunking factor: {cf}")
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2**31, 2))
                cf = 2**p
                print(f"using auto set chunking factor: {cf}")

            new_c = int(c // cf)
            if new_c == 0:
                print(
                    f"chunking factor {cf} is too high; splitting down to one channel."
                )
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                print(
                    f"chunking factor {cf} is too high; splitting down to one timestep."
                )
                new_t = 1

            print(
                f"conv dw subsampling: using split C size {new_c} and split T size {new_t}"
            )
            x = self.channel_chunked_conv(
                self.conv[i * 3 + 2], new_c, x
            )  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat(
                [self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2
            )  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk,
                    pad=(
                        self._kernel_size - 1,
                        self._stride - 1,
                        self._kernel_size - 1,
                        self._stride - 1,
                    ),
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

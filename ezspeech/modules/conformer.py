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
#

import torch
from torch import nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import AccessMixin

import math

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D, CausalConv2D
from nemo.utils import logging

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    LocalAttRelPositionalEncoding,
    MultiHeadAttention,
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    StackingSubsampling,
    SubsamplingReductionModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.asr.parts.utils.regularization_utils import compute_stochastic_depth_drop_probs
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, ChannelType, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging
__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerLayer']
class ConvSubsampling(torch.nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: Transformer-transducer: end-to-end speech recognition with self-attention (https://arxiv.org/pdf/1910.12977.pdf)
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
        subsampling,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super(ConvSubsampling, self).__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []


        if subsampling == 'dw_striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
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
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
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



        if subsampling in ["vggnet", "dw_striding", "striding"]:

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

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x, lengths):
        x=x.transpose(1, 2)

        lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        # Unsqueeze Channel Axis

        x = x.unsqueeze(1)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor == 1:
            # if subsampling_conv_chunking_factor is 1, we split only if needed
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
            if torch.numel(x) > x_ceil:
                need_to_split = True
            else:
                need_to_split = False
        else:
            # if subsampling_conv_chunking_factor > 1 we always split
            need_to_split = True

        if need_to_split:
            x, success = self.conv_split_by_batch(x)
            if not success:  # if unable to split by batch, try by channel
                if self._subsampling == 'dw_striding':
                    x = self.conv_split_by_channel(x)
                else:
                    x = self.conv(x)  # try anyway
        else:
            x = self.conv(x)


        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:

            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        return x, lengths

    def reset_parameters(self):
        # initialize weights
        if self._subsampling == 'dw_striding':
            with torch.no_grad():
                # init conv
                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size ** 2) ** -0.5
                pw_max = self._conv_channels ** -0.5

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
        """ Tries to split input by batch, run conv and concat results """
        b, _, _, _ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
            logging.debug(f'using manually set chunking factor: {cf}')
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2 ** p
            logging.debug(f'using auto set chunking factor: {cf}')

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, False

        logging.debug(f'conv subsampling: using split batch size {new_batch_size}')
        return torch.cat([self.conv(chunk) for chunk in torch.split(x, new_batch_size, 0)]), True

    def conv_split_by_channel(self, x):
        """ For dw convs, tries to split input by time, run conv and concat results """
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
                logging.debug(f'using manually set chunking factor: {cf}')
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2 ** 31, 2))
                cf = 2 ** p
                logging.debug(f'using auto set chunking factor: {cf}')

            new_c = int(c // cf)
            if new_c == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one channel.')
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one timestep.')
                new_t = 1

            logging.debug(f'conv dw subsampling: using split C size {new_c} and split T size {new_t}')
            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat([self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2)  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """ Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk, pad=(self._kernel_size - 1, self._stride - 1, self._kernel_size - 1, self._stride - 1)
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

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)

class ConformerLayer(torch.nn.Module):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos': absolute positional embedding and Transformer
            Default is rel_pos.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        use_bias (bool): Apply bias to all Linear and Conv1d layers from each ConformerLayer to improve activation flow and stabilize training of huge models.
            Defaults to True.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ):
        super(ConformerLayer, self).__init__()

        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
            )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'rel_pos_local_attn':
            x = self.self_attn(query=x, key=x, value=x, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, cache=cache_last_channel)
        else:
            x = None

        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x

        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)
        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
        use_bias (bool): Use bias in all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
            Defaults to True
    """

    def __init__(
        self,
        d_model,
        kernel_size,
        norm_type='batch_norm',
        conv_context_size=None,
        pointwise_activation='glu_',
        use_bias=True,
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, 'inplace'):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=self.use_bias,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type == 'fused_batch_norm':
            self.batch_norm = FusedBatchNorm1d(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model**-0.5
        dw_max = self.kernel_size**-0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            if self.use_bias:
                nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
                nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
                nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    use_bias (bool): Apply bias to all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish(), use_bias=True):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)


class ConformerEncoder(NeuralModule):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['vggnet', 'striding', 'dw-striding', 'stacking', 'stacking_norm']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_chunking_factor(int): optionally, force chunk inputs (helpful for large inputs)
            Should be power of 2, 1 (auto-chunking, default), or -1 (no chunking)
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
        reduction (str, Optional): the method of reduction, choices=['pooling', 'striding']. If no value
            is passed, then no reduction is performed and the models runs with the original 4x subsampling.
        reduction_position (int, Optional): the index of the layer to apply reduction. If -1, apply reduction
            at the end.
        reduction_factor (int): the reduction factor which should be either 1 or a power of 2
            Defaults to 1.
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): type of the attention layer and positional encoding

            'rel_pos':
                relative positional embedding and Transformer-XL

            'rel_pos_local_attn':
                relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.

            'abs_pos':
                absolute positional embedding and Transformer

            Default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaults to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        att_context_size (List[Union[List[int],int]]): specifies the context sizes on each side. Each context size should be a list of two integers like [100,100].
            A list of context sizes like [[100,100],[100,50]] can also be passed. -1 means unlimited context.
            Defaults to [-1,-1]
        att_context_probs (List[float]): a list of probabilities of each one of the att_context_size when a list of them is passed. If not specified, uniform distribution is being used.
            Defaults to None
        att_context_style (str): 'regular' or 'chunked_limited'.
            Defaults to 'regular'
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        conv_context_size (list): it can be"causal" or a list of two integers while conv_context_size[0]+conv_context_size[1]+1==conv_kernel_size.
            None means [(conv_kernel_size-1)//2, (conv_kernel_size-1)//2], and 'causal' means [(conv_kernel_size-1), 0].
            Defaults to None.
        conv_dual_mode (bool): specifies if convolution should be dual mode when dual_offline mode is being used. When enables, the left half of the convolution kernel would get masked in streaming cases.
            Defaults to False
        use_bias (bool): Use bias in all Linear and Conv1d layers from each ConformerLayer to improve activation flow and stabilize training of huge models.
            Defaults to True.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_pre_encoder (float): the dropout rate used before the encoder
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
        stochastic_depth_drop_prob (float): if non-zero, will randomly drop
            layers during training. The higher this value, the more often layers
            are dropped. Defaults to 0.0.
        stochastic_depth_mode (str): can be either "linear" or "uniform". If
            set to "uniform", all layers have the same probability of drop. If
            set to "linear", the drop probability grows linearly from 0 for the
            first layer to the desired value for the final layer. Defaults to
            "linear".
        stochastic_depth_start_layer (int): starting layer for stochastic depth.
            All layers before this will never be dropped. Note that drop
            probability will be adjusted accordingly if mode is "linear" when
            start layer is > 1. Defaults to 1.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        use_pytorch_sdpa (bool): use torch sdpa instead of manual attention.
            Defaults to False.
        use_pytorch_sdpa_backends (list[str]): list of backend names to use in sdpa. None or empty list means all backends. e.g. ["MATH"]
            Defaults to None
        sync_max_audio_length (bool): when true, performs NCCL all_reduce to allocate the same amount of memory for
            positional encoding buffers on all GPUs. Disabling this setting may help with deadlocks in certain
            scenarios such as model parallelism, or generally when this module is not being ran on some GPUs
            as a part of the training step.

    """

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length: bool = True,
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        self.self_attention_model = self_attention_model
        self.global_tokens = global_tokens
        self.global_attn_separate = global_attn_separate
        self.global_tokens_spacing = global_tokens_spacing
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.sync_max_audio_length = sync_max_audio_length

        # Setting up the att_context_size
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                # stacking_norm has an extra layer norm after stacking comparing to stacking
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                    activation=nn.ReLU(True),
                    is_causal=causal_downsampling,
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        # Reduction
        if reduction and reduction_factor > 1:
            assert reduction_position >= -1 and reduction_position < n_layers
            self.reduction_subsampling = SubsamplingReductionModule(
                reduction=reduction,
                d_model=d_model,
                reduction_factor=reduction_factor,
            )
            self.reduction_position = reduction_position
        else:
            self.reduction_subsampling = None
            self.reduction_position = None

        self._feat_out = d_model

        # Biases for relative positional encoding
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            if max(att_context_size) <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            self.pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=att_context_size,
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout_pre_encoder, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=self.conv_context_size,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True


        self.export_cache_support = False

        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers), stochastic_depth_drop_prob, stochastic_depth_mode, stochastic_depth_start_layer
        )
        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None


    @typecheck()
    def forward(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

    def forward_internal(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        audio_signal = torch.transpose(audio_signal, 1, 2)


        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        length = length.to(torch.int64)


        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )


        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:

            return audio_signal.transpose(1,2), length

    def update_max_seq_length(self, seq_length: int, device):
        # Find global max audio length across all nodes
        if self.sync_max_audio_length and torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)

            if self.att_context_style == "regular":
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
                if att_context_size[1] >= 0:
                    att_mask = att_mask.tril(diagonal=att_context_size[1])
            elif self.att_context_style == "chunked_limited":
                # When right context is unlimited, just the left side of the masking need to get updated
                if att_context_size[1] == -1:
                    if att_context_size[0] >= 0:
                        att_mask = att_mask.triu(diagonal=-att_context_size[0])
                else:
                    chunk_size = att_context_size[1] + 1
                    # left_chunks_num specifies the number of chunks to be visible by each chunk on the left side
                    if att_context_size[0] >= 0:
                        left_chunks_num = att_context_size[0] // chunk_size
                    else:
                        left_chunks_num = 10000

                    chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=att_mask.device)
                    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                    chunked_limited_mask = torch.logical_and(
                        torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                    )
                    att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
        else:
            att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def enable_pad_mask(self, on=True):
        # On inference, user may choose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask
    def _calc_context_sizes(
        self, att_context_size, att_context_probs, att_context_style, conv_context_size, conv_kernel_size
    ):
        # convert att_context_size to a standard list of lists
        if att_context_size:
            att_context_size_all = list(att_context_size)
            if isinstance(att_context_size_all[0], int):
                att_context_size_all = [att_context_size_all]
            for i, att_cs in enumerate(att_context_size_all):
                if isinstance(att_cs, ListConfig):
                    att_context_size_all[i] = list(att_cs)
                if att_context_style == "chunked_limited":
                    if att_cs[0] > 0 and att_cs[0] % (att_cs[1] + 1) > 0:
                        raise ValueError(f"att_context_size[{i}][0] % (att_context_size[{i}][1] + 1) should be zero!")
                    if att_cs[1] < 0 and len(att_context_size_all) <= 1:
                        raise ValueError(
                            f"Right context (att_context_size[{i}][1]) can not be unlimited for chunked_limited style!"
                        )
        else:
            att_context_size_all = [[-1, -1]]

        if att_context_probs:
            if len(att_context_probs) != len(att_context_size_all):
                raise ValueError("The size of the att_context_probs should be the same as att_context_size.")
            att_context_probs = list(att_context_probs)
            if sum(att_context_probs) != 1:
                raise ValueError(
                    "The sum of numbers in att_context_probs should be equal to one to be a distribution."
                )
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(att_context_size_all)

        if conv_context_size is not None:
            if isinstance(conv_context_size, ListConfig):
                conv_context_size = list(conv_context_size)
            if not isinstance(conv_context_size, list) and not isinstance(conv_context_size, str):
                raise ValueError(
                    f"Invalid conv_context_size! It should be the string 'causal' or a list of two integers."
                )
            if conv_context_size == "causal":
                conv_context_size = [conv_kernel_size - 1, 0]
            else:
                if conv_context_size[0] + conv_context_size[1] + 1 != conv_kernel_size:
                    raise ValueError(f"Invalid conv_context_size: {self.conv_context_size}!")
        else:
            conv_context_size = [(conv_kernel_size - 1) // 2, (conv_kernel_size - 1) // 2]
        return att_context_size_all, att_context_size_all[0], att_context_probs, conv_context_size




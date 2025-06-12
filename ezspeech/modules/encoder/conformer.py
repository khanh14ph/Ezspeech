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

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import torch
from torch import nn as nn
from torch.nn import LayerNorm
import torch
import torch.distributed
from torch import nn
from ezspeech.layers.positional_encoding import RelPositionalEncoding
from torch.nn import SiLU
from ezspeech.layers.sampling import ConvSubsampling
from ezspeech.utils.asr.regularization_utils import compute_stochastic_depth_drop_probs
from ezspeech.layers.attention import RelPositionMultiHeadAttention
from ezspeech.layers.causal_convs import CausalConv1D


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    use_bias (bool): Apply bias to all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
    """

    def __init__(self, d_model, d_ff, dropout, activation=SiLU(), use_bias=True):
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
        conv_context_size=None,
        use_bias=True,
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size

        self.use_bias = use_bias

        conv_context_size = (kernel_size - 1) // 2

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
        self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)

        self.activation = SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = self.pointwise_conv1(x)

        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)

        x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)

        return x

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
        self_attention_model="rel_pos",
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type="batch_norm",
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
        self.feed_forward1 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias
        )

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]
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

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(
        self,
        x,
        att_mask=None,
        pos_emb=None,
        pad_mask=None,
        cache_last_channel=None,
        cache_last_time=None,
    ):
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

        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=att_mask,
            pos_emb=pos_emb,
            cache=cache_last_channel,
        )

        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = x.transpose(1, 2)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)

        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        return x


class ConformerEncoder(nn.Module):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        feat_out
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        subsampling (str): the method of subsampling:
            choices = ['vggnet', 'striding', 'dw-striding', 'stacking', 'stacking_norm']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_chunking_factor(int): optionally, force chunk inputs (helpful for large inputs)
            Should be power of 2, 1 (auto-chunking, default), or -1 (no chunking)
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.

        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.

        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaults to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        att_context_size (List[Union[List[int],int]]): specifies the context sizes on each side.
            Each context size should be a list of two integers like `[100, 100]`.
            A list of context sizes like `[[100,100]`, `[100,50]]` can also be passed. -1 means unlimited context.
            Defaults to `[-1, -1]`
        att_context_probs (List[float]): a list of probabilities of each one of the att_context_size
            when a list of them is passed. If not specified, uniform distribution is being used.
            Defaults to None

        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        conv_context_size (list): it can be"causal" or a list of two integers
            while `conv_context_size[0]+conv_context_size[1]+1==conv_kernel_size`.
            `None` means `[(conv_kernel_size-1)//2`, `(conv_kernel_size-1)//2]`, and 'causal' means
            `[(conv_kernel_size-1), 0]`.
            Defaults to None.
        use_bias (bool): Use bias in all Linear and Conv1d layers from each ConformerLayer to improve
            activation flow and stabilize training of huge models.
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
            start layer is > 1. Defaults to 1
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        use_pytorch_sdpa (bool): use torch sdpa instead of manual attention.
            Defaults to False.
        use_pytorch_sdpa_backends (list[str]): list of backend names to use in sdpa.
            None or empty list means all backends. e.g. ["MATH"]
            Defaults to None.
        sync_max_audio_length (bool): when true, performs NCCL all_reduce to allocate the same amount of memory for
            positional encoding buffers on all GPUs. Disabling this setting may help with deadlocks in certain
            scenarios such as model parallelism, or generally when this module is not being ran on some GPUs
            as a part of the training step.
    """

    def __init__(
        self,
        feat_in,
        feat_out=-1,
        n_layers=12,
        d_model=512,
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        ff_expansion_factor=4,
        self_attention_model="rel_pos",
        n_heads=4,
        att_context_size=None,
        att_context_style="regular",
        pos_emb_max_len=5000,
        xscaling=False,
        untie_biases=True,
        conv_kernel_size=31,
        conv_norm_type="batch_norm",
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length: bool = True,
        **kwargs
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self._feat_out = feat_out
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        self.self_attention_model = self_attention_model

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
            att_context_size=att_context_size,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )
        if feat_out != -1:
            self.proj_out = nn.Linear(d_model, feat_out)
        self.pre_encode = ConvSubsampling(
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
            activation=nn.ReLU(True),
        )

        pos_bias_u = None
        pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout_pre_encoder,
            max_len=pos_emb_max_len,
            xscale=None,
            dropout_rate_emb=dropout_emb,
        )

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
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

        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers),
            stochastic_depth_drop_prob,
            stochastic_depth_mode,
            stochastic_depth_start_layer,
        )
        # will be set in self.forward() if defined in AccessMixin config

    def forward(self, audio_signal, length):

        self.update_max_seq_length(
            seq_length=audio_signal.size(2) * self.subsampling_factor,
            device=audio_signal.device,
        )

        return self.forward_internal(audio_signal, length)

    def forward_internal(self, audio_signal, length):
        # audio_signal=audio_signal.transpose(1,2)
        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(
                self.att_context_size_all, weights=self.att_context_probs
            )[0]
        else:
            cur_att_context_size = self.att_context_size

        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        length = length.to(torch.int64)
        max_audio_length = audio_signal.size(1)
        padding_length = length
        offset = None
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        for lth, (drop_prob, layer) in enumerate(
            zip(self.layer_drop_probs, self.layers)
        ):
            original_signal = audio_signal

            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )

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
                    audio_signal = (audio_signal - original_signal) / (
                        1.0 - drop_prob
                    ) + original_signal

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)
        audio_signal = audio_signal.transpose(1, 2)
        if self._feat_out != -1:
            audio_signal = self.proj_out(audio_signal)
        return audio_signal, length

    def update_max_seq_length(self, seq_length: int, device):
        """
        Updates the maximum sequence length for the model.

        Args:
            seq_length (int): New maximum sequence length.
            device (torch.device): Device to use for computations.
        """
        # Find global max audio length across all nodes
        if self.sync_max_audio_length and torch.distributed.is_initialized():
            global_max_len = torch.tensor(
                [seq_length], dtype=torch.float32, device=device
            )

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(
                global_max_len, op=torch.distributed.ReduceOp.MAX
            )

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def _create_masks(
        self, att_context_size, padding_length, max_audio_length, offset, device
    ):
        att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def _calc_context_sizes(
        self,
        att_context_size,
        conv_context_size,
        conv_kernel_size,
    ):

        # att_context_size_all = [[-1, -1]]
        att_context_size_all = [att_context_size]

        att_context_probs = [1.0 / len(att_context_size_all)] * len(
            att_context_size_all
        )

        conv_context_size = [
            (conv_kernel_size - 1) // 2,
            (conv_kernel_size - 1) // 2,
        ]
        return (
            att_context_size_all,
            att_context_size_all[0],
            att_context_probs,
            conv_context_size,
        )

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.

        Args:
            max_audio_length (int): New maximum sequence length.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

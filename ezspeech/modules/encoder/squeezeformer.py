import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ezspeech.layers.sampling import ConvolutionSubsampling

from ezspeech.layers.block import (
    AttentionBlock,
    FeedForwardBlock,
    ConvolutionBlock,
)

from ezspeech.utils.common import (
    make_padding_mask,
    time_reduction,
)

# from ezspeech.modules.conformer import SumaryMixing_ConformerLayer

class SqueezeformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        attn_num_heads: int,
        attn_group_size: int,
        attn_max_pos_encoding: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.attn = AttentionBlock(
            d_model=d_model,
            num_heads=attn_num_heads,
            group_size=attn_group_size,
            max_pos_encoding=attn_max_pos_encoding,
            dropout=dropout,
        )
        self.norm_attn = nn.LayerNorm(d_model)

        self.ffn1 = FeedForwardBlock(
            d_model=d_model,
            dropout=dropout,
        )
        self.norm_ffn1 = nn.LayerNorm(d_model)

        self.conv = ConvolutionBlock(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.norm_conv = nn.LayerNorm(d_model)

        self.ffn2 = FeedForwardBlock(
            d_model=d_model,
            dropout=dropout,
        )
        self.norm_ffn2 = nn.LayerNorm(d_model)

    def forward(
        self,
        xs: torch.Tensor,
        attn_masks: torch.Tensor,
        conv_masks: torch.Tensor,
    ) -> torch.Tensor:

        residual = xs.clone()
        xs = self.attn(xs, attn_masks)
        xs = xs + residual
        xs = self.norm_attn(xs)

        residual = xs.clone()
        xs = self.ffn1(xs)
        xs = xs + residual
        xs = self.norm_ffn1(xs)

        residual = xs.clone()
        xs = self.conv(xs, conv_masks)
        xs = xs + residual
        xs = self.norm_conv(xs)

        residual = xs.clone()
        xs = self.ffn2(xs)
        xs = xs + residual
        xs = self.norm_ffn2(xs)

        return xs

class SqueezeFormerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        subsampling_num_filters: int,
        subsampling_kernel_size: int,
        attn_num_heads: int,
        attn_group_size: int,
        attn_max_pos_encoding: int,
        conv_kernel_size: int,
        dropout: float,
        subsampling_factor: int = 4,
    ):
        super(SqueezeFormerEncoder, self).__init__()

        self.subsampling = ConvolutionSubsampling(
            input_dim=input_dim,
            output_dim=d_model,
            factor=subsampling_factor,
            num_filters=subsampling_num_filters,
            kernel_size=subsampling_kernel_size,
            dropout=dropout,
        )

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            encoder_layer = SqueezeformerBlock(
                d_model=d_model,
                attn_num_heads=attn_num_heads,
                attn_group_size=attn_group_size,
                attn_max_pos_encoding=attn_max_pos_encoding,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            self.encoder_layers.append(encoder_layer)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, x_lens = self.subsampling(xs, x_lens)

        __, max_time, __ = xs.size()
        masks = make_padding_mask(x_lens, max_time)
        attn_masks = masks.unsqueeze(1).repeat([1, max_time, 1])
        attn_masks = attn_masks & attn_masks.transpose(1, 2)
        attn_masks = ~attn_masks
        conv_masks = ~masks

        for layer in self.encoder_layers:
            xs = layer(xs, attn_masks, conv_masks)
        return xs, x_lens

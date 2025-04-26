import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Emformer
from torchaudio.transforms import MelSpectrogram, PitchShift

from lightspeech.layers.sampling import ConvolutionSubsampling

from lightspeech.layers.block import (
    SqueezeformerBlock,
    ConformerBlock,
    _lengths_to_padding_mask,
)

from lightspeech.utils.common import (
    make_padding_mask,
    time_reduction,
)
from transformers import AutoModel

# from lightspeech.modules.conformer import SumaryMixing_ConformerLayer


class AcousticEncoder(nn.Module):
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
        super(AcousticEncoder, self).__init__()

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

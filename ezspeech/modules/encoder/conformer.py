import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from lightspeech.layers.sampling import ConvolutionSubsampling

from lightspeech.layers.block import _lengths_to_padding_mask,


class ConformerBlock(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(
            input_dim, num_attention_heads, dropout=dropout
        )
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(
        self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x

class ConformerEncoder(nn.Module):
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
        super(ConformerEncoder, self).__init__()

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
            encoder_layer = ConformerBlock(
                input_dim=d_model,
                ffn_dim=4 * d_model,
                num_attention_heads=attn_num_heads,
                depthwise_conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            self.encoder_layers.append(encoder_layer)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, x_lens = self.subsampling(xs, x_lens)

        encoder_padding_mask = _lengths_to_padding_mask(x_lens)

        x = xs.transpose(0, 1)

        for layer in self.encoder_layers:
            x = layer(x, encoder_padding_mask)

        return x.transpose(0, 1), x_lens
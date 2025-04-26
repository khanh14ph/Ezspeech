import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from lightspeech.layers.attention import MultiHeadSelfAttention
from lightspeech.layers.normalization import ScaleBiasNorm


class _ConvolutionModule(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError(
                "depthwise_kernel_size must be odd to achieve 'SAME' padding."
            )
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            (
                torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
                if use_group_norm
                else torch.nn.BatchNorm1d(num_channels)
            ),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential(input)


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(
        max_length, device=lengths.device, dtype=lengths.dtype
    ).expand(batch_size, max_length) >= lengths.unsqueeze(1)
    return padding_mask


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


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = ScaleBiasNorm(d_model)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.pre_norm(xs)

        xs = self.linear1(xs)
        xs = self.activation(xs)
        xs = self.dropout(xs)

        xs = self.linear2(xs)
        xs = self.dropout(xs)

        return xs


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        group_size: int,
        max_pos_encoding: int,
        dropout: float,
    ):
        super(AttentionBlock, self).__init__()
        self.mhsa = MultiHeadSelfAttention(
            d_model,
            num_heads,
            group_size,
            max_pos_encoding,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = ScaleBiasNorm(d_model)

    def forward(self, xs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        xs = self.pre_norm(xs)

        xs = self.mhsa(xs, xs, xs, masks)
        xs = self.dropout(xs)

        return xs


class ConvolutionBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super(ConvolutionBlock, self).__init__()
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = ScaleBiasNorm(d_model)

    def forward(self, xs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        xs = self.pre_norm(xs)

        xs = xs.transpose(1, 2)
        xs = self.pointwise_conv1(xs)
        xs = F.silu(xs)

        masks = masks.unsqueeze(1)
        xs = xs.masked_fill(masks, 0.0)

        xs = self.depthwise_conv(xs)
        xs = self.norm(xs)
        xs = F.silu(xs)

        xs = self.pointwise_conv2(xs)
        xs = xs.transpose(1, 2)
        xs = self.dropout(xs)

        return xs

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ezspeech.layers.attention import MultiHeadSelfAttention
from ezspeech.layers.normalization import ScaleBiasNorm


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

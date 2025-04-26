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


class AcousticConformerEncoder(nn.Module):
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
        super(AcousticConformerEncoder, self).__init__()

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




class AcousticMixingEncoder(nn.Module):
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
        super(AcousticMixingEncoder, self).__init__()

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
            encoder_layer = SumaryMixing_ConformerLayer(
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

        __, max_time, __ = xs.size()
        masks = make_padding_mask(x_lens, max_time)
        xs = xs.transpose(0, 1)
        for layer in self.encoder_layers:
            xs = layer(xs, masks)
        xs = xs.transpose(0, 1)
        return xs, x_lens


class StreamingAcousticEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        segment_length: int,
        left_context_length: int,
        right_context_length: int,
        ffn_dim: int,
        num_layers: int,
        subsampling_factor: int = 4,
        num_heads: Optional[int] = 8,
        dropout: Optional[float] = 0.1,
        activation: Optional[str] = "gelu",
        max_memory_size: Optional[int] = 4,
        weight_init_scale_strategy: Optional[str] = "depthwise",
        tanh_on_mem: Optional[bool] = True,
    ):
        super(StreamingAcousticEncoder, self).__init__()

        self.stride = subsampling_factor
        self.right_padding = right_context_length

        assert (
            d_model % self.stride == 0
        ), "The model dimension must be divisible by the stride."

        self.input_linear = nn.Linear(
            in_features=input_dim,
            out_features=d_model // self.stride,
            bias=False,
        )

        self.encoder_layers = Emformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            segment_length=segment_length // self.stride,
            dropout=dropout,
            activation=activation,
            left_context_length=left_context_length // self.stride,
            right_context_length=right_context_length // self.stride,
            max_memory_size=max_memory_size,
            weight_init_scale_strategy=weight_init_scale_strategy,
            tanh_on_mem=tanh_on_mem,
        )

    def forward(
        self,
        xs: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training and non-streaming inference."""

        xs = F.pad(xs, (0, 0, 0, self.right_padding))
        xs = self.input_linear(xs)

        xs, x_lens = time_reduction(xs, x_lens, self.stride)
        xs, x_lens = self.encoder_layers(xs, x_lens)

        return xs, x_lens

    def infer(
        self,
        xs: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """Forward pass for streaming inference."""

        xs = self.input_linear(xs)

        xs, x_lens = time_reduction(xs, x_lens, self.stride)
        xs, x_lens, states = self.encoder_layers.infer(xs, x_lens, states)

        return xs, x_lens, states


class LinguisticEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        attn_num_heads: int,
        attn_group_size: int,
        attn_max_pos_encoding: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super(LinguisticEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                SqueezeformerBlock(
                    d_model=d_model,
                    attn_num_heads=attn_num_heads,
                    attn_group_size=attn_group_size,
                    attn_max_pos_encoding=attn_max_pos_encoding,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for __ in range(num_layers)
            ]
        )

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self.embedding(xs)

        __, max_time, __ = xs.size()
        masks = make_padding_mask(x_lens, max_time)
        attn_masks = masks.unsqueeze(1).repeat([1, max_time, 1])
        attn_masks = attn_masks & attn_masks.transpose(1, 2)
        attn_masks = ~attn_masks
        conv_masks = ~masks

        for layer in self.layers:
            xs = layer(xs, attn_masks, conv_masks)

        return xs, x_lens


class SpectrogramExtractor(nn.Module):
    def __init__(self, sample_rate: int, n_steps: List[int]):
        super().__init__()
        self.pitch_shift = [PitchShift(sample_rate, step) for step in n_steps]
        self.spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(0.05 * sample_rate),
            win_length=int(0.025 * sample_rate),
            hop_length=int(0.01 * sample_rate),
            n_mels=128,
        )

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor):
        pitch_shift = random.choice(self.pitch_shift).to(xs.device)
        xs = pitch_shift(xs).squeeze(1)

        ys = self.spectrogram(xs).clamp(1e-5).log().transpose(1, 2)
        y_lens = (x_lens * ys.size(1) / xs.size(1) - 1).long()

        masks = make_padding_mask(y_lens, ys.size(1))
        ys = ys.masked_fill(~masks[:, :, None], 0.0)

        return ys, y_lens


class TextEncoder(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_path).train()
        self.text_layer_norm = torch.nn.LayerNorm(768)

    def forward(self, input_features, attention_mask):
        text_context_vector = self.bert(input_features, attention_mask)[
            "last_hidden_state"
        ]
        text_context_vector = self.text_layer_norm(text_context_vector)
        return text_context_vector



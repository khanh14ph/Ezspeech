import logging
from typing import List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class MultiConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Multi Convolutional Spatial Gating Unit (M-CSGU)."""

    def __init__(
        self,
        size: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = LayerNorm(n_channels)

        kernel_sizes = list(map(int, kernel_sizes.split(",")))
        no_kernels = len(kernel_sizes)

        assert (
            n_channels % no_kernels == 0
        ), f"{n_channels} input channels cannot be divided between {no_kernels} kernels"

        self.arch_type = arch_type
        if arch_type in ["sum", "weighted_sum"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        elif arch_type in ["concat", "concat_fusion"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels // no_kernels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels // no_kernels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        else:
            raise NotImplementedError(
                f"Unknown architecture type for MultiConvCGMLP: {arch_type}"
            )
        self.use_non_linear = use_non_linear
        if arch_type == "weighted_sum":
            self.kernel_prob_gen = torch.nn.Sequential(
                torch.nn.Linear(n_channels * no_kernels, no_kernels),
                torch.nn.Softmax(dim=-1),
            )
            self.depthwise_conv_fusion = None
        elif arch_type == "concat_fusion":
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = torch.nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=merge_conv_kernel,
                stride=1,
                padding=(merge_conv_kernel - 1) // 2,
                groups=n_channels,
                bias=True,
            )
        else:
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = None

        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        self.model_act = activation
        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = get_activation(gate_activation)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, std=1e-6)
            torch.nn.init.ones_(conv.bias)
        if self.depthwise_conv_fusion is not None:
            torch.nn.init.normal_(self.depthwise_conv_fusion.weight, std=1e-6)
            torch.nn.init.ones_(self.depthwise_conv_fusion.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """
        x_r, x_i = x.chunk(2, dim=-1)

        x_i = self.norm(x_i).transpose(1, 2)  # (N, D/2, T)

        # TODO: Parallelize this convolution computation
        xs = []
        for conv in self.convs:
            xi = conv(x_i).transpose(1, 2)  # (N, T, D/2)
            if self.arch_type == "sum" and self.use_non_linear:
                xi = self.model_act(xi)
            xs.append(xi)

        if self.arch_type in ["sum", "weighted_sum"]:
            x = torch.stack(xs, dim=-2)
            if self.arch_type == "weighted_sum":
                prob = self.kernel_prob_gen(torch.cat(xs, dim=-1))
                x = prob.unsqueeze(-1) * x

            x_g = x.sum(dim=-2)
        else:
            x_concat = torch.cat(xs, dim=-1)  # (N, T, D)

            if self.arch_type == "concat_fusion":
                x_tmp = x_concat.transpose(1, 2)
                x_tmp = self.depthwise_conv_fusion(x_tmp)
                x_concat = x_concat + x_tmp.transpose(1, 2)

            x_g = x_concat

        if self.linear is not None:
            x_g = self.linear(x_g)

        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class MultiConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        if arch_type not in ["sum", "weighted_sum", "concat", "concat_fusion"]:
            raise NotImplementedError(f"Unknown MultiConvCGMLP type: {type}")

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = MultiConvolutionalSpatialGatingUnit(
            size=linear_units,
            arch_type=arch_type,
            kernel_sizes=kernel_sizes,
            merge_conv_kernel=merge_conv_kernel,
            use_non_linear=use_non_linear,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            activation=activation,
            gate_activation=gate_activation,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None

        xs_pad = self.channel_proj1(xs_pad)  # size -> linear_units
        xs_pad = self.csgu(xs_pad)  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)  # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out


class MultiConvConformerEncoder(AbsEncoder):
    """Multiconvformer encoder module.
    Link to the paper: https://arxiv.org/abs/2407.03718

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        cgmlp_linear_units (int): The number of units used in CGMLP block.
        multicgmlp_type (str): "sum", "weighted_sum", "concat" or "concat_fusion".
        multicgmlp_kernel_sizes (str): Comma seperated list of kernel sizes.
        multicgmlp_merge_conv_kernel (int): The number of kernels used in depthwise
            convolution fusion in MultiCGMLP.
        use_linear_after_conv (bool): Whether to use a linear layer after MultiCGMLP.
        gate_activation (str): The activation function used in CGMLP gating.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        cgmlp_linear_units: int = 2048,
        multicgmlp_type: str = "concat_fusion",
        multicgmlp_kernel_sizes: Union[int, str] = "7,15,23,31",
        multicgmlp_merge_conv_kernel: int = 31,
        multicgmlp_use_non_linear: int = True,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        input_layer: str = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
    ):
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        if isinstance(multicgmlp_kernel_sizes, int):
            multicgmlp_kernel_sizes = str(multicgmlp_kernel_sizes)

        convolution_layer = MultiConvolutionalGatingMLP
        convolution_layer_args = (
            output_size,
            cgmlp_linear_units,
            multicgmlp_type,
            multicgmlp_kernel_sizes,
            multicgmlp_merge_conv_kernel,
            multicgmlp_use_non_linear,
            dropout_rate,
            use_linear_after_conv,
            activation,
            gate_activation,
        )

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate[lnum],
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens

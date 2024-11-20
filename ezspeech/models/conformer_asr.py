from typing import Optional, Tuple
import torch
from ezspeech.modules.convolution import ConvolutionSubSampling
from ezspeech.modules.conformer import ConformerLayer, _lengths_to_padding_mask


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        num_heads: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        subsampling_num_filter: int = 256,
        subsampling_kernel_size: int = 3,
        vocab_size: int = 98,
    ):
        super().__init__()
        self.conv_subsample = ConvolutionSubSampling(
            d_input,
            d_hidden,
            subsampling_factor=4,
            num_filter=subsampling_num_filter,
            kernel_size=subsampling_kernel_size,
            dropout=dropout,
        )
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    d_hidden,
                    4 * d_hidden,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = torch.nn.Linear(d_hidden, vocab_size)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.

        """

        x, lengths = self.conv_subsample(x, lengths)

        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        x = x.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        x = self.lm_head(x.transpose(0, 1))
        x = x.log_softmax(2)

        return x, lengths

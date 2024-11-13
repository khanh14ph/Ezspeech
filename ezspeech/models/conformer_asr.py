from typing import Optional, Tuple
import torch
from ezspeech.modules.layer import Conv2dSubSampling
from ezspeech.modules.conformer import ConformerLayer,_lengths_to_padding_mask




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
        d_dim: int,
        num_heads: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        vocab_size: int = 98
    ):
        super().__init__()
        self.conv_subsample = Conv2dSubSampling(in_channels=1, out_channels=d_dim)

        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(d_dim * (((d_input - 1) // 2 - 1) // 2), d_dim),
            # torch.nn.Linear(d_dim * ((d_input - 1) // 2), d_dim),
            torch.nn.Dropout(p=dropout),
        )
        # self.input_projection = torch.nn.Linear(d_input,d_dim)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    d_dim,
                    4*d_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head=torch.nn.Linear(d_dim,vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            # print("length before", lengths)
            x, lengths = self.conv_subsample(x, lengths)
            x = self.input_projection(x)
            # print("length",lengths)
            encoder_padding_mask = _lengths_to_padding_mask(lengths)
            x = x.transpose(0, 1)
            for layer in self.conformer_layers:
                x = layer(x, encoder_padding_mask)
            x=self.lm_head(x.transpose(0,1))
            x=x.log_softmax(2)

            return x, lengths


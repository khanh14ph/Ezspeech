
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def rnn(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    norm: Optional[str] = None,
    forget_gate_bias: Optional[float] = 1.0,
    dropout: Optional[float] = 0.0,
    norm_first_rnn: Optional[bool] = None,
    t_max: Optional[int] = None,
    weights_init_scale: float = 1.0,
    hidden_hidden_bias_scale: float = 0.0,
    proj_size: int = 0,
) -> torch.nn.Module:
    """
    Utility function to provide unified interface to common LSTM RNN modules.

    Args:
        input_size: Input dimension.

        hidden_size: Hidden dimension of the RNN.

        num_layers: Number of RNN layers.

        norm: Optional string representing type of normalization to apply to the RNN.
            Supported values are None, batch and layer.

        forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

        dropout: Optional dropout to apply to end of multi-layered RNN.

        norm_first_rnn: Whether to normalize the first RNN layer.

        t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
            of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
            of training.
            Reference:
            [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

        weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
            sometimes helps reduce variance between runs.

        hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
            the default behaviour.

    Returns:
        A RNN module
    """
    if norm not in [None, "batch", "layer"]:
        raise ValueError(f"unknown norm={norm}")

    if norm is None:
        return LSTMDropout(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            weights_init_scale=weights_init_scale,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            proj_size=proj_size,
        )

    if norm == "batch":
        return BNRNNSum(
            input_size=input_size,
            hidden_size=hidden_size,
            rnn_layers=num_layers,
            batch_norm=True,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            norm_first_rnn=norm_first_rnn,
            weights_init_scale=weights_init_scale,
            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            proj_size=proj_size,
        )

    if norm == "layer":
        return torch.jit.script(
            ln_lstm(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                forget_gate_bias=forget_gate_bias,
                t_max=t_max,
                weights_init_scale=weights_init_scale,
                hidden_hidden_bias_scale=hidden_hidden_bias_scale,
            )
        )
def label_collate(labels, device=None):
    """Collates the label inputs for the rnn-t prediction network.
    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.
        device: Optional torch device to place the label on.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    max_len = max(len(label) for label in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)

    return labels
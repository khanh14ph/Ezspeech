
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

class LSTMDropout(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: Optional[float],
        forget_gate_bias: Optional[float],
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
        proj_size: int = 0,
    ):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.
        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            dropout: See `torch.nn.LSTM`.

            forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

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
            A `torch.nn.LSTM`.
        """
        super(LSTMDropout, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, proj_size=proj_size
        )

        if t_max is not None:
            # apply chrono init
            for name, v in self.lstm.named_parameters():
                if 'bias' in name:
                    p = getattr(self.lstm, name)
                    n = p.nelement()
                    hidden_size = n // 4
                    p.data.fill_(0)
                    p.data[hidden_size : 2 * hidden_size] = torch.log(
                        torch.nn.init.uniform_(p.data[0:hidden_size], 1, t_max - 1)
                    )
                    # forget gate biases = log(uniform(1, Tmax-1))
                    p.data[0:hidden_size] = -p.data[hidden_size : 2 * hidden_size]
                    # input gate biases = -(forget gate biases)

        elif forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size] *= float(hidden_hidden_bias_scale)

        self.dropout = torch.nn.Dropout(dropout) if dropout else None

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(weights_init_scale)

    def forward(
        self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, h = self.lstm(x, h)

        if self.dropout:
            x = self.dropout(x)

        return x, h
class BNRNNSum(torch.nn.Module):
    """RNN wrapper with optional batch norm.
    Instantiates an RNN. If it is an LSTM it initialises the forget gate
    bias =`lstm_gate_bias`. Optionally applies a batch normalisation layer to
    the input with the statistics computed over all time steps.  If dropout > 0
    then it is applied to all layer outputs except the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: torch.nn.Module = torch.nn.LSTM,
        rnn_layers: int = 1,
        batch_norm: bool = True,
        dropout: Optional[float] = 0.0,
        forget_gate_bias: Optional[float] = 1.0,
        norm_first_rnn: bool = False,
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
        proj_size: int = 0,
    ):
        super().__init__()
        self.rnn_layers = rnn_layers

        self.layers = torch.nn.ModuleList()
        for i in range(rnn_layers):
            final_layer = (rnn_layers - 1) == i

            self.layers.append(
                RNNLayer(
                    input_size,
                    hidden_size,
                    rnn_type=rnn_type,
                    batch_norm=batch_norm and (norm_first_rnn or i > 0),
                    forget_gate_bias=forget_gate_bias,
                    t_max=t_max,
                    weights_init_scale=weights_init_scale,
                    hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                    proj_size=proj_size,
                )
            )

            if dropout is not None and dropout > 0.0 and not final_layer:
                self.layers.append(torch.nn.Dropout(dropout))

            input_size = hidden_size

    def forward(
        self, x: torch.Tensor, hx: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx = self._parse_hidden_state(hx)

        hs = []
        cs = []
        rnn_idx = 0
        for layer in self.layers:
            if isinstance(layer, torch.nn.Dropout):
                x = layer(x)
            else:
                x, h_out = layer(x, hx=hx[rnn_idx])
                hs.append(h_out[0])
                cs.append(h_out[1])
                rnn_idx += 1
                del h_out

        h_0 = torch.stack(hs, dim=0)
        c_0 = torch.stack(cs, dim=0)
        return x, (h_0, c_0)

    def _parse_hidden_state(
        self, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[List[None], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Dealing w. hidden state:
        Typically in pytorch: (h_0, c_0)
            h_0 = ``[num_layers * num_directions, batch, hidden_size]``
            c_0 = ``[num_layers * num_directions, batch, hidden_size]``
        """
        if hx is None:
            return [None] * self.rnn_layers
        else:
            h_0, c_0 = hx

            if h_0.shape[0] != self.rnn_layers:
                raise ValueError(
                    'Provided initial state value `h_0` must be of shape : '
                    '[num_layers * num_directions, batch, hidden_size]'
                )

            return [(h_0[i], c_0[i]) for i in range(h_0.shape[0])]

    def _flatten_parameters(self):
        for layer in self.layers:
            if isinstance(layer, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
                layer._flatten_parameters()

def ln_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: Optional[float],
    forget_gate_bias: Optional[float],
    t_max: Optional[int],
    weights_init_scale: Optional[float] = None,  # ignored
    hidden_hidden_bias_scale: Optional[float] = None,  # ignored
) -> torch.nn.Module:
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""
    # The following are not implemented.
    if dropout is not None and dropout != 0.0:
        raise ValueError('`dropout` not supported with LayerNormLSTM')

    if t_max is not None:
        logging.warning("LayerNormLSTM does not support chrono init via `t_max`")

    if weights_init_scale is not None:
        logging.warning("`weights_init_scale` is ignored for LayerNormLSTM")

    if hidden_hidden_bias_scale is not None:
        logging.warning("`hidden_hidden_bias_scale` is ignored for LayerNormLSTM")

    return StackedLSTM(
        num_layers,
        LSTMLayer,
        first_layer_args=[LayerNormLSTMCell, input_size, hidden_size, forget_gate_bias],
        other_layer_args=[LayerNormLSTMCell, hidden_size, hidden_size, forget_gate_bias],
    )


class LSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

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
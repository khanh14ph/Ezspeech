from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ezspeech.utils.operation import init_weights


class PredictorNetwork(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        d_model: int,
        dropout: float,
    ):
        super(PredictorNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn_layer = nn.GRU(embedding_dim, d_model, batch_first=True)
        self.rnn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        token_idxs: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        embs = self.embedding(token_idxs)

        outputs, states = self.rnn_layer(embs, states)
        outputs = self.rnn_norm(outputs)
        outputs = self.dropout(outputs)

        return outputs, states


class JointNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(JointNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        enc_outs: torch.Tensor,
        pred_outs: torch.Tensor,
    ) -> torch.Tensor:
        # print("pred_outs",pred_outs)
        enc_outs = enc_outs.unsqueeze(2).contiguous()
        pred_outs = pred_outs.unsqueeze(1).contiguous()

        joint_outs = F.silu(enc_outs + pred_outs)
        outputs = self.linear(joint_outs)

        return outputs


class CTCDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(CTCDecoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim + 1)

    def forward(self, enc_outs: torch.Tensor) -> torch.Tensor:
        ctc_outs = F.silu(self.linear1(enc_outs))
        ctc_outs = self.linear2(ctc_outs)
        ctc_outs = ctc_outs.log_softmax(2)
        return ctc_outs


class ConvASRDecoder(nn.Module):
    """Simple ASR Decoder for use with CTC-based models such as JasperNet and QuartzNet

    Based on these papers:
       https://arxiv.org/pdf/1904.03288.pdf
       https://arxiv.org/pdf/1910.10261.pdf
       https://arxiv.org/pdf/2005.04290.pdf
    """

    def __init__(
        self,
        feat_in,
        num_classes,
        vocabulary=None,
        init_mode="xavier_uniform",
        add_blank=True,
    ):
        super().__init__()

        if vocabulary is None and num_classes < 0:
            raise ValueError(
                "Neither of the vocabulary and num_classes are set! At least one of them need to be set."
            )

        if num_classes <= 0:
            num_classes = len(vocabulary)
            print(
                f"num_classes of ConvASRDecoder is set to the size of the vocabulary: {num_classes}."
            )

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. \
                        Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1 if add_blank else num_classes

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        # Adapter module forward step
        encoder_output = encoder_output.transpose(1, 2)
        return torch.nn.functional.log_softmax(
            self.decoder_layers(encoder_output).transpose(1, 2), dim=-1
        )

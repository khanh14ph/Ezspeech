from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, enc_outs: torch.Tensor) -> torch.Tensor:
        ctc_outs = F.silu(self.linear1(enc_outs))
        ctc_outs = self.linear2(ctc_outs)
        ctc_outs = ctc_outs.log_softmax(2)
        return ctc_outs



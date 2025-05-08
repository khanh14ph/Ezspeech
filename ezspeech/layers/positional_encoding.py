import math
from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.attention
import torch.nn.functional as F

from ezspeech.utils.common import avoid_float16_autocast_context
INF_VAL = 10000.0
class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, 'pe') and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x: torch.Tensor, cache_len=0):
        """Adds positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        input_len = x.size(1) + cache_len
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb

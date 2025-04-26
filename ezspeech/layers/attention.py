import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    """
    Relative Sinusoidal Positional Encoding for grouped multi-head attention
    Positional encoding for left context (sin) and right context (cos)
    Total context = 2 * max_len - group_size
    """

    def __init__(
        self,
        max_len: int,
        d_model: int,
        group_size: int,
    ):
        super(PositionalEncoding, self).__init__()

        self.max_len = max_len
        self.group_size = group_size

        # PE
        pos_encoding = torch.zeros(2 * max_len - group_size % 2, d_model)

        # Positions (max_len - 1, ..., max_len - 1)
        pos_left = torch.arange(
            start=max_len - 1,
            end=group_size % 2 - 1,
            step=-1,
            dtype=torch.float,
        )
        pos_right = torch.arange(
            start=0,
            end=-max_len,
            step=-1,
            dtype=torch.float,
        )
        pos = torch.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        steps = torch.arange(0, d_model // 2, dtype=torch.float).unsqueeze(0)
        angles = pos / 10000 ** (2 * steps / d_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)

    def forward(self, batch_size: int, seq_len: int) -> torch.Tensor:
        # (B, Th + 2*T-G, D)
        left_context = self.max_len - seq_len + self.group_size // 2
        right_context = (
            self.max_len - self.group_size % 2 + seq_len - self.group_size // 2
        )
        R = self.pos_encoding[:, left_context:right_context]
        return R.repeat(batch_size, 1, 1)


class MultiHeadSelfAttention(nn.Module):

    """Grouped Multi-Head Self-Attention Layer
        with Relative Sinusoidal Positional Encodings
    Args:
        d_model: model feature dimension
        num_heads: number of attention heads
        group_size: attention group size
        max_pos_encoding: maximum relative distance between elements

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        group_size: int,
        max_pos_encoding: int,
    ):
        super(MultiHeadSelfAttention, self).__init__()

        # Attention Params
        self.d_model = d_model
        self.num_heads = num_heads
        self.group_size = group_size
        self.d_head = (group_size * d_model) // num_heads

        # Linear Layers
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)

        # Position Embedding Layer
        self.pos_layer = nn.Linear(d_model, d_model)

        # Global content and positional bias
        self.u = nn.Parameter(torch.Tensor(d_model))  # Content bias
        self.v = nn.Parameter(torch.Tensor(d_model))  # Pos bias
        nn.init.xavier_uniform_(self.u.reshape(num_heads, -1))
        nn.init.xavier_uniform_(self.v.reshape(num_heads, -1))

        # Grouped Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = PositionalEncoding(
            max_pos_encoding,
            d_model,
            group_size,
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, self.group_size)

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, Th + 2*T-G, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1)))

        # (B, T, D) -> (B, H, T//G, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.d_head)
        Qu = Qu.transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.d_head)
        Qv = Qv.transpose(1, 2)

        # (B, Th + T, D) -> (B, H, Th//G + T//G, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_head)
        K = K.transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_head)
        V = V.transpose(1, 2)

        # (B, Th + 2*T-G, D) -> (B, H, Th//G + 2*T//G-1, d) or
        # (B, Th + T, D) -> (B, H, Th//G + T//G, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.d_head)
        E = E.transpose(1, 2)

        # attn_scores (B, H, T//G, Th//G + T//G)
        attn_scores_K = Qu.matmul(K.transpose(2, 3))
        attn_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        attn_scores = (attn_scores_K + attn_scores_E) / K.shape[-1] ** 0.5

        # Slice Mask (B, 1, T, T) -> (B, 1, T//G, T//G)
        mask = mask.unsqueeze(1).bool()  # (batch, 1, time1, time2)
        mask = mask[:, :, :: self.group_size, :: self.group_size]

        # Apply mask
        min_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(mask, min_value)

        # Att weights (B, H, T//G, Th//G + T//G)
        attn_w = attn_scores.softmax(dim=-1)

        # Att output (B, H, T//G, d)
        output = attn_w.matmul(V)

        # Transpose and Reshape (B, H, T//G, d) -> (B, T, D)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        # Slice Padding
        output = output[:, : output.size(1) - padding]

        # output linear layer
        output = self.output_layer(output)

        return output

    def pad(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ):

        # Compute Overflows
        overflow_Q = Q.size(1) % chunk_size
        overflow_KV = K.size(1) % chunk_size

        padding_Q = (chunk_size - overflow_Q) % chunk_size
        padding_KV = (chunk_size - overflow_KV) % chunk_size

        batch_size, seq_len_KV, _ = K.size()

        # Input Padding (B, T, D) -> (B, T + P, D)
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0)

        # Update Padding Mask
        mask = mask.int()
        mask = F.pad(mask, pad=(0, padding_Q, 0, padding_KV), value=1)

        return Q, K, V, mask, padding_Q

    def rel_to_abs(self, attn_scores: torch.Tensor) -> torch.Tensor:

        """Relative to absolute position indexing
        Args:
            attn_scores: absolute-by-relative indexed attention scores of shape
                        (B, H, T, Th + 2*T-1) for full context and
                        (B, H, T, Th + T) for causal context
        Return:
            attn_scores: absolute-by-absolute indexed attention scores
                        of shape (B, H, T, Th + T)
        References:
            Attention Augmented Convolutional Networks, Bello et al.
            https://arxiv.org/abs/1904.09925
        """

        # Att Scores (B, H, T, Th + 2*T-1)
        batch_size, num_heads, seq_length1, seq_length2 = attn_scores.size()

        # Column Padding (B, H, T, Th + 2*T)
        attn_scores = F.pad(attn_scores, pad=(0, 1), value=0)

        # Flatten (B, H, TTh + 2*TT)
        attn_scores = attn_scores.reshape(batch_size, num_heads, -1)

        # End Padding (B, H, TTh + 2*TT + Th + T - 1)
        attn_scores = F.pad(
            attn_scores,
            pad=(0, seq_length2 - seq_length1),
            value=0,
        )

        # Reshape (B, H, T + 1, Th + 2*T-1)
        attn_scores = attn_scores.reshape(
            batch_size, num_heads, 1 + seq_length1, seq_length2
        )

        # Slice (B, H, T, Th + T)
        attn_scores = attn_scores[:, :, :seq_length1, seq_length1 - 1:]

        return attn_scores

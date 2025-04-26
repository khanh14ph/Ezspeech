import torch
import torch.nn as nn
import torch.nn.functional as F

from lightspeech.utils.common import make_padding_mask, compute_statistic


class AttentiveStatisticPooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Linear(3 * d_model, attention_dim),
            nn.SiLU(),
            nn.Linear(attention_dim, d_model),
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor):
        # Make binary mask of shape [N, L, 1]
        mask = make_padding_mask(x_lens, xs.size(1))
        mask = mask[:, :, None]

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        weights = mask / mask.sum(dim=1, keepdim=True)
        mean, std = compute_statistic(xs, weights)

        mean = mean.unsqueeze(1).repeat(1, xs.size(1), 1)
        std = std.unsqueeze(1).repeat(1, xs.size(1), 1)

        # Apply layers
        attn = torch.cat([xs, mean, std], dim=2)
        attn = self.tdnn(attn)

        # Filter out zero-paddings
        attn = attn.masked_fill(~mask, -1e3)
        attn = F.softmax(attn, dim=1)

        # Append mean and std of the batch
        mean, std = compute_statistic(xs, attn)
        outs = torch.cat((mean, std), dim=1)
        outs = self.proj(outs)

        return outs


class AttentiveStatisticClassifier(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        embedding_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Linear(3 * d_model, attention_dim),
            nn.SiLU(),
            nn.Linear(attention_dim, d_model),
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor):
        # Make binary mask of shape [N, L, 1]
        mask = make_padding_mask(x_lens, xs.size(1))
        mask = mask[:, :, None]

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        weights = mask / mask.sum(dim=1, keepdim=True)
        mean, std = compute_statistic(xs, weights)

        mean = mean.unsqueeze(1).repeat(1, xs.size(1), 1)
        std = std.unsqueeze(1).repeat(1, xs.size(1), 1)

        # Apply layers
        attn = torch.cat([xs, mean, std], dim=2)
        attn = self.tdnn(attn)

        # Filter out zero-paddings
        attn = attn.masked_fill(~mask, -1e3)
        attn = F.softmax(attn, dim=1)

        # Append mean and std of the batch
        mean, std = compute_statistic(xs, attn)
        outs = torch.cat((mean, std), dim=1)

        embeds = self.proj(outs)
        logits = self.head(outs)

        return embeds, logits


class AttentiveTransformerPooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token)

        self.attention = nn.TransformerEncoderLayer(
            d_model, num_heads, 4 * d_model, dropout, batch_first=True
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        cls_token = self.cls_token.repeat_interleave(xs.size(0), dim=0)

        xs = torch.cat((cls_token, xs), dim=1)
        x_lens = x_lens + 1

        masks = make_padding_mask(x_lens, xs.size(1))
        outs = self.attention(xs, src_key_padding_mask=~masks)

        embeds = self.projector(outs[:, 0, :])

        return embeds


class AttentiveTransformerClassifier(nn.Module):
    def __init__(
        self,
        d_model: int,
        embedding_dim: int,
        num_classes: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token)

        self.attention = nn.TransformerEncoderLayer(
            d_model, num_heads, 4 * d_model, dropout, batch_first=True
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )

        self.header = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        cls_token = self.cls_token.repeat_interleave(xs.size(0), dim=0)

        xs = torch.cat((cls_token, xs), dim=1)
        x_lens = x_lens + 1

        masks = make_padding_mask(x_lens, xs.size(1))
        outs = self.attention(xs, src_key_padding_mask=~masks)

        embeds = self.projector(outs[:, 0, :])
        logits = self.header(outs[:, 0, :])

        return embeds, logits

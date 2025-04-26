from typing import Tuple, Optional

# import numpy as np
# from numba import jit, prange

import torch
import torch.nn as nn

from lightspeech.utils.common import make_padding_mask, length_regulator


# @jit(nopython=True)
# def search(log_prob_attn):
#     neg_inf = log_prob_attn.dtype.type(-np.inf)
#     log_p = log_prob_attn.copy()
#     log_p[0, 1:] = neg_inf

#     for i in range(1, log_p.shape[0]):
#         prev_log1 = neg_inf
#         for j in range(log_p.shape[1]):
#             prev_log2 = log_p[i - 1, j]
#             log_p[i, j] += max(prev_log1, prev_log2)
#             prev_log1 = prev_log2

#     opt = np.zeros_like(log_p)
#     one = opt.dtype.type(1)
#     j = log_p.shape[1] - 1

#     for i in range(log_p.shape[0] - 1, 0, -1):
#         opt[i, j] = one
#         if log_p[i - 1, j - 1] >= log_p[i - 1, j]:
#             j -= 1
#             if j == 0:
#                 opt[1:i, j] = one
#                 break
#     opt[0, j] = one

#     return opt


# @jit(nopython=True, parallel=True)
# def batch_search(log_prob_attn, in_lens, out_lens):
#     attn_out = np.zeros_like(log_prob_attn)
#     for b in prange(log_prob_attn.shape[0]):
#         out = search(log_prob_attn[b, : out_lens[b], : in_lens[b]])
#         attn_out[b, : out_lens[b], : in_lens[b]] = out
#     return attn_out


# def binarize_alignment(soft_attn, text_lens, feat_lens):
#     device = soft_attn.device

#     soft_attn = soft_attn.detach().cpu().numpy()
#     text_lens = text_lens.detach().cpu().numpy()
#     feat_lens = feat_lens.detach().cpu().numpy()

#     hard_attn = batch_search(soft_attn, text_lens, feat_lens)
#     hard_attn = torch.from_numpy(hard_attn)
#     hard_attn = hard_attn.to(device)

#     return hard_attn


# class AlignmentAttention(nn.Module):
#     def __init__(self, vocab_size: int, feat_dim: int, d_model: int):
#         super(AlignmentAttention, self).__init__()
#         self.temperature = -0.0005

#         self.text_emb = nn.Embedding(vocab_size, d_model)
#         self.text_proj = nn.Sequential(
#             nn.Conv1d(d_model, 2 * d_model, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv1d(2 * d_model, d_model, 1),
#         )

#         self.feat_emb = nn.Linear(feat_dim, d_model)
#         self.feat_proj = nn.Sequential(
#             nn.Conv1d(d_model, 2 * d_model, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv1d(2 * d_model, d_model, 1),
#         )

#     def forward(
#         self,
#         tokens: torch.Tensor,
#         token_lens: torch.Tensor,
#         features: torch.Tensor,
#         feature_lens: torch.Tensor,
#         attn_priors: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

#         text_embs = self.text_emb(tokens).transpose(1, 2)
#         feat_embs = self.feat_emb(features).transpose(1, 2)

#         text_encs = self.text_proj(text_embs)
#         feat_encs = self.feat_proj(feat_embs)

#         distance = (feat_encs[:, :, :, None] - text_encs[:, :, None]) ** 2
#         distance = self.temperature * distance.sum(1)

#         masks = make_padding_mask(token_lens, text_encs.size(2))
#         masks = ~masks[:, None, :]

#         soft_attns = distance.log_softmax(2) + attn_priors
#         soft_attns = soft_attns.masked_fill(masks, -1e3)

#         hard_attns = binarize_alignment(soft_attns, token_lens, feature_lens)
#         dur_tgts = hard_attns.sum(1).long()

#         return soft_attns, hard_attns, dur_tgts


class ConditionalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super(ConditionalAttention, self).__init__()

        self.reference_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout, batch_first=True
        )

        self.duration_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        text_encs: torch.Tensor,
        text_lens: torch.Tensor,
        feat_encs: torch.Tensor,
        feat_lens: torch.Tensor,
        dur_tgts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        key_padding_mask = make_padding_mask(feat_lens, feat_encs.size(1))
        attn_outs = self.reference_attention(
            text_encs, feat_encs, feat_encs, ~key_padding_mask
        )[0]

        dur_mask = make_padding_mask(text_lens, text_encs.size(1))
        dur_outs = self.duration_predictor(attn_outs)
        dur_outs = dur_outs.squeeze(2).masked_fill(~dur_mask, -1e3)

        if dur_tgts is None:
            dur_tgts = dur_outs.exp().round().long().clamp(1)

        attn_outs, attn_lens = length_regulator(attn_outs, dur_mask, dur_tgts)

        return attn_outs, attn_lens, dur_outs

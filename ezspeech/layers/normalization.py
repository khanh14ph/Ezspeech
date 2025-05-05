import random

import torch
import torch.nn as nn

from ezspeech.utils.common import compute_statistic


class ScaleBiasNorm(nn.Module):
    def __init__(self, d_model: int):
        super(ScaleBiasNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, xs):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        xs = scale * xs + bias
        return xs


class AdaptiveNorm(nn.Module):
    def __init__(self, d_model: int, style_dim: int):
        super(AdaptiveNorm, self).__init__()
        self.d_model = d_model
        self.affine_layer = nn.Linear(style_dim, 2 * d_model, bias=False)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor, styles: torch.Tensor
    ) -> torch.Tensor:

        coeff = self.affine_layer(styles)
        scale, bias = coeff.split(self.d_model, dim=-1)

        mean, std = compute_statistic(xs, x_lens)
        xs = (xs - mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-5)

        xs = scale.unsqueeze(1) * xs + bias.unsqueeze(1)

        return xs


class MixStyleNorm(nn.Module):
    def __init__(self, d_model: int, style_dim: int):
        super(MixStyleNorm, self).__init__()
        self.d_model = d_model
        self.affine_layer = nn.Linear(style_dim, 2 * d_model, bias=False)
        self.distribution = torch.distributions.Beta(0.1, 0.1)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor, styles: torch.Tensor
    ) -> torch.Tensor:

        if (not self.training) or (random.random() > 0.2):
            return xs

        bs = xs.size(0)
        device = xs.device

        coeff = self.affine_layer(styles)
        mu1, sig1 = coeff.split(self.d_model, dim=-1)

        idxs = torch.randperm(bs)
        mu2, sig2 = mu1[idxs], sig1[idxs]

        weight = self.distribution.sample((bs, 1))
        weight = weight.to(device)

        scale = weight * mu1 + (1 - weight) * mu2
        bias = weight * sig1 + (1 - weight) * sig2

        mean, std = compute_statistic(xs, x_lens)
        xs = (xs - mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-5)

        xs = scale.unsqueeze(1) * xs + bias.unsqueeze(1)

        return xs

class ScaleBiasNorm(nn.Module):
    def __init__(self, d_model: int):
        super(ScaleBiasNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, xs):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        xs = scale * xs + bias
        return xs
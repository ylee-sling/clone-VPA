# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Ensemble JEPA latent predictor ``P_omega`` (Eq. 14) and the uncertainty filter.

``N_e = 5`` heads, each an MLP ``292 -> 512 -> 512 -> 256`` (544,000 parameters per
head, 2,720,000 in total, Table 3) with input ``(z_t, Emb(u_t), a_bar_t)``:
the current latent, the 32-dimensional primitive embedding and the most
recently dispatched setpoint (``d_a = 4``, chunk-normalized). The heads are
evaluated as one batched call (:class:`EnsembleLinear`), so the ensemble is a
single evaluation on the decision chain (Proposition 8). Eq. (14)::

    z_hat^(e) = P^(e)(z_t, Emb(u_t), a_bar_t),   z_hat = mean_e z_hat^(e),
    sigma = (1 / (sqrt(d) gamma_bar*)) * sqrt(mean_e ||z_hat^(e) - z_hat||^2).

:class:`UncertaintyFilter` implements the asymmetric filter
``sigma_bar_{t+1} = max(sigma, (1 - a) sigma_bar_t + a sigma)``, the chunk validity
``H_val = max(H_min, floor(H exp(-beta sigma_bar)))`` (Section 4.6.6) and the
velocity attenuation ``alpha(sigma_bar) = max(alpha_min, exp(-beta_alpha sigma_bar))``
carried over from Aero-VPA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["EnsembleLinear", "EnsemblePredictor", "PredictorOutput", "UncertaintyFilter"]


class EnsembleLinear(nn.Module):
    """``N_e`` independent affine maps in one batched product: ``(N_e, B, in) -> (N_e, B, out)``."""

    def __init__(self, members: int, in_features: int, out_features: int):
        super().__init__()
        self.members, self.in_features, self.out_features = members, in_features, out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(members, in_features, out_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(members, 1, out_features).uniform_(-bound, bound))

    def forward(self, x: Tensor) -> Tensor:
        return torch.baddbmm(self.bias, x, self.weight)


@dataclass
class PredictorOutput:
    z_hat: Tensor     # (B, d)
    sigma: Tensor     # (B,)
    heads: Tensor     # (N_e, B, d)


class EnsemblePredictor(nn.Module):
    """``P_omega`` with ``N_e`` heads; ``gamma_bar*`` is a buffer set at the end of Stage 1."""

    def __init__(self, latent_dim: int = 256, prim_dim: int = 32, action_dim: int = 4, hidden: int = 512,
                 n_hidden: int = 2, n_heads: int = 5):
        super().__init__()
        if n_heads < 2:
            raise ValueError("the ensemble needs at least two heads")
        self.latent_dim, self.prim_dim, self.action_dim, self.n_heads = latent_dim, prim_dim, action_dim, n_heads
        layers = [EnsembleLinear(n_heads, latent_dim + prim_dim + action_dim, hidden), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers += [EnsembleLinear(n_heads, hidden, hidden), nn.GELU()]
        layers.append(EnsembleLinear(n_heads, hidden, latent_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("gamma_star", torch.tensor(1.0))

    def set_gamma_star(self, gamma: float) -> None:
        if not gamma > 0:
            raise ValueError("gamma_bar* must be positive")
        self.gamma_star.fill_(float(gamma))

    def forward_heads(self, z_t: Tensor, prim_emb: Tensor, a_bar: Tensor) -> Tensor:
        x = torch.cat([z_t, prim_emb, a_bar], -1)
        return self.net(x.unsqueeze(0).expand(self.n_heads, *x.shape))

    def forward(self, z_t: Tensor, prim_emb: Tensor, a_bar: Tensor) -> PredictorOutput:
        """Eq. (14). One evaluation on the decision chain."""
        heads = self.forward_heads(z_t, prim_emb, a_bar)
        z_hat = heads[0] + (heads - heads[:1]).mean(0)          # exact when all heads agree
        msd = (heads - z_hat.unsqueeze(0)).pow(2).sum(-1).mean(0)
        sigma = msd.sqrt() / (math.sqrt(self.latent_dim) * self.gamma_star.to(z_hat.dtype))
        return PredictorOutput(z_hat, sigma, heads)


class UncertaintyFilter:
    """Fast-attack / slow-release filter, validity horizon and velocity attenuation (batched)."""

    def __init__(self, H: int = 32, H_min: int = 8, beta: float = 1.0, alpha_sigma: float = 0.1,
                 beta_alpha: float = 0.5, alpha_min: float = 0.3):
        if not 1 <= H_min <= H:
            raise ValueError("require 1 <= H_min <= H")
        if not 0.0 < alpha_sigma <= 1.0:
            raise ValueError("alpha_sigma must lie in (0, 1]")
        self.H, self.H_min, self.beta, self.alpha_sigma = H, H_min, beta, alpha_sigma
        self.beta_alpha, self.alpha_min = beta_alpha, alpha_min
        self.sigma_bar: Optional[Tensor] = None

    def reset(self, batch: int, device: Optional[torch.device] = None) -> None:
        self.sigma_bar = torch.zeros(batch, device=device)

    def update(self, sigma: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        prev = torch.zeros_like(sigma) if self.sigma_bar is None else self.sigma_bar.to(sigma)
        new = torch.maximum(sigma, (1.0 - self.alpha_sigma) * prev + self.alpha_sigma * sigma)
        if mask is not None:
            new = torch.where(mask, new, prev)
        self.sigma_bar = new
        return new

    def validity(self, sigma_bar: Tensor) -> Tensor:
        """``H_val`` in knots (int64)."""
        return torch.clamp(torch.floor(self.H * torch.exp(-self.beta * sigma_bar)).long(), min=self.H_min)

    def attenuation(self, sigma_bar: Tensor) -> Tensor:
        """Velocity attenuation ``alpha(sigma_bar)`` in ``[alpha_min, 1]``."""
        return torch.clamp(torch.exp(-self.beta_alpha * sigma_bar), min=self.alpha_min, max=1.0)

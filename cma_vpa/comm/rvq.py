# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Residual vector quantization of the latent bus (Definition 3, Lemma 4, Proposition 5).

The bus carries a vector ``x in R^{d_pi}`` (``d_pi = 64``) as ``B = 24`` one-byte
indices into codebooks ``Q_b in R^{256 x d_pi}``::

    r_0 = x,   k_b = argmin_k ||r_{b-1} - Q_b[k]||,   r_b = r_{b-1} - Q_b[k_b],   b = 1..B
    x_hat = sum_b Q_b[k_b]

Ties in the argmin are broken by the smallest index (``torch.argmin`` returns the
first minimizer). Row ``0`` of every codebook is the zero vector and is not a
parameter, so Lemma 4 holds by construction: choosing ``k_b = 0`` keeps the
residual, hence ``||r_b|| <= ||r_{b-1}||``, ``||x - x_hat|| = ||r_B|| <= ||x||`` and
``||x_hat|| <= 2 ||x||``. Because the argmin is evaluated with the expanded
distance ``||r||^2 - 2 r.Q + ||Q||^2`` (one matrix product per stage), the
selected codeword is re-checked against the zero codeword with the exact
residual norm, so the monotonicity is exact in floating point, not only up to
rounding.

The decoder sums the selected codewords in the fixed order ``b = 1..B``; it is
a deterministic function of the code and the (shared, frozen) codebooks, so
two receivers of the same message reconstruct bit-identical vectors on the
same kernels (Proposition 4(c)).

Training (Stage 2, Section 4.8) uses the codebook and commitment terms of
VQ-VAE/RVQ [68, 69] with a straight-through estimator; :meth:`ResidualVQ.init_from_data`
seeds the codebooks by per-stage k-means on the residuals of a data batch, and
:meth:`ResidualVQ.restart_dead_codes` revives unused codewords.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "N_STAGES",
    "CODEBOOK_SIZE",
    "BUS_DIM",
    "RVQOutput",
    "ResidualVQ",
    "gaussian_rd_floor",
    "rd_floor",
]

N_STAGES = 24        # B
CODEBOOK_SIZE = 256  # one byte per stage
BUS_DIM = 64         # d_pi


@dataclass
class RVQOutput:
    """Result of :meth:`ResidualVQ.forward`.

    ``quantized`` carries the straight-through gradient (value ``x_hat``,
    gradient of the identity with respect to ``x``); ``codes`` are ``uint8``
    ``(..., B)``; ``residual_norms`` are ``||r_b||`` for ``b = 0..B``,
    shape ``(..., B + 1)``.
    """

    quantized: Tensor
    x_hat: Tensor
    codes: Tensor
    loss: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor
    residual_norms: Tensor


class ResidualVQ(nn.Module):
    """``B``-stage residual vector quantizer with a fixed zero codeword per stage.

    Parameters
    ----------
    n_stages, codebook_size, dim:
        ``B``, codebook rows (at most 256 so that an index fits one byte) and ``d_pi``.
    beta:
        Commitment weight.
    init_scale:
        Standard deviation of the random initialization of stage 1; stage ``b``
        is initialized with ``init_scale * decay**(b-1)`` (residuals shrink).
    decay:
        Per-stage decay of the initialization scale.
    """

    def __init__(self, n_stages: int = N_STAGES, codebook_size: int = CODEBOOK_SIZE, dim: int = BUS_DIM,
                 beta: float = 0.25, init_scale: float = 1.0, decay: float = 0.7):
        super().__init__()
        if not 2 <= codebook_size <= 256:
            raise ValueError("codebook_size must lie in [2, 256] (one byte per stage)")
        self.n_stages, self.codebook_size, self.dim = n_stages, codebook_size, dim
        self.beta = beta
        scales = init_scale * decay ** torch.arange(n_stages, dtype=torch.float32) / math.sqrt(dim)
        free = torch.randn(n_stages, codebook_size - 1, dim) * scales.view(-1, 1, 1)
        self.codebook_free = nn.Parameter(free)                      # rows 1..K-1 of each stage
        self.register_buffer("usage", torch.zeros(n_stages, codebook_size))
        self._frozen = False

    # ------------------------------------------------------------------ codebooks
    @property
    def codebooks(self) -> Tensor:
        """All codebooks ``(B, K, d_pi)`` with the zero vector in row 0 (Lemma 4)."""
        zero = torch.zeros(self.n_stages, 1, self.dim, dtype=self.codebook_free.dtype,
                           device=self.codebook_free.device)
        return torch.cat([zero, self.codebook_free], dim=1)

    def freeze(self) -> "ResidualVQ":
        """Freeze the codebooks (end of Stage 2)."""
        self.codebook_free.requires_grad_(False)
        self._frozen = True
        return self

    # ------------------------------------------------------------------ core greedy pass
    def _greedy(self, x: Tensor, Q: Tensor):
        """Greedy residual search; returns ``(codes long (N, B), x_hat (N, d), residuals list)``."""
        r = x
        q_sq = (Q * Q).sum(-1)                                        # (B, K)
        r_sq = (r * r).sum(-1)
        codes, picked, norms = [], [], [torch.sqrt(r_sq)]
        for b in range(self.n_stages):
            dist = r_sq.unsqueeze(-1) - 2.0 * (r @ Q[b].T) + q_sq[b]          # (N, K)
            k = torch.argmin(dist, dim=-1)                                   # first minimizer on ties
            qk = Q[b][k]
            r_new = r - qk
            r_new_sq = (r_new * r_new).sum(-1)
            worse = r_new_sq > r_sq                                          # exact Lemma 4 guard
            k = torch.where(worse, torch.zeros_like(k), k)
            qk = torch.where(worse.unsqueeze(-1), torch.zeros_like(qk), qk)
            r = torch.where(worse.unsqueeze(-1), r, r_new)
            r_sq = torch.where(worse, r_sq, r_new_sq)
            codes.append(k)
            picked.append(qk)
            norms.append(torch.sqrt(r_sq))
        return torch.stack(codes, -1), picked, torch.stack(norms, -1)

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode ``x (..., d_pi)`` to ``uint8`` codes ``(..., B)``."""
        lead = x.shape[:-1]
        codes, _, _ = self._greedy(x.reshape(-1, self.dim), self.codebooks.to(x.dtype))
        return codes.to(torch.uint8).reshape(*lead, self.n_stages)

    def decode(self, codes: Tensor) -> Tensor:
        """``x_hat = sum_b Q_b[k_b]`` in the fixed order ``b = 1..B``; ``codes (..., B)``."""
        if codes.shape[-1] != self.n_stages:
            raise ValueError(f"expected {self.n_stages} code bytes, got {codes.shape[-1]}")
        lead = codes.shape[:-1]
        c = codes.reshape(-1, self.n_stages).long()
        if bool((c >= self.codebook_size).any()):
            raise ValueError("code index outside the codebook")
        Q = self.codebooks
        out = torch.zeros(c.shape[0], self.dim, dtype=Q.dtype, device=Q.device)
        for b in range(self.n_stages):
            out = out + Q[b][c[:, b]]
        return out.reshape(*lead, self.dim)

    def residual_norms(self, x: Tensor) -> Tensor:
        """``||r_b||`` for ``b = 0..B`` (Lemma 4 diagnostics), shape ``(..., B + 1)``."""
        lead = x.shape[:-1]
        with torch.no_grad():
            _, _, norms = self._greedy(x.reshape(-1, self.dim), self.codebooks.to(x.dtype))
        return norms.reshape(*lead, self.n_stages + 1)

    # ------------------------------------------------------------------ training pass
    def forward(self, x: Tensor, update_usage: bool = True) -> RVQOutput:
        """Quantize with straight-through gradients and the RVQ training losses.

        ``codebook_loss = sum_b ||sg(r_{b-1}) - Q_b[k_b]||^2`` moves the selected
        codewords toward the residuals they quantize; ``commitment_loss =
        ||x - sg(x_hat)||^2`` keeps the encoder output close to its
        quantization. Both are batch means; ``loss = codebook + beta * commitment``.
        """
        lead = x.shape[:-1]
        xf = x.reshape(-1, self.dim)
        Q = self.codebooks.to(x.dtype)
        with torch.no_grad():
            codes, _, norms = self._greedy(xf.detach(), Q.detach())
        # differentiable reconstruction of the selected codewords and residual targets
        picked = Q[torch.arange(self.n_stages, device=x.device).unsqueeze(0), codes]   # (N, B, d)
        x_hat = picked.sum(1)
        csum = torch.cumsum(picked.detach(), dim=1)
        r_prev = xf.detach().unsqueeze(1) - torch.cat([torch.zeros_like(csum[:, :1]), csum[:, :-1]], 1)
        codebook_loss = (r_prev - picked).pow(2).sum(-1).sum(-1).mean()
        commitment_loss = (xf - x_hat.detach()).pow(2).sum(-1).mean()
        quantized = xf + (x_hat - xf).detach()
        if update_usage and self.training:
            with torch.no_grad():
                one_hot = torch.zeros(self.n_stages, self.codebook_size, device=x.device)
                one_hot.scatter_add_(1, codes.T, torch.ones_like(codes.T, dtype=one_hot.dtype))
                self.usage.mul_(0.99).add_(one_hot.to(self.usage.dtype))
        return RVQOutput(quantized=quantized.reshape(*lead, self.dim), x_hat=x_hat.reshape(*lead, self.dim),
                         codes=codes.to(torch.uint8).reshape(*lead, self.n_stages),
                         loss=codebook_loss + self.beta * commitment_loss,
                         codebook_loss=codebook_loss, commitment_loss=commitment_loss,
                         residual_norms=norms.reshape(*lead, self.n_stages + 1))

    # ------------------------------------------------------------------ initialization / maintenance
    @torch.no_grad()
    def init_from_data(self, x: Tensor, iters: int = 10, generator: Optional[torch.Generator] = None) -> None:
        """Per-stage k-means on the residuals of ``x (N, d_pi)``; the zero row stays fixed.

        Stage ``b`` is fitted to the residuals left by stages ``1..b-1``; the
        zero codeword takes part in the assignment, so every cluster center is
        the mean of the residuals it captures.
        """
        x = x.reshape(-1, self.dim).to(self.codebook_free.dtype)
        n = x.shape[0]
        K = self.codebook_size
        r = x.clone()
        for b in range(self.n_stages):
            idx = torch.randint(0, n, (K - 1,), generator=generator, device=x.device) if n < K - 1 else \
                torch.randperm(n, generator=generator, device=x.device)[:K - 1]
            C = r[idx].clone()
            for _ in range(iters):
                full = torch.cat([torch.zeros_like(C[:1]), C], 0)
                a = torch.cdist(r, full).argmin(-1)
                sums = torch.zeros_like(full).index_add_(0, a, r)
                cnt = torch.bincount(a, minlength=K).to(r.dtype)
                upd = cnt[1:] > 0
                C = torch.where(upd.unsqueeze(-1), sums[1:] / cnt[1:].clamp(min=1).unsqueeze(-1), C)
            self.codebook_free.data[b] = C
            full = torch.cat([torch.zeros_like(C[:1]), C], 0)
            k = torch.cdist(r, full).argmin(-1)
            r_new = r - full[k]
            keep = (r_new * r_new).sum(-1) > (r * r).sum(-1)
            r = torch.where(keep.unsqueeze(-1), r, r_new)

    @torch.no_grad()
    def restart_dead_codes(self, x: Tensor, threshold: float = 1e-3,
                           generator: Optional[torch.Generator] = None) -> int:
        """Replace codewords with decayed usage below ``threshold`` by residuals of ``x``.

        Returns the number of restarted codewords. Has no effect once frozen.
        """
        if self._frozen:
            return 0
        x = x.reshape(-1, self.dim).to(self.codebook_free.dtype)
        _, picked, _ = self._greedy(x, self.codebooks)
        r = x
        restarted = 0
        for b in range(self.n_stages):
            dead = (self.usage[b, 1:] < threshold).nonzero(as_tuple=True)[0]
            if dead.numel() > 0 and r.shape[0] > 0:
                src = torch.randint(0, r.shape[0], (dead.numel(),), generator=generator, device=x.device)
                self.codebook_free.data[b, dead] = r[src]
                self.usage[b, dead + 1] = 1.0
                restarted += dead.numel()
            r = r - picked[b]
        return restarted

    # ------------------------------------------------------------------ metrics
    @torch.no_grad()
    def relative_mse(self, x: Tensor) -> Tensor:
        """``E||x - x_hat||^2 / E||x||^2`` over the batch (Proposition 5 readout)."""
        x_hat = self.decode(self.encode(x))
        return (x - x_hat).pow(2).sum(-1).mean() / x.pow(2).sum(-1).mean().clamp(min=1e-30)

    @torch.no_grad()
    def snr_db(self, x: Tensor) -> float:
        """Signal-to-quantization-noise ratio ``-10 log10(relative MSE)`` [dB]."""
        return float(-10.0 * torch.log10(self.relative_mse(x).clamp(min=1e-30)))


# ---------------------------------------------------------------------------
# Proposition 5: rate-distortion floor of the bus
# ---------------------------------------------------------------------------
def rd_floor(h_nats: float, d_pi: int = BUS_DIM, n_bytes: int = N_STAGES) -> float:
    """Eq. (10): ``E||pi - phi(f(pi))||^2 >= d/(2 pi e) exp(2/d (h - 8 B ln 2))``."""
    return d_pi / (2.0 * math.pi * math.e) * math.exp(2.0 / d_pi * (h_nats - 8.0 * n_bytes * math.log(2.0)))


def gaussian_rd_floor(d_pi: int = BUS_DIM, n_bytes: int = N_STAGES) -> float:
    """Relative MSE floor ``2^(-16 B / d_pi)`` for an isotropic Gaussian plan (1/64 at the defaults)."""
    return 2.0 ** (-16.0 * n_bytes / d_pi)

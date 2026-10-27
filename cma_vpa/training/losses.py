# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Training objectives of Section 4.8 (the barrier never enters any of them).

* :func:`vicreg_variance`, :func:`vicreg_covariance` - Eq. (8)::

      v(Z) = (1/d) sum_j max(0, gamma_v - sqrt(Var(Z_.j) + eps_v)),
      c(Z) = (1/d) sum_{j != k} [C(Z)]_jk^2

  with the unbiased sample covariance ``C(Z)``, ``gamma_v = 1`` and ``eps_v = 1e-4``
  (``0 < eps_v < gamma_v^2``).
* :func:`sigreg` - SIGReg [40]: the squared distance between the empirical
  characteristic function of projections ``a^T z`` along random unit directions
  and ``exp(-t^2/2)``, integrated against the weight ``w(t) = exp(-t^2/2)`` by the
  trapezoidal rule (Lemma 2 at the population level).
* :func:`jepa_loss` - ``(1/N_e) sum_e ||z_hat^(e) - sg(E_psi_bar(I_{t+1}))||^2 + lambda_v v(Z) + lambda_c c(Z)``
  (+ optional ``lambda_s`` SIGReg), with optional per-head bootstrap weights that
  keep the ensemble heads epistemically diverse.
* :func:`cfm_loss` - Eq. (17), squared Frobenius norm over the chunk.
* :func:`bus_loss` - Stage 2: tied linear reconstruction of the standardized
  latent from the quantized bus vector + RVQ codebook and commitment terms.
* :func:`master_loss` - Stage 3: plan regression to the EMA-target bus latent at
  ``T_M`` (teacher-forced modes), mode cross-entropies, variance-covariance
  regularization of the agent features.
* :func:`policy_loss` - Stage 4: ``L_CFM + lambda_h L_CE``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "vicreg_variance", "vicreg_covariance", "sigreg", "jepa_loss", "cfm_loss", "bus_loss",
    "master_loss", "policy_loss", "bootstrap_weights",
]


def _check(Z: Tensor) -> None:
    if Z.dim() != 2 or Z.shape[0] < 2:
        raise ValueError(f"expected a batch (B >= 2, d), got {tuple(Z.shape)}")


def vicreg_variance(Z: Tensor, gamma_v: float = 1.0, eps_v: float = 1e-4) -> Tensor:
    """Eq. (8) ``v(Z)``."""
    _check(Z)
    if not 0.0 < eps_v < gamma_v ** 2:
        raise ValueError("require 0 < eps_v < gamma_v^2")
    return F.relu(gamma_v - torch.sqrt(Z.var(0, unbiased=True) + eps_v)).mean()


def vicreg_covariance(Z: Tensor) -> Tensor:
    """Eq. (8) ``c(Z)``."""
    _check(Z)
    B, d = Z.shape
    Zc = Z - Z.mean(0, keepdim=True)
    C = Zc.T @ Zc / (B - 1)
    off = C - torch.diag_embed(torch.diagonal(C))
    return off.pow(2).sum() / d


def sigreg(Z: Tensor, n_directions: int = 64, t_max: float = 3.0, n_t: int = 17,
           generator: Optional[torch.Generator] = None) -> Tensor:
    """SIGReg statistic averaged over ``n_directions`` random unit directions."""
    _check(Z)
    d = Z.shape[1]
    A = torch.randn(d, n_directions, generator=generator, device=Z.device, dtype=Z.dtype)
    A = A / torch.linalg.vector_norm(A, dim=0, keepdim=True)
    s = Z @ A                                                                  # (B, M)
    t = torch.linspace(-t_max, t_max, n_t, device=Z.device, dtype=Z.dtype)     # (T,)
    ts = s.unsqueeze(-1) * t                                                   # (B, M, T)
    re = torch.cos(ts).mean(0) - torch.exp(-0.5 * t * t)
    im = torch.sin(ts).mean(0)
    w = torch.exp(-0.5 * t * t)
    integrand = (re * re + im * im) * w                                        # (M, T)
    return torch.trapezoid(integrand, t, dim=-1).mean()


def bootstrap_weights(n_heads: int, batch: int, generator: Optional[torch.Generator] = None,
                      device=None) -> Tensor:
    """Poisson(1) bootstrap weights ``(N_e, B)`` (online bagging of the ensemble heads)."""
    lam = torch.ones(n_heads, batch, device=device)
    return torch.poisson(lam, generator=generator)


def jepa_loss(heads: Tensor, target: Tensor, Z: Tensor, *, lambda_v: float = 25.0, lambda_c: float = 1.0,
              lambda_s: float = 0.0, gamma_v: float = 1.0, eps_v: float = 1e-4,
              head_weights: Optional[Tensor] = None,
              generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Stage-1 objective on per-head forecasts ``heads (N_e, B, d)``, targets ``(B, d)`` and online ``Z (B, d)``."""
    tgt = target.detach()                                                      # sg(.)
    err = (heads - tgt.unsqueeze(0)).pow(2).sum(-1)                            # (N_e, B)
    if head_weights is not None:
        pred = (err * head_weights).sum(-1) / head_weights.sum(-1).clamp(min=1.0)
        pred = pred.mean()
    else:
        pred = err.mean()
    v = vicreg_variance(Z, gamma_v, eps_v)
    c = vicreg_covariance(Z)
    total = pred + lambda_v * v + lambda_c * c
    s = torch.zeros((), device=Z.device)
    if lambda_s > 0:
        s = sigreg(Z, generator=generator)
        total = total + lambda_s * s
    return total, {"pred": pred.detach(), "var": v.detach(), "cov": c.detach(), "sigreg": s.detach()}


def cfm_loss(field, actions: Tensor, cond: Tensor, kv, generator: Optional[torch.Generator] = None) -> Tensor:
    """Eq. (17): ``E ||v_theta(tau A + (1 - tau) xi, tau | e, Z) - (A - xi)||_F^2`` (batch mean)."""
    xi = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
    tau = torch.rand(actions.shape[0], generator=generator, device=actions.device, dtype=actions.dtype)
    a_tau = tau.view(-1, 1, 1) * actions + (1.0 - tau.view(-1, 1, 1)) * xi
    return (field(a_tau, tau, cond, kv) - (actions - xi)).pow(2).sum((1, 2)).mean()


def bus_loss(z_tilde: Tensor, bus, rvq, lambda_rec: float = 1.0) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Stage 2: ``||z_tilde - P_bus^T(x_hat - b)||^2`` (straight-through) + RVQ codebook/commitment terms."""
    x = bus(z_tilde)
    out = rvq(x)
    rec = (z_tilde - bus.reconstruct(out.quantized)).pow(2).sum(-1).mean()
    total = lambda_rec * rec + out.loss
    return total, {"rec": rec.detach(), "codebook": out.codebook_loss.detach(),
                   "commit": out.commitment_loss.detach(), "out": out}


def master_loss(out, target: Tensor, present: Tensor, u_g: Tensor, u_M: Tensor, *, lambda_mode: float = 1.0,
                lambda_vc: float = 0.1, head_weights: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Stage 3 on a teacher-forced :class:`~cma_vpa.models.MasterOutput`.

    ``target (B, N, d_pi)`` is the EMA-target bus latent at ``T_M``; ``present (B, N)``
    masks absent agent slots; ``u_g (B,)`` and ``u_M (B, N)`` are the rule-based labels.
    """
    m = present.to(target.dtype)
    err = (out.plan_heads - target.unsqueeze(0)).pow(2).sum(-1)                # (n, B, N)
    w = m.unsqueeze(0) if head_weights is None else m.unsqueeze(0) * head_weights
    plan = (err * w).sum() / w.sum().clamp(min=1.0)
    ce_g = F.cross_entropy(out.global_logits, u_g)
    ll = out.local_logits[present]
    ce_l = F.cross_entropy(ll, u_M[present]) if ll.shape[0] > 0 else torch.zeros((), device=target.device)
    H = out.h[present]
    vc = (vicreg_variance(H) + vicreg_covariance(H)) if H.shape[0] >= 2 else torch.zeros((), device=target.device)
    total = plan + lambda_mode * (ce_g + ce_l) + lambda_vc * vc
    return total, {"plan": plan.detach(), "ce_g": ce_g.detach(), "ce_l": ce_l.detach(), "vc": vc.detach()}


def policy_loss(field, actions: Tensor, cond: Tensor, kv, logits: Tensor, prim: Tensor, lambda_h: float = 1.0,
                generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Stage 4: ``L_CFM + lambda_h L_CE``."""
    l_cfm = cfm_loss(field, actions, cond, kv, generator)
    l_ce = F.cross_entropy(logits, prim)
    return l_cfm + lambda_h * l_ce, {"cfm": l_cfm.detach(), "ce": l_ce.detach()}

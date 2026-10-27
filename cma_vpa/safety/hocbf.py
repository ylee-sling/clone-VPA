# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Braking-consistent high-order control barrier function of relative degree two.

Implements Section 4.6.1-4.6.2 (Eq. 21-25, Lemmas 7-11) and Appendix B.1-B.2 of
*Dynamic Latent Cloning with Continuous Barrier Guidance* (CMA-VPA).

For a pair ``(i, j)`` with relative position ``dp = p_i - p_j``, relative
velocity ``dv = v_i - v_j`` and collision distance ``D = r_i,eff + r_j,eff``::

    rho   = ||dp||,  m = dp . dv,  c = -m / rho          (closing speed)
    cbar(rho) = sqrt(2 a_max (rho - D) + varsigma^2) - varsigma   (Eq. 23)
    X     = cbar + varsigma,   k = a_max D - varsigma^2 / 2
    psi_0 = b = rho^2 - D^2
    psi_1 = 2 m + alpha_1(b) = 2 rho (cbar - c)
    alpha_1(b)   = 2 rho cbar                      with rho = sqrt(b + D^2)
    alpha_1'(b)  = cbar / rho + a_max / X                                  (Eq. 25)
    alpha_1''(b) = -(3/4 X^4 - varsigma X^3 + k^2) / (2 X^3 rho^3)         (Eq. 25)
    beta  = -2 ||dv||^2 - 2 alpha_1'(b) m - lambda_b psi_1                 (Eq. 24)

The HOCBF condition is ``2 dp . da >= beta`` for the relative realized
acceleration ``da``.

Numerical notes
---------------
* ``cbar`` is evaluated as ``2 a_max (rho - D) / (X + varsigma)``, which avoids
  the cancellation of ``X - varsigma`` near the collision distance.
* When the barrier is parametrized by ``b`` the offset ``rho - D`` is evaluated
  as ``b / (rho + D)``, which keeps finite-difference checks accurate at small
  ``b``.
* ``alpha_1`` is smooth only on ``[b_ext, inf)`` with
  ``rho_ext = D - varsigma^2 / (4 a_max)``. Every function returns a ``valid``
  mask (or documents its domain); values outside the domain are computed on a
  clamped radicand and are finite but meaningless. The margin module turns
  out-of-domain cases into ``+inf`` margins.

All functions are vectorized: scalar per-pair quantities have shape ``(...)``,
vector quantities ``(..., d)``. ``D`` may be a Python float or a tensor
broadcastable to ``(...)``. Double precision is recommended for the safety
layer; every function preserves the dtype and device of its tensor inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

from .params import DEFAULT_PARAMS, SafetyParams

__all__ = [
    "AlphaTerms",
    "HOCBFTerms",
    "as_tensor",
    "rho_ext",
    "b_ext",
    "alpha_terms_rho",
    "alpha_terms_b",
    "closing_speed_limit",
    "alpha1",
    "alpha1_prime",
    "alpha1_second",
    "alpha1_second_direct",
    "alpha1_second_numerator",
    "lambda_min",
    "hocbf_terms",
    "share_slack",
    "pair_slack",
    "spec_barrier",
    "pairwise_relative",
]

TensorLike = Union[Tensor, float, int]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def as_tensor(x: TensorLike, like: Tensor) -> Tensor:
    """Convert ``x`` to a tensor with the dtype and device of ``like``."""
    if isinstance(x, Tensor):
        if x.dtype == like.dtype and x.device == like.device:
            return x
        return x.to(dtype=like.dtype, device=like.device)
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def _ref(*xs: TensorLike) -> Tensor:
    """First tensor among ``xs``; a float64 scalar if all are Python numbers."""
    for x in xs:
        if isinstance(x, Tensor):
            if not x.is_floating_point():
                return x.to(torch.get_default_dtype())
            return x
    return torch.zeros((), dtype=torch.float64)


def rho_ext(D: TensorLike, cfg: SafetyParams = DEFAULT_PARAMS) -> TensorLike:
    """``rho_ext = D - varsigma^2 / (4 a_max)``: left end of the domain of alpha_1."""
    return cfg.rho_ext(D)


def b_ext(D: TensorLike, cfg: SafetyParams = DEFAULT_PARAMS) -> TensorLike:
    """``b_ext = rho_ext^2 - D^2`` (Lemma 8)."""
    r = cfg.rho_ext(D)
    return r * r - D * D


# ---------------------------------------------------------------------------
# class-K chain
# ---------------------------------------------------------------------------
class AlphaTerms(NamedTuple):
    """Braking-consistent class-K function and its derivatives at one ``rho``.

    Attributes
    ----------
    rho, X, c_bar:
        Distance, ``X = sqrt(2 a_max (rho - D) + varsigma^2)`` and the
        admissible closing speed ``cbar = X - varsigma``.
    alpha1, alpha1_p, alpha1_pp:
        ``alpha_1(b)``, ``alpha_1'(b)``, ``alpha_1''(b)`` at ``b = rho^2 - D^2``.
    valid:
        ``rho >= rho_ext`` (smooth domain of Lemma 8).
    """

    rho: Tensor
    X: Tensor
    c_bar: Tensor
    alpha1: Tensor
    alpha1_p: Tensor
    alpha1_pp: Tensor
    valid: Tensor


def _alpha_core(rho: Tensor, delta: Tensor, D: Tensor, cfg: SafetyParams) -> AlphaTerms:
    """Shared evaluation of Eq. (23) and Eq. (25) from ``rho`` and ``delta = rho - D``."""
    a = cfg.a_max
    s = cfg.varsigma
    rad = 2.0 * a * delta + s * s
    valid = delta >= -(s * s) / (4.0 * a)
    # Clamp only to keep values finite outside the domain; inside the domain
    # rad >= s^2 / 2 > 0 so the clamp is inactive.
    tiny = torch.finfo(rad.dtype).eps * s * s
    X = torch.sqrt(torch.clamp(rad, min=tiny))
    c_bar = 2.0 * a * delta / (X + s)  # == X - s without cancellation
    rho_safe = torch.clamp(rho, min=torch.finfo(rho.dtype).tiny)
    a1 = 2.0 * rho * c_bar
    a1p = c_bar / rho_safe + a / X
    k = a * D - 0.5 * s * s
    X3 = X * X * X
    num = 0.75 * X3 * X - s * X3 + k * k
    a1pp = -num / (2.0 * X3 * rho_safe * rho_safe * rho_safe)
    return AlphaTerms(rho, X, c_bar, a1, a1p, a1pp, valid)


def alpha_terms_rho(rho: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                    cfg: SafetyParams = DEFAULT_PARAMS) -> AlphaTerms:
    """Evaluate ``cbar``, ``alpha_1``, ``alpha_1'``, ``alpha_1''`` at distance ``rho``.

    ``rho`` and ``D`` broadcast; ``rho - D`` is formed directly.
    """
    ref = _ref(rho, D)
    rho_t = as_tensor(rho, ref)
    D_t = as_tensor(D, ref)
    return _alpha_core(rho_t, rho_t - D_t, D_t, cfg)


def alpha_terms_b(b: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                  cfg: SafetyParams = DEFAULT_PARAMS) -> AlphaTerms:
    """Evaluate the class-K chain as a function of ``b = rho^2 - D^2``.

    ``rho - D`` is evaluated as ``b / (rho + D)`` (exact for small ``b``).
    """
    ref = _ref(b, D)
    b_t = as_tensor(b, ref)
    D_t = as_tensor(D, ref)
    rho = torch.sqrt(torch.clamp(b_t + D_t * D_t, min=0.0))
    delta = b_t / (rho + D_t)
    return _alpha_core(rho, delta, D_t, cfg)


def closing_speed_limit(rho: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                        cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """Admissible closing speed ``cbar(rho)`` of Eq. (23)."""
    return alpha_terms_rho(rho, D, cfg).c_bar


def alpha1(b: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
           cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """First class-K function ``alpha_1(b) = 2 rho cbar(rho)|_{rho = sqrt(b + D^2)}``."""
    return alpha_terms_b(b, D, cfg).alpha1


def alpha1_prime(b: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                 cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """``alpha_1'(b) = cbar / rho + a_max / X`` (Eq. 25)."""
    return alpha_terms_b(b, D, cfg).alpha1_p


def alpha1_second(b: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                  cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """Closed form ``alpha_1''(b) = -(3/4 X^4 - varsigma X^3 + k^2) / (2 X^3 rho^3)``."""
    return alpha_terms_b(b, D, cfg).alpha1_pp


def alpha1_second_direct(b: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                         cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """Direct derivative of Appendix B.1 (before the simplification with ``k``).

    ``alpha_1''(b) = (a_max / (X rho) - (X - varsigma) / rho^2 - a_max^2 / X^3) / (2 rho)``.
    Used to check the closed form of Eq. (25).
    """
    t = alpha_terms_b(b, D, cfg)
    a = cfg.a_max
    rho, X = t.rho, t.X
    return (a / (X * rho) - t.c_bar / (rho * rho) - a * a / (X * X * X)) / (2.0 * rho)


def alpha1_second_numerator(X: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                            cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """Numerator ``3/4 X^4 - varsigma X^3 + k^2`` of ``-alpha_1''``.

    Lemma 8: its minimum over ``X > 0`` is ``a_max D (a_max D - varsigma^2)``,
    attained at ``X = varsigma`` (that is, at ``b = 0``).
    """
    ref = _ref(X, D)
    X_t = as_tensor(X, ref)
    D_t = as_tensor(D, ref)
    k = cfg.k(D_t)
    return 0.75 * X_t ** 4 - cfg.varsigma * X_t ** 3 + k * k


def lambda_min(D: TensorLike = DEFAULT_PARAMS.D_nominal,
               cfg: SafetyParams = DEFAULT_PARAMS) -> Tuple[Tensor, Tensor]:
    """Closed form of ``lambda_min = sup_{rho >= D} cbar(rho) / rho`` (Lemma 11).

    Stationarity of ``cbar / rho`` gives ``a_max rho = X (X - varsigma)``; with
    ``a_max rho = k + X^2 / 2`` this is ``X^2 - 2 varsigma X - 2k = 0``, so
    ``X* = varsigma + sqrt(2 a_max D)`` and
    ``rho* = D + (X*^2 - varsigma^2) / (2 a_max)``.

    Returns
    -------
    (lambda_min, rho_star)
    """
    ref = _ref(D)
    D_t = as_tensor(D, ref)
    a, s = cfg.a_max, cfg.varsigma
    X_star = s + torch.sqrt(2.0 * a * D_t)
    rho_star = D_t + (X_star * X_star - s * s) / (2.0 * a)
    return (X_star - s) / rho_star, rho_star


# ---------------------------------------------------------------------------
# HOCBF chain on relative states
# ---------------------------------------------------------------------------
@dataclass
class HOCBFTerms:
    """All HOCBF quantities of a batch of pairs (shapes ``(...)``).

    ``psi0 = b``, ``psi1 = 2m + alpha_1(b)``, ``beta`` of Eq. (24),
    ``closing_speed = -m / rho``.
    """

    rho: Tensor
    m: Tensor
    V2: Tensor
    b: Tensor
    closing_speed: Tensor
    c_bar: Tensor
    X: Tensor
    alpha1: Tensor
    alpha1_p: Tensor
    alpha1_pp: Tensor
    psi0: Tensor
    psi1: Tensor
    beta: Tensor
    valid: Tensor

    @property
    def V(self) -> Tensor:
        """Relative speed ``||dv||``."""
        return torch.sqrt(self.V2)

    @property
    def in_safe_set(self) -> Tensor:
        """Membership of the HOCBF safe set ``{psi0 >= 0, psi1 >= 0}``."""
        return (self.psi0 >= 0) & (self.psi1 >= 0)


def hocbf_terms(dp: Tensor, dv: Tensor, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                cfg: SafetyParams = DEFAULT_PARAMS) -> HOCBFTerms:
    """Evaluate the HOCBF chain (Eq. 23-24) for relative states ``(dp, dv)``.

    Parameters
    ----------
    dp, dv:
        Relative position ``p_i - p_j`` and velocity ``v_i - v_j``, shape ``(..., d)``.
    D:
        Collision distance ``D_ij``, float or tensor broadcastable to ``(...)``.
    """
    D_t = as_tensor(D, dp)
    rho = torch.linalg.vector_norm(dp, dim=-1)
    m = (dp * dv).sum(-1)
    V2 = (dv * dv).sum(-1)
    delta = rho - D_t
    t = _alpha_core(rho, delta, D_t, cfg)
    b = delta * (rho + D_t)
    psi1 = 2.0 * m + t.alpha1
    beta = -2.0 * V2 - 2.0 * t.alpha1_p * m - cfg.lambda_b * psi1
    rho_safe = torch.clamp(rho, min=torch.finfo(rho.dtype).tiny)
    closing = -m / rho_safe
    return HOCBFTerms(rho=rho, m=m, V2=V2, b=b, closing_speed=closing, c_bar=t.c_bar, X=t.X,
                      alpha1=t.alpha1, alpha1_p=t.alpha1_p, alpha1_pp=t.alpha1_pp,
                      psi0=b, psi1=psi1, beta=beta, valid=t.valid)


def share_slack(dp: Tensor, dv: Tensor, u: Tensor, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """Reciprocal share slack ``F_ij(x, u) = g_ij . u - beta_ij / 2`` (Eq. 28).

    ``F_ij(x, u_i) + F_ji(x, u_j) = 2 dp . (u_i - u_j) - beta`` is the slack of
    Eq. (24) at the commanded inputs.
    """
    t = hocbf_terms(dp, dv, D, cfg)
    return 2.0 * (dp * u).sum(-1) - 0.5 * t.beta


def pair_slack(dp: Tensor, dv: Tensor, du: Tensor, D: TensorLike = DEFAULT_PARAMS.D_nominal,
               cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """Pair slack ``phi~ = 2 dp . du - beta`` at relative commanded input ``du`` (App. B.2)."""
    t = hocbf_terms(dp, dv, D, cfg)
    return 2.0 * (dp * du).sum(-1) - t.beta


def spec_barrier(dp: Tensor, dv: Tensor, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                 cfg: SafetyParams = DEFAULT_PARAMS, varsigma: Optional[float] = None) -> Tensor:
    """Regularized braking-augmented specification barrier ``h^varsigma`` (Eq. 22).

    ``h = rho^2 - (D + (||dv||^2 + 2 varsigma ||dv||) / (2 a_max))^2``; with
    ``varsigma = 0`` and ``D = 2 r_s`` this is Eq. (21).
    """
    s = cfg.varsigma if varsigma is None else varsigma
    D_t = as_tensor(D, dp)
    rho2 = (dp * dp).sum(-1)
    w = torch.linalg.vector_norm(dv, dim=-1)
    stop = D_t + (w * w + 2.0 * s * w) / (2.0 * cfg.a_max)
    return rho2 - stop * stop


def pairwise_relative(pos: Tensor, vel: Tensor) -> Tuple[Tensor, Tensor]:
    """All pairwise relative states ``dp[..., i, j, :] = p_i - p_j`` (same for ``dv``).

    Parameters
    ----------
    pos, vel:
        Shape ``(..., N, d)``.

    Returns
    -------
    dp, dv:
        Shape ``(..., N, N, d)``; the diagonal is zero.
    """
    dp = pos.unsqueeze(-2) - pos.unsqueeze(-3)
    dv = vel.unsqueeze(-2) - vel.unsqueeze(-3)
    return dp, dv

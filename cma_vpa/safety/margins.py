# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Reciprocal shares with explicit sampled-data and estimation margins.

Implements Section 4.6.3 of the CMA-VPA paper:

* Remark 2   - estimation-inflated initial box,
* Lemma 13   - one-tick reachable box and one-tick decrease bound ``Lambda`` (Eq. 30),
* Lemma 12   - local estimation margins ``eps^x`` and ``eps^psi`` (Eq. 29),
* Lemma 14   - inactivity certificates (I1), (I2),
* Lemma 15   - communication radius ``R_com`` (Eq. 31),
* Eq. (27)   - the per-agent reciprocal share ``g_ij . u_i >= zeta_ij`` with
  ``eps_ij = T_c Lambda_ij + eps^x_ij + 4 rho^+_ij e_a``.

Every routine is vectorized over arbitrary leading batch dimensions. Quantities
that are undefined (a reachable box that leaves the smooth domain of
``alpha_1``) are returned as ``+inf`` margins, never as NaN, so that downstream
code can test ``torch.isfinite``.

Notation follows the paper: ``V = ||dv||``, ``rho = ||dp||``, ``m = dp . dv``,
``A_bar = 2 sqrt(d) u_inf + 2 e_a``, ``U_bar = 2 sqrt(d) u_inf``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .hocbf import TensorLike, _alpha_core, _ref, as_tensor, hocbf_terms, lambda_min
from .params import DEFAULT_PARAMS, SafetyParams

__all__ = [
    "StateBox",
    "ReachableBox",
    "EstimationMargin",
    "ShareSet",
    "point_box",
    "estimation_inflated_box",
    "one_tick_reachable_box",
    "curvature_majorant",
    "chi_bound",
    "one_tick_decrease_bound",
    "inactivity_rate_bound",
    "estimation_margin",
    "inactivity_certificates",
    "comm_radius_gamma",
    "communication_radius",
    "far_field_certified",
    "reciprocal_shares",
    "reciprocal_shares_reference",
    "check_params",
]


# ---------------------------------------------------------------------------
# boxes
# ---------------------------------------------------------------------------
@dataclass
class StateBox:
    """Box ``V in [V_lo, V_hi]``, ``rho in [rho_lo, rho_hi]``, ``m in [m_lo, m_hi]``.

    It must contain the true relative speed, distance and ``dp . dv`` at the tick.
    """

    V_lo: Tensor
    V_hi: Tensor
    rho_lo: Tensor
    rho_hi: Tensor
    m_lo: Tensor
    m_hi: Tensor


def point_box(rho: Tensor, V: Tensor, m: Tensor) -> StateBox:
    """Degenerate box for exact information (Lemma 13 as stated)."""
    return StateBox(V, V, rho, rho, m, m)


def estimation_inflated_box(rho_hat: Tensor, V_hat: Tensor, m_hat: Tensor,
                            e_p: TensorLike, e_v: TensorLike) -> StateBox:
    """Estimation-inflated box of Remark 2.

    Under ``||dp^ - dp|| <= e_p`` and ``||dv^ - dv|| <= e_v`` the true values
    lie in ``V in [V^ - e_v, V^ + e_v]``, ``rho in [rho^ - e_p, rho^ + e_p]`` and
    ``m in [m^ - dm, m^ + dm]`` with ``dm = rho^ e_v + V^ e_p + e_p e_v``.
    """
    e_p_t = as_tensor(e_p, rho_hat)
    e_v_t = as_tensor(e_v, rho_hat)
    dm = rho_hat * e_v_t + V_hat * e_p_t + e_p_t * e_v_t
    return StateBox(
        V_lo=torch.clamp(V_hat - e_v_t, min=0.0),
        V_hi=V_hat + e_v_t,
        rho_lo=rho_hat - e_p_t,
        rho_hi=rho_hat + e_p_t,
        m_lo=m_hat - dm,
        m_hi=m_hat + dm,
    )


@dataclass
class ReachableBox:
    """One-tick reachable box of Lemma 13 (all shapes ``(...)``).

    ``valid`` is ``rho^- >= rho_ext``, i.e. ``b^- >= b_ext`` (the comparison is
    made on ``rho`` so that a negative ``rho^-`` is never mistaken for a large
    ``b^-``).
    """

    V_minus: Tensor
    V_plus: Tensor
    rho_minus: Tensor
    rho_plus: Tensor
    m_minus: Tensor
    m_plus: Tensor
    b_minus: Tensor
    b_plus: Tensor
    X_minus: Tensor
    X_plus: Tensor
    alpha1_p_minus: Tensor  # alpha_1'(b^-)  (upper bound of alpha_1' on the box)
    alpha1_p_plus: Tensor   # alpha_1'(b^+)  (lower bound of alpha_1' on the box)
    valid: Tensor


def one_tick_reachable_box(box: StateBox, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                           cfg: SafetyParams = DEFAULT_PARAMS) -> ReachableBox:
    """Box that contains the relative state on ``[s_n, s_n + T_c]`` (Lemma 13, App. B.3).

    ``V^+ = V_hi + A T_c``, ``V^- = max(0, V_lo - A T_c)``,
    ``rho^+- = rho_hi/lo +- V^+ T_c``, ``m^- = m_lo - T_c rho^+ A``,
    ``m^+ = m_hi + T_c ((V^+)^2 + rho^+ A)``, ``b^+- = (rho^+-)^2 - D^2``.
    """
    A, Tc = cfg.A_bar, cfg.T_c
    D_t = as_tensor(D, box.rho_lo)
    V_plus = box.V_hi + A * Tc
    V_minus = torch.clamp(box.V_lo - A * Tc, min=0.0)
    rho_plus = box.rho_hi + V_plus * Tc
    rho_minus = box.rho_lo - V_plus * Tc
    m_minus = box.m_lo - Tc * rho_plus * A
    m_plus = box.m_hi + Tc * (V_plus * V_plus + rho_plus * A)
    tm = _alpha_core(rho_minus, rho_minus - D_t, D_t, cfg)
    tp = _alpha_core(rho_plus, rho_plus - D_t, D_t, cfg)
    b_minus = (rho_minus - D_t) * (rho_minus + D_t)
    b_plus = (rho_plus - D_t) * (rho_plus + D_t)
    return ReachableBox(V_minus=V_minus, V_plus=V_plus, rho_minus=rho_minus, rho_plus=rho_plus,
                        m_minus=m_minus, m_plus=m_plus, b_minus=b_minus, b_plus=b_plus,
                        X_minus=tm.X, X_plus=tp.X, alpha1_p_minus=tm.alpha1_p,
                        alpha1_p_plus=tp.alpha1_p, valid=tm.valid)


def curvature_majorant(reach: ReachableBox, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                       cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """``M_bar = (3/4 X^+ + k^2 / (X^-)^3) / (2 (rho^-)^3) >= |alpha_1''|`` on the box (App. B.1)."""
    D_t = as_tensor(D, reach.rho_minus)
    k = cfg.k(D_t)
    rho_m = torch.clamp(reach.rho_minus, min=torch.finfo(reach.rho_minus.dtype).tiny)
    Xm3 = reach.X_minus ** 3
    return (0.75 * reach.X_plus + k * k / Xm3) / (2.0 * rho_m ** 3)


def chi_bound(reach: ReachableBox) -> Tensor:
    """Lower bound ``chi`` of ``alpha_1'(b) m`` on the box (Lemma 13).

    ``chi = alpha_1'(b^-) m^-`` if ``m^- < 0`` and ``alpha_1'(b^+) m^-`` otherwise.
    """
    return torch.where(reach.m_minus < 0,
                       reach.alpha1_p_minus * reach.m_minus,
                       reach.alpha1_p_plus * reach.m_minus)


def one_tick_decrease_bound(reach: ReachableBox, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                            cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """One-tick decrease bound ``Lambda`` of Eq. (30).

    For frozen commanded inputs, ``phi~(s) >= phi~(s_n) - T_c Lambda`` on the
    tick. Returns ``+inf`` where ``b^- < b_ext``.
    """
    A, U, lam = cfg.A_bar, cfg.U_bar, cfg.lambda_b
    Mbar = curvature_majorant(reach, D, cfg)
    chi = chi_bound(reach)
    Vp, Vm = reach.V_plus, reach.V_minus
    m_sq = torch.maximum(reach.m_minus ** 2, reach.m_plus ** 2)
    Lam = (4.0 * Vp * A
           + 2.0 * Vp * U
           + 4.0 * Mbar * m_sq
           + 2.0 * reach.alpha1_p_minus * reach.rho_plus * A
           - 2.0 * reach.alpha1_p_plus * Vm * Vm
           - 2.0 * lam * (Vm * Vm - reach.rho_plus * A + chi))
    Lam = torch.clamp(Lam, min=0.0)
    return torch.where(reach.valid, Lam, torch.full_like(Lam, math.inf))


def _inactivity_rate_bound(reach: ReachableBox, cfg: SafetyParams) -> Tensor:
    chi = chi_bound(reach)
    Psi = 2.0 * reach.rho_plus * cfg.A_bar - 2.0 * reach.V_minus ** 2 - 2.0 * chi
    return torch.where(reach.valid, Psi, torch.full_like(Psi, math.inf))


def inactivity_rate_bound(reach: ReachableBox, cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """``Psi = 2 rho^+ A - 2 (V^-)^2 - 2 chi`` of Lemma 14 (``+inf`` where invalid).

    ``dpsi_1/ds >= -Psi`` on the tick for any admissible inputs.
    """
    return _inactivity_rate_bound(reach, cfg)


# ---------------------------------------------------------------------------
# Lemma 12: local estimation margin
# ---------------------------------------------------------------------------
@dataclass
class EstimationMargin:
    """Local estimation margins of Lemma 12.

    ``eps_x``  : ``|F(x, u) - F(x^, u)| <= eps_x / 2`` for every admissible ``u``.
    ``eps_psi``: ``|psi_1(x) - psi_1(x^)| <= eps_psi``.
    Both are ``+inf`` where ``rho^ - e_p < rho_ext``.
    """

    eps_x: Tensor
    eps_psi: Tensor
    G_p: Tensor
    G_v: Tensor
    valid: Tensor


def estimation_margin(rho_hat: Tensor, V_hat: Tensor, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                      cfg: SafetyParams = DEFAULT_PARAMS, e_p: Optional[TensorLike] = None,
                      e_v: Optional[TensorLike] = None) -> EstimationMargin:
    """Local estimation margins ``eps^x_ij`` and ``eps^psi_ij`` (Lemma 12, Eq. 29).

    With ``rho_lo = rho^ - e_p``, ``rho_hi = rho^ + e_p``, ``V_hi = V^ + e_v``,
    ``b_lo = rho_lo^2 - D^2``::

        M_x = (3/4 X(rho_hi) + k^2 / X(rho_lo)^3) / (2 rho_lo^3)
        G_v = 2 V_hi + (alpha_1'(b_lo) + lambda_b) rho_hi
        G_p = 2 sqrt(d) u_inf + 2 M_x rho_hi^2 V_hi + (alpha_1'(b_lo) + lambda_b) V_hi
              + lambda_b alpha_1'(b_lo) rho_hi
        eps_x   = 2 (e_p G_p + e_v G_v)
        eps_psi = 2 e_p (V_hi + alpha_1'(b_lo) rho_hi) + 2 e_v rho_hi
    """
    e_p = cfg.e_p if e_p is None else e_p
    e_v = cfg.e_v if e_v is None else e_v
    e_p_t = as_tensor(e_p, rho_hat)
    e_v_t = as_tensor(e_v, rho_hat)
    D_t = as_tensor(D, rho_hat)
    lam = cfg.lambda_b
    rho_lo = rho_hat - e_p_t
    rho_hi = rho_hat + e_p_t
    V_hi = V_hat + e_v_t
    t_lo = _alpha_core(rho_lo, rho_lo - D_t, D_t, cfg)
    t_hi = _alpha_core(rho_hi, rho_hi - D_t, D_t, cfg)
    k = cfg.k(D_t)
    rho_lo_s = torch.clamp(rho_lo, min=torch.finfo(rho_lo.dtype).tiny)
    M_x = (0.75 * t_hi.X + k * k / t_lo.X ** 3) / (2.0 * rho_lo_s ** 3)
    a1p = t_lo.alpha1_p
    G_v = 2.0 * V_hi + (a1p + lam) * rho_hi
    G_p = (2.0 * math.sqrt(cfg.d) * cfg.u_inf
           + 2.0 * M_x * rho_hi * rho_hi * V_hi
           + (a1p + lam) * V_hi
           + lam * a1p * rho_hi)
    eps_x = 2.0 * (e_p_t * G_p + e_v_t * G_v)
    eps_psi = 2.0 * e_p_t * (V_hi + a1p * rho_hi) + 2.0 * e_v_t * rho_hi
    inf = torch.full_like(eps_x, math.inf)
    valid = t_lo.valid
    return EstimationMargin(eps_x=torch.where(valid, eps_x, inf),
                            eps_psi=torch.where(valid, eps_psi, inf),
                            G_p=G_p, G_v=G_v, valid=valid)


# ---------------------------------------------------------------------------
# Lemma 14: inactivity certificates
# ---------------------------------------------------------------------------
def inactivity_certificates(reach: ReachableBox, psi1_lower: Tensor,
                            cfg: SafetyParams = DEFAULT_PARAMS) -> Tuple[Tensor, Tensor]:
    """Inactivity certificates (I1), (I2) of Lemma 14.

    Parameters
    ----------
    reach:
        One-tick reachable box (on the inflated box when states are estimated).
    psi1_lower:
        A lower bound of ``psi_1(s_n)``: ``psi_1`` itself with exact information,
        ``psi1^ - eps^psi`` with estimated states (Lemma 12).

    Returns
    -------
    (I1, I2):
        ``I1 = m^- >= 0``; ``I2 = psi1_lower >= T_c max(0, Psi)``.
    """
    I1 = reach.m_minus >= 0
    Psi = _inactivity_rate_bound(reach, cfg)
    I2 = psi1_lower >= cfg.T_c * torch.clamp(Psi, min=0.0)
    return I1, I2


# ---------------------------------------------------------------------------
# Lemma 15: communication radius
# ---------------------------------------------------------------------------
def comm_radius_gamma(R: TensorLike, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                      cfg: SafetyParams = DEFAULT_PARAMS,
                      V_max: Optional[float] = None) -> Tensor:
    """``Gamma(R)`` of Eq. (31); ``-inf`` where ``R - V^+_max T_c < rho_ext``.

    ``Gamma(R) = cbar(R) - V_max - T_c (1 + V^+ T_c / R)
    (A + alpha_1'((R - V^+ T_c)^2 - D^2) V^+)`` with ``V^+ = V_max + A T_c``.
    """
    Vm = cfg.V_max if V_max is None else V_max
    A, Tc = cfg.A_bar, cfg.T_c
    ref = _ref(R, D)
    R_t = as_tensor(R, ref)
    D_t = as_tensor(D, ref)
    Vp = Vm + A * Tc
    rho_in = R_t - Vp * Tc
    tR = _alpha_core(R_t, R_t - D_t, D_t, cfg)
    tin = _alpha_core(rho_in, rho_in - D_t, D_t, cfg)
    gamma = tR.c_bar - Vm - Tc * (1.0 + Vp * Tc / R_t) * (A + tin.alpha1_p * Vp)
    return torch.where(tin.valid & (R_t > D_t), gamma, torch.full_like(gamma, -math.inf))


@functools.lru_cache(maxsize=256)
def _communication_radius_cached(D: float, cfg: SafetyParams, V_max: Optional[float],
                                 iters: int) -> float:
    A, Tc = cfg.A_bar, cfg.T_c
    Vm = cfg.V_max if V_max is None else V_max
    Vp = Vm + A * Tc
    f64 = torch.float64
    lo = torch.tensor(max(D, cfg.rho_ext(D) + Vp * Tc) + 1e-9, dtype=f64)
    hi = torch.tensor(max(2.0 * float(lo), 1.0), dtype=f64)
    g = lambda r: comm_radius_gamma(r, D, cfg, V_max)  # noqa: E731
    if float(g(lo)) >= 0.0:
        return float(lo)
    while float(g(hi)) < 0.0:
        hi = hi * 2.0
        if float(hi) > 1e7:
            raise RuntimeError("communication radius did not bracket; check parameters")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(g(mid)) >= 0.0:
            hi = mid
        else:
            lo = mid
    return float(hi)  # upper end of the bracket: Gamma(hi) >= 0 (sound)


def communication_radius(D: float = DEFAULT_PARAMS.D_nominal, cfg: SafetyParams = DEFAULT_PARAMS,
                         V_max: Optional[float] = None, iters: int = 200) -> float:
    """Smallest radius ``R_com`` with ``Gamma(R_com) >= 0`` (Lemma 15), by bisection.

    ``Gamma`` is strictly increasing, so the returned upper end of the final
    bracket satisfies ``Gamma(R_com) >= 0`` exactly. For Table 4 and ``D = 1.2``
    this is ``29.864 m``; the paper rounds it to ``29.86 m``.
    """
    return _communication_radius_cached(float(D), cfg, None if V_max is None else float(V_max),
                                        int(iters))


def far_field_certified(rho_lo: Tensor, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                        cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """Pairs certified by Lemma 15 alone: ``Gamma(rho_lo; D_ij) >= 0``.

    ``rho_lo`` must lower-bound the true distance (``rho^ - e_p`` with estimated
    states). Because ``Gamma`` is strictly increasing, this is equivalent to
    ``rho >= R_com(D_ij)`` and handles per-pair collision distances exactly.
    Relies on Assumption 1 (``||dv|| <= V_max``) for the true state.
    """
    return comm_radius_gamma(rho_lo, D, cfg) >= 0.0


# ---------------------------------------------------------------------------
# Eq. (27): reciprocal shares
# ---------------------------------------------------------------------------
@dataclass
class ShareSet:
    """Per-agent reciprocal shares and their certificates (Eq. 27, Lemmas 12-15).

    Shapes: per-pair quantities ``(..., N)``, ``g`` is ``(..., N, d)``.

    Attributes
    ----------
    g, zeta:
        Constraint ``g . u_i >= zeta`` with ``g = 2 dp^`` and
        ``zeta = (beta^ + eps) / 2``. Entries of non-active pairs are zeroed
        (``g = 0``, ``zeta = 0``) so that they are trivially satisfied.
    active:
        Pairs that must enter the Stage-B QP.
    ill_posed:
        Active pairs whose margin is infinite (the reachable box leaves the
        domain of ``alpha_1``). Algorithm 2 returns *infeasible* for these, and
        the executor must brake.
    inactive_I1, inactive_I2, far:
        Which certificate removed the pair (Lemma 14 (I1), (I2), Lemma 15).
    eps, eps_dt, eps_x, eps_psi, Lambda, rho_plus, beta, psi1, psi0:
        Margin components and HOCBF quantities at the estimate.
    Psi:
        One-tick decrease rate bound of ``psi_1`` (Lemma 14) on the inflated box
        (``+inf`` where undefined).
    admission:
        Radius-admission condition of Section 4.6.6 for this pair at this ``D``,
        evaluated conservatively on the estimate: ``rho^ - e_p >= D`` (so
        ``psi_0 >= 0`` at the true state) and ``psi1^ - eps^psi >= T_c max(0, Psi)``.
    """

    g: Tensor
    zeta: Tensor
    active: Tensor
    ill_posed: Tensor
    inactive_I1: Tensor
    inactive_I2: Tensor
    far: Tensor
    eps: Tensor
    eps_dt: Tensor
    eps_x: Tensor
    eps_psi: Tensor
    Lambda: Tensor
    rho_plus: Tensor
    beta: Tensor
    psi1: Tensor
    psi0: Tensor
    Psi: Tensor
    admission: Tensor


def reciprocal_shares(dp_hat: Tensor, dv_hat: Tensor, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                      cfg: SafetyParams = DEFAULT_PARAMS, heard: Optional[Tensor] = None,
                      e_p: Optional[TensorLike] = None,
                      e_v: Optional[TensorLike] = None) -> ShareSet:
    """Build agent ``i``'s Stage-B constraints from its relative-state estimates.

    This is lines 20-28 of Algorithm 2 (KNOTPROJECT), vectorized over
    neighbours and over any batch dimensions. It computes exactly the same
    quantities as :func:`reciprocal_shares_reference` (the composition of the
    lemma-level functions above) but evaluates the class-K chain at all six
    distances it needs (``rho^``, ``rho^ -+ e_p``, ``rho^-+``, and
    ``rho^ - e_p - V^+_max T_c`` for Lemma 15) in one batched call, and has no
    data-dependent Python control flow, so it can be wrapped by ``torch.compile``
    for fixed neighbour-table shapes.

    Parameters
    ----------
    dp_hat, dv_hat:
        Agent ``i``'s estimates of ``p_i - p_j`` and ``v_i - v_j``, ``(..., N, d)``.
    D:
        Collision distances ``D_ij = r_i,eff + r_j,eff`` (float or ``(..., N)``).
        Both agents of a pair must use the same value (Assumption 4).
    heard:
        Boolean ``(..., N)``: neighbour broadcast received and fresh within
        ``R_link``. Pairs not heard are certified by Lemma 15 (Assumption 2).
    e_p, e_v:
        Estimation bounds; ``0`` gives the exact-information margins.
    """
    e_p = cfg.e_p if e_p is None else e_p
    e_v = cfg.e_v if e_v is None else e_v
    A, U, Tc, lam = cfg.A_bar, cfg.U_bar, cfg.T_c, cfg.lambda_b
    rho = torch.linalg.vector_norm(dp_hat, dim=-1)
    m = (dp_hat * dv_hat).sum(-1)
    V2 = (dv_hat * dv_hat).sum(-1)
    V = torch.sqrt(V2)
    D_t = as_tensor(D, dp_hat)
    shape = torch.broadcast_shapes(rho.shape, D_t.shape)
    rho, m, V2, V, D_t = (t.expand(shape) for t in (rho, m, V2, V, D_t))
    e_p_t = as_tensor(e_p, rho)
    e_v_t = as_tensor(e_v, rho)
    if heard is None:
        heard = torch.ones(shape, dtype=torch.bool, device=rho.device)

    # Remark 2 (estimation-inflated box) and Lemma 13 (one-tick reachable box)
    dm = rho * e_v_t + V * e_p_t + e_p_t * e_v_t
    rho_lo, rho_hi = rho - e_p_t, rho + e_p_t
    V_lo, V_hi = torch.clamp(V - e_v_t, min=0.0), V + e_v_t
    V_plus = V_hi + A * Tc
    V_minus = torch.clamp(V_lo - A * Tc, min=0.0)
    rho_plus = rho_hi + V_plus * Tc
    rho_minus = rho_lo - V_plus * Tc
    m_minus = (m - dm) - Tc * rho_plus * A
    m_plus = (m + dm) + Tc * (V_plus * V_plus + rho_plus * A)
    Vp_max = cfg.V_max + A * Tc
    rho_far = rho_lo - Vp_max * Tc

    # all six evaluations of the class-K chain in one call
    R = torch.stack([rho, rho_lo, rho_hi, rho_minus, rho_plus, rho_far])
    t = _alpha_core(R, R - D_t, D_t, cfg)
    X, cbar, a1, a1p, valid = t.X, t.c_bar, t.alpha1, t.alpha1_p, t.valid
    k = cfg.k(D_t)
    k2 = k * k
    tiny = torch.finfo(rho.dtype).tiny
    inf = torch.full_like(rho, math.inf)

    # HOCBF at the estimate (Eq. 23-24)
    psi0 = (rho - D_t) * (rho + D_t)
    psi1 = 2.0 * m + a1[0]
    beta = -2.0 * V2 - 2.0 * a1p[0] * m - lam * psi1

    # Lemma 12 (estimation margins)
    rho_lo_s = torch.clamp(rho_lo, min=tiny)
    M_x = (0.75 * X[2] + k2 / X[1] ** 3) / (2.0 * rho_lo_s ** 3)
    G_v = 2.0 * V_hi + (a1p[1] + lam) * rho_hi
    G_p = (2.0 * math.sqrt(cfg.d) * cfg.u_inf + 2.0 * M_x * rho_hi * rho_hi * V_hi
           + (a1p[1] + lam) * V_hi + lam * a1p[1] * rho_hi)
    eps_x = torch.where(valid[1], 2.0 * (e_p_t * G_p + e_v_t * G_v), inf)
    eps_psi = torch.where(valid[1], 2.0 * e_p_t * (V_hi + a1p[1] * rho_hi) + 2.0 * e_v_t * rho_hi, inf)

    # Lemma 13 (one-tick decrease bound) and Lemma 14 (inactivity certificates)
    rho_m_s = torch.clamp(rho_minus, min=tiny)
    Mbar = (0.75 * X[4] + k2 / X[3] ** 3) / (2.0 * rho_m_s ** 3)
    chi = torch.where(m_minus < 0, a1p[3] * m_minus, a1p[4] * m_minus)
    Vm2 = V_minus * V_minus
    Lam = (4.0 * V_plus * A + 2.0 * V_plus * U
           + 4.0 * Mbar * torch.maximum(m_minus * m_minus, m_plus * m_plus)
           + 2.0 * a1p[3] * rho_plus * A - 2.0 * a1p[4] * Vm2
           - 2.0 * lam * (Vm2 - rho_plus * A + chi))
    Lam = torch.where(valid[3], torch.clamp(Lam, min=0.0), inf)
    Psi = torch.where(valid[3], 2.0 * rho_plus * A - 2.0 * Vm2 - 2.0 * chi, inf)
    I1 = m_minus >= 0
    I2 = (psi1 - eps_psi) >= Tc * torch.clamp(Psi, min=0.0)

    # Lemma 15 (communication radius) at R = rho^ - e_p
    gamma = cbar[1] - cfg.V_max - Tc * (1.0 + Vp_max * Tc / rho_lo_s) * (A + a1p[5] * Vp_max)
    far = valid[5] & (rho_lo > D_t) & (gamma >= 0.0)

    certified = I1 | I2 | far
    active = heard & ~certified
    eps_dt = Tc * Lam
    eps = eps_dt + eps_x + 4.0 * rho_plus * cfg.e_a
    ill_posed = active & ~torch.isfinite(eps)
    g = torch.where(active.unsqueeze(-1), 2.0 * dp_hat, torch.zeros_like(dp_hat))
    zeta = torch.where(active, 0.5 * (beta + eps), torch.zeros_like(beta))
    admission = (rho_lo >= D_t) & I2
    return ShareSet(g=g, zeta=zeta, active=active, ill_posed=ill_posed,
                    inactive_I1=heard & I1, inactive_I2=heard & I2 & ~I1,
                    far=heard & far & ~I1 & ~I2, eps=eps, eps_dt=eps_dt, eps_x=eps_x,
                    eps_psi=eps_psi, Lambda=Lam, rho_plus=rho_plus, beta=beta,
                    psi1=psi1, psi0=psi0, Psi=Psi, admission=admission)


def reciprocal_shares_reference(dp_hat: Tensor, dv_hat: Tensor,
                                D: TensorLike = DEFAULT_PARAMS.D_nominal,
                                cfg: SafetyParams = DEFAULT_PARAMS, heard: Optional[Tensor] = None,
                                e_p: Optional[TensorLike] = None,
                                e_v: Optional[TensorLike] = None) -> ShareSet:
    """Reference composition of the lemma-level functions (Phase-1 implementation).

    Mathematically identical to :func:`reciprocal_shares`; kept as a readable,
    lemma-by-lemma specification and as the oracle in the tests.

    Parameters
    ----------
    dp_hat, dv_hat:
        Agent ``i``'s estimates of ``p_i - p_j`` and ``v_i - v_j``, ``(..., N, d)``.
    D:
        Collision distances ``D_ij = r_i,eff + r_j,eff`` (float or ``(..., N)``).
        Both agents of a pair must use the same value (Assumption 4).
    heard:
        Boolean ``(..., N)``: neighbour broadcast received and fresh within
        ``R_link``. Pairs not heard are certified by Lemma 15 (Assumption 2).
    e_p, e_v:
        Estimation bounds; ``0`` gives the exact-information margins.
    """
    e_p = cfg.e_p if e_p is None else e_p
    e_v = cfg.e_v if e_v is None else e_v
    D_t = as_tensor(D, dp_hat)
    t = hocbf_terms(dp_hat, dv_hat, D_t, cfg)
    rho_hat, m_hat = t.rho, t.m
    V_hat = torch.sqrt(t.V2)
    if heard is None:
        heard = torch.ones_like(rho_hat, dtype=torch.bool)

    box = estimation_inflated_box(rho_hat, V_hat, m_hat, e_p, e_v)
    reach = one_tick_reachable_box(box, D_t, cfg)
    est = estimation_margin(rho_hat, V_hat, D_t, cfg, e_p, e_v)

    I1, I2 = inactivity_certificates(reach, t.psi1 - est.eps_psi, cfg)
    far = far_field_certified(box.rho_lo, D_t, cfg)
    certified = I1 | I2 | far
    active = heard & ~certified

    Lam = one_tick_decrease_bound(reach, D_t, cfg)
    eps_dt = cfg.T_c * Lam
    eps = eps_dt + est.eps_x + 4.0 * reach.rho_plus * cfg.e_a
    ill_posed = active & ~torch.isfinite(eps)

    g = 2.0 * dp_hat
    zeta = 0.5 * (t.beta + eps)
    zero = torch.zeros_like(zeta)
    g = torch.where(active.unsqueeze(-1), g, torch.zeros_like(g))
    zeta = torch.where(active, zeta, zero)
    Psi = inactivity_rate_bound(reach, cfg)
    admission = (box.rho_lo >= D_t) & I2
    return ShareSet(g=g, zeta=zeta, active=active, ill_posed=ill_posed,
                    inactive_I1=heard & I1, inactive_I2=heard & I2 & ~I1,
                    far=heard & far & ~I1 & ~I2, eps=eps, eps_dt=eps_dt, eps_x=est.eps_x,
                    eps_psi=est.eps_psi, Lambda=Lam, rho_plus=reach.rho_plus, beta=t.beta,
                    psi1=t.psi1, psi0=t.psi0, Psi=Psi, admission=admission)


# ---------------------------------------------------------------------------
# consistency checks of a parameter set
# ---------------------------------------------------------------------------
def check_params(cfg: SafetyParams = DEFAULT_PARAMS) -> List[str]:
    """Return the standing conditions of Section 4.6 that ``cfg`` violates.

    Checked for every collision distance ``D in [2 r_s, 2 r_max]`` that Eq. (35)
    can produce:

    * Lemma 8:  ``varsigma^2 < a_max D``;
    * Lemma 11: ``lambda_b >= lambda_min(D)``;
    * Assumption 2 / Lemma 15: ``R_link >= R_com(D)``;
    * Corollary 6: ``u_inf >= a_max / 2 + e_a`` (strictly, for pairwise feasibility).
    """
    issues: List[str] = []
    for D in (cfg.D_nominal, cfg.D_max):
        if not cfg.varsigma ** 2 < cfg.a_max * D:
            issues.append(f"Lemma 8: varsigma^2 = {cfg.varsigma**2:.4g} >= a_max D = {cfg.a_max*D:.4g}")
        lam_min, _ = lambda_min(D, cfg)
        if cfg.lambda_b < float(lam_min):
            issues.append(f"Lemma 11: lambda_b = {cfg.lambda_b} < lambda_min(D={D}) = {float(lam_min):.4f}")
        R_com = communication_radius(D, cfg)
        if cfg.R_link < R_com:
            issues.append(f"Lemma 15: R_link = {cfg.R_link} m < R_com(D={D:.2f} m) = {R_com:.3f} m; "
                          "pairs that are not heard are not certified at this collision distance")
    if not cfg.u_inf > cfg.a_max / 2.0 + cfg.e_a:
        issues.append(f"Corollary 6: u_inf = {cfg.u_inf} <= a_max/2 + e_a = {cfg.a_max/2 + cfg.e_a}")
    return issues

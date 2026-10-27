# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Real-time quadratic-programming solvers of CBF-DeGG-Flow.

* :func:`project_box_halfspace` - exact closed form of Lemma 18 for one
  half-space and a box, ``O(d log d)`` by sorting the ``<= 2d`` breakpoints of
  the piecewise-linear map ``lambda -> g . clip(u^ + lambda g, l, h)``.
* :func:`hildreth` - Hildreth's dual coordinate ascent (Algorithm 2, lines
  11-17) for ``min ||x - y||_W^2  s.t.  G x <= h`` with a diagonal or dense
  weight, warm start, row masks, an iteration cap, convergence detection, row
  *groups* (see below) and an exact active-set polish. Used by Stage A
  (polyhedron ``P_t`` of the chunk) and by Stage B with several active
  neighbours.
* :func:`enumerate_small_qp` - exact active-set enumeration for small
  dimensions; the Stage-B backstop when Hildreth stops at its cap without a
  KKT certificate (nearly antiparallel constraints).
* :func:`solve_stage_b_qp` - the Stage-B knot projection of Eq. (34): routes
  each batch element to the closed form (one active constraint) or to Hildreth
  (two or more), returns warm-start multipliers, and reports *exact* primal
  feasibility of the returned input against the original constraints.
* :func:`stage_b_fixed` - a branch-free, fixed-shape version of the same
  projection for ``torch.compile`` (used by the real-time executor).

Grouped (coloured) Hildreth
---------------------------
Updating several rows whose gradients have disjoint support in the
``W^{-1}`` metric (``G_r W^{-1} G_s^T = 0``) simultaneously is *identical* to
updating them one after the other, because no update changes the residual of
another row in the group. A partition of the rows into such groups therefore
gives exactly Hildreth's iteration (for one row order) with one vectorized step
per group instead of one Python step per row. Stage A's rows touch at most two
consecutive knots, so ``O(H (N_c + 12))`` rows collapse into ``O(N_c)`` groups
(:func:`cma_vpa.safety.degg_flow.build_stage_a_polytope` builds them
structurally; :func:`greedy_row_coloring` builds them for any ``G``).

Soundness of the feasibility flag
---------------------------------
The Stage-B solvers are run on constraints tightened by a tolerance ``tau``
proportional to the floating-point resolution, the result is clamped to the
box, and the returned ``ok`` flag is an exact check ``g_j . u >= zeta_j`` of the
*original* constraints. ``ok = True`` therefore certifies the executed input
irrespective of solver accuracy; any numerical shortfall produces ``ok = False``
and the executor's fallback (Section 4.7), never an unsafe command.

Compile-friendliness
--------------------
:func:`project_box_halfspace`, :func:`_hildreth_sweeps`,
:func:`_goldfarb_idnani` with ``early_exit=False`` and :func:`stage_b_fixed`
contain no data-dependent Python control flow and no boolean-mask indexing, so
for fixed shapes they can be wrapped by ``torch.compile``. The drivers
(:func:`hildreth`, :func:`solve_stage_b_qp`) add early exits and routing on top
and stay eager.

All solvers are batched over leading dimensions and preserve dtype and device.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    "BoxHalfspaceResult",
    "HildrethResult",
    "StageBResult",
    "single_constraint_feasible",
    "project_box_halfspace",
    "hildreth",
    "greedy_row_coloring",
    "groups_conflict_free",
    "enumerate_small_qp",
    "stage_b_rows",
    "stage_b_groups",
    "solve_stage_b_qp",
    "stage_b_fixed",
]

GroupsLike = Optional[Sequence[Union[Sequence[int], Tensor]]]


# ---------------------------------------------------------------------------
# Lemma 18: one half-space and a box
# ---------------------------------------------------------------------------
@dataclass
class BoxHalfspaceResult:
    """Solution of ``min ||u - u^||^2 s.t. g . u >= zeta, l <= u <= h``.

    Attributes
    ----------
    u:
        Minimizer ``clip(u^ + lambda* g, l, h)``; for infeasible problems the
        limit ``lim_{lambda -> inf} u(lambda)`` (the maximizer of ``g . u``).
    lam:
        Multiplier ``lambda*`` of the half-space (``+inf`` if infeasible).
    lam_upper, lam_lower:
        KKT multipliers of ``u <= h`` and ``-u <= -l`` (warm start for Hildreth).
    feasible:
        ``sup_{l <= u <= h} g . u >= zeta``.
    sup_value:
        ``sup_{l <= u <= h} g . u`` (Corollary 4: ``u_inf ||g||_1`` for a symmetric box).
    """

    u: Tensor
    lam: Tensor
    lam_upper: Tensor
    lam_lower: Tensor
    feasible: Tensor
    sup_value: Tensor


def single_constraint_feasible(g: Tensor, zeta: Tensor, u_inf: float) -> Tensor:
    """Corollary 4: ``g . u >= zeta`` is feasible on ``[-u_inf, u_inf]^d`` iff ``zeta <= u_inf ||g||_1``."""
    return zeta <= u_inf * g.abs().sum(-1)


def project_box_halfspace(u_hat: Tensor, g: Tensor, zeta: Tensor, lower: Tensor | float,
                          upper: Tensor | float) -> BoxHalfspaceResult:
    """Exact closed form of Lemma 18 (vectorized, branch-free).

    Stationarity gives ``u(lambda) = clip(u^ + lambda g, l, h)``, and
    ``f(lambda) = g . u(lambda)`` is continuous, non-decreasing and piecewise
    linear. Coordinate ``l`` is free (strictly inside its bounds) on the
    interval ``(enter_l, exit_l)`` with ``enter = (l - u^)/g, exit = (h - u^)/g``
    for ``g > 0`` and the reverse for ``g < 0``; on a free interval it adds
    ``g_l^2`` to the slope. Sorting the ``2d`` interval end points and
    accumulating slopes gives ``f`` at every breakpoint; ``lambda*`` is the
    smallest root of ``f(lambda) = zeta`` on ``[0, inf)``.

    Parameters
    ----------
    u_hat, g:
        ``(..., d)``.
    zeta:
        ``(...)``.
    lower, upper:
        Box bounds, floats or tensors broadcastable to ``(..., d)``; ``lower < upper``.
    """
    dt, dev = u_hat.dtype, u_hat.device
    lo = torch.as_tensor(lower, dtype=dt, device=dev).expand_as(u_hat)
    hi = torch.as_tensor(upper, dtype=dt, device=dev).expand_as(u_hat)
    zeta = torch.as_tensor(zeta, dtype=dt, device=dev)

    u0 = torch.minimum(torch.maximum(u_hat, lo), hi)
    f0 = (g * u0).sum(-1)
    pos = g > 0
    neg = g < 0
    nz = pos | neg
    g_safe = torch.where(nz, g, torch.ones_like(g))
    t_lo = (lo - u_hat) / g_safe
    t_hi = (hi - u_hat) / g_safe
    enter = torch.where(pos, t_lo, t_hi)
    leave = torch.where(pos, t_hi, t_lo)
    inf = torch.full_like(enter, math.inf)
    enter = torch.where(nz, enter, inf)
    leave = torch.where(nz, leave, inf)
    g2 = g * g

    # slope of f on (0, first breakpoint)
    free0 = nz & (enter <= 0) & (leave > 0)
    s0 = torch.where(free0, g2, torch.zeros_like(g2)).sum(-1)

    ev_lam = torch.cat([torch.where(nz & (enter > 0), enter, inf),
                        torch.where(nz & (leave > 0), leave, inf)], dim=-1)
    ev_dslope = torch.cat([g2, -g2], dim=-1)
    lam_sorted, order = torch.sort(ev_lam, dim=-1)
    dslope = torch.gather(ev_dslope, -1, order)
    fin = torch.isfinite(lam_sorted)
    dslope = torch.where(fin, dslope, torch.zeros_like(dslope))

    zeros = torch.zeros_like(lam_sorted[..., :1])
    lam_f = torch.where(fin, lam_sorted, torch.zeros_like(lam_sorted))
    prev = torch.cat([zeros, lam_f[..., :-1]], dim=-1)
    seg_len = torch.where(fin, lam_sorted - prev, torch.zeros_like(lam_sorted))
    # slope on the segment that ends at breakpoint k
    slope = s0.unsqueeze(-1) + torch.cat([zeros, torch.cumsum(dslope, -1)[..., :-1]], dim=-1)
    f_at = f0.unsqueeze(-1) + torch.cumsum(slope * seg_len, dim=-1)

    sup_value = torch.where(pos, g * hi, torch.where(neg, g * lo, torch.zeros_like(g))).sum(-1)
    already = f0 >= zeta
    hit = fin & (f_at >= zeta.unsqueeze(-1))
    any_hit = hit.any(-1)
    k = torch.argmax(hit.to(torch.int8), dim=-1, keepdim=True)
    f_k = torch.gather(f_at, -1, k).squeeze(-1)
    s_k = torch.gather(slope, -1, k).squeeze(-1)
    len_k = torch.gather(seg_len, -1, k).squeeze(-1)
    prev_k = torch.gather(prev, -1, k).squeeze(-1)
    f_prev = f_k - s_k * len_k
    s_safe = torch.where(s_k > 0, s_k, torch.ones_like(s_k))
    lam_star = prev_k + torch.clamp((zeta - f_prev) / s_safe, min=0.0)
    lam_star = torch.minimum(lam_star, prev_k + len_k)  # stay on the segment

    feasible = already | any_hit
    lam_star = torch.where(already, torch.zeros_like(lam_star), lam_star)
    lam_star = torch.where(feasible, lam_star, torch.full_like(lam_star, math.inf))

    lam_fin = torch.where(feasible, lam_star, torch.zeros_like(lam_star))
    raw = u_hat + lam_fin.unsqueeze(-1) * g
    u_star = torch.minimum(torch.maximum(raw, lo), hi)
    u_sup = torch.where(pos, hi, torch.where(neg, lo, u0))
    u_star = torch.where(feasible.unsqueeze(-1), u_star, u_sup)
    lam_up = torch.where(feasible.unsqueeze(-1), torch.clamp(raw - hi, min=0.0), torch.zeros_like(raw))
    lam_lo = torch.where(feasible.unsqueeze(-1), torch.clamp(lo - raw, min=0.0), torch.zeros_like(raw))
    return BoxHalfspaceResult(u=u_star, lam=lam_star, lam_upper=lam_up, lam_lower=lam_lo,
                              feasible=feasible, sup_value=sup_value)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _apply_winv(W_inv: Optional[Tensor], M: Tensor) -> Tensor:
    """Return ``W^{-1} M`` for ``M`` of shape ``(..., n, k)``."""
    if W_inv is None:
        return M
    if W_inv.dim() >= 2 and W_inv.shape[-1] == W_inv.shape[-2] == M.shape[-2] and W_inv.dim() == M.dim():
        return W_inv @ M
    return W_inv.unsqueeze(-1) * M  # diagonal weight given as (..., n)


def _groups_key(groups: GroupsLike, m: int) -> Tuple[Tuple[int, ...], ...]:
    if groups is None:
        return tuple((r,) for r in range(m))
    return tuple(tuple(int(r) for r in (g.tolist() if isinstance(g, Tensor) else g)) for g in groups)


@functools.lru_cache(maxsize=512)
def _pack_groups(key: Tuple[Tuple[int, ...], ...], m: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Pad row groups to ``(n_groups, g_max)``; ``pos[r]`` is row ``r``'s flat slot."""
    ng = len(key)
    gmax = max((len(g) for g in key), default=1) or 1
    gidx = torch.zeros(ng, gmax, dtype=torch.long)
    gval = torch.zeros(ng, gmax, dtype=torch.bool)
    pos = torch.full((m,), -1, dtype=torch.long)
    count = 0
    for gi, rows in enumerate(key):
        for j, r in enumerate(rows):
            if not 0 <= r < m or pos[r] >= 0:
                raise ValueError("row groups must partition range(m)")
            gidx[gi, j] = r
            gval[gi, j] = True
            pos[r] = gi * gmax + j
            count += 1
    if count != m:
        raise ValueError(f"row groups cover {count} of {m} rows")
    return gidx, gval, pos


def greedy_row_coloring(G: Tensor, W_inv: Optional[Tensor] = None, tol: float = 0.0) -> List[Tensor]:
    """Partition the rows of ``G`` into groups with pairwise ``G_r W^-1 G_s^T = 0``.

    The conflict pattern is the union over batch elements. Greedy colouring in
    row order; returns a list of ``LongTensor`` row indices usable as
    ``groups`` in :func:`hildreth`.
    """
    WG = _apply_winv(W_inv, G.transpose(-1, -2))
    Q = (G @ WG).abs()
    conflict = (Q > tol).reshape(-1, Q.shape[-2], Q.shape[-1]).any(0).cpu()
    m = conflict.shape[0]
    color = torch.full((m,), -1, dtype=torch.long)
    for r in range(m):
        nb = conflict[r].clone()
        nb[r] = False
        used = set(color[nb & (color >= 0)].tolist())
        c = 0
        while c in used:
            c += 1
        color[r] = c
    return [torch.nonzero(color == c).flatten() for c in range(int(color.max()) + 1)] if m else []


def groups_conflict_free(G: Tensor, groups: GroupsLike, W_inv: Optional[Tensor] = None,
                         tol: float = 0.0) -> bool:
    """Check that no two rows of a group interact (``|G_r W^-1 G_s^T| <= tol``)."""
    WG = _apply_winv(W_inv, G.transpose(-1, -2))
    Q = (G @ WG).abs().reshape(-1, G.shape[-2], G.shape[-2]).amax(0)
    for rows in _groups_key(groups, G.shape[-2]):
        if len(rows) > 1:
            idx = torch.tensor(rows, device=Q.device)
            sub = Q[idx][:, idx].clone()
            sub.fill_diagonal_(0.0)
            if float(sub.max()) > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# Hildreth's dual coordinate ascent
# ---------------------------------------------------------------------------
@dataclass
class HildrethResult:
    """Output of :func:`hildreth`.

    ``x`` is the primal iterate ``y - W^{-1} G^T lam``; ``max_violation`` is
    ``max_r (G_r x - h_r)_+`` over unmasked rows; ``feasible`` compares it with
    ``feas_tol (1 + |h_r|)``; ``converged`` additionally requires the last block
    of sweeps to have changed the multipliers by less than ``tol`` (or an
    accepted polish); ``exact`` marks elements whose ``x`` carries a KKT
    certificate from the active-set polish.
    """

    x: Tensor
    lam: Tensor
    sweeps: int
    converged: Tensor
    feasible: Tensor
    max_violation: Tensor
    exact: Tensor


def _hildreth_sweeps(x: Tensor, lam: List[Tensor], Gg: Tensor, WGg: Tensor, hg: Tensor,
                     qinv: Tensor, live: Tensor, n_sweeps: int) -> Tuple[Tensor, List[Tensor]]:
    """``n_sweeps`` grouped Gauss-Seidel sweeps of Hildreth's method (pure, compile-friendly).

    ``Gg (ng, ..., g, n)``, ``WGg (ng, ..., n, g)``, ``hg, qinv (ng, ..., g)``;
    ``qinv`` is ``1 / (G_r W^-1 G_r^T)`` for live rows and 0 for masked or
    padded rows (whose multiplier therefore never moves); ``live (..., 1)``
    freezes converged batch elements.
    """
    lam = list(lam)
    for _ in range(n_sweeps):
        for gi in range(len(lam)):
            viol = (Gg[gi] @ x.unsqueeze(-1)).squeeze(-1) - hg[gi]
            new = torch.clamp(lam[gi] + viol * qinv[gi], min=0.0)
            dl = (new - lam[gi]) * live
            x = x - (WGg[gi] @ dl.unsqueeze(-1)).squeeze(-1)
            lam[gi] = lam[gi] + dl
    return x, lam


_COMPILED_SWEEP = None


def _one_sweep(x: Tensor, lam: List[Tensor], Gg: Tensor, WGg: Tensor, hg: Tensor, qinv: Tensor,
               live: Tensor) -> Tuple[Tensor, List[Tensor]]:
    return _hildreth_sweeps(x, lam, Gg, WGg, hg, qinv, live, 1)


def _compiled_sweeps():
    """Sweep kernel that runs a ``torch.compile``-d *single* sweep in a Python loop.

    Compiling one sweep (instead of unrolling ``n`` of them) keeps the graph at
    ``O(groups)`` nodes, so compilation time does not grow with the sweep count.
    """
    global _COMPILED_SWEEP
    if _COMPILED_SWEEP is None:
        _COMPILED_SWEEP = torch.compile(_one_sweep, dynamic=False)

    def run(x, lam, Gg, WGg, hg, qinv, live, n_sweeps):
        for _ in range(n_sweeps):
            x, lam = _COMPILED_SWEEP(x, lam, Gg, WGg, hg, qinv, live)
        return x, list(lam)
    return run


def _working_set_solve(Gs: Tensor, WGs: Tensor, c_y: Tensor, idx: Tensor, vf: Tensor):
    """Solve ``(G_A W^-1 G_A^T) mu = G_A y - h_A`` on the slots marked by ``vf``.

    A ridge of ``1e-13 (1 + max diag)`` makes the system nonsingular when
    active rows are linearly dependent (duplicate or degenerate constraints),
    which yields the minimum-norm split of the multipliers. Returns ``mu``, the
    *unregularized* masked matrix (for the residual test of the KKT check), the
    right-hand side and a singularity flag.
    """
    Q = (Gs @ WGs) * vf.unsqueeze(-1) * vf.unsqueeze(-2)
    eye_inactive = torch.diag_embed(1.0 - vf)
    Mmat = Q + eye_inactive
    rhs = torch.gather(c_y, -1, idx) * vf
    diag = torch.diagonal(Q, dim1=-2, dim2=-1)
    ridge = 1e-13 * (1.0 + diag.amax(-1, keepdim=True))
    mu, info = torch.linalg.solve_ex(Mmat + torch.diag_embed(ridge * vf), rhs)
    singular = (info != 0) | ~torch.isfinite(mu).all(-1)
    return mu, Mmat, rhs, singular


def _goldfarb_idnani(y: Tensor, G: Tensor, h: Tensor, WG: Tensor, mask: Tensor, feas_tol: float,
                     live: Tensor, iters: int, early_exit: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """Goldfarb-Idnani dual active-set method for ``min ||x - y||_W^2 s.t. G x <= h`` (batched).

    Starts from the unconstrained minimizer ``x = y`` with an empty working set
    ``A`` and repeatedly picks the most violated row ``p`` (scaled by
    ``||G_p||_{W^-1}``). Raising its multiplier by ``t`` while keeping the
    rows of ``A`` at equality moves ``x`` along::

        dlam_A = -(G_A W^-1 G_A^T)^{-1} G_A W^-1 G_p^T,
        z      = -W^-1 (G_p^T + G_A^T dlam_A).

    The full step ``t2 = (G_p x - h_p) / (-G_p z)`` makes ``p`` active and adds
    it to ``A``; the partial step ``t1 = min_{dlam_j < 0} lam_j / (-dlam_j)``
    drops the row whose multiplier reaches zero first. If ``G_p`` lies in the
    span of ``A`` (``z = 0``) only partial steps are possible, and if none is
    either the QP is infeasible. ``A`` stays linearly independent (so ``n``
    slots suffice), the dual objective increases strictly and the method
    terminates finitely (Goldfarb and Idnani, Math. Programming 27, 1983).

    The returned ``x`` is recomputed as ``y - W^-1 G^T lam`` (stationarity
    exact) and accepted only if every unmasked row is feasible within
    ``feas_tol (1 + |h_r|)`` and ``lam >= 0`` - the KKT conditions, so an
    accepted point is the unique optimum. ``h`` must be sanitized (0 on masked
    rows). With ``early_exit=False`` the loop has a fixed trip count.

    Returns ``(x, lam, accepted)``; entries that are not accepted are zero.
    """
    dt = y.dtype
    dev = y.device
    m, n = G.shape[-2], G.shape[-1]
    batch = y.shape[:-1]
    k = min(m, n)
    h_scale = 1.0 + h.abs()
    q = (G * WG.transpose(-1, -2)).sum(-1)
    qn = torch.sqrt(torch.where(mask, q, torch.ones_like(q)))
    x = y.clone()
    idx = torch.zeros(*batch, k, dtype=torch.long, device=dev)
    valid = torch.zeros(*batch, k, dtype=torch.bool, device=dev)
    u = torch.zeros(*batch, k, dtype=dt, device=dev)
    p = torch.full(batch, -1, dtype=torch.long, device=dev)
    up = torch.zeros(batch, dtype=dt, device=dev)
    live = live.clone()
    finished = torch.zeros_like(live)
    inf_k = torch.full((*batch, k), math.inf, dtype=dt, device=dev)
    ninf_m = torch.full((*batch, m), -math.inf, dtype=dt, device=dev)
    inf_b = torch.full(batch, math.inf, dtype=dt, device=dev)
    oh = torch.nn.functional.one_hot
    for _ in range(max(iters, 1)):
        r = (G @ x.unsqueeze(-1)).squeeze(-1) - h
        viol = mask & (r > feas_tol * h_scale)
        need = live & (p < 0)
        has_v = viol.any(-1)
        finished = finished | (need & ~has_v)
        live = live & ~(need & ~has_v)
        take = need & has_v
        p = torch.where(take, torch.argmax(torch.where(viol, r / qn, ninf_m), dim=-1), p)
        up = torch.where(take, torch.zeros_like(up), up)
        act = live & (p >= 0)
        if early_exit and not bool(act.any()):
            break
        p_s = torch.clamp(p, min=0)
        gp = torch.gather(G, -2, p_s.view(*batch, 1, 1).expand(*batch, 1, n)).squeeze(-2)
        Wgp = torch.gather(WG, -1, p_s.view(*batch, 1, 1).expand(*batch, n, 1)).squeeze(-1)
        Gs = torch.gather(G, -2, idx.unsqueeze(-1).expand(*batch, k, n))
        WGs = torch.gather(WG, -1, idx.unsqueeze(-2).expand(*batch, n, k))
        vf = valid.to(dt)
        Qs = (Gs @ WGs) * vf.unsqueeze(-1) * vf.unsqueeze(-2) + torch.diag_embed(1.0 - vf)
        b = (Gs @ Wgp.unsqueeze(-1)).squeeze(-1) * vf
        sol, info = torch.linalg.solve_ex(Qs, b)
        dlam = -sol * vf
        z = -(Wgp + (WGs @ dlam.unsqueeze(-1)).squeeze(-1))
        dec = -(gp * z).sum(-1)                                  # >= 0: violation decrease rate
        q_p = torch.gather(q, -1, p_s.unsqueeze(-1)).squeeze(-1)
        zero_z = dec <= 1e-10 * q_p                             # G_p in span(G_A)
        r_p = torch.clamp(torch.gather(r, -1, p_s.unsqueeze(-1)).squeeze(-1), min=0.0)
        t2 = torch.where(zero_z, inf_b, r_p / torch.where(zero_z, torch.ones_like(dec), dec))
        neg = valid & (dlam < 0)
        ratio = torch.where(neg, u / torch.where(neg, -dlam, torch.ones_like(dlam)), inf_k)
        t1, j1 = ratio.min(dim=-1)
        t = torch.minimum(t1, t2)
        failed = act & ((info != 0) | ~torch.isfinite(t))          # infeasible QP (or breakdown)
        live = live & ~failed
        act = act & ~failed
        tf = torch.where(act, t, torch.zeros_like(t))
        x = x + tf.unsqueeze(-1) * torch.where(zero_z.unsqueeze(-1), torch.zeros_like(z), z)
        u = u + tf.unsqueeze(-1) * dlam
        up = up + tf
        full = act & (t2 <= t1)
        partial = act & ~full
        drop = oh(j1, k).to(torch.bool) & partial.unsqueeze(-1)
        valid = valid & ~drop
        u = torch.where(drop, torch.zeros_like(u), u)
        empty = ~valid
        slot = torch.argmax(empty.to(torch.int8), dim=-1)
        can_add = full & empty.any(-1)
        add = oh(slot, k).to(torch.bool) & can_add.unsqueeze(-1)
        idx = torch.where(add, p.unsqueeze(-1), idx)
        valid = valid | add
        u = torch.where(add, up.unsqueeze(-1), u)
        live = live & ~(full & ~can_add)                           # cannot happen in exact arithmetic
        p = torch.where(full, torch.full_like(p, -1), p)
    lam = torch.zeros(*batch, m, dtype=dt, device=dev).scatter_add(-1, idx, u * valid.to(dt))
    x_final = y - (WG @ lam.unsqueeze(-1)).squeeze(-1)
    r = (G @ x_final.unsqueeze(-1)).squeeze(-1) - h
    feas = (torch.where(mask, r, torch.zeros_like(r)) <= feas_tol * h_scale).all(-1)
    accepted = finished & feas & (lam >= 0).all(-1)
    zero_x = torch.zeros_like(x_final)
    return (torch.where(accepted.unsqueeze(-1), x_final, zero_x),
            torch.where(accepted.unsqueeze(-1), lam, torch.zeros_like(lam)), accepted)


def _pdas_refine(y: Tensor, G: Tensor, h: Tensor, WG: Tensor, mask: Tensor, feas_tol: float,
                 lam_seed: Tensor, live: Tensor, iters: int,
                 early_exit: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """Primal-dual active-set (semismooth Newton) refinement with a KKT certificate.

    The method of Hintermueller, Ito and Kunisch for ``min ||x - y||_W^2 s.t.
    G x <= h``: with the current pair ``(x, lam)`` the working set is

        A = {r : lam_r + (G_r x - h_r) / (G_r W^-1 G_r^T) > 0},

    i.e. the rows that one Hildreth update would keep or make positive, and the
    next pair solves the equality-constrained projection on ``A``. It changes
    the whole working set per iteration and typically terminates in a handful of
    iterations from a good Hildreth warm start, but it may oscillate when rows
    interact with mixed signs (no M-matrix structure); :func:`_goldfarb_idnani`
    is the finite fallback. ``A`` is
    held in ``k = min(m, 2n)`` slots (dependent rows are handled by the ridge of
    :func:`_working_set_solve`); elements whose working set is larger are not
    accepted. Acceptance uses the KKT test (feasibility of every row,
    ``mu >= 0``, accurate solve), so an accepted point is the exact optimum.

    ``h`` must already be sanitized (0 on masked rows). Returns ``(x, mu, accepted)``.
    """
    dt = y.dtype
    m, n = G.shape[-2], G.shape[-1]
    batch = y.shape[:-1]
    k = min(m, 2 * n)          # room for degenerate vertices with more than n active rows
    h_scale = 1.0 + h.abs()
    c_y = (G @ y.unsqueeze(-1)).squeeze(-1) - h
    q = (G * WG.transpose(-1, -2)).sum(-1)
    qinv = torch.where(mask, 1.0 / torch.where(mask, q, torch.ones_like(q)), torch.zeros_like(q))
    lam = torch.where(mask, lam_seed.clamp(min=0.0), torch.zeros_like(lam_seed))
    x = y - (WG @ lam.unsqueeze(-1)).squeeze(-1)
    r = (G @ x.unsqueeze(-1)).squeeze(-1) - h
    live = live.clone()
    x_out = torch.zeros_like(y)
    mu_out = torch.zeros_like(lam)
    accepted = torch.zeros_like(live)
    ninf = torch.full_like(lam, -math.inf)
    A_prev = torch.zeros_like(mask)
    for it in range(max(iters, 1)):
        score = lam + r * qinv
        A = mask & (score > 0)
        if it > 0:
            # a repeated working set reproduces the previous (rejected) solution
            live = live & (A != A_prev).any(-1)
            if early_exit and not bool(live.any()):
                break
        A_prev = A
        over = A.sum(-1) > k
        top_v, idx = torch.topk(torch.where(A, score, ninf), k, dim=-1)
        valid = top_v > -math.inf
        Gs = torch.gather(G, -2, idx.unsqueeze(-1).expand(*batch, k, n))
        WGs = torch.gather(WG, -1, idx.unsqueeze(-2).expand(*batch, n, k))
        vf = valid.to(dt)
        mu, Mmat, rhs, singular = _working_set_solve(Gs, WGs, c_y, idx, vf)
        mu = torch.where(valid & ~singular.unsqueeze(-1), mu, torch.zeros_like(mu))
        x_new = y - (WGs @ mu.unsqueeze(-1)).squeeze(-1)
        r_new = (G @ x_new.unsqueeze(-1)).squeeze(-1) - h
        lam_new = torch.zeros_like(lam).scatter_add(-1, idx, mu)
        res = ((Mmat @ mu.unsqueeze(-1)).squeeze(-1) - rhs).abs().amax(-1)
        feas = (torch.where(mask, r_new, torch.zeros_like(r_new)) <= feas_tol * h_scale).all(-1)
        optimal = (live & ~singular & ~over & (mu >= 0).all(-1) & feas
                   & (res <= feas_tol * (1.0 + rhs.abs().amax(-1))))
        x_out = torch.where(optimal.unsqueeze(-1), x_new, x_out)
        mu_out = torch.where(optimal.unsqueeze(-1), lam_new, mu_out)
        accepted = accepted | optimal
        live = live & ~optimal & ~singular
        if early_exit and not bool(live.any()):
            break
        upd = live.unsqueeze(-1)
        lam = torch.where(upd, lam_new, lam)
        x = torch.where(upd, x_new, x)
        r = torch.where(upd, r_new, r)
    return x_out, mu_out, accepted


def hildreth(y: Tensor, G: Tensor, h: Tensor, *, W_inv: Optional[Tensor] = None,
             lam0: Optional[Tensor] = None, row_mask: Optional[Tensor] = None,
             groups: GroupsLike = None, max_sweeps: int = 50, tol: Optional[float] = None,
             feas_tol: Optional[float] = None, check_every: int = 1, polish: bool = False,
             polish_iters: Optional[int] = None, polish_during: bool = True,
             gi_iters: Optional[int] = None, finisher: bool = True,
             compile: bool = False) -> HildrethResult:
    """Weighted projection onto a polyhedron by Hildreth's method [77].

    Solves ``min_x ||x - y||_W^2  s.t.  G x <= h`` through its dual: with
    ``x = y - W^{-1} G^T lam``, each row ``r`` is updated in Gauss-Seidel order::

        lam_r' = max(0, lam_r + (G_r x - h_r) / (G_r W^{-1} G_r^T))
        x      = x - (lam_r' - lam_r) W^{-1} G_r^T

    With ``groups`` (a partition of the rows into sets of mutually
    non-interacting rows, see the module docstring) each group is updated in one
    vectorized step; the iterates are identical to row-by-row Hildreth.

    Hildreth's method converges linearly with a rate that degrades when active
    constraints are nearly parallel (Section 8 of the paper) and when rows form
    long chains (Stage A's acceleration box couples consecutive knots, which
    makes the dual Hessian a discrete Laplacian; thousands of sweeps can be
    needed); at an iteration cap its iterate may be feasible but not optimal.
    With ``polish=True`` the warm start and the checks after sweeps 1, 2, 4, 8,
    ... (and the cap) run the primal-dual active-set method of
    :func:`_pdas_refine` seeded by the current multipliers. After the sweeps,
    elements without a certificate (including those that stopped through the
    ordinary convergence test) get one more PDAS attempt and then the finite
    Goldfarb-Idnani dual active-set method (:func:`_goldfarb_idnani`). Every
    accepted point carries a KKT certificate and is the exact optimum.

    Parameters
    ----------
    y:
        Point to project, ``(..., n)``.
    G, h:
        Constraints ``(..., m, n)`` and ``(..., m)``.
    W_inv:
        Inverse weight: ``None`` (identity), diagonal ``(..., n)`` or dense
        ``(..., n, n)`` (symmetric positive definite).
    lam0:
        Warm-start multipliers ``(..., m)`` (negative entries are clipped to 0).
    row_mask:
        Boolean ``(..., m)``; masked-out rows are ignored (multiplier held at 0).
    groups:
        Optional row partition (sequences or ``LongTensor`` of row indices);
        default: one row per group. Rows inside a group must not interact.
    max_sweeps:
        Iteration cap (number of full sweeps over the rows).
    tol, feas_tol:
        Convergence tolerances on the multiplier change (relative to
        ``1 + max lam``) and on the primal violation (relative to ``1 + |h_r|``).
        Default to ``1e3 * eps`` and ``1e4 * eps`` of the dtype.
    check_every:
        Number of sweeps between convergence tests.
    polish, polish_iters:
        Enable the exact active-set polish; cap on PDAS iterations per attempt
        (default ``min(2 (m + n), 16)``).
    polish_during:
        Also polish at sweeps 1, 2, 4, ... (``True``, cheap for Stage B where the
        working set is ``3 x 3``); with ``False`` only the warm start and the
        final iterate are polished (large problems such as Stage A).
    gi_iters, finisher:
        Iteration cap of the final Goldfarb-Idnani finisher (default ``4 n + 32``)
        and whether to run it (callers with their own exact backstop disable it).
    compile:
        Run the sweep kernel through ``torch.compile`` (fixed shapes; the first
        call per shape pays the compilation). Measured on CPU it roughly halves
        the per-sweep time but compiling a Stage-A sweep takes minutes, so it is
        meant for long-running deployments with fixed shapes.
    """
    dt = y.dtype
    dev = y.device
    eps = torch.finfo(dt).eps
    tol = 1e3 * eps if tol is None else tol
    feas_tol = 1e4 * eps if feas_tol is None else feas_tol
    batch = torch.broadcast_shapes(y.shape[:-1], G.shape[:-2], h.shape[:-1])
    m, n = G.shape[-2], G.shape[-1]
    y = y.expand(*batch, n)
    G = G.expand(*batch, m, n)
    h = h.expand(*batch, m)
    refine_iters = polish_iters if polish_iters is not None else min(2 * (m + n), 16)
    gi_iters = gi_iters if gi_iters is not None else 4 * n + 32

    WG = _apply_winv(W_inv, G.transpose(-1, -2))          # (..., n, m)
    q = (G * WG.transpose(-1, -2)).sum(-1)                 # (..., m) = G_r W^-1 G_r^T
    mask = (q > 0) if row_mask is None else (row_mask.expand(*batch, m) & (q > 0))
    maskf = mask.to(dt)
    h_s = torch.where(mask, h, torch.zeros_like(h))        # masked rows can never be active
    qinv = torch.where(mask, 1.0 / torch.where(mask, q, torch.ones_like(q)), torch.zeros_like(q))
    h_scale = 1.0 + h_s.abs()

    gidx, gval, pos = _pack_groups(_groups_key(groups, m), m)
    gidx, gval_f, pos = gidx.to(dev), gval.to(device=dev, dtype=dt), pos.to(dev)
    ng, gmax = gidx.shape
    Gg = (G[..., gidx, :] * gval_f.unsqueeze(-1)).movedim(-3, 0)        # (ng, ..., g, n)
    WGg = (WG[..., gidx] * gval_f).movedim(-2, 0)                       # (ng, ..., n, g)
    hg = (h_s[..., gidx] * gval_f).movedim(-2, 0)                       # (ng, ..., g)
    qg = (qinv[..., gidx] * gval_f).movedim(-2, 0)

    def to_groups(lam_full: Tensor) -> List[Tensor]:
        return list((lam_full[..., gidx] * gval_f).movedim(-2, 0).unbind(0))

    def to_full(lam_list: List[Tensor]) -> Tensor:
        return torch.stack(lam_list, dim=-2).reshape(*batch, ng * gmax)[..., pos]

    if lam0 is None:
        lam_full = torch.zeros(*batch, m, dtype=dt, device=dev)
    else:
        lam_full = torch.clamp(lam0.expand(*batch, m).to(dt), min=0.0) * maskf
    x = y - (WG @ lam_full.unsqueeze(-1)).squeeze(-1)
    done = torch.zeros(batch, dtype=torch.bool, device=dev)
    polished = torch.zeros_like(done)

    def try_polish(x: Tensor, lam_full: Tensor, skip: Tensor, finisher: bool = False):
        # semismooth-Newton active set seeded by the Hildreth multipliers; at the
        # end, the finite Goldfarb-Idnani method for whatever is left
        xp, mup, ok = _pdas_refine(y, G, h_s, WG, mask, feas_tol, lam_full, ~skip, refine_iters)
        rest = ~skip & ~ok
        if finisher and bool(rest.any()):
            xq, muq, ok2 = _goldfarb_idnani(y, G, h_s, WG, mask, feas_tol, rest, gi_iters)
            xp = torch.where(ok2.unsqueeze(-1), xq, xp)
            mup = torch.where(ok2.unsqueeze(-1), muq, mup)
            ok = ok | ok2
        x = torch.where(ok.unsqueeze(-1), xp, x)
        lam_full = torch.where(ok.unsqueeze(-1), mup, lam_full)
        return x, lam_full, ok

    if polish and lam0 is not None:
        x, lam_full, ok = try_polish(x, lam_full, polished)
        polished = polished | ok
        done = done | ok

    sweep_fn = _compiled_sweeps() if compile else _hildreth_sweeps
    lam_list = to_groups(lam_full)
    sweeps = 0
    next_polish = 1
    while sweeps < max_sweeps and not bool(done.all()):
        nb = min(max(check_every, 1), max_sweeps - sweeps)
        before = lam_full
        live = (~done).to(dt).unsqueeze(-1)
        x, lam_list = sweep_fn(x, lam_list, Gg, WGg, hg, qg, live, nb)
        sweeps += nb
        lam_full = to_full(lam_list)
        v = torch.clamp((G @ x.unsqueeze(-1)).squeeze(-1) - h_s, min=0.0) * maskf
        feas = (v <= feas_tol * h_scale).all(-1)
        small = (lam_full - before).abs().amax(-1) <= tol * (1.0 + lam_full.amax(-1).clamp(min=0.0))
        done = done | (feas & small)
        if polish and polish_during and not bool(done.all()) and \
                (sweeps >= next_polish or sweeps >= max_sweeps):
            x, lam_full, ok = try_polish(x, lam_full, polished | done)
            polished = polished | ok
            done = done | ok
            lam_list = to_groups(lam_full)
            while next_polish <= sweeps:
                next_polish *= 2

    if polish and not bool(polished.all()):
        x, lam_full, ok = try_polish(x, lam_full, polished, finisher=finisher)
        polished = polished | ok
        done = done | ok

    v = torch.clamp((G @ x.unsqueeze(-1)).squeeze(-1) - h_s, min=0.0) * maskf
    max_violation = v.amax(-1) if m > 0 else torch.zeros(batch, dtype=dt, device=dev)
    feasible = (v <= feas_tol * h_scale).all(-1)
    return HildrethResult(x=x, lam=lam_full, sweeps=sweeps, converged=done & feasible,
                          feasible=feasible, max_violation=max_violation,
                          exact=polished & feasible)


# ---------------------------------------------------------------------------
# exact enumeration for small dimensions (Stage B backstop)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=128)
def _subsets(m: int, k: int) -> tuple[Tensor, Tensor]:
    """All subsets of ``range(m)`` of size ``<= k`` as padded indices ``(C, k)`` and a validity mask."""
    combos = [c for j in range(k + 1) for c in itertools.combinations(range(m), j)]
    idx = torch.zeros(len(combos), max(k, 1), dtype=torch.long)
    val = torch.zeros(len(combos), max(k, 1), dtype=torch.bool)
    for i, c in enumerate(combos):
        if c:
            idx[i, :len(c)] = torch.tensor(c)
            val[i, :len(c)] = True
    return idx, val


def enumerate_small_qp(y: Tensor, G: Tensor, h: Tensor, *, W_inv: Optional[Tensor] = None,
                       row_mask: Optional[Tensor] = None,
                       feas_tol: Optional[float] = None) -> tuple[Tensor, Tensor]:
    """Exact ``min ||x - y||_W^2 s.t. G x <= h`` by enumerating active sets of size ``<= n``.

    The optimum is the W-projection of ``y`` onto ``{G_S x = h_S}`` for a
    linearly independent subset ``S`` of its active rows (Caratheodory's theorem
    for the cone of active gradients), so the feasible candidate with the
    smallest objective ``mu_S^T (G_S W^-1 G_S^T) mu_S`` is the optimum. All
    ``sum_{j<=n} C(m, j)`` candidates are solved in one batched call; intended
    for ``n <= 3`` (Stage B) where this is a few hundred ``3 x 3`` solves.

    Returns
    -------
    (x, feasible):
        The optimum, and whether the QP is feasible (otherwise ``x = y``).
    """
    dt = y.dtype
    eps = torch.finfo(dt).eps
    feas_tol = 1e4 * eps if feas_tol is None else feas_tol
    batch = torch.broadcast_shapes(y.shape[:-1], G.shape[:-2], h.shape[:-1])
    m, n = G.shape[-2], G.shape[-1]
    y = y.expand(*batch, n)
    G = G.expand(*batch, m, n)
    h = h.expand(*batch, m)
    mask = torch.ones(*batch, m, dtype=torch.bool, device=y.device) if row_mask is None \
        else row_mask.expand(*batch, m)
    h = torch.where(mask, h, torch.zeros_like(h))
    idx, val = _subsets(m, min(n, m))
    idx, val = idx.to(y.device), val.to(y.device)
    vm = val.to(dt)

    WG = _apply_winv(W_inv, G.transpose(-1, -2))                      # (..., n, m)
    Qm = G @ WG                                                        # (..., m, m)
    c = (G @ y.unsqueeze(-1)).squeeze(-1) - h                          # (..., m)
    Qs = Qm[..., idx.unsqueeze(-1), idx.unsqueeze(-2)]                 # (..., C, k, k)
    Qs = Qs * vm.unsqueeze(-1) * vm.unsqueeze(-2) + torch.diag_embed(1.0 - vm)
    cs = c[..., idx] * vm                                              # (..., C, k)
    mu, info = torch.linalg.solve_ex(Qs, cs)
    mu = mu * vm
    WGs = WG[..., idx]                                                 # (..., n, C, k)
    xs = y.unsqueeze(-2) - torch.einsum("...nck,...ck->...cn", WGs, mu)
    r = torch.einsum("...cn,...mn->...cm", xs, G) - h.unsqueeze(-2)    # (..., C, m)
    row_ok = (r <= feas_tol * (1.0 + h.abs().unsqueeze(-2))) | ~mask.unsqueeze(-2)
    cand_rows_ok = (mask[..., idx] | ~val).all(-1)                     # no masked row in S
    feasible = (row_ok.all(-1) & (info == 0) & cand_rows_ok & torch.isfinite(mu).all(-1))
    obj = (mu * (Qs @ mu.unsqueeze(-1)).squeeze(-1)).sum(-1)
    obj = torch.where(feasible, obj, torch.full_like(obj, math.inf))
    best = torch.argmin(obj, dim=-1)                                   # (...)
    x = torch.gather(xs, -2, best[..., None, None].expand(*batch, 1, n)).squeeze(-2)
    any_feas = feasible.any(-1)
    return torch.where(any_feas.unsqueeze(-1), x, y), any_feas


# ---------------------------------------------------------------------------
# Stage B: Eq. (34)
# ---------------------------------------------------------------------------
@dataclass
class StageBResult:
    """Result of the Stage-B knot projection (Eq. 34).

    Attributes
    ----------
    u:
        Projected acceleration ``u*`` inside ``[-u_inf, u_inf]^d``, ``(..., d)``.
    ok:
        Exact primal feasibility of ``u`` for every active share. ``False``
        means the executor must engage the braking fallback (Algorithm 1, line 29).
    lam:
        Multipliers in the row layout of :func:`stage_b_rows`
        ``(..., N + 2d)`` - pass back as ``lam0`` at the next tick.
    n_active:
        Number of active shares per element.
    method:
        ``0`` no active share (clip), ``1`` closed form (Lemma 18), ``2`` Hildreth.
    sweeps:
        Hildreth sweeps used by the slowest element (0 if Hildreth was not run).
    max_violation:
        ``max_j (zeta_j - g_j . u)_+`` against the original constraints.
    optimal:
        ``u`` is certified to be the exact minimizer of Eq. (34) (closed form,
        KKT-certified polish or enumeration).
    """

    u: Tensor
    ok: Tensor
    lam: Tensor
    n_active: Tensor
    method: Tensor
    sweeps: int
    max_violation: Tensor
    optimal: Tensor


def stage_b_rows(g: Tensor, zeta: Tensor, u_inf: float) -> tuple[Tensor, Tensor]:
    """Rows ``G u <= h`` of Eq. (34): shares ``-g_j . u <= -zeta_j``, then ``u <= u_inf``, ``-u <= u_inf``.

    Returns ``G`` of shape ``(..., N + 2d, d)`` and ``h`` of shape ``(..., N + 2d)``.
    """
    d = g.shape[-1]
    batch = g.shape[:-2]
    eye = torch.eye(d, dtype=g.dtype, device=g.device).expand(*batch, d, d)
    G = torch.cat([-g, eye, -eye], dim=-2)
    box = torch.full((*batch, 2 * d), float(u_inf), dtype=g.dtype, device=g.device)
    h = torch.cat([-zeta, box], dim=-1)
    return G, h


def stage_b_groups(N: int, d: int) -> Tuple[Tuple[int, ...], ...]:
    """Row groups of :func:`stage_b_rows`: each share alone, then the ``d`` upper and the ``d`` lower box rows.

    Box rows of different axes are orthogonal, so each half of the box is one group.
    """
    return tuple((j,) for j in range(N)) + (tuple(range(N, N + d)), tuple(range(N + d, N + 2 * d)))


def _stage_b_prepare(u_hat: Tensor, g: Tensor, zeta: Tensor, active: Tensor, u_inf: float,
                     feas_tol: float):
    """Clip ``u^``, drop non-finite shares, count active rows and tighten (see module docstring)."""
    u_hat = torch.clamp(u_hat, -u_inf, u_inf)
    finite = torch.isfinite(zeta) & torch.isfinite(g).all(-1)
    bad = (active & ~finite).any(-1)          # ill-posed active share -> not ok
    act = active & finite
    n_act = act.sum(-1)
    g1 = g.abs().sum(-1)
    zeta_c = torch.where(act, zeta, torch.zeros_like(zeta))
    tau = 2.0 * feas_tol * (1.0 + zeta_c.abs() + g1 * (1.0 + u_inf))
    zeta_t = zeta_c + tau
    return u_hat, act, bad, n_act, zeta_t


def _stage_b_verify(u: Tensor, g: Tensor, zeta: Tensor, act: Tensor, bad: Tensor, u_inf: float):
    """Clamp to the box and check every active share exactly (original ``zeta``)."""
    u = torch.clamp(u, -u_inf, u_inf)
    slack = (g * u.unsqueeze(-2)).sum(-1) - zeta
    viol = torch.where(act, torch.clamp(-slack, min=0.0), torch.zeros_like(slack))
    max_violation = viol.amax(-1) if viol.shape[-1] > 0 else torch.zeros(u.shape[:-1], dtype=u.dtype,
                                                                          device=u.device)
    ok = (~bad) & (viol <= 0).all(-1)
    return u, ok, max_violation


def _closed_form_first_active(u_hat: Tensor, g: Tensor, zeta_t: Tensor, act: Tensor, u_inf: float):
    """Lemma 18 on the first active share of every element; multipliers in Stage-B row layout."""
    N, d = g.shape[-2], g.shape[-1]
    batch = u_hat.shape[:-1]
    idx = torch.argmax(act.to(torch.int8), dim=-1, keepdim=True)              # (..., 1)
    g1row = torch.gather(g, -2, idx.unsqueeze(-1).expand(*batch, 1, d)).squeeze(-2)
    z1 = torch.gather(zeta_t, -1, idx).squeeze(-1)
    res = project_box_halfspace(u_hat, g1row, z1, -u_inf, u_inf)
    lam1 = torch.where(res.feasible, res.lam, torch.zeros_like(res.lam))
    lam_share = torch.zeros(*batch, N, dtype=u_hat.dtype, device=u_hat.device).scatter(
        -1, idx, lam1.unsqueeze(-1))
    lam = torch.cat([lam_share, res.lam_upper, res.lam_lower], dim=-1)
    return res.u, lam


def solve_stage_b_qp(u_hat: Tensor, g: Tensor, zeta: Tensor, active: Tensor, u_inf: float, *,
                     lam0: Optional[Tensor] = None, max_sweeps: int = 50,
                     feas_tol: Optional[float] = None, polish: bool = True,
                     enumerate_max_rows: int = 24, enum_after: int = 2) -> StageBResult:
    """Knot-boundary projection ``argmin_{|u|_inf <= u_inf} ||u - u^||^2  s.t.  g_j . u >= zeta_j``.

    Each batch element is routed by its number of active shares: none - clip
    ``u^`` to the box; one - Lemma 18; two or more - grouped Hildreth
    warm-started with ``lam0`` and capped at ``max_sweeps``, with the exact
    active-set polish (``polish=True``) so that nearly parallel constraints do
    not exhaust the cap. When ``d <= 3`` and there are at most
    ``enumerate_max_rows`` rows, Hildreth is limited to ``enum_after`` sweeps
    (it normally certifies after one or two thanks to the warm start and the
    polish) and elements that still lack a KKT certificate are finished by
    :func:`enumerate_small_qp`; this bounds the per-tick cost, also for
    infeasible problems, and the returned input is the exact minimizer whenever
    the QP is feasible. Theorem 3(d): one sweep costs ``O(N d)``; the closed form
    ``O(d log d)``.

    Parameters
    ----------
    u_hat:
        Nominal accelerations ``(..., d)`` (Eq. 34 clips them to the box first).
    g, zeta, active:
        Shares from :func:`cma_vpa.safety.margins.reciprocal_shares`:
        ``(..., N, d)``, ``(..., N)``, ``(..., N)``.
    u_inf:
        Per-axis input bound.
    lam0:
        Warm start ``(..., N + 2d)`` from the previous tick (rows of
        :func:`stage_b_rows`); multipliers of rows that are no longer active
        are discarded.
    feas_tol:
        Relative solver tolerance used to tighten the constraints (default
        ``1e4 * eps`` of the dtype). See the module docstring.
    """
    dt, dev = u_hat.dtype, u_hat.device
    feas_tol = 1e4 * torch.finfo(dt).eps if feas_tol is None else feas_tol
    d = u_hat.shape[-1]
    batch = torch.broadcast_shapes(u_hat.shape[:-1], g.shape[:-2], zeta.shape[:-1], active.shape[:-1])
    if len(batch) == 0:  # single problem: add a batch axis for the routing below
        r = solve_stage_b_qp(u_hat.unsqueeze(0), g.unsqueeze(0), zeta.unsqueeze(0),
                             active.unsqueeze(0), u_inf,
                             lam0=None if lam0 is None else lam0.unsqueeze(0),
                             max_sweeps=max_sweeps, feas_tol=feas_tol, polish=polish,
                             enumerate_max_rows=enumerate_max_rows, enum_after=enum_after)
        return StageBResult(u=r.u[0], ok=r.ok[0], lam=r.lam[0], n_active=r.n_active[0],
                            method=r.method[0], sweeps=r.sweeps, max_violation=r.max_violation[0],
                            optimal=r.optimal[0])
    N = g.shape[-2]
    u_hat = u_hat.expand(*batch, d)
    g = g.expand(*batch, N, d)
    zeta = zeta.expand(*batch, N)
    active = active.expand(*batch, N)
    u_hat, act, bad, n_act, zeta_t = _stage_b_prepare(u_hat, g, zeta, active, u_inf, feas_tol)

    u_out = u_hat.clone()
    lam_out = torch.zeros(*batch, N + 2 * d, dtype=dt, device=dev)
    method = torch.zeros(batch, dtype=torch.int8, device=dev)
    optimal = n_act == 0
    sweeps = 0

    # -- exactly one active share: Lemma 18 ---------------------------------
    one = n_act == 1
    if bool(one.any()):
        u1, lam1 = _closed_form_first_active(u_hat, g, zeta_t, act, u_inf)
        u_out = torch.where(one.unsqueeze(-1), u1, u_out)
        lam_out = torch.where(one.unsqueeze(-1), lam1, lam_out)
        method = torch.where(one, torch.ones_like(method), method)
        optimal = optimal | one

    # -- two or more: grouped Hildreth + polish + enumeration backstop --------
    many = n_act >= 2
    if bool(many.any()):
        sel = many.nonzero(as_tuple=True)
        S_ = sel[0].shape[0]
        # compress to the active shares (k <= N rows per element, plus the box)
        act_s = act[sel]
        k = int(n_act[sel].max())
        cidx = torch.topk(act_s.to(torch.int8), k, dim=-1).indices                 # (S, k)
        valid = torch.gather(act_s, -1, cidx)
        g_c = torch.gather(g[sel], -2, cidx.unsqueeze(-1).expand(S_, k, d))
        z_c = torch.where(valid, torch.gather(zeta_t[sel], -1, cidx), torch.zeros(S_, k, dtype=dt, device=dev))
        G, h = stage_b_rows(g_c, z_c, u_inf)
        rmask = torch.cat([valid, torch.ones(S_, 2 * d, dtype=torch.bool, device=dev)], -1)
        l0 = None
        if lam0 is not None:
            l0f = lam0.expand(*batch, N + 2 * d)[sel]
            l0 = torch.cat([torch.gather(l0f[:, :N], -1, cidx), l0f[:, N:]], -1)
        enum_ok = polish and d <= 3 and G.shape[-2] <= enumerate_max_rows
        # With the exact enumeration available, warm-started Hildreth (polished at
        # the warm start and at sweeps 1, 2) handles the common case and the
        # enumeration settles the rest in one batched call - including infeasible
        # problems, for which Hildreth would otherwise run to its cap.
        hr = hildreth(u_hat[sel], G, h, lam0=l0, row_mask=rmask, groups=stage_b_groups(k, d),
                      max_sweeps=min(max_sweeps, enum_after) if enum_ok else max_sweeps,
                      feas_tol=feas_tol, polish=polish, polish_iters=4, finisher=not enum_ok)
        x = hr.x
        exact = hr.exact
        need = ~exact
        if enum_ok and bool(need.any()):
            xe, fe = enumerate_small_qp(u_hat[sel][need], G[need], h[need], row_mask=rmask[need],
                                        feas_tol=feas_tol)
            x = x.clone()
            x[need] = torch.where(fe.unsqueeze(-1), xe, x[need])
            exact = exact.clone()
            exact[need] = fe
        u_out[sel] = x
        lam_sh = torch.zeros(S_, N, dtype=dt, device=dev).scatter(
            -1, cidx, torch.where(valid, hr.lam[:, :k], torch.zeros_like(hr.lam[:, :k])))
        lam_out[sel] = torch.cat([lam_sh, hr.lam[:, k:]], -1)
        method[sel] = 2
        optimal = optimal.clone()
        optimal[sel] = exact
        sweeps = hr.sweeps

    u_out, ok, max_violation = _stage_b_verify(u_out, g, zeta, act, bad, u_inf)
    return StageBResult(u=u_out, ok=ok, lam=lam_out, n_active=n_act, method=method,
                        sweeps=sweeps, max_violation=max_violation, optimal=optimal & ok)


def stage_b_fixed(u_hat: Tensor, g: Tensor, zeta: Tensor, active: Tensor, u_inf: float,
                  lam0: Optional[Tensor] = None, *, sweeps: int = 4, refine_iters: int = 8,
                  feas_tol: Optional[float] = None, multi: bool = True) -> StageBResult:
    """Branch-free, fixed-shape Stage-B projection (compile target).

    Computes, for *every* element, the closed form of Lemma 18 on the first
    active share and (``multi=True``) ``sweeps`` grouped Hildreth sweeps
    followed by ``refine_iters`` Goldfarb-Idnani steps (fixed trip counts), then
    selects per element by the number of active shares. There is no
    data-dependent Python control flow, so ``torch.compile(stage_b_fixed)``
    produces one graph per neighbour-table shape. Elements with two or more
    active shares whose result is not certified (``optimal = False``) should be
    re-solved with :func:`solve_stage_b_qp`; ``ok`` is the same exact
    feasibility check in both paths, so an uncertified result is still safe to
    execute when ``ok`` holds. With ``multi=False`` only the closed form is
    computed (the common, at most one active share, case) and every element
    with several active shares is reported as not optimal.
    """
    dt, dev = u_hat.dtype, u_hat.device
    feas_tol = 1e4 * torch.finfo(dt).eps if feas_tol is None else feas_tol
    d = u_hat.shape[-1]
    N = g.shape[-2]
    batch = torch.broadcast_shapes(u_hat.shape[:-1], g.shape[:-2], zeta.shape[:-1], active.shape[:-1])
    u_hat = u_hat.expand(*batch, d)
    g = g.expand(*batch, N, d)
    zeta = zeta.expand(*batch, N)
    active = active.expand(*batch, N)
    u_hat, act, bad, n_act, zeta_t = _stage_b_prepare(u_hat, g, zeta, active, u_inf, feas_tol)

    u1, lam1 = _closed_form_first_active(u_hat, g, zeta_t, act, u_inf)
    if not multi:
        one = (n_act == 1).unsqueeze(-1)
        u = torch.where(one, u1, u_hat)
        lam = torch.where(one, lam1, torch.zeros_like(lam1))
        u, ok, max_violation = _stage_b_verify(u, g, zeta, act, bad, u_inf)
        return StageBResult(u=u, ok=ok, lam=lam, n_active=n_act,
                            method=torch.clamp(n_act, max=2).to(torch.int8), sweeps=0,
                            max_violation=max_violation, optimal=(n_act <= 1) & ok)

    G, h = stage_b_rows(g, torch.where(act, zeta_t, torch.zeros_like(zeta_t)), u_inf)
    m = N + 2 * d
    rmask = torch.cat([act, torch.ones(*batch, 2 * d, dtype=torch.bool, device=dev)], -1)
    q = (G * G).sum(-1)
    mask = rmask & (q > 0)
    h_s = torch.where(mask, h, torch.zeros_like(h))
    qinv = torch.where(mask, 1.0 / torch.where(mask, q, torch.ones_like(q)), torch.zeros_like(q))
    # the packing is a trace-time constant; call the uncached function while compiling
    pack = _pack_groups.__wrapped__ if torch.compiler.is_compiling() else _pack_groups
    gidx, gval, pos = pack(stage_b_groups(N, d), m)
    gidx, gval_f, pos = gidx.to(dev), gval.to(device=dev, dtype=dt), pos.to(dev)
    ng, gmax = gidx.shape
    Gg = (G[..., gidx, :] * gval_f.unsqueeze(-1)).movedim(-3, 0)
    WGg = Gg.transpose(-1, -2)
    hg = (h_s[..., gidx] * gval_f).movedim(-2, 0)
    qg = (qinv[..., gidx] * gval_f).movedim(-2, 0)
    lam_full = torch.zeros(*batch, m, dtype=dt, device=dev) if lam0 is None else \
        torch.clamp(lam0.expand(*batch, m), min=0.0) * mask.to(dt)
    x = u_hat - (G.transpose(-1, -2) @ lam_full.unsqueeze(-1)).squeeze(-1)
    lam_list = list((lam_full[..., gidx] * gval_f).movedim(-2, 0).unbind(0))
    live = torch.ones(*batch, 1, dtype=dt, device=dev)
    x, lam_list = _hildreth_sweeps(x, lam_list, Gg, WGg, hg, qg, live, sweeps)
    lam_h = torch.stack(lam_list, dim=-2).reshape(*batch, ng * gmax)[..., pos]
    multi = n_act >= 2
    xr, mur, acc = _goldfarb_idnani(u_hat, G, h_s, G.transpose(-1, -2), mask, feas_tol,
                                    multi, refine_iters, early_exit=False)
    u_multi = torch.where(acc.unsqueeze(-1), xr, x)
    lam_multi = torch.where(acc.unsqueeze(-1), mur, lam_h)

    one = (n_act == 1).unsqueeze(-1)
    u = torch.where(multi.unsqueeze(-1), u_multi, torch.where(one, u1, u_hat))
    lam = torch.where(multi.unsqueeze(-1), lam_multi,
                      torch.where(one, lam1, torch.zeros_like(lam1)))
    method = torch.clamp(n_act, max=2).to(torch.int8)
    u, ok, max_violation = _stage_b_verify(u, g, zeta, act, bad, u_inf)
    optimal = (~multi | acc) & ok
    return StageBResult(u=u, ok=ok, lam=lam, n_active=n_act, method=method, sweeps=sweeps,
                        max_violation=max_violation, optimal=optimal)

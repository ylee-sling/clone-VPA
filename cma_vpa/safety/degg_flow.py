# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage A of CBF-DeGG-Flow: flow-time guidance by endpoint projection (Section 4.6.4).

The clone's conditional flow-matching solver integrates ``dA/dtau = v_theta(A, tau)``
from ``A(0) = xi`` with ``K`` Euler steps. Because the probability path is
linear, the posterior-mean endpoint is explicit (Lemma 16)::

    A_hat_{1,k} = A_k + (1 - tau_k) v_theta(A_k, tau_k),        tau_k = k / K.

CBF-DeGG-Flow projects this endpoint onto a polyhedron ``P_t`` in a weighted
norm and adds the minimal guidance that moves it there (Eq. 32)::

    Gamma*(tau_k) = nu_k (Pi_W(A_hat) - A_hat) / (1 - tau_k)
    A_{k+1}       = A_k + (v_theta + Gamma*) / K
                  = (1 - w_k) A_k + w_k [(1 - nu_k) A_hat + nu_k Pi_W(A_hat)],  w_k = 1/(K-k)

(Lemma 17(a); the second form is what is computed, because at ``k = K - 1``
it gives ``A_K = Pi_W(A_hat)`` exactly when ``nu_{K-1} = 1`` - Lemma 17(c)).

Construction of ``P_t`` (:func:`build_stage_a_polytope`)
---------------------------------------------------------
Only the translational knots are constrained. Velocities are world-frame
``V_k = R (s * a_k + c)``, where ``a_k`` are the first three channels of the
normalized chunk (heading frame frozen at ``s_t``), so every constraint below
is a linear inequality in ``a``:

* reciprocal shares at knots ``k = 0..H-2`` (Eq. 27), linearized once per
  decision along the reference trajectory (previous chunk shifted by the
  elapsed time, or constant velocity for the first decision) with neighbours
  extrapolated at constant velocity::

      g_{ij,k} . u_k >= zeta_{ij,k},   u_k = (V_{k+1} - V_k) / Delta
      <=>  -g_{ij,k} . (V_{k+1} - V_k) <= -Delta zeta_{ij,k}

* acceleration box ``|V_{k+1,l} - V_{k,l}| <= Delta u_inf`` (world axes),
* speed box ``|V_{k,l}| <= v_inf`` (world axes),
* normalization box ``|a_{k,c}| <= 1`` so that Algorithm 2's final
  "denormalize and clip" does not leave ``P_t``.

Shares that are certified inactive (Lemmas 14, 15) or not heard are masked;
ill-posed shares (infinite margin) and shares that no input in the box can
satisfy (Corollary 4) are also masked here, because they would make ``P_t``
empty - Stage A carries no safety guarantee, and Stage B engages the fallback
if such a state materializes. The weight
``W`` is block diagonal: an arbitrary positive diagonal on the translational
channels (identity by default) and the yaw-rate channel is not constrained, so
the projection leaves it unchanged.

Every row touches at most two consecutive knots, so rows of the same kind and
slot at knots of equal parity never interact; :func:`build_stage_a_polytope`
returns this partition, which turns Hildreth's ``O(H (N_c + 18))`` row updates
per sweep into ``2 N_c + 24`` vectorized group updates with identical iterates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .margins import ShareSet, reciprocal_shares
from .params import DEFAULT_PARAMS, SafetyParams
from .qp_solvers import hildreth

__all__ = [
    "ChunkMap",
    "Polyhedron",
    "StageAPolytope",
    "GuidedSolveResult",
    "constant_velocity_reference",
    "shift_reference",
    "integrate_positions",
    "build_stage_a_polytope",
    "default_nu_schedule",
    "guided_solve",
]

VelocityField = Callable[[Tensor, float], Tensor]


# ---------------------------------------------------------------------------
# chunk coordinates
# ---------------------------------------------------------------------------
@dataclass
class ChunkMap:
    """Affine map from normalized translational chunk channels to world velocities.

    ``V_world = R (scale * a + offset)`` with ``R (..., 3, 3)`` the heading-to-world
    rotation frozen at the decision time ``s_t``.
    """

    rotation: Tensor
    scale: Tensor
    offset: Tensor

    @staticmethod
    def from_yaw(yaw: Tensor, scale: Union[Tensor, float, Sequence[float]] = 8.0,
                 offset: Union[Tensor, float, Sequence[float]] = 0.0) -> "ChunkMap":
        """Rotation about ``z`` by ``yaw`` (shape ``(...)``), per-axis ``scale`` and ``offset``."""
        c, s = torch.cos(yaw), torch.sin(yaw)
        z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
        R = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1),
                         torch.stack([z, z, o], -1)], -2)
        sc = torch.as_tensor(scale, dtype=yaw.dtype, device=yaw.device).expand(*yaw.shape, 3)
        of = torch.as_tensor(offset, dtype=yaw.dtype, device=yaw.device).expand(*yaw.shape, 3)
        return ChunkMap(R, sc, of)

    @property
    def M(self) -> Tensor:
        """Linear part ``R diag(scale)`` ``(..., 3, 3)``."""
        return self.rotation * self.scale.unsqueeze(-2)

    @property
    def c_world(self) -> Tensor:
        """Constant part ``R offset`` ``(..., 3)``."""
        return (self.rotation @ self.offset.unsqueeze(-1)).squeeze(-1)

    def to_world(self, a: Tensor) -> Tensor:
        """World velocities ``(..., H, 3)`` of normalized translational channels ``(..., H, 3)``."""
        return torch.einsum("...ij,...hj->...hi", self.M, a) + self.c_world.unsqueeze(-2)

    def to_normalized(self, V: Tensor) -> Tensor:
        """Inverse of :meth:`to_world`."""
        local = torch.einsum("...ji,...hj->...hi", self.rotation, V)   # R^T V
        return (local - self.offset.unsqueeze(-2)) / self.scale.unsqueeze(-2)


@dataclass
class Polyhedron:
    """``{x : G x <= h}`` on the flattened translational chunk ``x = a[..., :c].reshape(B, H c)``.

    ``row_mask`` disables rows, ``groups`` is an optional partition into
    non-interacting rows (see :func:`cma_vpa.safety.qp_solvers.hildreth`),
    ``W_inv`` the inverse projection weight (``None`` = identity).
    """

    G: Tensor
    h: Tensor
    row_mask: Optional[Tensor] = None
    groups: Optional[Tuple[Tuple[int, ...], ...]] = None
    W_inv: Optional[Tensor] = None

    def residual(self, x: Tensor) -> Tensor:
        """``(G x - h)_+`` on unmasked rows, ``(B, m)``."""
        r = torch.clamp((self.G @ x.unsqueeze(-1)).squeeze(-1) - self.h, min=0.0)
        if self.row_mask is not None:
            r = torch.where(self.row_mask, r, torch.zeros_like(r))
        return r

    def contains(self, x: Tensor, tol: float = 1e-9) -> Tensor:
        """``G x <= h + tol (1 + |h|)`` on unmasked rows."""
        h_abs = torch.where(self.row_mask, self.h.abs(), torch.zeros_like(self.h)) \
            if self.row_mask is not None else self.h.abs()
        return (self.residual(x) <= tol * (1.0 + h_abs)).all(-1)


@dataclass
class StageAPolytope(Polyhedron):
    """``P_t`` of one decision with its row layout and the shares it was built from.

    ``layout`` maps ``"share" | "accel" | "speed" | "chunk_box"`` to ``(start, stop)``
    row ranges; share rows are ordered ``(k, j)``, the others ``(k, slot)`` with
    slots ``(+x, +y, +z, -x, -y, -z)``.
    """

    layout: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    shares: Optional[ShareSet] = None
    chunk_map: Optional[ChunkMap] = None
    H: int = 0
    N: int = 0


# ---------------------------------------------------------------------------
# reference trajectory (frozen linearization point)
# ---------------------------------------------------------------------------
def constant_velocity_reference(v_hat: Tensor, H: int) -> Tensor:
    """Reference velocities ``(..., H, 3)`` equal to the current estimate (first decision)."""
    return v_hat.unsqueeze(-2).expand(*v_hat.shape[:-1], H, v_hat.shape[-1]).clone()


def shift_reference(V_prev: Tensor, shift_knots: Union[Tensor, float]) -> Tensor:
    """Previous world-frame chunk velocities shifted by ``shift_knots`` (may be fractional).

    ``V_ref[k] = V_prev(k + shift)`` with linear interpolation between knots and
    the last knot held beyond the end of the previous chunk.
    """
    H = V_prev.shape[-2]
    s = torch.as_tensor(shift_knots, dtype=V_prev.dtype, device=V_prev.device)
    t = torch.arange(H, dtype=V_prev.dtype, device=V_prev.device) + s.unsqueeze(-1)   # (..., H)
    t = torch.clamp(t, 0.0, H - 1.0)
    i0 = torch.floor(t).long()
    i1 = torch.clamp(i0 + 1, max=H - 1)
    frac = (t - i0.to(t.dtype)).unsqueeze(-1)
    shape = (*V_prev.shape[:-2], H, V_prev.shape[-1])
    v0 = torch.gather(V_prev, -2, i0.unsqueeze(-1).expand(shape))
    v1 = torch.gather(V_prev, -2, i1.unsqueeze(-1).expand(shape))
    return (1.0 - frac) * v0 + frac * v1


def integrate_positions(p0: Tensor, V: Tensor, dt: float) -> Tensor:
    """Exact positions of the piecewise-linear velocity ``V (..., H, 3)``: ``p_{k+1} = p_k + dt (V_k + V_{k+1}) / 2``."""
    inc = 0.5 * dt * (V[..., :-1, :] + V[..., 1:, :])
    return torch.cat([p0.unsqueeze(-2), p0.unsqueeze(-2) + torch.cumsum(inc, dim=-2)], dim=-2)


# ---------------------------------------------------------------------------
# P_t
# ---------------------------------------------------------------------------
_SLOT_AXIS = (0, 1, 2, 0, 1, 2)
_SLOT_SIGN = (1.0, 1.0, 1.0, -1.0, -1.0, -1.0)


def _stage_a_groups(H: int, N: int, include_chunk_box: bool) -> Tuple[Tuple[int, ...], ...]:
    nS, nA, nV = (H - 1) * N, (H - 1) * 6, H * 6
    groups: List[Tuple[int, ...]] = []
    for parity in (0, 1):
        ks = range(parity, H - 1, 2)
        for j in range(N):
            groups.append(tuple(k * N + j for k in ks))
        for s in range(6):
            groups.append(tuple(nS + k * 6 + s for k in ks))
    for s in range(6):
        groups.append(tuple(nS + nA + k * 6 + s for k in range(H)))
    if include_chunk_box:
        for s in range(6):
            groups.append(tuple(nS + nA + nV + k * 6 + s for k in range(H)))
    return tuple(g for g in groups if g)


def build_stage_a_polytope(p_hat: Tensor, v_hat: Tensor, V_ref: Tensor, nbr_pos: Tensor,
                           nbr_vel: Tensor, chunk_map: ChunkMap, *,
                           D: Union[Tensor, float] = DEFAULT_PARAMS.D_nominal,
                           heard: Optional[Tensor] = None, cfg: SafetyParams = DEFAULT_PARAMS,
                           v_inf: Optional[float] = None, u_inf: Optional[float] = None,
                           include_chunk_box: bool = True, W_trans: Optional[Tensor] = None,
                           e_p: Optional[float] = None, e_v: Optional[float] = None) -> StageAPolytope:
    """Construct ``P_t = {a : G a <= h}`` for one decision of every agent in the batch.

    Parameters
    ----------
    p_hat, v_hat:
        Own position and velocity estimate at the decision time ``s_t``, ``(B, 3)``.
    V_ref:
        World-frame reference velocities at the knots ``(B, H, 3)`` along which
        the shares are linearized (:func:`shift_reference` or
        :func:`constant_velocity_reference`). Reference positions are integrated
        from ``p_hat`` with the trapezoidal rule (exact for the piecewise-linear
        velocity of the chunk).
    nbr_pos, nbr_vel:
        Neighbour positions and velocities at ``s_t`` (latest broadcast already
        extrapolated), ``(B, N, 3)``; they are extrapolated at constant velocity
        to every knot.
    chunk_map:
        Normalized heading-frame chunk to world-velocity map (batch ``B``).
    D, heard:
        Collision distances and heard-masks of the neighbours, ``(B, N)``.
    v_inf, u_inf:
        Per-axis speed and acceleration bounds (defaults ``cfg.v_bar``, ``cfg.u_inf``).
    include_chunk_box:
        Add ``|a| <= 1`` rows (see the module docstring).
    W_trans:
        Optional positive diagonal weight ``(B, H, 3)`` or ``(H, 3)`` on the
        translational channels (default identity).
    e_p, e_v:
        Estimation bounds passed to :func:`reciprocal_shares`.
    """
    dt_, dev = p_hat.dtype, p_hat.device
    B, H = V_ref.shape[0], V_ref.shape[-2]
    N = nbr_pos.shape[-2]
    Dt = cfg.T_c
    v_inf = cfg.v_bar if v_inf is None else v_inf
    u_inf = cfg.u_inf if u_inf is None else u_inf
    if heard is None:
        heard = torch.ones(B, N, dtype=torch.bool, device=dev)
    D_t = torch.as_tensor(D, dtype=dt_, device=dev).expand(B, N)

    # -- reciprocal shares along the reference (knots 0 .. H-2) ----------------
    P_ref = integrate_positions(p_hat, V_ref, Dt)                               # (B, H, 3)
    tk = Dt * torch.arange(H - 1, dtype=dt_, device=dev)                        # (H-1,)
    Pj = nbr_pos.unsqueeze(1) + tk.view(1, -1, 1, 1) * nbr_vel.unsqueeze(1)     # (B, H-1, N, 3)
    dp = P_ref[:, :-1].unsqueeze(2) - Pj
    dv = V_ref[:, :-1].unsqueeze(2) - nbr_vel.unsqueeze(1)
    S = reciprocal_shares(dp, dv, D_t.unsqueeze(1).expand(B, H - 1, N), cfg,
                          heard=heard.unsqueeze(1).expand(B, H - 1, N), e_p=e_p, e_v=e_v)
    # Corollary 4 pre-screen: a share that no input in the box satisfies would
    # empty P_t; it is left to Stage B (which brakes if the state materializes).
    share_on = (S.active & torch.isfinite(S.zeta)
                & (S.zeta <= u_inf * S.g.abs().sum(-1)))
    g = torch.where(share_on.unsqueeze(-1), S.g, torch.zeros_like(S.g))
    zeta = torch.where(share_on, S.zeta, torch.zeros_like(S.zeta))

    # -- rows in world-velocity coordinates V (H, 3) ---------------------------
    eyeH = torch.eye(H, dtype=dt_, device=dev)
    diff = eyeH[:-1] - eyeH[1:]                         # row k: V_k - V_{k+1}
    G_share = torch.einsum("kh,bkjl->bkjhl", diff, g).reshape(B, (H - 1) * N, H, 3)
    h_share = (-Dt * zeta).reshape(B, (H - 1) * N)

    sign = torch.tensor(_SLOT_SIGN, dtype=dt_, device=dev)
    axis = torch.eye(3, dtype=dt_, device=dev)[list(_SLOT_AXIS)]                # (6, 3)
    slot = sign.unsqueeze(-1) * axis                                            # (6, 3)
    G_acc = torch.einsum("kh,sl->kshl", -diff, slot).reshape(1, (H - 1) * 6, H, 3).expand(B, -1, -1, -1)
    h_acc = torch.full((B, (H - 1) * 6), Dt * u_inf, dtype=dt_, device=dev)
    G_spd = torch.einsum("kh,sl->kshl", eyeH, slot).reshape(1, H * 6, H, 3).expand(B, -1, -1, -1)
    h_spd = torch.full((B, H * 6), v_inf, dtype=dt_, device=dev)

    G_w = torch.cat([G_share, G_acc, G_spd], dim=1)                             # (B, m_w, H, 3)
    h_w = torch.cat([h_share, h_acc, h_spd], dim=1)
    # pull back through V = M a + c
    G_n = torch.einsum("bmhi,bij->bmhj", G_w, chunk_map.M)
    h_n = h_w - torch.einsum("bmhi,bi->bm", G_w, chunk_map.c_world)
    G_list, h_list = [G_n.reshape(B, -1, 3 * H)], [h_n]
    if include_chunk_box:
        G_box = torch.einsum("kh,sl->kshl", eyeH, slot).reshape(1, H * 6, 3 * H).expand(B, -1, -1)
        G_list.append(G_box)
        h_list.append(torch.ones(B, H * 6, dtype=dt_, device=dev))
    G = torch.cat(G_list, dim=1)
    h = torch.cat(h_list, dim=1)
    m = G.shape[1]
    nS, nA, nV = (H - 1) * N, (H - 1) * 6, H * 6
    row_mask = torch.ones(B, m, dtype=torch.bool, device=dev)
    row_mask[:, :nS] = share_on.reshape(B, nS)
    layout = {"share": (0, nS), "accel": (nS, nS + nA), "speed": (nS + nA, nS + nA + nV)}
    if include_chunk_box:
        layout["chunk_box"] = (nS + nA + nV, m)
    W_inv = None
    if W_trans is not None:
        W_inv = (1.0 / torch.as_tensor(W_trans, dtype=dt_, device=dev)).expand(B, H, 3).reshape(B, 3 * H)
    return StageAPolytope(G=G, h=h, row_mask=row_mask, groups=_stage_a_groups(H, N, include_chunk_box),
                          W_inv=W_inv, layout=layout, shares=S, chunk_map=chunk_map, H=H, N=N)


# ---------------------------------------------------------------------------
# Algorithm 2: GUIDEDSOLVE
# ---------------------------------------------------------------------------
def _w_norm(x: Tensor, W_inv: Optional[Tensor]) -> Tensor:
    """``||x||_W = sqrt(x^T W x)`` given ``W^{-1}`` (``None``, diagonal ``(..., n)`` or dense ``(..., n, n)``)."""
    if W_inv is None:
        return torch.linalg.vector_norm(x, dim=-1)
    if W_inv.dim() == x.dim():
        return torch.sqrt((x * x / W_inv).sum(-1))
    z = torch.linalg.solve(W_inv, x.unsqueeze(-1)).squeeze(-1)
    return torch.sqrt(torch.clamp((x * z).sum(-1), min=0.0))


def default_nu_schedule(K: int) -> List[float]:
    """Table 4: ``nu_k = 0.5`` for ``k < K - 1`` and ``nu_{K-1} = 1``."""
    return [0.5] * (K - 1) + [1.0]


@dataclass
class GuidedSolveResult:
    """Output of :func:`guided_solve`.

    Attributes
    ----------
    A:
        Normalized chunk ``(B, H, d_a)`` (clipped to ``[-1, 1]`` if requested).
    lam:
        Final dual multipliers (warm start for the next decision with the same row layout).
    terminal_feasible:
        ``A_K`` (after clipping) lies in ``P_t`` within the solver tolerance.
    terminal_exact:
        The final projection carries a KKT certificate (exact ``Pi_W``).
    guided:
        Guidance was applied at the final step (``False`` where the final
        projection failed, e.g. an empty ``P_t``; those elements return the
        unguided endpoint).
    intervention:
        ``||Gamma*(tau_k)||_W`` per step, ``(K, B)``.
    endpoints:
        Endpoint estimates ``A_hat_{1,k}`` per step, ``(K, B, H, d_a)``.
    sweeps:
        Hildreth sweeps used at each step.
    """

    A: Tensor
    lam: Tensor
    terminal_feasible: Tensor
    terminal_exact: Tensor
    guided: Tensor
    intervention: Tensor
    endpoints: Tensor
    sweeps: List[int]


def guided_solve(velocity_fn: VelocityField, xi: Tensor, polytope: Polyhedron, K: int,
                 nu: Optional[Sequence[float]] = None, *, n_channels: int = 3, n_sweeps: int = 5,
                 final_sweeps: int = 10, lam0: Optional[Tensor] = None, polish_final: bool = True,
                 clip: bool = True, compile: bool = False, feas_tol: Optional[float] = None) -> GuidedSolveResult:
    """Real-time guided flow-matching solver (Algorithm 2, ``GUIDEDSOLVE``).

    Parameters
    ----------
    velocity_fn:
        ``v_theta(A, tau)`` with the conditioning bound in (one network
        evaluation per call); ``A`` has shape ``(B, H, d_a)``.
    xi:
        Initial noise ``A(0)`` ``(B, H, d_a)`` (normalized chunk space).
    polytope:
        ``P_t`` on the first ``n_channels`` channels flattened in knot-major
        order (:func:`build_stage_a_polytope`, or any :class:`Polyhedron`).
    K:
        Number of Euler steps.
    nu:
        Guidance schedule (default :func:`default_nu_schedule`); ``nu[K-1]`` must
        be 1 for Lemma 17(c).
    n_sweeps, final_sweeps:
        Hildreth sweeps per intermediate step (warm-started across steps, as in
        Algorithm 2) and before the exact polish of the final projection.
        Intermediate projections are approximate by design; the final one is
        made exact by the polish (PDAS, then Goldfarb-Idnani), so more final
        sweeps only shorten the polish.
    lam0:
        Dual warm start from the previous decision.
    polish_final:
        Exact active-set polish of the final projection (KKT certificate).
    clip:
        Clip the output to ``[-1, 1]`` ("denormalize and clip").
    compile:
        Use the compiled Hildreth sweep kernel.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    nu = list(default_nu_schedule(K) if nu is None else nu)
    if len(nu) != K or not all(0.0 <= v <= 1.0 for v in nu):
        raise ValueError("nu must have K entries in [0, 1]")
    B, H, da = xi.shape
    c = n_channels
    A = xi.clone()
    lam = lam0
    interventions, endpoints, sweeps = [], [], []
    guided = torch.ones(B, dtype=torch.bool, device=xi.device)
    exact = torch.zeros_like(guided)
    W_inv = polytope.W_inv
    for k in range(K):
        tau = k / K
        V = velocity_fn(A, tau)                                   # one network evaluation
        A1 = A + (1.0 - tau) * V                                  # Lemma 16
        endpoints.append(A1)
        y = A1[..., :c].reshape(B, H * c)
        last = k == K - 1
        hr = hildreth(y, polytope.G, polytope.h, W_inv=W_inv, lam0=lam, row_mask=polytope.row_mask,
                      groups=polytope.groups, max_sweeps=final_sweeps if last else n_sweeps,
                      polish=polish_final and last, polish_during=False, compile=compile,
                      feas_tol=feas_tol)
        lam = hr.lam
        sweeps.append(hr.sweeps)
        P = hr.x
        nu_k = nu[k]
        step_ok = hr.feasible if last else torch.ones_like(guided)
        if last:
            guided = step_ok
            exact = hr.exact
        target = torch.where(step_ok.unsqueeze(-1), y + nu_k * (P - y), y)
        delta = target - y                                         # = (1 - tau) Gamma*
        interventions.append(_w_norm(delta, W_inv) / (1.0 - tau))
        w = 1.0 / (K - k)
        A_next = (1.0 - w) * A + w * A1
        trans = (1.0 - w) * A[..., :c] + w * target.view(B, H, c)
        A = torch.cat([trans, A_next[..., c:]], dim=-1)
    if clip:
        A = torch.clamp(A, -1.0, 1.0)
    x_out = A[..., :c].reshape(B, H * c)
    tol = 1e4 * torch.finfo(A.dtype).eps if feas_tol is None else feas_tol
    terminal_feasible = polytope.contains(x_out, tol=10 * tol)
    return GuidedSolveResult(A=A, lam=lam, terminal_feasible=terminal_feasible, terminal_exact=exact & guided,
                             guided=guided, intervention=torch.stack(interventions),
                             endpoints=torch.stack(endpoints), sweeps=sweeps)

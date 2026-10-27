# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Safety layer for arms commanded by end-effector twists (Remark 1, Task T2).

An arm is abstracted at the centers of collision spheres rigidly attached to its
end effector (wrist / flange, hand, finger tips). With TCP position ``p``,
orientation ``R`` and world-frame twist ``(v, w)``, sphere ``k`` with offset
``o_k`` (TCP frame) has lever arm ``r_k = R o_k`` and

    c_k     = p + r_k,
    c_k_dot = v + w x r_k                          (linear in the twist),
    c_k_ddot = v_dot + w_dot x r_k + w x (w x r_k)  (affine in the twist rate).

The twist rate ``u = (v_dot, w_dot) in R^6`` is the commanded input. The drift
``w x (w x r)`` is bounded by ``||w||^2 ||r||`` and is added to the disturbance
bound (:func:`arm_safety_params`):

    e_a = e_track + w_max^2 r_max.

Every sphere center therefore satisfies Assumption 1 with a per-axis input bound
``u_inf = a_lin + sqrt(2) a_ang r_max`` (``|(w_dot x r)_k| <= sqrt(2) a_ang ||r||``)
and speed bound ``v_bar = v_max + w_max r_max``. The barrier, the reciprocal
shares and every margin of Section 4.6 are then evaluated for **sphere pairs**
between different arms with :func:`~cma_vpa.safety.margins.reciprocal_shares`.
A share ``g^T c_ddot >= zeta`` of sphere ``k`` becomes a row in twist-rate space,

    g^T (v_dot + w_dot x r_k) = [g ; r_k x g]^T u >= zeta          (:func:`twist_rows`),

so Stage B (Eq. 34) stays a QP with ``d = 6`` variables. The per-axis speed
limits are imposed as box bounds on ``u``: ``(-xi_bar - xi_hat)/T_c <= u <= (xi_bar - xi_hat)/T_c``.
Together with the acceleration box this gives an asymmetric box, and Lemma 18
applies unchanged (:func:`~cma_vpa.safety.qp_solvers.project_box_halfspace`
takes general bounds). The QP is solved in coordinates normalized by the
acceleration bounds (``y = u / a``), i.e. Eq. (34) is the projection in the metric
``sum_k (u_k - u_hat_k)^2 / a_k^2``. This is the plain Euclidean projection when all
axes share a bound, and it makes linear and angular channels commensurable.
The gripper channel passes through unmodified (Section 4.5).

The fallback brakes the whole twist with the jerk-optimal profile of Eq. (36),
``xi(sigma) = xi_0 (1 - 3 x^2 + 2 x^3)`` with ``x = sigma / T_b``, one common duration
``T_b = 3/2 max(||v_0|| / a_brk_lin, ||w_0|| / a_brk_ang)`` (both channels stop
together and neither exceeds its braking bound), and then holds the pose.

Collisions between proximal links are outside the barrier (workspace
partitioning, Remark 1); the simulator reports them as a separate metric.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .fallback import fallback_envelope
from .margins import ShareSet, reciprocal_shares
from .params import SafetyParams
from .qp_solvers import hildreth, project_box_halfspace

__all__ = [
    "ArmLimits", "SphereSet", "PANDA_SPHERES", "SphereState", "sphere_kinematics", "sphere_jacobian",
    "sphere_acceleration", "centrifugal_drift", "drift_bound", "arm_safety_params", "twist_rows", "twist_box",
    "TwistQPResult", "solve_twist_qp", "ArmShareSet", "arm_shares", "TwistBraking", "SphereTable",
    "SphereBroadcast", "ArmExecutorConfig", "ArmCommand", "ArmExecutor", "TwistChunkBuffer",
]


# ---------------------------------------------------------------------------
# limits and collision spheres
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmLimits:
    """Per-axis twist and twist-rate bounds of one arm (world frame), and the tracking error.

    The defaults are within the Franka Panda Cartesian limits (1.7 m/s, 2.5 rad/s,
    13 m/s^2, 25 rad/s^2) and are typical of assembly motions.
    """

    v_axis: float = 0.4        # |v_k| bound [m/s]
    w_axis: float = 0.7        # |w_k| bound [rad/s]
    a_lin: float = 6.0         # |v_dot_k| bound [m/s^2]
    a_ang: float = 10.0        # |w_dot_k| bound [rad/s^2]
    e_track: float = 0.3       # sphere-level acceleration tracking error [m/s^2]
    a_brk_lin: float = 3.0     # braking deceleration bounds of the fallback
    a_brk_ang: float = 6.0

    @property
    def v_max(self) -> float:
        return math.sqrt(3.0) * self.v_axis

    @property
    def w_max(self) -> float:
        return math.sqrt(3.0) * self.w_axis

    def twist_bound(self, dtype=torch.float64, device=None) -> Tensor:
        return torch.tensor([self.v_axis] * 3 + [self.w_axis] * 3, dtype=dtype, device=device)

    def accel_bound(self, dtype=torch.float64, device=None) -> Tensor:
        return torch.tensor([self.a_lin] * 3 + [self.a_ang] * 3, dtype=dtype, device=device)


@dataclass(frozen=True)
class SphereSet:
    """Collision spheres in the TCP frame: names, offsets ``o_k`` [m], physical radii, safety margin."""

    names: Tuple[str, ...]
    offsets: Tuple[Tuple[float, float, float], ...]
    radii: Tuple[float, ...]
    margin: float = 0.01

    def __post_init__(self):
        if not (len(self.names) == len(self.offsets) == len(self.radii)):
            raise ValueError("names, offsets and radii must have the same length")

    @property
    def n(self) -> int:
        return len(self.names)

    def offset_tensor(self, dtype=torch.float64, device=None) -> Tensor:
        return torch.tensor(self.offsets, dtype=dtype, device=device)

    def radius_tensor(self, dtype=torch.float64, device=None) -> Tensor:
        return torch.tensor(self.radii, dtype=dtype, device=device)

    def safety_radius_tensor(self, dtype=torch.float64, device=None) -> Tensor:
        return self.radius_tensor(dtype, device) + self.margin

    @property
    def lever_max(self) -> float:
        return max(math.sqrt(sum(c * c for c in o)) for o in self.offsets)


#: Franka Panda hand (TCP frame: z along the approach direction, y across the fingers).
#: Finger tips (one sphere covers both open fingers), the two halves of the hand
#: housing, the flange and the link-7 body - all rigidly attached to the end effector.
PANDA_SPHERES = SphereSet(
    names=("tcp", "hand_a", "hand_b", "flange", "wrist"),
    offsets=((0.0, 0.0, -0.015), (0.0, 0.055, -0.065), (0.0, -0.055, -0.065), (0.0, 0.0, -0.115),
             (0.0, 0.0, -0.175)),
    radii=(0.035, 0.045, 0.045, 0.055, 0.06),
)


# ---------------------------------------------------------------------------
# Remark 1: sphere kinematics
# ---------------------------------------------------------------------------
@dataclass
class SphereState:
    center: Tensor     # (..., S, 3)
    velocity: Tensor   # (..., S, 3)
    lever: Tensor      # (..., S, 3)  r_k = c_k - p


def sphere_kinematics(p: Tensor, R: Tensor, v: Tensor, w: Tensor, offsets: Tensor) -> SphereState:
    """Centers, velocities ``v + w x r`` and lever arms of spheres with TCP-frame ``offsets (S, 3)``.

    ``p (..., 3)``, ``R (..., 3, 3)``, ``v``, ``w`` ``(..., 3)`` (world frame).
    """
    r = torch.einsum("...ij,sj->...si", R, offsets.to(R))
    c = p.unsqueeze(-2) + r
    cd = v.unsqueeze(-2) + torch.cross(w.unsqueeze(-2).expand_as(r), r, dim=-1)
    return SphereState(c, cd, r)


def sphere_jacobian(r: Tensor) -> Tensor:
    """``J_c = [I, -[r]_x]`` with ``c_dot = J_c (v, w)``: ``(..., 3) -> (..., 3, 6)``."""
    z = torch.zeros_like(r[..., 0])
    skew = torch.stack([torch.stack([z, -r[..., 2], r[..., 1]], -1),
                        torch.stack([r[..., 2], z, -r[..., 0]], -1),
                        torch.stack([-r[..., 1], r[..., 0], z], -1)], -2)
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(*r.shape[:-1], 3, 3)
    return torch.cat([eye, -skew], -1)


def centrifugal_drift(w: Tensor, r: Tensor) -> Tensor:
    """``w x (w x r)``."""
    return torch.cross(w, torch.cross(w, r, dim=-1), dim=-1)


def drift_bound(w: Tensor, r: Tensor) -> Tensor:
    """``||w||^2 ||r||`` (attained when ``w`` is orthogonal to ``r``)."""
    return torch.linalg.vector_norm(w, dim=-1) ** 2 * torch.linalg.vector_norm(r, dim=-1)


def sphere_acceleration(r: Tensor, w: Tensor, u: Tensor) -> Tensor:
    """``c_ddot = v_dot + w_dot x r + w x (w x r)`` for twist rate ``u = (v_dot, w_dot)``."""
    return u[..., :3] + torch.cross(u[..., 3:], r, dim=-1) + centrifugal_drift(w, r)


def arm_safety_params(limits: ArmLimits = ArmLimits(), spheres: SphereSet = PANDA_SPHERES, *,
                      a_max: float = 3.0, varsigma: float = 0.4, lambda_b: float = 6.0, e_p: float = 0.003,
                      e_v: float = 0.02, e_tr: float = 0.05, T_c: float = 0.01, T_r: float = 0.02,
                      R_link: float = 3.0) -> SafetyParams:
    """Sphere-space :class:`SafetyParams` of an arm (Remark 1).

    ``u_inf = a_lin + sqrt(2) a_ang r_max`` bounds every axis of ``v_dot + w_dot x r``;
    ``e_a = e_track + w_max^2 r_max`` adds the centrifugal drift bound to the
    tracking error; ``v_bar = v_max + w_max r_max``; ``r_s`` / ``r_max`` are the
    smallest and largest safety radius of the sphere set; ``e_p`` / ``e_v`` cover
    the 1 mm / 1 mm/s broadcast quantization and the state estimate.

    The defaults are chosen for collision distances of 0.09-0.14 m. A larger
    regularizer ``varsigma = 0.4 m/s`` than for quadrotors keeps the curvature
    ``alpha_1''(0) = -a (a D - varsigma^2) / (2 varsigma^3 D^2)`` moderate (-328 instead of
    -2280 for ``varsigma = 0.25``). That curvature enters the one-tick margin ``T_c Lambda``
    (Lemma 13), and with it every well-posed pairwise share is feasible with the
    linear channel alone (``tests/test_dual_arm.py``).
    """
    lever = spheres.lever_max
    rs = spheres.safety_radius_tensor()
    return SafetyParams(a_max=a_max, varsigma=varsigma, lambda_b=lambda_b, r_s=float(rs.min()),
                        r_max=float(rs.max()), kappa_sigma=0.0,
                        u_inf=limits.a_lin + math.sqrt(2.0) * limits.a_ang * lever,
                        e_a=limits.e_track + limits.w_max ** 2 * lever, e_p=e_p, e_v=e_v, e_tr=e_tr,
                        v_bar=limits.v_max + limits.w_max * lever, T_c=T_c, d=3, a_brk=limits.a_brk_lin,
                        T_r=T_r, R_link=R_link)


# ---------------------------------------------------------------------------
# twist-space Stage B
# ---------------------------------------------------------------------------
def twist_rows(g: Tensor, lever: Tensor) -> Tensor:
    """Rows ``[g ; r x g]`` of the shares in twist-rate space: ``(..., N, 3) x (..., N, 3) -> (..., N, 6)``."""
    return torch.cat([g, torch.cross(lever, g, dim=-1)], -1)


def twist_box(xi_hat: Tensor, limits: ArmLimits, T_c: float) -> Tuple[Tensor, Tensor]:
    """Per-axis bounds ``[lo, hi]`` on the twist rate: acceleration box intersected with the speed limits.

    If the current twist already exceeds a speed limit by more than one tick of
    full acceleration, both bounds equal the maximal deceleration on that axis.
    """
    vb = limits.twist_bound(xi_hat.dtype, xi_hat.device)
    ab = limits.accel_bound(xi_hat.dtype, xi_hat.device)
    hi = torch.minimum(torch.maximum((vb - xi_hat) / T_c, -ab), ab)
    lo = torch.minimum(torch.maximum((-vb - xi_hat) / T_c, -ab), ab)
    return torch.minimum(lo, hi), hi


@dataclass
class TwistQPResult:
    u: Tensor               # (B, 6) projected twist rate
    ok: Tensor              # (B,) exact feasibility of every active share
    n_active: Tensor        # (B,)
    method: Tensor          # (B,) 0 none, 1 closed form (Lemma 18), 2 Hildreth + exact polish
    optimal: Tensor         # (B,) KKT-certified minimizer
    max_violation: Tensor   # (B,)


def solve_twist_qp(u_hat: Tensor, rows: Tensor, zeta: Tensor, active: Tensor, lo: Tensor, hi: Tensor,
                   scale: Tensor, *, max_sweeps: int = 50, feas_tol: Optional[float] = None) -> TwistQPResult:
    """Eq. (34) for twist rates: ``min sum_k ((u_k - u^_k)/a_k)^2  s.t.  rows_j . u >= zeta_j, lo <= u <= hi``.

    ``u_hat, lo, hi (B, d)``, ``rows (B, N, d)``, ``zeta, active (B, N)``, ``scale (d,)``.
    Routed like :func:`~cma_vpa.safety.qp_solvers.solve_stage_b_qp`: no active
    share - clip; one - Lemma 18 (closed form, general box); several - grouped
    Hildreth with the exact PDAS / Goldfarb-Idnani polish. Constraints are
    tightened by a tolerance-proportional amount and checked exactly afterwards.
    """
    dt, dev = u_hat.dtype, u_hat.device
    feas_tol = 1e4 * torch.finfo(dt).eps if feas_tol is None else feas_tol
    B, N, d = rows.shape
    s = scale.to(dt)
    lo_y, hi_y = lo / s, hi / s
    y_hat = torch.minimum(torch.maximum(u_hat / s, lo_y), hi_y)
    ry = rows * s
    finite = torch.isfinite(zeta) & torch.isfinite(ry).all(-1)
    bad = (active & ~finite).any(-1)
    act = active & finite
    n_act = act.sum(-1)
    zc = torch.where(act, zeta, torch.zeros_like(zeta))
    ry_c = torch.where(act.unsqueeze(-1), ry, torch.zeros_like(ry))
    box_mag = torch.maximum(lo_y.abs(), hi_y.abs()).amax(-1, keepdim=True)
    tau = 2.0 * feas_tol * (1.0 + zc.abs() + ry_c.abs().sum(-1) * (1.0 + box_mag))
    zt = zc + tau

    y = y_hat.clone()
    method = torch.zeros(B, dtype=torch.int8, device=dev)
    optimal = n_act == 0
    one = n_act == 1
    if bool(one.any()):
        idx = torch.argmax(act.to(torch.int8), -1)
        ar = torch.arange(B, device=dev)
        res = project_box_halfspace(y_hat, ry[ar, idx], zt[ar, idx], lo_y, hi_y)
        y = torch.where(one.unsqueeze(-1), res.u, y)
        method = torch.where(one, torch.ones_like(method), method)
        optimal = optimal | (one & res.feasible)
    many = n_act >= 2
    if bool(many.any()):
        sel = many.nonzero(as_tuple=True)[0]
        k = int(n_act[sel].max())
        cidx = torch.topk(act[sel].to(torch.int8), k, dim=-1).indices
        valid = torch.gather(act[sel], -1, cidx)
        g_c = torch.gather(ry[sel], 1, cidx.unsqueeze(-1).expand(-1, k, d))
        z_c = torch.where(valid, torch.gather(zt[sel], -1, cidx), torch.zeros_like(cidx, dtype=dt))
        eye = torch.eye(d, dtype=dt, device=dev).expand(sel.shape[0], d, d)
        G = torch.cat([-g_c, eye, -eye], 1)
        h = torch.cat([-z_c, hi_y[sel], -lo_y[sel]], -1)
        rmask = torch.cat([valid, torch.ones(sel.shape[0], 2 * d, dtype=torch.bool, device=dev)], -1)
        groups = tuple((j,) for j in range(k)) + (tuple(range(k, k + d)), tuple(range(k + d, k + 2 * d)))
        hr = hildreth(y_hat[sel], G, h, row_mask=rmask, groups=groups, max_sweeps=max_sweeps,
                      feas_tol=feas_tol, polish=True, polish_iters=4, finisher=True)
        y[sel] = hr.x
        method[sel] = 2
        optimal = optimal.clone()
        optimal[sel] = hr.exact
    y = torch.minimum(torch.maximum(y, lo_y), hi_y)
    slack = (ry * y.unsqueeze(-2)).sum(-1) - zeta
    viol = torch.where(act, torch.clamp(-slack, min=0.0), torch.zeros_like(slack))
    max_violation = viol.amax(-1) if N > 0 else torch.zeros(B, dtype=dt, device=dev)
    ok = ~bad & (viol <= 0).all(-1)
    return TwistQPResult(u=y * s, ok=ok, n_active=n_act, method=method, optimal=optimal & ok,
                         max_violation=max_violation)


@dataclass
class ArmShareSet:
    shares: ShareSet        # sphere-pair shares, slots (B, S * M)
    rows: Tensor            # (B, S * M, 6)
    own: Tensor             # (B, S * M) own sphere index of each slot
    other: Tensor           # (B, S * M) neighbour sphere slot of each slot
    D: Tensor               # (B, S * M) collision distances r_k + r_l
    rho: Tensor             # (B, S * M) estimated center distances
    dv_norm: Tensor         # (B, S * M) estimated relative speeds
    heard: Tensor           # (B, S * M)


def arm_shares(own: SphereState, own_radius: Tensor, nbr_pos: Tensor, nbr_vel: Tensor, nbr_radius: Tensor,
               heard: Tensor, cfg: SafetyParams) -> ArmShareSet:
    """Reciprocal shares (Eq. 27) of every own sphere against every heard neighbour sphere.

    ``own`` has ``(B, S, 3)`` tensors, ``own_radius (B, S)`` or ``(S,)``;
    neighbour spheres ``(B, M, 3)``, ``nbr_radius / heard (B, M)``.
    """
    B, S, _ = own.center.shape
    M = nbr_pos.shape[1]
    dp = own.center.unsqueeze(2) - nbr_pos.unsqueeze(1)                     # (B, S, M, 3)
    dv = own.velocity.unsqueeze(2) - nbr_vel.unsqueeze(1)
    r_own = own_radius.expand(B, S) if own_radius.dim() == 1 else own_radius
    D = r_own.unsqueeze(-1) + nbr_radius.unsqueeze(1)                       # (B, S, M)
    hd = heard.unsqueeze(1).expand(B, S, M)
    sh = reciprocal_shares(dp.reshape(B, S * M, 3), dv.reshape(B, S * M, 3), D.reshape(B, S * M), cfg,
                           heard=hd.reshape(B, S * M))
    lever = own.lever.unsqueeze(2).expand(B, S, M, 3).reshape(B, S * M, 3)
    idx_s = torch.arange(S, device=dp.device).view(1, S, 1).expand(B, S, M).reshape(B, S * M)
    idx_m = torch.arange(M, device=dp.device).view(1, 1, M).expand(B, S, M).reshape(B, S * M)
    return ArmShareSet(shares=sh, rows=twist_rows(sh.g, lever), own=idx_s, other=idx_m, D=D.reshape(B, S * M),
                       rho=torch.linalg.vector_norm(dp, dim=-1).reshape(B, S * M),
                       dv_norm=torch.linalg.vector_norm(dv, dim=-1).reshape(B, S * M), heard=hd.reshape(B, S * M))


# ---------------------------------------------------------------------------
# fallback: jerk-optimal twist braking
# ---------------------------------------------------------------------------
class TwistBraking:
    """Eq. (36) applied to the whole twist with one common duration, then pose hold."""

    def __init__(self, n: int, limits: ArmLimits, dtype=torch.float64, device=None):
        self.limits = limits
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.xi0 = torch.zeros(n, 6, dtype=dtype, device=device)
        self.s0 = torch.zeros(n, dtype=dtype, device=device)

    def duration(self, xi0: Tensor) -> Tensor:
        L = self.limits
        return 1.5 * torch.maximum(torch.linalg.vector_norm(xi0[..., :3], dim=-1) / L.a_brk_lin,
                                   torch.linalg.vector_norm(xi0[..., 3:], dim=-1) / L.a_brk_ang)

    def engage(self, mask: Tensor, xi: Tensor, s: float) -> Tensor:
        new = mask & ~self.active
        self.xi0 = torch.where(new.unsqueeze(-1), xi, self.xi0)
        self.s0 = torch.where(new, torch.full_like(self.s0, s), self.s0)
        self.active = self.active | new
        return new

    def release(self, mask: Tensor) -> None:
        self.active = self.active & ~mask

    def command(self, s: float) -> Tuple[Tensor, Tensor, Tensor]:
        """``(twist (n, 6), twist_rate (n, 6), stopped (n,))`` at time ``s``."""
        Tb = self.duration(self.xi0)
        moving = Tb > 0
        Ts = torch.where(moving, Tb, torch.ones_like(Tb))
        x = torch.clamp((s - self.s0) / Ts, 0.0, 1.0)
        x = torch.where(moving, x, torch.ones_like(x))
        beta = 1.0 - 3.0 * x * x + 2.0 * x ** 3
        dbeta = torch.where(moving & (x < 1.0), (-6.0 * x + 6.0 * x * x) / Ts, torch.zeros_like(x))
        return self.xi0 * beta.unsqueeze(-1), self.xi0 * dbeta.unsqueeze(-1), x >= 1.0


# ---------------------------------------------------------------------------
# chunk buffer, neighbour table, broadcast
# ---------------------------------------------------------------------------
class TwistChunkBuffer:
    """Double-buffered chunks of world-frame twists ``(B, H, 6)`` and gripper commands ``(B, H)``."""

    def __init__(self, n: int, H: int, dtype=torch.float64, device=None):
        self.xi = torch.zeros(2, n, H, 6, dtype=dtype, device=device)
        self.grip = torch.zeros(2, n, H, dtype=dtype, device=device)
        self.stamp = torch.full((2, n), -math.inf, dtype=dtype, device=device)
        self.H_val = torch.zeros(2, n, dtype=torch.long, device=device)
        self.valid = torch.zeros(2, n, dtype=torch.bool, device=device)
        self.front = torch.zeros(n, dtype=torch.long, device=device)
        self.switch_time = torch.full((n,), -math.inf, dtype=dtype, device=device)
        self._ar = torch.arange(n, device=device)

    def publish(self, mask: Tensor, xi: Tensor, grip: Tensor, stamp: Tensor, H_val: Tensor, now: float) -> None:
        sel = mask.nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return
        b = 1 - self.front[sel]
        self.xi[b, sel] = xi[sel].to(self.xi)
        self.grip[b, sel] = grip[sel].to(self.grip)
        self.stamp[b, sel] = stamp[sel].to(self.stamp)
        self.H_val[b, sel] = H_val[sel].long()
        self.valid[b, sel] = True
        self.front[sel] = b
        self.switch_time[sel] = now

    def _eval(self, which: Tensor, s: float, T_c: float):
        ar = self._ar
        xi, gr, st, hv, va = (self.xi[which, ar], self.grip[which, ar], self.stamp[which, ar],
                              self.H_val[which, ar], self.valid[which, ar])
        H = xi.shape[1]
        ds = s - st
        ok = va & (ds >= 0) & (ds <= hv.to(ds.dtype) * T_c)
        t = torch.clamp(torch.where(ok, ds, torch.zeros_like(ds)) / T_c, 0.0, H - 1.0)
        i0 = torch.floor(t).long()
        i1 = torch.clamp(i0 + 1, max=H - 1)
        w = t - i0.to(t.dtype)
        x = (1 - w).unsqueeze(-1) * xi[ar, i0] + w.unsqueeze(-1) * xi[ar, i1]
        g = (1 - w) * gr[ar, i0] + w * gr[ar, i1]
        return x, g, ok

    def nominal(self, s_eval: float, s_valid: float, T_c: float, T_bl: float):
        """Blended twist and gripper command at ``s_eval``; validity of the newest chunk at ``s_valid``."""
        x_new, g_new, _ = self._eval(self.front, s_eval, T_c)
        _, _, valid_now = self._eval(self.front, s_valid, T_c)
        x_old, g_old, ok_old = self._eval(1 - self.front, s_eval, T_c)
        w = torch.clamp((s_eval - self.switch_time) / max(T_bl, 1e-9), 0.0, 1.0)
        blend = ok_old & (w < 1.0)
        wv = torch.where(blend, w, torch.ones_like(w))
        x = (1 - wv).unsqueeze(-1) * x_old + wv.unsqueeze(-1) * x_new
        g = (1 - wv) * g_old + wv * g_new
        return x, g, valid_now


@dataclass
class SphereTable:
    """Latest sphere broadcasts of the *other* arms as seen by each arm, ``(B, M)`` slots."""

    pos: Tensor
    vel: Tensor
    radius: Tensor
    flags: Tensor
    stamp: Tensor
    present: Tensor


@dataclass
class SphereBroadcast:
    """Per-sphere content of the 22-byte safety packets of every arm, ``(B, S)``."""

    arm_id: Tensor          # (B,)
    seq: Tensor             # (B,)
    stamp: Tensor           # (B,)
    pos: Tensor             # (B, S, 3)
    vel: Tensor             # (B, S, 3)
    radius: Tensor          # (B, S) safety radii
    flags: Tensor           # (B,) uint8


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------
@dataclass
class ArmExecutorConfig:
    limits: ArmLimits = field(default_factory=ArmLimits)
    spheres: SphereSet = PANDA_SPHERES
    safety: Optional[SafetyParams] = None
    H: int = 32
    T_bl: float = 0.05
    odom_timeout: float = 0.05
    nbr_timeout: float = 0.05
    release_ticks: int = 10
    w_hold: float = 0.02          # twist norm below which a braking neighbour counts as holding
    brake_on_neighbor: bool = True
    max_sweeps: int = 5           # Hildreth sweeps before the exact polish / Goldfarb-Idnani finisher

    def params(self) -> SafetyParams:
        return self.safety if self.safety is not None else arm_safety_params(self.limits, self.spheres)


@dataclass
class ArmCommand:
    time: float
    twist: Tensor           # (B, 6) world-frame twist setpoint for the next tick
    twist_rate: Tensor      # (B, 6) commanded twist rate u*
    gripper: Tensor         # (B,) gripper command (passes through)
    braking: Tensor
    flags: Tensor
    cause: Tensor
    n_active: Tensor
    stage_b_ok: Tensor
    intervention: Tensor    # (B,) ||u* - u^|| over the linear channels [m/s^2]
    sphere_intervention: Tensor   # (B,) max over spheres of ||J_c (u* - u^)|| [m/s^2]
    u_hat: Tensor
    u_star: Tensor
    tick_ms: float
    shares: Optional[ArmShareSet] = None


class ArmExecutor:
    """Batched 100 Hz executor for arms (Algorithm 1 TICK with twist-space Stage B)."""

    def __init__(self, n_arms: int, n_nbr_spheres: int, config: ArmExecutorConfig = ArmExecutorConfig(),
                 arm_ids: Optional[Tensor] = None, dtype=torch.float64, device=None):
        self.B, self.M = n_arms, n_nbr_spheres
        self.config = config
        self.cfg = config.params()
        self.limits = config.limits
        self.dtype, self.device = dtype, device
        self.offsets = config.spheres.offset_tensor(dtype, device)
        self.radius = config.spheres.safety_radius_tensor(dtype, device)
        self.buffer = TwistChunkBuffer(n_arms, config.H, dtype, device)
        self.brake = TwistBraking(n_arms, config.limits, dtype, device)
        self.clear_count = torch.zeros(n_arms, dtype=torch.long, device=device)
        self.flags = torch.zeros(n_arms, dtype=torch.uint8, device=device)
        self.seq = torch.zeros(n_arms, dtype=torch.long, device=device)
        self.arm_ids = torch.arange(n_arms, device=device) if arm_ids is None else arm_ids
        self.last_gripper = torch.full((n_arms,), -1.0, dtype=dtype, device=device)
        self.events: List[Dict] = []
        self.tick_ms: List[float] = []

    def publish_chunk(self, mask: Tensor, xi: Tensor, grip: Tensor, stamp: Tensor, H_val: Tensor, now: float):
        self.buffer.publish(mask, xi, grip, stamp, H_val, now)

    def spheres(self, p: Tensor, R: Tensor, v: Tensor, w: Tensor) -> SphereState:
        return sphere_kinematics(p, R, v, w, self.offsets)

    def project(self, xi_hat: Tensor, u_hat: Tensor, own: SphereState, nbr: SphereTable, heard: Tensor):
        sh = arm_shares(own, self.radius, nbr.pos, nbr.vel, nbr.radius, heard, self.cfg)
        lo, hi = twist_box(xi_hat, self.limits, self.cfg.T_c)
        res = solve_twist_qp(u_hat, sh.rows, sh.shares.zeta, sh.shares.active, lo, hi,
                             self.limits.accel_bound(self.dtype, self.device),
                             max_sweeps=self.config.max_sweeps)
        return sh, res

    def tick(self, s_n: float, p: Tensor, R: Tensor, v: Tensor, w: Tensor, nbr: SphereTable, *,
             odom_stamp: Optional[Tensor] = None, keep_shares: bool = False) -> ArmCommand:
        """One executor period for every arm: nominal twist rate, Stage B, fallback, outputs."""
        t0 = time.perf_counter()
        cfg, ec, L = self.cfg, self.config, self.limits
        Tc = cfg.T_c
        xi_hat = torch.cat([v, w], -1)
        stale_odom = torch.zeros(self.B, dtype=torch.bool, device=p.device) if odom_stamp is None \
            else (s_n - odom_stamp) > ec.odom_timeout
        xi_nom, grip_nom, chunk_ok = self.buffer.nominal(s_n + Tc, s_n, Tc, ec.T_bl)
        # neighbours (constant-velocity extrapolation of the broadcast state)
        age = (s_n - nbr.stamp).unsqueeze(-1)
        nbr_pos = nbr.pos + nbr.vel * age
        own = self.spheres(p, R, v, w)
        dist = torch.linalg.vector_norm(own.center.unsqueeze(2) - nbr_pos.unsqueeze(1), dim=-1).amin(1)
        near = nbr.present & (dist <= cfg.R_link)
        fresh = (s_n - nbr.stamp) <= ec.nbr_timeout
        heard = near & fresh
        stale_nbr = (near & ~fresh).any(-1)
        fl = nbr.flags.to(torch.int32)
        moving = torch.linalg.vector_norm(nbr.vel, dim=-1) > ec.w_hold
        nbr_brakes = heard & (((fl & 0x01) != 0) | (((fl & 0x02) != 0) & moving))
        # nominal twist rate and Stage B
        braking_before = self.brake.active
        acc = L.accel_bound(self.dtype, self.device)
        u_hat = torch.clamp((xi_nom - xi_hat) / Tc, -acc, acc)
        u_hat = torch.where((chunk_ok & ~braking_before).unsqueeze(-1), u_hat, torch.zeros_like(u_hat))
        tbl = SphereTable(nbr_pos, nbr.vel, nbr.radius, nbr.flags, nbr.stamp, nbr.present)
        sh, res = self.project(xi_hat, u_hat, own, tbl, heard)
        # fallback triggers
        cause = torch.zeros(self.B, dtype=torch.long, device=p.device)
        cause |= torch.where(~res.ok, 1, 0)
        cause |= torch.where(~chunk_ok, 2, 0)
        cause |= torch.where(stale_odom, 4, 0)
        cause |= torch.where(stale_nbr, 8, 0)
        primary = cause != 0
        cause |= torch.where(nbr_brakes.any(-1) & ec.brake_on_neighbor, 16, 0)
        want = cause != 0
        new = self.brake.engage(want, xi_hat, s_n)
        if bool(new.any()):
            self._log(new, s_n, cause, sh)
        brk_xi, brk_rate, stopped = self.brake.command(s_n + Tc)
        clear = self.brake.active & stopped & ~want
        self.clear_count = torch.where(clear, self.clear_count + 1, torch.zeros_like(self.clear_count))
        rel = self.brake.active & (self.clear_count >= ec.release_ticks)
        self.brake.release(rel)
        self.clear_count = torch.where(rel, torch.zeros_like(self.clear_count), self.clear_count)
        braking = self.brake.active
        twist = torch.where(braking.unsqueeze(-1), brk_xi, xi_hat + Tc * res.u)
        rate = torch.where(braking.unsqueeze(-1), brk_rate, res.u)
        grip = torch.where(chunk_ok & ~braking, grip_nom, self.last_gripper)
        self.last_gripper = grip
        flags = torch.where(braking, 0x02, 0) | torch.where(braking & primary, 0x01, 0)
        self.flags = flags.to(torch.uint8)
        self.seq = self.seq + 1
        du = res.u - u_hat
        sph_int = torch.linalg.vector_norm(du[:, None, :3] + torch.cross(du[:, None, 3:].expand_as(own.lever),
                                                                          own.lever, dim=-1), dim=-1).amax(-1)
        ms = (time.perf_counter() - t0) * 1e3
        self.tick_ms.append(ms)
        return ArmCommand(time=s_n, twist=twist, twist_rate=rate, gripper=grip, braking=braking, flags=self.flags,
                          cause=cause, n_active=res.n_active, stage_b_ok=res.ok,
                          intervention=torch.linalg.vector_norm(du[:, :3], dim=-1), sphere_intervention=sph_int,
                          u_hat=u_hat, u_star=res.u, tick_ms=ms, shares=sh if keep_shares else None)

    def _log(self, new: Tensor, s_n: float, cause: Tensor, sh: ArmShareSet) -> None:
        """Record a fallback onset with the Eq. (41) margin of every heard sphere pair.

        The envelope is the point-mass bound of Proposition 9 evaluated with the
        sphere-space parameters; for spheres on a braking end effector it is an
        approximation (the lever arms rotate while the twist decays), so it is
        logged as a diagnostic, not as a certificate.
        """
        need = fallback_envelope(sh.dv_norm, sh.D, self.cfg)
        margin = torch.where(sh.heard, sh.rho - need, torch.full_like(sh.rho, math.inf))
        for i in new.nonzero(as_tuple=True)[0].tolist():
            mm = float(margin[i].min()) if margin.shape[-1] else math.inf
            self.events.append({"arm": int(self.arm_ids[i]), "time": float(s_n), "cause": int(cause[i]),
                                "envelope_margin": mm, "n_pairs": int(sh.heard[i].sum())})

    def broadcast(self, s_n: float, p: Tensor, R: Tensor, v: Tensor, w: Tensor) -> SphereBroadcast:
        own = self.spheres(p, R, v, w)
        return SphereBroadcast(arm_id=self.arm_ids, seq=self.seq.clone(),
                               stamp=torch.full((self.B,), float(s_n), dtype=self.dtype, device=p.device),
                               pos=own.center, vel=own.velocity, radius=self.radius.expand(self.B, -1).clone(),
                               flags=self.flags.clone())

    def latency_stats(self) -> Dict[str, float]:
        if not self.tick_ms:
            return {}
        t = torch.tensor(self.tick_ms, dtype=torch.float64)
        return {"p50": float(t.quantile(0.5)), "p95": float(t.quantile(0.95)), "p99": float(t.quantile(0.99)),
                "max": float(t.max())}

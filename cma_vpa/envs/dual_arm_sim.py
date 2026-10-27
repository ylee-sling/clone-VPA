# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Lightweight dual-arm simulator for Task T2: two Franka Emika Panda arms at a shared table.

Kinematics
----------
:class:`PandaKinematics` implements the Panda with the manufacturer's modified
Denavit-Hartenberg parameters (Craig convention,
``T_i = RotX(alpha_{i-1}) TransX(a_{i-1}) RotZ(q_i) TransZ(d_i)``), the flange
(``d = 0.107 m``), the hand (rotated by ``-pi/4``) and the tool center point
(``0.1034 m`` beyond the hand). At the ready pose
``q = (0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4)`` the TCP is at ``(0.3069, 0, 0.4869) m``
with the tool axis pointing down, as on the real robot. It also provides the
geometric Jacobian at the TCP (world frame, linear rows first) and damped
least-squares inverse kinematics with a null-space posture term and the joint
position and velocity limits of the Panda datasheet. Everything is batched
over leading dimensions and every arm has its own base pose.

Dynamics
--------
:class:`DualArmSim` advances each end effector under zero-order hold at
``T_c = 10 ms`` (the 100 Hz control frequency of Section 7.1). The controller
input is a world-frame **twist setpoint** for the next tick, as published by
the executor (``xi_hat + T_c u*``). The low-level Cartesian controller turns it
into the twist rate ``u = clip((clip(xi_sp) - xi) / T_c, +-a)`` and integrates
exactly:

    p <- p + T_c v + T_c^2/2 (v_dot + d_lin),   v <- v + T_c (v_dot + d_lin),
    w <- w + T_c (w_dot + d_ang),   R <- Exp((w + T_c/2 (w_dot + d_ang)) T_c) R.

The bounded disturbance ``(d_lin, d_ang)`` models the tracking error of an
operational-space controller. It is scaled so that every collision sphere sees
at most ``e_track`` (Assumption 1); the centrifugal drift ``w x (w x r)``
arises from the rigid-body motion itself and is covered by the
``w_max^2 r_max`` term of :func:`~cma_vpa.safety.arm_barrier.arm_safety_params`.
The joints follow the end effector through damped least-squares IK every tick
(joint limits respected); the remaining pose residual is reported. The
proximal links are not covered by the barrier (Remark 1); their points
(:meth:`PandaKinematics.link_points`) are used for collision metrics and
rendering.

The gripper opening ``g in [0, 1]`` (``1`` = fully open, 8 cm) follows the
command (``> 0`` closes) at a finite rate and stops at ``grip_min`` (an object
between the fingers).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..safety.arm_barrier import PANDA_SPHERES, ArmLimits, SphereSet, SphereState, sphere_kinematics

__all__ = [
    "PANDA_DH", "PANDA_Q_MIN", "PANDA_Q_MAX", "PANDA_QD_MAX", "PANDA_Q_READY", "PANDA_FINGER_MAX",
    "rotvec_to_matrix", "matrix_to_rotvec", "matrix_to_quat", "yaw_matrix", "PandaKinematics", "DualArmSim",
]

#: Modified DH rows ``(a_{i-1}, d_i, alpha_{i-1})`` of joints 1..7 (Franka documentation).
PANDA_DH = ((0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
            (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2))
PANDA_FLANGE_D = 0.107
PANDA_HAND_YAW = -math.pi / 4
PANDA_TCP_D = 0.1034
PANDA_Q_MIN = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
PANDA_Q_MAX = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
PANDA_QD_MAX = (2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100)
PANDA_Q_READY = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4)
PANDA_FINGER_MAX = 0.08          # maximal opening between the fingers [m]


# ---------------------------------------------------------------------------
# SO(3) helpers (batched, numerically safe near 0 and pi)
# ---------------------------------------------------------------------------
def _skew(v: Tensor) -> Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rotvec_to_matrix(rv: Tensor) -> Tensor:
    """Rodrigues' formula ``Exp(rv)`` for rotation vectors ``(..., 3)``."""
    th = torch.linalg.vector_norm(rv, dim=-1, keepdim=True).unsqueeze(-1)       # (..., 1, 1)
    K = _skew(rv)
    small = th < 1e-6
    ths = torch.where(small, torch.ones_like(th), th)
    a = torch.where(small, 1.0 - th ** 2 / 6.0, torch.sin(ths) / ths)
    b = torch.where(small, 0.5 - th ** 2 / 24.0, (1.0 - torch.cos(ths)) / ths ** 2)
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device).expand(K.shape)
    return eye + a * K + b * (K @ K)


def matrix_to_rotvec(R: Tensor) -> Tensor:
    """``Log(R)`` as a rotation vector ``(..., 3)`` (stable near the identity and near pi)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    th = torch.arccos(cos)
    vee = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    sin = torch.sin(th)
    small = th < 1e-6
    near_pi = th > math.pi - 1e-4
    fac = torch.where(small, 0.5 + th ** 2 / 12.0, th / (2.0 * torch.where(small, torch.ones_like(sin), sin)))
    rv = fac.unsqueeze(-1) * vee
    # near pi: axis from the diagonal of (R + I) / 2
    B = (R + torch.eye(3, dtype=R.dtype, device=R.device)) / 2.0
    diag = torch.stack([B[..., 0, 0], B[..., 1, 1], B[..., 2, 2]], -1).clamp(min=0.0)
    k = torch.argmax(diag, -1, keepdim=True)
    col = torch.gather(B, -1, k.unsqueeze(-2).expand(*B.shape[:-1], 1)).squeeze(-1)
    axis = col / torch.linalg.vector_norm(col, dim=-1, keepdim=True).clamp(min=1e-12)
    sign = torch.sign((axis * vee).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    rv_pi = axis * th.unsqueeze(-1) * sign
    return torch.where(near_pi.unsqueeze(-1), rv_pi, rv)


def matrix_to_quat(R: Tensor) -> Tensor:
    """Unit quaternion ``(w, x, y, z)`` with ``w >= 0``."""
    rv = matrix_to_rotvec(R)
    th = torch.linalg.vector_norm(rv, dim=-1, keepdim=True)
    axis = rv / th.clamp(min=1e-12)
    q = torch.cat([torch.cos(th / 2), axis * torch.sin(th / 2)], -1)
    return torch.where(q[..., :1] < 0, -q, q)


def yaw_matrix(yaw: Tensor) -> Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def _mdh(a: float, d: float, alpha: float, theta: Tensor) -> Tensor:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([torch.stack([ct, -st, z, a * o], -1),
                        torch.stack([st * ca, ct * ca, -sa * o, -d * sa * o], -1),
                        torch.stack([st * sa, ct * sa, ca * o, d * ca * o], -1),
                        torch.stack([z, z, z, o], -1)], -2)


# ---------------------------------------------------------------------------
# Panda kinematics
# ---------------------------------------------------------------------------
class PandaKinematics:
    """Batched Panda kinematics for ``A`` arms; ``q`` has shape ``(..., A, 7)``.

    Each joint transform is ``C_i Z(q_i)`` with the constant factor
    ``C_i = RotX(alpha_{i-1}) TransX(a_{i-1})`` precomputed and
    ``Z(q) = RotZ(q) TransZ(d_i)``; one forward pass builds all frames.

    Parameters
    ----------
    base_pos, base_yaw:
        Base positions ``(A, 3)`` and yaws ``(A,)`` in the world frame.
    """

    def __init__(self, base_pos: Tensor, base_yaw: Tensor, dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        self.dtype, self.device = dtype, device
        self.base_p = torch.as_tensor(base_pos, dtype=dtype, device=device)
        self.base_R = yaw_matrix(torch.as_tensor(base_yaw, dtype=dtype, device=device))
        self.A = self.base_p.shape[0]
        kw = dict(dtype=dtype, device=device)
        self.q_min = torch.tensor(PANDA_Q_MIN, **kw)
        self.q_max = torch.tensor(PANDA_Q_MAX, **kw)
        self.qd_max = torch.tensor(PANDA_QD_MAX, **kw)
        self.q_ready = torch.tensor(PANDA_Q_READY, **kw)
        zero = torch.zeros((), **kw)
        self._C = torch.stack([_mdh(a, 0.0, al, zero) for a, _, al in PANDA_DH])          # (7, 4, 4)
        self._d = torch.tensor([d for _, d, _ in PANDA_DH], **kw)
        self._T_base = torch.eye(4, **kw).repeat(self.A, 1, 1)
        self._T_base[:, :3, :3] = self.base_R
        self._T_base[:, :3, 3] = self.base_p
        self._T_flange = _mdh(0.0, PANDA_FLANGE_D, 0.0, zero)
        self._T_tcp = _mdh(0.0, 0.0, 0.0, torch.tensor(PANDA_HAND_YAW, **kw)) @ _mdh(0.0, PANDA_TCP_D, 0.0, zero)
        self._T_flange_tcp = self._T_flange @ self._T_tcp

    def frames(self, q: Tensor) -> Tensor:
        """World frames ``(..., A, 9, 4, 4)``: joints 1..7, flange, TCP."""
        c, s = torch.cos(q), torch.sin(q)
        Z = torch.zeros(*q.shape, 4, 4, dtype=q.dtype, device=q.device)
        Z[..., 0, 0] = c
        Z[..., 0, 1] = -s
        Z[..., 1, 0] = s
        Z[..., 1, 1] = c
        Z[..., 2, 2] = 1.0
        Z[..., 2, 3] = self._d
        Z[..., 3, 3] = 1.0
        M = self._C @ Z                                                    # (..., A, 7, 4, 4)
        T = self._T_base.expand(*q.shape[:-1], 4, 4)
        out = []
        for i in range(7):
            T = T @ M[..., i, :, :]
            out.append(T)
        out.append(T @ self._T_flange)
        out.append(T @ self._T_flange_tcp)
        return torch.stack(out, -3)

    def fk(self, q: Tensor) -> Tuple[Tensor, Tensor]:
        """TCP position ``(..., A, 3)`` and orientation ``(..., A, 3, 3)``."""
        T = self.frames(q)[..., -1, :, :]
        return T[..., :3, 3], T[..., :3, :3]

    @staticmethod
    def jacobian_from_frames(Fr: Tensor) -> Tensor:
        z = Fr[..., :7, :3, 2]
        o = Fr[..., :7, :3, 3]
        p = Fr[..., -1, :3, 3].unsqueeze(-2)
        Jv = torch.cross(z, p - o, dim=-1)
        return torch.cat([Jv, z], -1).transpose(-1, -2)

    def jacobian(self, q: Tensor) -> Tensor:
        """Geometric Jacobian at the TCP, world frame, ``(..., A, 6, 7)`` (linear rows first)."""
        return self.jacobian_from_frames(self.frames(q))

    @staticmethod
    def link_points_from_frames(Fr: Tensor) -> Tensor:
        o = Fr[..., :8, :3, 3]                                    # joints 1..7 and flange
        mids = 0.5 * (o[..., 1:5, :] + o[..., 2:6, :])
        return torch.cat([o[..., 1:6, :], mids], -2)

    def link_points(self, q: Tensor) -> Tensor:
        """Points on the proximal links (joint origins 2..6 and link midpoints), ``(..., A, 9, 3)``."""
        return self.link_points_from_frames(self.frames(q))

    @staticmethod
    def _pose_error_from_frames(Fr: Tensor, p_t: Tensor, R_t: Tensor) -> Tensor:
        p, R = Fr[..., -1, :3, 3], Fr[..., -1, :3, :3]
        return torch.cat([p_t - p, matrix_to_rotvec(R_t @ R.transpose(-1, -2))], -1)

    def pose_error(self, q: Tensor, p_t: Tensor, R_t: Tensor) -> Tensor:
        return self._pose_error_from_frames(self.frames(q), p_t, R_t)

    def _dls(self, J: Tensor, e: Tensor, damping: float, posture: Optional[Tensor] = None) -> Tensor:
        """``J^T (J J^T + l^2 I)^-1 e`` plus the null-space projection of ``posture``."""
        JJt = J @ J.transpose(-1, -2)
        eye = torch.eye(6, dtype=J.dtype, device=J.device)
        K = torch.linalg.inv(JJt + damping ** 2 * eye)
        Jp = J.transpose(-1, -2) @ K                                      # damped pseudo-inverse (..., 7, 6)
        dq = (Jp @ e.unsqueeze(-1)).squeeze(-1)
        if posture is not None:
            dq = dq + posture - (Jp @ (J @ posture.unsqueeze(-1))).squeeze(-1)
        return dq

    def ik(self, q0: Tensor, p_t: Tensor, R_t: Tensor, iters: int = 200, damping: float = 0.02,
           tol: float = 1e-7, posture_gain: float = 0.05) -> Tuple[Tensor, Tensor]:
        """Damped least-squares IK to a TCP pose; returns ``(q, ||error||)``."""
        q = q0.clone()
        for it in range(iters):
            Fr = self.frames(q)
            e = self._pose_error_from_frames(Fr, p_t, R_t)
            if bool((torch.linalg.vector_norm(e, dim=-1) < tol).all()):
                break
            # posture term and damping are faded out over the second half so the iteration converges exactly
            fade = max(0.0, 1.0 - 2.0 * it / iters)
            posture = fade * posture_gain * (self.q_ready - q) if fade > 0 else None
            dq = self._dls(self.jacobian_from_frames(Fr), e, damping * fade + 1e-6, posture)
            q = torch.minimum(torch.maximum(q + dq, self.q_min + 1e-3), self.q_max - 1e-3)
        return q, torch.linalg.vector_norm(self.pose_error(q, p_t, R_t), dim=-1)

    def track(self, q: Tensor, p_t: Tensor, R_t: Tensor, dt: float, iters: int = 1, damping: float = 0.02,
              posture_gain: float = 0.5) -> Tuple[Tensor, Tensor, Tensor]:
        """One control period of joint tracking: DLS steps limited by ``qd_max dt`` and the joint limits.

        Returns ``(q, ||pose error||, frames)``; the frames of the new configuration
        are returned so that callers need no further forward pass.
        """
        Fr = self.frames(q)
        for _ in range(iters):
            e = self._pose_error_from_frames(Fr, p_t, R_t)
            dq = self._dls(self.jacobian_from_frames(Fr), e, damping, posture_gain * dt * (self.q_ready - q))
            lim = self.qd_max * dt / iters
            scale = torch.clamp((lim / dq.abs().clamp(min=1e-12)).amin(-1, keepdim=True), max=1.0)
            q = torch.minimum(torch.maximum(q + scale * dq, self.q_min + 1e-3), self.q_max - 1e-3)
            Fr = self.frames(q)
        err = torch.linalg.vector_norm(self._pose_error_from_frames(Fr, p_t, R_t), dim=-1)
        return q, err, Fr


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------
class DualArmSim:
    """Two (or more) Panda end effectors with twist-rate dynamics under zero-order hold, batched over envs.

    State tensors have shape ``(E, A, ...)``: TCP position ``p``, orientation
    ``R``, world-frame twist ``(v, w)``, joints ``q``, gripper opening ``grip``.
    """

    def __init__(self, num_envs: int = 1, base_pos: Sequence[Sequence[float]] = ((0.0, -0.55, 0.0), (0.0, 0.55, 0.0)),
                 base_yaw: Sequence[float] = (math.pi / 2, -math.pi / 2), limits: ArmLimits = ArmLimits(),
                 spheres: SphereSet = PANDA_SPHERES, T_c: float = 0.01, e_track: float = 0.0,
                 disturbance_tau: float = 0.2, gripper_rate: float = 4.0,
                 generator: Optional[torch.Generator] = None, dtype: torch.dtype = torch.float64):
        self.E, self.T_c, self.limits, self.spheres = num_envs, T_c, limits, spheres
        self.kin = PandaKinematics(torch.tensor(base_pos, dtype=dtype), torch.tensor(base_yaw, dtype=dtype), dtype)
        self.A = self.kin.A
        self.dtype = dtype
        self.g = generator
        self.e_track, self.disturbance_tau, self.gripper_rate = e_track, disturbance_tau, gripper_rate
        self.offsets = spheres.offset_tensor(dtype)
        self.t = 0.0
        E, A = self.E, self.A
        self.q = self.kin.q_ready.expand(E, A, 7).clone()
        self.p, self.R = self.kin.fk(self.q)
        self.v = torch.zeros(E, A, 3, dtype=dtype)
        self.w = torch.zeros(E, A, 3, dtype=dtype)
        self.grip = torch.ones(E, A, dtype=dtype)
        self.ik_err = torch.zeros(E, A, dtype=dtype)
        self.d = torch.zeros(E, A, 6, dtype=dtype)
        self.last_u = torch.zeros(E, A, 6, dtype=dtype)
        self._proximal: Optional[Tensor] = None

    @property
    def base_R(self) -> Tensor:
        return self.kin.base_R

    def reset(self, p_tcp: Tensor, R_tcp: Tensor) -> None:
        """Place every TCP at ``(p_tcp, R_tcp)`` ``(E, A, 3)``, ``(E, A, 3, 3)`` at rest (IK from the ready pose)."""
        q, err = self.kin.ik(self.kin.q_ready.expand(self.E, self.A, 7).clone(), p_tcp.to(self.dtype),
                             R_tcp.to(self.dtype))
        if bool((err > 1e-4).any()):
            raise ValueError(f"initial TCP pose not reachable (IK residual {float(err.max()):.2e})")
        self.q = q
        self.p, self.R = self.kin.fk(q)
        self.v.zero_()
        self.w.zero_()
        self.grip.fill_(1.0)
        self.ik_err.zero_()
        self.d.zero_()
        self._proximal = None
        self.t = 0.0

    @property
    def twist(self) -> Tensor:
        return torch.cat([self.v, self.w], -1)

    def sphere_state(self) -> SphereState:
        return sphere_kinematics(self.p, self.R, self.v, self.w, self.offsets)

    def _disturbance(self) -> Tensor:
        if self.e_track <= 0:
            return torch.zeros(self.E, self.A, 6, dtype=self.dtype)
        a = math.exp(-self.T_c / self.disturbance_tau)
        n = torch.randn(self.E, self.A, 6, generator=self.g, dtype=self.dtype)
        self.d = a * self.d + math.sqrt(1 - a * a) * n
        lin = self.d[..., :3]
        ang = self.d[..., 3:]
        lin = lin * torch.clamp(0.6 * self.e_track / torch.linalg.vector_norm(lin, dim=-1, keepdim=True)
                                .clamp(min=1e-12), max=1.0)
        cap = 0.4 * self.e_track / max(self.spheres.lever_max, 1e-6)
        ang = ang * torch.clamp(cap / torch.linalg.vector_norm(ang, dim=-1, keepdim=True).clamp(min=1e-12), max=1.0)
        return torch.cat([lin, ang], -1)

    def step(self, twist_sp: Tensor, grip_cmd: Tensor, grip_min: Optional[Tensor] = None) -> None:
        """Advance one tick with world-frame twist setpoints ``(E, A, 6)`` and gripper commands ``(E, A)``."""
        L, Tc = self.limits, self.T_c
        vb = L.twist_bound(self.dtype)
        ab = L.accel_bound(self.dtype)
        sp = torch.minimum(torch.maximum(twist_sp.to(self.dtype), -vb), vb)
        u = torch.minimum(torch.maximum((sp - self.twist) / Tc, -ab), ab)
        a = u + self._disturbance()
        al, aa = a[..., :3], a[..., 3:]
        self.p = self.p + Tc * self.v + 0.5 * Tc * Tc * al
        self.R = rotvec_to_matrix((self.w + 0.5 * Tc * aa) * Tc) @ self.R
        self.v = self.v + Tc * al
        self.w = self.w + Tc * aa
        self.last_u = u
        # gripper: > 0 closes
        target = torch.where(grip_cmd.to(self.dtype) > 0, torch.zeros_like(self.grip), torch.ones_like(self.grip))
        step = self.gripper_rate * Tc
        g = self.grip + torch.clamp(target - self.grip, -step, step)
        if grip_min is not None:
            g = torch.maximum(g, grip_min.to(self.dtype))
        self.grip = g.clamp(0.0, 1.0)
        # joints follow the end effector
        self.q, self.ik_err, Fr = self.kin.track(self.q, self.p, self.R, Tc)
        self._proximal = PandaKinematics.link_points_from_frames(Fr)
        self.t += Tc

    def proximal_points(self) -> Tensor:
        if self._proximal is None:
            self._proximal = self.kin.link_points(self.q)
        return self._proximal

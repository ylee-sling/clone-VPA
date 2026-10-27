# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Privileged centralized expert for the antipodal crossing (Section 4.8, stand-in for the SCP planner [82]).

The expert sees the ground-truth states of the whole team and a shared passing
handedness ``h in {+1, -1}`` (+1: counter-clockwise roundabout, i.e. every
vehicle passes the others on their right). Its desired velocity is the
saturated sum of

* a goal term ``v_h (g - p)/||g - p||``, slowed inside ``slow_radius`` of the goal;
* a roundabout term ``h v_rot gate(rho_c) (z x r_hat)`` around the conflict
  sphere, active within ``round_radius`` of the centre, plus a radial push that
  keeps vehicles out of the inner ``ring_radius``;
* pairwise separation within ``avoid_radius``, rotated by ``h * 60 deg`` about
  ``z`` so that close encounters are also resolved with the shared handedness;
* an altitude-hold term.

It is equivariant under the reflection ``S_y`` combined with ``h -> -h``, so the
two handedness modes are mirror images and a symmetric start has exactly two
expert modes (the situation of Lemma 3 and Proposition 7).

:meth:`CrossingExpert.plan` forward-simulates the team kinematically for ``H``
knots and returns each vehicle's chunk of desired velocities (the label used
for behaviour cloning and DAgger relabelling). The fixed labelling rules of
Section 4.8 are in :func:`handedness_label`, :func:`local_mode_labels` and
:func:`primitive_labels`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .quadrotor_sim import clip_norm

__all__ = ["ExpertConfig", "CrossingExpert", "handedness_label", "local_mode_labels", "primitive_labels"]


@dataclass
class ExpertConfig:
    v_h: float = 5.0                 # horizontal speed limit [m/s]
    v_z_max: float = 1.0
    slow_radius: float = 3.0
    round_radius: float = 10.0
    ring_radius: float = 4.0
    v_rot_frac: float = 0.8
    k_ring: float = 1.5
    avoid_radius: float = 4.0
    k_avoid: float = 3.0
    avoid_rot_deg: float = 60.0
    k_alt: float = 1.0
    centre: tuple = (0.0, 0.0)


class CrossingExpert:
    """Privileged team planner with a shared handedness ``h`` (batched over the team)."""

    def __init__(self, goals: Tensor, handedness: Tensor, cfg: ExpertConfig = ExpertConfig()):
        self.goals = goals.to(torch.float64)
        self.h = handedness.to(torch.float64).view(-1)          # (N,) +1 / -1 (usually all equal)
        self.cfg = cfg

    def velocity(self, p: Tensor, v: Optional[Tensor] = None) -> Tensor:
        """Desired world velocity ``(N, 3)`` at team state ``p (N, 3)``."""
        c = self.cfg
        p = p.to(torch.float64)
        h = self.h.view(-1, 1)
        zhat = torch.tensor([0.0, 0.0, 1.0], dtype=p.dtype)
        dg = self.goals - p
        dg_xy = dg.clone()
        dg_xy[:, 2] = 0
        dist = torch.linalg.vector_norm(dg_xy, dim=-1, keepdim=True)
        goal = c.v_h * dg_xy / dist.clamp(min=1e-9) * torch.clamp(dist / c.slow_radius, max=1.0)
        ctr = torch.tensor([*c.centre, 0.0], dtype=p.dtype)
        r = p - ctr
        r[:, 2] = 0
        rho = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
        rhat = r / rho.clamp(min=1e-6)
        tang = torch.cross(zhat.expand_as(rhat), rhat, dim=-1)
        gate = torch.clamp((c.round_radius - rho) / (c.round_radius - c.ring_radius), 0.0, 1.0)
        near_goal = torch.clamp(dist / c.slow_radius, max=1.0)
        roundabout = h * c.v_rot_frac * c.v_h * gate * near_goal * tang
        ring = c.k_ring * torch.clamp(c.ring_radius - rho, min=0.0) * rhat
        # pairwise separation, rotated by the shared handedness
        dp = p.unsqueeze(1) - p.unsqueeze(0)                                   # (N, N, 3) p_i - p_j
        dp[..., 2] = 0
        dd = torch.linalg.vector_norm(dp, dim=-1, keepdim=True)
        n = dp / dd.clamp(min=1e-6)
        w = torch.clamp(1.0 / dd.clamp(min=0.3) - 1.0 / c.avoid_radius, min=0.0)
        w = w * (1 - torch.eye(p.shape[0], dtype=p.dtype)).unsqueeze(-1)
        th = math.radians(c.avoid_rot_deg) * h.view(-1, 1, 1)
        rot_n = torch.cos(th) * n + torch.sin(th) * torch.cross(zhat.expand_as(n), n, dim=-1)
        avoid = c.k_avoid * (w * rot_n).sum(1)
        v_xy = clip_norm(goal + roundabout + ring + avoid, c.v_h)
        v_z = torch.clamp(c.k_alt * dg[:, 2:3], -c.v_z_max, c.v_z_max)
        return torch.cat([v_xy[:, :2], v_z], -1)

    def plan(self, p: Tensor, v: Optional[Tensor], H: int, dt: float) -> Tensor:
        """Chunks of desired velocities ``(N, H, 3)`` by kinematic team rollout (knot spacing ``dt``)."""
        p = p.to(torch.float64).clone()
        out = []
        for _ in range(H):
            vd = self.velocity(p)
            out.append(vd)
            p = p + dt * vd
        return torch.stack(out, 1)


def handedness_label(pos_traj: Tensor, vel_traj: Tensor, centre=(0.0, 0.0), radius: float = 10.0) -> Tensor:
    """Team passing handedness from a trajectory ``(T, N, 3)`` by the sign of the angular momentum.

    Returns the global mode label ``0`` (counter-clockwise, ``h = +1``) or ``1``
    (clockwise, ``h = -1``) using only samples within ``radius`` of the centre.
    """
    c = torch.tensor([*centre, 0.0], dtype=pos_traj.dtype)
    r = pos_traj - c
    Lz = r[..., 0] * vel_traj[..., 1] - r[..., 1] * vel_traj[..., 0]
    inside = torch.linalg.vector_norm(r[..., :2], dim=-1) < radius
    s = (Lz * inside).sum()
    return torch.tensor(0 if float(s) >= 0 else 1)


def local_mode_labels(pos_traj: Tensor, vel_traj: Tensor, centre=(0.0, 0.0), radius: float = 10.0) -> Tensor:
    """Per-vehicle side label (``0`` counter-clockwise, ``1`` clockwise) by the same rule, ``(N,)``."""
    c = torch.tensor([*centre, 0.0], dtype=pos_traj.dtype)
    r = pos_traj - c
    Lz = r[..., 0] * vel_traj[..., 1] - r[..., 1] * vel_traj[..., 0]
    inside = torch.linalg.vector_norm(r[..., :2], dim=-1) < radius
    return ((Lz * inside).sum(0) < 0).long()


def primitive_labels(v_des: Tensor, p: Tensor, goals: Tensor, v_h: float) -> Tensor:
    """Primitive label per vehicle from the expert command (fixed rule, :data:`PRIMITIVES` order).

    hover (5) near the goal and slow; climb_descend (4) for large vertical
    speed; yield (1) when slow; veer_left (2) / veer_right (3) for a large
    lateral component relative to the goal direction; cruise (0) otherwise.
    """
    dg = goals - p
    dg[..., 2] = 0
    ghat = dg / torch.linalg.vector_norm(dg, dim=-1, keepdim=True).clamp(min=1e-9)
    left = torch.stack([-ghat[..., 1], ghat[..., 0], torch.zeros_like(ghat[..., 0])], -1)
    lat = (v_des * left).sum(-1)
    speed = torch.linalg.vector_norm(v_des, dim=-1)
    dist = torch.linalg.vector_norm(dg, dim=-1)
    lab = torch.zeros(v_des.shape[:-1], dtype=torch.long)
    lab = torch.where(lat > 0.3 * v_h, torch.full_like(lab, 2), lab)
    lab = torch.where(lat < -0.3 * v_h, torch.full_like(lab, 3), lab)
    lab = torch.where(speed < 0.5 * v_h, torch.full_like(lab, 1), lab)
    lab = torch.where(v_des[..., 2].abs() > 0.4 * v_h, torch.full_like(lab, 4), lab)
    lab = torch.where((dist < 1.0) & (speed < 0.3), torch.full_like(lab, 5), lab)
    return lab

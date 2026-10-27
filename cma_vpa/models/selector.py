# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Neuro-symbolic primitive selector ``pi_phi^h`` (Eq. 13), milestone pointer and goal feature.

``u_t = argmax_u pi_phi^h(u | z_t, g_t, S_t, c_tilde_text, Emb(u_M^i))`` over
``N_u = 6`` primitives. The MLP is ``433 -> 384 -> 384 -> 6`` (316,806 parameters,
Table 3); its input is the concatenation

    z_t (256) | g_t (4) | S_tilde_t (13) | c_tilde_text (128) | Emb(u_M^i) (32)  = 433.

``g_t in R^4`` is the goal feature of Aero-VPA [26, Eq. (3)]; that paper is not
part of this repository, so :func:`goal_feature` fixes a continuous definition
that only needs the relative goal position in the heading frame:
``g = (tanh(r/l) n, tanh(r/l))`` with ``r = ||dp||``, ``n = dp / r`` and length
scale ``l``. It is zero at the goal and saturates far away.

:class:`MilestoneTracker` holds the symbolic milestone pointer ``m_t`` (1-based)
with the frozen completion threshold ``tau* = kappa sqrt(2 d) gamma_bar*`` of VPA
[24, Eq. (9)]; the master may also assign the milestone index through the
downlink (:meth:`MilestoneTracker.assign`), and the pointer never decreases.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["PrimitiveSelector", "MilestoneTracker", "goal_feature", "PRIMITIVES"]

#: Default primitive dictionary for the aerial task (N_u = 6).
PRIMITIVES = ("cruise", "yield", "veer_left", "veer_right", "climb_descend", "hover")


class PrimitiveSelector(nn.Module):
    """MLP classifier over the primitive dictionary (Eq. 13).

    Parameters
    ----------
    latent_dim, goal_dim, proprio_dim, text_dim, mode_dim:
        Input slot widths (256, 4, 13, 128, 32 by default; 433 in total).
    hidden:
        Hidden widths (``(384, 384)``).
    num_primitives:
        ``N_u``.
    """

    def __init__(self, latent_dim: int = 256, goal_dim: int = 4, proprio_dim: int = 13, text_dim: int = 128,
                 mode_dim: int = 32, hidden: Sequence[int] = (384, 384), num_primitives: int = 6):
        super().__init__()
        if num_primitives < 2:
            raise ValueError("the primitive dictionary needs at least two primitives")
        self.slots = (latent_dim, goal_dim, proprio_dim, text_dim, mode_dim)
        self.in_dim = sum(self.slots)
        self.num_primitives = num_primitives
        layers, w = [], self.in_dim
        for h in hidden:
            layers += [nn.Linear(w, h), nn.GELU()]
            w = h
        layers.append(nn.Linear(w, num_primitives))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_t: Tensor, g_t: Tensor, s_t: Tensor, c_text: Tensor, mode_emb: Tensor) -> Tensor:
        """Logits ``(B, N_u)``; all inputs ``(B, slot)``. One evaluation on the decision chain."""
        parts = (z_t, g_t, s_t, c_text, mode_emb)
        for p, w in zip(parts, self.slots):
            if p.shape[-1] != w:
                raise ValueError(f"selector input slot has width {p.shape[-1]}, expected {w}")
        return self.mlp(torch.cat(parts, -1))

    @staticmethod
    def select(logits: Tensor) -> Tensor:
        """``argmax`` (softmax is monotone, so the logits suffice)."""
        return logits.argmax(-1)

    @staticmethod
    def loss(logits: Tensor, labels: Tensor) -> Tensor:
        """Cross-entropy ``L_CE`` of Stage 4."""
        return F.cross_entropy(logits, labels)


def goal_feature(rel_goal_heading: Tensor, length_scale: float = 5.0) -> Tensor:
    """``g_t = (tanh(r/l) dp/r, tanh(r/l))`` for heading-frame goal offsets ``(..., 3)``."""
    r = torch.linalg.vector_norm(rel_goal_heading, dim=-1, keepdim=True)
    s = torch.tanh(r / length_scale)
    n = rel_goal_heading / r.clamp(min=1e-9)
    return torch.cat([s * n, s], -1)


class MilestoneTracker:
    """Milestone pointer ``m_t in {1..M}`` with the frozen threshold ``tau*`` (batched).

    ``step`` advances the pointer when ``||z_t - z_g^(m_t)|| < tau*`` and
    ``m_t < M``; ``assign`` applies a master-assigned index (never decreasing);
    ``active`` gathers ``z_g^(m_t)``.
    """

    def __init__(self, kappa: float = 0.1, latent_dim: int = 256, gamma_star: float = 1.0):
        if not 0.0 < kappa < 1.0:
            raise ValueError("kappa must lie in (0, 1)")
        self.kappa, self.latent_dim, self.gamma_star = kappa, latent_dim, gamma_star
        self.m: Optional[Tensor] = None
        self.M = 0

    @property
    def threshold(self) -> float:
        return self.kappa * math.sqrt(2.0 * self.latent_dim) * self.gamma_star

    def reset(self, batch: int, num_milestones: int, device: Optional[torch.device] = None) -> None:
        if num_milestones < 1:
            raise ValueError("at least one milestone is required")
        self.M = num_milestones
        self.m = torch.ones(batch, dtype=torch.long, device=device)

    def active(self, milestones: Tensor) -> Tensor:
        """``z_g^(m_t)`` from ``(B, M, d)``."""
        assert self.m is not None, "call reset() first"
        idx = (self.m - 1).clamp(0, self.M - 1).view(-1, 1, 1).expand(-1, 1, milestones.shape[-1])
        return milestones.gather(1, idx).squeeze(1)

    def step(self, z_t: Tensor, milestones: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply the threshold rule; returns the boolean ``reached`` mask."""
        assert self.m is not None, "call reset() first"
        dist = torch.linalg.vector_norm(z_t - self.active(milestones), dim=-1)
        reached = dist < self.threshold
        adv = reached & (self.m < self.M)
        if mask is not None:
            adv = adv & mask
        self.m = self.m + adv.long()
        return reached

    def assign(self, m_new: Tensor, mask: Optional[Tensor] = None) -> None:
        """Master-assigned milestone (1-based); the pointer is monotone and clipped to ``[1, M]``."""
        assert self.m is not None, "call reset() first"
        m_new = m_new.to(self.m.device).long().clamp(1, self.M)
        upd = torch.maximum(self.m, m_new)
        self.m = upd if mask is None else torch.where(mask, upd, self.m)

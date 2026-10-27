# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Jerk-optimal braking fallback (Section 4.7, Eq. 36) and its safety envelope (Prop. 9, Eq. 41).

When Stage B is infeasible, when no valid chunk exists, when odometry or
neighbour broadcasts are stale, or when a neighbour brakes, the executor
commands the braking profile of Aero-VPA [26, Eq. (26)]::

    v_brk(sigma) = v0 (1 - 3 (sigma/T_b)^2 + 2 (sigma/T_b)^3),   T_b = 3 ||v0|| / (2 a_brk),

for ``sigma in [0, T_b]`` and ``0`` afterwards, then holds position. With
``x = sigma / T_b`` and ``beta(x) = 1 - 3x^2 + 2x^3``:

* acceleration ``v0 beta'(x) / T_b = v0 (-6 sigma / T_b^2 + 6 sigma^2 / T_b^3)``,
  whose magnitude peaks at ``x = 1/2`` with value exactly ``a_brk``;
* jerk ``v0 (-6 / T_b^2 + 12 sigma / T_b^3)``;
* displacement ``v0 T_b (x - x^3 + x^4 / 2)``, in total ``v0 T_b / 2``;
* velocity and acceleration are continuous at both ends (``beta'(0) = beta'(1) = 0``).

Proposition 9 certifies the fallback for a pair whose separation at onset
satisfies Eq. (41); :func:`is_fallback_certified` evaluates it and the
executor logs the result for every fallback, as the protocol requires.

All functions are batched over leading dimensions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from .params import DEFAULT_PARAMS, SafetyParams

__all__ = [
    "BrakingState",
    "braking_duration",
    "braking_profile",
    "BrakingFallback",
    "fallback_envelope",
    "is_fallback_certified",
]

logger = logging.getLogger(__name__)
TensorLike = Union[Tensor, float]


# ---------------------------------------------------------------------------
# Eq. (36): the profile
# ---------------------------------------------------------------------------
def braking_duration(v0: Tensor, a_brk: float = DEFAULT_PARAMS.a_brk) -> Tensor:
    """``T_b = 3 ||v0|| / (2 a_brk)`` for velocities ``v0`` of shape ``(..., d)``."""
    return 1.5 * torch.linalg.vector_norm(v0, dim=-1) / a_brk


@dataclass
class BrakingState:
    """Reference of the braking profile at elapsed times ``sigma`` (all batched).

    ``velocity``, ``acceleration``, ``jerk``, ``displacement`` have shape
    ``(..., d)``; ``T_b`` and ``stopped`` (``sigma >= T_b``) have shape ``(...)``.
    """

    velocity: Tensor
    acceleration: Tensor
    jerk: Tensor
    displacement: Tensor
    T_b: Tensor
    stopped: Tensor


def braking_profile(v0: Tensor, sigma: TensorLike, a_brk: float = DEFAULT_PARAMS.a_brk) -> BrakingState:
    """Evaluate Eq. (36) and its derivatives at elapsed time ``sigma >= 0`` since onset.

    Parameters
    ----------
    v0:
        Velocity at braking onset ``(..., d)``.
    sigma:
        Time since onset, float or tensor broadcastable to ``(...)``.
    a_brk:
        Peak deceleration of the profile.
    """
    sigma_t = torch.as_tensor(sigma, dtype=v0.dtype, device=v0.device)
    T_b = braking_duration(v0, a_brk)
    T_b, sigma_t = torch.broadcast_tensors(T_b, sigma_t)
    sigma_t = torch.clamp(sigma_t, min=0.0)
    moving = T_b > 0
    T_safe = torch.where(moving, T_b, torch.ones_like(T_b))
    x = torch.where(moving, torch.clamp(sigma_t / T_safe, max=1.0), torch.ones_like(T_b))
    running = moving & (sigma_t < T_b)
    beta = 1.0 - 3.0 * x * x + 2.0 * x * x * x
    dbeta = torch.where(running, (-6.0 * x + 6.0 * x * x) / T_safe, torch.zeros_like(x))
    ddbeta = torch.where(running, (-6.0 + 12.0 * x) / (T_safe * T_safe), torch.zeros_like(x))
    integral = T_b * (x - x ** 3 + 0.5 * x ** 4)
    v0b = v0.expand(*T_b.shape, v0.shape[-1])
    return BrakingState(velocity=v0b * beta.unsqueeze(-1),
                        acceleration=v0b * dbeta.unsqueeze(-1),
                        jerk=v0b * ddbeta.unsqueeze(-1),
                        displacement=v0b * integral.unsqueeze(-1),
                        T_b=T_b, stopped=~running)


class BrakingFallback:
    """Batched state machine of the braking fallback for ``n_agents`` agents.

    ``engage`` latches the onset position, velocity and time for the selected
    agents (agents already braking keep their original onset, so the profile is
    never restarted). ``command`` returns the profile reference at a time
    ``s`` - velocity, acceleration feedforward and the hold position after
    stopping (``p0 + v0 T_b / 2``). ``release`` clears the latch.
    """

    def __init__(self, n_agents: int, d: int = 3, a_brk: float = DEFAULT_PARAMS.a_brk,
                 dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None):
        self.a_brk = a_brk
        self.active = torch.zeros(n_agents, dtype=torch.bool, device=device)
        self.p0 = torch.zeros(n_agents, d, dtype=dtype, device=device)
        self.v0 = torch.zeros(n_agents, d, dtype=dtype, device=device)
        self.s0 = torch.zeros(n_agents, dtype=dtype, device=device)

    def engage(self, mask: Tensor, p: Tensor, v: Tensor, s: TensorLike) -> Tensor:
        """Start braking for ``mask & ~active``; returns the mask of new onsets."""
        new = mask & ~self.active
        s_t = torch.as_tensor(s, dtype=self.s0.dtype, device=self.s0.device).expand_as(self.s0)
        self.p0 = torch.where(new.unsqueeze(-1), p, self.p0)
        self.v0 = torch.where(new.unsqueeze(-1), v, self.v0)
        self.s0 = torch.where(new, s_t, self.s0)
        self.active = self.active | new
        return new

    def release(self, mask: Tensor) -> None:
        """Clear the latch for the selected agents."""
        self.active = self.active & ~mask

    def command(self, s: TensorLike) -> BrakingState:
        """Profile reference at absolute time ``s`` for every agent (inactive rows are meaningless)."""
        s_t = torch.as_tensor(s, dtype=self.s0.dtype, device=self.s0.device)
        return braking_profile(self.v0, s_t - self.s0, self.a_brk)

    def hold_position(self) -> Tensor:
        """Position at which the reference stops, ``p0 + v0 T_b / 2``."""
        T_b = braking_duration(self.v0, self.a_brk)
        return self.p0 + 0.5 * T_b.unsqueeze(-1) * self.v0

    def stopped(self, s: TensorLike) -> Tensor:
        """``True`` where the profile has finished (``s - s0 >= T_b``)."""
        return self.command(s).stopped


# ---------------------------------------------------------------------------
# Proposition 9, Eq. (41): the envelope
# ---------------------------------------------------------------------------
def fallback_envelope(dv_norm: Tensor, D: TensorLike = DEFAULT_PARAMS.D_nominal,
                      cfg: SafetyParams = DEFAULT_PARAMS, v_prime: Optional[float] = None,
                      A_bar: Optional[float] = None, T_r: Optional[float] = None) -> Tensor:
    """Separation required at fallback onset by Eq. (41).

    ``D + T_r w + A T_r^2 / 2 + 21 v' / (8 a_brk) (w + A T_r) + 3 e_tr v' / a_brk``
    with ``w = ||dv_ij||`` at onset.

    Parameters
    ----------
    dv_norm:
        Relative speed ``||dv_ij||`` at onset, shape ``(...)``.
    v_prime:
        Bound on each agent's speed at braking onset (default ``cfg.v_bar``,
        which Assumption 1 guarantees).
    A_bar, T_r:
        Relative acceleration bound before onset and reaction time (defaults
        ``cfg.A_bar``, ``cfg.T_r``).
    """
    vp = cfg.v_bar if v_prime is None else v_prime
    A = cfg.A_bar if A_bar is None else A_bar
    Tr = cfg.T_r if T_r is None else T_r
    w = dv_norm
    a = cfg.a_brk
    return (D + Tr * w + 0.5 * A * Tr * Tr + 21.0 * vp / (8.0 * a) * (w + A * Tr)
            + 3.0 * cfg.e_tr * vp / a)


def is_fallback_certified(rho: Tensor, delta_v: Tensor, D_ij: TensorLike = DEFAULT_PARAMS.D_nominal,
                          params: SafetyParams = DEFAULT_PARAMS, v_prime: Optional[float] = None,
                          log: bool = True) -> Tensor:
    """Proposition 9: ``rho_ij(s_0) >=`` Eq. (41), per pair.

    ``rho`` has shape ``(...)`` and ``delta_v`` (relative velocity vectors)
    shape ``(..., d)``. Returns a boolean tensor of shape ``(...)``. With ``log=True`` the number
    of certified pairs and the smallest margin ``rho - envelope`` are logged at
    INFO level (WARNING if any pair is outside the envelope), which is the
    record the pre-registered protocol requires for every fallback.
    """
    need = fallback_envelope(torch.linalg.vector_norm(delta_v, dim=-1), D_ij, params, v_prime)
    ok = rho >= need
    if log and ok.numel() > 0:
        margin = float((rho - need).min())
        n_ok, n = int(ok.sum()), ok.numel()
        (logger.info if n_ok == n else logger.warning)(
            "fallback envelope (Prop. 9): %d/%d pairs certified, min margin %.3f m", n_ok, n, margin)
    return ok

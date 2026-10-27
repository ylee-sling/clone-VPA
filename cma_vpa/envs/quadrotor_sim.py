# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Fast multi-UAV simulator: disturbed double integrator under zero-order hold (Assumption 1).

Each vehicle obeys ``p_dot = v``, ``v_dot = u_n + d(s)`` on ``[s_n, s_n + T_c)``, where
``u_n in [-u_inf, u_inf]^3`` is the acceleration commanded at tick ``n`` (held
constant) and ``||d|| <= e_a`` collects acceleration-tracking error and wind
(Section 3.3). With ``d`` held over the tick as well, the update is exact::

    p_{n+1} = p_n + T_c v_n + T_c^2 / 2 (u_n + d_n),     v_{n+1} = v_n + T_c (u_n + d_n).

Wind (Section 7.2, Task T1):

* ``W0`` - no wind;
* ``W1`` - steady 3 m/s wind;
* ``W2`` - steady 3 m/s plus Dryden turbulence [87], gust vector clipped so that
  the gust adds at most 2 m/s (total wind at most 5 m/s, "gusts up to 5 m/s").

The wind enters as a drag-type tracking error ``k_d (w - w_hat)`` where ``w_hat``
is the low-level controller's slowly adapting wind estimate (integral action,
time constant ``tau_i``), plus a residual fraction ``rho_s`` of the drag the
controller never compensates. The realized error is clipped to
``e_a_realized <= e_a`` (default ``0.95 e_a``), so Assumption 1 holds by construction.

The Dryden model is the low-altitude MIL-F-8785C form with one first-order
(AR(1)) shaping filter per axis, discretized exactly at ``T_c``; scale lengths
and intensities follow from the altitude and the 20-ft wind speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

__all__ = ["WindConfig", "DrydenGust", "WindModel", "QuadrotorTeam", "clip_norm"]


def clip_norm(x: Tensor, max_norm: float) -> Tensor:
    """Scale rows of ``x (..., d)`` to norm at most ``max_norm``."""
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * torch.clamp(max_norm / n.clamp(min=1e-12), max=1.0)


@dataclass
class WindConfig:
    """Wind condition of Task T1 (``kind`` in ``{"W0", "W1", "W2"}``)."""

    kind: str = "W0"
    steady_speed: float = 3.0
    direction_deg: float = 30.0          # wind *towards* this azimuth (ENU, from +x)
    gust_max: float = 2.0                # max gust magnitude added to the steady wind (W2)
    w20: float = 5.0                     # 20-ft wind speed for the Dryden intensities [m/s]
    altitude: float = 5.0                # [m]
    airspeed: float = 5.0                # nominal airspeed for the Dryden time constants [m/s]
    k_drag: float = 0.35                 # drag per unit mass [1/s]
    tau_i: float = 1.5                   # wind-estimate time constant of the controller [s]
    rho_s: float = 0.15                  # uncompensated fraction of the steady drag

    @staticmethod
    def named(kind: str, **kw) -> "WindConfig":
        if kind not in ("W0", "W1", "W2"):
            raise ValueError(f"unknown wind condition {kind!r}")
        return WindConfig(kind=kind, **kw)


class DrydenGust:
    """Per-vehicle Dryden turbulence (low-altitude form, first-order filters, exact discretization)."""

    def __init__(self, n: int, cfg: WindConfig, dt: float, generator: Optional[torch.Generator] = None,
                 dtype: torch.dtype = torch.float64):
        h_ft = max(cfg.altitude, 0.5) / 0.3048
        L_w = h_ft
        L_u = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2
        sigma_w = 0.1 * cfg.w20
        sigma_u = sigma_w / (0.177 + 0.000823 * h_ft) ** 0.4
        L = torch.tensor([L_u, L_u, L_w], dtype=dtype) * 0.3048                 # back to metres
        self.sigma = torch.tensor([sigma_u, sigma_u, sigma_w], dtype=dtype)
        self.a = torch.exp(-cfg.airspeed * dt / L)                              # AR(1) coefficients
        self.b = self.sigma * torch.sqrt(1.0 - self.a ** 2)
        self.g = generator
        self.state = torch.randn(n, 3, generator=generator, dtype=dtype) * self.sigma
        self.gust_max = cfg.gust_max

    def step(self) -> Tensor:
        e = torch.randn(self.state.shape, generator=self.g, dtype=self.state.dtype)
        self.state = self.a * self.state + self.b * e
        return clip_norm(self.state, self.gust_max)


class WindModel:
    """Wind velocity per vehicle and the induced acceleration-tracking error."""

    def __init__(self, n: int, cfg: WindConfig, dt: float, e_a_realized: float,
                 generator: Optional[torch.Generator] = None, dtype: torch.dtype = torch.float64):
        self.cfg, self.dt, self.e_a = cfg, dt, e_a_realized
        az = math.radians(cfg.direction_deg)
        steady = torch.tensor([math.cos(az), math.sin(az), 0.0], dtype=dtype) * cfg.steady_speed
        self.steady = steady if cfg.kind in ("W1", "W2") else torch.zeros(3, dtype=dtype)
        self.gust = DrydenGust(n, cfg, dt, generator, dtype) if cfg.kind == "W2" else None
        self.w_hat = torch.zeros(n, 3, dtype=dtype)
        self.w = self.steady.expand(n, 3).clone()

    def step(self) -> Tensor:
        """Advance one tick; returns the tracking error ``d (n, 3)`` with ``||d|| <= e_a_realized``."""
        w = self.steady.expand_as(self.w_hat).clone()
        if self.gust is not None:
            w = w + self.gust.step()
        self.w = w
        k = self.cfg.k_drag
        d = k * (w - self.w_hat) + self.cfg.rho_s * k * self.w_hat
        self.w_hat = self.w_hat + (self.dt / self.cfg.tau_i) * (w - self.w_hat)
        return clip_norm(d, self.e_a)


class QuadrotorTeam:
    """Batched disturbed double integrators with zero-order hold at ``T_c`` (default 10 ms).

    Parameters
    ----------
    p0, v0:
        Initial states ``(n, 3)``.
    T_c, u_inf:
        Tick and per-axis acceleration bound (commands are clipped to the box).
    wind:
        :class:`WindConfig` (``W0`` gives ``d = 0``).
    e_a:
        Assumption-1 bound; the realized error is clipped to ``e_a_frac * e_a``.
    hold_gain, hold_damping, hold_accel:
        PD position hold used when the executor emits a hold position (after a
        completed braking profile), clipped to ``hold_accel``.
    """

    def __init__(self, p0: Tensor, v0: Optional[Tensor] = None, *, T_c: float = 0.01, u_inf: float = 8.0,
                 wind: WindConfig = WindConfig(), e_a: float = 1.0, e_a_frac: float = 0.95,
                 hold_gain: float = 2.0, hold_damping: float = 3.0, hold_accel: float = 2.0,
                 generator: Optional[torch.Generator] = None):
        self.p = p0.to(torch.float64).clone()
        self.v = torch.zeros_like(self.p) if v0 is None else v0.to(torch.float64).clone()
        self.n = self.p.shape[0]
        self.T_c, self.u_inf = T_c, u_inf
        self.wind = WindModel(self.n, wind, T_c, e_a * e_a_frac, generator)
        self.hold_gain, self.hold_damping, self.hold_accel = hold_gain, hold_damping, hold_accel
        self.t = 0.0
        self.last_d = torch.zeros_like(self.p)
        self.last_u = torch.zeros_like(self.p)

    def command_to_accel(self, acceleration: Tensor, hold_position: Optional[Tensor] = None) -> Tensor:
        """Commanded acceleration; rows with a finite hold position use the clipped PD hold law."""
        u = acceleration.to(torch.float64)
        if hold_position is not None:
            hp = hold_position.to(torch.float64)
            hold = torch.isfinite(hp).all(-1)
            if bool(hold.any()):
                pd = clip_norm(self.hold_gain * (torch.nan_to_num(hp) - self.p) - self.hold_damping * self.v,
                               self.hold_accel)
                u = torch.where(hold.unsqueeze(-1), pd, u)
        return torch.clamp(u, -self.u_inf, self.u_inf)

    def step(self, acceleration: Tensor, hold_position: Optional[Tensor] = None) -> None:
        """Advance one tick under zero-order hold."""
        u = self.command_to_accel(acceleration, hold_position)
        d = self.wind.step()
        a = u + d
        Tc = self.T_c
        self.p = self.p + Tc * self.v + 0.5 * Tc * Tc * a
        self.v = self.v + Tc * a
        self.last_u, self.last_d = u, d
        self.t += Tc

    def pairwise_distances(self) -> Tensor:
        return torch.cdist(self.p, self.p)

    def min_separation(self) -> float:
        if self.n < 2:
            return math.inf
        d = self.pairwise_distances()
        d = d + torch.diag(torch.full((self.n,), math.inf, dtype=d.dtype))
        return float(d.min())

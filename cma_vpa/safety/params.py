# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Pre-registered default parameters of the CMA-VPA safety layer (Table 4).

All quantities are SI (m, s, m/s, m/s^2). The dataclass is frozen and hashable
so that derived quantities that require a numerical search (the communication
radius of Lemma 15) can be cached per parameter set.

Derived bounds (Assumption 1)
-----------------------------
``U_bar = 2 sqrt(d) u_inf``
    bound on the Euclidean norm of a relative *commanded* acceleration
    ``u_i - u_j`` when every component of each input lies in ``[-u_inf, u_inf]``.
``A_bar = U_bar + 2 e_a``
    bound on the relative *realized* acceleration (commanded plus disturbance).
``V_max = 2 v_bar``
    bound on the relative speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["SafetyParams", "DEFAULT_PARAMS"]


@dataclass(frozen=True)
class SafetyParams:
    """Parameters of the braking-consistent HOCBF and the Stage-B executor.

    Attributes
    ----------
    a_max:
        Relative braking deceleration budget ``a_max`` [m/s^2] (Eq. 21-23).
    varsigma:
        Regularizer ``varsigma`` of the admissible closing speed [m/s]; must
        satisfy ``0 < varsigma^2 < a_max * D`` (Lemma 8).
    lambda_b:
        Gain of the linear second class-K function ``alpha_2(s) = lambda_b s``
        [1/s]; Lemma 11 requires ``lambda_b >= lambda_min(D)``.
    r_s, r_max:
        Nominal and maximal effective radius [m] (Eq. 35). ``D_ij = r_i + r_j``.
    kappa_sigma:
        Sensitivity of the radius to the filtered ensemble dispersion (Eq. 35).
    u_inf:
        Per-axis acceleration bound of the commanded input [m/s^2].
    e_a:
        Disturbance / acceleration-tracking bound ``||d_i|| <= e_a`` [m/s^2].
    e_p, e_v:
        Relative-state estimation bounds (Assumption 2) [m], [m/s].
    e_tr:
        Velocity tracking error of the braking fallback (Prop. 9) [m/s].
    v_bar:
        Speed bound of a single agent [m/s]; ``V_max = 2 v_bar``.
    T_c:
        Executor period = knot spacing ``Delta`` [s].
    d:
        Spatial dimension.
    a_brk, T_r:
        Fallback braking deceleration [m/s^2] and reaction time [s].
    R_link:
        Radius within which every safety broadcast is received [m]
        (Assumption 2); must be at least ``R_com(D)`` (Lemma 15) for every
        collision distance Eq. (35) can produce. The default 32 m covers
        ``R_com(2 r_max = 3.0 m) = 31.646 m``; Table 4's 30 m only covers the
        nominal ``D = 1.2 m`` (``R_com = 29.864 m``).
    """

    a_max: float = 5.0
    varsigma: float = 0.5
    lambda_b: float = 4.0
    r_s: float = 0.6
    r_max: float = 1.5
    kappa_sigma: float = 0.5
    u_inf: float = 8.0
    e_a: float = 1.0
    e_p: float = 0.05
    e_v: float = 0.10
    e_tr: float = 0.3
    v_bar: float = 8.0
    T_c: float = 0.01
    d: int = 3
    a_brk: float = 5.0
    T_r: float = 0.05
    R_link: float = 32.0

    def __post_init__(self) -> None:
        positive = ("a_max", "varsigma", "lambda_b", "r_s", "r_max", "u_inf",
                    "v_bar", "T_c", "a_brk", "R_link")
        for name in positive:
            if not getattr(self, name) > 0.0:
                raise ValueError(f"SafetyParams.{name} must be positive, got {getattr(self, name)}")
        nonneg = ("kappa_sigma", "e_a", "e_p", "e_v", "e_tr", "T_r")
        for name in nonneg:
            if getattr(self, name) < 0.0:
                raise ValueError(f"SafetyParams.{name} must be non-negative, got {getattr(self, name)}")
        if self.d < 1:
            raise ValueError("SafetyParams.d must be >= 1")
        if self.r_max < self.r_s:
            raise ValueError("SafetyParams.r_max must be >= r_s")
        if not self.varsigma ** 2 < self.a_max * self.D_nominal:
            raise ValueError("Lemma 8 requires varsigma^2 < a_max * D for the nominal D = 2 r_s")

    # ------------------------------------------------------------------ derived
    @property
    def D_nominal(self) -> float:
        """Collision distance for nominal radii, ``2 r_s`` [m]."""
        return 2.0 * self.r_s

    @property
    def D_max(self) -> float:
        """Largest collision distance reachable through Eq. (35), ``2 r_max`` [m]."""
        return 2.0 * self.r_max

    @property
    def V_max(self) -> float:
        """Relative speed bound ``2 v_bar`` [m/s]."""
        return 2.0 * self.v_bar

    @property
    def U_bar(self) -> float:
        """Relative commanded-acceleration norm bound ``2 sqrt(d) u_inf``."""
        return 2.0 * math.sqrt(self.d) * self.u_inf

    @property
    def A_bar(self) -> float:
        """Relative realized-acceleration norm bound ``U_bar + 2 e_a``."""
        return self.U_bar + 2.0 * self.e_a

    def k(self, D: Any) -> Any:
        """Constant ``k = a_max D - varsigma^2 / 2`` of Lemma 8 (float or tensor ``D``)."""
        return self.a_max * D - 0.5 * self.varsigma ** 2

    def rho_ext(self, D: Any) -> Any:
        """Left end of the smooth domain of ``alpha_1``: ``D - varsigma^2 / (4 a_max)``."""
        return D - self.varsigma ** 2 / (4.0 * self.a_max)

    def replace(self, **changes: Any) -> "SafetyParams":
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)


DEFAULT_PARAMS = SafetyParams()

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Proposition 7: symmetric standoff and its resolution by the master's joint mode.

Setting of the proposition and of Appendix A (Table 10): two identical agents
at rest at ``p_i = (-10, 0, 0)``, ``p_j = (10, 0, 0)`` (``L = 20 m``), goals swapped,
exact information (``e_p = e_v = 0``), no disturbance, Stage B (Eq. 34) with the
reciprocal shares of Eq. (27) at every tick of ``T_c = 10 ms``, nominal input
``u^ = nu + l`` clipped to the box, where ``nu`` is a goal-tracking law capped at
5 m/s (equivariant under ``S_y``, ``S_z`` and ``x -> -x``) and
``l = -2 h v_lat chi(|p_x|) h_own`` with ``v_lat = 1.5``, ``chi = 1{|p_x| < 6 m}``,
``h_own = +y`` for agent ``i`` and ``-y`` for agent ``j``.

(a) ``l = 0``: the lateral coordinates stay exactly zero, ``p_i,x = -p_j,x``,
    the agents stall in front of each other and never reach their goals.
(b) independent sides: the two incompatible choices (probability 1/2) keep the
    relative state on the x-axis (relative lateral offset exactly zero) and
    neither agent reaches its goal; the compatible choices pass.
(c) joint mode: both agents use the handedness of one global mode sampled by
    the master (:class:`~cma_vpa.models.MasterPlanner`, Eq. 11) and delivered in
    the 36-byte downlink; at the first tick with ``chi > 0`` the relative lateral
    acceleration is exactly ``-4 h v_lat``, the relative state leaves the axis,
    the separation stays at least ``D`` and both goals are reached.
"""

from __future__ import annotations

import math
from typing import Dict

import pytest
import torch

from cma_vpa.comm import DownlinkPacket
from cma_vpa.models import MasterConfig, MasterPlanner, handedness_of_mode
from cma_vpa.safety import DEFAULT_PARAMS, reciprocal_shares, solve_stage_b_qp

CFG = DEFAULT_PARAMS.replace(e_p=0.0, e_v=0.0)
D = CFG.D_nominal
V_LAT = 1.5
GATE = 6.0
R_GOAL = 1.0          # goal tolerance of Task T1
T_END = 12.0


def nominal(p: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Goal tracking capped at 5 m/s: ``nu = k_v (sat_5(k_p (w - p)) - v)``."""
    vd = w - p
    n = torch.linalg.vector_norm(vd, dim=-1, keepdim=True)
    vd = vd * torch.clamp(5.0 / n.clamp(min=1e-12), max=1.0)
    return 2.0 * (vd - v)


def simulate_pair(h_i: int, h_j: int, lateral: bool = True, T: float = T_END) -> Dict[str, torch.Tensor]:
    dt = torch.float64
    p = torch.tensor([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=dt)
    v = torch.zeros(2, 3, dtype=dt)
    w = p.flip(0).clone()
    h = torch.tensor([float(h_i), float(h_j)], dtype=dt)
    h_own = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=dt)
    Tc = CFG.T_c
    P, U, OK, LAT = [p.clone()], [], [], []
    reached = torch.zeros(2, dtype=torch.bool)
    t_reach = torch.full((2,), math.nan, dtype=dt)
    first_active = None
    for n in range(int(round(T / Tc))):
        chi = (p[:, 0].abs() < GATE).to(dt)
        ell = -2.0 * h.unsqueeze(-1) * V_LAT * chi.unsqueeze(-1) * h_own if lateral else torch.zeros_like(p)
        u_hat = torch.clamp(nominal(p, v, w) + ell, -CFG.u_inf, CFG.u_inf)
        dp = torch.stack([p[0] - p[1], p[1] - p[0]]).unsqueeze(1)          # (2, 1, 3)
        dv = torch.stack([v[0] - v[1], v[1] - v[0]]).unsqueeze(1)
        S = reciprocal_shares(dp, dv, torch.full((2, 1), D, dtype=dt), CFG)
        r = solve_stage_b_qp(u_hat, S.g, S.zeta, S.active, CFG.u_inf)
        u = r.u
        if lateral and first_active is None and bool((chi > 0).any()):
            first_active = n
            LAT.append(u[0, 1] - u[1, 1])
        p = p + Tc * v + 0.5 * Tc * Tc * u
        v = v + Tc * u
        P.append(p.clone())
        U.append(u.clone())
        OK.append(r.ok.clone())
        d_goal = torch.linalg.vector_norm(p - w, dim=-1)
        new = ~reached & (d_goal < R_GOAL)
        t_reach = torch.where(new, torch.full_like(t_reach, (n + 1) * Tc), t_reach)
        reached |= new
    P = torch.stack(P)
    return {"P": P, "U": torch.stack(U), "ok": torch.stack(OK), "reached": reached, "t_reach": t_reach,
            "sep": torch.linalg.vector_norm(P[:, 0] - P[:, 1], dim=-1), "first_active": first_active,
            "lat_acc": LAT[0] if LAT else None}


@pytest.fixture(scope="module")
def runs():
    return {(hi, hj, lat): simulate_pair(hi, hj, lat)
            for hi, hj, lat in [(1, 1, False), (1, -1, True), (-1, 1, True), (1, 1, True), (-1, -1, True)]}


def test_prop7a_symmetric_policies_stall_on_axis(runs):
    r = runs[(1, 1, False)]
    P = r["P"]
    assert bool((P[..., 1:] == 0).all())                                 # y = z = 0 exactly
    assert float((P[:, 0, 0] + P[:, 1, 0]).abs().max()) <= 1e-9          # p_i,x = -p_j,x
    assert bool((P[:, 0, 0] < 0).all() and (P[:, 1, 0] > 0).all())
    assert not bool(r["reached"].any())
    assert bool(r["ok"].all())                                           # Stage B feasible throughout
    assert float(r["sep"].min()) >= D - 1e-6
    stall = float(P[-1, 1, 0])
    assert abs(stall - 0.616) < 1e-3                                     # Table 10: x_i = -x_j = -0.616 m
    assert abs(float(r["sep"].min()) - 1.231) < 1e-3                     # Table 10: d_min = 1.231 m
    assert float(torch.linalg.vector_norm(P[-1] - P[-50], dim=-1).max()) < 1e-3   # at rest


@pytest.mark.parametrize("hi,hj", [(1, -1), (-1, 1)])
def test_prop7b_incompatible_independent_sides_stall(runs, hi, hj):
    r = runs[(hi, hj, True)]
    P = r["P"]
    assert bool((P[:, 0, 1] == P[:, 1, 1]).all())                        # relative lateral offset exactly 0
    assert bool((P[:, 0, 2] == P[:, 1, 2]).all())
    assert float((P[:, 0, 0] + P[:, 1, 0]).abs().max()) <= 1e-9
    assert abs(float(P[-1, 0, 1]) + hi * 3.33) < 0.01                    # Table 10: drift to y = -/+3.33 m
    assert not bool(r["reached"].any())
    assert float(r["sep"].min()) >= D - 1e-6


def test_prop7b_standoff_probability_one_half(runs):
    stall = {(hi, hj): not bool(runs[(hi, hj, True)]["reached"].all()) for hi, hj in
             [(1, -1), (-1, 1), (1, 1), (-1, -1)]}
    assert stall == {(1, -1): True, (-1, 1): True, (1, 1): False, (-1, -1): False}
    g = torch.Generator().manual_seed(0)
    hs = 1 - 2 * torch.randint(0, 2, (4000, 2), generator=g)              # independent uniform sides
    frac = sum(stall[(int(a), int(b))] for a, b in hs.tolist()) / hs.shape[0]
    assert abs(frac - 0.5) < 0.03


def _master_mode(seed: int) -> int:
    """Sample the joint mode with a master and deliver it through 36-byte downlinks to both agents."""
    torch.manual_seed(1234)
    cfg = MasterConfig(width=32, heads=4, depth=2, mode_emb_dim=8, plan_hidden=32, bus_dim=16, latent_dim=16,
                       text_dim=8, n_global_modes=4)
    m = MasterPlanner(cfg).eval()
    x = torch.zeros(1, 2, 16)                         # symmetric team: identical uplinks
    with torch.no_grad():
        out = m(x, torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 8), torch.zeros(1, 2, 16),
                generator=torch.Generator().manual_seed(seed))
    assert int(out.u_M[0, 0]) == int(out.u_M[0, 1])  # identical agents receive the same local mode
    side = int(out.u_g[0]) % 2
    msgs = [DownlinkPacket.build(a, 1, 0.0, side, 1, [0] * 24, 0.0, 0.0).pack() for a in (0, 1)]
    modes = [DownlinkPacket.unpack(raw).mode for raw in msgs]
    assert modes[0] == modes[1]
    return int(handedness_of_mode(torch.tensor(modes[0])))


def test_prop7c_joint_mode_leaves_axis_and_reaches_goals(runs):
    seen = set()
    for seed in range(40):
        h = _master_mode(seed)
        seen.add(h)
        if len(seen) == 2:
            break
    assert seen == {1, -1}                           # the master samples both modes
    for h in (1, -1):
        r = runs[(h, h, True)]
        assert r["first_active"] is not None
        assert math.isclose(float(r["lat_acc"]), -4 * h * V_LAT, abs_tol=1e-12)   # exactly -4 h v_lat
        P = r["P"]
        k = r["first_active"] + 2
        assert float((P[k, 0, 1] - P[k, 1, 1]).abs()) > 0                  # off the axis right after
        assert abs(float((P[:, 0, 1] - P[:, 1, 1]).abs().max()) - 4.55) < 0.01   # Table 10: up to 4.55 m
        assert abs(float(r["sep"].min()) - 3.108) < 1e-3                         # Table 10: d_min = 3.108 m
        assert torch.allclose(r["t_reach"], torch.full((2,), 5.24, dtype=torch.float64))   # Table 10: 5.24 s
        assert bool(r["ok"].all())
        assert float(r["sep"].min()) >= D - 1e-6                           # Theorem 1
        assert bool(r["reached"].all()) and float(r["t_reach"].max()) < T_END

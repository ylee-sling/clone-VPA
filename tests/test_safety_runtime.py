# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Runtime safety layer: Stage A (Lemma 17), braking fallback (Eq. 36, Prop. 9), 100 Hz executor.

Run with ``pytest -q tests/test_safety_runtime.py``; ``python tests/test_safety_runtime.py``
prints a report with the measured values (Prop. 9 margins, Stage-A solve
times, executor latency percentiles). Set ``CMA_VPA_TEST_COMPILE=1`` to also
check the ``torch.compile`` executor path against eager execution (the first
compilation takes minutes on CPU).
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cma_vpa.safety import degg_flow as DF  # noqa: E402
from cma_vpa.safety import executor as EX  # noqa: E402
from cma_vpa.safety import fallback as FB  # noqa: E402
from cma_vpa.safety import hocbf as H  # noqa: E402
from cma_vpa.safety import margins as M  # noqa: E402
from cma_vpa.safety import qp_solvers as Q  # noqa: E402
from cma_vpa.safety.params import DEFAULT_PARAMS  # noqa: E402
from test_safety import brute_force_qp  # noqa: E402

F64 = torch.float64
P = DEFAULT_PARAMS
D = P.D_nominal


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def rand_unit(n: int, d: int, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, d, generator=g, dtype=F64)
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


# ===========================================================================
# Eq. (36): the braking profile
# ===========================================================================
def test_braking_profile_shape_and_derivatives():
    g = gen(1)
    v0 = rand_unit(200, 3, g) * (8 * torch.rand(200, 1, generator=g, dtype=F64))
    T_b = FB.braking_duration(v0)
    assert torch.allclose(T_b, 1.5 * torch.linalg.vector_norm(v0, dim=-1) / P.a_brk)
    x = torch.linspace(0.0, 1.2, 1201, dtype=F64)
    sig = T_b.unsqueeze(-1) * x                                          # (200, 1201)
    st = FB.braking_profile(v0.unsqueeze(1), sig)
    assert torch.allclose(st.velocity[:, 0], v0) and float(st.velocity[:, 1000:].abs().max()) < 1e-12
    assert float(st.acceleration[:, 0].abs().max()) == 0.0
    assert float(st.acceleration[:, 1000:].abs().max()) == 0.0         # zero at and after T_b
    amax = torch.linalg.vector_norm(st.acceleration, dim=-1).amax(-1)
    moving = torch.linalg.vector_norm(v0, dim=-1) > 1e-6
    assert torch.allclose(amax[moving], torch.full_like(amax[moving], P.a_brk), rtol=1e-6)  # peak a_brk at T_b/2
    assert torch.allclose(st.displacement[:, -1], 0.5 * T_b.unsqueeze(-1) * v0, atol=1e-12)
    # finite differences: d/dsigma velocity = acceleration, d/dsigma displacement = velocity
    h = 1e-6
    for xi in (0.1, 0.37, 0.5, 0.83):
        s0 = T_b * xi
        a = FB.braking_profile(v0, s0)
        fd_v = (FB.braking_profile(v0, s0 + h).velocity - FB.braking_profile(v0, s0 - h).velocity) / (2 * h)
        fd_x = (FB.braking_profile(v0, s0 + h).displacement - FB.braking_profile(v0, s0 - h).displacement) / (2 * h)
        fd_a = (FB.braking_profile(v0, s0 + h).acceleration - FB.braking_profile(v0, s0 - h).acceleration) / (2 * h)
        assert torch.allclose(fd_v[moving], a.acceleration[moving], atol=1e-5)
        assert torch.allclose(fd_x[moving], a.velocity[moving], atol=1e-7)
        assert torch.allclose(fd_a[moving], a.jerk[moving], atol=1e-3)
    zero = FB.braking_profile(torch.zeros(1, 3, dtype=F64), 0.3)
    assert bool(zero.stopped.all()) and float(zero.velocity.abs().max()) == 0.0


def test_braking_fallback_latch_hold_and_release():
    fb = FB.BrakingFallback(3)
    p = torch.zeros(3, 3, dtype=F64)
    v = torch.tensor([[4.0, 0, 0], [0, -2.0, 0], [0, 0, 0]], dtype=F64)
    new = fb.engage(torch.tensor([True, True, False]), p, v, 1.0)
    assert new.tolist() == [True, True, False]
    new2 = fb.engage(torch.tensor([True, False, True]), p + 1.0, 3 * v, 2.0)   # no restart for agent 0
    assert new2.tolist() == [False, False, True]
    assert float(fb.s0[0]) == 1.0 and torch.equal(fb.v0[0], v[0])
    T0 = 1.5 * 4.0 / P.a_brk
    assert not bool(fb.stopped(1.0 + 0.5 * T0)[0]) and bool(fb.stopped(1.0 + T0)[0])
    assert torch.allclose(fb.hold_position()[0], torch.tensor([0.5 * T0 * 4.0, 0, 0], dtype=F64))
    fb.release(torch.tensor([True, False, False]))
    assert fb.active.tolist() == [False, True, True]


# ===========================================================================
# Proposition 9: the braking envelope
# ===========================================================================
def prop9_simulation(n: int = 4000, dt: float = 2e-3, seed: int = 9, cfg=P) -> Dict[str, float]:
    """Reciprocal braking from the boundary of Eq. (41) with worst-case reaction and tracking.

    During the reaction time each agent accelerates by ``A/2`` toward the
    other; then both follow Eq. (36) along their own onset velocity while a
    tracking error of magnitude ``e_tr`` pushes them together until their
    profile ends; afterwards they hold position. The initial separation equals
    the right-hand side of Eq. (41) exactly.
    """
    g = gen(seed)
    A, Tr, vp, a_brk = cfg.A_bar, cfg.T_r, cfg.v_bar, cfg.a_brk
    s_max = vp - 0.5 * A * Tr                         # so that onset speeds stay <= v'
    v_i = rand_unit(n, 3, g) * (s_max * torch.rand(n, 1, generator=g, dtype=F64))
    v_j = rand_unit(n, 3, g) * (s_max * torch.rand(n, 1, generator=g, dtype=F64))
    dv0 = v_i - v_j
    w0 = torch.linalg.vector_norm(dv0, dim=-1)
    rho0 = FB.fallback_envelope(w0, D, cfg, v_prime=vp)
    # geometry: half head-on (dp antiparallel to dv), half random direction
    head = -dv0 / torch.clamp(w0, min=1e-9).unsqueeze(-1)
    rnd = rand_unit(n, 3, g)
    nrm = torch.where((torch.arange(n) % 2 == 0).unsqueeze(-1), head, rnd)
    p_i, p_j = 0.5 * rho0.unsqueeze(-1) * nrm, -0.5 * rho0.unsqueeze(-1) * nrm
    min_gap = torch.full((n,), math.inf, dtype=F64)
    # reaction phase
    steps = int(round(Tr / dt))
    for _ in range(steps):
        u = (p_j - p_i) / torch.linalg.vector_norm(p_j - p_i, dim=-1, keepdim=True)
        a_i, a_j = 0.5 * A * u, -0.5 * A * u
        p_i, p_j = p_i + v_i * dt + 0.5 * a_i * dt * dt, p_j + v_j * dt + 0.5 * a_j * dt * dt
        v_i, v_j = v_i + a_i * dt, v_j + a_j * dt
        min_gap = torch.minimum(min_gap, torch.linalg.vector_norm(p_i - p_j, dim=-1) - D)
    onset_speed = torch.maximum(torch.linalg.vector_norm(v_i, dim=-1), torch.linalg.vector_norm(v_j, dim=-1))
    vi0, vj0, pi0, pj0 = v_i.clone(), v_j.clone(), p_i.clone(), p_j.clone()
    dvo = torch.linalg.vector_norm(vi0 - vj0, dim=-1)
    T_i, T_j = FB.braking_duration(vi0, a_brk), FB.braking_duration(vj0, a_brk)
    T_end = float(torch.maximum(T_i, T_j).max()) + 0.1
    sig = 0.0
    xi, xj = pi0.clone(), pj0.clone()
    dv_ratio_max = 0.0
    sj = torch.clamp(onset_speed, min=1e-9)
    while sig < T_end:
        u = (xj - xi) / torch.clamp(torch.linalg.vector_norm(xj - xi, dim=-1, keepdim=True), min=1e-12)
        bi = FB.braking_profile(vi0, sig + 0.5 * dt, a_brk)
        bj = FB.braking_profile(vj0, sig + 0.5 * dt, a_brk)
        e_i = torch.where(bi.stopped.unsqueeze(-1), torch.zeros_like(u), cfg.e_tr * u)
        e_j = torch.where(bj.stopped.unsqueeze(-1), torch.zeros_like(u), -cfg.e_tr * u)
        xi = xi + (bi.velocity + e_i) * dt
        xj = xj + (bj.velocity + e_j) * dt
        sig += dt
        min_gap = torch.minimum(min_gap, torch.linalg.vector_norm(xi - xj, dim=-1) - D)
        # proof inequality ||dv(sigma)|| <= (1 + a_brk sigma / s_j) ||dv(0)|| (exact tracking)
        dvs = torch.linalg.vector_norm(FB.braking_profile(vi0, sig, a_brk).velocity
                                       - FB.braking_profile(vj0, sig, a_brk).velocity, dim=-1)
        bound = (1 + a_brk * sig / sj) * dvo
        dv_ratio_max = max(dv_ratio_max, float((dvs - bound).max()))
    return {"min_margin": float(min_gap.min()), "max_onset_speed": float(onset_speed.max()),
            "dv_bound_excess": dv_ratio_max, "n": n}


def test_prop9_braking_envelope_simulation():
    r = prop9_simulation()
    assert r["max_onset_speed"] <= P.v_bar + 1e-9
    assert r["min_margin"] >= 0.0                          # distance never below D
    assert r["dv_bound_excess"] <= 1e-9


def test_is_fallback_certified_and_logging(caplog):
    rho = torch.tensor([100.0, 5.0], dtype=F64)
    dv = torch.tensor([[10.0, 0, 0], [10.0, 0, 0]], dtype=F64)
    need = FB.fallback_envelope(torch.tensor(10.0, dtype=F64), D)
    expected = D + P.T_r * 10 + 0.5 * P.A_bar * P.T_r ** 2 + 21 * P.v_bar / (8 * P.a_brk) * (10 + P.A_bar * P.T_r) \
        + 3 * P.e_tr * P.v_bar / P.a_brk
    assert float(need) == pytest.approx(expected, rel=1e-12)
    with caplog.at_level(logging.INFO, logger="cma_vpa.safety.fallback"):
        ok = FB.is_fallback_certified(rho, dv, D)
    assert ok.tolist() == [True, False]
    assert any("1/2 pairs certified" in rec.message for rec in caplog.records)


# ===========================================================================
# Stage A: Lemma 17 and the polytope
# ===========================================================================
def random_field(B: int, H: int, da: int, g: torch.Generator, L: float = 1.5, smooth: bool = False):
    """Random nonlinear field ``v(A, tau)``; ``smooth`` makes it vary slowly across knots."""
    n = H * da
    W1 = torch.randn(n, 64, generator=g, dtype=F64) * (L / math.sqrt(n))
    W2 = torch.randn(64, n, generator=g, dtype=F64) * (1.0 / math.sqrt(64))
    b = torch.randn(B, n, generator=g, dtype=F64) * 0.5
    if smooth:
        # project the output on a few low-frequency knot profiles
        k = torch.arange(H, dtype=F64)
        basis = torch.stack([torch.cos(math.pi * f * (k + 0.5) / H) for f in range(3)], -1)  # (H, 3)
        Pm = basis @ torch.linalg.pinv(basis)
    def v(A, tau):
        out = (torch.tanh(A.reshape(B, n) @ W1 + tau) @ W2 + b).reshape(B, H, da)
        if smooth:
            out = torch.einsum("hk,bkc->bhc", Pm, out)
        return out
    return v


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_lemma17c_random_fields_random_polyhedra(K):
    g = gen(170 + K)
    B, H, da = 40, 4, 3
    n, m = H * da, 30
    G = torch.randn(B, m, n, generator=g, dtype=F64)
    x0 = 0.3 * torch.randn(B, n, generator=g, dtype=F64)
    h = (G @ x0.unsqueeze(-1)).squeeze(-1) + 0.2 * torch.rand(B, m, generator=g, dtype=F64)
    poly = DF.Polyhedron(G=G, h=h)
    vf = random_field(B, H, da, g)
    xi = torch.randn(B, H, da, generator=g, dtype=F64)
    r = DF.guided_solve(vf, xi, poly, K, n_channels=da, clip=False)
    resid = poly.residual(r.A.reshape(B, n))
    assert float((resid / (1 + h.abs())).max()) <= 1e-6
    assert bool(r.terminal_feasible.all()) and bool(r.terminal_exact.all()) and bool(r.guided.all())


def _stage_a_case(B: int, seed: int, N: int = 3):
    g = gen(seed)
    Hk = 32
    p_hat = torch.zeros(B, 3, dtype=F64)
    v_hat = torch.cat([3 + torch.rand(B, 1, generator=g, dtype=F64), torch.zeros(B, 2, dtype=F64)], -1)
    # an oncoming neighbour near the braking boundary (closing 5-6 m/s at 5.5-7 m, inside C^HO,
    # so its shares are active), one to the side, one far away
    nbr_pos = torch.tensor([[5.5, 0.4, 0.0], [3.0, 4.0, 0.0], [25.0, -3.0, 1.0]], dtype=F64)[:N].expand(B, N, 3).clone()
    nbr_pos[:, 0, 0] += 1.5 * torch.rand(B, generator=g, dtype=F64)
    nbr_vel = torch.tensor([[-2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=F64)[:N].expand(B, N, 3).clone()
    t = H.hocbf_terms(p_hat.unsqueeze(1) - nbr_pos, v_hat.unsqueeze(1) - nbr_vel)
    assert bool(t.in_safe_set.all())
    yaw = (2 * torch.rand(B, generator=g, dtype=F64) - 1) * math.pi
    cmap = DF.ChunkMap.from_yaw(yaw, scale=8.0)
    poly = DF.build_stage_a_polytope(p_hat, v_hat, DF.constant_velocity_reference(v_hat, Hk), nbr_pos,
                                     nbr_vel, cmap)
    return poly, cmap, v_hat


@pytest.mark.parametrize("K", [2, 3])
def test_lemma17c_stage_a_polytope(K):
    B = 8
    poly, cmap, v_hat = _stage_a_case(B, 30 + K)
    assert int(poly.row_mask[:, :poly.layout["share"][1]].sum()) > 0       # shares are active
    g = gen(40 + K)
    target = torch.cat([cmap.to_normalized(v_hat.unsqueeze(1).expand(B, 32, 3)),
                        torch.zeros(B, 32, 1, dtype=F64)], -1)
    base = random_field(B, 32, 4, g, smooth=True)
    vf = lambda A, tau: (target - A) + 0.3 * base(A, tau)  # noqa: E731
    xi = 0.3 * torch.randn(B, 32, 4, generator=g, dtype=F64)
    r = DF.guided_solve(vf, xi, poly, K)
    x = r.A[..., :3].reshape(B, -1)
    h_abs = torch.where(poly.row_mask, poly.h.abs(), torch.zeros_like(poly.h))
    assert float((poly.residual(x) / (1 + h_abs)).max()) <= 1e-6
    assert bool(r.terminal_exact.all()) and bool(r.terminal_feasible.all())
    assert float(r.A.abs().max()) <= 1.0


def test_lemma17a_averaged_form_equals_direct_euler_update():
    g = gen(171)
    B, H, da, K = 10, 3, 2, 4
    n, m = H * da, 12
    G = torch.randn(B, m, n, generator=g, dtype=F64)
    h = (G @ (0.2 * torch.randn(B, n, 1, generator=g, dtype=F64))).squeeze(-1) + 0.3
    poly = DF.Polyhedron(G=G, h=h)
    vf = random_field(B, H, da, g)
    xi = torch.randn(B, H, da, generator=g, dtype=F64)
    nu = [0.3, 0.7, 0.5, 1.0]
    r = DF.guided_solve(vf, xi, poly, K, nu=nu, n_channels=da, n_sweeps=20000, final_sweeps=20000, clip=False)
    A = xi.clone()
    for k in range(K):                                   # Eq. (32) literally, with exact projections
        tau = k / K
        V = vf(A, tau)
        A1 = A + (1 - tau) * V
        y = A1.reshape(B, n)
        Pk = Q.hildreth(y, G, h, max_sweeps=20000, polish=True).x
        Gamma = nu[k] * (Pk - y) / (1 - tau)
        A = A + (V + Gamma.reshape(B, H, da)) / K
    assert torch.allclose(r.A, A, atol=1e-9)


def test_lemma17b_minimal_intervention_against_brute_force():
    g = gen(172)
    B, H, da = 30, 2, 2
    n, m = H * da, 7
    G = torch.randn(B, m, n, generator=g, dtype=F64)
    h = (G @ (0.2 * torch.randn(B, n, 1, generator=g, dtype=F64))).squeeze(-1) + 0.2
    w = 0.5 + torch.rand(B, n, generator=g, dtype=F64)            # diagonal W
    poly = DF.Polyhedron(G=G, h=h, W_inv=1.0 / w)
    vf = random_field(B, H, da, g, L=3.0)
    xi = 2 * torch.randn(B, H, da, generator=g, dtype=F64)
    r = DF.guided_solve(vf, xi, poly, 1, n_channels=da, clip=False)  # K = 1: A_1 = Pi_W(A_hat)
    A_hat = r.endpoints[0].reshape(B, n)
    for b in range(B):
        x = brute_force_qp(A_hat[b].numpy(), G[b].numpy(), h[b].numpy(), np.diag(1.0 / w[b].numpy()))
        assert np.allclose(r.A[b].reshape(-1).numpy(), x, atol=1e-8)
        gam = (x - A_hat[b].numpy())                               # (1 - tau) Gamma*, tau = 0
        assert float(r.intervention[0, b]) == pytest.approx(math.sqrt(float((gam * gam * w[b].numpy()).sum())),
                                                            rel=1e-6, abs=1e-9)


def test_nu_zero_reduces_to_plain_euler():
    g = gen(173)
    B, H, da, K = 5, 4, 3, 3
    G = torch.randn(B, 6, H * da, generator=g, dtype=F64)
    poly = DF.Polyhedron(G=G, h=torch.ones(B, 6, dtype=F64))
    vf = random_field(B, H, da, g)
    xi = torch.randn(B, H, da, generator=g, dtype=F64)
    r = DF.guided_solve(vf, xi, poly, K, nu=[0.0] * K, n_channels=da, clip=False)
    A = xi.clone()
    for k in range(K):
        A = A + vf(A, k / K) / K
    assert torch.allclose(r.A, A, atol=1e-12)
    assert float(r.intervention.abs().max()) == 0.0


def test_stage_a_rows_match_their_definitions_and_groups_are_independent():
    B = 4
    poly, cmap, v_hat = _stage_a_case(B, 77)
    H, N, Dt = poly.H, poly.N, P.T_c
    g = gen(78)
    a = torch.rand(B, H, 3, generator=g, dtype=F64) * 2 - 1
    V = cmap.to_world(a)
    lhs = (poly.G @ a.reshape(B, -1).unsqueeze(-1)).squeeze(-1) - poly.h
    S = poly.shares
    u = (V[:, 1:] - V[:, :-1]) / Dt                                         # (B, H-1, 3)
    s0, s1 = poly.layout["share"]
    share_expected = -Dt * ((S.g * u.unsqueeze(2)).sum(-1) - S.zeta)       # (B, H-1, N)
    on = poly.row_mask[:, s0:s1]
    assert torch.allclose(lhs[:, s0:s1][on], share_expected.reshape(B, -1)[on], atol=1e-9)
    a0, a1 = poly.layout["accel"]
    sign = torch.tensor([1, 1, 1, -1, -1, -1], dtype=F64)
    acc_expected = Dt * (sign * (u.unsqueeze(2).expand(B, H - 1, 2, 3).reshape(B, H - 1, 6)) - P.u_inf)
    assert torch.allclose(lhs[:, a0:a1], acc_expected.reshape(B, -1), atol=1e-9)
    v0, v1 = poly.layout["speed"]
    spd_expected = sign * V.unsqueeze(2).expand(B, H, 2, 3).reshape(B, H, 6) - P.v_bar
    assert torch.allclose(lhs[:, v0:v1], spd_expected.reshape(B, -1), atol=1e-9)
    c0, c1 = poly.layout["chunk_box"]
    box_expected = sign * a.unsqueeze(2).expand(B, H, 2, 3).reshape(B, H, 6) - 1.0
    assert torch.allclose(lhs[:, c0:c1], box_expected.reshape(B, -1), atol=1e-12)
    assert Q.groups_conflict_free(poly.G, poly.groups)
    assert len(poly.groups) == 2 * N + 24


def test_chunk_map_roundtrip_and_reference_helpers():
    g = gen(79)
    cmap = DF.ChunkMap.from_yaw(torch.rand(3, generator=g, dtype=F64) * 6, scale=[8.0, 8.0, 4.0],
                                offset=[0.0, 0.0, 0.5])
    a = torch.rand(3, 32, 3, generator=g, dtype=F64) * 2 - 1
    assert torch.allclose(cmap.to_normalized(cmap.to_world(a)), a, atol=1e-12)
    V = torch.randn(2, 32, 3, generator=g, dtype=F64)
    assert torch.allclose(DF.shift_reference(V, 0.0), V)
    sh = DF.shift_reference(V, 2.5)
    assert torch.allclose(sh[:, 0], 0.5 * (V[:, 2] + V[:, 3])) and torch.allclose(sh[:, -1], V[:, -1])
    p = DF.integrate_positions(torch.zeros(2, 3, dtype=F64), V, 0.01)
    assert torch.allclose(p[:, -1], 0.01 * (0.5 * V[:, 0] + V[:, 1:-1].sum(1) + 0.5 * V[:, -1]))


# ===========================================================================
# solvers used by Stage A / Stage B
# ===========================================================================
def test_grouped_hildreth_is_exactly_rowwise_hildreth():
    poly, _, _ = _stage_a_case(3, 80)
    g = gen(81)
    y = torch.rand(3, poly.G.shape[-1], generator=g, dtype=F64) * 1.6 - 0.8
    order = [r for grp in poly.groups for r in grp]
    singles = [(r,) for r in order]
    kw = dict(row_mask=poly.row_mask, max_sweeps=7, tol=-1.0)    # tol < 0: run every sweep
    a = Q.hildreth(y, poly.G, poly.h, groups=poly.groups, **kw)
    b = Q.hildreth(y, poly.G, poly.h, groups=singles, **kw)
    assert a.sweeps == b.sweeps == 7
    assert torch.allclose(a.x, b.x, atol=1e-11) and torch.allclose(a.lam, b.lam, atol=1e-9)
    assert Q.groups_conflict_free(poly.G, Q.greedy_row_coloring(poly.G))


def test_goldfarb_idnani_matches_brute_force():
    g = gen(82)
    n_prob, n, m = 120, 4, 9
    y = 2 * torch.randn(n_prob, n, generator=g, dtype=F64)
    G = torch.randn(n_prob, m, n, generator=g, dtype=F64)
    h = (G @ (0.3 * torch.randn(n_prob, n, 1, generator=g, dtype=F64))).squeeze(-1) \
        + 0.4 * torch.rand(n_prob, m, generator=g, dtype=F64) - 0.15    # some problems infeasible
    mask = torch.ones(n_prob, m, dtype=torch.bool)
    x, lam, ok = Q._goldfarb_idnani(y, G, h, G.transpose(-1, -2), mask, 1e-11,
                                    torch.ones(n_prob, dtype=torch.bool), 200)
    n_feas = 0
    for i in range(n_prob):
        xb = brute_force_qp(y[i].numpy(), G[i].numpy(), h[i].numpy())
        assert (xb is not None) == bool(ok[i]), i
        if xb is not None:
            n_feas += 1
            assert np.allclose(x[i].numpy(), xb, atol=1e-8)
            assert float(lam[i].min()) >= 0.0
    assert 10 < n_feas < n_prob


def test_stage_b_fixed_matches_routed_solver():
    g = gen(83)
    n, N, d = 600, 4, 3
    u_hat = (2 * torch.rand(n, d, generator=g, dtype=F64) - 1) * P.u_inf
    gg = 2 * (1.2 + 4 * torch.rand(n, N, 1, generator=g, dtype=F64)) * rand_unit(n * N, d, g).reshape(n, N, d)
    u0 = (2 * torch.rand(n, d, generator=g, dtype=F64) - 1) * 0.9 * P.u_inf
    zeta = (gg * u0[:, None]).sum(-1) - 0.5 * torch.rand(n, N, generator=g, dtype=F64)
    zeta[: n // 10] += 40.0                                     # some infeasible
    act = torch.rand(n, N, generator=g) < 0.6
    a = Q.solve_stage_b_qp(u_hat, gg, zeta, act, P.u_inf)
    b = Q.stage_b_fixed(u_hat, gg, zeta, act, P.u_inf)
    assert torch.equal(a.n_active, b.n_active)
    both = a.optimal & b.optimal
    assert float(both.double().mean()) > 0.6
    assert torch.allclose(a.u[both], b.u[both], atol=1e-9)
    assert not bool((b.ok & ~a.ok).any())                      # never feasible where the exact solver is not
    slack = (gg * b.u.unsqueeze(-2)).sum(-1) - zeta
    assert bool((slack[b.ok.unsqueeze(-1).expand_as(act) & act] >= 0).all())  # ok is an exact certificate


# ===========================================================================
# Executor: 100 Hz mock runs
# ===========================================================================
def make_chunk(v: torch.Tensor, s: float, H: int = 32, H_val: Optional[int] = None) -> EX.Chunk:
    B = v.shape[0]
    return EX.Chunk(v.unsqueeze(1).expand(B, H, 3).clone(), torch.zeros(B, H, dtype=F64),
                    torch.full((B,), s, dtype=F64), torch.full((B,), H if H_val is None else H_val, dtype=torch.long))


def run_team(ex: EX.Executor, p: torch.Tensor, v: torch.Tensor, goal: torch.Tensor, n_ticks: int,
             speed: float = 4.0, decision_period: float = 0.033, n_sub: int = 5,
             sigma_bar: Optional[Callable[[int], torch.Tensor]] = None,
             hooks: Optional[Callable[[int, Dict], None]] = None, t0: float = 0.0,
             dist_frac: float = 0.8, seed: int = 0) -> Dict[str, object]:
    """Close the loop: straight-line 'clone' chunks -> executor -> disturbed double integrator.

    The plant realizes the commanded acceleration feed-forward plus a random
    disturbance of norm ``<= dist_frac * e_a`` (Assumption 1), and holds
    position after a completed braking profile.
    """
    B = p.shape[0]
    gd = gen(1000 + seed)
    log: Dict[str, object] = {"min_dist": math.inf, "flags": [], "cause": [], "radius": [], "cmds": []}
    next_dec = t0
    for n in range(n_ticks):
        s = t0 + n * P.T_c
        if s >= next_dec - 1e-12:
            d = goal - p
            dist = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
            vdes = d / torch.clamp(dist, min=1e-6) * torch.clamp(dist, max=speed)
            ex.publish_chunk(torch.ones(B, dtype=torch.bool), make_chunk(vdes, s), s)
            next_dec += decision_period
        ctx = {"s": s, "p": p, "v": v}
        bc = ex.broadcast(s, p, v).quantized()
        ctx["bc"] = bc
        nb = EX.neighbor_tables_from_broadcasts(bc)
        ctx["nb"] = nb
        ctx["odom_stamp"] = None
        if hooks is not None:
            hooks(n, ctx)
        cmd = ex.tick(s, p, v, ctx["nb"], odom_stamp=ctx["odom_stamp"],
                      sigma_bar=None if sigma_bar is None else sigma_bar(n))
        log["cmds"].append(cmd)
        log["flags"].append(cmd.flags.clone())
        log["cause"].append(cmd.cause.clone())
        log["radius"].append(cmd.radius.clone())
        h = P.T_c / n_sub
        dist = rand_unit(B, 3, gd) * (dist_frac * P.e_a * torch.rand(B, 1, generator=gd, dtype=F64))
        hold = ~torch.isnan(cmd.position[:, 0])
        for _ in range(n_sub):
            a = cmd.acceleration + dist
            a = torch.where(hold.unsqueeze(-1), torch.clamp(-v / 0.05, -P.u_inf, P.u_inf), a)
            p = p + v * h + 0.5 * a * h * h
            v = v + a * h
        if B > 1:
            dd = torch.cdist(p, p) + torch.eye(B, dtype=F64) * 1e9
            log["min_dist"] = min(log["min_dist"], float(dd.min()))
            dp, dv = H.pairwise_relative(p, v)
            Dij = cmd.radius.unsqueeze(-1) + cmd.radius.unsqueeze(0)
            off = ~torch.eye(B, dtype=torch.bool)
            t = H.hocbf_terms(dp[off], dv[off], Dij[off])
            log["min_psi0"] = min(log.get("min_psi0", math.inf), float(t.psi0.min()))
            log["min_psi1"] = min(log.get("min_psi1", math.inf), float(t.psi1.min()))
    log["p"], log["v"] = p, v
    return log


def crossing_team(B: int = 6, radius: float = 12.0, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    g = gen(seed)
    ang = torch.arange(B, dtype=F64) * 2 * math.pi / B + 0.15 * torch.rand(B, generator=g, dtype=F64)
    p = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang),
                     0.5 * torch.rand(B, generator=g, dtype=F64)], -1)
    goal = -p + torch.stack([torch.zeros(B, dtype=F64), 1.5 * torch.ones(B, dtype=F64), torch.zeros(B, dtype=F64)], -1)
    return p, goal


def _check_flags_consistent(log: Dict[str, object]) -> None:
    for c in log["cmds"]:
        f = c.flags.to(torch.int64)
        assert torch.equal((f & EX.FLAG_BRAKING) != 0, c.braking)
        primary = (c.cause & ~EX.CAUSE_NEIGHBOR_BRAKING) != 0
        assert torch.equal((f & EX.FLAG_FALLBACK) != 0, c.braking & primary)
        holding = ~torch.isnan(c.position[:, 0])
        assert bool((~holding | c.braking).all())             # position hold only while braking


def test_executor_100_ticks_latency_and_safety():
    """Stage-B real-time mock: 6 quadrotors already crossing, 100 consecutive 100 Hz ticks."""
    B = 6
    p, goal = crossing_team(B, radius=6.0)
    v0 = 4.0 * (goal - p) / torch.linalg.vector_norm(goal - p, dim=-1, keepdim=True)
    t = H.hocbf_terms(*(x.reshape(-1, 3) for x in H.pairwise_relative(p, v0)))
    off = ~torch.eye(B, dtype=torch.bool).reshape(-1)
    assert bool(t.in_safe_set[off].all())                     # start inside C^HO
    run_team(EX.Executor(B, B - 1), p.clone(), v0.clone(), goal, 3)     # warm-up (first-call overheads)
    ex = EX.Executor(B, B - 1)
    log = run_team(ex, p, v0, goal, 100)
    st = ex.latency_stats()
    assert len(ex.tick_ms) == 100
    assert st["p99"] < 50.0                                    # eager CPU, whole team batched per tick
    n_act = torch.stack([c.n_active for c in log["cmds"]])
    assert int((n_act >= 2).sum()) > 0                         # Stage B engaged, incl. Hildreth path
    _check_flags_consistent(log)
    # The dense symmetric crossing crowds the centre until agents are surrounded
    # (cone condition of Theorem 3(b) fails), Stage B becomes infeasible and the
    # team brakes in cascade - the behaviour Section 8 predicts without the
    # master's joint mode. Every onset is logged; separation stays above D in
    # this deterministic run (a regression check, not a theorem).
    assert len(ex.events) > 0 and all(ev.cause != 0 for ev in ex.events)
    assert log["min_dist"] >= 2 * P.r_s


def test_executor_long_crossing_safety():
    """900 ticks of the dense crossing: logged fallbacks, consistent flags, no contact."""
    B = 6
    p, goal = crossing_team(B, seed=3)
    ex = EX.Executor(B, B - 1)
    log = run_team(ex, p, torch.zeros(B, 3, dtype=F64), goal, 900, seed=3)
    _check_flags_consistent(log)
    for ev in ex.events:                                       # every fallback is logged with Prop. 9 status
        assert isinstance(ev.certified, bool) and ev.cause != 0 and ev.n_pairs == B - 1
    assert log["min_dist"] >= 2 * P.r_s                        # deterministic regression check


THEOREM1_CASES = {
    "pair head-on, 1.0 m offset": ([[-10.0, 0, 0], [10.0, 1.0, 0]], [[10.0, 0, 0], [-10.0, 1.0, 0]], True),
    "pair head-on, 0.3 m offset": ([[-10.0, 0, 0], [10.0, 0.3, 0]], [[10.0, 0, 0], [-10.0, 0.3, 0]], False),
    "3 agents, altitude-staggered": ([[-10.0, 0, 0], [5.0, 8.66, 0.6], [5.0, -8.66, -0.6]],
                                     [[10.0, 0.5, 0], [-5.0, -8.66, 0.6], [-5.0, 8.66, -0.6]], True),
}


@pytest.mark.parametrize("case", list(THEOREM1_CASES))
def test_executor_theorem1_closed_loop(case):
    """Closed loop with quantized broadcasts and |d| <= 0.8 e_a: while Stage B stays
    feasible (Assumption 3), psi_0 and psi_1 stay non-negative (Theorem 1)."""
    p0, goal, reaches = THEOREM1_CASES[case]
    p, goal = torch.tensor(p0, dtype=F64), torch.tensor(goal, dtype=F64)
    B = p.shape[0]
    ex = EX.Executor(B, B - 1)
    log = run_team(ex, p, torch.zeros(B, 3, dtype=F64), goal, 700)
    n_act = torch.stack([c.n_active for c in log["cmds"]])
    assert int((n_act > 0).sum()) > 100                       # the barrier is active for a long time
    assert all(bool(c.stage_b_ok.all()) for c in log["cmds"]) and not ex.events
    assert log["min_psi0"] >= 0.0 and log["min_psi1"] >= 0.0
    assert log["min_dist"] >= 2 * P.r_s
    if B == 3:
        assert int((n_act >= 2).sum()) > 0                    # Hildreth path exercised
    if reaches:
        assert float(torch.linalg.vector_norm(log["p"] - goal, dim=-1).max()) < 1.5


def test_executor_no_chunk_and_expired_chunk_trigger_fallback():
    ex = EX.Executor(1, 1)
    nb = EX.NeighborTable(pos=torch.full((1, 1, 3), 100.0, dtype=F64), vel=torch.zeros(1, 1, 3, dtype=F64),
                          radius=torch.full((1, 1), P.r_s, dtype=F64), flags=torch.zeros(1, 1, dtype=torch.uint8),
                          stamp=torch.zeros(1, 1, dtype=F64), present=torch.zeros(1, 1, dtype=torch.bool),
                          ids=torch.ones(1, 1, dtype=torch.long))
    p, v = torch.zeros(1, 3, dtype=F64), torch.tensor([[3.0, 0, 0]], dtype=F64)
    cmd = ex.tick(0.0, p, v, nb)                                # nothing published yet
    assert bool(cmd.braking[0]) and int(cmd.cause[0]) & EX.CAUSE_NO_CHUNK
    assert int(cmd.flags[0]) == EX.FLAG_BRAKING | EX.FLAG_FALLBACK
    # braking reference follows Eq. (36) from the onset velocity
    ref = FB.braking_profile(v, P.T_c)
    assert torch.allclose(cmd.velocity, ref.velocity) and torch.allclose(cmd.acceleration, ref.acceleration)
    # a chunk with a short validity horizon expires
    ex2 = EX.Executor(1, 1)
    ex2.publish_chunk(torch.ones(1, dtype=torch.bool), make_chunk(v, 0.0, H_val=5), 0.0)
    assert not bool(ex2.tick(0.04, p, v, nb).braking[0])
    c = ex2.tick(0.06, p, v, nb)
    assert bool(c.braking[0]) and int(c.cause[0]) & EX.CAUSE_NO_CHUNK


def test_executor_stale_odometry_brake_hold_and_release():
    ex = EX.Executor(1, 1, EX.ExecutorConfig(release_ticks=5))
    nb = EX.NeighborTable(pos=torch.full((1, 1, 3), 100.0, dtype=F64), vel=torch.zeros(1, 1, 3, dtype=F64),
                          radius=torch.full((1, 1), P.r_s, dtype=F64), flags=torch.zeros(1, 1, dtype=torch.uint8),
                          stamp=torch.zeros(1, 1, dtype=F64), present=torch.zeros(1, 1, dtype=torch.bool),
                          ids=torch.ones(1, 1, dtype=torch.long))
    p, v = torch.zeros(1, 3, dtype=F64), torch.tensor([[2.0, 0, 0]], dtype=F64)
    braking, holds, released_at = [], [], None
    for n in range(200):
        s = n * P.T_c
        if n % 3 == 0:
            ex.publish_chunk(torch.ones(1, dtype=torch.bool), make_chunk(torch.tensor([[2.0, 0, 0]], dtype=F64), s), s)
        stale = 20 <= n < 40
        odom = torch.tensor([s - 0.2 if stale else s], dtype=F64)
        cmd = ex.tick(s, p, v, nb, odom_stamp=odom)
        braking.append(bool(cmd.braking[0]))
        holds.append(not math.isnan(float(cmd.position[0, 0])))
        if n == 20:
            assert int(cmd.cause[0]) & EX.CAUSE_STALE_ODOM and int(cmd.flags[0]) & EX.FLAG_FALLBACK
        v = cmd.velocity.clone()                                # perfect tracking
        p = p + v * P.T_c
        if released_at is None and n > 20 and not braking[-1]:
            released_at = n
    T_b = 1.5 * 2.0 / P.a_brk                                  # 0.6 s = 60 ticks
    assert not any(braking[:20]) and all(braking[20:40])
    assert any(holds) and released_at is not None
    assert released_at >= 20 + int(T_b / P.T_c) - 1           # never released before the profile ends
    assert not any(braking[released_at:])


def test_executor_neighbor_flags_and_stale_neighbors():
    B = 2
    ex = EX.Executor(B, 1)
    p = torch.tensor([[0.0, 0, 0], [8.0, 0, 0]], dtype=F64)
    v = torch.zeros(B, 3, dtype=F64)
    ex.publish_chunk(torch.ones(B, dtype=torch.bool), make_chunk(v, 0.0), 0.0)
    bc = ex.broadcast(0.0, p, v)
    # neighbour 1 reports FALLBACK -> agent 0 brakes sympathetically
    bc.flags[1] = EX.FLAG_FALLBACK | EX.FLAG_BRAKING
    cmd = ex.tick(0.0, p, v, EX.neighbor_tables_from_broadcasts(bc))
    assert bool(cmd.braking[0]) and int(cmd.cause[0]) & EX.CAUSE_NEIGHBOR_BRAKING
    assert not (int(cmd.flags[0]) & EX.FLAG_FALLBACK)         # sympathetic only: no primary cause
    # a holding (stopped, BRAKING-only) neighbour does not trigger braking
    ex2 = EX.Executor(B, 1)
    ex2.publish_chunk(torch.ones(B, dtype=torch.bool), make_chunk(v, 0.0), 0.0)
    bc2 = ex2.broadcast(0.0, p, v)
    bc2.flags[1] = EX.FLAG_BRAKING
    assert not bool(ex2.tick(0.0, p, v, EX.neighbor_tables_from_broadcasts(bc2)).braking[0])
    # a stale broadcast of a nearby neighbour triggers braking, of a far one does not
    ex3 = EX.Executor(B, 1)
    ex3.publish_chunk(torch.ones(B, dtype=torch.bool), make_chunk(v, 0.0), 0.0)
    nb = EX.neighbor_tables_from_broadcasts(ex3.broadcast(0.0, p, v))
    nb.stamp[:] = -0.5
    c = ex3.tick(0.0, p, v, nb)
    assert bool(c.braking.all()) and int(c.cause[0]) & EX.CAUSE_STALE_NEIGHBOR
    ex4 = EX.Executor(B, 1)
    pf = torch.tensor([[0.0, 0, 0], [60.0, 0, 0]], dtype=F64)
    ex4.publish_chunk(torch.ones(B, dtype=torch.bool), make_chunk(v, 0.0), 0.0)
    nb4 = EX.neighbor_tables_from_broadcasts(ex4.broadcast(0.0, pf, v))
    nb4.stamp[:] = -0.5
    assert not bool(ex4.tick(0.0, pf, v, nb4).braking.any())


def test_executor_stage_b_infeasible_logs_prop9_event():
    B = 2
    ex = EX.Executor(B, 1)
    # head-on pair barely outside the thin layer: Stage B ill-posed for exact grazing is not needed;
    # use a state outside C^HO (closing faster than cbar) so the share cannot be met
    p = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]], dtype=F64)
    v = torch.tensor([[4.0, 0, 0], [-4.0, 0, 0]], dtype=F64)
    ex.publish_chunk(torch.ones(B, dtype=torch.bool), make_chunk(v, 0.0), 0.0)
    cmd = ex.tick(0.0, p, v, EX.neighbor_tables_from_broadcasts(ex.broadcast(0.0, p, v)))
    assert not bool(cmd.stage_b_ok.any()) and bool(cmd.braking.all())
    assert all(int(c) & EX.CAUSE_STAGE_B for c in cmd.cause)
    assert len(ex.events) == 2 and all(not e.certified for e in ex.events) and ex.events[0].n_pairs == 1
    need = FB.fallback_envelope(torch.tensor(8.0, dtype=F64), 2 * P.r_s)
    assert ex.events[0].min_margin == pytest.approx(3.0 - float(need), rel=1e-9)


def test_radius_admission_bisection():
    B = 3
    ex = EX.Executor(B, 1)
    v0 = torch.zeros(B, 3, dtype=F64)
    # agent 0: no neighbour in range; agent 1: neighbour close; agent 2: decrease request
    p = torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]], dtype=F64)
    nb_pos = torch.tensor([[[100.0, 0, 0]], [[2.0, 0, 0]], [[50.0, 0, 0]]], dtype=F64)
    nb_vel = torch.tensor([[[0.0, 0, 0]], [[-1.5, 0, 0]], [[0.0, 0, 0]]], dtype=F64)
    nb = EX.NeighborTable(pos=nb_pos, vel=nb_vel, radius=torch.full((B, 1), P.r_s, dtype=F64),
                          flags=torch.zeros(B, 1, dtype=torch.uint8), stamp=torch.zeros(B, 1, dtype=F64),
                          present=torch.ones(B, 1, dtype=torch.bool), ids=torch.ones(B, 1, dtype=torch.long))
    ex.radius = torch.tensor([P.r_s, P.r_s, 1.2], dtype=F64)
    ex.publish_chunk(torch.ones(B, dtype=torch.bool), make_chunk(v0, 0.0), 0.0)
    sigma = torch.tensor([10.0, 10.0, 0.0], dtype=F64)            # requests r_max, r_max, r_s
    cmd = ex.tick(0.0, p, v0, nb, sigma_bar=sigma)
    r = cmd.radius_next
    assert float(r[0]) == pytest.approx(P.r_max)                    # free: full increase
    assert float(r[2]) == pytest.approx(P.r_s)                      # decrease admitted at once
    assert P.r_s <= float(r[1]) < P.r_max                           # limited by the close neighbour
    # the admitted value passes the test and a value one resolution step above fails it
    dp = (p[1] - nb_pos[1, 0]).view(1, 1, 3)
    dv = (v0[1] - nb_vel[1, 0]).view(1, 1, 3)
    ok_at = M.reciprocal_shares(dp, dv, float(r[1]) + P.r_s).admission
    res = (P.r_max - P.r_s) / (ex.config.radius_points + 1) ** ex.config.radius_rounds
    ok_above = M.reciprocal_shares(dp, dv, float(r[1]) + 2 * res + P.r_s).admission
    assert bool(ok_at.all()) and not bool(ok_above.all())
    assert torch.allclose(ex.radius, r)                            # takes effect at the next tick
    bc = ex.broadcast(0.01, p, v0)
    assert torch.allclose(bc.radius, r)                            # and is broadcast now


def test_radius_admission_in_closed_loop():
    B = 4
    p, goal = crossing_team(B, radius=10.0, seed=5)
    ex = EX.Executor(B, B - 1)
    sig = lambda n: torch.full((B,), 2.0 if 30 <= n < 250 else 0.0, dtype=F64)  # noqa: E731
    log = run_team(ex, p, torch.zeros(B, 3, dtype=F64), goal, 300, sigma_bar=sig)
    R = torch.stack(log["radius"])
    assert float(R[:30].max()) == pytest.approx(P.r_s)
    assert float(R.max()) > P.r_s                                    # increases admitted
    assert float(R.max()) <= P.r_max + 1e-12
    assert float(R[-1].max()) == pytest.approx(P.r_s)               # decreases admitted at once
    assert log["min_dist"] >= 2 * P.r_s


def test_chunk_blending_over_T_bl():
    ex = EX.Executor(1, 1, EX.ExecutorConfig(T_bl=0.05))
    v_old, v_new = torch.tensor([[2.0, 0, 0]], dtype=F64), torch.tensor([[0.0, 2.0, 0]], dtype=F64)
    ex.publish_chunk(torch.ones(1, dtype=torch.bool), make_chunk(v_old, 0.0), 0.0)
    ex.publish_chunk(torch.ones(1, dtype=torch.bool), make_chunk(v_new, 0.1), 0.1)
    for s, w in ((0.1, 0.0), (0.125, 0.5), (0.15, 1.0), (0.2, 1.0)):
        vb, _, ok = ex.nominal_velocity(s, s)
        assert bool(ok[0])
        assert torch.allclose(vb, (1 - w) * v_old + w * v_new, atol=1e-12), (s, vb)


def test_px4_message_content():
    ex = EX.Executor(1, 1)
    p, v = torch.zeros(1, 3, dtype=F64), torch.tensor([[1.0, 2.0, 3.0]], dtype=F64)
    ex.publish_chunk(torch.ones(1, dtype=torch.bool), make_chunk(v, 0.0), 0.0)
    nb = EX.NeighborTable(pos=torch.full((1, 1, 3), 100.0, dtype=F64), vel=torch.zeros(1, 1, 3, dtype=F64),
                          radius=torch.full((1, 1), P.r_s, dtype=F64), flags=torch.zeros(1, 1, dtype=torch.uint8),
                          stamp=torch.zeros(1, 1, dtype=F64), present=torch.zeros(1, 1, dtype=torch.bool),
                          ids=torch.ones(1, 1, dtype=torch.long))
    cmd = ex.tick(0.0, p, v, nb)
    msg = EX.px4_trajectory_setpoint(cmd, 0, 1_000_000)
    assert msg["velocity"] == pytest.approx([2.0, 1.0, -3.0])         # ENU -> NED
    assert all(math.isnan(x) for x in msg["position"]) and math.isnan(msg["yaw"])
    assert msg["acceleration"] == pytest.approx([0.0, 0.0, 0.0])
    ocm = EX.px4_offboard_control_mode(cmd, 0, 1_000_000)
    assert ocm["velocity"] and ocm["acceleration"] and not ocm["position"]
    q = EX.SafetyBroadcast(torch.zeros(1, dtype=torch.long), torch.tensor([300]), torch.zeros(1, dtype=F64),
                           torch.tensor([[1.234567, -400.0, 0.004]], dtype=F64), torch.tensor([[0.016, 0, 0]], dtype=F64),
                           torch.tensor([0.6012], dtype=F64), torch.zeros(1, dtype=torch.uint8)).quantized()
    assert q.pos.tolist()[0] == pytest.approx([1.23, -327.67, 0.0]) and float(q.radius) == pytest.approx(0.61)
    assert int(q.seq) == 300 % 256


@pytest.mark.skipif(os.environ.get("CMA_VPA_TEST_COMPILE") != "1", reason="set CMA_VPA_TEST_COMPILE=1")
def test_compiled_executor_matches_eager():
    B = 4
    p, goal = crossing_team(B, seed=7)
    ex_e = EX.Executor(B, B - 1)
    ex_c = EX.Executor(B, B - 1, EX.ExecutorConfig(compile=True))
    le = run_team(ex_e, p.clone(), torch.zeros(B, 3, dtype=F64), goal, 120)
    lc = run_team(ex_c, p.clone(), torch.zeros(B, 3, dtype=F64), goal, 120)
    for a, b in zip(le["cmds"], lc["cmds"]):
        assert torch.allclose(a.velocity, b.velocity, atol=1e-8)
        assert torch.equal(a.braking, b.braking)


# ===========================================================================
# report
# ===========================================================================
def stage_b_latency_rows(compiled: bool = False, N: int = 8) -> List[Tuple[str, str, str, str]]:  # pragma: no cover
    """Per-tick Stage-B cost (shares + projection) for one vehicle with an ``N``-slot neighbour table."""
    torch.set_num_threads(1)
    g = gen(3)
    u_hat = torch.tensor([[3.0, -1.0, 0.5]], dtype=F64)
    heard = torch.ones(1, N, dtype=torch.bool)
    Dm = torch.full((1, N), D, dtype=F64)

    def state(dirs_close: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = dirs_close.shape[0]
        dirs = rand_unit(N, 3, g).unsqueeze(0)
        dirs[0, :k] = dirs_close
        rho = torch.full((1, N, 1), 25.0, dtype=F64)
        rho[0, :k, 0] = 2.0 + 0.5 * torch.rand(k, generator=g, dtype=F64)
        return rho * dirs, -0.9 * H.closing_speed_limit(rho) * dirs

    cases = {"0 active": torch.zeros(0, 3, dtype=F64),
             "1 active": rand_unit(1, 3, g),
             "2 active (feasible)": torch.nn.functional.normalize(torch.tensor([[1.0, 0.2, 0], [1.0, -0.2, 0]], dtype=F64), dim=-1),
             "3 active (surrounded)": torch.nn.functional.normalize(torch.tensor([[1.0, 0, 0], [-0.5, 0.86, 0], [-0.5, -0.86, 0]], dtype=F64), dim=-1)}
    kern = torch.compile(EX._knot_kernel, dynamic=False) if compiled else None
    rows = []
    for name, dc in cases.items():
        dp, dv = state(dc)
        S = M.reciprocal_shares(dp, dv, Dm, heard=heard)
        lam = Q.solve_stage_b_qp(u_hat, S.g, S.zeta, S.active, P.u_inf).lam

        def eager():
            s = M.reciprocal_shares(dp, dv, Dm, heard=heard)
            return Q.solve_stage_b_qp(u_hat, s.g, s.zeta, s.active, P.u_inf, lam0=lam)

        for _ in range(30):
            r = eager()
        t0 = time.perf_counter()
        for _ in range(300):
            eager()
        te = (time.perf_counter() - t0) / 300 * 1e3
        txt = f"eager {te:.3f} ms (n_active {int(r.n_active)}, ok {bool(r.ok)})"
        if kern is not None:
            for _ in range(5):
                kern(dp, dv, Dm, heard, u_hat, lam, P, 50)
            t0 = time.perf_counter()
            for _ in range(300):
                kern(dp, dv, Dm, heard, u_hat, lam, P, 50)
            tc = (time.perf_counter() - t0) / 300 * 1e3
            txt += f"; compiled kernel {tc:.3f} ms" + ("" if int(r.n_active) <= 1 else " + eager finish")
        rows.append(("Stage B", f"1 vehicle, N={N} slots, {name}", txt, "<= 1 ms (H1)"))
    return rows


def _report() -> None:  # pragma: no cover - manual run
    rows = []
    r = prop9_simulation(n=20000)
    rows.append(("Prop. 9", "20,000 reciprocal braking scenarios on Eq. (41)",
                 f"min margin above D {r['min_margin']:.2f} m; dv bound excess {r['dv_bound_excess']:.1e}",
                 "7.12 m; never violated"))
    for K in (2, 3, 5):
        B = 1
        poly, cmap, v_hat = _stage_a_case(B, 30 + K)
        g = gen(40 + K)
        target = torch.cat([cmap.to_normalized(v_hat.unsqueeze(1).expand(B, 32, 3)), torch.zeros(B, 32, 1, dtype=F64)], -1)
        base = random_field(B, 32, 4, g, smooth=True)
        vf = lambda A, tau: (target - A) + 0.3 * base(A, tau)  # noqa: E731
        xi = 0.3 * torch.randn(B, 32, 4, generator=g, dtype=F64)
        DF.guided_solve(vf, xi, poly, K)
        t0 = time.perf_counter()
        res = DF.guided_solve(vf, xi, poly, K)
        rows.append(("Lemma 17", f"Stage-A guided solve, K={K}, {poly.G.shape[1]} rows",
                     f"{(time.perf_counter() - t0) * 1e3:.1f} ms; exact={bool(res.terminal_exact.all())}; "
                     f"sweeps {res.sweeps}", "A_K in P_t"))
    for B in (1, 6):
        p, goal = crossing_team(B if B > 1 else 2)
        p, goal = p[:max(B, 1)], goal[:max(B, 1)]
        ex = EX.Executor(max(B, 1), max(B - 1, 1))
        if B == 1:
            nbh = lambda n, ctx: ctx.__setitem__("nb", EX.NeighborTable(  # noqa: E731
                pos=torch.tensor([[[6.0, 0.5, 0.0]]], dtype=F64), vel=torch.tensor([[[-1.0, 0, 0]]], dtype=F64),
                radius=torch.full((1, 1), P.r_s, dtype=F64), flags=torch.zeros(1, 1, dtype=torch.uint8),
                stamp=torch.full((1, 1), ctx["s"], dtype=F64), present=torch.ones(1, 1, dtype=torch.bool),
                ids=torch.ones(1, 1, dtype=torch.long)))
            run_team(EX.Executor(1, 1), p.clone(), torch.zeros(1, 3, dtype=F64), goal, 5, hooks=nbh)
            run_team(ex, p, torch.zeros(1, 3, dtype=F64), goal, 300, hooks=nbh)
        else:
            run_team(EX.Executor(B, B - 1), p.clone(), torch.zeros(B, 3, dtype=F64), goal, 5)
            run_team(ex, p, torch.zeros(B, 3, dtype=F64), goal, 300)
        st = ex.latency_stats()
        rows.append(("Executor", f"tick latency, {B} agent(s) batched, eager CPU",
                     f"p50 {st['p50']:.2f} / p95 {st['p95']:.2f} / p99 {st['p99']:.2f} ms", "Stage B <= 1 ms (H1)"))
    rows += stage_b_latency_rows(compiled=os.environ.get("CMA_VPA_TEST_COMPILE") == "1")
    w = [max(len(r_[i]) for r_ in rows) for i in range(3)]
    print(f"{'Statement':<{w[0]}}  {'Check':<{w[1]}}  {'Measured':<{w[2]}}  Paper")
    for a, b, c, d in rows:
        print(f"{a:<{w[0]}}  {b:<{w[1]}}  {c:<{w[2]}}  {d}")


if __name__ == "__main__":  # pragma: no cover
    _report()

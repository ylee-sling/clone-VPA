# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Numerical verification of the CMA-VPA safety engine (Appendix A, Table 9).

Run with ``pytest -q tests/test_safety.py`` or, for a Table-9 style report that
prints every measured value next to the value reported in the paper,
``python tests/test_safety.py``.

Ground truths are independent of the code under test:

* finite differences and autograd for the derivatives of ``alpha_1``;
* exhaustive active-set enumeration for every QP (exact for convex QPs);
* KKT certificates for the closed form of Lemma 18;
* fine-step simulation of the disturbed double integrator for Lemmas 13-14
  and Theorem 1.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cma_vpa.safety import hocbf as H  # noqa: E402
from cma_vpa.safety import margins as M  # noqa: E402
from cma_vpa.safety import qp_solvers as Q  # noqa: E402
from cma_vpa.safety.params import DEFAULT_PARAMS, SafetyParams  # noqa: E402

F64 = torch.float64
P = DEFAULT_PARAMS
D = P.D_nominal


# ===========================================================================
# helpers
# ===========================================================================
def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def rand_unit(n: int, d: int, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, d, generator=g, dtype=F64)
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def rand_ball(n: int, d: int, radius: float, g: torch.Generator) -> torch.Tensor:
    r = radius * torch.rand(n, 1, generator=g, dtype=F64) ** (1.0 / d)
    return r * rand_unit(n, d, g)


def sample_safe_states(n: int, V_max: float, rho_lo: float, rho_hi: float, g: torch.Generator,
                       Dv: float = D, cfg: SafetyParams = P) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sample ``n`` relative states in C^HO with rho ~ U[rho_lo, rho_hi], dv ~ U(ball)."""
    dps, dvs, have = [], [], 0
    while have < n:
        k = 4 * (n - have) + 64
        rho = rho_lo + (rho_hi - rho_lo) * torch.rand(k, 1, generator=g, dtype=F64)
        dp = rho * rand_unit(k, cfg.d, g)
        dv = rand_ball(k, cfg.d, V_max, g)
        t = H.hocbf_terms(dp, dv, Dv, cfg)
        keep = t.in_safe_set
        dps.append(dp[keep])
        dvs.append(dv[keep])
        have += int(keep.sum())
    return torch.cat(dps)[:n], torch.cat(dvs)[:n]


def sample_closing_states(n: int, g: torch.Generator, frac_lo: float = 0.3, frac_hi: float = 1.0,
                          rho_lo: float = 1.3, rho_hi: float = 8.0, v_tan: float = 3.0):
    """Approaching pairs in C^HO: closing speed ``U[frac_lo, frac_hi] * cbar(rho)`` plus tangential motion."""
    rho = rho_lo + (rho_hi - rho_lo) * torch.rand(n, generator=g, dtype=F64)
    nrm = rand_unit(n, 3, g)
    tang = torch.linalg.cross(nrm, rand_unit(n, 3, g))
    tang = tang / torch.linalg.vector_norm(tang, dim=-1, keepdim=True)
    frac = frac_lo + (frac_hi - frac_lo) * torch.rand(n, generator=g, dtype=F64)
    dv = (-frac * H.closing_speed_limit(rho))[:, None] * nrm \
        + v_tan * torch.rand(n, 1, generator=g, dtype=F64) * tang
    return rho[:, None] * nrm, dv


def brute_force_qp(y: np.ndarray, G: np.ndarray, h: np.ndarray, Winv: Optional[np.ndarray] = None,
                   tol: float = 1e-9) -> Optional[np.ndarray]:
    """Exact solution of ``min ||x - y||_W^2 s.t. G x <= h`` by active-set enumeration.

    The optimum is the W-projection of ``y`` onto ``{G_S x = h_S}`` for some
    linearly independent subset ``S`` of the active rows (Caratheodory), so the
    feasible candidate with the smallest objective is the optimum. Returns
    ``None`` if no candidate is feasible (the QP is infeasible).
    """
    n, m = y.size, G.shape[0]
    Winv = np.eye(n) if Winv is None else Winv
    W = np.linalg.inv(Winv)
    best, best_obj = None, math.inf
    for k in range(0, min(n, m) + 1):
        for S in itertools.combinations(range(m), k):
            if k == 0:
                x = y.copy()
            else:
                GS = G[list(S)]
                Mx = GS @ Winv @ GS.T
                if np.linalg.cond(Mx) > 1e10:
                    continue
                mu = np.linalg.solve(Mx, GS @ y - h[list(S)])
                x = y - Winv @ GS.T @ mu
            if np.all(G @ x - h <= tol * (1.0 + np.abs(h))):
                obj = float((x - y) @ W @ (x - y))
                if obj < best_obj - 1e-15:
                    best, best_obj = x, obj
    return best


def simulate_tick(dp: torch.Tensor, dv: torch.Tensor, du: torch.Tensor, n_sub: int,
                  g: torch.Generator, cfg: SafetyParams = P, adversarial: bool = True):
    """Integrate the relative double integrator over one tick with frozen ``du``.

    The relative disturbance has norm ``<= 2 e_a`` and is piecewise constant on
    ``n_sub`` substeps: random directions, or (adversarial) pointing along
    ``-dp`` to push the pair together. Yields the state after every substep.
    """
    h = cfg.T_c / n_sub
    for _ in range(n_sub):
        if adversarial:
            n = dp / torch.linalg.vector_norm(dp, dim=-1, keepdim=True)
            dd = -2.0 * cfg.e_a * n
        else:
            dd = rand_ball(dp.shape[0], dp.shape[1], 2.0 * cfg.e_a, g)
        a = du + dd
        dp = dp + dv * h + 0.5 * a * h * h
        dv = dv + a * h
        yield dp, dv


# ===========================================================================
# Lemma 8: braking-consistent class-K function
# ===========================================================================
def lemma8_measurements() -> Dict[str, float]:
    out: Dict[str, float] = {}
    b = torch.logspace(-3, math.log10(400.0), 4000, dtype=F64)
    hstep = 1e-5 * (1.0 + b)
    fd1 = (H.alpha1(b + hstep) - H.alpha1(b - hstep)) / (2 * hstep)
    fd2 = (H.alpha1_prime(b + hstep) - H.alpha1_prime(b - hstep)) / (2 * hstep)
    a1p, a1pp = H.alpha1_prime(b), H.alpha1_second(b)
    out["fd_rel_err_alpha1p"] = float(((fd1 - a1p).abs() / a1p.abs()).max())
    out["fd_rel_err_alpha1pp"] = float(((fd2 - a1pp).abs() / a1pp.abs()).max())

    bb = torch.linspace(float(H.b_ext(D)), 500.0, 200001, dtype=F64)
    cf, dr = H.alpha1_second(bb), H.alpha1_second_direct(bb)
    out["closed_vs_direct_rel_err"] = float(((cf - dr).abs() / dr.abs()).max())

    bgrid = torch.cat([bb, torch.zeros(1, dtype=F64)])
    X = H.alpha_terms_b(bgrid).X
    num = H.alpha1_second_numerator(X)
    i = int(torch.argmin(num))
    out["numerator_min"] = float(num[i])
    out["numerator_argmin_b"] = float(bgrid[i])
    out["numerator_min_exact"] = P.a_max * D * (P.a_max * D - P.varsigma ** 2)
    out["alpha1p_at_0"] = float(H.alpha1_prime(torch.zeros((), dtype=F64)))
    out["alpha1pp_at_0"] = float(H.alpha1_second(torch.zeros((), dtype=F64)))
    return out


def test_lemma8_finite_differences():
    m = lemma8_measurements()
    assert m["fd_rel_err_alpha1p"] < 1e-6
    assert m["fd_rel_err_alpha1pp"] < 1e-6


def test_lemma8_autograd_matches_closed_form():
    b = torch.linspace(float(H.b_ext(D)) + 1e-6, 400.0, 3001, dtype=F64, requires_grad=True)
    a1 = H.alpha1(b)
    (g1,) = torch.autograd.grad(a1.sum(), b, create_graph=True)
    (g2,) = torch.autograd.grad(g1.sum(), b)
    with torch.no_grad():
        assert torch.allclose(g1, H.alpha1_prime(b), rtol=1e-12, atol=1e-12)
        assert torch.allclose(g2, H.alpha1_second(b), rtol=1e-10, atol=1e-10)


def test_lemma8_closed_form_matches_direct_derivative():
    assert lemma8_measurements()["closed_vs_direct_rel_err"] < 1e-12


def test_lemma8_numerator_minimum_is_34_50_at_b0():
    m = lemma8_measurements()
    assert m["numerator_min_exact"] == pytest.approx(34.5, abs=1e-12)
    assert m["numerator_min"] == pytest.approx(34.5, abs=1e-9)
    assert abs(m["numerator_argmin_b"]) < 1e-12
    assert m["alpha1p_at_0"] == pytest.approx(P.a_max / P.varsigma, rel=1e-14)   # = 10
    assert m["alpha1pp_at_0"] == pytest.approx(-79.8611111111, rel=1e-10)


def test_lemma8_class_k_shape():
    b = torch.linspace(float(H.b_ext(D)), 400.0, 100001, dtype=F64)
    t = H.alpha_terms_b(b)
    assert bool(t.valid.all())
    assert float(H.alpha1(torch.zeros((), dtype=F64))) == 0.0
    assert bool((torch.diff(t.alpha1) > 0).all())          # strictly increasing
    assert bool((t.alpha1_p > 0).all())
    assert bool((t.alpha1_pp < 0).all())                   # strictly concave
    bound = -(P.a_max * D * (P.a_max * D - P.varsigma ** 2)) / (2 * t.X ** 3 * t.rho ** 3)
    assert bool((t.alpha1_pp <= bound * (1 - 1e-12)).all())  # Eq. (25) upper bound


def test_lemma8_psi1_equals_braking_gap():
    g = gen(1)
    dp = (D + 10 * torch.rand(5000, 1, generator=g, dtype=F64)) * rand_unit(5000, 3, g)
    dv = rand_ball(5000, 3, 16.0, g)
    t = H.hocbf_terms(dp, dv)
    assert torch.allclose(t.psi1, 2 * t.rho * (t.c_bar - t.closing_speed), atol=1e-10, rtol=1e-12)
    assert torch.equal(t.psi1 >= 0, t.closing_speed <= t.c_bar)


# ===========================================================================
# Lemma 11: input-bounded feasibility
# ===========================================================================
def lemma11_measurements(n_states: int = 200_000) -> Dict[str, float]:
    rho = torch.linspace(D, D + 50.0, 3_000_000, dtype=F64)
    ratio = H.closing_speed_limit(rho) / rho
    i = int(torch.argmax(ratio))
    lam_cf, rho_cf = H.lambda_min(D)
    dp, dv = sample_safe_states(n_states, 16.0, D, 10.0, gen(11))
    t = H.hocbf_terms(dp, dv)
    r = t.beta / (2 * t.rho)
    gap = r - P.a_max * t.c_bar / (t.c_bar + P.varsigma)
    return {"lambda_min_grid": float(ratio[i]), "rho_star_grid": float(rho[i]),
            "lambda_min_closed": float(lam_cf), "rho_star_closed": float(rho_cf),
            "max_beta_over_2rho": float(r.max()), "max_gap": float(gap.max())}


def test_lemma11_lambda_min_and_push_apart():
    m = lemma11_measurements()
    assert m["lambda_min_closed"] == pytest.approx(1.2613, abs=5e-5)
    assert m["rho_star_closed"] == pytest.approx(2.7464, abs=5e-5)
    assert m["lambda_min_grid"] == pytest.approx(m["lambda_min_closed"], abs=1e-9)
    assert m["rho_star_grid"] == pytest.approx(m["rho_star_closed"], abs=1e-4)
    assert P.lambda_b >= m["lambda_min_closed"]
    assert m["max_beta_over_2rho"] < P.a_max
    assert m["max_gap"] < 0.0                                   # Eq. (26)


# ===========================================================================
# Lemmas 9 and 10: nesting and monotonicity in D
# ===========================================================================
def test_lemma9_nesting():
    g = gen(9)
    n = 400_000
    dp = (D + 20 * torch.rand(n, 1, generator=g, dtype=F64)) * rand_unit(n, 3, g)
    dv = rand_ball(n, 3, 16.0, g)
    hs = H.spec_barrier(dp, dv) >= 0
    t = H.hocbf_terms(dp, dv)
    assert int(hs.sum()) > 10_000
    assert bool(t.in_safe_set[hs].all())                      # {h^s >= 0} in C^HO
    h0 = H.spec_barrier(dp, dv, varsigma=0.0) >= 0              # Eq. (21)
    brake = torch.sqrt(2 * P.a_max * torch.clamp(t.rho - D, min=0.0))
    assert int(h0.sum()) > 10_000
    assert bool((t.closing_speed[h0] <= brake[h0] + 1e-12).all())
    assert bool((t.rho[t.in_safe_set] >= D - 1e-12).all())      # C^HO in {rho >= D}


def test_lemma10_monotone_in_collision_distance():
    g = gen(10)
    dp = (1.6 + 10 * torch.rand(20000, 1, generator=g, dtype=F64)) * rand_unit(20000, 3, g)
    dv = rand_ball(20000, 3, 16.0, g)
    lo, hi = H.hocbf_terms(dp, dv, 1.2), H.hocbf_terms(dp, dv, 1.5)
    assert bool((hi.psi0 <= lo.psi0).all()) and bool((hi.psi1 <= lo.psi1 + 1e-12).all())


# ===========================================================================
# Lemma 12: local estimation margin
# ===========================================================================
def test_lemma12_grid_range():
    rho = torch.linspace(1.5, 3.0, 301, dtype=F64)[:, None].expand(301, 801)
    V = torch.linspace(0.0, 8.0, 801, dtype=F64)[None, :].expand(301, 801)
    em = M.estimation_margin(rho, V, D)
    r = em.eps_x / (4 * rho)
    assert float(r.min()) == pytest.approx(0.817, abs=5e-4)    # paper: 0.817
    assert float(r.max()) == pytest.approx(3.814, abs=5e-4)    # paper: 3.814
    assert bool((em.eps_psi <= 0.5 * em.eps_x).all())         # App. B.4 (lambda_b >= 2)


def test_lemma12_soundness():
    g = gen(12)
    n = 100_000
    dp, dv = sample_safe_states(n, 16.0, 1.4, 6.0, g)
    dp_hat = dp + rand_ball(n, 3, P.e_p, g)
    dv_hat = dv + rand_ball(n, 3, P.e_v, g)
    u = (2 * torch.rand(n, 3, generator=g, dtype=F64) - 1) * P.u_inf
    t_hat = H.hocbf_terms(dp_hat, dv_hat)
    em = M.estimation_margin(t_hat.rho, t_hat.V, D)
    ok = torch.isfinite(em.eps_x)
    dF = (H.share_slack(dp, dv, u) - H.share_slack(dp_hat, dv_hat, u)).abs()
    dpsi = (H.hocbf_terms(dp, dv).psi1 - t_hat.psi1).abs()
    assert int(ok.sum()) > 0.9 * n
    assert bool((dF[ok] <= 0.5 * em.eps_x[ok]).all())
    assert bool((dpsi[ok] <= em.eps_psi[ok]).all())


# ===========================================================================
# Lemma 13 / Remark 2: one-tick decrease bound
# ===========================================================================
def _lambda_point(dp, dv, Dv=D):
    t = H.hocbf_terms(dp, dv, Dv)
    reach = M.one_tick_reachable_box(M.point_box(t.rho, t.V, t.m), Dv)
    return M.one_tick_decrease_bound(reach, Dv), reach, t


def test_lemma13_one_tick_bound_is_sound():
    g = gen(13)
    n = 20_000
    dp, dv = sample_safe_states(n, 16.0, 1.3, 8.0, g)
    du = (2 * torch.rand(n, 3, generator=g, dtype=F64) - 1) * 2 * P.u_inf   # u_i - u_j in box
    Lam, reach, _ = _lambda_point(dp, dv)
    phi0 = H.pair_slack(dp, dv, du)
    fin = torch.isfinite(Lam)
    worst = torch.full((n,), math.inf, dtype=F64)
    inside = torch.ones(n, dtype=torch.bool)
    for adv in (True, False):
        for p_, v_ in simulate_tick(dp, dv, du, 50, g, adversarial=adv):
            phi = H.pair_slack(p_, v_, du)
            worst = torch.minimum(worst, phi - (phi0 - P.T_c * Lam))
            rho, V, m = (torch.linalg.vector_norm(p_, dim=-1), torch.linalg.vector_norm(v_, dim=-1),
                         (p_ * v_).sum(-1))
            tol = 1e-12
            inside &= ((V <= reach.V_plus + tol) & (V >= reach.V_minus - tol)
                       & (rho <= reach.rho_plus + tol) & (rho >= reach.rho_minus - tol)
                       & (m <= reach.m_plus + tol) & (m >= reach.m_minus - tol))
    assert int(fin.sum()) > 0.95 * n
    assert bool(inside.all())
    assert float(worst[fin].min()) >= -1e-9


def test_lemma13_monotone_in_box_endpoints():
    g = gen(131)
    n = 50_000
    dp, dv = sample_safe_states(n, 16.0, 1.4, 10.0, g)
    t = H.hocbf_terms(dp, dv)
    small = M.point_box(t.rho, t.V, t.m)
    r = lambda: 0.2 * torch.rand(n, generator=g, dtype=F64)  # noqa: E731
    big = M.StateBox(V_lo=torch.clamp(t.V - r(), min=0), V_hi=t.V + r(), rho_lo=t.rho - 0.5 * r(),
                     rho_hi=t.rho + r(), m_lo=t.m - r(), m_hi=t.m + r())
    rs, rb = M.one_tick_reachable_box(small), M.one_tick_reachable_box(big)
    Ls, Lb = M.one_tick_decrease_bound(rs), M.one_tick_decrease_bound(rb)
    Ps, Pb = M.inactivity_rate_bound(rs), M.inactivity_rate_bound(rb)
    assert bool((Lb >= Ls - 1e-9 * (1 + Ls.abs())).all())
    assert bool((Pb >= Ps - 1e-9 * (1 + Ps.abs())).all())
    assert bool((rb.rho_plus >= rs.rho_plus).all()) and bool((rb.m_minus <= rs.m_minus).all())


def test_lambda_monotone_in_estimation_bounds_and_inflation_sound():
    g = gen(132)
    n = 50_000
    dp, dv = sample_safe_states(n, 16.0, 1.5, 10.0, g)
    t = H.hocbf_terms(dp, dv)
    prev = None
    for ep in (0.0, 0.02, 0.05, 0.10):
        box = M.estimation_inflated_box(t.rho, t.V, t.m, ep, 2 * ep)
        Lam = M.one_tick_decrease_bound(M.one_tick_reachable_box(box))
        if prev is not None:
            both = torch.isfinite(prev)
            assert bool((Lam[both] >= prev[both] - 1e-9 * (1 + prev[both])).all())
        prev = Lam
    # Remark 2: the inflated box built from an estimate contains the truth, so its
    # Lambda upper-bounds the true-state Lambda.
    dp_hat = dp + rand_ball(n, 3, P.e_p, g)
    dv_hat = dv + rand_ball(n, 3, P.e_v, g)
    th = H.hocbf_terms(dp_hat, dv_hat)
    box = M.estimation_inflated_box(th.rho, th.V, th.m, P.e_p, P.e_v)
    assert bool(((box.rho_lo <= t.rho) & (t.rho <= box.rho_hi) & (box.V_lo <= t.V) & (t.V <= box.V_hi)
                 & (box.m_lo <= t.m) & (t.m <= box.m_hi)).all())
    L_hat = M.one_tick_decrease_bound(M.one_tick_reachable_box(box))
    L_true, _, _ = _lambda_point(dp, dv)
    fin = torch.isfinite(L_hat)
    assert bool((L_hat[fin] >= L_true[fin] - 1e-9 * (1 + L_true[fin])).all())


# ===========================================================================
# Lemma 14: inactivity certificates
# ===========================================================================
def test_lemma14_certificates_are_sound():
    g = gen(14)
    n = 40_000
    dp, dv = sample_safe_states(n, 16.0, 1.25, 12.0, g)
    t = H.hocbf_terms(dp, dv)
    reach = M.one_tick_reachable_box(M.point_box(t.rho, t.V, t.m))
    I1, I2 = M.inactivity_certificates(reach, t.psi1)
    cert = I1 | I2
    assert int(I1.sum()) > 100 and int((I2 & ~I1).sum()) > 100
    dpc, dvc = dp[cert], dv[cert]
    k = dpc.shape[0]
    du = (2 * torch.rand(k, 3, generator=g, dtype=F64) - 1) * 2 * P.u_inf
    worst0 = torch.full((k,), math.inf, dtype=F64)
    worst1 = torch.full((k,), math.inf, dtype=F64)
    for p_, v_ in simulate_tick(dpc, dvc, du, 40, g, adversarial=True):
        tt = H.hocbf_terms(p_, v_)
        worst0 = torch.minimum(worst0, tt.psi0)
        worst1 = torch.minimum(worst1, tt.psi1)
    assert float(worst0.min()) >= -1e-12 and float(worst1.min()) >= -1e-12


def test_lemma14_certificates_monotone_under_inflation():
    """A certificate obtained on the inflated box also holds at the true state."""
    g = gen(141)
    n = 100_000
    dp, dv = sample_safe_states(n, 16.0, 1.3, 15.0, g)
    dp_hat = dp + rand_ball(n, 3, P.e_p, g)
    dv_hat = dv + rand_ball(n, 3, P.e_v, g)
    S = M.reciprocal_shares(dp_hat.unsqueeze(1), dv_hat.unsqueeze(1), D)
    t = H.hocbf_terms(dp, dv)
    reach_t = M.one_tick_reachable_box(M.point_box(t.rho, t.V, t.m))
    I1t, I2t = M.inactivity_certificates(reach_t, t.psi1)
    assert bool(I1t[S.inactive_I1[:, 0]].all())
    assert bool(I2t[S.inactive_I2[:, 0]].all())


# ===========================================================================
# Lemma 15: communication radius
# ===========================================================================
def test_lemma15_communication_radius():
    R = M.communication_radius(D)
    assert R == pytest.approx(29.864, abs=1e-3)                 # paper: 29.864 m
    assert float(M.comm_radius_gamma(torch.tensor(R, dtype=F64))) >= 0.0
    assert M.communication_radius(D, P, V_max=12.0) == pytest.approx(17.9, abs=0.05)
    assert M.communication_radius(D, P, V_max=8.0) == pytest.approx(9.1, abs=0.05)
    Rg = torch.linspace(R - 10, R + 50, 20001, dtype=F64)
    gam = M.comm_radius_gamma(Rg)
    assert bool((torch.diff(gam) > 0).all())                    # strictly increasing


def test_lemma15_far_states_are_certified():
    g = gen(15)
    n = 100_000
    R = M.communication_radius(D)
    rho = R + 30 * torch.rand(n, 1, generator=g, dtype=F64) ** 2
    dp = rho * rand_unit(n, 3, g)
    dv = rand_ball(n, 3, 16.0, g)
    dv[: n // 4] = -16.0 * dp[: n // 4] / rho[: n // 4]          # worst case: head-on at V_max
    t = H.hocbf_terms(dp, dv)
    reach = M.one_tick_reachable_box(M.point_box(t.rho, t.V, t.m))
    I1, I2 = M.inactivity_certificates(reach, t.psi1)
    assert bool((t.psi0 > 0).all()) and bool((t.psi1 >= 0).all())
    assert bool((I1 | I2).all())
    assert bool(M.far_field_certified(t.rho).all())


def test_check_params_flags_link_radius_for_inflated_radii():
    assert P.R_link == 32.0 and M.check_params(P) == []           # default covers D = 2 r_max
    assert P.R_link >= M.communication_radius(P.D_max)
    issues = M.check_params(P.replace(R_link=30.0))                 # Table 4 value
    assert M.communication_radius(P.D_max) > 30.0
    assert len(issues) == 1 and "Lemma 15" in issues[0]


def test_fused_shares_match_reference_composition():
    g = gen(16)
    n = 30_000
    dp, dv = sample_safe_states(n, 16.0, 1.2, 35.0, g)
    dp, dv = dp.reshape(-1, 6, 3), dv.reshape(-1, 6, 3)
    Dv = 1.2 + 0.6 * torch.rand(dp.shape[:-1], generator=g, dtype=F64)
    heard = torch.rand(dp.shape[:-1], generator=g) < 0.9
    for ep, ev in ((0.0, 0.0), (P.e_p, P.e_v)):
        a = M.reciprocal_shares(dp, dv, Dv, heard=heard, e_p=ep, e_v=ev)
        b = M.reciprocal_shares_reference(dp, dv, Dv, heard=heard, e_p=ep, e_v=ev)
        for name in ("active", "ill_posed", "inactive_I1", "inactive_I2", "far", "admission"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
        for name in ("g", "zeta", "eps", "Lambda", "eps_x", "eps_psi", "beta", "psi1", "Psi"):
            x, y = getattr(a, name), getattr(b, name)
            assert torch.allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=False) or \
                bool((torch.isinf(x) == torch.isinf(y)).all() and torch.allclose(
                    x[torch.isfinite(x)], y[torch.isfinite(y)], rtol=1e-12, atol=1e-12)), name


# ===========================================================================
# Lemma 18 / Corollary 4: closed form for one active constraint
# ===========================================================================
def _kkt_residual(res: Q.BoxHalfspaceResult, u_hat, g, zeta, lo, hi):
    u, lam = res.u, res.lam
    stat = u - u_hat - lam[:, None] * g + res.lam_upper - res.lam_lower
    prim = torch.clamp(zeta - (g * u).sum(-1), min=0)
    comp = (lam * ((g * u).sum(-1) - zeta)).abs() + (res.lam_upper * (hi - u)).abs().sum(-1) \
        + (res.lam_lower * (u - lo)).abs().sum(-1)
    box = torch.clamp(u - hi, min=0).sum(-1) + torch.clamp(lo - u, min=0).sum(-1)
    return stat.abs().amax(-1), prim, comp, box


@pytest.mark.parametrize("d", [1, 2, 3, 5, 8])
def test_lemma18_kkt_certificate(d):
    g_ = gen(180 + d)
    n = 20_000
    lo = -1.0 - 4 * torch.rand(n, d, generator=g_, dtype=F64)
    hi = 1.0 + 4 * torch.rand(n, d, generator=g_, dtype=F64)
    u_hat = 8 * torch.randn(n, d, generator=g_, dtype=F64)       # also outside the box
    g = torch.randn(n, d, generator=g_, dtype=F64)
    g[torch.rand(n, d, generator=g_) < 0.15] = 0.0               # exercise zero coordinates
    sup = torch.where(g > 0, g * hi, g * lo).sum(-1)
    zeta = sup * (2 * torch.rand(n, generator=g_, dtype=F64) - 0.8) - 0.1
    res = Q.project_box_halfspace(u_hat, g, zeta, lo, hi)
    feas_true = zeta <= sup
    assert torch.equal(res.feasible, feas_true)
    f = res.feasible
    stat, prim, comp, box = _kkt_residual(res, u_hat, g, zeta, lo, hi)
    scale = 1 + u_hat.abs().amax(-1) + zeta.abs()
    assert float((stat[f] / scale[f]).max()) < 1e-12
    assert float((prim[f] / scale[f]).max()) < 1e-12
    assert float((comp[f] / scale[f] ** 2).max()) < 1e-11
    assert float(box[f].max()) == 0.0
    assert bool((res.lam[f] >= 0).all())
    assert torch.allclose(res.u[~f], torch.where(g > 0, hi, torch.where(g < 0, lo,
                          torch.minimum(torch.maximum(u_hat, lo), hi)))[~f])


def test_lemma18_matches_brute_force_qp():
    g_ = gen(18)
    n, d, u_inf = 600, 3, P.u_inf
    u_hat = (2 * torch.rand(n, d, generator=g_, dtype=F64) - 1) * u_inf
    g = 2 * (1.2 + 5 * torch.rand(n, 1, generator=g_, dtype=F64)) * rand_unit(n, d, g_)
    zeta = u_inf * g.abs().sum(-1) * (1.4 * torch.rand(n, generator=g_, dtype=F64) - 0.3)
    res = Q.project_box_halfspace(u_hat, g, zeta, -u_inf, u_inf)
    assert torch.equal(res.feasible, Q.single_constraint_feasible(g, zeta, u_inf))   # Cor. 4
    eye = np.eye(d)
    for i in range(n):
        G = np.vstack([-g[i].numpy()[None], eye, -eye])
        h = np.concatenate([[-zeta[i].item()], np.full(2 * d, u_inf)])
        x = brute_force_qp(u_hat[i].numpy(), G, h)
        assert (x is not None) == bool(res.feasible[i])
        if x is not None:
            assert np.allclose(res.u[i].numpy(), x, atol=1e-9)


# ===========================================================================
# Hildreth's method and the Stage-B dispatcher
# ===========================================================================
def _random_feasible_stage_b(n: int, N: int, g_: torch.Generator, u_inf: float = P.u_inf):
    d = 3
    u_hat = (2 * torch.rand(n, d, generator=g_, dtype=F64) - 1) * u_inf
    u0 = (2 * torch.rand(n, d, generator=g_, dtype=F64) - 1) * 0.9 * u_inf       # interior point
    g = 2 * (1.2 + 4 * torch.rand(n, N, 1, generator=g_, dtype=F64)) \
        * rand_unit(n * N, d, g_).reshape(n, N, d)
    slack = 0.5 * torch.rand(n, N, generator=g_, dtype=F64)
    zeta = (g * u0[:, None]).sum(-1) - slack
    return u_hat, g, zeta


@pytest.mark.parametrize("N", [2, 3, 4])
def test_hildreth_matches_brute_force(N):
    g_ = gen(200 + N)
    n, u_inf = 150, P.u_inf
    u_hat, g, zeta = _random_feasible_stage_b(n, N, g_)
    active = torch.ones(n, N, dtype=torch.bool)
    # production path: default cap of 50 sweeps with the exact active-set polish
    res = Q.solve_stage_b_qp(u_hat, g, zeta, active, u_inf)
    assert bool(res.ok.all()) and bool((res.method == 2).all()) and res.sweeps <= 50
    # pure Hildreth (Algorithm 2 verbatim) with a generous cap; its linear rate can
    # be very slow for nearly antiparallel rows, so only converged elements are
    # compared and nearly all must converge.
    G, h = Q.stage_b_rows(g, zeta, u_inf)
    pure = Q.hildreth(u_hat, G, h, max_sweeps=20_000, polish=False)
    assert float(pure.converged.double().mean()) >= 0.95
    for i in range(n):
        x = brute_force_qp(u_hat[i].numpy(), G[i].numpy(), h[i].numpy())
        assert x is not None
        assert np.allclose(res.u[i].numpy(), x, atol=1e-8), (i, res.u[i], x)
        if bool(pure.converged[i]):
            assert np.allclose(pure.x[i].numpy(), x, atol=1e-6), (i, pure.x[i], x)


def test_hildreth_weighted_metric_matches_brute_force():
    g_ = gen(21)
    n_prob, n, m = 60, 5, 8
    for i in range(n_prob):
        y = 3 * torch.randn(n, generator=g_, dtype=F64)
        G = torch.randn(m, n, generator=g_, dtype=F64)
        x0 = 0.3 * torch.randn(n, generator=g_, dtype=F64)
        h = G @ x0 + 0.5 * torch.rand(m, generator=g_, dtype=F64)
        if i % 2 == 0:
            w = 0.2 + 3 * torch.rand(n, generator=g_, dtype=F64)
            Winv_t, Winv_np = 1.0 / w, np.diag(1.0 / w.numpy())
        else:
            A = torch.randn(n, n, generator=g_, dtype=F64)
            W = A @ A.T + n * torch.eye(n, dtype=F64)
            Winv_t = torch.linalg.inv(W)
            Winv_np = Winv_t.numpy()
        r = Q.hildreth(y, G, h, W_inv=Winv_t, max_sweeps=20000)
        x = brute_force_qp(y.numpy(), G.numpy(), h.numpy(), Winv_np)
        assert bool(r.feasible) and np.allclose(r.x.numpy(), x, atol=1e-7)


def test_hildreth_detects_infeasibility():
    g_ = gen(22)
    n = 200
    gdir = rand_unit(n, 3, g_)
    g = torch.stack([4 * gdir, -4 * gdir], dim=1)                  # g.u >= z1 and -g.u >= z2
    z = 1.0 + torch.rand(n, 2, generator=g_, dtype=F64)            # z1 + z2 > 0: empty
    res = Q.solve_stage_b_qp(torch.zeros(n, 3, dtype=F64), g, z, torch.ones(n, 2, dtype=torch.bool),
                             P.u_inf, max_sweeps=200)
    assert not bool(res.ok.any())


def test_hildreth_warm_start_reduces_sweeps():
    g_ = gen(23)
    u_hat, g, zeta = _random_feasible_stage_b(300, 4, g_)
    act = torch.ones(300, 4, dtype=torch.bool)
    cold = Q.solve_stage_b_qp(u_hat, g, zeta, act, P.u_inf, max_sweeps=5000)
    warm = Q.solve_stage_b_qp(u_hat + 1e-3, g, zeta, act, P.u_inf, lam0=cold.lam, max_sweeps=5000)
    assert bool(warm.ok.all()) and warm.sweeps <= cold.sweeps


def test_stage_b_routing_and_unbatched_call():
    g_ = gen(24)
    u_hat, g, zeta = _random_feasible_stage_b(400, 3, g_)
    act = torch.rand(400, 3, generator=g_) < 0.5
    res = Q.solve_stage_b_qp(u_hat, g, zeta, act, P.u_inf, max_sweeps=5000)
    assert torch.equal(res.n_active, act.sum(-1))
    assert torch.equal(res.method, torch.clamp(act.sum(-1), max=2).to(torch.int8))
    none = act.sum(-1) == 0
    assert torch.equal(res.u[none], torch.clamp(u_hat[none], -P.u_inf, P.u_inf))
    assert bool(res.ok.all())
    single = Q.solve_stage_b_qp(u_hat[7], g[7], zeta[7], act[7], P.u_inf, max_sweeps=5000)
    assert torch.allclose(single.u, res.u[7]) and bool(single.ok) == bool(res.ok[7])
    # an ill-posed (infinite) active share is reported as not ok
    zeta_bad = zeta.clone()
    zeta_bad[0, 0] = math.inf
    act_bad = act.clone()
    act_bad[0, 0] = True
    assert not bool(Q.solve_stage_b_qp(u_hat, g, zeta_bad, act_bad, P.u_inf).ok[0])


# ===========================================================================
# Theorem 3 (a), (c)
# ===========================================================================
def test_theorem3a_parallel_projection_is_sound():
    """Exact information: both agents use the same data, project independently,
    and the joint input satisfies 2 dp.(u_i - u_j) >= beta + eps."""
    g_ = gen(31)
    n = 20_000
    dp, dv = sample_closing_states(n, g_)
    Si = M.reciprocal_shares(dp[:, None], dv[:, None], D, e_p=0.0, e_v=0.0)
    Sj = M.reciprocal_shares(-dp[:, None], -dv[:, None], D, e_p=0.0, e_v=0.0)
    assert torch.equal(Si.active, Sj.active)
    assert torch.allclose(Si.zeta, Sj.zeta, rtol=1e-12, atol=1e-12)
    uh = (2 * torch.rand(2, n, 3, generator=g_, dtype=F64) - 1) * P.u_inf
    ri = Q.solve_stage_b_qp(uh[0], Si.g, Si.zeta, Si.active, P.u_inf)
    rj = Q.solve_stage_b_qp(uh[1], Sj.g, Sj.zeta, Sj.active, P.u_inf)
    both = Si.active[:, 0] & ri.ok & rj.ok
    assert int(both.sum()) > 1000
    lhs = 2 * (dp * (ri.u - rj.u)).sum(-1)
    rhs = Si.beta[:, 0] + Si.eps[:, 0]
    assert bool((lhs[both] >= rhs[both] - 1e-9 * (1 + rhs[both].abs())).all())


def test_theorem3c_admissible_fraction():
    g_ = gen(32)
    c0, u_inf, d = 3.5, P.u_inf, 3
    bound = 0.5 - math.sqrt(d) * c0 / (2 * u_inf)
    assert bound == pytest.approx(0.1211, abs=1e-4)
    U = (2 * torch.rand(400_000, d, generator=g_, dtype=F64) - 1) * u_inf
    for n_ in rand_unit(64, d, g_):
        frac = float(((U @ n_) >= c0).double().mean())
        assert frac >= bound - 3e-3


# ===========================================================================
# Lemmas 13-14 / Corollary 4: well-posedness of Stage B (Table 9)
# ===========================================================================
def well_posedness(V_range: float, n: int = 40_000, seed: int = 41) -> Dict[str, float]:
    g_ = gen(seed)
    dp, dv = sample_safe_states(n, V_range, D, 10.0, g_)
    out = {}
    for tag, (ep, ev) in {"exact": (0.0, 0.0), "est": (P.e_p, P.e_v)}.items():
        dph = dp + rand_ball(n, 3, ep, g_) if ep > 0 else dp
        dvh = dv + rand_ball(n, 3, ev, g_) if ev > 0 else dv
        S = M.reciprocal_shares(dph[:, None], dvh[:, None], D, e_p=ep, e_v=ev)
        act = S.active[:, 0]
        infeas = act & (~torch.isfinite(S.zeta[:, 0])
                        | ~Q.single_constraint_feasible(S.g[:, 0], S.zeta[:, 0], P.u_inf))
        rho = torch.linalg.vector_norm(dp, dim=-1)
        out[f"{tag}_ill_posed_pct"] = 100.0 * float(infeas.double().mean())
        out[f"{tag}_max_dist_from_D"] = float((rho[infeas] - D).max()) if bool(infeas.any()) else 0.0
        out[f"{tag}_active_pct"] = 100.0 * float(act.double().mean())
    return out


@pytest.mark.parametrize("V_range", [16.0, 8.0, 4.0])
def test_stage_b_well_posedness_thin_layer(V_range):
    m = well_posedness(V_range)
    assert m["exact_ill_posed_pct"] < 0.2 and m["est_ill_posed_pct"] < 1.0
    assert m["exact_max_dist_from_D"] < 0.25 and m["est_max_dist_from_D"] < 0.25


# ===========================================================================
# Theorem 1: forward invariance under Stage-B execution (Monte Carlo)
# ===========================================================================
def run_encounters(p_i, p_j, v_i, v_j, T: float, estimated: bool, seed: int,
                   v_nom: float = 5.0, n_sub: int = 10, cfg: SafetyParams = P) -> Dict[str, float]:
    """Two disturbed double integrators per encounter, both applying Stage B.

    Goals are antipodal (each agent heads to the other's start). The
    disturbance pushes the agents toward each other with magnitude ``e_a``.
    ``psi_0`` and ``psi_1`` are evaluated at every substep on the true state.
    """
    g_ = gen(seed)
    n = p_i.shape[0]
    goal_i, goal_j = p_j.clone(), p_i.clone()
    p = torch.cat([p_i, p_j])
    v = torch.cat([v_i, v_j])
    goal = torch.cat([goal_i, goal_j])
    ep, ev = (cfg.e_p, cfg.e_v) if estimated else (0.0, 0.0)
    h = cfg.T_c / n_sub
    stats = dict(min_psi0=math.inf, min_psi1=math.inf, min_dist=math.inf, active_ticks=0,
                 infeasible_ticks=0, max_speed=0.0, infeasible_max_gap=0.0)
    enc_psi0 = torch.full((n,), math.inf, dtype=F64)
    enc_psi1 = torch.full((n,), math.inf, dtype=F64)
    enc_dist = torch.full((n,), math.inf, dtype=F64)
    enc_infeasible = torch.zeros(n, dtype=torch.bool)
    for _ in range(int(round(T / cfg.T_c))):
        dp = p[:n] - p[n:]
        dv = v[:n] - v[n:]
        dp_hat = torch.cat([dp, -dp])
        dv_hat = torch.cat([dv, -dv])
        if estimated:
            dp_hat = dp_hat + rand_ball(2 * n, 3, ep, g_)
            dv_hat = dv_hat + rand_ball(2 * n, 3, ev, g_)
        S = M.reciprocal_shares(dp_hat[:, None], dv_hat[:, None], D, cfg, e_p=ep, e_v=ev)
        to_goal = goal - p
        dist = torch.linalg.vector_norm(to_goal, dim=-1, keepdim=True)
        v_des = to_goal * torch.clamp(v_nom / torch.clamp(dist, min=1e-9), max=2.0)  # min(v_nom, 2 dist)
        u_hat = torch.clamp((v_des - v) / 0.25, -cfg.u_inf, cfg.u_inf)
        r = Q.solve_stage_b_qp(u_hat, S.g, S.zeta, S.active, cfg.u_inf)
        stats["active_ticks"] += int(S.active.sum())
        bad = ~r.ok
        if bool(bad.any()):
            stats["infeasible_ticks"] += int(bad.sum())
            rho_now = torch.linalg.vector_norm(dp, dim=-1)
            bad_enc = bad[:n] | bad[n:]
            enc_infeasible |= bad_enc
            stats["infeasible_max_gap"] = max(stats["infeasible_max_gap"],
                                              float((rho_now[bad_enc] - D).max()))
        u = r.u   # best effort when infeasible (the executor would brake instead)
        for _s in range(n_sub):
            dpt = p[:n] - p[n:]
            nij = dpt / torch.linalg.vector_norm(dpt, dim=-1, keepdim=True)
            dist_ = torch.cat([-cfg.e_a * nij, cfg.e_a * nij])     # push toward each other
            a = u + dist_
            p = p + v * h + 0.5 * a * h * h
            v = v + a * h
            t = H.hocbf_terms(p[:n] - p[n:], v[:n] - v[n:], D, cfg)
            enc_psi0 = torch.minimum(enc_psi0, t.psi0)
            enc_psi1 = torch.minimum(enc_psi1, t.psi1)
            enc_dist = torch.minimum(enc_dist, t.rho)
        stats["max_speed"] = max(stats["max_speed"], float(torch.linalg.vector_norm(v, dim=-1).max()))
    ok = ~enc_infeasible   # Assumption 3 held throughout the encounter
    stats["n_encounters"] = n
    stats["n_feasible_throughout"] = int(ok.sum())
    if bool(ok.any()):
        stats["min_psi0"] = float(enc_psi0[ok].min())
        stats["min_psi1"] = float(enc_psi1[ok].min())
        stats["min_dist"] = float(enc_dist[ok].min())
    stats["min_dist_all"] = float(enc_dist.min())
    return stats


def head_on_initial(n: int, seed: int, v0: float = 5.0):
    g_ = gen(seed)
    lat = (2 * torch.rand(n, 2, generator=g_, dtype=F64) - 1) * torch.tensor([1.0, 0.5], dtype=F64)
    p_i = torch.zeros(n, 3, dtype=F64)
    p_i[:, 0] = -15.0
    p_j = torch.zeros(n, 3, dtype=F64)
    p_j[:, 0] = 15.0
    p_j[:, 1:] = lat
    v_i = torch.zeros(n, 3, dtype=F64)
    v_i[:, 0] = v0
    v_j = -v_i
    return p_i, p_j, v_i, v_j


def boundary_initial(n: int, seed: int, frac: float = 0.98, ep: float = 0.0, ev: float = 0.0):
    """Starts close to the braking boundary: closing speed ``frac * cbar(rho)``."""
    g_ = gen(seed)
    rho = 1.6 + 6.0 * torch.rand(n, generator=g_, dtype=F64)
    nrm = rand_unit(n, 3, g_)
    cbar = H.closing_speed_limit(rho)
    tang = torch.linalg.cross(nrm, rand_unit(n, 3, g_))
    tang = tang / torch.linalg.vector_norm(tang, dim=-1, keepdim=True)
    dv = -frac * cbar[:, None] * nrm + (2 * torch.rand(n, 1, generator=g_, dtype=F64)) * tang
    dp = rho[:, None] * nrm
    t = H.hocbf_terms(dp, dv)
    keep = t.in_safe_set
    dp, dv = dp[keep], dv[keep]
    c = torch.zeros_like(dp)
    return c + dp / 2, c - dp / 2, dv / 2, -dv / 2, H.hocbf_terms(dp, dv).psi1.min()


@pytest.mark.parametrize("estimated", [False, True])
def test_theorem1_head_on_encounters(estimated):
    s = run_encounters(*head_on_initial(300, 51 + estimated), T=5.0, estimated=estimated, seed=7)
    assert s["active_ticks"] > 1000
    assert s["infeasible_ticks"] == 0                            # Assumption 3 held throughout
    assert s["min_psi0"] >= 0.0 and s["min_psi1"] >= 0.0
    assert s["min_dist"] >= D


@pytest.mark.parametrize("estimated", [False, True])
def test_theorem1_boundary_start_encounters(estimated):
    """Theorem 1 is conditional on Assumption 3 (feasibility at every tick).

    Encounters whose Stage B stayed feasible must keep psi_0, psi_1 >= 0.
    Infeasible ticks may occur only in the thin layer near D that Section 8
    describes: with exact information a pair can graze the barrier with large
    tangential speed, the one-tick box of Lemma 13 then leaves the domain of
    alpha_1 (Lambda = inf), and the executor must engage the fallback.
    """
    p_i, p_j, v_i, v_j, psi1_0 = boundary_initial(400, 61 + estimated)
    s = run_encounters(p_i, p_j, v_i, v_j, T=3.0, estimated=estimated, seed=8)
    assert float(psi1_0) < 0.5                                   # starts near the boundary
    assert s["n_feasible_throughout"] >= 0.9 * s["n_encounters"]
    assert s["min_psi0"] >= 0.0 and s["min_psi1"] >= 0.0
    assert s["min_dist"] >= D
    assert s["infeasible_ticks"] <= 0.005 * s["active_ticks"] + 1
    assert s["infeasible_max_gap"] <= 0.16                       # thin layer (paper: 0.152 m)


# ===========================================================================
# Table-9 style report
# ===========================================================================
def _report() -> None:  # pragma: no cover - manual run
    rows = []
    m8 = lemma8_measurements()
    rows += [("Lemma 8", "FD rel. err alpha1'", f"{m8['fd_rel_err_alpha1p']:.1e}", "1.2e-07"),
             ("Lemma 8", "FD rel. err alpha1''", f"{m8['fd_rel_err_alpha1pp']:.1e}", "3.4e-07"),
             ("Lemma 8", "closed vs direct alpha1''", f"{m8['closed_vs_direct_rel_err']:.1e}", "9.2e-16"),
             ("Lemma 8", "min 3/4X^4 - sX^3 + k^2", f"{m8['numerator_min']:.2f} at b={m8['numerator_argmin_b']:.1g}",
              "34.50 at b=0"),
             ("Lemma 8", "alpha1'(0), alpha1''(0)", f"{m8['alpha1p_at_0']:.2f}, {m8['alpha1pp_at_0']:.2f}",
              "10, -79.86")]
    m11 = lemma11_measurements()
    rows += [("Lemma 11", "lambda_min (rho*)", f"{m11['lambda_min_grid']:.3f} ({m11['rho_star_grid']:.2f} m)",
              "1.261 (2.75 m)"),
             ("Lemma 11", "max beta/(2 rho)", f"{m11['max_beta_over_2rho']:.2f}", "4.44"),
             ("Lemma 11", "max[beta/2rho - a cbar/(cbar+s)]", f"{m11['max_gap']:.2f}", "-0.14")]
    rho = torch.linspace(1.5, 3.0, 301, dtype=F64)[:, None].expand(301, 801)
    V = torch.linspace(0.0, 8.0, 801, dtype=F64)[None, :].expand(301, 801)
    r = M.estimation_margin(rho, V, D).eps_x / (4 * rho)
    rows += [("Lemma 12", "eps^x/(4 rho) range", f"{float(r.min()):.3f} - {float(r.max()):.3f}", "0.817 - 3.814")]
    rows += [("Lemma 15", "R_com (D=1.2)", f"{M.communication_radius(D):.3f} m", "29.864 m"),
             ("Lemma 15", "R_com (V_max=12 / 8)",
              f"{M.communication_radius(D, P, 12.0):.1f} / {M.communication_radius(D, P, 8.0):.1f} m", "17.9 / 9.1 m"),
             ("Lemma 15", "R_com (D=2 r_max=3.0)", f"{M.communication_radius(P.D_max):.3f} m", "(not reported)")]
    for Vr in (16.0, 8.0, 4.0):
        w = well_posedness(Vr)
        rows.append(("Lem. 13-14", f"ill-posed, |dv|<={Vr:g}",
                     f"{w['exact_ill_posed_pct']:.3f}% / {w['est_ill_posed_pct']:.3f}% "
                     f"(<= {max(w['exact_max_dist_from_D'], w['est_max_dist_from_D']):.3f} m)",
                     {16.0: "0.033% / 0.122%", 8.0: "0.015% / 0.130%", 4.0: "0.013% / 0.205%"}[Vr]))
    for est in (False, True):
        t0 = time.time()
        s = run_encounters(*head_on_initial(300, 51 + est), T=5.0, estimated=est, seed=7)
        rows.append(("Theorem 1", f"head-on, {'estimated' if est else 'exact'}",
                     f"{s['active_ticks']} active, {s['infeasible_ticks']} infeasible; min psi0={s['min_psi0']:.3f} "
                     f"psi1={s['min_psi1']:.3f} d_min={s['min_dist']:.3f} ({time.time()-t0:.1f}s)",
                     "none infeasible; d_min 1.396 / 1.321"))
        p_i, p_j, v_i, v_j, _ = boundary_initial(400, 61 + est)
        s = run_encounters(p_i, p_j, v_i, v_j, T=3.0, estimated=est, seed=8)
        rows.append(("Theorem 1", f"boundary start, {'estimated' if est else 'exact'}",
                     f"{s['active_ticks']} active, {s['infeasible_ticks']} infeasible "
                     f"(<= {s['infeasible_max_gap']:.3f} m from D, "
                     f"{s['n_encounters'] - s['n_feasible_throughout']} enc.); min psi0={s['min_psi0']:.3f} "
                     f"psi1={s['min_psi1']:.3f} d_min={s['min_dist']:.3f} (all: {s['min_dist_all']:.4f})",
                     "none infeasible; d_min 1.389 / 1.331"))
    # latency of one Stage-B call (one agent, d = 3)
    g_ = gen(99)
    u_hat, g, zeta = _random_feasible_stage_b(1, 3, g_)
    for N, act in ((1, torch.tensor([[True, False, False]])), (3, torch.ones(1, 3, dtype=torch.bool))):
        for _ in range(20):
            Q.solve_stage_b_qp(u_hat, g, zeta, act, P.u_inf)
        t0 = time.perf_counter()
        for _ in range(200):
            Q.solve_stage_b_qp(u_hat, g, zeta, act, P.u_inf)
        rows.append(("Stage B", f"latency, {N} active (torch, CPU)",
                     f"{(time.perf_counter() - t0) / 200 * 1e3:.3f} ms", "<= 1 ms (H1)"))
    w = max(len(r[0]) for r in rows), max(len(r[1]) for r in rows), max(len(r[2]) for r in rows)
    print(f"{'Statement':<{w[0]}}  {'Check':<{w[1]}}  {'Measured':<{w[2]}}  Paper")
    for a, b, c, d in rows:
        print(f"{a:<{w[0]}}  {b:<{w[1]}}  {c:<{w[2]}}  {d}")
    print("\ncheck_params(DEFAULT_PARAMS):", M.check_params(P))


if __name__ == "__main__":  # pragma: no cover
    _report()

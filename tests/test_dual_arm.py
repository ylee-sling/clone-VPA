# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Task T2 tests: Panda kinematics, collision-sphere kinematics (Remark 1), twist-space Stage B with
``d_a = 7``, the ManiSkill3-style assembly environment, its expert, the role-order modes and the runner.
"""

from __future__ import annotations

import math

import pytest
import torch

from cma_vpa.comm import (ARM_UNITS, TABLE1_UNITS, SafetyPacket, decode_sphere_broadcast, encode_sphere_broadcast,
                          sphere_agent_id)
from cma_vpa.envs import (AGENT_UIDS, AssemblyConfig, AssemblyModeMaster, AssemblyRunner, DualArmAssemblyEnv,
                          DualArmSim, ExpertAssemblyPolicy, LearnedAssemblyPolicy, PandaKinematics,
                          matrix_to_rotvec, order_label, pd_ee_delta_pose_to_twist, role_labels, rotvec_to_matrix,
                          twist_to_pd_ee_delta_pose)
from cma_vpa.envs.dual_arm_sim import PANDA_Q_READY
from cma_vpa.models import CloneConfig, ClonePolicy
from cma_vpa.safety import (PANDA_SPHERES, ArmExecutor, ArmExecutorConfig, ArmLimits, SphereTable,
                            arm_safety_params, arm_shares, centrifugal_drift, check_params, drift_bound,
                            lambda_min, solve_twist_qp, sphere_acceleration, sphere_jacobian, sphere_kinematics,
                            twist_box, twist_rows)

DT = torch.float64
R_DOWN = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=DT)


# ---------------------------------------------------------------------------
# Panda kinematics
# ---------------------------------------------------------------------------
def test_panda_fk_ready_pose():
    kin = PandaKinematics(torch.zeros(1, 3), torch.zeros(1))
    q = torch.tensor(PANDA_Q_READY, dtype=DT).view(1, 7)
    p, R = kin.fk(q)
    assert torch.allclose(p, torch.tensor([[0.30689, 0.0, 0.48688]], dtype=DT), atol=2e-5)   # Franka ready pose
    assert torch.allclose(R, torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=DT)).unsqueeze(0), atol=1e-9)
    flange = kin.frames(q)[0, 7, :3, 3]
    assert torch.allclose(flange, torch.tensor([0.30689, 0.0, 0.59028], dtype=DT), atol=2e-5)


def test_panda_jacobian_matches_finite_differences():
    kin = PandaKinematics(torch.tensor([[0.0, -0.55, 0.0], [0.0, 0.55, 0.0]]), torch.tensor([math.pi / 2, -math.pi / 2]))
    g = torch.Generator().manual_seed(0)
    q = kin.q_ready + 0.3 * torch.randn(3, 2, 7, generator=g, dtype=DT)
    J = kin.jacobian(q)
    p0, R0 = kin.fk(q)
    eps = 1e-6
    for i in range(7):
        dq = torch.zeros_like(q)
        dq[..., i] = eps
        p1, R1 = kin.fk(q + dq)
        assert torch.allclose(J[..., :3, i], (p1 - p0) / eps, atol=1e-5)
        assert torch.allclose(J[..., 3:, i], matrix_to_rotvec(R1 @ R0.transpose(-1, -2)) / eps, atol=1e-5)


def test_ik_reaches_every_task_pose():
    env = DualArmAssemblyEnv(AssemblyConfig(seed=0))
    poses = [env.part_pos, env.part_pos + torch.tensor([0, 0, 0.10], dtype=DT),
             env.slot_position(torch.zeros(1, 2, dtype=torch.long)), env.slot_position(torch.ones(1, 2, dtype=torch.long)),
             env.staging().unsqueeze(0), env.home().unsqueeze(0)]
    for target in poses:
        q, err = env.sim.kin.ik(env.sim.kin.q_ready.expand(1, 2, 7).clone(), target, R_DOWN.expand(1, 2, 3, 3))
        assert float(err.max()) < 1e-6
        assert bool(((q >= env.sim.kin.q_min) & (q <= env.sim.kin.q_max)).all())


def test_so3_helpers_roundtrip():
    g = torch.Generator().manual_seed(1)
    rv = torch.randn(2000, 3, generator=g, dtype=DT)
    rv = rv / rv.norm(dim=-1, keepdim=True) * torch.rand(2000, 1, generator=g, dtype=DT) * math.pi * 0.9999
    assert torch.allclose(matrix_to_rotvec(rotvec_to_matrix(rv)), rv, atol=1e-8)
    R = rotvec_to_matrix(rv)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=DT).expand_as(R), atol=1e-12)


# ---------------------------------------------------------------------------
# Remark 1: collision-sphere kinematics and the drift bound
# ---------------------------------------------------------------------------
def _rigid_states(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, generator=g, dtype=DT) * 0.3
    R = rotvec_to_matrix(torch.randn(n, 3, generator=g, dtype=DT))
    v = torch.randn(n, 3, generator=g, dtype=DT) * 0.4
    w = torch.randn(n, 3, generator=g, dtype=DT) * 1.5
    u = torch.randn(n, 6, generator=g, dtype=DT) * 5.0
    return p, R, v, w, u


def test_sphere_velocity_is_linear_in_the_twist():
    p, R, v, w, _ = _rigid_states(64, 2)
    offs = PANDA_SPHERES.offset_tensor(DT)
    st = sphere_kinematics(p, R, v, w, offs)
    # c_dot = v + w x (c - p) = J_c (v, w)
    assert torch.allclose(st.lever, st.center - p.unsqueeze(1), atol=1e-14)
    J = sphere_jacobian(st.lever)                                                # (n, S, 3, 6)
    xi = torch.cat([v, w], -1)
    assert torch.allclose(st.velocity, (J @ xi.view(-1, 1, 6, 1)).squeeze(-1), atol=1e-12)
    # finite differences of the rigid motion (constant twist)
    h = 1e-6
    p2 = p + h * v
    R2 = rotvec_to_matrix(h * w) @ R
    st2 = sphere_kinematics(p2, R2, v, w, offs)
    assert torch.allclose((st2.center - st.center) / h, st.velocity, atol=1e-5)


def test_sphere_acceleration_drift_and_bound():
    p, R, v, w, u = _rigid_states(64, 3)
    offs = PANDA_SPHERES.offset_tensor(DT)
    st = sphere_kinematics(p, R, v, w, offs)
    # c_ddot by central differences of the exact motion with constant twist rate u
    h = 1e-4

    def at(t):
        vt = v + t * u[:, :3]
        wt = w + t * u[:, 3:]
        pt = p + t * v + 0.5 * t * t * u[:, :3]
        # orientation with linearly varying angular velocity, accurate to O(t^3) via the midpoint rule
        Rt = rotvec_to_matrix((w + 0.5 * t * u[:, 3:]) * t) @ R
        return sphere_kinematics(pt, Rt, vt, wt, offs).center

    acc_fd = (at(h) - 2 * st.center + at(-h)) / (h * h)
    acc = sphere_acceleration(st.lever, w.unsqueeze(1).expand_as(st.lever), u.unsqueeze(1).expand(-1, offs.shape[0], -1))
    assert torch.allclose(acc_fd, acc, atol=2e-3, rtol=1e-3)
    # ||w x (w x r)|| <= ||w||^2 ||r||, with equality for w orthogonal to r
    drift = centrifugal_drift(w.unsqueeze(1).expand_as(st.lever), st.lever)
    bound = drift_bound(w.unsqueeze(1).expand_as(st.lever), st.lever)
    assert bool((torch.linalg.vector_norm(drift, dim=-1) <= bound + 1e-12).all())
    r = torch.tensor([[0.0, 0.0, 0.18]], dtype=DT)
    wo = torch.tensor([[1.3, -0.7, 0.0]], dtype=DT)
    assert torch.allclose(torch.linalg.vector_norm(centrifugal_drift(wo, r), dim=-1), drift_bound(wo, r))


def test_arm_safety_params_fold_drift_into_e_a():
    L = ArmLimits()
    cfg = arm_safety_params(L, PANDA_SPHERES)
    lever = PANDA_SPHERES.lever_max
    assert math.isclose(cfg.e_a, L.e_track + L.w_max ** 2 * lever)
    assert math.isclose(cfg.u_inf, L.a_lin + math.sqrt(2) * L.a_ang * lever)
    assert math.isclose(cfg.v_bar, L.v_max + L.w_max * lever)
    assert check_params(cfg) == []                                    # Lemmas 8, 11, 15 and Corollary 6
    for D in (cfg.D_nominal, cfg.D_max):
        assert cfg.lambda_b >= float(lambda_min(D, cfg)[0])
    # u_inf bounds every axis of v_dot + w_dot x r over the twist-rate box (Assumption 1 for the spheres)
    g = torch.Generator().manual_seed(4)
    a = L.accel_bound(DT)
    u = (2 * torch.rand(20000, 6, generator=g, dtype=DT) - 1) * a
    rr = torch.randn(20000, 3, generator=g, dtype=DT)
    rr = rr / rr.norm(dim=-1, keepdim=True) * lever
    sph = u[:, :3] + torch.cross(u[:, 3:], rr, dim=-1)
    assert float(sph.abs().max()) <= cfg.u_inf + 1e-12


def test_twist_rows_match_the_sphere_jacobian():
    g = torch.Generator().manual_seed(5)
    gvec = torch.randn(10, 7, 3, generator=g, dtype=DT)
    r = torch.randn(10, 7, 3, generator=g, dtype=DT) * 0.2
    u = torch.randn(10, 6, generator=g, dtype=DT)
    rows = twist_rows(gvec, r)
    lhs = (rows * u.unsqueeze(1)).sum(-1)
    rhs = (gvec * (sphere_jacobian(r) @ u.view(10, 1, 6, 1)).squeeze(-1)).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-12)


# ---------------------------------------------------------------------------
# Stage B in twist space (d_a = 7: twist + gripper)
# ---------------------------------------------------------------------------
def _ee_pair_problem(seed: int, closing: float = 0.25):
    g = torch.Generator().manual_seed(seed)
    ex = ArmExecutor(2, PANDA_SPHERES.n, ArmExecutorConfig())
    p = torch.tensor([[-0.15, -0.14, 0.25], [-0.15, 0.14, 0.25]], dtype=DT) + 0.03 * torch.randn(2, 3, generator=g, dtype=DT)
    R = torch.stack([rotvec_to_matrix(0.3 * torch.randn(3, generator=g, dtype=DT)) @ R_DOWN for _ in range(2)])
    v = closing * torch.randn(2, 3, generator=g, dtype=DT)
    w = 0.5 * torch.randn(2, 3, generator=g, dtype=DT)
    own = ex.spheres(p, R, v, w)
    other = torch.tensor([1, 0])
    sh = arm_shares(own, ex.radius, own.center[other], own.velocity[other], ex.radius.expand(2, -1),
                    torch.ones(2, PANDA_SPHERES.n, dtype=torch.bool), ex.cfg)
    xi = torch.cat([v, w], -1)
    lo, hi = twist_box(xi, ex.limits, ex.cfg.T_c)
    u_hat = torch.randn(2, 6, generator=g, dtype=DT) * ex.limits.accel_bound(DT)
    return ex, sh, xi, lo, hi, u_hat


def test_twist_qp_matches_reference_solver():
    scipy_opt = pytest.importorskip("scipy.optimize")
    import numpy as np

    a = ArmLimits().accel_bound(DT)
    checked = 0
    for seed in range(400):
        ex, sh, xi, lo, hi, u_hat = _ee_pair_problem(seed)
        S = sh.shares
        if not ((S.psi1 >= 0).all() and (S.psi0 >= 0).all()) or int(S.active.sum()) < 4:
            continue
        res = solve_twist_qp(u_hat, sh.rows, S.zeta, S.active, lo, hi, a)
        for b in range(2):
            act = S.active[b]
            y0 = (u_hat[b] / a).clamp(lo[b] / a, hi[b] / a).numpy()
            ry = (sh.rows[b] * a)[act].numpy()
            z = S.zeta[b][act].numpy()
            ref = scipy_opt.minimize(lambda y: 0.5 * np.sum((y - y0) ** 2), y0, jac=lambda y: y - y0, method="SLSQP",
                                     bounds=list(zip((lo[b] / a).numpy(), (hi[b] / a).numpy())),
                                     constraints=[{"type": "ineq", "fun": lambda y: ry @ y - z, "jac": lambda y: ry}],
                                     options={"ftol": 1e-14, "maxiter": 500})
            feasible = ref.success and float(np.min(ry @ ref.x - z)) > -1e-7
            assert bool(res.ok[b]) == feasible
            if feasible:
                assert bool(res.optimal[b])
                assert np.abs((res.u[b] / a).numpy() - ref.x).max() < 1e-6
                checked += 1
        if checked >= 40:
            break
    assert checked >= 40


def test_single_share_feasibility_in_the_safe_set():
    """Twist-space Stage B is feasible for sphere-pair states of C_HO outside a thin layer near D.

    Ill-posed shares (the conservative one-tick reachable box leaves the domain of
    alpha_1) occur only within 1 cm of D, and every active, well-posed share more
    than 2 cm beyond D is feasible with the Lemma 18 closed form, as for T1
    (Table 9). Inside the layer the executor brakes, as Algorithm 2 prescribes.
    """
    cfg = arm_safety_params()
    L = ArmLimits()
    g = torch.Generator().manual_seed(6)
    n = 100000
    D = 0.10
    rho = D + 0.4 * torch.rand(n, generator=g, dtype=DT)
    nh = torch.randn(n, 3, generator=g, dtype=DT)
    nh = nh / nh.norm(dim=-1, keepdim=True)
    dv = torch.randn(n, 3, generator=g, dtype=DT)
    dv = dv / dv.norm(dim=-1, keepdim=True) * 1.0 * torch.rand(n, 1, generator=g, dtype=DT)
    from cma_vpa.safety import reciprocal_shares
    S = reciprocal_shares((rho.unsqueeze(-1) * nh).unsqueeze(1), dv.unsqueeze(1), torch.full((n, 1), D, dtype=DT), cfg)
    inC = (S.psi1[:, 0] >= 0) & (rho >= D)
    act = S.active[:, 0] & inC
    ill = S.ill_posed[:, 0] & inC
    assert float((rho[ill] - D).max()) < 0.01                         # thin layer only
    keep = act & ~ill & (rho - D > 0.02)
    lever = torch.randn(n, 3, generator=g, dtype=DT)
    lever = lever / lever.norm(dim=-1, keepdim=True) * PANDA_SPHERES.lever_max * torch.rand(n, 1, generator=g, dtype=DT)
    rows = twist_rows(S.g[:, 0], lever)[keep].unsqueeze(1)
    zeta = S.zeta[:, 0][keep].unsqueeze(1)
    m = rows.shape[0]
    xi = torch.zeros(m, 6, dtype=DT)
    lo, hi = twist_box(xi, L, cfg.T_c)
    u_hat = torch.zeros(m, 6, dtype=DT)
    res = solve_twist_qp(u_hat, rows, zeta, torch.ones(m, 1, dtype=torch.bool), lo, hi, L.accel_bound(DT))
    assert m > 5000 and bool(res.ok.all()) and bool((res.method == 1).all())     # Lemma 18 closed form
    slack = (rows[:, 0] * res.u).sum(-1) - zeta[:, 0]
    assert float(slack.min()) >= 0


def _head_on(stage_b: bool, T: float = 2.5, e_track: float = 0.3, spin: float = 0.0):
    g = torch.Generator().manual_seed(7)
    sim = DualArmSim(num_envs=1, e_track=e_track, generator=g)
    p0 = torch.tensor([[[-0.15, -0.28, 0.25], [-0.15, 0.28, 0.25]]], dtype=DT)
    sim.reset(p0, R_DOWN.expand(1, 2, 3, 3))
    ex = ArmExecutor(2, PANDA_SPHERES.n, ArmExecutorConfig())
    H = 32
    chunk = torch.zeros(2, H, 6, dtype=DT)
    chunk[0, :, 1], chunk[1, :, 1] = 0.4, -0.4
    chunk[0, :, 5], chunk[1, :, 5] = spin, -spin
    grip = torch.tensor([[0.7] * H, [-0.3] * H], dtype=DT)
    D = ex.radius.view(-1, 1) + ex.radius.view(1, -1)
    phys = PANDA_SPHERES.radius_tensor(DT)
    out = {"margin": math.inf, "clearance": math.inf, "ok": True, "braking": 0, "grip_ok": True}
    for n in range(int(round(T / 0.01))):
        s = n * 0.01
        if n % 3 == 0:
            ex.publish_chunk(torch.ones(2, dtype=torch.bool), chunk, grip, torch.full((2,), s, dtype=DT),
                             torch.full((2,), H), s)
        bc = ex.broadcast(s, sim.p[0], sim.R[0], sim.v[0], sim.w[0])
        o = torch.tensor([1, 0])
        nbr = SphereTable(pos=bc.pos[o], vel=bc.vel[o], radius=bc.radius[o],
                          flags=bc.flags[o].unsqueeze(-1).expand(2, PANDA_SPHERES.n),
                          stamp=bc.stamp[o].unsqueeze(-1).expand(2, PANDA_SPHERES.n),
                          present=torch.ones(2, PANDA_SPHERES.n, dtype=torch.bool))
        cmd = ex.tick(s, sim.p[0], sim.R[0], sim.v[0], sim.w[0], nbr)
        twist = cmd.twist if stage_b else torch.cat([sim.v[0], sim.w[0]], -1) + 0.01 * cmd.u_hat
        out["grip_ok"] &= bool(torch.allclose(cmd.gripper, grip[:, 0]))          # gripper passes through
        sim.step(twist.unsqueeze(0), cmd.gripper.unsqueeze(0))
        c = sim.sphere_state().center[0]
        dist = torch.cdist(c[0], c[1])
        out["margin"] = min(out["margin"], float((dist - D).min()))
        out["clearance"] = min(out["clearance"], float((dist - (phys.view(-1, 1) + phys.view(1, -1))).min()))
        out["ok"] &= bool(cmd.stage_b_ok.all())
        out["braking"] += int(cmd.braking.sum())
    return out


def test_closed_loop_stage_b_keeps_sphere_pairs_apart():
    on = _head_on(True)
    assert on["ok"] and on["braking"] == 0 and on["grip_ok"]
    assert on["margin"] >= 0.0                       # every sphere pair at least D_kl apart (Theorem 1)
    spin = _head_on(True, spin=1.0)                  # rotating hands: drift term w x (w x r) active
    assert spin["ok"] and spin["margin"] >= 0.0
    off = _head_on(False)
    assert off["clearance"] < 0.0                    # without Stage B the hands collide


def test_clone_with_seven_dimensional_chunks():
    torch.manual_seed(0)
    cfg = CloneConfig.tiny(in_channels=8, action_dim=7, attenuate_channels=6, K=2)
    assert CloneConfig.arm().in_channels == 8 and CloneConfig.arm().action_dim == 7
    clone = ClonePolicy(cfg).eval()
    g = torch.Generator().manual_seed(1)
    C, H, W = cfg.in_channels, *cfg.image_size
    clone.reset_mission(torch.randn(1, cfg.text_dim, generator=g), milestone_frames=torch.rand(2, 1, C, H, W, generator=g))
    d = clone.decide(torch.rand(2, C, H, W, generator=g), torch.randn(2, 13, generator=g), torch.randn(2, 3, generator=g), 0.0)
    assert d.chunk.shape == (2, cfg.horizon, 7) and len(d.chain) == cfg.K + 3
    assert torch.allclose(d.chunk[..., :6], d.raw_chunk[..., :6] * d.alpha.view(-1, 1, 1))   # twist attenuated
    assert torch.equal(d.chunk[..., 6], d.raw_chunk[..., 6])                                # gripper passes


# ---------------------------------------------------------------------------
# sphere packets
# ---------------------------------------------------------------------------
def test_sphere_packets_millimetre_units():
    ex = ArmExecutor(2, PANDA_SPHERES.n, ArmExecutorConfig())
    p = torch.tensor([[0.1234, -0.4321, 0.2], [0.05, 0.3, 0.15]], dtype=DT)
    R = R_DOWN.expand(2, 3, 3)
    v = torch.tensor([[0.1, 0.2, -0.3], [0.0, -0.25, 0.0]], dtype=DT)
    w = torch.tensor([[0.0, 0.0, 0.5], [0.2, 0.0, 0.0]], dtype=DT)
    bc = ex.broadcast(3.21, p, R, v, w)
    raw = encode_sphere_broadcast(bc)
    assert len(raw) == 2 * PANDA_SPHERES.n and all(len(r) == 22 for r in raw)
    dec = decode_sphere_broadcast(raw, 3.22)
    for arm in range(2):
        for k in range(PANDA_SPHERES.n):
            pos, vel, rad, fl, st = dec[arm][k]
            assert max(abs(a - b) for a, b in zip(pos, bc.pos[arm, k].tolist())) <= 0.0005 + 1e-12
            assert max(abs(a - b) for a, b in zip(vel, bc.vel[arm, k].tolist())) <= 0.0005 + 1e-12
            assert rad >= float(bc.radius[arm, k]) - 1e-12 and abs(st - 3.21) < 1e-6
    assert sphere_agent_id(3, 4) == 28
    pk = SafetyPacket.build(1, 0, 0.0, (1.234, 0, 0), (0, 0, 0), 0.6)
    assert pk.pos_cm == (123, 0, 0) and pk.position(TABLE1_UNITS) == pk.pos      # Table 1 unchanged
    assert SafetyPacket.build(1, 0, 0.0, (1.234, 0, 0), (0, 0, 0), 0.06, units=ARM_UNITS).pos_cm == (1234, 0, 0)


# ---------------------------------------------------------------------------
# the environment (ManiSkill3-style API)
# ---------------------------------------------------------------------------
def test_env_api_spaces_and_observations():
    env = DualArmAssemblyEnv(AssemblyConfig(num_envs=2, obs_mode="state_dict", seed=0))
    assert tuple(env.action_space.spaces.keys()) == AGENT_UIDS and env.single_action_space.shape == (7,)
    obs, info = env.reset(seed=1)
    assert set(obs["agent"]) == set(AGENT_UIDS) and obs["agent"]["panda-0"]["qpos"].shape == (2, 7)
    assert obs["agent"]["panda-1"]["tcp_quat"].shape == (2, 4) and obs["extra"]["part_pos"].shape == (2, 2, 3)
    act = {u: torch.zeros(2, 7) for u in AGENT_UIDS}
    obs, rew, term, trunc, info = env.step(act)
    assert rew.shape == (2,) and term.shape == (2,) and not bool(term.any()) and not bool(info["collision"].any())
    flat = DualArmAssemblyEnv(AssemblyConfig(num_envs=2, obs_mode="state", seed=0)).get_obs()
    assert flat.dim() == 2 and flat.shape[0] == 2
    rgbd = DualArmAssemblyEnv(AssemblyConfig(obs_mode="rgbd", image_size=(32, 40), occlusion="severe", seed=0))
    o = rgbd.get_obs()
    assert o["sensor_data"]["panda-0"]["rgbd"].shape == (1, 8, 32, 40)
    assert rgbd.occluders.shape == (1, 5, 3) and rgbd.render().shape == (1, 32, 40, 3)
    # action normalization round trip (base frame <-> world frame)
    a = (torch.rand(2, 2, 7) * 2 - 1).double()
    tw = env.to_world_twist(a)
    assert torch.allclose(env.to_action(tw, a[..., 6])[..., :6], a[..., :6], atol=1e-12)
    # ManiSkill3 pd_ee_delta_pose conversion round trip
    twist = torch.tensor([0.3, -0.2, 0.1, 0.5, 0.0, -0.4], dtype=DT)
    ms = twist_to_pd_ee_delta_pose(twist, torch.tensor(1.0, dtype=DT), dt=0.01)
    assert ms.shape == (7,) and float(ms[6]) == -1.0
    back, gr = pd_ee_delta_pose_to_twist(ms, dt=0.01)
    assert torch.allclose(back, twist) and float(gr) == 1.0


def test_mode_labels_role_order():
    assert order_label(torch.tensor([0, 1])).tolist() == [0, 1]
    assert role_labels(torch.tensor([0, 1])).tolist() == [[0, 1], [1, 0]]


@pytest.fixture(scope="module")
def batched_master_run():
    env = DualArmAssemblyEnv(AssemblyConfig(num_envs=3, seed=11))
    env.reset(seed=11, options={"instructions": ["red_first", "blue_first", "any"]})
    g = torch.Generator().manual_seed(1)
    pol = ExpertAssemblyPolicy(env, generator=g)
    master = AssemblyModeMaster(env, g)
    return env, master, AssemblyRunner(env, pol, master).run()


def test_expert_assembles_in_the_instructed_order(batched_master_run):
    env, master, r = batched_master_run
    assert r.success.tolist() == [True, True, True]
    assert r.first_placed[0] == 0 and r.first_placed[1] == 1                     # red first, blue first
    assert int(r.first_placed[2]) == int(master.u_g[2])                          # "any": the sampled order
    assert not bool(r.collision.any()) and bool((r.min_clearance > 0).all())
    assert bool((r.assembly_time < 40.0).all()) and r.infeasible_ticks_pct == 0.0
    assert float(env.sim.ik_err.max()) < 1e-3


@pytest.fixture(scope="module")
def conflict_runs():
    env = DualArmAssemblyEnv(AssemblyConfig(num_envs=3, instruction="red_first", seed=4))
    roles = torch.tensor([[1, 0], [1, 1], [0, 0]])              # wrong order, both followers, both leaders
    pol = ExpertAssemblyPolicy(env, mode_source="fixed", roles=roles)
    with_b = AssemblyRunner(env, pol, None).run(2000)
    env2 = DualArmAssemblyEnv(AssemblyConfig(num_envs=1, instruction="any", seed=4))
    pol2 = ExpertAssemblyPolicy(env2, mode_source="fixed", roles=torch.tensor([[0, 0]]))
    without_b = AssemblyRunner(env2, pol2, None, stage_b=False).run(1200)
    return with_b, without_b


def test_incompatible_roles_fail_safely(conflict_runs):
    with_b, without_b = conflict_runs
    # roles contrary to the instruction: both parts placed, assembly predicate false
    assert int(with_b.n_placed[0]) == 2 and int(with_b.first_placed[0]) == 1 and not bool(with_b.success[0])
    # two followers wait for each other (deadlock), two leaders meet at the fixture (standoff)
    assert int(with_b.n_placed[1]) == 0 and int(with_b.n_placed[2]) == 0
    assert not bool(with_b.collision.any())                                     # the barrier keeps them apart
    assert without_b.collision.tolist() == [True]                               # without Stage B they collide


def test_learned_clone_drives_the_runner():
    env = DualArmAssemblyEnv(AssemblyConfig(instruction="any", image_size=(32, 40), seed=2))
    torch.manual_seed(0)
    clone = ClonePolicy(CloneConfig.tiny(in_channels=8, action_dim=7, attenuate_channels=6)).eval()
    pol = LearnedAssemblyPolicy(env, clone, generator=torch.Generator().manual_seed(0))
    master = AssemblyModeMaster(env, torch.Generator().manual_seed(0))
    r = AssemblyRunner(env, pol, master).run(120)
    assert r.ticks == 120 and pol.n_decisions >= 36
    assert clone.u_M.tolist() == role_labels(master.u_g)[0].tolist()            # roles from the downlink
    assert not bool(r.collision.any())

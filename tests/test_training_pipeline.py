# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Phase 4 tests: losses and the five training stages (tiny configuration, CPU).

For every stage: five optimizer steps without NaN, the modules that the stage
must not train are bit-identical before and after and receive no gradient, the
checkpoint reloads into deterministic outputs, and no barrier function is
evaluated inside any training loss.
"""

from __future__ import annotations

import copy
import math
import sys

import pytest
import torch

import cma_vpa.safety.degg_flow as degg_flow
import cma_vpa.safety.margins as margins
import cma_vpa.safety.qp_solvers as qp_solvers
from cma_vpa.models import CloneConfig, ClonePolicy
from cma_vpa.training import (cfm_loss, jepa_loss, sigreg, stage1_repr, stage2_bus, stage3_master, stage4_policy,
                              stage5_dagger, vicreg_covariance, vicreg_variance)
from cma_vpa.training.common import build_clone, load_checkpoint, pick_device, seed_all, team_batch
from cma_vpa.training.data import Sampler, collect_episodes
from cma_vpa.training.stage4_policy import MAX_CLONE_PARAMETERS, PolicyTrainer, load_stage3

STEPS = 5
COMMON = ["--tiny", "--steps", str(STEPS), "--teams", "2", "--device", "cpu", "--log-every", "0"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def episodes():
    eps = collect_episodes(2, n_agents=(3,), wind=("W0", "W1"), H=8, T_max=3.0, f_dec=10.0, seed=0)
    assert len(eps) == 2 and all(e.T >= 20 for e in eps)
    return eps


@pytest.fixture(scope="module")
def ckdir(tmp_path_factory):
    return tmp_path_factory.mktemp("ckpt")


@pytest.fixture(scope="module")
def no_barrier():
    """Make every barrier entry point raise, wherever it was imported; used around training losses."""
    names = [(margins, "reciprocal_shares"), (margins, "reciprocal_shares_reference"),
             (qp_solvers, "solve_stage_b_qp"), (qp_solvers, "stage_b_fixed"), (qp_solvers, "hildreth"),
             (qp_solvers, "project_box_halfspace"), (degg_flow, "guided_solve"),
             (degg_flow, "build_stage_a_polytope")]
    originals = {id(getattr(m, n)) for m, n in names}

    class Guard:
        def __enter__(self):
            self.saved = []

            def boom(*a, **k):
                raise AssertionError("a barrier function was evaluated inside a training loss")

            for modname, mod in list(sys.modules.items()):
                if not modname.startswith("cma_vpa") or mod is None:
                    continue
                for attr, val in list(vars(mod).items()):
                    if callable(val) and id(val) in originals:
                        self.saved.append((mod, attr, val))
                        setattr(mod, attr, boom)
            assert len(self.saved) >= len(names)
            return self

        def __exit__(self, *exc):
            for m, n, f in self.saved:
                setattr(m, n, f)
            return False

    return Guard


def _state(mod):
    return {k: v.detach().clone() for k, v in mod.state_dict().items()}


def _same(a, b) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def _changed(a, b) -> bool:
    return any(not torch.equal(a[k], b[k]) for k in a if a[k].is_floating_point())


def _finite(history, keys=("loss",)) -> bool:
    return len(history) == STEPS and all(math.isfinite(h[k]) for h in history for k in keys)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def test_vicreg_terms_eq8():
    torch.manual_seed(0)
    Z = torch.randn(64, 8) * torch.linspace(0.2, 2.0, 8)
    v_ref = sum(max(0.0, 1.0 - math.sqrt(float(Z[:, j].var()) + 1e-4)) for j in range(8)) / 8
    assert math.isclose(float(vicreg_variance(Z)), v_ref, rel_tol=1e-5)
    C = torch.cov(Z.T)
    c_ref = float((C - torch.diag(torch.diag(C))).pow(2).sum()) / 8
    assert math.isclose(float(vicreg_covariance(Z)), c_ref, rel_tol=1e-5)
    with pytest.raises(ValueError):
        vicreg_variance(Z, gamma_v=1.0, eps_v=2.0)
    with pytest.raises(ValueError):
        vicreg_variance(Z[:1])


def test_sigreg_detects_collapse():
    g = torch.Generator().manual_seed(1)
    gauss = torch.randn(4096, 16, generator=g)
    collapsed = torch.zeros(4096, 16)
    s_g = float(sigreg(gauss, generator=torch.Generator().manual_seed(2)))
    s_c = float(sigreg(collapsed, generator=torch.Generator().manual_seed(2)))
    assert s_g < 0.01 < s_c


def test_jepa_loss_stop_gradient_and_bootstrap():
    torch.manual_seed(3)
    heads = torch.randn(3, 16, 8, requires_grad=True)
    target = torch.randn(16, 8, requires_grad=True)
    Z = torch.randn(16, 8, requires_grad=True)
    w = torch.ones(3, 16)
    loss, terms = jepa_loss(heads, target, Z, head_weights=w)
    loss.backward()
    assert target.grad is None and heads.grad is not None and Z.grad is not None
    l2, _ = jepa_loss(heads.detach(), target.detach(), Z.detach())
    assert torch.allclose(loss.detach(), l2)                            # unit weights = plain mean


def test_cfm_loss_zero_for_exact_field():
    class Exact(torch.nn.Module):     # knows xi through a closure: v = A1 - xi exactly
        def forward(self, a_tau, tau, cond, kv):
            return self.A - self.xi

    A = torch.randn(4, 8, 4)
    g = torch.Generator().manual_seed(5)
    xi = torch.randn(A.shape, generator=torch.Generator().manual_seed(5))
    f = Exact()
    f.A, f.xi = A, xi
    assert float(cfm_loss(f, A, torch.zeros(4, 1), None, generator=g)) < 1e-10


# ---------------------------------------------------------------------------
# the five stages
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stage_runs(episodes, ckdir, no_barrier):
    runs = {}
    with no_barrier():
        runs[1] = stage1_repr.run(stage1_repr.build_parser().parse_args(
            COMMON + ["--out", str(ckdir / "s1.pt"), "--stats-teams", "4"]), episodes)
        runs[2] = stage2_bus.run(stage2_bus.build_parser().parse_args(
            COMMON + ["--init", str(ckdir / "s1.pt"), "--out", str(ckdir / "s2.pt"), "--init-teams", "4"]), episodes)
        runs[3] = stage3_master.run(stage3_master.build_parser().parse_args(
            COMMON + ["--init", str(ckdir / "s2.pt"), "--out", str(ckdir / "s3.pt"), "--T-M", "0.5"]), episodes)
        runs[4] = stage4_policy.run(stage4_policy.build_parser().parse_args(
            COMMON + ["--init", str(ckdir / "s3.pt"), "--out", str(ckdir / "s4.pt")]), episodes)
    runs[5] = stage5_dagger.run(stage5_dagger.build_parser().parse_args(
        COMMON + ["--init", str(ckdir / "s4.pt"), "--out", str(ckdir / "s5.pt"), "--iters", "1", "--rollouts", "1",
                  "--rollout-t-max", "1.0", "--agents", "3", "--f-dec", "10"]), episodes)
    return runs


def test_stage1_representation(stage_runs, ckdir):
    r = stage_runs[1]
    assert _finite(r["history"], ("loss", "pred", "var", "cov"))
    c = r["clone"]
    assert r["gamma_star"] > 0 and math.isclose(float(c.gamma_star), r["gamma_star"], rel_tol=1e-6)
    assert math.isclose(float(c.predictor.gamma_star), r["gamma_star"], rel_tol=1e-6)
    assert not any(p.requires_grad for m in (c.encoder, c.predictor, c.prim_emb) for p in m.parameters())
    # modules outside the stage are untouched (identical to a fresh initialization with the same seed)
    seed_all(0)
    fresh = ClonePolicy(CloneConfig.tiny())
    for name in ("selector", "field", "plan_decoder", "bus", "text_proj", "mode_emb"):
        assert _same(_state(getattr(fresh, name)), _state(getattr(c, name))), name
    assert _changed(_state(fresh.encoder), _state(c.encoder)) and _changed(_state(fresh.predictor), _state(c.predictor))
    ck = load_checkpoint(str(ckdir / "s1.pt"))
    assert ck["stage"] == 1 and "ema_encoder" in ck and ck["calib"]["gamma_star"] == r["gamma_star"]


def test_stage2_bus(stage_runs, ckdir):
    r = stage_runs[2]
    assert _finite(r["history"], ("loss", "rec", "codebook", "commit"))
    s1 = load_checkpoint(str(ckdir / "s1.pt"))["clone"]
    s2 = load_checkpoint(str(ckdir / "s2.pt"))["clone"]
    for k in s1:
        if k.startswith(("encoder.", "predictor.", "prim_emb.", "selector.", "field.", "text_proj.")):
            assert torch.equal(s1[k], s2[k]), k
    assert not torch.equal(s1["bus.proj.weight"], s2["bus.proj.weight"])
    assert not torch.equal(s1["rvq.codebook_free"], s2["rvq.codebook_free"])
    Q = r["clone"].rvq.codebooks
    assert torch.equal(Q[:, 0], torch.zeros_like(Q[:, 0]))              # Lemma 4 zero codeword kept
    assert math.isfinite(r["snr_db"])


def test_stage3_master(stage_runs, ckdir):
    r = stage_runs[3]
    assert _finite(r["history"], ("loss", "plan", "ce_g", "ce_l", "vc"))
    s2 = load_checkpoint(str(ckdir / "s2.pt"))["clone"]
    s3 = load_checkpoint(str(ckdir / "s3.pt"))
    for k in s2:
        if not k.startswith("text_proj."):
            assert torch.equal(s2[k], s3["clone"][k]), k
    assert not torch.equal(s2["text_proj.proj.weight"], s3["clone"]["text_proj.proj.weight"])
    assert not any(p.requires_grad for p in r["master"].parameters())
    assert float(r["master"].gamma_star) == float(r["clone"].gamma_star)


def test_stage4_policy_and_frozen_gradients(stage_runs, ckdir, episodes, no_barrier):
    r = stage_runs[4]
    assert _finite(r["history"], ("loss", "cfm", "ce"))
    s3 = load_checkpoint(str(ckdir / "s3.pt"))
    s4 = load_checkpoint(str(ckdir / "s4.pt"))
    trained = ("selector.", "mode_emb.", "field.", "plan_decoder.")
    for k in s3["clone"]:
        if not k.startswith(trained):
            assert torch.equal(s3["clone"][k], s4["clone"][k]), k
    assert any(not torch.equal(s3["clone"][k], s4["clone"][k]) for k in s3["clone"] if k.startswith("field."))
    for k in s3["master"]:
        assert torch.equal(s3["master"][k], s4["master"][k])
    assert ClonePolicy().parameter_budget()["total"] < MAX_CLONE_PARAMETERS
    # one more step: frozen modules receive no gradient at all
    dev = pick_device("cpu")
    _, cfg, clone, master = load_stage3(str(ckdir / "s4.pt"), dev)
    args = stage4_policy.build_parser().parse_args(COMMON)
    tr = PolicyTrainer(clone, master, args, dev, torch.Generator().manual_seed(0))
    with no_barrier():
        tr.step(team_batch(Sampler(episodes).team(2), cfg, dev))
    trained_ids = {id(p) for p in tr.params}
    for name, p in list(clone.named_parameters()) + list(master.named_parameters()):
        if id(p) not in trained_ids:
            assert p.grad is None and not p.requires_grad, name
    assert any(p.grad is not None for p in tr.params)


def test_stage5_dagger(stage_runs, ckdir):
    r = stage_runs[5]
    assert len(r["log"]) == 1 and r["log"][0]["episodes"] >= 1
    assert all(math.isfinite(h["loss"]) for h in r["history"])
    s4 = load_checkpoint(str(ckdir / "s4.pt"))
    s5 = load_checkpoint(str(ckdir / "s5.pt"))
    assert s5["stage"] == 5
    for k in s4["clone"]:
        if k.startswith(("encoder.", "predictor.", "bus.", "rvq.", "text_proj.", "prim_emb.")):
            assert torch.equal(s4["clone"][k], s5["clone"][k]), k


def test_checkpoint_reload_is_deterministic(stage_runs, ckdir, episodes):
    ck = load_checkpoint(str(ckdir / "s4.pt"))
    cfg = CloneConfig(**ck["config"])
    outs = []
    for _ in range(2):
        c = build_clone(cfg, ck["clone"]).eval()
        g = torch.Generator().manual_seed(9)
        C, H, W = cfg.in_channels, *cfg.image_size
        c.reset_mission(torch.randn(1, cfg.text_dim, generator=g),
                        milestone_frames=torch.rand(3, 1, C, H, W, generator=g))
        d = c.decide(torch.rand(3, C, H, W, generator=g), torch.randn(3, 13, generator=g),
                     torch.randn(3, 3, generator=g), 0.0, generator=torch.Generator().manual_seed(4))
        outs.append((d.chunk, d.primitive, d.sigma))
    a, b = outs
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    # the in-memory model of the stage gives the same output as the reloaded one
    live = copy.deepcopy(stage_runs[4]["clone"]).eval()
    g = torch.Generator().manual_seed(9)
    C, H, W = cfg.in_channels, *cfg.image_size
    live.reset_mission(torch.randn(1, cfg.text_dim, generator=g), milestone_frames=torch.rand(3, 1, C, H, W, generator=g))
    d = live.decide(torch.rand(3, C, H, W, generator=g), torch.randn(3, 13, generator=g),
                    torch.randn(3, 3, generator=g), 0.0, generator=torch.Generator().manual_seed(4))
    assert torch.equal(d.chunk, a[0])


def test_device_selection():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in ("cpu", "cuda", "mps")

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Phase 3 tests: latent bus (RVQ, Table 1 codecs), bridge, master and the clone decision chain.

Covers Lemma 4 (RVQ residuals), Proposition 4 (payload sizes, clone consistency),
Proposition 5 (Gaussian floor), Proposition 8 (sequential depth K + 3, bridge
K/V off the chain), Eq. 11-12 (mode sampling, plan spread), Eq. 16 (plan
tokens), noise persistence (Section 4.5) and the parameter budget of Table 3.
"""

from __future__ import annotations

import math
import struct

import pytest
import torch

from cma_vpa.comm import (DOWNLINK_SIZE, SAFETY_SIZE, UPLINK_SIZE, DownlinkPacket, PacketCRCError,
                          PacketFieldError, PacketLengthError, ResidualVQ, SafetyPacket, SequenceFilter,
                          UplinkPacket, crc16_ccitt, decode_safety_broadcast, encode_safety_broadcast,
                          gaussian_rd_floor, timestamp_us, unwrap_timestamp)
from cma_vpa.models import (TABLE3_BUDGET, CloneConfig, ClonePolicy, MasterConfig, MasterPlanner,
                            NoisePersistence, handedness_of_mode)
from cma_vpa.safety import (ChunkMap, Executor, ExecutorConfig, build_stage_a_polytope,
                            constant_velocity_reference, neighbor_tables_from_broadcasts)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tiny_clone(seed: int = 0, B: int = 3, M: int = 2, **kw) -> ClonePolicy:
    torch.manual_seed(seed)
    cfg = CloneConfig.tiny(**kw)
    c = ClonePolicy(cfg).eval()
    g = torch.Generator().manual_seed(seed + 1)
    C, H, W = cfg.in_channels, *cfg.image_size
    c.reset_mission(torch.randn(1, cfg.text_dim, generator=g),
                    milestone_frames=torch.randn(B, M, C, H, W, generator=g), agent_ids=list(range(10, 10 + B)))
    return c


def tiny_master(cfg: CloneConfig, seed: int = 0) -> MasterPlanner:
    torch.manual_seed(seed)
    return MasterPlanner(MasterConfig(bus_dim=cfg.bus_dim, latent_dim=cfg.latent_dim, text_dim=cfg.text_proj_dim,
                                      width=32, heads=4, depth=2, mode_emb_dim=8, plan_hidden=32,
                                      n_local_modes=cfg.num_modes)).eval()


def frames(c: ClonePolicy, B: int, g: torch.Generator) -> torch.Tensor:
    return torch.randn(B, c.cfg.in_channels, *c.cfg.image_size, generator=g)


def master_round(c: ClonePolicy, m: MasterPlanner, now: float, validity: float = 1.5, gen=None):
    ids = c.agent_ids
    dec = m.decode_uplinks(c.uplinks(now), ids, c.rvq)
    out = m.cycle(c.rvq, dec["codes"][None], dec["sigma"][None], dec["present"][None], c.c_tilde[:1],
                  c.tracker.active(c.milestones)[None], generator=gen)
    return out, m.downlinks(out, ids, now, c.tracker.m, validity)


# ---------------------------------------------------------------------------
# CRC and packet codecs (Table 1)
# ---------------------------------------------------------------------------
def test_crc16_ccitt_check_value():
    assert crc16_ccitt(b"123456789") == 0x29B1
    assert crc16_ccitt(b"") == 0xFFFF


def _examples():
    up = UplinkPacket.build(3, 65535, 12.345678, list(range(24)), 0.37, 0x05)
    dn = DownlinkPacket.build(250, 1234, 4300.0, 11, 7, [255] * 24, 0.3, 1.5, status=0xA)
    sf = SafetyPacket.build(9, 300, 1.0, (1.234, -20.005, 3.0), (-7.99, 0.004, 15.0), 0.6, 0x02)
    return up, dn, sf


def test_exact_sizes_and_layout():
    assert (UPLINK_SIZE, DOWNLINK_SIZE, SAFETY_SIZE) == (35, 36, 22)
    up, dn, sf = _examples()
    assert [len(p.pack()) for p in (up, dn, sf)] == [35, 36, 22]
    # Proposition 4(a): downlink + uplink at most 2 B_max = 80 bytes per cycle per agent
    assert len(up.pack()) <= 40 and len(dn.pack()) <= 40
    raw = up.pack()
    assert raw[0] == 3 and struct.unpack("<H", raw[1:3])[0] == 65535 and raw[7:31] == bytes(range(24))
    assert struct.unpack("<H", raw[-2:])[0] == crc16_ccitt(raw[:-2])


def test_roundtrip_all_messages():
    up, dn, sf = _examples()
    for p in (up, dn, sf):
        assert type(p).unpack(p.pack()) == p
    u = UplinkPacket.unpack(up.pack())
    assert u.sigma_bar >= 0.37 and u.sigma_bar - 0.37 < 1 / 64          # rounded up
    d = DownlinkPacket.unpack(dn.pack())
    assert (d.mode, d.status, d.milestone, d.plan_valid) == (11, 0xA, 7, True)
    assert d.validity_s == 1.5 and d.sigma_M >= 0.3 and d.sigma_M - 0.3 < 0.125
    s = SafetyPacket.unpack(sf.pack())
    assert s.pos == (1.23, -20.01, 3.0) or s.pos == (1.23, -20.0, 3.0)
    assert s.radius_cm == 60 and s.seq == 300 % 256 and s.flags == 0x02
    assert abs(s.vel[0] + 7.99) < 1e-9


@pytest.mark.parametrize("which", [0, 1, 2])
def test_crc_detects_every_single_bit_error(which):
    pkt = _examples()[which]
    raw = bytearray(pkt.pack())
    for i in range(len(raw)):
        for bit in range(8):
            bad = bytearray(raw)
            bad[i] ^= 1 << bit
            with pytest.raises(PacketCRCError):
                type(pkt).unpack(bytes(bad))


def test_length_and_field_validation():
    up, dn, sf = _examples()
    for p in (up, dn, sf):
        raw = p.pack()
        for bad in (raw[:-1], raw + b"\x00", b""):
            with pytest.raises(PacketLengthError):
                type(p).unpack(bad)
    with pytest.raises(PacketFieldError):
        DownlinkPacket(1, 1, 1, 16, 0, bytes(24), 0, 1).pack()             # mode needs 4 bits
    with pytest.raises(PacketFieldError):
        UplinkPacket(1, 1, 1, bytes(23), 0, 0).pack()                      # 24-byte code
    with pytest.raises(PacketFieldError):
        UplinkPacket(256, 1, 1, bytes(24), 0, 0).pack()


def test_quantization_semantics():
    # radius is never under-reported; position/velocity saturate at int16
    s = SafetyPacket.build(0, 0, 0.0, (500.0, -500.0, 0.0), (0, 0, 0), 0.6001, 0)
    assert s.pos_cm == (32767, -32767, 0) and s.radius_cm == 61
    assert SafetyPacket.build(0, 0, 0.0, (0, 0, 0), (0, 0, 0), 9.0).radius_cm == 255
    assert UplinkPacket.build(0, 0, 0.0, [0] * 24, float("nan")).uncertainty == 255
    assert UplinkPacket.build(0, 0, 0.0, [0] * 24, -1.0).uncertainty == 0
    assert DownlinkPacket.build(0, 0, 0.0, 0, 0, [0] * 24, 0.0, 0.0).plan_valid is False
    assert DownlinkPacket.build(0, 0, 0.0, 0, 0, [0] * 24, 0.0, 100.0).validity_code == 15


def test_timestamp_wrap_and_sequence_filter():
    period = 2 ** 32 * 1e-6
    for t in (0.0, 1.5, period - 0.001, period + 0.25, 3 * period + 10.0):
        ts = timestamp_us(t)
        assert abs(unwrap_timestamp(ts, t + 0.02) - t) < 1e-6        # received 20 ms later
        assert abs(unwrap_timestamp(ts, t - 0.02) - t) < 1e-6        # clock skew the other way
    f = SequenceFilter(16)
    assert f.accept(1, 65534) and f.accept(1, 65535) and f.accept(1, 0)      # wraps
    assert not f.accept(1, 0) and not f.accept(1, 65535)                     # duplicate / older
    assert f.accept(2, 5)                                                    # per sender
    f8 = SequenceFilter(8)
    assert f8.accept(0, 255) and f8.accept(0, 3) and not f8.accept(0, 200)


def test_safety_broadcast_wiring_to_executor():
    torch.manual_seed(0)
    ex = Executor(4, 3, ExecutorConfig(), agent_ids=torch.tensor([3, 4, 5, 6]))
    p = torch.randn(4, 3, dtype=torch.float64) * 10
    v = torch.randn(4, 3, dtype=torch.float64) * 3
    ex.radius = torch.tensor([0.6, 0.6123, 1.5, 0.75], dtype=torch.float64)
    ex.flags = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
    bc = ex.broadcast(12.34, p, v)
    raw = encode_safety_broadcast(bc)
    assert all(len(r) == 22 for r in raw)
    raw[2] = raw[2][:5] + bytes([raw[2][5] ^ 0x10]) + raw[2][6:]        # corrupt one packet
    dec = decode_safety_broadcast(raw, now_s=12.35)
    keep = [0, 1, 3]
    q = bc.quantized()
    assert dec.agent_id.tolist() == [3, 4, 6]
    assert torch.allclose(dec.pos, q.pos[keep], atol=1e-12) and torch.allclose(dec.vel, q.vel[keep], atol=1e-12)
    assert torch.allclose(dec.radius, q.radius[keep], atol=1e-12)
    assert bool((dec.radius >= bc.radius[keep]).all())
    assert torch.allclose(dec.stamp, torch.full((3,), 12.34, dtype=torch.float64), atol=1e-6)
    assert dec.flags.tolist() == [0, 1, 3]
    nb = neighbor_tables_from_broadcasts(dec)
    assert nb.pos.shape == (3, 2, 3)


# ---------------------------------------------------------------------------
# RVQ (Lemma 4, Proposition 5)
# ---------------------------------------------------------------------------
def test_rvq_zero_codeword_and_shapes():
    q = ResidualVQ()
    Q = q.codebooks
    assert Q.shape == (24, 256, 64) and Q.numel() == TABLE3_BUDGET["rvq"]
    assert torch.equal(Q[:, 0], torch.zeros(24, 64))
    assert q.codebook_free.numel() == 24 * 255 * 64


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e3])
def test_rvq_lemma4_residual_monotonicity(scale):
    torch.manual_seed(1)
    q = ResidualVQ()
    q.init_from_data(torch.randn(2000, 64), iters=3)
    x = torch.randn(3000, 64) * scale
    n = q.residual_norms(x)
    assert bool((n[:, 1:] <= n[:, :-1]).all())                   # ||r_b|| <= ||r_{b-1}|| exactly
    x_hat = q.decode(q.encode(x))
    err = torch.linalg.vector_norm(x - x_hat, dim=-1)
    xn = torch.linalg.vector_norm(x, dim=-1)
    tol = 1e-5 * xn + 1e-30
    assert bool((err <= xn + tol).all())                         # ||x - x_hat|| <= ||x||
    assert bool((torch.linalg.vector_norm(x_hat, dim=-1) <= 2 * xn + tol).all())
    assert torch.allclose(err, n[:, -1], rtol=1e-4, atol=1e-6 * scale)   # x - x_hat = r_B


def test_rvq_ties_and_determinism():
    q = ResidualVQ(n_stages=2, codebook_size=4, dim=2)
    with torch.no_grad():
        q.codebook_free.copy_(torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                                            [[0.1, 0.0], [0.0, 0.1], [0.1, 0.0]]]))
    codes = q.encode(torch.tensor([[1.0, 0.0], [0.9, 0.0]]))
    assert codes[:, 0].tolist() == [1, 1]          # rows 1 and 2 tie: the smallest index wins
    q2 = ResidualVQ()
    x = torch.randn(64, 64)
    c1, c2 = q2.encode(x), q2.encode(x.clone())
    assert torch.equal(c1, c2) and c1.dtype == torch.uint8
    assert torch.equal(q2.decode(c1), q2.decode(c2))
    with pytest.raises(ValueError):
        ResidualVQ(n_stages=1, codebook_size=4, dim=2).decode(torch.tensor([[7]]))


def test_rvq_training_step_and_straight_through():
    torch.manual_seed(2)
    q = ResidualVQ().train()
    data = torch.randn(4096, 64) @ torch.diag(torch.linspace(2.0, 0.05, 64))
    q.init_from_data(data[:2048], iters=5)
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    x = data[2048:].clone().requires_grad_(True)
    out = q(x)
    assert out.codes.shape == (2048, 24) and torch.allclose(out.quantized, out.x_hat, atol=1e-5)
    out.quantized.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))          # straight-through
    losses = []
    for _ in range(25):
        opt.zero_grad()
        o = q(data[2048:])
        o.loss.backward()
        opt.step()
        losses.append(float(o.loss.detach()))
    assert losses[-1] < losses[0]
    assert q.snr_db(data[2048:]) > 3.0
    q.freeze()
    assert not q.codebook_free.requires_grad and q.restart_dead_codes(data) == 0


def test_rvq_rate_distortion_floor():
    assert math.isclose(gaussian_rd_floor(), 1 / 64)
    assert math.isclose(10 * math.log10(64), 18.06, abs_tol=0.01)
    torch.manual_seed(3)
    q = ResidualVQ()
    q.init_from_data(torch.randn(4000, 64), iters=5)
    assert float(q.relative_mse(torch.randn(4000, 64))) >= gaussian_rd_floor()   # converse bound


# ---------------------------------------------------------------------------
# parameter budget (Table 3) and master size
# ---------------------------------------------------------------------------
def test_table3_parameter_budget():
    c = ClonePolicy()
    b = c.parameter_budget()
    for k, v in TABLE3_BUDGET.items():
        assert b[k] == v, (k, b[k], v)
    assert sum(TABLE3_BUDGET.values()) == 18_669_514
    assert b["total"] == 18_669_514 + 6 * 32 + 16 * 32 < 18_700_000
    assert c.cfg.cond_dim == 721 and c.selector.in_dim == 433
    m = MasterPlanner()
    n = sum(p.numel() for p in m.parameters())
    assert 4.2e6 < n < 4.6e6                                     # paper: 4.40M (no breakdown given)


# ---------------------------------------------------------------------------
# Proposition 8: sequential depth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [1, 2, 3, 5])
@pytest.mark.parametrize("H", [4, 16])
def test_prop8_depth_is_K_plus_3(K, H):
    c = tiny_clone(K=K, horizon=H, H_min=2)
    log = []
    for name, mod in (("E", c.encoder), ("pi", c.selector), ("P", c.predictor), ("v", c.field)):
        mod.register_forward_hook(lambda *_a, n=name: log.append(n))
    g = torch.Generator().manual_seed(4)
    out = c.decide(frames(c, 3, g), torch.randn(3, 13, generator=g), torch.randn(3, 3, generator=g), 0.0)
    assert log == ["E", "pi", "P"] + ["v"] * K
    assert len(out.chain) == K + 3 and out.chunk.shape == (3, H, 4)


def test_depth_guard_detects_extra_evaluation():
    c = tiny_clone()
    orig = c.selector.forward

    def leaky(*a):
        c.encoder(torch.zeros(1, c.cfg.in_channels, *c.cfg.image_size))
        return orig(*a)

    c.selector.forward = leaky
    g = torch.Generator().manual_seed(5)
    with pytest.raises(RuntimeError, match="sequential-depth"):
        c.decide(frames(c, 3, g), torch.zeros(3, 13), torch.zeros(3, 3), 0.0)


# ---------------------------------------------------------------------------
# bridge: K/V off the chain, double buffer, validity, null token
# ---------------------------------------------------------------------------
def test_bridge_kv_computed_once_per_message_not_in_solver():
    c = tiny_clone(K=3)
    m = tiny_master(c.cfg)
    calls = []
    for layer in c.field.cross:
        for proj in (layer.k_proj, layer.v_proj):
            proj.register_forward_hook(lambda *_a: calls.append(1))
    _, dl = master_round(c, m, 0.0, gen=torch.Generator().manual_seed(0))
    calls.clear()
    assert bool(c.on_downlink(dl, 0.01).all())
    assert len(calls) == 2 * c.field.n_cross and c.bridge.n_writes == 1      # once per message
    g = torch.Generator().manual_seed(6)
    calls.clear()
    for t in range(3):
        c.decide(frames(c, 3, g), torch.zeros(3, 13), torch.zeros(3, 3), 0.02 + 0.033 * t)
    assert calls == []                                                       # never inside a decision
    # cached K/V equal a fresh projection of the same plan tokens
    kv = c.bridge.read(0.05)
    p = DownlinkPacket.unpack(dl[0])
    pi_hat = c.rvq.decode(torch.tensor([list(p.code)], dtype=torch.uint8))
    tok = c.plan_tokens(pi_hat, torch.tensor([p.mode]), c.tracker.active(c.milestones)[:1])
    fresh = c.field.project_kv(tok)
    for a, b in zip(kv.K + kv.V, fresh.K + fresh.V):
        assert torch.allclose(a[:1], b, atol=1e-6)


def test_bridge_double_buffer_validity_and_null_plan():
    c = tiny_clone()
    m = tiny_master(c.cfg)
    front0 = c.bridge.front.clone()
    _, dl = master_round(c, m, 0.0, validity=0.5, gen=torch.Generator().manual_seed(1))
    c.on_downlink(dl, 0.0)
    assert torch.equal(c.bridge.front, 1 - front0)                      # flipped after the write
    assert bool(c.bridge.valid(0.4).all()) and not bool(c.bridge.valid(0.6).any())
    null = c.field.null_kv(3)
    late = c.bridge.read(0.6)
    for a, b in zip(late.K, null.K):
        assert torch.allclose(a, b)
    # a duplicate (same sequence number) is rejected; a corrupted one too
    assert not bool(c.on_downlink(dl, 0.1).any())
    bad = [d[:-1] + bytes([d[-1] ^ 1]) for d in dl]
    assert not bool(c.on_downlink(bad, 0.1).any())
    # a "no valid plan" downlink switches the agent to the null plan immediately
    m.seq += 1
    nd = [DownlinkPacket.build(a, m.seq, 0.2, 3, 1, [0] * 24, 0.0, 0.0).pack() for a in c.agent_ids]
    c.on_downlink(nd, 0.2)
    assert not bool(c.bridge.valid(0.2).any()) and c.u_M.tolist() == [3, 3, 3]
    flags = [UplinkPacket.unpack(u).flags for u in c.uplinks(0.21)]
    assert all(f & 0x04 for f in flags)                                     # stale-plan flag


def test_prop4c_identical_clones_reconstruct_identical_tokens():
    c1 = tiny_clone(seed=7)
    c2 = tiny_clone(seed=7)
    m = tiny_master(c1.cfg)
    _, dl = master_round(c1, m, 0.0, gen=torch.Generator().manual_seed(2))
    c1.on_downlink(dl, 0.0)
    c2.on_downlink(dl, 0.0)
    k1, k2 = c1.bridge.read(0.1), c2.bridge.read(0.1)
    for a, b in zip(k1.K + k1.V, k2.K + k2.V):
        assert torch.equal(a, b)                                            # bit-exact


# ---------------------------------------------------------------------------
# master (Eq. 11-12)
# ---------------------------------------------------------------------------
def _team(B=2, N=5, seed=0, cfg=None):
    g = torch.Generator().manual_seed(seed)
    cfg = cfg or MasterConfig(width=32, heads=4, depth=2, mode_emb_dim=8, plan_hidden=32, bus_dim=16,
                              latent_dim=32, text_dim=16)
    return cfg, (torch.randn(B, N, cfg.bus_dim, generator=g), torch.rand(B, N, generator=g),
                 torch.ones(B, N, dtype=torch.bool), torch.randn(B, cfg.text_dim, generator=g),
                 torch.randn(B, N, cfg.latent_dim, generator=g))


def test_master_permutation_equivariance_and_no_identity():
    cfg, (x, s, pres, c, zg) = _team()
    torch.manual_seed(0)
    m = MasterPlanner(cfg).eval()
    perm = torch.tensor([3, 0, 4, 1, 2])
    u_g = torch.tensor([1, 2])
    with torch.no_grad():
        a = m(x, s, pres, c, zg, u_g=u_g)
        b = m(x[:, perm], s[:, perm], pres[:, perm], c, zg[:, perm], u_g=u_g)
    assert torch.allclose(a.h[:, perm], b.h, atol=1e-5) and torch.allclose(a.h0, b.h0, atol=1e-5)
    assert torch.equal(a.u_M[:, perm], b.u_M)
    assert torch.allclose(a.plan[:, perm], b.plan, atol=1e-5)
    # identical agents receive identical features (no identity embeddings)
    x2, s2, zg2 = x.clone(), s.clone(), zg.clone()
    x2[:, 1], s2[:, 1], zg2[:, 1] = x2[:, 0], s2[:, 0], zg2[:, 0]
    with torch.no_grad():
        d = m(x2, s2, pres, c, zg2, u_g=u_g)
    assert torch.allclose(d.h[:, 0], d.h[:, 1], atol=1e-6)


def test_master_variable_team_size():
    cfg, (x, s, pres, c, zg) = _team(B=1, N=6)
    torch.manual_seed(1)
    m = MasterPlanner(cfg).eval()
    pres = torch.tensor([[True, True, True, False, True, False]])
    u_g = torch.tensor([0])
    x_alt = x.clone()
    x_alt[:, 3] += 100.0                                     # garbage in absent slots must not matter
    x_alt[:, 5] -= 50.0
    with torch.no_grad():
        a = m(x, s, pres, c, zg, u_g=u_g)
        b = m(x_alt, s, pres, c, zg, u_g=u_g)
        keep = pres[0]
        small = m(x[:, keep], s[:, keep], pres[:, keep], c, zg[:, keep], u_g=u_g)
    assert torch.allclose(a.h, b.h, atol=1e-5) and torch.allclose(a.h0, b.h0, atol=1e-5)
    assert torch.allclose(a.h[:, keep], small.h, atol=1e-5) and torch.allclose(a.h0, small.h0, atol=1e-5)
    assert bool((a.h[0, ~keep] == 0).all())


def test_master_mode_sampling_eq11_and_plan_spread_eq12():
    cfg, (x, s, pres, c, zg) = _team(B=64, N=3, seed=2)
    torch.manual_seed(2)
    m = MasterPlanner(cfg).eval()
    with torch.no_grad():
        o1 = m(x, s, pres, c, zg, generator=torch.Generator().manual_seed(9))
        o2 = m(x, s, pres, c, zg, generator=torch.Generator().manual_seed(9))
        o3 = m(x, s, pres, c, zg, generator=torch.Generator().manual_seed(10))
        h, h0 = m.encode(x, s, pres, c, zg)
    assert torch.equal(o1.u_g, o2.u_g) and not torch.equal(o1.u_g, o3.u_g)     # sampled, reproducible
    # local modes are the deterministic argmax given u_g (all randomness in u_g)
    assert torch.equal(o1.u_M, m.local_logits(h, o1.u_g).argmax(-1))
    with torch.no_grad():
        o4 = m(x, s, pres, c, zg, u_g=o1.u_g)
    assert torch.equal(o4.u_M, o1.u_M)
    # Eq. 12: mean of the heads and the normalized spread
    pi = o1.plan_heads.mean(0)
    assert torch.allclose(o1.plan, pi, atol=1e-6)
    spread = ((o1.plan_heads - pi).pow(2).sum(-1).mean(0)).sqrt() / (math.sqrt(cfg.bus_dim) * 1.0)
    assert torch.allclose(o1.sigma_M, spread, atol=1e-6)
    assert handedness_of_mode(torch.tensor([0, 1, 2, 3])).tolist() == [1, -1, 1, -1]


def test_end_to_end_bus_roundtrip():
    c = tiny_clone(seed=11)
    m = tiny_master(c.cfg, seed=11)
    g = torch.Generator().manual_seed(12)
    c.decide(frames(c, 3, g), torch.zeros(3, 13), torch.zeros(3, 3), 0.0)
    ups = c.uplinks(0.0)
    assert all(len(u) == 35 for u in ups)
    ups[1] = ups[1][:-2] + b"\x00\x00"                                   # dropped by the CRC check
    dec = m.decode_uplinks(ups, c.agent_ids, c.rvq)
    assert dec["present"].tolist() == [True, False, True]
    x_ref = c.rvq.decode(c.rvq.encode(c.bus_latent(c.last_z)))
    assert torch.equal(dec["x_hat"][0], x_ref[0])
    out = m.cycle(c.rvq, dec["codes"][None], dec["sigma"][None], dec["present"][None], c.c_tilde[:1],
                  c.tracker.active(c.milestones)[None], generator=g)
    dl = m.downlinks(out, c.agent_ids, 0.1, c.tracker.m, 1.0)
    assert all(len(d) == 36 for d in dl)
    assert bool(c.on_downlink(dl, 0.11).all())
    assert torch.equal(c.u_M, out.u_M[0])
    d = c.decide(frames(c, 3, g), torch.zeros(3, 13), torch.zeros(3, 3), 0.12)
    assert bool(d.plan_valid.all())


# ---------------------------------------------------------------------------
# noise persistence, outputs, Stage-A guidance inside DECIDE
# ---------------------------------------------------------------------------
def test_noise_persistence_rules():
    npst = NoisePersistence((4, 2), N_xi=3)
    g = torch.Generator().manual_seed(0)
    u, md = torch.tensor([0, 1]), torch.tensor([2, 2])
    x0 = npst.draw(u, md, g).clone()                                  # t = 0: drawn
    x1 = npst.draw(u, md, g).clone()                                  # t = 1: kept
    assert torch.equal(x0, x1)
    x2 = npst.draw(torch.tensor([0, 3]), md, g).clone()               # primitive of agent 1 changed
    assert torch.equal(x2[0], x1[0]) and not torch.equal(x2[1], x1[1])
    x3 = npst.draw(torch.tensor([0, 3]), md, g).clone()               # t = 3 = 0 mod N_xi: redraw all
    assert not torch.equal(x3[0], x2[0]) and not torch.equal(x3[1], x2[1])
    x4 = npst.draw(torch.tensor([0, 3]), torch.tensor([2, 5]), g)     # master mode changed for agent 1
    assert torch.equal(x4[0], x3[0]) and not torch.equal(x4[1], x3[1])


def test_decision_outputs_and_chunk_buffer():
    c = tiny_clone()
    g = torch.Generator().manual_seed(13)
    d = c.decide(frames(c, 3, g), torch.zeros(3, 13), torch.zeros(3, 3), 1.0)
    assert bool((d.chunk.abs() <= 1).all()) and bool((d.H_val >= c.cfg.H_min).all())
    assert bool((d.r_eff >= 0.6 - 1e-9).all() & (d.r_eff <= 1.5 + 1e-9).all())
    assert torch.allclose(d.chunk[..., :3], d.raw_chunk[..., :3] * d.alpha.view(-1, 1, 1))
    assert torch.equal(d.chunk[..., 3], d.raw_chunk[..., 3])            # yaw rate passes unmodified
    A, stamp, hv = c.chunks.newest()
    assert torch.equal(A, d.chunk) and bool((stamp == 1.0).all()) and torch.equal(hv, d.H_val)
    assert not bool(d.plan_valid.any())                                 # no master yet: null plan


def test_decide_with_stage_a_guidance_terminal_feasible():
    c = tiny_clone(K=2, horizon=8)
    g = torch.Generator().manual_seed(14)
    B, H = 3, 8
    p = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=torch.float64)
    v = torch.zeros(B, 3, dtype=torch.float64)
    nbr_pos = torch.stack([p[[1, 2]], p[[0, 2]], p[[0, 1]]])
    nbr_pos[0, 0] = torch.tensor([1.6, 0.0, 0.0], dtype=torch.float64)     # a close neighbour
    nbr_vel = torch.zeros(B, 2, 3, dtype=torch.float64)
    cm = ChunkMap.from_yaw(torch.zeros(B, dtype=torch.float64), scale=8.0)
    poly = build_stage_a_polytope(p, v, constant_velocity_reference(v, H), nbr_pos, nbr_vel, cm,
                                  D=torch.full((B, 2), 1.2, dtype=torch.float64))
    d = c.decide(frames(c, B, g), torch.zeros(B, 13), torch.zeros(B, 3), 0.0, polytope=poly)
    assert len(d.chain) == c.cfg.K + 3
    assert d.guided is not None and bool(d.guided.terminal_feasible.all())   # Lemma 17(c)

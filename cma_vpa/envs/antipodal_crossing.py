# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Benchmark Task T1: high-density multi-UAV antipodal crossing (Section 7.2) and the closed-loop runner.

``N_a in {4, 8, 12, 16}`` quadrotors start on a horizontal circle of radius
20 m (altitude jitter +-0.5 m) and must reach the antipodal point within 1 m,
crossing a central conflict sphere of radius 5 m, under wind W0/W1/W2. An
episode succeeds if every vehicle reaches its goal within ``T_max = 60 s``
without collision; a collision is a separation below ``2 r_v = 0.70 m``.
Makespan is the time at which the last vehicle reaches its goal.

:class:`CrossingRunner` executes the asynchronous loops of Algorithm 1 in
simulated time on one thread, in a deterministic interleaving:

* master cycle every ``1/f_M`` s (default 2 Hz) - uplinks in, downlinks out,
  all through the 35/36-byte codecs;
* clone decision every ``1/f_dec`` s (default 30 Hz); the chunk becomes visible
  to the executor after ``latency`` s, stamped with its decision time, so the
  executor skips the knots that correspond to inference and transport latency;
* executor tick every ``T_c = 10 ms`` (100 Hz) with Stage B and the fallback,
  and the 22-byte safety broadcast of every vehicle (one-tick transport delay,
  optional packet loss).

Chunk sources (``policy``):

* :class:`ExpertChunkPolicy` - privileged expert chunks with a handedness taken
  from the master's downlink mode token (``"master"``), sampled independently
  per vehicle (``"independent"``), or fixed;
* :class:`LearnedChunkPolicy` - the neural clones (:class:`~cma_vpa.models.ClonePolicy`)
  on rendered RGB-D frames, optionally with Stage-A guidance.

Masters: :class:`ModeMaster` (samples one global handedness ``u_g`` per mission
and broadcasts it, the mechanism of Proposition 7(c)) and :class:`LearnedMaster`
(:class:`~cma_vpa.models.MasterPlanner` on decoded uplinks).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from ..comm.packet import (DN_NEW_EPOCH, DownlinkPacket, PacketError, SequenceFilter, decode_safety_broadcast,
                           encode_safety_broadcast)
from ..safety.degg_flow import ChunkMap, build_stage_a_polytope, shift_reference
from ..safety.executor import (Chunk, Executor, ExecutorConfig, NeighborTable, SafetyBroadcast,
                               neighbor_tables_from_broadcasts)
from ..safety.params import DEFAULT_PARAMS, SafetyParams
from .expert import CrossingExpert, ExpertConfig, primitive_labels
from .quadrotor_sim import QuadrotorTeam, WindConfig
from .render import render_rgbd

__all__ = [
    "CrossingConfig", "AntipodalCrossing", "ExpertChunkPolicy", "LearnedChunkPolicy", "ModeMaster",
    "LearnedMaster", "CrossingRunner", "EpisodeResult", "proprio_state", "V_SCALE", "YAW_RATE_SCALE",
]

V_SCALE = 8.0           # chunk normalization of velocities [m/s] (= v_bar)
YAW_RATE_SCALE = 1.0    # chunk normalization of the yaw rate [rad/s]


@dataclass
class CrossingConfig:
    n_agents: int = 8
    radius: float = 20.0
    altitude: float = 5.0
    z_jitter: float = 0.5
    angle_jitter_deg: float = 0.0
    goal_tol: float = 1.0
    conflict_radius: float = 5.0
    v_h: float = 5.0
    wind: str = "W0"
    T_max: float = 60.0
    r_v: float = 0.35
    image_size: tuple = (128, 160)
    seed: int = 0


def proprio_state(v_heading: Tensor, a_bar: Tensor, dz: Tensor, yaw: Tensor, sigma_bar: Tensor,
                  flags: Tensor) -> Tensor:
    """``S_t in R^13``: heading-frame velocity / v_scale (3), last setpoint (4), altitude error (1),
    sin/cos yaw (2), sigma_bar (1), braking and fallback flags (2)."""
    f = flags.to(torch.int64)
    return torch.cat([v_heading / V_SCALE, a_bar, dz.unsqueeze(-1), torch.sin(yaw).unsqueeze(-1),
                      torch.cos(yaw).unsqueeze(-1), sigma_bar.unsqueeze(-1),
                      ((f & 2) != 0).float().unsqueeze(-1), ((f & 1) != 0).float().unsqueeze(-1)], -1).float()


class AntipodalCrossing:
    """State, observations and metrics of one T1 episode."""

    def __init__(self, cfg: CrossingConfig = CrossingConfig(), safety: SafetyParams = DEFAULT_PARAMS):
        self.cfg, self.safety = cfg, safety
        self.reset(cfg.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        c = self.cfg
        g = torch.Generator().manual_seed(c.seed if seed is None else seed)
        self.generator = g
        N = c.n_agents
        ang = 2 * math.pi * torch.arange(N, dtype=torch.float64) / N
        ang = ang + math.radians(c.angle_jitter_deg) * (2 * torch.rand(N, generator=g, dtype=torch.float64) - 1)
        z = c.altitude + c.z_jitter * (2 * torch.rand(N, generator=g, dtype=torch.float64) - 1)
        self.start = torch.stack([c.radius * torch.cos(ang), c.radius * torch.sin(ang), z], -1)
        self.goals = torch.stack([-self.start[:, 0], -self.start[:, 1], z], -1)
        d = self.goals - self.start
        self.yaw = torch.atan2(d[:, 1], d[:, 0])                         # heading frozen toward the goal
        self.sim = QuadrotorTeam(self.start, wind=WindConfig.named(c.wind), T_c=self.safety.T_c,
                                 u_inf=self.safety.u_inf, e_a=self.safety.e_a, generator=g)
        self.arrival = torch.full((N,), math.nan, dtype=torch.float64)
        self.d_min = math.inf
        self.collisions = 0
        self.collided_pairs: set = set()

    @property
    def t(self) -> float:
        return self.sim.t

    def chunk_map(self) -> ChunkMap:
        return ChunkMap.from_yaw(self.yaw, scale=V_SCALE)

    def rel_goal_heading(self) -> Tensor:
        d = self.goals - self.sim.p
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], -1)

    def v_heading(self) -> Tensor:
        v = self.sim.v
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)

    def render(self, image_size: Optional[tuple] = None, positions: Optional[Tensor] = None) -> Tensor:
        """Front-camera RGB-D frames ``(N, 4, H, W)`` for every vehicle."""
        p = self.sim.p if positions is None else positions
        N = p.shape[0]
        idx = torch.tensor([[j for j in range(N) if j != i] for i in range(N)], dtype=torch.long)
        others = p[idx] if N > 1 else torch.zeros(N, 0, 3, dtype=p.dtype)
        rad = torch.full(others.shape[:2], self.cfg.r_v, dtype=torch.float32)
        return render_rgbd(p, self.yaw, others, rad, self.goals, image_size=image_size or self.cfg.image_size)

    def milestone_frames(self, image_size: Optional[tuple] = None) -> Tensor:
        """One milestone per vehicle: the view from just before its goal, ``(N, 1, 4, H, W)``."""
        approach = self.goals - 2.0 * torch.stack([torch.cos(self.yaw), torch.sin(self.yaw),
                                                   torch.zeros_like(self.yaw)], -1)
        return self.render(image_size, positions=approach).unsqueeze(1)

    def update_metrics(self) -> None:
        p = self.sim.p
        N = p.shape[0]
        if N > 1:
            D = torch.cdist(p, p) + torch.diag(torch.full((N,), math.inf, dtype=p.dtype))
            self.d_min = min(self.d_min, float(D.min()))
            close = (D < 2 * self.cfg.r_v).nonzero().tolist()
            for i, j in close:
                if i < j and (i, j) not in self.collided_pairs:
                    self.collided_pairs.add((i, j))
                    self.collisions += 1
        dist = torch.linalg.vector_norm(self.goals - p, dim=-1)
        new = torch.isnan(self.arrival) & (dist < self.cfg.goal_tol)
        self.arrival = torch.where(new, torch.full_like(self.arrival, self.t), self.arrival)

    @property
    def all_arrived(self) -> bool:
        return not bool(torch.isnan(self.arrival).any())


# ---------------------------------------------------------------------------
# masters
# ---------------------------------------------------------------------------
class ModeMaster:
    """Samples one global handedness per mission and broadcasts it in 36-byte downlinks.

    ``u_g`` is drawn once (``DN_NEW_EPOCH`` set in the first cycle) and kept, so
    that every vehicle receives the same mode token; the plan code is empty and
    the plan validity is zero (the expert clones use only the mode).
    """

    def __init__(self, n_agents: int, generator: Optional[torch.Generator] = None, u_g: Optional[int] = None):
        self.N = n_agents
        self.u_g = int(torch.randint(0, 2, (1,), generator=generator)) if u_g is None else int(u_g)
        self.seq = 0
        self.first = True

    def cycle(self, uplinks: Sequence[bytes], now: float, agent_ids: Sequence[int],
              milestones: Sequence[int]) -> List[bytes]:
        self.seq = (self.seq + 1) % 65536
        status = DN_NEW_EPOCH if self.first else 0
        self.first = False
        return [DownlinkPacket.build(a, self.seq, now, self.u_g % 2, milestones[k], [0] * 24, 0.0, 0.0,
                                     status).pack() for k, a in enumerate(agent_ids)]


class LearnedMaster:
    """:class:`~cma_vpa.models.MasterPlanner` on decoded uplinks (Algorithm 1 MASTERCYCLE)."""

    def __init__(self, planner, rvq, c_tilde: Tensor, z_goal: Tensor, validity_s: float = 1.5,
                 generator: Optional[torch.Generator] = None, resample_every: int = 0):
        self.planner, self.rvq = planner, rvq
        self.c_tilde, self.z_goal = c_tilde, z_goal
        self.validity_s, self.g = validity_s, generator
        self.resample_every = resample_every
        self.n_cycles = 0
        self.u_g: Optional[Tensor] = None

    @torch.no_grad()
    def cycle(self, uplinks: Sequence[bytes], now: float, agent_ids: Sequence[int],
              milestones: Sequence[int]) -> List[bytes]:
        dec = self.planner.decode_uplinks(uplinks, agent_ids, self.rvq)
        x_hat = dec["x_hat"][None].to(self.c_tilde.dtype)
        resample = self.u_g is None or (self.resample_every and self.n_cycles % self.resample_every == 0)
        if resample:
            out = self.planner(x_hat, dec["sigma"][None], dec["present"][None], self.c_tilde[:1],
                               self.z_goal[None], generator=self.g)
            self.u_g = out.u_g
        else:
            out = self.planner(x_hat, dec["sigma"][None], dec["present"][None], self.c_tilde[:1],
                               self.z_goal[None], u_g=self.u_g)
        out.codes = self.rvq.encode(out.plan)
        self.n_cycles += 1
        return self.planner.downlinks(out, agent_ids, now, torch.as_tensor(milestones), self.validity_s,
                                      status=DN_NEW_EPOCH if resample else 0)


# ---------------------------------------------------------------------------
# chunk sources
# ---------------------------------------------------------------------------
class ExpertChunkPolicy:
    """Privileged expert chunks; the handedness comes from the downlink mode token or is sampled.

    ``mode_source``: ``"master"`` (mode token of the latest valid downlink;
    until one arrives the vehicle uses ``default_h``), ``"independent"`` (each
    vehicle samples its own side once, Proposition 7(b)), or ``"fixed"``.
    """

    def __init__(self, env: AntipodalCrossing, H: int = 32, mode_source: str = "master",
                 default_h: int = 1, expert_cfg: Optional[ExpertConfig] = None,
                 generator: Optional[torch.Generator] = None):
        self.env, self.H = env, H
        N = env.cfg.n_agents
        if mode_source == "independent":
            h = 1 - 2 * torch.randint(0, 2, (N,), generator=generator)
        else:
            h = torch.full((N,), default_h)
        self.h = h.to(torch.float64)
        self.mode_source = mode_source
        if expert_cfg is None:
            ring = max(4.0, 2.5 * N / (2 * math.pi))          # ring spacing of about 2.5 m per vehicle
            expert_cfg = ExpertConfig(v_h=env.cfg.v_h, ring_radius=ring, round_radius=ring + 6.0)
        self.ecfg = expert_cfg
        self.filter = SequenceFilter(16)
        self.sigma_bar = torch.zeros(N)
        self.n_decisions = 0

    def on_downlink(self, msgs: Sequence[bytes], now: float) -> None:
        if self.mode_source != "master":
            return
        for raw in msgs:
            try:
                p = DownlinkPacket.unpack(raw)
            except PacketError:
                continue
            if 0 <= p.agent_id < self.h.shape[0] and self.filter.accept(p.agent_id, p.seq):
                self.h[p.agent_id] = 1.0 if p.mode % 2 == 0 else -1.0

    def uplinks(self, now: float, flags: Tensor) -> List[bytes]:
        return []

    def decide(self, now: float, nbrs: Optional[NeighborTable] = None) -> Dict[str, Tensor]:
        env = self.env
        expert = CrossingExpert(env.goals, self.h, self.ecfg)
        V = expert.plan(env.sim.p, env.sim.v, self.H, env.safety.T_c)          # (N, H, 3) world
        A = torch.zeros(V.shape[0], self.H, 4, dtype=torch.float64)
        A[..., :3] = env.chunk_map().to_normalized(V)
        self.n_decisions += 1
        return {"A": A.clamp(-1, 1), "H_val": torch.full((V.shape[0],), self.H, dtype=torch.long),
                "sigma_bar": self.sigma_bar, "prim": primitive_labels(V[:, 0], env.sim.p, env.goals, env.cfg.v_h)}


class LearnedChunkPolicy:
    """Neural clones on rendered frames (DECIDE), optionally with Stage-A guidance."""

    def __init__(self, env: AntipodalCrossing, clone, stage_a: bool = False,
                 generator: Optional[torch.Generator] = None, executor: Optional[Executor] = None):
        self.env, self.clone, self.stage_a, self.g = env, clone, stage_a, generator
        self.executor = executor
        self.prev_V: Optional[Tensor] = None
        self.prev_stamp = 0.0
        self.lam: Optional[Tensor] = None
        self.n_decisions = 0

    def on_downlink(self, msgs: Sequence[bytes], now: float) -> None:
        self.clone.on_downlink(msgs, now)

    def uplinks(self, now: float, flags: Tensor) -> List[bytes]:
        return self.clone.uplinks(now, flags)

    def decide(self, now: float, nbrs: Optional[NeighborTable] = None) -> Dict[str, Tensor]:
        env, clone = self.env, self.clone
        cfg = clone.cfg
        frames = env.render(cfg.image_size)
        flags = self.executor.flags if self.executor is not None else torch.zeros(env.cfg.n_agents, dtype=torch.uint8)
        sb = clone.filter.sigma_bar if clone.filter.sigma_bar is not None else torch.zeros(env.cfg.n_agents)
        s_t = proprio_state(env.v_heading().float(), clone.a_bar.float(),
                            (env.sim.p[:, 2] - env.goals[:, 2]).float(), env.yaw.float(), sb.float(), flags)
        poly = None
        if self.stage_a and nbrs is not None:
            cm = env.chunk_map()
            V_ref = (shift_reference(self.prev_V, (now - self.prev_stamp) / env.safety.T_c)
                     if self.prev_V is not None else env.sim.v.unsqueeze(1).expand(-1, cfg.horizon, -1).clone())
            age = now - nbrs.stamp
            p_j = nbrs.pos + nbrs.vel * age.unsqueeze(-1)
            D = self.executor.radius.unsqueeze(-1) + nbrs.radius if self.executor is not None else \
                torch.full(nbrs.present.shape, env.safety.D_nominal, dtype=torch.float64)
            poly = build_stage_a_polytope(env.sim.p, env.sim.v, V_ref, p_j, nbrs.vel, cm, D=D,
                                          heard=nbrs.present, cfg=env.safety)
        out = clone.decide(frames, s_t, env.rel_goal_heading().float(), now, polytope=poly, lam0=self.lam,
                           generator=self.g)
        if out.guided is not None:
            self.lam = out.guided.lam
        A = out.chunk.to(torch.float64)
        self.prev_V = env.chunk_map().to_world(A[..., :3])
        self.prev_stamp = now
        self.n_decisions += 1
        return {"A": A, "H_val": out.H_val, "sigma_bar": out.sigma_bar.to(torch.float64), "prim": out.primitive}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    success: bool
    makespan: float
    d_min: float
    collisions: int
    arrival: Tensor
    duration: float
    ticks: int
    interventions: int          # agent-ticks in which Stage B modified the nominal input
    fallback_events: int
    braking_ticks: int
    causes: Dict[str, int]
    n_master_cycles: int
    n_decisions: int
    executor_latency: Dict[str, float]
    wall_time: float
    trajectory: Optional[Tensor] = None
    records: List[dict] = field(default_factory=list)


class CrossingRunner:
    """Deterministic simulated-time interleaving of the master, clone and executor loops."""

    def __init__(self, env: AntipodalCrossing, policy, master=None, *, f_master: float = 2.0,
                 f_dec: float = 30.0, latency: float = 0.015, packet_loss: float = 0.0,
                 executor_cfg: Optional[ExecutorConfig] = None, use_packets: bool = True,
                 record_every: int = 0, keep_trajectory_every: int = 0,
                 generator: Optional[torch.Generator] = None, expert_mix: Optional[float] = None,
                 expert_policy: Optional[ExpertChunkPolicy] = None):
        self.env, self.policy, self.master = env, policy, master
        self.f_master, self.f_dec, self.latency = f_master, f_dec, latency
        self.packet_loss, self.use_packets = packet_loss, use_packets
        self.g = generator or torch.Generator().manual_seed(env.cfg.seed + 101)
        self.record_every, self.keep_traj = record_every, keep_trajectory_every
        self.expert_mix, self.expert_policy = expert_mix, expert_policy
        N = env.cfg.n_agents
        # a holding vehicle drifts under gusts; 0.25 m/s keeps such drift from re-triggering sympathetic braking
        ec = executor_cfg or ExecutorConfig(safety=env.safety, v_hold=0.25)
        if hasattr(policy, "clone"):
            ec = ExecutorConfig(**{**ec.__dict__, "H": policy.clone.cfg.horizon})
        self.executor = Executor(N, max(N - 1, 1), ec)
        if hasattr(policy, "executor"):
            policy.executor = self.executor
        self.ids = list(range(N))
        # last-heard store of safety broadcasts (per sender)
        dt = torch.float64
        self.store = SafetyBroadcast(agent_id=torch.arange(N), seq=torch.zeros(N, dtype=torch.long),
                                     stamp=torch.full((N,), -math.inf, dtype=dt), pos=env.sim.p.clone(),
                                     vel=torch.zeros(N, 3, dtype=dt),
                                     radius=torch.full((N,), env.safety.r_s, dtype=dt),
                                     flags=torch.zeros(N, dtype=torch.uint8))
        self.inflight: List[bytes] = []
        self.pending: List[tuple] = []

    # ------------------------------------------------------------------ safety channel
    def _receive(self, now: float) -> NeighborTable:
        if self.inflight:
            if self.use_packets:
                bc = decode_safety_broadcast(self.inflight, now)
            else:
                bc = self.inflight[0]
            for k, a in enumerate(bc.agent_id.tolist()):
                self.store.pos[a] = bc.pos[k]
                self.store.vel[a] = bc.vel[k]
                self.store.radius[a] = bc.radius[k]
                self.store.flags[a] = bc.flags[k]
                self.store.stamp[a] = bc.stamp[k]
                self.store.seq[a] = bc.seq[k]
        nb = neighbor_tables_from_broadcasts(self.store)
        nb.present = nb.present & torch.isfinite(nb.stamp)
        nb.stamp = torch.where(torch.isfinite(nb.stamp), nb.stamp, torch.full_like(nb.stamp, -1e9))
        return nb

    def _send(self, now: float) -> None:
        bc = self.executor.broadcast(now, self.env.sim.p, self.env.sim.v)
        if self.use_packets:
            pk = encode_safety_broadcast(bc)
            if self.packet_loss > 0:
                keep = torch.rand(len(pk), generator=self.g) >= self.packet_loss
                pk = [p for p, k in zip(pk, keep.tolist()) if k]
            self.inflight = pk
        else:
            self.inflight = [bc.quantized()]

    # ------------------------------------------------------------------ main loop
    def run(self, T_max: Optional[float] = None, stop_when_done: bool = True, settle: float = 0.5) -> EpisodeResult:
        env, ex = self.env, self.executor
        Tc = env.safety.T_c
        T_max = env.cfg.T_max if T_max is None else T_max
        n_ticks = int(round(T_max / Tc))
        next_master, next_dec = 0.0, 0.0
        n_master = n_dec = 0
        interventions = braking_ticks = 0
        causes = {"stage_b": 0, "no_chunk": 0, "stale_odom": 0, "stale_neighbor": 0, "neighbor_braking": 0}
        traj, records = [], []
        done_at = None
        sigma_bar = torch.zeros(env.cfg.n_agents, dtype=torch.float64)
        wall0 = time.perf_counter()
        self._send(0.0)
        n = 0
        for n in range(n_ticks):
            s = n * Tc
            nbrs = self._receive(s)
            # master loop
            if self.master is not None and s + 1e-9 >= next_master:
                ups = self.policy.uplinks(s, ex.flags)
                msgs = self.master.cycle(ups, s, self.ids, [1] * len(self.ids))
                self.policy.on_downlink(msgs, s + 0.005)
                if self.expert_policy is not None and self.expert_policy is not self.policy:
                    self.expert_policy.on_downlink(msgs, s + 0.005)
                n_master += 1
                next_master += 1.0 / self.f_master
            # clone loop: decide now, visible after the latency
            if s + 1e-9 >= next_dec:
                out = self.policy.decide(s, nbrs)
                if self.expert_mix is not None and self.expert_policy is not None:
                    e = self.expert_policy.decide(s, nbrs)
                    use_e = torch.rand(env.cfg.n_agents, generator=self.g) < self.expert_mix
                    out = {**out, "A": torch.where(use_e.view(-1, 1, 1), e["A"], out["A"].to(e["A"]))}
                    if self.record_every:
                        out["expert_A"], out["expert_prim"] = e["A"], e["prim"]
                sigma_bar = out["sigma_bar"].to(torch.float64)
                # the first plan of the mission is available at mission start (no start-up fallback)
                vis = s if n == 0 else s + self.latency
                self.pending.append((vis, s, out["A"], out["H_val"]))
                n_dec += 1
                if self.record_every and n_dec % self.record_every == 0:
                    records.append({"t": s, "p": env.sim.p.clone(), "v": env.sim.v.clone(),
                                    "a_bar": ex.last_setpoint.clone(), "flags": ex.flags.clone(),
                                    "A": out.get("expert_A", out["A"]).clone(),
                                    "prim": out.get("expert_prim", out["prim"]).clone()})
                next_dec += 1.0 / self.f_dec
            while self.pending and self.pending[0][0] <= s + 1e-9:
                _, stamp, A, H_val = self.pending.pop(0)
                cm = env.chunk_map()
                ch = Chunk.from_normalized(A.to(torch.float64), cm, YAW_RATE_SCALE,
                                           torch.full((A.shape[0],), stamp, dtype=torch.float64), H_val)
                ex.publish_chunk(torch.ones(A.shape[0], dtype=torch.bool), ch, s)
            # executor tick
            cmd = ex.tick(s, env.sim.p, env.sim.v, nbrs, sigma_bar=sigma_bar)
            interventions += int(((cmd.intervention > 1e-3) & ~cmd.braking).sum())
            braking_ticks += int(cmd.braking.sum())
            for key, bit in (("stage_b", 1), ("no_chunk", 2), ("stale_odom", 4), ("stale_neighbor", 8),
                             ("neighbor_braking", 16)):
                causes[key] += int(((cmd.cause & bit) != 0).sum())
            env.sim.step(cmd.acceleration, cmd.position)
            env.update_metrics()
            self._send(s + Tc)
            if self.keep_traj and n % self.keep_traj == 0:
                traj.append(env.sim.p.clone())
            if stop_when_done and env.all_arrived:
                done_at = done_at if done_at is not None else s
                if s - done_at >= settle:
                    break
        arrival = env.arrival.clone()
        success = env.all_arrived and env.collisions == 0
        makespan = float(arrival.max()) if env.all_arrived else math.nan
        return EpisodeResult(success=success, makespan=makespan, d_min=env.d_min, collisions=env.collisions,
                             arrival=arrival, duration=(n + 1) * Tc, ticks=n + 1, interventions=interventions,
                             fallback_events=len(ex.events), braking_ticks=braking_ticks, causes=causes,
                             n_master_cycles=n_master, n_decisions=getattr(self.policy, "n_decisions", 0),
                             executor_latency=ex.latency_stats(), wall_time=time.perf_counter() - wall0,
                             trajectory=torch.stack(traj) if traj else None, records=records)

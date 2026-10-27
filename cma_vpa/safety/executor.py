# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""100 Hz executor: Algorithm 1 ``TICK`` (Sections 4.6.5, 4.6.6 and 4.7).

Every tick ``s_n`` the executor, for each agent of the batch,

1. evaluates the newest valid chunk at its chunk-local time (skipping knots that
   correspond to inference and transport latency) and blends it with the
   previous chunk over ``T_bl``; the chunk is world-frame (rotated from the
   heading frame frozen at its decision time, see :meth:`Chunk.from_normalized`);
2. forms the nominal acceleration ``u^ = clip((v_nom_{n+1} - v^) / T_c, -u_inf, u_inf)``;
3. builds the reciprocal shares of every neighbour heard within ``R_link``
   (Eq. 27, with the certificates of Lemmas 14-15) and projects ``u^`` by
   Stage B (Eq. 34): closed form (Lemma 18) for one active share, Hildreth with
   the exact polish for several, warm-started from the previous tick;
4. engages the jerk-optimal braking fallback (Eq. 36) if Stage B is infeasible,
   if no valid chunk exists, if odometry or a nearby neighbour's broadcast is
   stale, or if a neighbour brakes, and logs whether the envelope of
   Proposition 9 held at onset;
5. admits a requested effective-radius change (Eq. 35, Section 4.6.6): a
   decrease immediately, an increase only up to the largest value at which
   every heard pair keeps ``psi_0 >= 0`` and ``psi_1 >= T_c max(0, Psi)``;
6. emits the PX4 ``TrajectorySetpoint`` content: velocity ``v^ + T_c u*`` with
   acceleration feed-forward ``u*`` (or the braking reference / hold position)
   and the 22-byte safety broadcast fields.

Semantics fixed here where the paper leaves them open
------------------------------------------------------
* Flags (Table 1, "fallback, braking"): ``FLAG_FALLBACK`` marks a *primary*
  cause in this tick, ``FLAG_BRAKING`` marks that the braking profile (or the
  position hold after it) is being executed. A neighbour triggers sympathetic
  braking if it reports ``FLAG_FALLBACK``, or ``FLAG_BRAKING`` while it still
  moves faster than ``v_hold``; a stopped, holding neighbour does not, so the
  cascade required by Proposition 9 cannot deadlock.
* Release: a braking agent resumes once it has stopped and no trigger has
  been present for ``release_ticks`` consecutive ticks.
* Radius timing: the radius admitted at tick ``n`` is broadcast in tick ``n``'s
  packet and takes effect at tick ``n + 1``, so both agents of a pair switch at
  the same tick when the broadcast latency is at most one tick (Assumption 4).
* Radius search: the "bisection" of Section 4.6.6 is evaluated as a batched
  multisection (``radius_points`` probes per round, ``radius_rounds`` rounds),
  one fused share evaluation per round; with one probe per round it is plain
  bisection. The returned radius always satisfies the admission test.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .degg_flow import ChunkMap
from .fallback import BrakingFallback, fallback_envelope
from .margins import ShareSet, reciprocal_shares
from .params import DEFAULT_PARAMS, SafetyParams
from .qp_solvers import StageBResult, solve_stage_b_qp, stage_b_fixed

__all__ = [
    "FLAG_FALLBACK",
    "FLAG_BRAKING",
    "CAUSE_STAGE_B",
    "CAUSE_NO_CHUNK",
    "CAUSE_STALE_ODOM",
    "CAUSE_STALE_NEIGHBOR",
    "CAUSE_NEIGHBOR_BRAKING",
    "ExecutorConfig",
    "Chunk",
    "ChunkBuffer",
    "NeighborTable",
    "SafetyBroadcast",
    "ExecutorCommand",
    "FallbackEvent",
    "Executor",
    "neighbor_tables_from_broadcasts",
    "requested_radius",
    "px4_trajectory_setpoint",
    "px4_offboard_control_mode",
]

FLAG_FALLBACK = 0x01
FLAG_BRAKING = 0x02

CAUSE_STAGE_B = 0x01
CAUSE_NO_CHUNK = 0x02
CAUSE_STALE_ODOM = 0x04
CAUSE_STALE_NEIGHBOR = 0x08
CAUSE_NEIGHBOR_BRAKING = 0x10


# ---------------------------------------------------------------------------
# configuration and data containers
# ---------------------------------------------------------------------------
@dataclass
class ExecutorConfig:
    """Executor settings (safety parameters come from :class:`SafetyParams`).

    Attributes
    ----------
    H:
        Chunk length (knots spaced ``T_c``).
    T_bl:
        Blending window between consecutive chunks [s].
    odom_timeout, nbr_timeout:
        Age beyond which odometry / a neighbour broadcast is stale [s].
    release_ticks:
        Consecutive trigger-free ticks after stopping before braking is released.
    v_hold:
        Speed below which a braking neighbour counts as holding [m/s].
    radius_rounds, radius_points:
        Multisection search of the radius admission.
    brake_on_neighbor:
        Sympathetic braking (Algorithm 1, line 24, "or a neighbour brakes").
    max_sweeps:
        Hildreth cap of Stage B.
    compile:
        Run shares + Stage B through ``torch.compile`` (fixed neighbour-table
        shape; first tick pays the compilation, see :meth:`Executor.warmup`).
    """

    safety: SafetyParams = DEFAULT_PARAMS
    H: int = 32
    T_bl: float = 0.05
    odom_timeout: float = 0.05
    nbr_timeout: float = 0.05
    release_ticks: int = 10
    v_hold: float = 0.05
    radius_rounds: int = 3
    radius_points: int = 8
    brake_on_neighbor: bool = True
    max_sweeps: int = 50
    compile: bool = False


@dataclass
class Chunk:
    """World-frame chunk as published by a clone decision (batched over agents).

    ``velocity (B, H, 3)`` and ``yaw_rate (B, H)`` at knots ``stamp + k T_c``;
    ``H_val (B,)`` is the validity horizon in knots (Section 4.6.6).
    """

    velocity: Tensor
    yaw_rate: Tensor
    stamp: Tensor
    H_val: Tensor

    @staticmethod
    def from_normalized(A: Tensor, chunk_map: ChunkMap, yaw_rate_scale: float, stamp: Tensor,
                        H_val: Tensor, yaw_rate_offset: float = 0.0) -> "Chunk":
        """Denormalize a solver chunk ``A (B, H, 4)`` (heading frame) and rotate it to the world frame."""
        vel = chunk_map.to_world(A[..., :3])
        yr = yaw_rate_scale * A[..., 3] + yaw_rate_offset
        return Chunk(vel, yr, stamp, H_val)


class ChunkBuffer:
    """Double-buffered single-slot chunk store (writer and reader never wait).

    ``publish`` writes into the back slot of the selected agents and then flips
    their front index; the reader always sees a complete chunk. The previous
    front chunk is kept for blending together with the switch time.
    """

    def __init__(self, n_agents: int, H: int, dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        self.vel = torch.zeros(2, n_agents, H, 3, dtype=dtype, device=device)
        self.yaw = torch.zeros(2, n_agents, H, dtype=dtype, device=device)
        self.stamp = torch.full((2, n_agents), -math.inf, dtype=dtype, device=device)
        self.H_val = torch.zeros(2, n_agents, dtype=torch.long, device=device)
        self.valid = torch.zeros(2, n_agents, dtype=torch.bool, device=device)
        self.front = torch.zeros(n_agents, dtype=torch.long, device=device)
        self.switch_time = torch.full((n_agents,), -math.inf, dtype=dtype, device=device)
        self._ar = torch.arange(n_agents, device=device)

    def publish(self, mask: Tensor, chunk: Chunk, now: float) -> None:
        """Store ``chunk`` for the agents in ``mask`` and make it current."""
        back = 1 - self.front
        sel = mask.nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return
        b = back[sel]
        self.vel[b, sel] = chunk.velocity[sel].to(self.vel.dtype)
        self.yaw[b, sel] = chunk.yaw_rate[sel].to(self.yaw.dtype)
        self.stamp[b, sel] = chunk.stamp[sel].to(self.stamp.dtype)
        self.H_val[b, sel] = chunk.H_val[sel].to(torch.long)
        self.valid[b, sel] = True
        self.front[sel] = b                      # the flip
        self.switch_time[sel] = now

    def slot(self, which: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Gather ``(vel, yaw, stamp, H_val, valid)`` of slot ``which[i]`` for every agent."""
        ar = self._ar
        return (self.vel[which, ar], self.yaw[which, ar], self.stamp[which, ar],
                self.H_val[which, ar], self.valid[which, ar])


@dataclass
class NeighborTable:
    """Latest neighbour broadcasts as seen by each agent, ``(B, N)`` slots.

    Slots must keep a fixed neighbour identity between ticks so that the
    Stage-B warm start stays aligned.
    """

    pos: Tensor
    vel: Tensor
    radius: Tensor
    flags: Tensor
    stamp: Tensor
    present: Tensor
    ids: Tensor


@dataclass
class SafetyBroadcast:
    """Fields of the 22-byte safety packet (Table 1) for every agent of the batch."""

    agent_id: Tensor
    seq: Tensor
    stamp: Tensor
    pos: Tensor
    vel: Tensor
    radius: Tensor
    flags: Tensor

    def quantized(self) -> "SafetyBroadcast":
        """Apply the wire resolution of Table 1: int16 cm position, int16 cm/s velocity, uint8 cm radius.

        The rounding error (at most ``0.5 sqrt(3)`` cm and cm/s per agent) is
        part of the estimation bounds ``e_p``, ``e_v`` of Assumption 2.
        """
        lim = 32767.0
        pos = torch.clamp(torch.round(self.pos * 100.0), -lim, lim) / 100.0
        vel = torch.clamp(torch.round(self.vel * 100.0), -lim, lim) / 100.0
        rad = torch.clamp(torch.ceil(self.radius * 100.0), 0.0, 255.0) / 100.0   # never under-report
        return SafetyBroadcast(self.agent_id, self.seq % 256, self.stamp, pos, vel, rad, self.flags)


def neighbor_tables_from_broadcasts(bc: SafetyBroadcast, receive_time: Optional[Tensor] = None) -> NeighborTable:
    """All-others neighbour tables for a simulated team: agent ``i`` sees every ``j != i``.

    Slot ``k`` of agent ``i`` holds agent ``k`` if ``k < i`` and ``k + 1``
    otherwise, which keeps identities fixed. ``receive_time`` (optional, per
    sender) overrides the stamp (to model transport latency).
    """
    B = bc.pos.shape[0]
    dev = bc.pos.device
    ar = torch.arange(B, device=dev)
    k = torch.arange(B - 1, device=dev)
    others = k.unsqueeze(0) + (k.unsqueeze(0) >= ar.unsqueeze(1)).long()      # (B, B-1)
    stamp = bc.stamp if receive_time is None else receive_time
    return NeighborTable(pos=bc.pos[others], vel=bc.vel[others], radius=bc.radius[others],
                         flags=bc.flags[others], stamp=stamp[others],
                         present=torch.ones(B, B - 1, dtype=torch.bool, device=dev),
                         ids=bc.agent_id[others])


@dataclass
class FallbackEvent:
    """Log record of a fallback onset (protocol readout of Section 7.4)."""

    agent: int
    time: float
    cause: int
    certified: bool
    min_margin: float
    n_pairs: int


@dataclass
class ExecutorCommand:
    """Executor output for one tick (all batched over agents).

    ``position`` is NaN except while holding after a completed braking profile
    (PX4 ignores NaN fields of ``TrajectorySetpoint``).
    """

    time: float
    velocity: Tensor
    acceleration: Tensor
    yaw_rate: Tensor
    position: Tensor
    braking: Tensor
    flags: Tensor
    cause: Tensor
    radius: Tensor
    radius_next: Tensor
    n_active: Tensor
    stage_b_ok: Tensor
    intervention: Tensor
    u_hat: Tensor
    u_star: Tensor
    tick_ms: float
    shares: Optional[ShareSet] = None


def requested_radius(sigma_bar: Tensor, cfg: SafetyParams = DEFAULT_PARAMS) -> Tensor:
    """Eq. (35): ``r_eff = min(r_max, r_s exp(kappa_sigma sigma_bar))``."""
    return torch.clamp(cfg.r_s * torch.exp(cfg.kappa_sigma * sigma_bar), max=cfg.r_max)


# ---------------------------------------------------------------------------
# compiled kernel (fixed shapes)
# ---------------------------------------------------------------------------
def _knot_kernel(dp_hat: Tensor, dv_hat: Tensor, D: Tensor, heard: Tensor, u_hat: Tensor,
                 lam0: Tensor, cfg: SafetyParams, max_sweeps: int):
    """Fixed-shape tick kernel: fused shares + Lemma 18 (the at-most-one-active case).

    Elements with several active shares are flagged ``optimal = False`` and
    finished by the exact eager path (warm-started Hildreth + enumeration).
    """
    S = reciprocal_shares(dp_hat, dv_hat, D, cfg, heard=heard)
    r = stage_b_fixed(u_hat, S.g, S.zeta, S.active, cfg.u_inf, lam0, multi=False)
    return S, r


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------
class Executor:
    """Batched 100 Hz executor implementing Algorithm 1 ``TICK``.

    Parameters
    ----------
    n_agents:
        Batch size ``B`` (1 on a vehicle; the whole team in simulation).
    n_neighbors:
        Neighbour-table width ``N`` (fixed, so that compiled kernels and warm
        starts keep their shapes).
    config:
        :class:`ExecutorConfig`.
    agent_ids:
        Identifiers written into the safety broadcast (default ``0..B-1``).
    """

    def __init__(self, n_agents: int, n_neighbors: int, config: ExecutorConfig = ExecutorConfig(),
                 agent_ids: Optional[Tensor] = None, dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None):
        self.B, self.N = n_agents, n_neighbors
        self.config = config
        self.cfg = config.safety
        self.dtype, self.device = dtype, device
        d = self.cfg.d
        self.buffer = ChunkBuffer(n_agents, config.H, dtype, device)
        self.fallback = BrakingFallback(n_agents, d, self.cfg.a_brk, dtype, device)
        self.radius = torch.full((n_agents,), self.cfg.r_s, dtype=dtype, device=device)
        self.lam = torch.zeros(n_agents, n_neighbors + 2 * d, dtype=dtype, device=device)
        self.clear_count = torch.zeros(n_agents, dtype=torch.long, device=device)
        self.flags = torch.zeros(n_agents, dtype=torch.uint8, device=device)
        self.seq = torch.zeros(n_agents, dtype=torch.long, device=device)
        self.agent_ids = torch.arange(n_agents, device=device) if agent_ids is None else agent_ids
        self.last_setpoint = torch.zeros(n_agents, d + 1, dtype=dtype, device=device)
        self.events: List[FallbackEvent] = []
        self.tick_ms: List[float] = []
        self._kernel = torch.compile(_knot_kernel, dynamic=False) if config.compile else None

    # ------------------------------------------------------------------ chunks
    def publish_chunk(self, mask: Tensor, chunk: Chunk, now: float) -> None:
        """Make ``chunk`` the newest chunk of the agents in ``mask`` (called by the clone loop)."""
        self.buffer.publish(mask, chunk, now)

    def _eval_slot(self, which: Tensor, s: float) -> Tuple[Tensor, Tensor, Tensor]:
        vel, yaw, stamp, H_val, valid = self.buffer.slot(which)
        Tc, H = self.cfg.T_c, vel.shape[-2]
        ds = s - stamp
        ok = valid & (ds >= 0) & (ds <= H_val.to(ds.dtype) * Tc)
        t = torch.clamp(torch.where(ok, ds, torch.zeros_like(ds)) / Tc, 0.0, H - 1.0)
        i0 = torch.floor(t).long()
        i1 = torch.clamp(i0 + 1, max=H - 1)
        w = (t - i0.to(t.dtype))
        ar = torch.arange(self.B, device=vel.device)
        v = (1 - w).unsqueeze(-1) * vel[ar, i0] + w.unsqueeze(-1) * vel[ar, i1]
        y = (1 - w) * yaw[ar, i0] + w * yaw[ar, i1]
        return v, y, ok

    def nominal_velocity(self, s_eval: float, s_valid: float) -> Tuple[Tensor, Tensor, Tensor]:
        """Blended chunk velocity and yaw rate at ``s_eval``; validity of the newest chunk at ``s_valid``."""
        front = self.buffer.front
        v_new, y_new, _ = self._eval_slot(front, s_eval)
        _, _, valid_now = self._eval_slot(front, s_valid)
        v_old, y_old, ok_old = self._eval_slot(1 - front, s_eval)
        w = torch.clamp((s_eval - self.buffer.switch_time) / max(self.config.T_bl, 1e-9), 0.0, 1.0)
        blend = ok_old & (w < 1.0)
        wv = torch.where(blend, w, torch.ones_like(w))
        v = (1 - wv).unsqueeze(-1) * v_old + wv.unsqueeze(-1) * v_new
        y = (1 - wv) * y_old + wv * y_new
        return v, y, valid_now

    # ------------------------------------------------------------- Stage B
    def _project(self, dp_hat: Tensor, dv_hat: Tensor, D: Tensor, heard: Tensor,
                 u_hat: Tensor) -> Tuple[ShareSet, StageBResult]:
        if self._kernel is not None:
            S, r = self._kernel(dp_hat, dv_hat, D, heard, u_hat, self.lam, self.cfg, self.config.max_sweeps)
            need = (r.n_active >= 2) & ~r.optimal
            if bool(need.any()):                       # rare: finish exactly in eager mode
                sel = need.nonzero(as_tuple=True)[0]
                re = solve_stage_b_qp(u_hat[sel], S.g[sel], S.zeta[sel], S.active[sel], self.cfg.u_inf,
                                      lam0=self.lam[sel], max_sweeps=self.config.max_sweeps)
                u, ok, lam, opt = r.u.clone(), r.ok.clone(), r.lam.clone(), r.optimal.clone()
                u[sel], ok[sel], lam[sel], opt[sel] = re.u, re.ok, re.lam, re.optimal
                r = StageBResult(u=u, ok=ok, lam=lam, n_active=r.n_active, method=r.method,
                                 sweeps=max(r.sweeps, re.sweeps), max_violation=r.max_violation,
                                 optimal=opt)
            return S, r
        S = reciprocal_shares(dp_hat, dv_hat, D, self.cfg, heard=heard)
        r = solve_stage_b_qp(u_hat, S.g, S.zeta, S.active, self.cfg.u_inf, lam0=self.lam,
                             max_sweeps=self.config.max_sweeps)
        return S, r

    # ------------------------------------------------------------- radius
    def _admissible(self, r_cand: Tensor, r_nbr: Tensor, dp_hat: Tensor, dv_hat: Tensor,
                    heard: Tensor) -> Tensor:
        """Admission test for candidate radii ``r_cand (B, C)``; returns ``(B, C)``."""
        C = r_cand.shape[-1]
        D = r_cand.unsqueeze(-1) + r_nbr.unsqueeze(1)                       # (B, C, N)
        S = reciprocal_shares(dp_hat.unsqueeze(1).expand(-1, C, -1, -1),
                              dv_hat.unsqueeze(1).expand(-1, C, -1, -1), D, self.cfg,
                              heard=heard.unsqueeze(1).expand(-1, C, -1))
        return (S.admission | ~heard.unsqueeze(1)).all(-1)

    def admit_radius(self, r_req: Tensor, r_nbr: Tensor, dp_hat: Tensor, dv_hat: Tensor,
                     heard: Tensor) -> Tensor:
        """Section 4.6.6: decreases at once, increases to the largest admissible value.

        The search keeps the invariant "the lower bracket end is admissible"
        (it starts at the current radius, which must itself pass the test for any
        increase to be admitted), so the result always satisfies the condition.
        """
        r_cur = self.radius
        r_req = torch.clamp(r_req, self.cfg.r_s, self.cfg.r_max)
        out = torch.where(r_req <= r_cur, r_req, r_cur)
        inc = r_req > r_cur
        if not bool(inc.any()):
            return out
        ends = self._admissible(torch.stack([r_cur, r_req], -1), r_nbr, dp_hat, dv_hat, heard)
        full = inc & ends[:, 1]
        out = torch.where(full, r_req, out)
        search = inc & ends[:, 0] & ~ends[:, 1]
        if bool(search.any()):
            lo, hi = r_cur.clone(), r_req.clone()
            P = max(self.config.radius_points, 1)
            frac = torch.arange(1, P + 1, dtype=self.dtype, device=lo.device) / (P + 1)
            for _ in range(max(self.config.radius_rounds, 1)):
                cand = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * frac          # (B, P)
                adm = self._admissible(cand, r_nbr, dp_hat, dv_hat, heard)
                lead = torch.cumprod(adm.to(torch.int8), dim=-1).sum(-1)       # leading admissible probes
                idx_lo = torch.clamp(lead - 1, min=0)
                new_lo = torch.where(lead > 0, cand.gather(-1, idx_lo.unsqueeze(-1)).squeeze(-1), lo)
                new_hi = torch.where(lead < P, cand.gather(-1, torch.clamp(lead, max=P - 1).unsqueeze(-1)).squeeze(-1), hi)
                lo = torch.where(search, new_lo, lo)
                hi = torch.where(search, new_hi, hi)
            out = torch.where(search, lo, out)
        return out

    # ------------------------------------------------------------------ tick
    def tick(self, s_n: float, p_hat: Tensor, v_hat: Tensor, nbrs: NeighborTable, *,
             odom_stamp: Optional[Tensor] = None, sigma_bar: Optional[Tensor] = None,
             keep_shares: bool = False) -> ExecutorCommand:
        """One executor period for every agent (Algorithm 1, lines 22-34).

        Parameters
        ----------
        s_n:
            Tick time [s].
        p_hat, v_hat:
            Own position and velocity estimates (world frame), ``(B, 3)``.
        nbrs:
            Neighbour table ``(B, N)`` from the latest safety broadcasts.
        odom_stamp:
            Time of the odometry sample (default ``s_n``).
        sigma_bar:
            Filtered ensemble dispersion ``(B,)`` for the radius request
            (Eq. 35); ``None`` keeps the current radius.
        keep_shares:
            Attach the :class:`ShareSet` to the command (diagnostics).
        """
        t0 = time.perf_counter()
        cfg, ec = self.cfg, self.config
        Tc = cfg.T_c
        dt_ = self.dtype

        # -- odometry and chunk --------------------------------------------
        stale_odom = torch.zeros(self.B, dtype=torch.bool, device=p_hat.device) if odom_stamp is None \
            else (s_n - odom_stamp) > ec.odom_timeout
        v_nom, yaw_nom, chunk_ok = self.nominal_velocity(s_n + Tc, s_n)

        # -- neighbours ------------------------------------------------------
        age = s_n - nbrs.stamp
        p_j = nbrs.pos + nbrs.vel * age.unsqueeze(-1)                # constant-velocity extrapolation
        dp_hat = p_hat.unsqueeze(1) - p_j
        dv_hat = v_hat.unsqueeze(1) - nbrs.vel
        dist = torch.linalg.vector_norm(dp_hat, dim=-1)
        near = nbrs.present & (dist <= cfg.R_link)
        fresh = age <= ec.nbr_timeout
        heard = near & fresh
        stale_nbr = (near & ~fresh).any(-1)
        flags_j = nbrs.flags.to(torch.int32)
        moving_j = torch.linalg.vector_norm(nbrs.vel, dim=-1) > ec.v_hold
        nbr_brakes = heard & (((flags_j & FLAG_FALLBACK) != 0) | (((flags_j & FLAG_BRAKING) != 0) & moving_j))
        D = self.radius.unsqueeze(-1) + nbrs.radius

        # -- nominal input and Stage B -----------------------------------------
        braking_before = self.fallback.active
        u_hat = torch.clamp((v_nom - v_hat) / Tc, -cfg.u_inf, cfg.u_inf)
        u_hat = torch.where((chunk_ok & ~braking_before).unsqueeze(-1), u_hat, torch.zeros_like(u_hat))
        S, r = self._project(dp_hat, dv_hat, D, heard, u_hat)
        self.lam = r.lam.to(dt_)

        # -- fallback triggers (Algorithm 1, lines 24 and 29) ----------------
        cause = torch.zeros(self.B, dtype=torch.long, device=p_hat.device)
        cause |= torch.where(~r.ok, CAUSE_STAGE_B, 0)
        cause |= torch.where(~chunk_ok, CAUSE_NO_CHUNK, 0)
        cause |= torch.where(stale_odom, CAUSE_STALE_ODOM, 0)
        cause |= torch.where(stale_nbr, CAUSE_STALE_NEIGHBOR, 0)
        primary = cause != 0
        sympathetic = nbr_brakes.any(-1) & ec.brake_on_neighbor
        cause |= torch.where(sympathetic, CAUSE_NEIGHBOR_BRAKING, 0)
        want = cause != 0

        new = self.fallback.engage(want, p_hat, v_hat, s_n)
        if bool(new.any()):
            self._log_onsets(new, s_n, cause, dp_hat, dv_hat, D, heard)
        stopped = self.fallback.stopped(s_n)
        clear = self.fallback.active & stopped & ~want
        self.clear_count = torch.where(clear, self.clear_count + 1, torch.zeros_like(self.clear_count))
        release = self.fallback.active & (self.clear_count >= ec.release_ticks)
        self.fallback.release(release)
        self.clear_count = torch.where(release, torch.zeros_like(self.clear_count), self.clear_count)
        braking = self.fallback.active

        # -- radius admission (takes effect next tick) -------------------------
        r_next = self.radius if sigma_bar is None else \
            self.admit_radius(requested_radius(sigma_bar, cfg), nbrs.radius, dp_hat, dv_hat, heard)

        # -- outputs -----------------------------------------------------------
        nan3 = torch.full_like(v_hat, math.nan)
        brk = self.fallback.command(s_n + Tc)
        hold = braking & brk.stopped
        vel = torch.where(braking.unsqueeze(-1), brk.velocity, v_hat + Tc * r.u)
        acc = torch.where(braking.unsqueeze(-1), brk.acceleration, r.u)
        yaw = torch.where(braking, torch.zeros_like(yaw_nom), yaw_nom)
        pos = torch.where(hold.unsqueeze(-1), self.fallback.hold_position(), nan3)
        flags = (torch.where(braking, FLAG_BRAKING, 0) | torch.where(braking & primary, FLAG_FALLBACK, 0))
        self.flags = flags.to(torch.uint8)
        self.last_setpoint = torch.cat([vel, yaw.unsqueeze(-1)], -1)
        radius_now = self.radius
        self.radius = r_next
        self.seq = self.seq + 1
        ms = (time.perf_counter() - t0) * 1e3
        self.tick_ms.append(ms)
        return ExecutorCommand(time=s_n, velocity=vel, acceleration=acc, yaw_rate=yaw, position=pos,
                               braking=braking, flags=self.flags, cause=cause, radius=radius_now,
                               radius_next=r_next, n_active=r.n_active, stage_b_ok=r.ok,
                               intervention=torch.linalg.vector_norm(r.u - u_hat, dim=-1),
                               u_hat=u_hat, u_star=r.u, tick_ms=ms, shares=S if keep_shares else None)

    def _log_onsets(self, new: Tensor, s_n: float, cause: Tensor, dp_hat: Tensor, dv_hat: Tensor,
                    D: Tensor, heard: Tensor) -> None:
        rho = torch.linalg.vector_norm(dp_hat, dim=-1)
        need = fallback_envelope(torch.linalg.vector_norm(dv_hat, dim=-1), D, self.cfg)
        margin = torch.where(heard, rho - need, torch.full_like(rho, math.inf))
        for i in new.nonzero(as_tuple=True)[0].tolist():
            n_pairs = int(heard[i].sum())
            mm = float(margin[i].min()) if n_pairs else math.inf
            self.events.append(FallbackEvent(agent=int(self.agent_ids[i]), time=float(s_n),
                                             cause=int(cause[i]), certified=mm >= 0.0,
                                             min_margin=mm, n_pairs=n_pairs))

    # ------------------------------------------------------------- broadcast
    def broadcast(self, s_n: float, p_hat: Tensor, v_hat: Tensor) -> SafetyBroadcast:
        """Safety packet fields for this tick (radius = the value in effect from the next tick)."""
        stamp = torch.full((self.B,), float(s_n), dtype=self.dtype, device=p_hat.device)
        return SafetyBroadcast(agent_id=self.agent_ids, seq=self.seq.clone(), stamp=stamp, pos=p_hat.clone(),
                               vel=v_hat.clone(), radius=self.radius.clone(), flags=self.flags.clone())

    def warmup(self, p_hat: Tensor, v_hat: Tensor, nbrs: NeighborTable) -> float:
        """Run the Stage-B path once without side effects (triggers compilation); returns seconds."""
        t0 = time.perf_counter()
        age = torch.zeros_like(nbrs.stamp)
        dp_hat = p_hat.unsqueeze(1) - nbrs.pos
        dv_hat = v_hat.unsqueeze(1) - nbrs.vel
        self._project(dp_hat, dv_hat, self.radius.unsqueeze(-1) + nbrs.radius, nbrs.present & (age >= 0),
                      torch.zeros_like(v_hat))
        return time.perf_counter() - t0

    def latency_stats(self) -> Dict[str, float]:
        """Percentiles of the measured tick time [ms]."""
        if not self.tick_ms:
            return {}
        t = torch.tensor(self.tick_ms, dtype=torch.float64)
        return {"mean": float(t.mean()), "p50": float(t.quantile(0.5)), "p95": float(t.quantile(0.95)),
                "p99": float(t.quantile(0.99)), "max": float(t.max())}


# ---------------------------------------------------------------------------
# PX4 message content (px4_msgs/TrajectorySetpoint, OffboardControlMode)
# ---------------------------------------------------------------------------
def _enu_to_ned(v: List[float]) -> List[float]:
    return [v[1], v[0], -v[2]]


def px4_trajectory_setpoint(cmd: ExecutorCommand, agent: int, timestamp_us: int,
                            enu_to_ned: bool = True) -> Dict[str, object]:
    """Fields of ``px4_msgs/msg/TrajectorySetpoint`` for one agent (NaN = unused).

    The executor works in an ENU world frame; PX4 expects NED, so by default
    ``x_ned = y_enu``, ``y_ned = x_enu``, ``z_ned = -z_enu`` and the yaw rate
    changes sign.
    """
    pos = cmd.position[agent].tolist()
    vel = cmd.velocity[agent].tolist()
    acc = cmd.acceleration[agent].tolist()
    yr = float(cmd.yaw_rate[agent])
    if enu_to_ned:
        pos, vel, acc, yr = _enu_to_ned(pos), _enu_to_ned(vel), _enu_to_ned(acc), -yr
    return {"timestamp": int(timestamp_us), "position": pos, "velocity": vel, "acceleration": acc,
            "jerk": [math.nan] * 3, "yaw": math.nan, "yawspeed": yr}


def px4_offboard_control_mode(cmd: ExecutorCommand, agent: int, timestamp_us: int) -> Dict[str, object]:
    """Fields of ``px4_msgs/msg/OffboardControlMode`` (the proof-of-life message, sent every tick)."""
    holding = not math.isnan(float(cmd.position[agent, 0]))
    return {"timestamp": int(timestamp_us), "position": holding, "velocity": True, "acceleration": True,
            "attitude": False, "body_rate": False, "thrust_and_torque": False, "direct_actuator": False}

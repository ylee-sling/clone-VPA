# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Expert data for the staged training pipeline: collection, storage and batch samplers.

Episodes of Task T1 are rolled out with the privileged expert
(:class:`~cma_vpa.envs.ExpertChunkPolicy`) behind the real executor (Stage B and
fallback active), with a :class:`~cma_vpa.envs.ModeMaster` that samples the
team's passing handedness. Only states are stored - frames are rendered on the
fly (:func:`render_frames`), so a dataset of many episodes stays small.

Per decision ``t`` and vehicle ``i`` an episode stores the position, velocity,
the last dispatched setpoint ``a_bar`` (chunk-normalized), the executor flags,
the expert chunk label ``A (H, 4)`` and the primitive label; per episode the
goals, headings and the rule-based mode labels (Section 4.8: passing
handedness ``u_g`` and per-vehicle side ``u_M``).

Run ``python -m cma_vpa.training.data --out t1_expert.pt --episodes 40`` to
collect a dataset file.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..envs import (V_SCALE, YAW_RATE_SCALE, AntipodalCrossing, CrossingConfig, CrossingRunner,
                    ExpertChunkPolicy, ModeMaster, handedness_label, local_mode_labels, render_rgbd)
from ..safety.degg_flow import ChunkMap
from ..safety.executor import ExecutorConfig

__all__ = ["Episode", "collect_episodes", "episode_from_records", "rel_goal_batch", "render_frames", "proprio_batch", "save_episodes", "load_episodes",
           "INSTRUCTIONS", "Sampler"]

INSTRUCTIONS = (
    "cross the conflict zone and reach the antipodal point",
    "swap sides with the vehicle opposite you",
    "fly through the centre to the far side of the circle",
)


@dataclass
class Episode:
    p: Tensor          # (T, N, 3) float64
    v: Tensor          # (T, N, 3) float64
    a_bar: Tensor      # (T, N, 4) float32, chunk-normalized last setpoint
    flags: Tensor      # (T, N) uint8
    A: Tensor          # (T, N, H, 4) float32 expert chunk labels
    prim: Tensor       # (T, N) int64
    goals: Tensor      # (N, 3)
    yaw: Tensor        # (N,)
    u_g: int
    u_M: Tensor        # (N,)
    dt: float          # decision period [s]
    r_v: float
    text: str

    @property
    def T(self) -> int:
        return self.p.shape[0]

    @property
    def N(self) -> int:
        return self.p.shape[1]


def _normalize_setpoint(sp: Tensor, yaw: Tensor) -> Tensor:
    cm = ChunkMap.from_yaw(yaw, scale=V_SCALE)
    vel = cm.to_normalized(sp[:, None, :3].to(torch.float64))[:, 0]
    return torch.cat([vel, sp[:, 3:4] / YAW_RATE_SCALE], -1).float().clamp(-1, 1)


def episode_from_records(records: List[dict], env: AntipodalCrossing, f_dec: float, text: str) -> Optional[Episode]:
    """Assemble an :class:`Episode` from :class:`~cma_vpa.envs.CrossingRunner` records (expert labels)."""
    if len(records) < 4:
        return None
    p = torch.stack([r["p"] for r in records])
    v = torch.stack([r["v"] for r in records])
    a_bar = torch.stack([_normalize_setpoint(r["a_bar"], env.yaw) for r in records])
    return Episode(p=p, v=v, a_bar=a_bar, flags=torch.stack([r["flags"] for r in records]),
                   A=torch.stack([r["A"] for r in records]).float(),
                   prim=torch.stack([r["prim"] for r in records]).long(),
                   goals=env.goals.clone(), yaw=env.yaw.clone(),
                   u_g=int(handedness_label(p, v, radius=10.0)), u_M=local_mode_labels(p, v, radius=10.0),
                   dt=1.0 / f_dec, r_v=env.cfg.r_v, text=text)


def collect_episodes(n_episodes: int, n_agents: Sequence[int] = (4, 8), wind: Sequence[str] = ("W0", "W1"),
                     H: int = 32, T_max: float = 30.0, f_dec: float = 30.0, seed: int = 0,
                     v_h: Sequence[float] = (5.0,), angle_jitter_deg: float = 4.0,
                     verbose: bool = False) -> List[Episode]:
    """Roll out ``n_episodes`` expert episodes (team size, wind and speed drawn per episode)."""
    rng = random.Random(seed)
    eps: List[Episode] = []
    for k in range(n_episodes):
        N = rng.choice(list(n_agents))
        cfg = CrossingConfig(n_agents=N, wind=rng.choice(list(wind)), v_h=rng.choice(list(v_h)),
                             T_max=T_max, seed=seed * 1000 + k, angle_jitter_deg=angle_jitter_deg)
        env = AntipodalCrossing(cfg)
        g = torch.Generator().manual_seed(seed * 7919 + k)
        master = ModeMaster(N, g)
        pol = ExpertChunkPolicy(env, H=H, mode_source="master", default_h=1 if master.u_g % 2 == 0 else -1)
        runner = CrossingRunner(env, pol, master, f_dec=f_dec, record_every=1, generator=g,
                                executor_cfg=ExecutorConfig(safety=env.safety, H=H, v_hold=0.25))
        res = runner.run()
        ep = episode_from_records(res.records, env, f_dec, rng.choice(INSTRUCTIONS))
        if ep is None:
            continue
        eps.append(ep)
        if verbose:
            print(f"episode {k}: N={N} wind={cfg.wind} success={res.success} makespan={res.makespan:.2f} "
                  f"u_g={ep.u_g} (master {master.u_g}) T={ep.T}")
    return eps


def render_frames(ep: Episode, t: Tensor, image_size: Tuple[int, int]) -> Tensor:
    """Frames of every vehicle at decision indices ``t (K,)``: ``(K, N, 4, H, W)``."""
    K, N = t.shape[0], ep.N
    p = ep.p[t]                                                        # (K, N, 3)
    idx = torch.tensor([[j for j in range(N) if j != i] for i in range(N)], dtype=torch.long)
    others = p[:, idx] if N > 1 else torch.zeros(K, N, 0, 3, dtype=p.dtype)
    flat_p = p.reshape(K * N, 3)
    flat_o = others.reshape(K * N, max(N - 1, 0), 3)
    rad = torch.full(flat_o.shape[:2], ep.r_v)
    yaw = ep.yaw.repeat(K)
    goals = ep.goals.repeat(K, 1)
    fr = render_rgbd(flat_p, yaw, flat_o, rad, goals, image_size=image_size)
    return fr.reshape(K, N, *fr.shape[1:])


def proprio_batch(ep: Episode, t: Tensor, sigma: Optional[Tensor] = None) -> Tensor:
    """``S_t (K, N, 13)`` (see :func:`~cma_vpa.envs.proprio_state`)."""
    from ..envs import proprio_state

    K, N = t.shape[0], ep.N
    v = ep.v[t]
    c, s = torch.cos(ep.yaw), torch.sin(ep.yaw)
    vh = torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)
    dz = ep.p[t][..., 2] - ep.goals[:, 2]
    sig = torch.zeros(K, N) if sigma is None else sigma
    out = proprio_state(vh.reshape(K * N, 3).float(), ep.a_bar[t].reshape(K * N, 4), dz.reshape(-1).float(),
                        ep.yaw.repeat(K).float(), sig.reshape(-1).float(), ep.flags[t].reshape(-1))
    return out.reshape(K, N, 13)


def rel_goal_batch(ep: Episode, t: Tensor) -> Tensor:
    """Heading-frame goal offsets ``(K, N, 3)``."""
    d = ep.goals - ep.p[t]
    c, s = torch.cos(ep.yaw), torch.sin(ep.yaw)
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], -1).float()


class Sampler:
    """Random decision indices over a list of episodes (``(episode, t)`` pairs, team-level)."""

    def __init__(self, episodes: List[Episode], min_future: int = 0, seed: int = 0):
        self.eps = [e for e in episodes if e.T > min_future + 1]
        if not self.eps:
            raise ValueError("no episode is long enough for the requested horizon")
        self.min_future = min_future
        self.g = torch.Generator().manual_seed(seed)

    def team(self, n_snapshots: int) -> List[Tuple[Episode, Tensor]]:
        """``n_snapshots`` (episode, t-vector) groups; t is drawn so that ``t + min_future < T``."""
        out = []
        for _ in range(n_snapshots):
            e = self.eps[int(torch.randint(0, len(self.eps), (1,), generator=self.g))]
            t = torch.randint(0, e.T - self.min_future, (1,), generator=self.g)
            out.append((e, t))
        return out


def save_episodes(eps: List[Episode], path: str) -> None:
    torch.save([asdict(e) for e in eps], path)


def load_episodes(path: str) -> List[Episode]:
    return [Episode(**d) for d in torch.load(path, weights_only=False)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Collect T1 expert episodes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--agents", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--wind", nargs="+", default=["W0", "W1", "W2"])
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--t-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    eps = collect_episodes(a.episodes, a.agents, a.wind, a.horizon, a.t_max, seed=a.seed, verbose=True)
    save_episodes(eps, a.out)
    print(f"saved {len(eps)} episodes to {a.out}")


if __name__ == "__main__":
    main()

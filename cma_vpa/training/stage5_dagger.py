# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 5 (Section 4.8 (v)): dataset aggregation (DAgger [81]) with privileged expert relabelling.

Each iteration ``k`` rolls the whole team out in simulation (master, clones,
executor with Stage B and the fallback, all through the latent bus and the
safety channel) under a mixture of its own and the expert's commands: at
every decision each vehicle executes the expert chunk with probability
``beta_k = beta_0 * decay^k`` and its own chunk otherwise. The privileged
expert relabels every visited state with its chunk and primitive (with the
handedness that the master broadcast, read from the downlink mode token); the
relabelled states are aggregated with the data of previous iterations, and
the Stage-4 objective is optimized on the aggregate. The barrier acts only in
the rollouts, never in the loss.

Usage::

    python -m cma_vpa.training.stage5_dagger --init ckpt/stage4.pt --iters 5 --rollouts 4 --out ckpt/stage5.pt
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional

import torch

from ..envs import (AntipodalCrossing, CrossingConfig, CrossingRunner, ExpertChunkPolicy, LearnedChunkPolicy,
                    LearnedMaster)
from ..models import HashedTextEmbedder
from ..safety.executor import ExecutorConfig
from .common import add_common_args, episodes_from_args, freeze, pick_device, save_checkpoint, seed_all, team_batch
from .data import INSTRUCTIONS, Episode, Sampler, episode_from_records
from .stage4_policy import PolicyTrainer, load_stage3

__all__ = ["build_parser", "rollout", "run", "main"]


def build_parser() -> argparse.ArgumentParser:
    ap = add_common_args(argparse.ArgumentParser(description="Stage 5: DAgger with expert relabelling"))
    ap.add_argument("--init", default="checkpoints/stage4.pt", help="stage-4 checkpoint")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=4, help="team rollouts per iteration")
    ap.add_argument("--beta0", type=float, default=0.5, help="initial expert mixing probability")
    ap.add_argument("--beta-decay", type=float, default=0.5)
    ap.add_argument("--rollout-t-max", type=float, default=20.0)
    ap.add_argument("--lambda-h", type=float, default=1.0)
    ap.add_argument("--plan-dropout", type=float, default=0.1)
    ap.add_argument("--keep-initial", action="store_true", help="aggregate with the initial expert data")
    ap.set_defaults(out="checkpoints/stage5.pt", steps=500, lr=1e-4)
    return ap


@torch.no_grad()
def rollout(clone, master, *, n_agents: int, wind: str, seed: int, beta: float, T_max: float, f_dec: float,
            text: str) -> Episode:
    """One closed-loop team rollout with expert mixing ``beta``; returns the relabelled episode."""
    cfg = clone.cfg
    env = AntipodalCrossing(CrossingConfig(n_agents=n_agents, wind=wind, T_max=T_max, seed=seed,
                                           image_size=cfg.image_size, angle_jitter_deg=4.0))
    g = torch.Generator().manual_seed(seed)
    c_text = HashedTextEmbedder(cfg.text_dim)(text).to(clone.mu_z.device)
    clone.reset_mission(c_text, milestone_frames=env.milestone_frames(cfg.image_size).to(clone.mu_z.device))
    lm = LearnedMaster(master, clone.rvq, clone.c_tilde, clone.tracker.active(clone.milestones), generator=g)
    pol = LearnedChunkPolicy(env, clone, generator=g)
    expert = ExpertChunkPolicy(env, H=cfg.horizon, mode_source="master")
    runner = CrossingRunner(env, pol, lm, f_dec=f_dec, record_every=1, generator=g, expert_mix=beta,
                            expert_policy=expert,
                            executor_cfg=ExecutorConfig(safety=env.safety, H=cfg.horizon, v_hold=0.25))
    res = runner.run(T_max)
    ep = episode_from_records(res.records, env, f_dec, text)
    if ep is not None and lm.u_g is not None:
        ep.u_g = int(lm.u_g[0])                    # the mode that was broadcast (and relabelled with)
    return ep, res


def run(args, episodes: Optional[List] = None) -> Dict:
    dev = pick_device(args.device)
    g = seed_all(args.seed)
    rng = random.Random(args.seed)
    ck, cfg, clone, master = load_stage3(args.init, dev)
    data: List[Episode] = list(episodes if episodes is not None else
                               (episodes_from_args(args, cfg.horizon) if args.keep_initial or args.data else []))
    tr = PolicyTrainer(clone, master, args, dev, g)
    log = []
    for k in range(args.iters):
        beta = args.beta0 * args.beta_decay ** k
        freeze(clone, master)
        stats = []
        for r in range(args.rollouts):
            ep, res = rollout(clone, master, n_agents=rng.choice(args.agents), wind=rng.choice(args.wind),
                              seed=args.seed * 100003 + k * 101 + r, beta=beta, T_max=args.rollout_t_max,
                              f_dec=args.f_dec, text=rng.choice(INSTRUCTIONS))
            stats.append({"success": res.success, "d_min": res.d_min, "fallbacks": res.fallback_events,
                          "interventions": res.interventions})
            if ep is not None:
                data.append(ep)
        tr.enable()
        sampler = Sampler(data, seed=args.seed + k)
        for _ in range(args.steps):
            tr.step(team_batch(sampler.team(args.teams), cfg, dev))
        log.append({"iter": k, "beta": beta, "episodes": len(data), "rollouts": stats,
                    "last": tr.history[-1] if tr.history else None})
        print(f"[stage5] iter {k} beta {beta:.3f} aggregate {len(data)} episodes; "
              f"success {sum(s['success'] for s in stats)}/{len(stats)}; last loss "
              f"{tr.history[-1]['loss'] if tr.history else float('nan'):.4f}")
    freeze(clone)
    ck.update(stage=5, clone=clone.state_dict(), args5=vars(args), dagger_log=log)
    path = save_checkpoint(args.out, **ck)
    print(f"[stage5] saved {path}")
    return {"checkpoint": path, "log": log, "history": tr.history, "clone": clone, "master": master}


def main(argv=None) -> Dict:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

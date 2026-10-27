# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Shared utilities of the stage scripts: devices, seeding, checkpoints, freezing, team batches."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..models import CloneConfig, ClonePolicy, HashedTextEmbedder, MasterConfig, MasterPlanner
from .data import Episode, collect_episodes, load_episodes, proprio_batch, rel_goal_batch, render_frames

__all__ = [
    "pick_device", "seed_all", "freeze", "trainable", "save_checkpoint", "load_checkpoint", "build_clone",
    "build_master", "add_common_args", "episodes_from_args", "TeamBatch", "team_batch", "assert_finite",
]


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def freeze(*modules: nn.Module) -> None:
    for m in modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def trainable(*modules: nn.Module) -> List[nn.Parameter]:
    out, seen = [], set()
    for m in modules:
        m.train()
        for p in m.parameters():
            if id(p) not in seen:
                p.requires_grad_(True)
                out.append(p)
                seen.add(id(p))
    return out


def assert_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} is not finite: {value}")


def save_checkpoint(path: str, **payload) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict:
    return torch.load(path, map_location=device or "cpu", weights_only=False)


def build_clone(cfg: CloneConfig, state: Optional[Dict] = None, device: Optional[torch.device] = None) -> ClonePolicy:
    c = ClonePolicy(cfg)
    if state is not None:
        c.load_state_dict(state)
    return c.to(device) if device is not None else c


def build_master(clone_cfg: CloneConfig, state: Optional[Dict] = None, mcfg: Optional[MasterConfig] = None,
                 tiny: bool = False, device: Optional[torch.device] = None) -> MasterPlanner:
    if mcfg is None:
        mcfg = MasterConfig(bus_dim=clone_cfg.bus_dim, latent_dim=clone_cfg.latent_dim,
                            text_dim=clone_cfg.text_proj_dim, n_local_modes=clone_cfg.num_modes)
        if tiny:
            mcfg = MasterConfig(**{**asdict(mcfg), "width": 32, "depth": 2, "mode_emb_dim": 8, "plan_hidden": 32})
    m = MasterPlanner(mcfg)
    if state is not None:
        m.load_state_dict(state)
    return m.to(device) if device is not None else m


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--data", default=None, help="episode file from cma_vpa.training.data (else collect)")
    ap.add_argument("--collect", type=int, default=8, help="expert episodes to collect when --data is absent")
    ap.add_argument("--agents", type=int, nargs="+", default=[4, 8], help="team sizes for collection")
    ap.add_argument("--wind", nargs="+", default=["W0", "W1"], help="wind conditions for collection")
    ap.add_argument("--t-max", type=float, default=30.0, help="episode length for collection [s]")
    ap.add_argument("--f-dec", type=float, default=30.0, help="decision rate f_dec [Hz] (Table 4)")
    ap.add_argument("--tiny", action="store_true", help="small model widths (tests, quick runs)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--teams", type=int, default=8, help="team snapshots per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", required=False, default="checkpoints/stage.pt")
    return ap


def episodes_from_args(args, H: int) -> List[Episode]:
    if args.data:
        return load_episodes(args.data)
    return collect_episodes(args.collect, args.agents, args.wind, H=H, T_max=args.t_max, f_dec=args.f_dec,
                            seed=args.seed)


class TeamBatch:
    """Flattened per-agent tensors of a batch of team snapshots (agents padded to ``N_max``)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def team_batch(snapshots: List[Tuple[Episode, Tensor]], cfg: CloneConfig, device: torch.device,
               future: int = 0, text: Optional[HashedTextEmbedder] = None) -> TeamBatch:
    """Render and stack ``(episode, t)`` snapshots.

    Returns per-slot tensors of shape ``(B, N_max, ...)`` with ``present (B, N_max)``:
    frames at ``t`` (and ``t + future`` if ``future > 0``), proprioception (with
    ``sigma = 0``), heading-frame goal offsets, setpoints, flags, chunk and
    primitive labels, mode labels, milestone frames and instruction embeddings.
    """
    N_max = max(e.N for e, _ in snapshots)
    B = len(snapshots)
    C, H, W = cfg.in_channels, *cfg.image_size
    fr = torch.zeros(B, N_max, C, H, W)
    fr_f = torch.zeros(B, N_max, C, H, W) if future > 0 else None
    mil = torch.zeros(B, N_max, C, H, W)
    S = torch.zeros(B, N_max, cfg.proprio_dim)
    rg = torch.zeros(B, N_max, 3)
    a_bar = torch.zeros(B, N_max, cfg.action_dim)
    A = torch.zeros(B, N_max, cfg.horizon, cfg.action_dim)
    prim = torch.zeros(B, N_max, dtype=torch.long)
    u_M = torch.zeros(B, N_max, dtype=torch.long)
    u_g = torch.zeros(B, dtype=torch.long)
    present = torch.zeros(B, N_max, dtype=torch.bool)
    texts = []
    from ..envs.render import render_rgbd
    for b, (e, t) in enumerate(snapshots):
        n = e.N
        fr[b, :n] = render_frames(e, t, (H, W))[0]
        if future > 0:
            fr_f[b, :n] = render_frames(e, t + future, (H, W))[0]
        approach = e.goals - 2.0 * torch.stack([torch.cos(e.yaw), torch.sin(e.yaw), torch.zeros_like(e.yaw)], -1)
        idx = torch.tensor([[j for j in range(n) if j != i] for i in range(n)], dtype=torch.long)
        others = approach[idx] if n > 1 else torch.zeros(n, 0, 3, dtype=approach.dtype)
        mil[b, :n] = render_rgbd(approach, e.yaw, others, torch.full(others.shape[:2], e.r_v), e.goals,
                                 image_size=(H, W))
        S[b, :n] = proprio_batch(e, t)[0]
        rg[b, :n] = rel_goal_batch(e, t)[0]
        a_bar[b, :n] = e.a_bar[t][0]
        Hc = min(cfg.horizon, e.A.shape[2])
        A[b, :n, :Hc] = e.A[t][0, :, :Hc]
        if Hc < cfg.horizon:
            A[b, :n, Hc:] = A[b, :n, Hc - 1:Hc]
        prim[b, :n] = e.prim[t][0]
        u_M[b, :n] = e.u_M
        u_g[b] = e.u_g
        present[b, :n] = True
        texts.append(e.text)
    c_text = (text or HashedTextEmbedder(cfg.text_dim))(texts)
    out = TeamBatch(frames=fr, frames_future=fr_f, milestones=mil, S=S, rel_goal=rg, a_bar=a_bar, A=A, prim=prim,
                    u_M=u_M, u_g=u_g, present=present, c_text=c_text)
    for k, v in list(out.__dict__.items()):
        if isinstance(v, Tensor):
            setattr(out, k, v.to(device))
    return out

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 1 (Section 4.8 (i)): representation learning of ``E_psi`` and the ensemble ``P_omega``.

Objective (JEPA, no reconstruction)::

    L = (1/N_e) sum_e || z_hat^(e)_{t+1} - sg(E_psi_bar(I_{t+1})) ||^2 + lambda_v v(Z) + lambda_c c(Z) [+ lambda_s SIGReg]

with ``z_hat^(e) = P^(e)(z_t, Emb(u_t), a_bar_t)``, the EMA target encoder
``E_psi_bar`` (momentum annealed from 0.996 to 1.0 by a cosine schedule), VICReg
penalties on the batch of online latents ``Z`` (Eq. 8, ``gamma_v = 1``,
``eps_v = 1e-4``) and Poisson bootstrap weights per head (online bagging, which
keeps the heads' disagreement an epistemic signal). The primitive embedding
``Emb(u)`` is learned here and frozen with the representation.

At the end the representation is frozen, which fixes ``gamma_bar* = sqrt(mean_j Var(Z_.j) + eps)``
and the standardization statistics ``mu_Z``, ``mu_S``, ``sigma_S``; they are
installed in the clone (:meth:`~cma_vpa.models.ClonePolicy.calibrate`) and stored
in the checkpoint together with the EMA target encoder (used as the target of Stage 3).

Usage::

    python -m cma_vpa.training.stage1_repr --data t1_expert.pt --steps 20000 --out ckpt/stage1.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from ..models import CloneConfig, EMATarget, latent_scale, momentum_schedule
from .common import (add_common_args, assert_finite, build_clone, episodes_from_args, freeze, pick_device,
                     save_checkpoint, seed_all, team_batch, trainable)
from .data import Sampler
from .losses import bootstrap_weights, jepa_loss

__all__ = ["build_parser", "run", "main"]


def build_parser() -> argparse.ArgumentParser:
    ap = add_common_args(argparse.ArgumentParser(description="Stage 1: JEPA representation (E_psi, P_omega)"))
    ap.add_argument("--stride", type=int, default=1, help="prediction stride in decisions (I_{t+1})")
    ap.add_argument("--lambda-v", type=float, default=25.0)
    ap.add_argument("--lambda-c", type=float, default=1.0)
    ap.add_argument("--lambda-s", type=float, default=0.0, help="SIGReg weight (0 disables)")
    ap.add_argument("--gamma-v", type=float, default=1.0)
    ap.add_argument("--eps-v", type=float, default=1e-4)
    ap.add_argument("--ema-start", type=float, default=0.996)
    ap.add_argument("--ema-end", type=float, default=1.0)
    ap.add_argument("--no-bootstrap", action="store_true", help="disable per-head bootstrap weights")
    ap.add_argument("--stats-teams", type=int, default=64, help="team snapshots for the frozen statistics")
    ap.set_defaults(out="checkpoints/stage1.pt")
    return ap


def run(args, episodes: Optional[List] = None) -> Dict:
    dev = pick_device(args.device)
    g = seed_all(args.seed)
    cfg = CloneConfig.tiny() if args.tiny else CloneConfig()
    clone = build_clone(cfg, device=dev)
    eps = episodes if episodes is not None else episodes_from_args(args, cfg.horizon)
    sampler = Sampler(eps, min_future=args.stride, seed=args.seed)
    params = trainable(clone.encoder, clone.predictor, clone.prim_emb)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    ema = EMATarget(clone.encoder)
    history = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        tb = team_batch(sampler.team(args.teams), cfg, dev, future=args.stride)
        m = tb.present
        z = clone.encoder(tb.frames[m])
        target = ema(tb.frames_future[m])
        heads = clone.predictor.forward_heads(z, clone.prim_emb(tb.prim[m]), tb.a_bar[m])
        w = None if args.no_bootstrap else bootstrap_weights(heads.shape[0], heads.shape[1], g).to(dev)
        loss, terms = jepa_loss(heads, target, z, lambda_v=args.lambda_v, lambda_c=args.lambda_c,
                                lambda_s=args.lambda_s, gamma_v=args.gamma_v, eps_v=args.eps_v, head_weights=w,
                                generator=g if dev.type == "cpu" else None)
        assert_finite("stage-1 loss", loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        ema.update(clone.encoder, momentum_schedule(step, args.steps, args.ema_start, args.ema_end))
        history.append({k: float(v) for k, v in terms.items()} | {"loss": float(loss.detach())})
        if args.log_every and step % args.log_every == 0:
            h = history[-1]
            print(f"[stage1] step {step:6d} loss {h['loss']:.4f} pred {h['pred']:.4f} var {h['var']:.4f} "
                  f"cov {h['cov']:.4f} ({time.perf_counter() - t0:.1f}s)")

    # ---- freeze and fix gamma_bar* and the standardization statistics
    freeze(clone.encoder, clone.predictor, clone.prim_emb)
    with torch.no_grad():
        Z, S = [], []
        for _ in range(max(1, args.stats_teams // max(args.teams, 1))):
            tb = team_batch(sampler.team(args.teams), cfg, dev)
            Z.append(clone.encoder(tb.frames[tb.present]))
            S.append(tb.S[tb.present])
        Z, S = torch.cat(Z), torch.cat(S)
        gamma = latent_scale(Z, args.eps_v)
        mu_z = Z.mean(0)
        mu_s = S.mean(0)
        sd_s = S.std(0).clamp(min=1e-3)
        sd_s = torch.where(S.std(0) < 1e-6, torch.ones_like(sd_s), sd_s)     # constant channels: leave unscaled
    clone.calibrate(gamma, mu_z, mu_s, sd_s)
    ckpt = save_checkpoint(args.out, stage=1, config=asdict(cfg), clone=clone.state_dict(),
                           ema_encoder=ema.module.state_dict(),
                           calib={"gamma_star": gamma, "mu_z": mu_z.cpu(), "mu_s": mu_s.cpu(), "sd_s": sd_s.cpu()},
                           args=vars(args), history=history)
    print(f"[stage1] gamma_bar* = {gamma:.4f}; saved {ckpt}")
    return {"checkpoint": ckpt, "history": history, "gamma_star": gamma, "clone": clone, "ema": ema}


def main(argv=None) -> Dict:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 2 (Section 4.8 (ii)): bus space - ``P_bus`` and the RVQ codebooks on frozen latents.

The standardized latent ``z_tilde = (z - mu_Z) / gamma_bar*`` of the frozen
encoder is projected to the bus, ``x = P_bus z_tilde in R^64``, quantized by the
24-stage RVQ, and reconstructed by the tied map ``z_tilde ~ P_bus^T (x_hat - b)``.
The loss is the reconstruction error (straight-through through the quantizer)
plus the RVQ codebook and commitment terms. ``P_bus`` is initialized with the
leading principal directions of ``z_tilde`` and the codebooks by per-stage
k-means on the residuals; unused codewords are restarted periodically. Both
are frozen at the end.

Usage::

    python -m cma_vpa.training.stage2_bus --init ckpt/stage1.pt --steps 5000 --out ckpt/stage2.pt
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import torch

from ..models import CloneConfig
from .common import (add_common_args, assert_finite, build_clone, episodes_from_args, freeze, load_checkpoint,
                     pick_device, save_checkpoint, seed_all, team_batch, trainable)
from .data import Sampler
from .losses import bus_loss

__all__ = ["build_parser", "run", "main"]


def build_parser() -> argparse.ArgumentParser:
    ap = add_common_args(argparse.ArgumentParser(description="Stage 2: bus projection and RVQ codebooks"))
    ap.add_argument("--init", required=False, default="checkpoints/stage1.pt", help="stage-1 checkpoint")
    ap.add_argument("--init-teams", type=int, default=64, help="team snapshots for PCA / k-means init")
    ap.add_argument("--kmeans-iters", type=int, default=10)
    ap.add_argument("--restart-every", type=int, default=200)
    ap.add_argument("--lambda-rec", type=float, default=1.0)
    ap.set_defaults(out="checkpoints/stage2.pt", lr=1e-3, weight_decay=0.0)
    return ap


def run(args, episodes: Optional[List] = None) -> Dict:
    dev = pick_device(args.device)
    g = seed_all(args.seed)
    ck = load_checkpoint(args.init, dev)
    cfg = CloneConfig(**ck["config"])
    clone = build_clone(cfg, ck["clone"], dev)
    freeze(clone)
    eps = episodes if episodes is not None else episodes_from_args(args, cfg.horizon)
    sampler = Sampler(eps, seed=args.seed + 2)

    def latents(n_teams: int) -> torch.Tensor:
        with torch.no_grad():
            tb = team_batch(sampler.team(n_teams), cfg, dev)
            return clone.standardize_z(clone.encoder(tb.frames[tb.present]))

    # ---- initialization: PCA for P_bus, k-means for the codebooks
    with torch.no_grad():
        Zt = torch.cat([latents(args.teams) for _ in range(max(1, args.init_teams // max(args.teams, 1)))])
        mu = Zt.mean(0)
        _, _, Vh = torch.linalg.svd(Zt - mu, full_matrices=False)
        k = min(cfg.bus_dim, Vh.shape[0])
        W = torch.zeros(cfg.bus_dim, cfg.latent_dim, device=dev)
        W[:k] = Vh[:k]
        clone.bus.proj.weight.copy_(W)
        clone.bus.proj.bias.copy_(-(W @ mu))
        clone.rvq.init_from_data(clone.bus(Zt), iters=args.kmeans_iters, generator=g)

    params = trainable(clone.bus, clone.rvq)
    opt = torch.optim.Adam(params, lr=args.lr)
    history = []
    for step in range(args.steps):
        z = latents(args.teams)
        loss, terms = bus_loss(z, clone.bus, clone.rvq, args.lambda_rec)
        assert_finite("stage-2 loss", loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        if args.restart_every and step % args.restart_every == args.restart_every - 1:
            clone.rvq.restart_dead_codes(clone.bus(z).detach(), generator=g)
        history.append({"loss": float(loss.detach()), "rec": float(terms["rec"]),
                        "codebook": float(terms["codebook"]), "commit": float(terms["commit"])})
        if args.log_every and step % args.log_every == 0:
            print(f"[stage2] step {step:6d} loss {history[-1]['loss']:.4f} rec {history[-1]['rec']:.4f}")
    freeze(clone.bus)
    clone.rvq.freeze()
    with torch.no_grad():
        z = latents(args.teams)
        x = clone.bus(z)
        snr = clone.rvq.snr_db(x)
        rel_rec = float((z - clone.bus.reconstruct(clone.rvq.decode(clone.rvq.encode(x)))).pow(2).sum(-1).mean()
                        / z.pow(2).sum(-1).mean().clamp(min=1e-12))
    ck.update(stage=2, clone=clone.state_dict(), bus_metrics={"snr_db": snr, "relative_reconstruction": rel_rec},
              args2=vars(args), history2=history)
    path = save_checkpoint(args.out, **ck)
    print(f"[stage2] bus SNR {snr:.2f} dB, relative latent reconstruction {rel_rec:.4f}; saved {path}")
    return {"checkpoint": path, "history": history, "snr_db": snr, "clone": clone}


def main(argv=None) -> Dict:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

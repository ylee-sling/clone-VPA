# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 3 (Section 4.8 (iii)): the master - ``J_Omega``, the mode heads and the plan heads.

Inputs are the **decoded quantized uplinks** of every vehicle of a team
snapshot, ``x_hat^(i) = RVQ^-1(RVQ(P_bus z_tilde_t^(i)))``, with the
uncertainty ``sigma^(i)`` of the frozen predictor, the projected instruction
``c_tilde = P_c c_text`` and the milestone latents. Targets are each vehicle's
bus-space latent at ``T_M = 1 s``, ``P_bus((E(I_{t+T_M}) - mu_Z) / gamma_bar*)``, computed
under stop-gradient with the target encoder ``E``. By default (``--target-encoder
online``) ``E`` is the frozen ``E_psi``: once ``E_psi`` is frozen its EMA copy converges
to it, and using it keeps the targets on exactly the scale of the uplinks; with
``--target-encoder ema`` the stored Stage-1 EMA encoder ``E_psi_bar`` is used. The loss is

    plan regression (teacher-forced with the label modes, bootstrap-weighted heads)
    + lambda_mode [CE(pi^g(h_0), u_g) + CE(pi^loc(h_i, u_g), u_M^i)]
    + lambda_vc [v(H) + c(H)]  on the agent features,

with mode labels from the fixed rules of :mod:`cma_vpa.envs.expert` (passing
handedness). The text projection ``P_c`` is trained here together with the
master (it is the first module that consumes ``c_tilde``) and frozen with it.

Usage::

    python -m cma_vpa.training.stage3_master --init ckpt/stage2.pt --steps 10000 --out ckpt/stage3.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from ..models import CloneConfig, SiameseViTEncoder
from .common import (add_common_args, assert_finite, build_clone, build_master, episodes_from_args, freeze,
                     load_checkpoint, pick_device, save_checkpoint, seed_all, team_batch, trainable)
from .data import Sampler
from .losses import bootstrap_weights, master_loss

__all__ = ["build_parser", "run", "main", "master_inputs"]


def build_parser() -> argparse.ArgumentParser:
    ap = add_common_args(argparse.ArgumentParser(description="Stage 3: PR-JEPA master planner"))
    ap.add_argument("--init", default="checkpoints/stage2.pt", help="stage-2 checkpoint")
    ap.add_argument("--T-M", type=float, default=1.0, help="macro horizon T_M [s]")
    ap.add_argument("--lambda-mode", type=float, default=1.0)
    ap.add_argument("--lambda-vc", type=float, default=0.1)
    ap.add_argument("--target-encoder", choices=["online", "ema"], default="online")
    ap.set_defaults(out="checkpoints/stage3.pt", lr=3e-4)
    return ap


@torch.no_grad()
def master_inputs(clone, tb) -> Dict[str, torch.Tensor]:
    """Decoded quantized uplinks, uncertainty and milestone latents of a :class:`TeamBatch` (frozen clone)."""
    B, N = tb.present.shape
    m = tb.present
    z = clone.encoder(tb.frames[m])
    pred = clone.predictor(z, clone.prim_emb(tb.prim[m]), tb.a_bar[m])
    codes = clone.rvq.encode(clone.bus_latent(z))
    x_hat = torch.zeros(B, N, clone.cfg.bus_dim, device=z.device)
    x_hat[m] = clone.rvq.decode(codes).to(x_hat.dtype)
    sigma = torch.zeros(B, N, device=z.device)
    sigma[m] = pred.sigma
    zg = torch.zeros(B, N, clone.cfg.latent_dim, device=z.device)
    zg[m] = clone.encoder(tb.milestones[m])
    zt = torch.zeros(B, N, clone.cfg.latent_dim, device=z.device)
    zt[m] = z
    return {"x_hat": x_hat, "sigma": sigma, "z_goal": zg, "z": zt, "codes": codes, "z_hat": pred.z_hat,
            "sigma_flat": pred.sigma}


def run(args, episodes: Optional[List] = None) -> Dict:
    dev = pick_device(args.device)
    g = seed_all(args.seed)
    ck = load_checkpoint(args.init, dev)
    cfg = CloneConfig(**ck["config"])
    clone = build_clone(cfg, ck["clone"], dev)
    freeze(clone)
    if args.target_encoder == "ema":
        target_enc = SiameseViTEncoder(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.vit_width,
                                       cfg.vit_depth, cfg.vit_heads, latent_dim=cfg.latent_dim).to(dev)
        target_enc.load_state_dict(ck["ema_encoder"])
        freeze(target_enc)
    else:
        target_enc = clone.encoder
    master = build_master(cfg, tiny=args.tiny, device=dev)
    master.gamma_star.fill_(float(clone.gamma_star))
    eps = episodes if episodes is not None else episodes_from_args(args, cfg.horizon)
    dt = eps[0].dt
    fut = max(1, int(round(args.T_M / dt)))
    sampler = Sampler(eps, min_future=fut, seed=args.seed + 3)
    params = trainable(master, clone.text_proj)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for step in range(args.steps):
        tb = team_batch(sampler.team(args.teams), cfg, dev, future=fut)
        inp = master_inputs(clone, tb)
        with torch.no_grad():
            m = tb.present
            target = torch.zeros_like(inp["x_hat"])
            target[m] = clone.bus(clone.standardize_z(target_enc(tb.frames_future[m])))
        c_tilde = clone.text_proj(tb.c_text)
        out = master(inp["x_hat"], inp["sigma"], tb.present, c_tilde, inp["z_goal"], u_g=tb.u_g, u_M=tb.u_M)
        hw = bootstrap_weights(master.cfg.n_ens, tb.present.numel(), g).to(dev).view(master.cfg.n_ens,
                                                                                  *tb.present.shape)
        loss, terms = master_loss(out, target, tb.present, tb.u_g, tb.u_M, lambda_mode=args.lambda_mode,
                                  lambda_vc=args.lambda_vc, head_weights=hw)
        assert_finite("stage-3 loss", loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        acc = float((out.global_logits.argmax(-1) == tb.u_g).float().mean())
        history.append({k: float(v) for k, v in terms.items()} | {"loss": float(loss.detach()), "acc_g": acc})
        if args.log_every and step % args.log_every == 0:
            h = history[-1]
            print(f"[stage3] step {step:6d} loss {h['loss']:.4f} plan {h['plan']:.4f} ce_g {h['ce_g']:.3f} "
                  f"ce_l {h['ce_l']:.3f} acc_g {acc:.2f}")
    freeze(master, clone.text_proj)
    ck.update(stage=3, clone=clone.state_dict(), master=master.state_dict(), master_config=asdict(master.cfg),
              args3=vars(args), history3=history)
    path = save_checkpoint(args.out, **ck)
    print(f"[stage3] master {sum(p.numel() for p in master.parameters()):,} parameters; saved {path}")
    return {"checkpoint": path, "history": history, "master": master, "clone": clone}


def main(argv=None) -> Dict:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

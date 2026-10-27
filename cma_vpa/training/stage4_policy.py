# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 4 (Section 4.8 (iv)): clone policy - the selector ``pi_phi^h`` and the CFM solver ``v_theta``.

Training minimizes ``L_CFM + lambda_h L_CE`` on expert chunks. The plan
conditioning is produced exactly as at deployment, by the **frozen master on
decoded quantized uplinks**: ``x_hat -> J_Omega -> (u_M, pi) -> RVQ -> pi_hat -> Z_tilde``
(Eq. 16), never from target-encoder views of the future, which would leak the
answer into the conditioning [23]. The master's global mode is set to the
episode's executed handedness label (so that the conditioning matches the
mode the expert chunk realizes), and its local mode is its own argmax.
With probability ``plan_dropout = 0.1`` the plan tokens are replaced by the
learned null token. Trained here: selector, the clone mode embedding
``Emb(u_M)``, ``v_theta`` (with the null token) and the plan-token decoder;
everything upstream is frozen. The barrier never enters the loss.

Usage::

    python -m cma_vpa.training.stage4_policy --init ckpt/stage3.pt --steps 20000 --out ckpt/stage4.pt
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import torch

from ..models import CloneConfig, MasterConfig, goal_feature
from .common import (add_common_args, assert_finite, build_clone, build_master, episodes_from_args, freeze,
                     load_checkpoint, pick_device, save_checkpoint, seed_all, team_batch, trainable)
from .data import Sampler
from .losses import policy_loss
from .stage3_master import master_inputs

__all__ = ["build_parser", "PolicyTrainer", "run", "main", "MAX_CLONE_PARAMETERS"]

MAX_CLONE_PARAMETERS = 18_700_000


def build_parser() -> argparse.ArgumentParser:
    ap = add_common_args(argparse.ArgumentParser(description="Stage 4: clone selector and CFM solver"))
    ap.add_argument("--init", default="checkpoints/stage3.pt", help="stage-3 checkpoint")
    ap.add_argument("--lambda-h", type=float, default=1.0)
    ap.add_argument("--plan-dropout", type=float, default=0.1)
    ap.set_defaults(out="checkpoints/stage4.pt", lr=3e-4)
    return ap


class PolicyTrainer:
    """One optimizer over the Stage-4 trainable modules; shared with Stage 5 (DAgger)."""

    def __init__(self, clone, master, args, device, generator: torch.Generator):
        self.clone, self.master, self.args, self.dev, self.g = clone, master, args, device, generator
        freeze(clone, master)
        mods = [clone.selector, clone.mode_emb, clone.field, clone.plan_decoder]
        if clone.goal_token is not None:
            mods.append(clone.goal_token)
        self.modules = mods
        self.params = trainable(*mods)
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=args.weight_decay)
        self.history: List[Dict] = []

    def enable(self) -> None:
        """Re-enable training of the Stage-4 modules (after a frozen closed-loop rollout)."""
        for mod in self.modules:
            mod.train()
        for p in self.params:
            p.requires_grad_(True)

    def step(self, tb) -> Dict[str, float]:
        clone, master, a = self.clone, self.master, self.args
        m = tb.present
        inp = master_inputs(clone, tb)
        with torch.no_grad():
            c_tilde = clone.text_proj(tb.c_text)
            mo = master(inp["x_hat"], inp["sigma"], tb.present, c_tilde, inp["z_goal"], u_g=tb.u_g)
            pi_hat = clone.rvq.decode(clone.rvq.encode(mo.plan))           # what the downlink conveys
            u_M = mo.u_M[m]
            z = inp["z"][m]
            S = tb.S[m].clone()
            S[:, 10] = inp["sigma_flat"]                                    # sigma_bar slot of S_t
            g_t = goal_feature(tb.rel_goal[m])
            c_rep = c_tilde.unsqueeze(1).expand(-1, m.shape[1], -1)[m]
        drop = torch.rand(u_M.shape[0], generator=self.g).to(self.dev) < a.plan_dropout
        kv = clone.plan_kv(pi_hat[m], u_M, inp["z_goal"][m], drop=drop)
        logits = clone.selector(z, g_t, clone.standardize_s(S), c_rep, clone.mode_emb(u_M))
        cond = clone.conditioning(tb.prim[m], z, inp["z_hat"], S, g_t, c_rep, u_M)
        loss, terms = policy_loss(clone.field, tb.A[m], cond, kv, logits, tb.prim[m], a.lambda_h,
                                  generator=self.g if self.dev.type == "cpu" else None)
        assert_finite("stage-4 loss", loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, a.grad_clip)
        self.opt.step()
        acc = float((logits.argmax(-1) == tb.prim[m]).float().mean())
        h = {"loss": float(loss.detach()), "cfm": float(terms["cfm"]), "ce": float(terms["ce"]), "acc": acc}
        self.history.append(h)
        return h


def load_stage3(path: str, dev: torch.device):
    ck = load_checkpoint(path, dev)
    cfg = CloneConfig(**ck["config"])
    clone = build_clone(cfg, ck["clone"], dev)
    master = build_master(cfg, ck["master"], MasterConfig(**ck["master_config"]), device=dev)
    return ck, cfg, clone, master


def run(args, episodes: Optional[List] = None) -> Dict:
    dev = pick_device(args.device)
    g = seed_all(args.seed)
    ck, cfg, clone, master = load_stage3(args.init, dev)
    budget = clone.parameter_budget()
    if not args.tiny and budget["total"] >= MAX_CLONE_PARAMETERS:
        raise RuntimeError(f"clone has {budget['total']:,} parameters (limit {MAX_CLONE_PARAMETERS:,})")
    eps = episodes if episodes is not None else episodes_from_args(args, cfg.horizon)
    sampler = Sampler(eps, seed=args.seed + 4)
    tr = PolicyTrainer(clone, master, args, dev, g)
    for step in range(args.steps):
        h = tr.step(team_batch(sampler.team(args.teams), cfg, dev))
        if args.log_every and step % args.log_every == 0:
            print(f"[stage4] step {step:6d} loss {h['loss']:.4f} cfm {h['cfm']:.4f} ce {h['ce']:.3f} "
                  f"acc {h['acc']:.2f}")
    freeze(clone)
    ck.update(stage=4, clone=clone.state_dict(), args4=vars(args), history4=tr.history, budget=budget)
    path = save_checkpoint(args.out, **ck)
    print(f"[stage4] clone {budget['total']:,} parameters; saved {path}")
    return {"checkpoint": path, "history": tr.history, "clone": clone, "master": master, "budget": budget}


def main(argv=None) -> Dict:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

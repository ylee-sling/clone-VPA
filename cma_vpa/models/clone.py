# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Clone pipeline: the ``K + 3`` decision chain (Section 4.3, Algorithm 1 DECIDE, Proposition 8).

One decision of clone ``i``::

    z_t   = E_psi(I_t)                                              evaluation 1
    u_t   = argmax pi_phi^h(u | z_t, g_t, S_t, c_tilde, Emb(u_M))    evaluation 2
    z_hat, sigma = P_omega(z_t, Emb(u_t), a_bar_t)                 evaluation 3 (heads batched)
    sigma_bar, xi_t (noise persistence), e_t (Eq. 15)
    A_t = GUIDEDSOLVE(xi_t, e_t, Z_tilde, P_t, K)                    evaluations 4 .. K + 3
    alpha(sigma_bar), H_val, r_eff (Eq. 35); publish to the chunk double buffer

The bridge keys and values, the RVQ decoding, the text projection, the bus
projection and the milestone latents are computed on message arrival or once
per mission and are read from buffers, so they are not on the chain. Every
:meth:`ClonePolicy.decide` audits the chain with forward hooks and raises if
the evaluations differ from ``[E_psi, pi_phi_h, P_omega] + [v_theta] * K`` or if
any bridge K/V projection ran inside the decision.

Conditioning (Eq. 15), ``d_e = 721``::

    e_t = [Emb(u_t) 32 | z_tilde_t 256 | z_hat_tilde 256 | S_tilde_t 13 | g_t 4 | c_tilde 128 | Emb(u_M) 32]

with ``z_tilde = (z - mu_Z) / gamma_bar*`` and ``S_tilde = (S - mu_S) / sigma_S``.

Parameter budget (Table 3): the modules listed there total exactly 18,669,514
parameters; the clone additionally holds the two 32-dimensional embedding
tables ``Emb(u)`` (6 x 32) and ``Emb(u_M)`` (16 x 32) that Table 3 folds into
its rounding, 18,670,218 in all (:meth:`ClonePolicy.parameter_budget`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..comm.packet import (UP_BRAKING, UP_FALLBACK, UP_NULL_PLAN, UP_STALE_PLAN, DownlinkPacket, PacketError,
                           SequenceFilter, UplinkPacket, unwrap_timestamp)
from ..comm.rvq import ResidualVQ
from ..safety.params import DEFAULT_PARAMS, SafetyParams
from .bridge import BridgeCache, PlanTokenDecoder
from .encoder import SiameseViTEncoder, TextProjection
from .predictor import EnsemblePredictor, UncertaintyFilter
from .selector import MilestoneTracker, PrimitiveSelector, goal_feature
from .solver import FlowSolver, FlowVectorField, NoisePersistence, PlanKV

__all__ = ["CloneConfig", "BusProjection", "ClonePolicy", "DecisionOutput", "ChunkSlots", "TABLE3_BUDGET"]

#: Analytic parameter counts of Table 3.
TABLE3_BUDGET = {
    "encoder": 11_171_200, "text_proj": 65_664, "selector": 316_806, "predictor": 2_720_000,
    "solver": 3_915_524, "rvq": 393_216, "bus": 16_448, "plan_decoder": 70_656,
}


@dataclass
class CloneConfig:
    """Clone hyper-parameters; defaults are Table 3 and Table 4."""

    image_size: Tuple[int, int] = (128, 160)
    in_channels: int = 4
    patch_size: int = 16
    vit_width: int = 384
    vit_depth: int = 6
    vit_heads: int = 6
    latent_dim: int = 256
    text_dim: int = 512
    text_proj_dim: int = 128
    goal_dim: int = 4
    proprio_dim: int = 13
    num_primitives: int = 6
    num_modes: int = 16
    emb_dim: int = 32
    selector_hidden: Tuple[int, ...] = (384, 384)
    n_heads: int = 5
    pred_hidden: int = 512
    pred_layers: int = 2
    horizon: int = 32
    action_dim: int = 4
    field_width: int = 256
    field_depth: int = 4
    field_heads: int = 4
    time_dim: int = 128
    cross_after: Tuple[int, ...] = (1, 3)
    n_plan_tokens: int = 4
    bus_dim: int = 64
    rvq_stages: int = 24
    rvq_size: int = 256
    K: int = 2
    N_xi: int = 30
    H_min: int = 8
    beta: float = 1.0
    alpha_sigma: float = 0.1
    kappa: float = 0.1
    plan_dropout: float = 0.1
    #: leading action channels scaled by the velocity attenuation alpha(sigma_bar):
    #: 3 for quadrotor velocity setpoints (d_a = 4), 6 for arm twists (d_a = 7); the rest pass through
    attenuate_channels: int = 3

    @staticmethod
    def arm(**kw) -> "CloneConfig":
        """Task-T2 configuration: two stacked RGB-D views (8 channels), twist + gripper chunks (``d_a = 7``)."""
        return replace(CloneConfig(in_channels=8, action_dim=7, attenuate_channels=6), **kw)

    @property
    def cond_dim(self) -> int:
        return (2 * self.emb_dim + 2 * self.latent_dim + self.proprio_dim + self.goal_dim + self.text_proj_dim)

    @staticmethod
    def tiny(**kw) -> "CloneConfig":
        """A small configuration for tests and quick experiments (same structure, small widths)."""
        base = CloneConfig(image_size=(32, 40), patch_size=8, vit_width=48, vit_depth=2, vit_heads=4,
                           latent_dim=32, text_dim=32, text_proj_dim=16, selector_hidden=(32, 32),
                           n_heads=3, pred_hidden=32, pred_layers=2, horizon=8, field_width=32,
                           field_depth=2, field_heads=4, time_dim=16, cross_after=(0, 1), bus_dim=16,
                           rvq_stages=24, rvq_size=16, emb_dim=8, H_min=2)
        return replace(base, **kw)


class BusProjection(nn.Module):
    """``P_bus : R^256 -> R^64`` (16,448 parameters) on standardized latents.

    Stage 2 trains it with a tied linear reconstruction ``z_tilde ~ W^T (P_bus z_tilde - b)``
    (a PCA-type objective that cannot collapse) together with the RVQ losses.
    """

    def __init__(self, latent_dim: int = 256, bus_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(latent_dim, bus_dim)

    def forward(self, z_tilde: Tensor) -> Tensor:
        return self.proj(z_tilde)

    def reconstruct(self, x: Tensor) -> Tensor:
        return (x - self.proj.bias) @ self.proj.weight


@dataclass
class DecisionOutput:
    """One decision of a batch of clones."""

    chunk: Tensor           # (B, H, d_a) normalized chunk after attenuation
    raw_chunk: Tensor       # (B, H, d_a) solver output before attenuation
    primitive: Tensor       # (B,)
    selector_logits: Tensor
    z_t: Tensor
    z_hat: Tensor
    sigma: Tensor
    sigma_bar: Tensor
    H_val: Tensor           # (B,) int64 knots
    alpha: Tensor           # (B,)
    r_eff: Tensor           # (B,) requested effective radius (Eq. 35)
    cond: Tensor            # (B, d_e)
    plan_valid: Tensor      # (B,) bool
    noise_redrawn: Tensor   # (B,) bool
    chain: List[str]
    guided: Optional[object] = None


class ChunkSlots:
    """Double-buffered single-slot chunk output with stamp and validity ``H_val`` (Algorithm 1, line 20)."""

    def __init__(self, batch: int, H: int, d_a: int, device: Optional[torch.device] = None):
        self.A = torch.zeros(2, batch, H, d_a, device=device)
        self.stamp = torch.full((2, batch), -math.inf, dtype=torch.float64, device=device)
        self.H_val = torch.zeros(2, batch, dtype=torch.long, device=device)
        self.front = torch.zeros(batch, dtype=torch.long, device=device)
        self._ar = torch.arange(batch, device=device)

    def publish(self, A: Tensor, stamp: float, H_val: Tensor, mask: Optional[Tensor] = None) -> None:
        mask = torch.ones(A.shape[0], dtype=torch.bool, device=A.device) if mask is None else mask
        sel = mask.nonzero(as_tuple=True)[0]
        back = 1 - self.front[sel]
        self.A[back, sel] = A[sel].detach().to(self.A.dtype)
        self.stamp[back, sel] = float(stamp)
        self.H_val[back, sel] = H_val[sel]
        self.front[sel] = back

    def newest(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.A[self.front, self._ar], self.stamp[self.front, self._ar], self.H_val[self.front, self._ar]


class ClonePolicy(nn.Module):
    """All clone networks plus the per-agent runtime state of DECIDE / ONDOWNLINK (batched over agents)."""

    CHAIN = ("E_psi", "pi_phi_h", "P_omega", "v_theta")

    def __init__(self, cfg: CloneConfig = CloneConfig(), safety: SafetyParams = DEFAULT_PARAMS):
        super().__init__()
        self.cfg, self.safety = cfg, safety
        self.encoder = SiameseViTEncoder(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.vit_width,
                                         cfg.vit_depth, cfg.vit_heads, latent_dim=cfg.latent_dim)
        self.text_proj = TextProjection(cfg.text_dim, cfg.text_proj_dim)
        self.prim_emb = nn.Embedding(cfg.num_primitives, cfg.emb_dim)
        self.mode_emb = nn.Embedding(cfg.num_modes, cfg.emb_dim)
        self.selector = PrimitiveSelector(cfg.latent_dim, cfg.goal_dim, cfg.proprio_dim, cfg.text_proj_dim,
                                          cfg.emb_dim, cfg.selector_hidden, cfg.num_primitives)
        self.predictor = EnsemblePredictor(cfg.latent_dim, cfg.emb_dim, cfg.action_dim, cfg.pred_hidden,
                                           cfg.pred_layers, cfg.n_heads)
        self.field = FlowVectorField(cfg.horizon, cfg.action_dim, cfg.cond_dim, cfg.field_width,
                                     cfg.field_depth, cfg.field_heads, time_dim=cfg.time_dim,
                                     cross_after=cfg.cross_after)
        self.solver = FlowSolver(self.field, cfg.K)
        self.plan_decoder = PlanTokenDecoder(cfg.bus_dim, cfg.field_width, cfg.n_plan_tokens, cfg.num_modes)
        self.bus = BusProjection(cfg.latent_dim, cfg.bus_dim)
        self.rvq = ResidualVQ(cfg.rvq_stages, cfg.rvq_size, cfg.bus_dim)
        if cfg.field_width != cfg.latent_dim:
            self.goal_token = nn.Linear(cfg.latent_dim, cfg.field_width, bias=False)   # tiny configs only
        else:
            self.goal_token = None
        self.register_buffer("mu_z", torch.zeros(cfg.latent_dim))
        self.register_buffer("gamma_star", torch.tensor(1.0))
        self.register_buffer("mu_s", torch.zeros(cfg.proprio_dim))
        self.register_buffer("sd_s", torch.ones(cfg.proprio_dim))
        # runtime state (not in the state dict)
        self._chain: List[str] = []
        self._kv_calls = 0
        self.B = 0

    # ------------------------------------------------------------------ hooks / audit
    def _attach_audit(self) -> List:
        """Register the depth-audit hooks for one decision (removed afterwards; copy-safe)."""
        handles = []
        chain = self._chain

        def make(name: str) -> Callable:
            return lambda module, inputs, output: chain.append(name)

        for name, mod in zip(self.CHAIN, (self.encoder, self.selector, self.predictor, self.field)):
            handles.append(mod.register_forward_hook(make(name)))

        def kv_hook(module, inputs, output):
            self._kv_calls += 1

        for layer in self.field.cross:
            for proj in (layer.k_proj, layer.v_proj):
                handles.append(proj.register_forward_hook(kv_hook))
        return handles

    @property
    def expected_chain(self) -> List[str]:
        return [self.CHAIN[0], self.CHAIN[1], self.CHAIN[2]] + [self.CHAIN[3]] * self.solver.K

    # ------------------------------------------------------------------ budget
    def parameter_budget(self) -> Dict[str, int]:
        """Per-module parameter counts in the layout of Table 3 (+ the embedding tables)."""
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        out = {"encoder": n(self.encoder), "text_proj": n(self.text_proj), "selector": n(self.selector),
               "predictor": n(self.predictor), "solver": n(self.field), "rvq": self.rvq.codebooks.numel(),
               "bus": n(self.bus), "plan_decoder": n(self.plan_decoder),
               "embeddings": n(self.prim_emb) + n(self.mode_emb)}
        if self.goal_token is not None:
            out["goal_token"] = n(self.goal_token)
        out["total"] = sum(out.values())
        return out

    # ------------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate(self, gamma_star: float, mu_z: Tensor, mu_s: Tensor, sd_s: Tensor) -> None:
        """Install the statistics frozen at the end of Stage 1 (shared by every downstream module)."""
        if not gamma_star > 0 or not bool((sd_s > 0).all()):
            raise ValueError("gamma_bar* and sigma_S must be positive")
        self.gamma_star.fill_(float(gamma_star))
        self.predictor.set_gamma_star(gamma_star)
        self.mu_z.copy_(mu_z)
        self.mu_s.copy_(mu_s)
        self.sd_s.copy_(sd_s)

    def standardize_z(self, z: Tensor) -> Tensor:
        return (z - self.mu_z) / self.gamma_star

    def standardize_s(self, s: Tensor) -> Tensor:
        return (s - self.mu_s) / self.sd_s

    def bus_latent(self, z: Tensor) -> Tensor:
        """``P_bus z_tilde`` (uplink content before quantization)."""
        return self.bus(self.standardize_z(z))

    # ------------------------------------------------------------------ conditioning (Eq. 15) and tokens (Eq. 16)
    def conditioning(self, u_t: Tensor, z_t: Tensor, z_hat: Tensor, s_t: Tensor, g_t: Tensor,
                     c_tilde: Tensor, u_M: Tensor) -> Tensor:
        e = torch.cat([self.prim_emb(u_t.long()), self.standardize_z(z_t), self.standardize_z(z_hat),
                       self.standardize_s(s_t), g_t, c_tilde, self.mode_emb(u_M.long())], -1)
        assert e.shape[-1] == self.cfg.cond_dim
        return e

    def plan_tokens(self, pi_hat: Tensor, u_M: Tensor, z_goal: Tensor) -> Tensor:
        zg = z_goal if self.goal_token is None else self.goal_token(z_goal)
        return self.plan_decoder(pi_hat, u_M, zg, float(self.gamma_star))

    def plan_kv(self, pi_hat: Tensor, u_M: Tensor, z_goal: Tensor, drop: Optional[Tensor] = None) -> PlanKV:
        """Training-time K/V; ``drop (B,)`` replaces the plan by the null token (plan dropout)."""
        kv = self.field.project_kv(self.plan_tokens(pi_hat, u_M, z_goal))
        if drop is None or not bool(drop.any()):
            return kv
        return self.field.null_kv(pi_hat.shape[0], self.plan_decoder.n_tokens).select(drop, kv)

    # ------------------------------------------------------------------ mission / episode
    @torch.no_grad()
    def reset_mission(self, c_text: Tensor, milestone_frames: Optional[Tensor] = None,
                      milestone_latents: Optional[Tensor] = None, agent_ids: Optional[Sequence[int]] = None) -> None:
        """Begin a mission for ``B`` clones: ``c_tilde = P_c c_text`` and ``z_g = E_psi(I_g)`` once.

        ``c_text (B or 1, 512)``; milestones as frames ``(B, M, C, H, W)`` or latents ``(B, M, d)``.
        """
        if (milestone_frames is None) == (milestone_latents is None):
            raise ValueError("give exactly one of milestone_frames / milestone_latents")
        zg = self.encoder.encode_milestones(milestone_frames) if milestone_latents is None else milestone_latents
        B, M = zg.shape[:2]
        dev = zg.device
        self.B = B
        c = self.text_proj(c_text)
        self.c_tilde = c.expand(B, -1).clone() if c.shape[0] == 1 else c
        self.milestones = zg
        self.agent_ids = list(range(B)) if agent_ids is None else list(agent_ids)
        self._row = {a: k for k, a in enumerate(self.agent_ids)}
        self.tracker = MilestoneTracker(self.cfg.kappa, self.cfg.latent_dim, float(self.gamma_star))
        self.tracker.reset(B, M, dev)
        self.filter = UncertaintyFilter(self.cfg.horizon, self.cfg.H_min, self.cfg.beta, self.cfg.alpha_sigma)
        self.filter.reset(B, dev)
        self.noise = NoisePersistence((self.cfg.horizon, self.cfg.action_dim), self.cfg.N_xi)
        self.bridge = BridgeCache(self.field, B, self.plan_decoder.n_tokens, self.gamma_star.dtype, dev)
        self.bridge.refresh_null()
        self.chunks = ChunkSlots(B, self.cfg.horizon, self.cfg.action_dim, dev)
        self.u_M = torch.zeros(B, dtype=torch.long, device=dev)
        self.a_bar = torch.zeros(B, self.cfg.action_dim, device=dev)
        self.last_z = torch.zeros(B, self.cfg.latent_dim, device=dev)
        self.ever_planned = torch.zeros(B, dtype=torch.bool, device=dev)
        self.down_filter = SequenceFilter(16)
        self.up_seq = 0

    # ------------------------------------------------------------------ ONDOWNLINK
    @torch.no_grad()
    def on_downlink(self, messages: Sequence[bytes], now: float) -> Tensor:
        """Algorithm 1 lines 8-10 for a batch of raw downlinks; returns the mask of agents updated.

        Each message is CRC- and sequence-checked; the plan is decoded and the
        K/V of its six plan tokens are computed **here**, once, and written to the
        bridge double buffer together with the receiver-clock expiry time.
        """
        B, dev = self.B, self.u_M.device
        upd = torch.zeros(B, dtype=torch.bool, device=dev)
        valid_msg = torch.zeros(B, dtype=torch.bool, device=dev)
        codes = torch.zeros(B, self.cfg.rvq_stages, dtype=torch.uint8, device=dev)
        until = torch.full((B,), -math.inf, dtype=torch.float64, device=dev)
        mode = self.u_M.clone()
        mil = self.tracker.m.clone()
        for raw in messages:
            try:
                p = DownlinkPacket.unpack(raw)
            except PacketError:
                continue
            k = self._row.get(p.agent_id)
            if k is None or not self.down_filter.accept(p.agent_id, p.seq):
                continue
            upd[k] = True
            mode[k] = p.mode
            mil[k] = max(1, p.milestone)
            if p.plan_valid:
                valid_msg[k] = True
                codes[k] = torch.tensor(list(p.code), dtype=torch.uint8)
                until[k] = unwrap_timestamp(p.timestamp, now) + p.validity_s
        if not bool(upd.any()):
            return upd
        self.u_M = torch.where(upd, mode, self.u_M)
        self.tracker.assign(mil, upd)
        if bool(valid_msg.any()):
            pi_hat = self.rvq.decode(codes)
            zg = self.tracker.active(self.milestones)
            tokens = self.plan_tokens(pi_hat.to(zg.dtype), self.u_M, zg)
            self.bridge.write(valid_msg, tokens, until)
            self.ever_planned |= valid_msg
        self.bridge.invalidate(upd & ~valid_msg)
        return upd

    # ------------------------------------------------------------------ uplink
    @torch.no_grad()
    def uplinks(self, now: float, executor_flags: Optional[Tensor] = None) -> List[bytes]:
        """35-byte uplinks with ``RVQ(P_bus z_t)``, ``sigma_bar_{t+1}`` and status flags (Algorithm 1, line 11)."""
        codes = self.rvq.encode(self.bus_latent(self.last_z))
        sb = self.filter.sigma_bar
        valid = self.bridge.valid(now)
        self.up_seq = (self.up_seq + 1) % 65536
        out = []
        for k, a in enumerate(self.agent_ids):
            fl = 0
            if executor_flags is not None:
                ef = int(executor_flags[k])
                fl |= (UP_FALLBACK if ef & 0x01 else 0) | (UP_BRAKING if ef & 0x02 else 0)
            if not bool(self.ever_planned[k]):
                fl |= UP_NULL_PLAN
            elif not bool(valid[k]):
                fl |= UP_STALE_PLAN
            out.append(UplinkPacket.build(a, self.up_seq, now, codes[k].tolist(), float(sb[k]), fl).pack())
        return out

    # ------------------------------------------------------------------ DECIDE
    @torch.no_grad()
    def decide(self, frames: Tensor, s_t: Tensor, rel_goal_heading: Tensor, now: float, *,
               polytope=None, guidance_kw: Optional[dict] = None, lam0: Optional[Tensor] = None,
               generator: Optional[torch.Generator] = None, publish: bool = True) -> DecisionOutput:
        """Algorithm 1 DECIDE for every clone of the batch (lines 13-21).

        ``frames (B, 4, 128, 160)``, ``s_t (B, 13)`` raw proprioception,
        ``rel_goal_heading (B, 3)`` goal offset in the heading frame. With a
        Stage-A ``polytope`` (:func:`~cma_vpa.safety.degg_flow.build_stage_a_polytope`)
        the chunk is produced by GUIDEDSOLVE, otherwise by plain Euler steps.
        """
        self._chain, self._kv_calls = [], 0
        handles = self._attach_audit()
        try:
            z_t = self.encoder(frames)                                                   # 1
            self.tracker.step(z_t, self.milestones)
            g_t = goal_feature(rel_goal_heading.to(z_t.dtype))
            s_tilde = self.standardize_s(s_t.to(z_t.dtype))
            logits = self.selector(z_t, g_t, s_tilde, self.c_tilde, self.mode_emb(self.u_M))   # 2
            u_t = PrimitiveSelector.select(logits)
            pred = self.predictor(z_t, self.prim_emb(u_t), self.a_bar)                 # 3
            sigma_bar = self.filter.update(pred.sigma)
            xi = self.noise.draw(u_t, self.u_M, generator, dtype=z_t.dtype)
            e_t = self.conditioning(u_t, z_t, pred.z_hat, s_t.to(z_t.dtype), g_t, self.c_tilde, self.u_M)
            kv = self.bridge.read(now)
            guided = None
            if polytope is None:
                A = self.solver.sample(e_t, kv, xi)                                    # 4 .. K + 3
            else:
                guided = self.solver.guided_sample(e_t, kv, xi.to(polytope.G.dtype), polytope,
                                                   lam0=lam0, **(guidance_kw or {}))
                A = guided.A.to(z_t.dtype)
        finally:
            for h in handles:
                h.remove()
        if self._chain != self.expected_chain:
            raise RuntimeError(f"sequential-depth violation: {self._chain} != {self.expected_chain}")
        if self._kv_calls:
            raise RuntimeError(f"bridge K/V recomputed {self._kv_calls} times inside a decision")
        A_raw = A.clamp(-1.0, 1.0)
        alpha = self.filter.attenuation(sigma_bar)
        A_out = A_raw.clone()
        n_att = self.cfg.attenuate_channels
        A_out[..., :n_att] = A_out[..., :n_att] * alpha.view(-1, 1, 1)
        H_val = self.filter.validity(sigma_bar)
        r_eff = torch.clamp(self.safety.r_s * torch.exp(self.safety.kappa_sigma * sigma_bar), max=self.safety.r_max)
        if publish:
            self.chunks.publish(A_out, now, H_val)
        self.last_z = z_t
        self.a_bar = A_out[:, 0]
        return DecisionOutput(chunk=A_out, raw_chunk=A_raw, primitive=u_t, selector_logits=logits, z_t=z_t,
                              z_hat=pred.z_hat, sigma=pred.sigma, sigma_bar=sigma_bar, H_val=H_val, alpha=alpha,
                              r_eff=r_eff, cond=e_t, plan_valid=self.bridge.valid(now), noise_redrawn=self.noise.last_redraw,
                              chain=list(self._chain), guided=guided)

    def set_dispatched(self, a_bar: Tensor) -> None:
        """Record the most recently dispatched setpoint ``a_bar_t`` (normalized, ``(B, d_a)``)."""
        self.a_bar = a_bar.to(self.a_bar)

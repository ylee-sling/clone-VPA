# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""PR-JEPA master planner over the set of agents (Section 4.2, Eqs. 11-12, Algorithm 1 MASTERCYCLE).

Per cycle (``f_M in [1, 5]`` Hz, off the clone critical path):

1. decode every uplink heard in the cycle, ``x_hat^(i) = sum_b Q_b[k_b^(i)] in R^64``;
2. agent tokens ``chi_i = W_in [x_hat^(i) ; sigma_bar_i] in R^256`` **without** agent
   identity embeddings;
3. a set transformer ``J_Omega`` (4 layers of width 256: masked self-attention
   over the agents present, cross-attention to the mission tokens, MLP) gives
   per-agent features ``h_i`` and a pooled feature ``h_0``. It uses no positional
   or identity information, so it is permutation-equivariant (``h``) and
   -invariant (``h_0``) and the team size may change between cycles;
4. the global mode is **sampled**, ``u_g ~ pi^g(. | h_0)``, and the local modes are
   deterministic, ``u_M^i = argmax_u pi^loc(u | h_i, u_g)`` (Eq. 11), so that all
   randomness of the joint mode resides in ``u_g`` and ``TC(A | c, u_g) = 0``
   (Proposition 3(b), Proposition 7(c));
5. ``n_ens = 3`` plan heads predict each agent's bus-space latent at
   ``T_M = 1 s``; ``pi^(i)`` is their mean and ``sigma_bar_{M,i}`` their spread (Eq. 12)::

       pi^(i) = (1/n) sum_e G^(e)(h_i, Emb(u_g), Emb(u_M^i)),
       sigma_bar_{M,i} = (1 / (sqrt(d_pi) gamma_bar*)) sqrt((1/n) sum_e ||G^(e) - pi^(i)||^2);

6. ``pi^(i)`` is RVQ-encoded with the shared codebooks and sent in a 36-byte
   downlink (:meth:`MasterPlanner.downlinks`).

Mission tokens. Agent token ``i`` cross-attends to ``(c_tilde_text, z_g^(i, m^i))``:
the instruction token shared by all agents and the agent's own active milestone
latent. The milestone tokens are permuted together with the agents, so
equivariance is preserved without identity embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..comm.packet import DownlinkPacket, PacketError, SequenceFilter, UplinkPacket
from ..comm.rvq import ResidualVQ
from .predictor import EnsembleLinear

__all__ = ["MasterConfig", "SetLayer", "MasterPlanner", "MasterOutput", "handedness_of_mode"]


@dataclass
class MasterConfig:
    bus_dim: int = 64
    width: int = 256
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 3.0
    text_dim: int = 128
    latent_dim: int = 256
    n_global_modes: int = 4
    n_local_modes: int = 16
    mode_emb_dim: int = 64
    n_ens: int = 3
    plan_hidden: int = 256


class SetLayer(nn.Module):
    """Pre-norm masked self-attention over agents + cross-attention to per-agent mission tokens + MLP."""

    def __init__(self, width: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(width)
        hid = int(round(width * mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(width, hid), nn.GELU(), nn.Linear(hid, width))

    def forward(self, x: Tensor, pad: Tensor, mission: Tensor) -> Tensor:
        """``x (B, N, W)``, ``pad (B, N)`` True for absent agents, ``mission (B, N, T, W)``."""
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        B, N, W = x.shape
        q = self.norm2(x).reshape(B * N, 1, W)
        mem = mission.reshape(B * N, mission.shape[2], W)
        x = x + self.cross_attn(q, mem, mem, need_weights=False)[0].reshape(B, N, W)
        x = x + self.mlp(self.norm3(x))
        return x.masked_fill(pad.unsqueeze(-1), 0.0)


@dataclass
class MasterOutput:
    """One master cycle for a batch of teams (``B`` teams, ``N`` agent slots)."""

    h: Tensor               # (B, N, W)
    h0: Tensor              # (B, W)
    global_logits: Tensor   # (B, n_g)
    local_logits: Tensor    # (B, N, n_loc), conditioned on the sampled u_g
    u_g: Tensor             # (B,)
    u_M: Tensor             # (B, N)
    plan_heads: Tensor      # (n_ens, B, N, d_pi)
    plan: Tensor            # (B, N, d_pi)
    sigma_M: Tensor         # (B, N)
    codes: Optional[Tensor] = None   # (B, N, 24) uint8


def handedness_of_mode(u_g: Tensor) -> Tensor:
    """Passing side encoded by a global mode for crossings: even -> +1 (right), odd -> -1 (left)."""
    return 1 - 2 * (u_g.long() % 2)


class MasterPlanner(nn.Module):
    """``J_Omega`` with the global/local mode heads and the plan ensemble (4.4M parameters at the defaults)."""

    def __init__(self, cfg: MasterConfig = MasterConfig()):
        super().__init__()
        self.cfg = cfg
        W = cfg.width
        self.W_in = nn.Linear(cfg.bus_dim + 1, W)
        self.text_in = nn.Linear(cfg.text_dim, W)
        self.goal_in = nn.Linear(cfg.latent_dim, W)
        self.layers = nn.ModuleList([SetLayer(W, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(W)
        self.pool = nn.Sequential(nn.Linear(W, W), nn.GELU())
        self.global_head = nn.Sequential(nn.Linear(W, W), nn.GELU(), nn.Linear(W, cfg.n_global_modes))
        self.global_emb = nn.Embedding(cfg.n_global_modes, cfg.mode_emb_dim)
        self.local_head = nn.Sequential(nn.Linear(W + cfg.mode_emb_dim, W), nn.GELU(),
                                        nn.Linear(W, cfg.n_local_modes))
        self.local_emb = nn.Embedding(cfg.n_local_modes, cfg.mode_emb_dim)
        self.plan_heads = nn.Sequential(
            EnsembleLinear(cfg.n_ens, W + 2 * cfg.mode_emb_dim, cfg.plan_hidden), nn.GELU(),
            EnsembleLinear(cfg.n_ens, cfg.plan_hidden, cfg.bus_dim))
        self.register_buffer("gamma_star", torch.tensor(1.0))
        self.seq = 0
        self.uplink_filter = SequenceFilter(16)

    # ------------------------------------------------------------------ J_Omega
    def encode(self, x_hat: Tensor, sigma: Tensor, present: Tensor, c_text: Tensor, z_goal: Tensor):
        """``(h (B, N, W), h0 (B, W))`` from decoded uplinks.

        ``x_hat (B, N, d_pi)``, ``sigma (B, N)``, ``present (B, N)`` bool,
        ``c_text (B, 128)`` (projected instruction), ``z_goal (B, N, 256)``.
        """
        pad = ~present
        x = self.W_in(torch.cat([x_hat, sigma.unsqueeze(-1)], -1)).masked_fill(pad.unsqueeze(-1), 0.0)
        B, N = present.shape
        mission = torch.stack([self.text_in(c_text).unsqueeze(1).expand(B, N, -1), self.goal_in(z_goal)], 2)
        safe_pad = pad & ~pad.all(-1, keepdim=True)      # an empty team must not produce NaN attention
        for layer in self.layers:
            x = layer(x, safe_pad, mission)
        h = self.norm(x).masked_fill(pad.unsqueeze(-1), 0.0)
        cnt = present.sum(-1, keepdim=True).clamp(min=1).to(h.dtype)
        h0 = self.pool(h.sum(1) / cnt)
        return h, h0

    # ------------------------------------------------------------------ modes
    def global_logits(self, h0: Tensor) -> Tensor:
        return self.global_head(h0)

    def local_logits(self, h: Tensor, u_g: Tensor) -> Tensor:
        e = self.global_emb(u_g.long()).unsqueeze(1).expand(-1, h.shape[1], -1)
        return self.local_head(torch.cat([h, e], -1))

    def sample_modes(self, h: Tensor, h0: Tensor, generator: Optional[torch.Generator] = None,
                     greedy: bool = False, temperature: float = 1.0):
        """Eq. (11): ``u_g ~ pi^g(. | h0)`` (or its argmax if ``greedy``), ``u_M = argmax pi^loc``."""
        gl = self.global_logits(h0)
        if greedy:
            u_g = gl.argmax(-1)
        else:
            p = F.softmax(gl / temperature, -1)
            u_g = torch.multinomial(p, 1, generator=generator).squeeze(-1)
        ll = self.local_logits(h, u_g)
        return u_g, ll.argmax(-1), gl, ll

    # ------------------------------------------------------------------ plans
    def plan(self, h: Tensor, u_g: Tensor, u_M: Tensor):
        """Eq. (12): ``(heads (n, B, N, d_pi), pi (B, N, d_pi), sigma_M (B, N))``."""
        B, N, W = h.shape
        x = torch.cat([h, self.global_emb(u_g.long()).unsqueeze(1).expand(B, N, -1),
                       self.local_emb(u_M.long())], -1)
        heads = self.plan_heads(x.reshape(1, B * N, -1).expand(self.cfg.n_ens, -1, -1))
        heads = heads.reshape(self.cfg.n_ens, B, N, self.cfg.bus_dim)
        pi = heads[0] + (heads - heads[:1]).mean(0)
        spread = (heads - pi.unsqueeze(0)).pow(2).sum(-1).mean(0).sqrt()
        sigma_M = spread / (math.sqrt(self.cfg.bus_dim) * self.gamma_star.to(spread.dtype))
        return heads, pi, sigma_M

    def forward(self, x_hat: Tensor, sigma: Tensor, present: Tensor, c_text: Tensor, z_goal: Tensor,
                u_g: Optional[Tensor] = None, u_M: Optional[Tensor] = None,
                generator: Optional[torch.Generator] = None, greedy: bool = False) -> MasterOutput:
        """Full cycle; passing ``u_g`` / ``u_M`` (labels) gives teacher forcing for Stage 3."""
        h, h0 = self.encode(x_hat, sigma, present, c_text, z_goal)
        gl = self.global_logits(h0)
        if u_g is None:
            u_g, u_M_s, gl, ll = self.sample_modes(h, h0, generator, greedy)
            u_M = u_M_s if u_M is None else u_M
        else:
            ll = self.local_logits(h, u_g)
            if u_M is None:
                u_M = ll.argmax(-1)
        heads, pi, sigma_M = self.plan(h, u_g, u_M)
        return MasterOutput(h, h0, gl, ll, u_g, u_M, heads, pi, sigma_M)

    # ------------------------------------------------------------------ deployment cycle
    @torch.no_grad()
    def cycle(self, rvq: ResidualVQ, uplink_codes: Tensor, sigma: Tensor, present: Tensor, c_text: Tensor,
              z_goal: Tensor, generator: Optional[torch.Generator] = None, greedy: bool = False) -> MasterOutput:
        """MASTERCYCLE lines 2-5 from raw uplink codes ``(B, N, 24)``; the plan codes are attached."""
        x_hat = rvq.decode(uplink_codes)
        out = self(x_hat, sigma, present, c_text, z_goal, generator=generator, greedy=greedy)
        out.codes = rvq.encode(out.plan)
        return out

    def decode_uplinks(self, packets: Sequence[bytes], agent_order: Sequence[int], rvq: ResidualVQ,
                       check_sequence: bool = True) -> Dict[str, Tensor]:
        """Validate and decode uplinks into slot order ``agent_order`` (absent agents masked).

        Packets failing the length/CRC check, or older than the last accepted
        sequence number of their sender, are dropped.
        """
        slot = {a: k for k, a in enumerate(agent_order)}
        N = len(agent_order)
        codes = torch.zeros(N, 24, dtype=torch.uint8)
        sig = torch.zeros(N)
        present = torch.zeros(N, dtype=torch.bool)
        flags = torch.zeros(N, dtype=torch.long)
        for raw in packets:
            try:
                p = UplinkPacket.unpack(raw)
            except PacketError:
                continue
            if p.agent_id not in slot or (check_sequence and not self.uplink_filter.accept(p.agent_id, p.seq)):
                continue
            k = slot[p.agent_id]
            codes[k] = torch.tensor(list(p.code), dtype=torch.uint8)
            sig[k] = p.sigma_bar
            present[k] = True
            flags[k] = p.flags
        return {"codes": codes, "sigma": sig, "present": present, "flags": flags,
                "x_hat": rvq.decode(codes)}

    def downlinks(self, out: MasterOutput, agent_order: Sequence[int], t_s: float, milestones: Tensor,
                  validity_s: float, team: int = 0, status: int = 0) -> List[bytes]:
        """Pack one 36-byte downlink per present agent of team ``team`` (Algorithm 1, line 6)."""
        if out.codes is None:
            raise ValueError("MasterOutput has no plan codes; use cycle()")
        msgs = []
        self.seq = (self.seq + 1) % 65536
        for k, a in enumerate(agent_order):
            msgs.append(DownlinkPacket.build(
                agent_id=int(a), seq=self.seq, t_s=t_s, mode=int(out.u_M[team, k]),
                milestone=int(milestones[k]), code=out.codes[team, k].tolist(),
                sigma_M=float(out.sigma_M[team, k]), validity_s=validity_s, status=status).pack())
        return msgs

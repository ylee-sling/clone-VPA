# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Cross-attention bridge (Section 4.4, Eq. 16): plan tokens and their double-buffered K/V cache.

On arrival of a downlink message the clone decodes ``pi_hat = sum_b Q_b[k_b]`` and
forms ``M_p + 2 = 6`` plan tokens::

    Z_tilde = (1 / gamma_bar*) [ W_pi pi_hat (reshaped to M_p = 4 tokens) ; Emb(u_M) ; z_g^(i, m^i) ]

with ``W_pi : R^64 -> R^{4 x 256}`` and the mode table ``Emb : {0..15} -> R^256``
(together the 70,656-parameter plan-token decoder of Table 3). Scaling by the
constant ``1/gamma_bar*`` instead of a LayerNorm keeps the map from plan to
tokens affine, hence Lipschitz (Proposition 4(d)).

For each cross-attention layer ``l`` of ``v_theta`` the keys and values
``K^(l) = Z_tilde W_K^(l)``, ``V^(l) = Z_tilde W_V^(l)`` are computed once and written
to the back slot of a double buffer; the front index is then flipped, so the
decision loop always reads a complete pair and never waits (Algorithm 1,
lines 8-10). Once a plan exceeds its transmitted validity, or before any plan
has arrived, :meth:`BridgeCache.read` returns the K/V of the learned null token,
which degrades the clone to decentralized operation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .solver import FlowVectorField, PlanKV

__all__ = ["PlanTokenDecoder", "BridgeCache"]


class PlanTokenDecoder(nn.Module):
    """``W_pi`` (64 -> 4 x 256) and the 16 x 256 mode table of Eq. (16)."""

    def __init__(self, bus_dim: int = 64, width: int = 256, n_plan_tokens: int = 4, n_modes: int = 16):
        super().__init__()
        self.bus_dim, self.width, self.n_plan_tokens, self.n_modes = bus_dim, width, n_plan_tokens, n_modes
        self.W_pi = nn.Linear(bus_dim, n_plan_tokens * width)
        self.mode_table = nn.Embedding(n_modes, width)

    @property
    def n_tokens(self) -> int:
        return self.n_plan_tokens + 2

    def forward(self, pi_hat: Tensor, mode: Tensor, z_goal: Tensor, gamma_star: float) -> Tensor:
        """Plan tokens ``(B, 6, 256)`` from ``pi_hat (B, 64)``, ``mode (B,)`` and ``z_goal (B, 256)``."""
        if z_goal.shape[-1] != self.width:
            raise ValueError("milestone latent width must equal the token width (d = 256)")
        B = pi_hat.shape[0]
        plan = self.W_pi(pi_hat).view(B, self.n_plan_tokens, self.width)
        toks = torch.cat([plan, self.mode_table(mode.long()).unsqueeze(1), z_goal.unsqueeze(1)], 1)
        return toks / float(gamma_star)


class BridgeCache:
    """Double-buffered K/V of the plan tokens for a batch of clones.

    Storage is ``(2, B, L, T, W)`` for keys and values (``L`` cross-attention
    layers, ``T`` plan tokens); ``front[b]`` selects the slot that readers use.
    ``valid_until[b]`` is the receiver-clock time at which the plan of agent ``b``
    expires. ``n_writes`` counts K/V computations (one per accepted message).
    """

    def __init__(self, field: FlowVectorField, batch: int, n_tokens: int = 6,
                 dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None):
        self.field, self.B, self.T = field, batch, n_tokens
        L, W = field.n_cross, field.width
        self.K = torch.zeros(2, batch, L, n_tokens, W, dtype=dtype, device=device)
        self.V = torch.zeros_like(self.K)
        self.front = torch.zeros(batch, dtype=torch.long, device=device)
        self.valid_until = torch.full((batch,), -math.inf, dtype=torch.float64, device=device)
        self.n_writes = 0
        self._null: Optional[PlanKV] = None
        self._ar = torch.arange(batch, device=device)

    @torch.no_grad()
    def refresh_null(self) -> PlanKV:
        """(Re)compute the null-token K/V (after loading weights)."""
        self._null = self.field.null_kv(self.B, self.T).detach()
        return self._null

    @torch.no_grad()
    def write(self, mask: Tensor, plan_tokens: Tensor, valid_until: Tensor) -> None:
        """Compute K/V for the agents in ``mask`` from ``plan_tokens (B, T, W)`` and flip their buffer."""
        sel = mask.nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return
        kv = self.field.project_kv(plan_tokens[sel].to(self.K.dtype))
        self.n_writes += 1
        back = 1 - self.front[sel]
        self.K[back, sel] = torch.stack(kv.K, 1)
        self.V[back, sel] = torch.stack(kv.V, 1)
        self.front[sel] = back
        self.valid_until[sel] = valid_until[sel].to(self.valid_until)

    def invalidate(self, mask: Tensor) -> None:
        """Mark the plans of ``mask`` as expired (the null plan is used from now on)."""
        self.valid_until = torch.where(mask, torch.full_like(self.valid_until, -math.inf), self.valid_until)

    def valid(self, now: float) -> Tensor:
        return self.valid_until >= now

    def read(self, now: float) -> PlanKV:
        """K/V of the front slot for valid plans, of the null token otherwise."""
        null = self._null if self._null is not None else self.refresh_null()
        Kf = self.K[self.front, self._ar]                 # (B, L, T, W)
        Vf = self.V[self.front, self._ar]
        cur = PlanKV(tuple(Kf[:, l] for l in range(Kf.shape[1])), tuple(Vf[:, l] for l in range(Vf.shape[1])))
        return cur.select(self.valid(now).to(Kf.device), null)

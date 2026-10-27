# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Conditional flow-matching action-chunk solver ``v_theta`` (Section 4.5, Eqs. 17-18).

``v_theta(A, tau | e_t, Z_tilde)`` is a non-causal transformer over the ``H = 32``
knots of a chunk ``A in R^{H x d_a}`` (``d_a = 4``: heading-frame velocity and yaw
rate, normalized to ``[-1, 1]``). Every evaluation processes the whole chunk in
parallel. Exactly the 3,915,524 parameters of Table 3::

    action input        Linear(4, 256)                                   1,280
    position table      33 x 256  (conditioning token + 32 knots)        8,448
    conditioning input  Linear(721 + 128, 256)  on [e_t ; time(tau)]   217,600
    4 self-attention    pre-norm blocks, width 256, 4 heads, MLP 1024 3,159,040
    2 cross-attention   q, k, v, o projections + query LayerNorm        527,360
    output              LayerNorm + Linear(256, 4)                       1,540
    null plan token     1 x 256                                            256

The two cross-attention layers (after self-attention blocks 2 and 4) attend to
the six plan tokens of the bridge (Eq. 16). Their keys and values
``K^(l) = Z_tilde W_K^(l)``, ``V^(l) = Z_tilde W_V^(l)`` are computed **once per
downlink message** by :meth:`FlowVectorField.project_kv` and passed into every
solver evaluation as a :class:`PlanKV`, so the bridge adds no sequential
network evaluation (Proposition 8): inside an evaluation only the query and
output projections of the cross-attention run.

Training (Eq. 17)::

    L_CFM = E || v_theta(tau A + (1 - tau) xi, tau | e, Z) - (A - xi) ||_F^2,   tau ~ U[0, 1].

Sampling (Eq. 18): ``K`` explicit Euler steps ``A_{k+1} = A_k + v_theta(A_k, k/K) / K``
from ``A_0 = xi``; :meth:`FlowSolver.guided_sample` runs the same loop through
GUIDEDSOLVE (Algorithm 2) with Stage-A endpoint projection.

:class:`NoisePersistence` keeps ``xi_t = xi_{t-1}`` unless the local primitive or
the master mode changed or ``t = 0 (mod N_xi)`` (Section 4.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "SinusoidalTime",
    "CachedCrossAttention",
    "PlanKV",
    "FlowVectorField",
    "FlowSolver",
    "NoisePersistence",
]


class SinusoidalTime(nn.Module):
    """Sinusoidal features of the flow time ``tau in [0, 1]``: ``(B,) -> (B, dim)``."""

    def __init__(self, dim: int = 128, scale: float = 1000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding width must be even")
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000.0) * torch.arange(half) / half),
                             persistent=False)
        self.scale = scale

    def forward(self, tau: Tensor) -> Tensor:
        a = self.scale * tau.unsqueeze(-1) * self.freqs.to(tau.dtype)
        return torch.cat([a.sin(), a.cos()], -1)


@dataclass
class PlanKV:
    """Cached keys and values of the cross-attention layers: one ``(B, T, W)`` pair per layer."""

    K: Tuple[Tensor, ...]
    V: Tuple[Tensor, ...]

    def select(self, mask: Tensor, other: "PlanKV") -> "PlanKV":
        """Per-agent choice ``mask ? self : other``."""
        m = mask.view(-1, 1, 1)
        return PlanKV(tuple(torch.where(m, a, b) for a, b in zip(self.K, other.K)),
                      tuple(torch.where(m, a, b) for a, b in zip(self.V, other.V)))

    def detach(self) -> "PlanKV":
        return PlanKV(tuple(k.detach() for k in self.K), tuple(v.detach() for v in self.V))


class CachedCrossAttention(nn.Module):
    """Pre-norm residual cross-attention ``x + O softmax(Q K^T / sqrt(d_k)) V`` with external K, V."""

    def __init__(self, width: int = 256, heads: int = 4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by the number of heads")
        self.width, self.heads = width, heads
        self.norm_q = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)

    def project_kv(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        """``K = Z W_K``, ``V = Z W_V`` for plan tokens ``(B, T, W)`` (once per message)."""
        return self.k_proj(tokens), self.v_proj(tokens)

    def forward(self, x: Tensor, K: Tensor, V: Tensor) -> Tensor:
        B, L, W = x.shape
        h, dk = self.heads, W // self.heads
        q = self.q_proj(self.norm_q(x)).view(B, L, h, dk).transpose(1, 2)
        k = K.view(B, K.shape[1], h, dk).transpose(1, 2)
        v = V.view(B, V.shape[1], h, dk).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, W)
        return x + self.o_proj(out)


class FlowVectorField(nn.Module):
    """``v_theta``: width 256, depth 4, two KV-cached cross-attention layers, 33 tokens."""

    def __init__(self, horizon: int = 32, action_dim: int = 4, cond_dim: int = 721, width: int = 256,
                 depth: int = 4, heads: int = 4, mlp_ratio: float = 4.0, time_dim: int = 128,
                 cross_after: Sequence[int] = (1, 3)):
        super().__init__()
        if any(not 0 <= c < depth for c in cross_after):
            raise ValueError("cross_after indices must refer to self-attention blocks")
        self.horizon, self.action_dim, self.cond_dim, self.width = horizon, action_dim, cond_dim, width
        self.cross_after = tuple(cross_after)
        self.action_in = nn.Linear(action_dim, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon + 1, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_embed = SinusoidalTime(time_dim)
        self.cond_in = nn.Linear(cond_dim + time_dim, width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, int(round(width * mlp_ratio)), dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.cross = nn.ModuleList([CachedCrossAttention(width, heads) for _ in self.cross_after])
        self.norm = nn.LayerNorm(width)
        self.action_out = nn.Linear(width, action_dim)
        self.null_token = nn.Parameter(torch.zeros(1, width))
        nn.init.trunc_normal_(self.null_token, std=0.02)

    @property
    def n_cross(self) -> int:
        return len(self.cross)

    def project_kv(self, plan_tokens: Tensor) -> PlanKV:
        """Bridge keys and values for every cross-attention layer (on message arrival)."""
        ks, vs = zip(*(layer.project_kv(plan_tokens) for layer in self.cross))
        return PlanKV(tuple(ks), tuple(vs))

    def null_kv(self, batch: int, n_tokens: int = 6) -> PlanKV:
        """KV of the learned null plan (used under plan dropout and after a plan expires)."""
        tok = self.null_token.unsqueeze(0).expand(batch, n_tokens, self.width)
        return self.project_kv(tok)

    def forward(self, actions: Tensor, tau: Union[float, Tensor], cond: Tensor, kv: PlanKV) -> Tensor:
        """``(B, H, d_a), tau, (B, d_e), PlanKV -> (B, H, d_a)``. One network evaluation."""
        B = actions.shape[0]
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"expected actions (B, {self.horizon}, {self.action_dim}), got {tuple(actions.shape)}")
        if cond.shape != (B, self.cond_dim):
            raise ValueError(f"expected conditioning (B, {self.cond_dim}), got {tuple(cond.shape)}")
        tau_t = torch.as_tensor(tau, dtype=actions.dtype, device=actions.device)
        if tau_t.dim() == 0:
            tau_t = tau_t.expand(B)
        c = self.cond_in(torch.cat([cond, self.time_embed(tau_t)], -1)).unsqueeze(1)    # (B, 1, W)
        x = torch.cat([c, self.action_in(actions) + c], 1) + self.pos_embed
        j = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if j < len(self.cross_after) and i == self.cross_after[j]:
                x = self.cross[j](x, kv.K[j], kv.V[j])
                j += 1
        return self.action_out(self.norm(x[:, 1:]))


class FlowSolver:
    """Training loss and samplers around a :class:`FlowVectorField` (holds no parameters of its own)."""

    def __init__(self, field: FlowVectorField, K: int = 2):
        if K < 1:
            raise ValueError("K must be at least 1")
        self.field, self.K = field, K

    def cfm_loss(self, actions: Tensor, cond: Tensor, kv: PlanKV,
                 generator: Optional[torch.Generator] = None) -> Tensor:
        """Eq. (17): squared Frobenius norm over the chunk, mean over the batch."""
        xi = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
        tau = torch.rand(actions.shape[0], generator=generator, device=actions.device, dtype=actions.dtype)
        a_tau = tau.view(-1, 1, 1) * actions + (1.0 - tau.view(-1, 1, 1)) * xi
        v = self.field(a_tau, tau, cond, kv)
        return (v - (actions - xi)).pow(2).sum((1, 2)).mean()

    def sample(self, cond: Tensor, kv: PlanKV, xi: Tensor, K: Optional[int] = None) -> Tensor:
        """Unguided Euler integration of Eq. (18): exactly ``K`` field evaluations."""
        K = self.K if K is None else K
        a = xi
        for k in range(K):
            a = a + self.field(a, k / K, cond, kv) / K
        return a

    def guided_sample(self, cond: Tensor, kv: PlanKV, xi: Tensor, polytope, K: Optional[int] = None,
                      nu: Optional[Sequence[float]] = None, lam0: Optional[Tensor] = None, **kw):
        """GUIDEDSOLVE (Algorithm 2) with Stage-A endpoint projection; returns ``GuidedSolveResult``."""
        from ..safety.degg_flow import guided_solve

        K = self.K if K is None else K

        def velocity(a: Tensor, tau: float) -> Tensor:
            return self.field(a.to(cond.dtype), tau, cond, kv).to(a.dtype)

        return guided_solve(velocity, xi, polytope, K, nu, lam0=lam0, **kw)


class NoisePersistence:
    """Initial-noise persistence ``xi_t = xi_{t-1}`` (Section 4.5), batched over agents.

    The noise is redrawn for an agent when its primitive or master mode changed,
    when ``t = 0 (mod N_xi)`` (``t`` counts decisions), or at the first decision.
    """

    def __init__(self, shape: Tuple[int, int], N_xi: int = 30):
        self.shape, self.N_xi = shape, N_xi
        self.xi: Optional[Tensor] = None
        self.prim: Optional[Tensor] = None
        self.mode: Optional[Tensor] = None
        self.t = 0
        self.last_redraw: Optional[Tensor] = None

    def reset(self) -> None:
        self.xi = self.prim = self.mode = self.last_redraw = None
        self.t = 0

    def draw(self, prim: Tensor, mode: Tensor, generator: Optional[torch.Generator] = None,
             dtype: torch.dtype = torch.float32) -> Tensor:
        B = prim.shape[0]
        fresh = torch.randn(B, *self.shape, generator=generator, device=prim.device, dtype=dtype)
        if self.xi is None:
            redraw = torch.ones(B, dtype=torch.bool, device=prim.device)
        else:
            redraw = (prim != self.prim) | (mode != self.mode)
            if self.t % self.N_xi == 0:
                redraw = torch.ones_like(redraw)
        self.xi = fresh if self.xi is None else torch.where(redraw.view(-1, 1, 1), fresh, self.xi.to(dtype))
        self.prim, self.mode, self.last_redraw = prim.clone(), mode.clone(), redraw
        self.t += 1
        return self.xi

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Siamese non-reconstructive ViT encoder ``E_psi`` (Section 3.4, Eq. 7, Table 3) and text projection.

``E_psi`` maps an RGB-D frame ``I in R^{4 x 128 x 160}`` to ``z = E_psi(I) in R^256``.
It is Siamese in the three senses of Section 3.4: the same weights embed
current frames and milestone goal frames, and every clone holds identical
weights. It is trained without any reconstruction objective (Stage 1, with the
JEPA loss and an EMA target copy, :class:`EMATarget`).

Architecture (exactly the 11,171,200 parameters of Table 3)::

    patch embedding   Conv2d(4, 384, 16, stride 16)           393,600
    [CLS] token       1 x 384                                     384
    position table    81 x 384  (80 patches + [CLS])           31,104
    6 pre-norm blocks width 384, 6 heads, MLP 1536         10,646,784
    final LayerNorm   384                                         768
    head              Linear(384, 256)  on the [CLS] token     98,560

The text projection ``P_c : R^512 -> R^128`` (65,664 parameters) maps a frozen
CLIP text embedding once per mission. :class:`HashedTextEmbedder` is a
dependency-free deterministic stand-in for CLIP used by the tests and the
simulator demo; :class:`ClipTextEmbedder` wraps a Hugging Face CLIP model when
``transformers`` and the weights are available.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "SiameseViTEncoder",
    "TextProjection",
    "HashedTextEmbedder",
    "ClipTextEmbedder",
    "EMATarget",
    "momentum_schedule",
    "latent_scale",
]


class SiameseViTEncoder(nn.Module):
    """ViT-S/16-style encoder (width 384, depth 6, d = 256) without reconstruction.

    Parameters
    ----------
    image_size:
        ``(H, W)`` of the input; both must be multiples of ``patch_size``.
    in_channels:
        4 for RGB-D.
    width, depth, heads, mlp_ratio:
        Transformer hyper-parameters.
    latent_dim:
        ``d``.
    """

    def __init__(self, image_size: Tuple[int, int] = (128, 160), patch_size: int = 16, in_channels: int = 4,
                 width: int = 384, depth: int = 6, heads: int = 6, mlp_ratio: float = 4.0,
                 latent_dim: int = 256):
        super().__init__()
        h, w = image_size
        if h % patch_size or w % patch_size:
            raise ValueError("image size must be a multiple of the patch size")
        self.image_size, self.patch_size, self.in_channels = (h, w), patch_size, in_channels
        self.latent_dim, self.width = latent_dim, width
        n_patches = (h // patch_size) * (w // patch_size)
        self.patch_embed = nn.Conv2d(in_channels, width, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, int(round(width * mlp_ratio)), dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, latent_dim)

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        return (self.in_channels, *self.image_size)

    def forward(self, frames: Tensor) -> Tensor:
        """``(B, C, H, W) -> (B, d)``. One evaluation on the decision chain."""
        if frames.shape[1:] != self.frame_shape:
            raise ValueError(f"expected frames (B, {self.frame_shape}), got {tuple(frames.shape)}")
        x = self.patch_embed(frames).flatten(2).transpose(1, 2)              # (B, P, W)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])

    @torch.no_grad()
    def encode_milestones(self, frames: Tensor, chunk: int = 64) -> Tensor:
        """Milestone latents ``z_g`` for ``(..., C, H, W)`` frames (once per mission, off the chain)."""
        lead = frames.shape[:-3]
        flat = frames.reshape(-1, *self.frame_shape)
        out = torch.cat([self(flat[i:i + chunk]) for i in range(0, flat.shape[0], chunk)], 0)
        return out.reshape(*lead, self.latent_dim)


class TextProjection(nn.Module):
    """``c_tilde = P_c c_text``: ``R^512 -> R^128`` (Table 3), once per mission."""

    def __init__(self, text_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.text_dim, self.out_dim = text_dim, out_dim
        self.proj = nn.Linear(text_dim, out_dim)

    def forward(self, c_text: Tensor) -> Tensor:
        return self.proj(c_text)


class HashedTextEmbedder:
    """Deterministic, dependency-free stand-in for a frozen CLIP text encoder.

    Each whitespace token is hashed (SHA-256) into a seed for a unit Gaussian
    vector; the instruction embedding is the normalized sum. Identical strings
    map to identical vectors on every machine, which is all the pipeline needs
    from a *frozen* text encoder.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _token(self, tok: str) -> Tensor:
        seed = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:8], "little")
        g = torch.Generator().manual_seed(seed)
        return torch.randn(self.dim, generator=g)

    def __call__(self, texts: Union[str, Sequence[str]]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            toks = t.lower().split() or [""]
            v = torch.stack([self._token(k) for k in toks]).sum(0)
            out.append(v / v.norm().clamp(min=1e-12))
        return torch.stack(out)


class ClipTextEmbedder:
    """Frozen CLIP text embedding (``transformers`` required; weights loaded lazily)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        from transformers import CLIPModel, CLIPTokenizer   # optional dependency

        self.tok = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).eval()
        if device:
            self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def __call__(self, texts: Union[str, Sequence[str]]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        batch = self.tok(list(texts), padding=True, return_tensors="pt").to(self.model.device)
        return self.model.get_text_features(**batch)


class EMATarget:
    """Exponential-moving-average target copy ``E_psi_bar`` (Stage 1, [15, 26]).

    ``update(online, m)`` sets ``theta_bar <- m theta_bar + (1 - m) theta``; the
    target never receives gradients (stop-gradient by construction).
    """

    def __init__(self, online: nn.Module):
        self.module = copy.deepcopy(online).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module, momentum: float) -> None:
        for pt, po in zip(self.module.parameters(), online.parameters()):
            pt.mul_(momentum).add_(po.detach(), alpha=1.0 - momentum)
        for bt, bo in zip(self.module.buffers(), online.buffers()):
            bt.copy_(bo)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        return self.module(x)


def momentum_schedule(step: int, total: int, m0: float = 0.996, m1: float = 1.0) -> float:
    """Cosine schedule of the EMA momentum from ``m0`` to ``m1`` over ``total`` steps."""
    if total <= 1:
        return m1
    t = min(max(step / (total - 1), 0.0), 1.0)
    return m1 - (m1 - m0) * 0.5 * (1.0 + math.cos(math.pi * t))


def latent_scale(Z: Tensor, eps: float = 1e-4) -> float:
    """``gamma_bar = sqrt(mean_j Var(Z_.j) + eps)`` (VPA Eq. 9b), frozen as ``gamma_bar*`` after Stage 1."""
    if Z.dim() != 2 or Z.shape[0] < 2:
        raise ValueError("latent_scale needs a batch (N >= 2, d)")
    return float(torch.sqrt(Z.double().var(0, unbiased=True).mean() + eps))

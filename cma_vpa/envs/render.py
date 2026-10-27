# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Procedural RGB-D renderer for the front-facing camera of every vehicle (stand-in for Isaac Sim).

Produces ``4 x H x W`` frames (default ``128 x 160``, Table 3) from simulator
state, batched over vehicles, so that the encoder can be trained and run in
closed loop without a game engine:

* channel 0 (R) - other vehicles as anti-aliased discs of their physical radius;
* channel 1 (G) - the agent's own goal beacon (a 0.5 m sphere at the goal);
* channel 2 (B) - ground plane shading by distance (gives altitude and attitude cues);
* channel 3 (D) - inverse depth ``1 / (1 + z)`` of the nearest surface along each ray.

Camera: pinhole with horizontal field of view ``fov_deg`` along the vehicle's
yaw, level (zero pitch and roll), image ``y`` axis pointing down.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = ["render_rgbd", "render_spheres", "look_at", "to_uint8", "from_uint8"]


def render_rgbd(pos: Tensor, yaw: Tensor, others: Tensor, others_radius: Tensor, goal: Tensor, *,
                image_size: Tuple[int, int] = (128, 160), fov_deg: float = 90.0, goal_radius: float = 0.5,
                others_mask: Optional[Tensor] = None, max_ground_range: float = 50.0) -> Tensor:
    """Render ``(B, 4, H, W)`` frames in ``[0, 1]``.

    Parameters
    ----------
    pos, yaw:
        Camera positions ``(B, 3)`` and yaws ``(B,)`` (world ENU).
    others, others_radius:
        Positions ``(B, J, 3)`` and radii ``(B, J)`` of the other vehicles seen by each camera.
    goal:
        Goal positions ``(B, 3)``.
    others_mask:
        Optional ``(B, J)`` visibility mask (e.g. vehicles that exist).
    """
    H, W = image_size
    dev, dt = pos.device, torch.float32
    pos, yaw, others, goal = pos.to(dt), yaw.to(dt), others.to(dt), goal.to(dt)
    f = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    fwd = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)        # (B, 3)
    right = torch.stack([torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)], -1)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=dt, device=dev).expand_as(fwd)

    objs = torch.cat([others, goal.unsqueeze(1)], 1)                                       # (B, J+1, 3)
    rad = torch.cat([others_radius.to(dt), torch.full_like(others_radius[:, :1], goal_radius, dtype=dt)], 1)
    vis = torch.ones(objs.shape[:2], dtype=torch.bool, device=dev)
    if others_mask is not None:
        vis[:, :-1] = others_mask
    rel = objs - pos.unsqueeze(1)
    xc = (rel * right.unsqueeze(1)).sum(-1)
    yc = (rel * down.unsqueeze(1)).sum(-1)
    zc = (rel * fwd.unsqueeze(1)).sum(-1)
    front = (zc > 0.2) & vis
    zs = zc.clamp(min=0.2)
    u = cx + f * xc / zs
    v = cy + f * yc / zs
    rpx = (f * rad / zs).clamp(min=0.5)

    uu = torch.arange(W, dtype=dt, device=dev).view(1, 1, 1, W)
    vv = torch.arange(H, dtype=dt, device=dev).view(1, 1, H, 1)
    dist = torch.sqrt((uu - u[..., None, None]) ** 2 + (vv - v[..., None, None]) ** 2)        # (B, J+1, H, W)
    cover = torch.sigmoid((rpx[..., None, None] - dist) / 0.7) * front[..., None, None]
    occ_others = cover[:, :-1].amax(1) if cover.shape[1] > 1 else torch.zeros_like(cover[:, 0])
    occ_goal = cover[:, -1]

    # ground plane z = 0 seen from altitude h with a level camera
    h = pos[:, 2].clamp(min=0.05).view(-1, 1, 1)
    below = (vv[0] - cy).clamp(min=1e-3)                                                    # (1, H, 1)
    ground_depth = torch.where(vv[0] > cy, h * f / below, torch.full_like(below, math.inf))   # (B, H, 1)
    ground = torch.clamp(1.0 - ground_depth / max_ground_range, 0.0, 1.0).expand(-1, H, W)

    z_obj = torch.where(cover > 0.5, zs[..., None, None].expand_as(cover), torch.full_like(cover, math.inf))
    z_near = torch.minimum(z_obj.amin(1), ground_depth.expand(-1, H, W))
    depth = torch.where(torch.isfinite(z_near), 1.0 / (1.0 + z_near), torch.zeros_like(z_near))
    return torch.stack([occ_others, occ_goal, ground, depth], 1).clamp(0.0, 1.0)


def look_at(cam_pos: Tensor, target: Tensor, up: Optional[Tensor] = None) -> Tensor:
    """Camera rotation ``(..., 3, 3)`` with columns (right, down, forward) looking from ``cam_pos`` at ``target``."""
    fwd = target - cam_pos
    fwd = fwd / torch.linalg.vector_norm(fwd, dim=-1, keepdim=True).clamp(min=1e-12)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=fwd.dtype, device=fwd.device).expand_as(fwd) if up is None else up
    right = torch.cross(fwd, up, dim=-1)
    right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp(min=1e-12)
    down = torch.cross(fwd, right, dim=-1)
    return torch.stack([right, down, fwd], -1)


def render_spheres(cam_pos: Tensor, cam_R: Tensor, centers: Tensor, radii: Tensor, colors: Tensor, *,
                   mask: Optional[Tensor] = None, image_size: Tuple[int, int] = (128, 160), fov_deg: float = 70.0,
                   ground_z: Optional[float] = 0.0, ground_color: Tuple[float, float, float] = (0.55, 0.45, 0.35),
                   max_range: float = 3.0) -> Tensor:
    """Z-buffered RGB-D rendering of a sphere scene above a ground plane: ``(B, 4, H, W)`` in ``[0, 1]``.

    ``cam_pos (B, 3)``, ``cam_R (B, 3, 3)`` with columns (right, down, forward)
    (:func:`look_at`), ``centers (B, J, 3)``, ``radii (B, J)``, ``colors (B, J, 3)``.
    Channels: RGB and inverse depth ``1 / (1 + z)`` (0 where nothing is hit).
    """
    H, W = image_size
    dt, dev = torch.float32, cam_pos.device
    cam_pos, cam_R, centers = cam_pos.to(dt), cam_R.to(dt), centers.to(dt)
    radii, colors = radii.to(dt), colors.to(dt)
    f = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    rel = torch.einsum("bji,bik->bjk", centers - cam_pos.unsqueeze(1), cam_R)       # camera coordinates
    xc, yc, zc = rel[..., 0], rel[..., 1], rel[..., 2]
    vis = zc > 0.02
    if mask is not None:
        vis = vis & mask
    zs = zc.clamp(min=0.02)
    u = cx + f * xc / zs
    v = cy + f * yc / zs
    rpx = (f * radii / zs).clamp(min=0.5)
    uu = torch.arange(W, dtype=dt, device=dev).view(1, 1, 1, W)
    vv = torch.arange(H, dtype=dt, device=dev).view(1, 1, H, 1)
    dist = torch.sqrt((uu - u[..., None, None]) ** 2 + (vv - v[..., None, None]) ** 2)    # (B, J, H, W)
    cover = torch.sigmoid((rpx[..., None, None] - dist) / 0.7) * vis[..., None, None]
    z_front = (zs - radii).clamp(min=0.01)[..., None, None].expand_as(cover)
    z_obj = torch.where(cover > 0.5, z_front, torch.full_like(cover, math.inf))
    z_min, j_min = z_obj.min(1)                                                          # (B, H, W)
    col_obj = torch.gather(colors, 1, j_min.flatten(1).unsqueeze(-1).expand(-1, -1, 3)).view(
        cover.shape[0], H, W, 3).permute(0, 3, 1, 2)
    alpha = torch.gather(cover, 1, j_min.unsqueeze(1)).squeeze(1)
    # ground plane
    B = cam_pos.shape[0]
    if ground_z is not None:
        dirs = torch.stack([(uu[0, 0] - cx).expand(H, W) / f, (vv[0, 0] - cy).expand(H, W) / f,
                            torch.ones(H, W, dtype=dt, device=dev)], -1)                 # (H, W, 3) camera frame
        dw = torch.einsum("bik,hwk->bhwi", cam_R, dirs)                                   # world directions
        t = (ground_z - cam_pos[:, 2]).view(B, 1, 1) / torch.where(dw[..., 2].abs() > 1e-9, dw[..., 2],
                                                                     torch.full_like(dw[..., 2], -1e-9))
        hit = (dw[..., 2] < 0) & (t > 0) & (t < max_range)
        z_g = torch.where(hit, t, torch.full_like(t, math.inf))
        shade = torch.clamp(1.0 - z_g / max_range, 0.2, 1.0)
        gcol = torch.tensor(ground_color, dtype=dt, device=dev).view(1, 3, 1, 1) * shade.unsqueeze(1)
        gcol = torch.where(hit.unsqueeze(1), gcol, torch.zeros_like(gcol))
    else:
        z_g = torch.full((B, H, W), math.inf, dtype=dt, device=dev)
        gcol = torch.zeros(B, 3, H, W, dtype=dt, device=dev)
    obj_front = z_min < z_g
    rgb = torch.where(obj_front.unsqueeze(1), alpha.unsqueeze(1) * col_obj + (1 - alpha.unsqueeze(1)) * gcol, gcol)
    z_near = torch.where(obj_front, z_min, z_g)
    depth = torch.where(torch.isfinite(z_near), 1.0 / (1.0 + z_near), torch.zeros_like(z_near))
    return torch.cat([rgb, depth.unsqueeze(1)], 1).clamp(0.0, 1.0)


def to_uint8(frames: Tensor) -> Tensor:
    """Quantize ``[0, 1]`` frames for compact storage."""
    return torch.round(frames.clamp(0, 1) * 255).to(torch.uint8)


def from_uint8(frames: Tensor) -> Tensor:
    return frames.to(torch.float32) / 255.0

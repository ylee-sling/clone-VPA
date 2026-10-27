# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Task T2: collaborative dual-arm tabletop assembly (Sections 7.1-7.2, Table 6), ManiSkill3-style interface.

Two Franka Panda arms stand on opposite sides of a shared table (bases at
``y = -+0.55 m`` facing each other). Each arm picks its part (red for
``panda-0``, blue for ``panda-1``) from a shared bin region and inserts it into
a shared fixture, in the order specified by the instruction:

* ``"red_first"``  - "insert the red part first, then the blue part";
* ``"blue_first"`` - "insert the blue part first, then the red part";
* ``"any"``        - "assemble both parts in the fixture in any order".

The first part goes to the bottom slot of the fixture, the second on top of it.
**Success** is the assembly predicate within 40 s (4000 control steps at 100 Hz):
both parts placed, in the instructed order. A **collision** is any arm-arm contact:
two physical collision spheres of the end effectors, or two points of the
proximal links, closer than the sum of their radii. Occlusion levels
(``none``, ``moderate``, ``severe``) place 0, 2 or 5 random occluders between the
third-person cameras and the workspace; they only affect the rendered views.

Interface (mirrors ManiSkill3 multi-agent tasks such as ``TwoRobotStackCube-v1``)
-------------------------------------------------------------------------------
``env = DualArmAssemblyEnv(AssemblyConfig(num_envs=..., obs_mode="state_dict" | "state" | "rgbd"))``,
``obs, info = env.reset(seed=...)``,
``obs, reward, terminated, truncated, info = env.step(action)`` with batched torch
tensors over ``num_envs``. ``env.action_space`` is a ``Dict`` keyed by the agent uids
``("panda-0", "panda-1")`` of ``Box(-1, 1, (7,))`` actions: the normalized end-effector
twist ``(v, w)`` in the arm's **base frame** (scaled by the per-axis limits of
:class:`~cma_vpa.safety.arm_barrier.ArmLimits`) and a gripper command
(``> 0`` closes), i.e. ``d_a = 7`` (Section 3.3). The action is the twist setpoint for
the next control step; the simulator's low-level controller turns it into a
bounded twist rate (:class:`~cma_vpa.envs.dual_arm_sim.DualArmSim`).
:func:`twist_to_pd_ee_delta_pose` and :func:`pd_ee_delta_pose_to_twist` convert to and
from ManiSkill3's ``pd_ee_delta_pose`` controller (7-dimensional per Panda: position
delta, rotation delta, gripper, normalized by 0.1 m / 0.1 rad per control step).

Expert, modes and runner
------------------------
:class:`AssemblyExpert` is the privileged stand-in for the motion generator of
Section 4.8: a fixed-priority phase machine per arm (approach, descend, grasp,
lift, staging, wait, above fixture, insert, release, retreat, home). The leader
inserts first, and the follower waits at its staging pose until the leader has
placed its part and left the shared zone, treating the other arm as a moving
obstacle. The rule-based mode labels are the **role order**: the global mode is
the order (``0`` red first, ``1`` blue first) and the local mode of each arm is its
role (``0`` leader, ``1`` follower) (:func:`order_label`).

:class:`AssemblyRunner` runs the master (2 Hz, mode token = role in the 36-byte
downlink), the clones (30 Hz, chunks of ``H`` normalized twist knots) and the
:class:`~cma_vpa.safety.arm_barrier.ArmExecutor` (100 Hz, twist-space Stage B on
sphere-pair shares, braking fallback, one 22-byte safety packet per collision
sphere at 1 mm resolution) in a deterministic multi-rate interleaving, and
reports the readouts of Table 6.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ..comm.packet import (ARM_UNITS, DownlinkPacket, PacketError, SequenceFilter, decode_sphere_broadcast,
                           encode_sphere_broadcast)
from ..safety.arm_barrier import (PANDA_SPHERES, ArmCommand, ArmExecutor, ArmExecutorConfig, ArmLimits,
                                  SphereSet, SphereTable)
from .dual_arm_sim import DualArmSim, matrix_to_quat, matrix_to_rotvec
from .render import look_at, render_spheres

__all__ = [
    "AGENT_UIDS", "INSTRUCTIONS", "OCCLUSION_LEVELS", "T2_PRIMITIVES", "PHASES", "AssemblyConfig",
    "BoxSpace", "DictSpace", "DualArmAssemblyEnv", "AssemblyExpert", "order_label", "role_labels",
    "AssemblyModeMaster", "ExpertAssemblyPolicy", "LearnedAssemblyPolicy", "AssemblyRunner", "AssemblyResult",
    "twist_to_pd_ee_delta_pose", "pd_ee_delta_pose_to_twist", "make_maniskill_env",
]

AGENT_UIDS = ("panda-0", "panda-1")
INSTRUCTIONS = {
    "red_first": "insert the red part first, then the blue part",
    "blue_first": "insert the blue part first, then the red part",
    "any": "assemble both parts in the fixture in any order",
}
OCCLUSION_LEVELS = {"none": 0, "moderate": 2, "severe": 5}
#: Primitive dictionary of the arm task (N_u = 6), mapped from the expert phases.
T2_PRIMITIVES = ("approach", "descend", "grasp", "transport", "insert", "retreat")
PHASES = ("above_part", "descend", "grasp", "lift", "staging", "wait", "above_fixture", "insert", "release",
          "retreat", "home")
_PHASE_TO_PRIM = (0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5)
(P_ABOVE, P_DESCEND, P_GRASP, P_LIFT, P_STAGING, P_WAIT, P_ABOVE_FIX, P_INSERT, P_RELEASE, P_RETREAT,
 P_HOME) = range(11)

#: Top-down grasp orientation (tool x = world y, tool y = world x, tool z down).
_R_DOWN = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))


# ---------------------------------------------------------------------------
# minimal spaces (gymnasium's are used when it is installed)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    from gymnasium import spaces as _gym_spaces
except Exception:  # pragma: no cover
    _gym_spaces = None


class BoxSpace:
    """Minimal ``gymnasium.spaces.Box`` stand-in (``low``, ``high``, ``shape``, ``sample``, ``contains``)."""

    def __init__(self, low: float, high: float, shape: Tuple[int, ...]):
        self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), torch.float32

    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        return self.low + (self.high - self.low) * torch.rand(self.shape, generator=generator)

    def contains(self, x) -> bool:
        x = torch.as_tensor(x)
        return tuple(x.shape) == self.shape and bool(((x >= self.low) & (x <= self.high)).all())

    def __repr__(self) -> str:
        return f"Box({self.low}, {self.high}, {self.shape})"


class DictSpace(dict):
    """Minimal ``gymnasium.spaces.Dict`` stand-in (a dict of spaces with ``sample``)."""

    @property
    def spaces(self):
        return self

    def sample(self, generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
        return {k: v.sample(generator) for k, v in self.items()}


# ---------------------------------------------------------------------------
# configuration and environment
# ---------------------------------------------------------------------------
@dataclass
class AssemblyConfig:
    num_envs: int = 1
    instruction: str = "red_first"
    occlusion: str = "none"
    obs_mode: str = "state_dict"
    reward_mode: str = "dense"
    control_freq: int = 100
    max_episode_steps: int = 4000             # 40 s at 100 Hz
    image_size: Tuple[int, int] = (128, 160)
    limits: ArmLimits = field(default_factory=ArmLimits)
    spheres: SphereSet = PANDA_SPHERES
    e_track: float = 0.3                       # tracking-error disturbance at the spheres [m/s^2]
    base_y: float = 0.55
    # parts 0.30 m apart: simultaneous grasps keep about 14 cm beyond the sphere collision distances, more
    # than the standoff that the barrier holds between static sphere pairs (about D + 7 cm at e_a = 0.56 m/s^2)
    part_xy: Tuple[Tuple[float, float], ...] = ((0.30, -0.15), (0.30, 0.15))
    part_jitter: float = 0.015
    part_size: float = 0.04
    fixture_xy: Tuple[float, float] = (-0.15, 0.0)
    staging_offset: float = 0.28
    staging_z: float = 0.25
    home: Tuple[Tuple[float, float, float], ...] = ((0.10, -0.30, 0.30), (0.10, 0.30, 0.30))
    shared_zone_radius: float = 0.20
    grasp_tol: float = 0.012
    place_tol: float = 0.015
    proximal_radius: float = 0.06
    seed: int = 0


class DualArmAssemblyEnv:
    """Batched Task-T2 environment with a ManiSkill3-style API (see the module docstring)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, cfg: AssemblyConfig = AssemblyConfig()):
        if cfg.instruction not in INSTRUCTIONS:
            raise ValueError(f"unknown instruction {cfg.instruction!r}")
        if cfg.occlusion not in OCCLUSION_LEVELS:
            raise ValueError(f"unknown occlusion level {cfg.occlusion!r}")
        if cfg.obs_mode not in ("state", "state_dict", "rgbd"):
            raise ValueError(f"unknown obs_mode {cfg.obs_mode!r}")
        self.cfg = cfg
        self.agent_uids = AGENT_UIDS
        self.num_envs = cfg.num_envs
        self.T_c = 1.0 / cfg.control_freq
        self.g = torch.Generator().manual_seed(cfg.seed)
        by = cfg.base_y
        self.sim = DualArmSim(cfg.num_envs, base_pos=((0.0, -by, 0.0), (0.0, by, 0.0)),
                              base_yaw=(math.pi / 2, -math.pi / 2), limits=cfg.limits, spheres=cfg.spheres,
                              T_c=self.T_c, e_track=cfg.e_track, generator=self.g)
        self.dtype = torch.float64
        self.R_down = torch.tensor(_R_DOWN, dtype=self.dtype)
        self.fixture = torch.tensor([*cfg.fixture_xy, 0.0], dtype=self.dtype)
        self.twist_scale = cfg.limits.twist_bound(self.dtype)
        self.single_action_space = BoxSpace(-1.0, 1.0, (7,))
        if _gym_spaces is not None:  # pragma: no cover - optional dependency
            import numpy as np
            box = _gym_spaces.Box(-1.0, 1.0, (cfg.num_envs, 7), dtype=np.float32)
            self.action_space = _gym_spaces.Dict({u: box for u in AGENT_UIDS})
        else:
            self.action_space = DictSpace({u: BoxSpace(-1.0, 1.0, (cfg.num_envs, 7)) for u in AGENT_UIDS})
        self.reset(seed=cfg.seed)

    # ------------------------------------------------------------------ frames and conversions
    @property
    def base_R(self) -> Tensor:
        return self.sim.base_R                                          # (2, 3, 3)

    def to_world_twist(self, action: Tensor) -> Tensor:
        """Normalized base-frame actions ``(E, 2, >=6)`` to world twists ``(E, 2, 6)``."""
        tw = action[..., :6].to(self.dtype).clamp(-1, 1) * self.twist_scale
        Rb = self.base_R
        return torch.cat([torch.einsum("aij,eaj->eai", Rb, tw[..., :3]),
                          torch.einsum("aij,eaj->eai", Rb, tw[..., 3:])], -1)

    def to_action(self, twist_world: Tensor, grip: Tensor) -> Tensor:
        """World twists ``(E, 2, 6)`` and gripper commands ``(E, 2)`` to normalized actions ``(E, 2, 7)``."""
        Rb = self.base_R
        v = torch.einsum("aji,eaj->eai", Rb, twist_world[..., :3].to(self.dtype))
        w = torch.einsum("aji,eaj->eai", Rb, twist_world[..., 3:].to(self.dtype))
        return torch.cat([torch.cat([v, w], -1) / self.twist_scale, grip.to(self.dtype).unsqueeze(-1)], -1)

    def _stack_action(self, action: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        if isinstance(action, dict):
            a = torch.stack([torch.as_tensor(action[u], dtype=self.dtype).reshape(self.num_envs, 7)
                             for u in AGENT_UIDS], 1)
        else:
            a = torch.as_tensor(action, dtype=self.dtype).reshape(self.num_envs, 2, 7)
        return a

    # ------------------------------------------------------------------ task geometry
    def slot_position(self, k: Tensor) -> Tensor:
        """Center of fixture slot ``k`` (0 bottom, 1 top) ``(..., 3)``."""
        k = k.to(self.dtype)
        z = self.cfg.part_size / 2 + 0.005 + self.cfg.part_size * k
        return torch.stack([torch.full_like(z, float(self.fixture[0])), torch.full_like(z, float(self.fixture[1])),
                            z], -1)

    def staging(self) -> Tensor:
        c = self.cfg
        return torch.tensor([[c.fixture_xy[0], c.fixture_xy[1] - c.staging_offset, c.staging_z],
                             [c.fixture_xy[0], c.fixture_xy[1] + c.staging_offset, c.staging_z]], dtype=self.dtype)

    def home(self) -> Tensor:
        return torch.tensor(self.cfg.home, dtype=self.dtype)

    @property
    def order(self) -> Tensor:
        """Instructed order per env: ``0`` red first, ``1`` blue first, ``-1`` any."""
        return self._order

    # ------------------------------------------------------------------ reset / step
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        c = self.cfg
        if seed is not None:
            self.g.manual_seed(seed)
        E = self.num_envs
        jit = c.part_jitter * (2 * torch.rand(E, 2, 2, generator=self.g, dtype=self.dtype) - 1)
        xy = torch.tensor(c.part_xy, dtype=self.dtype).unsqueeze(0) + jit
        self.part_pos = torch.cat([xy, torch.full((E, 2, 1), c.part_size / 2, dtype=self.dtype)], -1)
        self.part_attached = torch.zeros(E, 2, dtype=torch.bool)
        self.part_offset = torch.zeros(E, 2, 3, dtype=self.dtype)
        self.part_placed = torch.zeros(E, 2, dtype=torch.bool)
        self.part_slot = torch.full((E, 2), -1, dtype=torch.long)
        self.n_placed = torch.zeros(E, dtype=torch.long)
        self.first_placed = torch.full((E,), -1, dtype=torch.long)
        self.elapsed_steps = torch.zeros(E, dtype=torch.long)
        self.collided = torch.zeros(E, dtype=torch.bool)
        self.success_time = torch.full((E,), math.nan, dtype=self.dtype)
        self.min_clearance = torch.full((E,), math.inf, dtype=self.dtype)
        self.min_proximal_clearance = torch.full((E,), math.inf, dtype=self.dtype)
        opts = options or {}
        instr = opts.get("instruction", c.instruction)
        per_env = opts.get("instructions", [instr] * E)          # optional per-env instructions
        if len(per_env) != E or any(i not in INSTRUCTIONS for i in per_env):
            raise ValueError("options['instructions'] must name one known instruction per env")
        self.instruction = instr
        self.instructions = list(per_env)
        code = {"red_first": 0, "blue_first": 1, "any": -1}
        self._order = torch.tensor([code[i] for i in per_env], dtype=torch.long)
        self.sim.reset(self.home().expand(E, 2, 3).clone(), self.R_down.expand(E, 2, 3, 3).clone())
        # occluders on the segments between the third-person cameras and the workspace
        n_occ = OCCLUSION_LEVELS[opts.get("occlusion", c.occlusion)]
        cams = self._third_person_cameras()                              # (2, 3)
        centre = torch.tensor([0.08, 0.0, 0.10], dtype=self.dtype)
        occ = []
        for k in range(n_occ):
            cam = cams[k % 2]
            f = 0.35 + 0.35 * torch.rand(E, 1, generator=self.g, dtype=self.dtype)
            pt = cam + f * (centre - cam) + 0.05 * torch.randn(E, 3, generator=self.g, dtype=self.dtype)
            occ.append(pt)
        self.occluders = torch.stack(occ, 1) if occ else torch.zeros(E, 0, 3, dtype=self.dtype)
        self.occluder_radius = 0.05 + 0.03 * torch.rand(E, self.occluders.shape[1], generator=self.g,
                                                        dtype=self.dtype)
        self._update_metrics()
        return self.get_obs(), self._info()

    def step(self, action: Union[Tensor, Dict[str, Tensor]]):
        """Apply normalized twist / gripper actions (Dict keyed by agent uid, or ``(E, 2, 7)``)."""
        c = self.cfg
        a = self._stack_action(action)
        active = ~self._done()
        twist_w = self.to_world_twist(a)
        twist_w = torch.where(active.view(-1, 1, 1), twist_w, torch.zeros_like(twist_w))
        grip_cmd = a[..., 6]
        holding = self.part_attached | self._closing_on_part(grip_cmd)
        grip_min = torch.where(holding, torch.full_like(self.sim.grip, c.part_size / 0.08), torch.zeros_like(self.sim.grip))
        self.sim.step(twist_w, grip_cmd, grip_min)
        self._update_parts(grip_cmd)
        self.elapsed_steps = self.elapsed_steps + active.long()
        self._update_metrics()
        info = self._info()
        newly = info["success"] & torch.isnan(self.success_time)
        self.success_time = torch.where(newly, self.elapsed_steps.to(self.dtype) * self.T_c, self.success_time)
        info["assembly_time"] = self.success_time.clone()
        reward = self._reward(info)
        terminated = info["success"] | info["fail"]
        truncated = self.elapsed_steps >= c.max_episode_steps
        return self.get_obs(), reward, terminated, truncated, info

    def _done(self) -> Tensor:
        return self.evaluate()["success"] | (self.elapsed_steps >= self.cfg.max_episode_steps)

    def _closing_on_part(self, grip_cmd: Tensor) -> Tensor:
        d = self.sim.p - self.part_pos
        near = (torch.linalg.vector_norm(d[..., :2], dim=-1) < self.cfg.grasp_tol) & (d[..., 2].abs() < self.cfg.grasp_tol)
        return near & (grip_cmd > 0) & ~self.part_placed

    def _update_parts(self, grip_cmd: Tensor) -> None:
        c = self.cfg
        width = c.part_size / 0.08
        attach = self._closing_on_part(grip_cmd) & ~self.part_attached & (self.sim.grip <= width + 1e-6)
        if bool(attach.any()):
            local = torch.einsum("eaji,eaj->eai", self.sim.R, self.part_pos - self.sim.p)
            self.part_offset = torch.where(attach.unsqueeze(-1), local, self.part_offset)
            self.part_attached = self.part_attached | attach
        held = self.part_attached
        world = self.sim.p + torch.einsum("eaij,eaj->eai", self.sim.R, self.part_offset)
        self.part_pos = torch.where(held.unsqueeze(-1), world, self.part_pos)
        release = held & (grip_cmd <= 0) & (self.sim.grip > width + 0.1)
        if bool(release.any()):
            slot = self.slot_position(self.n_placed).unsqueeze(1).expand(-1, 2, -1)
            dxy = torch.linalg.vector_norm(self.part_pos[..., :2] - slot[..., :2], dim=-1)
            dz = (self.part_pos[..., 2] - slot[..., 2]).abs()
            ok = release & (dxy < c.place_tol) & (dz < c.place_tol)
            # place at most one part per env per step (the second waits for the next step)
            first = ok & (torch.cumsum(ok.long(), 1) == 1)
            self.part_pos = torch.where(first.unsqueeze(-1), slot, self.part_pos)
            self.part_slot = torch.where(first, self.n_placed.unsqueeze(1).expand(-1, 2), self.part_slot)
            self.first_placed = torch.where((self.n_placed == 0) & first.any(1), first.long().argmax(1),
                                            self.first_placed)
            self.n_placed = self.n_placed + first.long().sum(1)
            self.part_placed = self.part_placed | first
            dropped = release & ~first
            drop_pos = self.part_pos.clone()
            drop_pos[..., 2] = c.part_size / 2
            self.part_pos = torch.where(dropped.unsqueeze(-1), drop_pos, self.part_pos)
            self.part_attached = self.part_attached & ~release

    def _update_metrics(self) -> None:
        st = self.sim.sphere_state()
        r = self.cfg.spheres.radius_tensor(self.dtype)
        d = torch.cdist(st.center[:, 0], st.center[:, 1]) - (r.view(-1, 1) + r.view(1, -1))
        clear = d.flatten(1).amin(1)
        pp = self.sim.proximal_points()                                        # (E, 2, P, 3)
        rp = self.cfg.proximal_radius
        pts0 = torch.cat([pp[:, 0], st.center[:, 0]], 1)
        pts1 = torch.cat([pp[:, 1], st.center[:, 1]], 1)
        r0 = torch.cat([torch.full((pp.shape[2],), rp, dtype=self.dtype), r])
        prox = (torch.cdist(pts0, pts1) - (r0.view(-1, 1) + r0.view(1, -1)))
        P = pp.shape[2]
        prox[:, P:, P:] = math.inf                                             # EE-EE pairs counted above
        pclear = prox.flatten(1).amin(1)
        self.min_clearance = torch.minimum(self.min_clearance, clear)
        self.min_proximal_clearance = torch.minimum(self.min_proximal_clearance, pclear)
        self.collided = self.collided | (clear < 0) | (pclear < 0)
        self.clearance = clear

    def evaluate(self) -> Dict[str, Tensor]:
        """Assembly predicate and collision flags (ManiSkill3 ``evaluate``)."""
        both = self.n_placed >= 2
        order_ok = (self._order < 0) | (self.first_placed == self._order)
        success = both & order_ok
        fail = both & ~order_ok
        return {"success": success, "fail": fail, "collision": self.collided.clone(), "n_placed": self.n_placed.clone(),
                "first_placed": self.first_placed.clone()}

    def _info(self) -> Dict[str, Tensor]:
        info = self.evaluate()
        info.update(elapsed_steps=self.elapsed_steps.clone(), min_clearance=self.min_clearance.clone(),
                    min_proximal_clearance=self.min_proximal_clearance.clone(), ik_residual=self.sim.ik_err.clone(),
                    part_attached=self.part_attached.clone(), part_placed=self.part_placed.clone())
        return info

    def _reward(self, info: Dict[str, Tensor]) -> Tensor:
        if self.cfg.reward_mode == "sparse":
            return info["success"].to(self.dtype)
        target = torch.where(self.part_attached.unsqueeze(-1),
                             self.slot_position(self.n_placed).unsqueeze(1).expand(-1, 2, -1), self.sim.p)
        dist = torch.where(self.part_placed, torch.zeros_like(self.n_placed, dtype=self.dtype).unsqueeze(1).expand(-1, 2),
                           torch.linalg.vector_norm(self.part_pos - target, dim=-1))
        return (-dist.sum(1) + 2.0 * self.n_placed.to(self.dtype) + 10.0 * info["success"].to(self.dtype)
                - 5.0 * info["collision"].to(self.dtype))

    # ------------------------------------------------------------------ observations
    def tcp_twist_base(self) -> Tensor:
        Rb = self.base_R
        return torch.cat([torch.einsum("aji,eaj->eai", Rb, self.sim.v), torch.einsum("aji,eaj->eai", Rb, self.sim.w)], -1)

    def tcp_pos_base(self, points: Optional[Tensor] = None) -> Tensor:
        """World points ``(E, 2, 3)`` (default: the TCPs) in each arm's base frame."""
        pts = self.sim.p if points is None else points
        Rb = self.base_R
        return torch.einsum("aji,eaj->eai", Rb, pts - self.sim.kin.base_p)

    def current_goal(self) -> Tensor:
        """Task-level goal of each arm: its part until grasped, then the next fixture slot, then home."""
        slot = self.slot_position(self.n_placed).unsqueeze(1).expand(-1, 2, -1)
        g = torch.where(self.part_attached.unsqueeze(-1), slot, self.part_pos)
        return torch.where(self.part_placed.unsqueeze(-1), self.home().expand_as(g), g)

    def rel_goal_base(self) -> Tensor:
        Rb = self.base_R
        return torch.einsum("aji,eaj->eai", Rb, self.current_goal() - self.sim.p)

    def proprio(self, sigma_bar: Optional[Tensor] = None, braking: Optional[Tensor] = None) -> Tensor:
        """``S_t in R^13`` per arm: joints (7), gripper opening (1), TCP relative to the fixture in the base
        frame (3), sigma_bar (1), braking flag (1). Shape ``(E, 2, 13)``."""
        E = self.num_envs
        sb = torch.zeros(E, 2, dtype=self.dtype) if sigma_bar is None else sigma_bar.to(self.dtype).view(E, 2)
        br = torch.zeros(E, 2, dtype=self.dtype) if braking is None else braking.to(self.dtype).view(E, 2)
        Rb = self.base_R
        rel = torch.einsum("aji,eaj->eai", Rb, self.sim.p - self.fixture)
        return torch.cat([self.sim.q, self.sim.grip.unsqueeze(-1), rel, sb.unsqueeze(-1), br.unsqueeze(-1)], -1).float()

    def _third_person_cameras(self) -> Tensor:
        by = self.cfg.base_y
        return torch.tensor([[-0.35, -by - 0.45, 0.85], [-0.35, by + 0.45, 0.85]], dtype=self.dtype)

    def render_agents(self, image_size: Optional[Tuple[int, int]] = None) -> Tensor:
        """RGB-D views ``(E, 2, 8, H, W)``: per arm the wrist camera (channels 0-3) and its third-person camera (4-7)."""
        hw = image_size or self.cfg.image_size
        E = self.num_envs
        st = self.sim.sphere_state()
        pp = self.sim.proximal_points()
        r_ee = self.cfg.spheres.radius_tensor(self.dtype)
        cols, cents, rads = [], [], []
        for a in range(2):
            cents += [st.center[:, a], pp[:, a]]
            rads += [r_ee.expand(E, -1), torch.full(pp.shape[:1] + pp.shape[2:3], self.cfg.proximal_radius, dtype=self.dtype)]
            cols += [torch.tensor([0.75, 0.75, 0.8], dtype=self.dtype).expand(E, st.center.shape[2], 3),
                     torch.tensor([0.9, 0.9, 0.9], dtype=self.dtype).expand(E, pp.shape[2], 3)]
        part_col = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.2, 0.9]], dtype=self.dtype).expand(E, 2, 3)
        fix = torch.cat([self.fixture.view(1, 1, 3).expand(E, 1, 3).clone()], 1)
        fix[..., 2] = 0.005
        cents += [self.part_pos, fix, self.occluders]
        rads += [torch.full((E, 2), self.cfg.part_size * 0.55, dtype=self.dtype),
                 torch.full((E, 1), 0.035, dtype=self.dtype), self.occluder_radius]
        cols += [part_col, torch.tensor([0.1, 0.7, 0.2], dtype=self.dtype).expand(E, 1, 3),
                 torch.tensor([0.15, 0.15, 0.15], dtype=self.dtype).expand(E, self.occluders.shape[1], 3)]
        C, Rr, K = torch.cat(cents, 1), torch.cat(rads, 1), torch.cat(cols, 1)
        J = C.shape[1]
        out = []
        cams3 = self._third_person_cameras()
        centre = torch.tensor([0.08, 0.0, 0.10], dtype=self.dtype)
        for a in range(2):
            R = self.sim.R[:, a]
            wrist_pos = self.sim.p[:, a] + torch.einsum("eij,j->ei", R, torch.tensor([0.06, 0.0, -0.10], dtype=self.dtype))
            wrist_R = torch.stack([R[..., :, 1], -R[..., :, 0], R[..., :, 2]], -1)
            # hide the arm's own end-effector spheres in its wrist view
            own_mask = torch.ones(E, J, dtype=torch.bool)
            start = a * (st.center.shape[2] + pp.shape[2])
            own_mask[:, start:start + st.center.shape[2]] = False
            w_img = render_spheres(wrist_pos, wrist_R, C, Rr, K, mask=own_mask, image_size=hw, fov_deg=80.0)
            tp = cams3[a].expand(E, 3)
            t_img = render_spheres(tp, look_at(tp, centre.expand(E, 3)), C, Rr, K, image_size=hw, fov_deg=60.0)
            out.append(torch.cat([w_img, t_img], 1))
        return torch.stack(out, 1)

    def goal_frames(self, image_size: Optional[Tuple[int, int]] = None) -> Tensor:
        """Milestone views ``(E, 2, 8, H, W)``: the current arm configuration with both parts in the fixture.

        At reset the arms are at their home poses, so this is the view of the
        completed assembly (instructed order; red at the bottom for ``"any"``).
        """
        saved = self.part_pos.clone()
        first = torch.where(self._order < 0, torch.zeros_like(self._order), self._order)
        slot_of = torch.stack([(first != 0).long(), (first != 1).long()], 1)             # bottom slot for the first part
        self.part_pos = self.slot_position(slot_of)
        try:
            return self.render_agents(image_size)
        finally:
            self.part_pos = saved

    def render(self) -> Tensor:
        """Third-person RGB of ``panda-0``'s camera, ``(E, H, W, 3)`` uint8 (ManiSkill3 ``render``)."""
        img = self.render_agents()[:, 0, 4:7]
        return (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1)

    def get_obs(self) -> Union[Tensor, Dict]:
        E = self.num_envs
        quat = matrix_to_quat(self.sim.R)
        tw = self.tcp_twist_base()
        obs = {"agent": {}, "extra": {}}
        for a, uid in enumerate(AGENT_UIDS):
            obs["agent"][uid] = {"qpos": self.sim.q[:, a].float(), "tcp_pos": self.tcp_pos_base()[:, a].float(),
                                 "tcp_quat": quat[:, a].float(), "tcp_twist": tw[:, a].float(),
                                 "gripper": self.sim.grip[:, a:a + 1].float()}
        obs["extra"] = {"part_pos": self.part_pos.float(), "part_attached": self.part_attached.float(),
                        "part_placed": self.part_placed.float(), "fixture_pos": self.fixture.expand(E, 3).float(),
                        "n_placed": self.n_placed.float().unsqueeze(-1),
                        "instruction": torch.nn.functional.one_hot(self._order + 1, 3).float()}
        if self.cfg.obs_mode == "rgbd":
            views = self.render_agents()
            obs["sensor_data"] = {uid: {"wrist_camera": views[:, a, :4].float(), "third_person": views[:, a, 4:].float(),
                                        "rgbd": views[:, a].float()} for a, uid in enumerate(AGENT_UIDS)}
        if self.cfg.obs_mode == "state":
            flat = [obs["agent"][u][k].reshape(E, -1) for u in AGENT_UIDS for k in obs["agent"][u]]
            flat += [v.reshape(E, -1) for v in obs["extra"].values()]
            return torch.cat(flat, -1)
        return obs


# ---------------------------------------------------------------------------
# expert and labelling rules
# ---------------------------------------------------------------------------
def order_label(first_placed: Tensor) -> Tensor:
    """Global mode label from the trajectory (fixed rule): ``0`` if the red part was placed first, else ``1``."""
    return (first_placed == 1).long()


def role_labels(u_g: Tensor) -> Tensor:
    """Local mode labels ``(E, 2)``: ``0`` for the leader (the arm whose part goes first), ``1`` for the follower."""
    leader = u_g.long()
    arm = torch.arange(2).view(1, 2)
    return (arm != leader.view(-1, 1)).long()


class AssemblyExpert:
    """Privileged fixed-priority expert (see the module docstring); vectorized over envs and arms.

    ``roles (E, 2)``: ``0`` leader, ``1`` follower. Two leaders, or two followers,
    are allowed (they model incompatible independent choices).
    """

    def __init__(self, env: DualArmAssemblyEnv, roles: Tensor, k_p: float = 3.0, k_r: float = 3.0,
                 v_fast: float = 0.35, v_fine: float = 0.08, w_cap: float = 0.6):
        self.env = env
        self.roles = roles.clone().long()
        self.k_p, self.k_r, self.v_fast, self.v_fine, self.w_cap = k_p, k_r, v_fast, v_fine, w_cap
        self.phase = torch.zeros(env.num_envs, 2, dtype=torch.long)
        self.lift_xy = env.part_pos[..., :2].clone()

    def set_roles(self, roles: Tensor, mask: Optional[Tensor] = None) -> None:
        self.roles = roles.long() if mask is None else torch.where(mask, roles.long(), self.roles)

    def primitive(self) -> Tensor:
        return torch.tensor(_PHASE_TO_PRIM, dtype=torch.long)[self.phase]

    def _target(self, p: Tensor, phase: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Target positions ``(E, 2, 3)``, gripper commands ``(E, 2)``, speed caps ``(E, 2)``."""
        env = self.env
        part = env.part_pos
        stage = env.staging().unsqueeze(0).expand_as(p)
        home = env.home().unsqueeze(0).expand_as(p)
        slot = env.slot_position(env.n_placed).unsqueeze(1).expand_as(p)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=p.dtype)
        lift = torch.cat([self.lift_xy, torch.full_like(self.lift_xy[..., :1], 0.17)], -1)
        retreat = torch.cat([p[..., :2], torch.full_like(p[..., :1], 0.30)], -1)
        tgt = torch.stack([part + 0.10 * up, part, part, lift, stage, stage, slot + 0.10 * up, slot, slot, retreat,
                           home], 2)                                                  # (E, 2, 11, 3)
        t = torch.gather(tgt, 2, phase.view(*phase.shape, 1, 1).expand(*phase.shape, 1, 3)).squeeze(2)
        close = (phase >= P_GRASP) & (phase <= P_INSERT)
        grip = torch.where(close, torch.ones_like(t[..., 0]), -torch.ones_like(t[..., 0]))
        fine = (phase == P_DESCEND) | (phase == P_INSERT)
        cap = torch.where(fine, torch.full_like(grip, self.v_fine), torch.full_like(grip, self.v_fast))
        return t, grip, cap

    def update(self) -> None:
        """Phase transitions from the current (privileged) state."""
        env = self.env
        p = env.sim.p
        t, _, _ = self._target(p, self.phase)
        err = torch.linalg.vector_norm(t - p, dim=-1)
        speed = torch.linalg.vector_norm(env.sim.v, dim=-1)
        ph = self.phase
        nxt = ph.clone()
        reached = err < 0.008
        settled = (err < 0.003) & (speed < 0.03)
        nxt = torch.where((ph == P_ABOVE) & reached, P_DESCEND, nxt)
        nxt = torch.where((ph == P_DESCEND) & settled, P_GRASP, nxt)
        nxt = torch.where((ph == P_GRASP) & env.part_attached, P_LIFT, nxt)
        nxt = torch.where((ph == P_LIFT) & (err < 0.015), P_STAGING, nxt)
        nxt = torch.where((ph == P_STAGING) & (err < 0.015), P_WAIT, nxt)
        other = self.phase.flip(1)
        other_p = p.flip(1)
        zone = env.cfg.shared_zone_radius
        other_clear = torch.linalg.vector_norm(other_p[..., :2] - env.fixture[:2], dim=-1) > zone
        follower_ok = (env.n_placed.unsqueeze(1) >= 1) & (other >= P_RETREAT) & other_clear
        allowed = (self.roles == 0) | follower_ok
        nxt = torch.where((ph == P_WAIT) & allowed, P_ABOVE_FIX, nxt)
        nxt = torch.where((ph == P_ABOVE_FIX) & reached, P_INSERT, nxt)
        nxt = torch.where((ph == P_INSERT) & settled, P_RELEASE, nxt)
        nxt = torch.where((ph == P_RELEASE) & env.part_placed, P_RETREAT, nxt)
        nxt = torch.where((ph == P_RETREAT) & (p[..., 2] > 0.28), P_HOME, nxt)
        # a part that was dropped is picked up again
        dropped = (ph >= P_LIFT) & (ph <= P_RELEASE) & ~env.part_attached & ~env.part_placed
        nxt = torch.where(dropped, P_ABOVE, nxt)
        grasped_now = (nxt == P_LIFT) & (ph == P_GRASP)
        self.lift_xy = torch.where(grasped_now.unsqueeze(-1), env.part_pos[..., :2], self.lift_xy)
        self.phase = nxt

    def _twist(self, p: Tensor, R: Tensor, phase: Tensor) -> Tuple[Tensor, Tensor]:
        t, grip, cap = self._target(p, phase)
        v = self.k_p * (t - p)
        v = v * torch.clamp(cap.unsqueeze(-1) / v.abs().amax(-1, keepdim=True).clamp(min=1e-12), max=1.0)
        hold = phase == P_GRASP
        hold = hold | (phase == P_RELEASE)
        v = torch.where(hold.unsqueeze(-1), self.k_p * (t - p) * 0.5, v)
        e_R = matrix_to_rotvec(self.env.R_down @ R.transpose(-1, -2))
        w = self.k_r * e_R
        w = w * torch.clamp(self.w_cap / w.abs().amax(-1, keepdim=True).clamp(min=1e-12), max=1.0)
        return torch.cat([v, w], -1), grip

    def act(self) -> Tensor:
        """Normalized actions ``(E, 2, 7)`` for the current tick."""
        tw, grip = self._twist(self.env.sim.p, self.env.sim.R, self.phase)
        return self.env.to_action(tw, grip)

    def plan(self, H: int) -> Tensor:
        """Normalized chunks ``(E, 2, H, 7)``: kinematic rollout of the phase controller (phase held fixed).

        Targets are fixed over the chunk; the orientation error is propagated with
        the small-angle update ``e <- e - w dt`` (the error is at most a few degrees
        once the arm has settled into its grasp orientation).
        """
        env = self.env
        p = env.sim.p.clone()
        t, grip, cap = self._target(p, self.phase)
        eR = matrix_to_rotvec(env.R_down @ env.sim.R.transpose(-1, -2))
        hold = ((self.phase == P_GRASP) | (self.phase == P_RELEASE)).unsqueeze(-1)
        dt = env.T_c
        V, Wt = [], []
        for _ in range(H):
            v = self.k_p * (t - p)
            v_cap = v * torch.clamp(cap.unsqueeze(-1) / v.abs().amax(-1, keepdim=True).clamp(min=1e-12), max=1.0)
            v = torch.where(hold, 0.5 * v, v_cap)
            w = self.k_r * eR
            w = w * torch.clamp(self.w_cap / w.abs().amax(-1, keepdim=True).clamp(min=1e-12), max=1.0)
            V.append(v)
            Wt.append(w)
            p = p + dt * v
            eR = eR - dt * w
        V, Wt = torch.stack(V, 2), torch.stack(Wt, 2)                      # (E, 2, H, 3) world frame
        Rb = env.base_R
        vb = torch.einsum("aji,eahj->eahi", Rb, V)
        wb = torch.einsum("aji,eahj->eahi", Rb, Wt)
        tw = torch.cat([vb, wb], -1) / env.twist_scale
        return torch.cat([tw, grip.unsqueeze(-1).unsqueeze(-1).expand(*tw.shape[:3], 1)], -1)


# ---------------------------------------------------------------------------
# master, chunk source and runner
# ---------------------------------------------------------------------------
class AssemblyModeMaster:
    """Broadcasts the role order as mode tokens (36-byte downlinks, mode = role of each arm).

    The global mode is the instructed order; with ``"any"`` it is sampled once per
    mission (Proposition 3: one shared draw instead of two independent ones).
    """

    def __init__(self, env: DualArmAssemblyEnv, generator: Optional[torch.Generator] = None):
        self.env = env
        o = env.order.clone()
        sampled = torch.randint(0, 2, o.shape, generator=generator)
        self.u_g = torch.where(o < 0, sampled, o)
        self.seq = 0

    def cycle(self, now: float) -> List[List[bytes]]:
        """Downlinks per env: ``[[panda-0 msg, panda-1 msg], ...]``."""
        self.seq = (self.seq + 1) % 65536
        roles = role_labels(self.u_g)
        return [[DownlinkPacket.build(a, self.seq, now, int(roles[e, a]), 1, [0] * 24, 0.0, 0.0).pack()
                 for a in range(2)] for e in range(self.env.num_envs)]


class ExpertAssemblyPolicy:
    """Expert chunks; the roles come from the downlink mode tokens, are sampled per arm, or are fixed.

    ``mode_source``: ``"master"`` (until a downlink arrives both arms wait as
    followers), ``"independent"`` (each arm draws its own role once), ``"fixed"``
    (roles from ``roles``).
    """

    def __init__(self, env: DualArmAssemblyEnv, H: int = 32, mode_source: str = "master",
                 roles: Optional[Tensor] = None, generator: Optional[torch.Generator] = None):
        E = env.num_envs
        if mode_source == "independent":
            r = torch.randint(0, 2, (E, 2), generator=generator)
        elif mode_source == "fixed":
            if roles is None:
                raise ValueError("fixed roles required")
            r = roles
        else:
            r = torch.ones(E, 2, dtype=torch.long)
        self.env, self.H, self.mode_source = env, H, mode_source
        self.expert = AssemblyExpert(env, r)
        self.filters = [SequenceFilter(16) for _ in range(E)]
        self.n_decisions = 0

    def on_downlink(self, msgs: List[List[bytes]], now: float) -> None:
        if self.mode_source != "master":
            return
        roles = self.expert.roles.clone()
        for e, per_env in enumerate(msgs):
            for raw in per_env:
                try:
                    p = DownlinkPacket.unpack(raw)
                except PacketError:
                    continue
                if 0 <= p.agent_id < 2 and self.filters[e].accept(p.agent_id, p.seq):
                    roles[e, p.agent_id] = p.mode
        self.expert.set_roles(roles)

    def decide(self, now: float) -> Dict[str, Tensor]:
        self.expert.update()
        self.n_decisions += 1
        return {"A": self.expert.plan(self.H), "prim": self.expert.primitive(),
                "H_val": torch.full((self.env.num_envs, 2), self.H, dtype=torch.long)}


class LearnedAssemblyPolicy:
    """Neural clones (:class:`~cma_vpa.models.ClonePolicy` with :meth:`~cma_vpa.models.CloneConfig.arm`) for T2.

    Every arm is one clone of the batch (``E * 2`` clones with Siamese weights).
    The clone sees its wrist and third-person RGB-D views stacked into 8
    channels, ``S_t`` from :meth:`DualArmAssemblyEnv.proprio` and the base-frame
    offset to its current goal; its chunks are normalized base-frame twists and
    gripper commands (``d_a = 7``). Downlinks of env ``e`` are routed to clones
    ``2 e`` and ``2 e + 1``.
    """

    def __init__(self, env: DualArmAssemblyEnv, clone, generator: Optional[torch.Generator] = None):
        from ..models import HashedTextEmbedder

        cfg = clone.cfg
        if cfg.action_dim != 7:
            raise ValueError("the T2 clone needs action_dim = 7 (CloneConfig.arm())")
        self.env, self.clone, self.g, self.H = env, clone, generator, cfg.horizon
        E = env.num_envs
        emb = HashedTextEmbedder(cfg.text_dim)([INSTRUCTIONS[i] for i in env.instructions for _ in range(2)])
        text = emb.to(clone.mu_z.device)
        goals = env.goal_frames(cfg.image_size).float()
        clone.reset_mission(text, milestone_frames=goals.reshape(E * 2, 1, *goals.shape[2:]).to(clone.mu_z.device),
                            agent_ids=list(range(E * 2)))
        self.n_decisions = 0
        self.braking: Optional[Tensor] = None

    def on_downlink(self, msgs: List[List[bytes]], now: float) -> None:
        flat = []
        for e, per_env in enumerate(msgs):
            for raw in per_env:
                try:
                    p = DownlinkPacket.unpack(raw)
                except PacketError:
                    continue
                flat.append(DownlinkPacket(agent_id=2 * e + p.agent_id, seq=p.seq, timestamp=p.timestamp, mode=p.mode,
                                           milestone=p.milestone, code=p.code, sigma_code=p.sigma_code,
                                           validity_code=p.validity_code, status=p.status).pack())
        self.clone.on_downlink(flat, now)

    def decide(self, now: float) -> Dict[str, Tensor]:
        env, clone = self.env, self.clone
        E = env.num_envs
        frames = env.render_agents(clone.cfg.image_size).float().reshape(E * 2, -1, *clone.cfg.image_size)
        sb = clone.filter.sigma_bar if clone.filter.sigma_bar is not None else torch.zeros(E * 2)
        S = env.proprio(sb.view(E, 2).double(), self.braking).reshape(E * 2, -1)
        out = clone.decide(frames.to(clone.mu_z.device), S.to(clone.mu_z.device),
                           env.rel_goal_base().reshape(E * 2, 3).float().to(clone.mu_z.device), now, generator=self.g)
        self.n_decisions += 1
        A = out.chunk.double().cpu().view(E, 2, self.H, 7)
        return {"A": A, "prim": out.primitive.cpu().view(E, 2), "H_val": out.H_val.cpu().view(E, 2)}


@dataclass
class AssemblyResult:
    success: Tensor
    assembly_time: Tensor
    collision: Tensor
    min_clearance: Tensor
    min_proximal_clearance: Tensor
    n_placed: Tensor
    first_placed: Tensor
    intervention_p95: float          # sphere-level ||J_c (u* - u^)|| over active ticks [m/s^2]
    infeasible_ticks_pct: float
    fallback_events: int
    braking_ticks: int
    ticks: int
    sim_time: float
    wall_time: float
    executor_latency: Dict[str, float]
    records: List[dict] = field(default_factory=list)


class AssemblyRunner:
    """Master / clone / executor loops for Task T2 in simulated time (deterministic interleaving)."""

    def __init__(self, env: DualArmAssemblyEnv, policy, master=None, *, f_master: float = 2.0, f_dec: float = 30.0,
                 latency: float = 0.015, packet_loss: float = 0.0, stage_b: bool = True,
                 executor_cfg: Optional[ArmExecutorConfig] = None, record: bool = False,
                 generator: Optional[torch.Generator] = None):
        self.env, self.policy, self.master = env, policy, master
        self.f_master, self.f_dec, self.latency, self.packet_loss = f_master, f_dec, latency, packet_loss
        self.stage_b, self.record = stage_b, record
        self.g = generator or torch.Generator().manual_seed(env.cfg.seed + 7)
        E = env.num_envs
        ec = executor_cfg or ArmExecutorConfig(limits=env.cfg.limits, spheres=env.cfg.spheres, H=policy.H)
        self.executor = ArmExecutor(E * 2, env.cfg.spheres.n, ec, arm_ids=torch.arange(2).repeat(E))
        S = env.cfg.spheres.n
        self.S = S
        dt = torch.float64
        self.store_pos = torch.zeros(E, 2, S, 3, dtype=dt)
        self.store_vel = torch.zeros(E, 2, S, 3, dtype=dt)
        self.store_rad = torch.zeros(E, 2, S, dtype=dt)
        self.store_flags = torch.zeros(E, 2, S, dtype=torch.uint8)
        self.store_stamp = torch.full((E, 2, S), -math.inf, dtype=dt)
        self.inflight: List[List[bytes]] = [[] for _ in range(E)]
        self.pending: List[tuple] = []

    def _flat(self, x: Tensor) -> Tensor:
        return x.reshape(self.env.num_envs * 2, *x.shape[2:])

    def _send(self, now: float) -> None:
        sim = self.env.sim
        bc = self.executor.broadcast(now, self._flat(sim.p), self._flat(sim.R), self._flat(sim.v), self._flat(sim.w))
        E = self.env.num_envs
        for e in range(E):
            sub = type(bc)(arm_id=bc.arm_id[2 * e:2 * e + 2], seq=bc.seq[2 * e:2 * e + 2],
                           stamp=bc.stamp[2 * e:2 * e + 2], pos=bc.pos[2 * e:2 * e + 2], vel=bc.vel[2 * e:2 * e + 2],
                           radius=bc.radius[2 * e:2 * e + 2], flags=bc.flags[2 * e:2 * e + 2])
            pk = encode_sphere_broadcast(sub, ARM_UNITS)
            if self.packet_loss > 0:
                keep = torch.rand(len(pk), generator=self.g) >= self.packet_loss
                pk = [p for p, k in zip(pk, keep.tolist()) if k]
            self.inflight[e] = pk

    def _receive(self, now: float) -> SphereTable:
        E, S = self.env.num_envs, self.S
        idx, pos, vel, rad, fl, st = [], [], [], [], [], []
        for e in range(E):
            for arm, spheres in decode_sphere_broadcast(self.inflight[e], now, ARM_UNITS).items():
                if arm > 1:
                    continue
                for k, (p, v, r, f, t) in spheres.items():
                    if k < S:
                        idx.append((e, arm, k))
                        pos.append(p)
                        vel.append(v)
                        rad.append(r)
                        fl.append(f)
                        st.append(t)
        if idx:
            i = torch.tensor(idx, dtype=torch.long)
            e_i, a_i, k_i = i[:, 0], i[:, 1], i[:, 2]
            self.store_pos[e_i, a_i, k_i] = torch.tensor(pos, dtype=torch.float64)
            self.store_vel[e_i, a_i, k_i] = torch.tensor(vel, dtype=torch.float64)
            self.store_rad[e_i, a_i, k_i] = torch.tensor(rad, dtype=torch.float64)
            self.store_flags[e_i, a_i, k_i] = torch.tensor(fl, dtype=torch.uint8)
            self.store_stamp[e_i, a_i, k_i] = torch.tensor(st, dtype=torch.float64)
        other = self.store_pos.flip(1)                          # arm a sees arm 1 - a
        present = torch.isfinite(self.store_stamp.flip(1))
        stamp = torch.where(present, self.store_stamp.flip(1), torch.full_like(self.store_stamp, -1e9))
        return SphereTable(pos=self._flat(other), vel=self._flat(self.store_vel.flip(1)),
                           radius=self._flat(self.store_rad.flip(1)), flags=self._flat(self.store_flags.flip(1)),
                           stamp=self._flat(stamp), present=self._flat(present))

    def _publish(self, stamp: float, A: Tensor, H_val: Tensor) -> None:
        env = self.env
        E, H = env.num_envs, A.shape[2]
        knots = A.permute(0, 2, 1, 3).reshape(E * H, 2, 7)                  # (E H, 2, 7): one knot of both arms
        tw = env.to_world_twist(knots).view(E, H, 2, 6).permute(0, 2, 1, 3)  # (E, 2, H, 6) world frame
        grip = A[..., 6]
        n = E * 2
        self.executor.publish_chunk(torch.ones(n, dtype=torch.bool), tw.reshape(n, H, 6), grip.reshape(n, H).to(torch.float64),
                                    torch.full((n,), stamp, dtype=torch.float64), self._flat(H_val), stamp)

    def run(self, max_steps: Optional[int] = None) -> AssemblyResult:
        env, ex = self.env, self.executor
        Tc = env.T_c
        steps = env.cfg.max_episode_steps if max_steps is None else max_steps
        next_master = next_dec = 0.0
        interventions: List[Tensor] = []
        infeasible = active_ticks = braking_ticks = 0
        records = []
        wall0 = time.perf_counter()
        self._send(0.0)
        n = 0
        for n in range(steps):
            s = n * Tc
            nbr = self._receive(s)
            if self.master is not None and s + 1e-9 >= next_master:
                msgs = self.master.cycle(s)
                self.policy.on_downlink(msgs, s + 0.005)
                next_master += 1.0 / self.f_master
            if s + 1e-9 >= next_dec:
                out = self.policy.decide(s)
                vis = s if n == 0 else s + self.latency
                self.pending.append((vis, s, out["A"], out["H_val"]))
                if self.record:
                    records.append({"t": s, "q": env.sim.q.clone(), "p": env.sim.p.clone(), "R": env.sim.R.clone(),
                                    "v": env.sim.v.clone(), "w": env.sim.w.clone(), "grip": env.sim.grip.clone(),
                                    "part_pos": env.part_pos.clone(), "attached": env.part_attached.clone(),
                                    "placed": env.part_placed.clone(), "n_placed": env.n_placed.clone(),
                                    "A": out["A"].clone(), "prim": out["prim"].clone()})
                next_dec += 1.0 / self.f_dec
            while self.pending and self.pending[0][0] <= s + 1e-9:
                _, stamp, A, H_val = self.pending.pop(0)
                self._publish(stamp, A, H_val)
            sim = env.sim
            cmd: ArmCommand = ex.tick(s, self._flat(sim.p), self._flat(sim.R), self._flat(sim.v), self._flat(sim.w), nbr)
            if self.stage_b:
                twist = cmd.twist
            else:   # unfiltered nominal (baseline without the barrier)
                twist = torch.cat([self._flat(sim.v), self._flat(sim.w)], -1) + Tc * cmd.u_hat
            act = env.to_action(twist.view(-1, 2, 6), cmd.gripper.view(-1, 2))
            obs, rew, term, trunc, info = env.step(act)
            on = cmd.n_active > 0
            active_ticks += int(on.sum())
            infeasible += int((~cmd.stage_b_ok).sum())
            braking_ticks += int(cmd.braking.sum())
            if bool(on.any()):
                interventions.append(cmd.sphere_intervention[on])
            self._send(s + Tc)
            if bool((term | trunc).all()):
                break
        info = env.evaluate()
        iv = torch.cat(interventions) if interventions else torch.zeros(1, dtype=torch.float64)
        return AssemblyResult(success=info["success"], assembly_time=env.success_time.clone(), collision=env.collided.clone(),
                              min_clearance=env.min_clearance.clone(),
                              min_proximal_clearance=env.min_proximal_clearance.clone(), n_placed=env.n_placed.clone(),
                              first_placed=env.first_placed.clone(), intervention_p95=float(iv.quantile(0.95)),
                              infeasible_ticks_pct=100.0 * infeasible / max(1, (n + 1) * env.num_envs * 2),
                              fallback_events=len(ex.events), braking_ticks=braking_ticks, ticks=n + 1,
                              sim_time=(n + 1) * Tc, wall_time=time.perf_counter() - wall0,
                              executor_latency=ex.latency_stats(), records=records)


# ---------------------------------------------------------------------------
# ManiSkill3 compatibility
# ---------------------------------------------------------------------------
def twist_to_pd_ee_delta_pose(twist: Tensor, grip: Tensor, dt: float, pos_bound: float = 0.1,
                              rot_bound: float = 0.1) -> Tensor:
    """Convert twists ``(..., 6)`` held for ``dt`` and gripper commands to ManiSkill3 ``pd_ee_delta_pose`` actions.

    ManiSkill3's Panda ``pd_ee_delta_pose`` controller takes ``(dx, dy, dz, drx, dry, drz, gripper)``
    normalized to ``[-1, 1]`` by ``pos_bound`` [m] and ``rot_bound`` [rad] per control step
    (gripper ``-1`` close, ``+1`` open, the opposite sign convention to this repository).
    """
    a = torch.cat([twist[..., :3] * dt / pos_bound, twist[..., 3:] * dt / rot_bound], -1).clamp(-1, 1)
    return torch.cat([a, -grip.unsqueeze(-1).clamp(-1, 1)], -1)


def pd_ee_delta_pose_to_twist(action: Tensor, dt: float, pos_bound: float = 0.1,
                              rot_bound: float = 0.1) -> Tuple[Tensor, Tensor]:
    """Inverse of :func:`twist_to_pd_ee_delta_pose` (within the bounds)."""
    tw = torch.cat([action[..., :3] * pos_bound / dt, action[..., 3:6] * rot_bound / dt], -1)
    return tw, -action[..., 6]


def make_maniskill_env(env_id: str = "TwoRobotStackCube-v1", num_envs: int = 1, control_freq: int = 100,
                       **kwargs):  # pragma: no cover - requires mani_skill
    """Create the ManiSkill3 two-robot tabletop scene used by the paper (requires ``mani_skill``).

    Returns the ManiSkill3 environment with the ``pd_ee_delta_pose`` controller at
    ``control_freq``; actions can be produced from this repository's twists with
    :func:`twist_to_pd_ee_delta_pose`. This helper is not exercised by the tests of
    this repository (ManiSkill3 and its GPU simulator are not dependencies).
    """
    import importlib

    try:
        import gymnasium as gym
        importlib.import_module("mani_skill.envs")          # registers the ManiSkill3 environments
    except ImportError as exc:
        raise ImportError("make_maniskill_env needs `pip install mani_skill gymnasium`") from exc
    sim_config = kwargs.pop("sim_config", {"control_freq": control_freq, "sim_freq": 5 * control_freq})
    return gym.make(env_id, num_envs=num_envs, control_mode="pd_ee_delta_pose", obs_mode=kwargs.pop("obs_mode", "state"),
                    sim_config=sim_config, **kwargs)

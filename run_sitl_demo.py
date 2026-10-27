#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Software-in-the-loop demo of CMA-VPA on Task T1 (antipodal crossing) and Task T2 (dual-arm assembly).

Three asynchronous loops run as ``asyncio`` coroutines on a virtual clock
(discrete-event time, so the result does not depend on the speed of the host):

* **master**   at ``f_M``  (default 2 Hz):  uplinks -> MASTERCYCLE -> 36-byte downlinks
  (delivered after the link latency);
* **clone**    at ``f_dec`` (default 30 Hz): DECIDE -> chunk, visible to the executor after
  the inference latency and stamped with its decision time;
* **executor** at ``f_c``  (100 Hz):        safety broadcasts in -> TICK (Stage B, fallback,
  radius admission) -> plant step under zero-order hold -> 22-byte safety broadcasts out.

All messages pass through the bit-exact codecs of Table 1 (CRC-16 checked).
The default chunk source is the privileged expert driven by the **master's mode
token**, which demonstrates the joint-mode mechanism of Proposition 7 at team
scale; ``--mode independent`` lets every vehicle pick its passing side on its
own, and ``--policy learned --checkpoint ckpt/stage5.pt`` runs the trained
neural clones and master (``--stage-a`` adds flow-time guidance).

For Task T2 (``--task t2``) the same three loops run for two Franka Panda arms:
the master broadcasts the **role order** (leader / follower) as mode tokens, the
clones produce chunks of end-effector twists and gripper commands (``d_a = 7``),
and the arm executor applies twist-space Stage B to the sphere-pair shares,
with one 22-byte safety packet per collision sphere at 1 mm resolution.

Examples::

    python run_sitl_demo.py                       # T1: N_a = 8, wind W1, master joint mode
    python run_sitl_demo.py --mode independent    # independent sides (Prop. 7(b) at team scale)
    python run_sitl_demo.py --agents 16 --wind W2 --episodes 5
    python run_sitl_demo.py --policy learned --checkpoint checkpoints/stage5.pt --stage-a
    python run_sitl_demo.py --task t2                              # T2: red part first
    python run_sitl_demo.py --task t2 --instruction any --occlusion severe --episodes 3
    python run_sitl_demo.py --task t2 --instruction any --mode independent   # roles chosen per arm
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import itertools
import math
import statistics
import time
from typing import Callable, List, Optional

import torch

from cma_vpa.comm import DOWNLINK_SIZE, SAFETY_SIZE, UPLINK_SIZE
from cma_vpa.envs import (YAW_RATE_SCALE, AntipodalCrossing, CrossingConfig, CrossingRunner, ExpertChunkPolicy,
                          LearnedChunkPolicy, LearnedMaster, ModeMaster)
from cma_vpa.safety import Chunk, ExecutorConfig

PRIO_MASTER, PRIO_DELIVERY, PRIO_CLONE, PRIO_EXECUTOR = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# virtual clock for asyncio coroutines
# ---------------------------------------------------------------------------
class VirtualClock:
    """Discrete-event clock: coroutines ``await sleep_until(t)``; callbacks run at their time."""

    def __init__(self):
        self.now = 0.0
        self._heap: list = []
        self._seq = itertools.count()
        self.stopped = False

    def sleep_until(self, t: float, prio: int) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (t, prio, next(self._seq), fut, None))
        return fut

    def call_at(self, t: float, prio: int, fn: Callable[[], None]) -> None:
        heapq.heappush(self._heap, (t, prio, next(self._seq), None, fn))

    async def drive(self, t_end: float) -> None:
        while self._heap and not self.stopped:
            t, _, _, fut, fn = heapq.heappop(self._heap)
            if t > t_end:
                break
            self.now = t
            if fn is not None:
                fn()
            else:
                fut.set_result(None)
                await asyncio.sleep(0)        # let the woken loop run until its next await
        self.stopped = True


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------
def build(args, seed: int):
    env_cfg = dict(n_agents=args.agents, wind=args.wind, v_h=args.v_h, T_max=args.t_max, seed=seed,
                   angle_jitter_deg=args.angle_jitter)
    g = torch.Generator().manual_seed(seed + 17)
    if args.policy == "expert":
        env = AntipodalCrossing(CrossingConfig(**env_cfg))
        policy = ExpertChunkPolicy(env, mode_source="fixed" if args.mode == "fixed" else
                                   ("master" if args.mode == "master" else "independent"), generator=g)
        master = ModeMaster(args.agents, g) if args.mode == "master" else None
        H = 32
    else:
        from cma_vpa.models import HashedTextEmbedder, MasterConfig
        from cma_vpa.training.common import build_clone, build_master, load_checkpoint
        from cma_vpa.models import CloneConfig
        ck = load_checkpoint(args.checkpoint)
        ccfg = CloneConfig(**ck["config"])
        clone = build_clone(ccfg, ck["clone"]).eval()
        env = AntipodalCrossing(CrossingConfig(**env_cfg, image_size=ccfg.image_size))
        mstr = build_master(ccfg, ck["master"], MasterConfig(**ck["master_config"])).eval()
        c_text = HashedTextEmbedder(ccfg.text_dim)("cross the conflict zone and reach the antipodal point")
        clone.reset_mission(c_text, milestone_frames=env.milestone_frames(ccfg.image_size))
        policy = LearnedChunkPolicy(env, clone, stage_a=args.stage_a, generator=g)
        master = LearnedMaster(mstr, clone.rvq, clone.c_tilde, clone.tracker.active(clone.milestones), generator=g)
        if args.mode != "master":
            master = None
        H = ccfg.horizon
    runner = CrossingRunner(env, policy, master, f_master=args.f_master, f_dec=args.f_dec, latency=args.latency,
                            packet_loss=args.packet_loss, generator=g,
                            executor_cfg=ExecutorConfig(safety=env.safety, H=H, v_hold=0.25))
    return env, policy, master, runner


async def episode(args, seed: int) -> dict:
    env, policy, master, runner = build(args, seed)
    ex = runner.executor
    clock = VirtualClock()
    Tc = env.safety.T_c
    stats = {"interventions": 0, "braking_ticks": 0, "uplinks": 0, "downlinks": 0, "safety_packets": 0,
             "decisions": 0, "master_cycles": 0, "decide_ms": [], "master_ms": []}
    shared = {"nbrs": runner._receive(0.0), "sigma_bar": torch.zeros(env.cfg.n_agents, dtype=torch.float64),
              "done_at": None}
    runner._send(0.0)

    def publish(stamp: float, A: torch.Tensor, H_val: torch.Tensor) -> None:
        ch = Chunk.from_normalized(A.to(torch.float64), env.chunk_map(), YAW_RATE_SCALE,
                                   torch.full((A.shape[0],), stamp, dtype=torch.float64), H_val)
        ex.publish_chunk(torch.ones(A.shape[0], dtype=torch.bool), ch, clock.now)

    async def master_loop():
        if master is None:
            return
        k = 0
        while not clock.stopped:
            await clock.sleep_until(k / args.f_master, PRIO_MASTER)
            t0 = time.perf_counter()
            ups = policy.uplinks(clock.now, ex.flags)
            msgs = master.cycle(ups, clock.now, runner.ids, [1] * len(runner.ids))
            stats["master_ms"].append((time.perf_counter() - t0) * 1e3)
            stats["uplinks"] += len(ups)
            stats["downlinks"] += len(msgs)
            stats["master_cycles"] += 1
            t_rx = clock.now + args.link_latency
            clock.call_at(t_rx, PRIO_DELIVERY, lambda m=msgs, t=t_rx: policy.on_downlink(m, t))
            k += 1

    async def clone_loop():
        k = 0
        while not clock.stopped:
            await clock.sleep_until(k / args.f_dec, PRIO_CLONE)
            t0 = time.perf_counter()
            out = policy.decide(clock.now, shared["nbrs"])
            stats["decide_ms"].append((time.perf_counter() - t0) * 1e3)
            shared["sigma_bar"] = out["sigma_bar"].to(torch.float64)
            stamp = clock.now
            if k == 0:
                publish(stamp, out["A"], out["H_val"])        # the mission starts with a plan
            else:
                clock.call_at(stamp + args.latency, PRIO_DELIVERY,
                              lambda s=stamp, A=out["A"], h=out["H_val"]: publish(s, A, h))
            stats["decisions"] += 1
            k += 1

    async def executor_loop():
        n = 0
        while not clock.stopped:
            await clock.sleep_until(n * Tc, PRIO_EXECUTOR)
            s = n * Tc
            nbrs = runner._receive(s)
            shared["nbrs"] = nbrs
            cmd = ex.tick(s, env.sim.p, env.sim.v, nbrs, sigma_bar=shared["sigma_bar"])
            stats["interventions"] += int(((cmd.intervention > 1e-3) & ~cmd.braking).sum())
            stats["braking_ticks"] += int(cmd.braking.sum())
            env.sim.step(cmd.acceleration, cmd.position)
            env.update_metrics()
            runner._send(s + Tc)
            stats["safety_packets"] += env.cfg.n_agents
            if env.all_arrived:
                shared["done_at"] = shared["done_at"] if shared["done_at"] is not None else s
                if s - shared["done_at"] >= 0.5:
                    clock.stopped = True
            n += 1

    wall = time.perf_counter()
    tasks = [asyncio.create_task(c()) for c in (master_loop, clone_loop, executor_loop)]
    await asyncio.sleep(0)                    # every loop registers its first wake-up
    await clock.drive(args.t_max)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    causes = {}
    for e in ex.events:
        causes[e.cause] = causes.get(e.cause, 0) + 1
    return {
        "seed": seed, "success": env.all_arrived and env.collisions == 0,
        "makespan": float(env.arrival.max()) if env.all_arrived else math.nan,
        "arrived": int((~torch.isnan(env.arrival)).sum()), "d_min": env.d_min, "collisions": env.collisions,
        "fallbacks": len(ex.events), "fallback_certified": sum(e.certified for e in ex.events),
        "fallback_causes": causes, "sim_time": clock.now, "wall": time.perf_counter() - wall,
        "tick_ms": ex.latency_stats(), **stats,
    }


# ---------------------------------------------------------------------------
# Task T2: dual-arm assembly
# ---------------------------------------------------------------------------
def build_t2(args, seed: int):
    from cma_vpa.envs import (AssemblyConfig, AssemblyModeMaster, AssemblyRunner, DualArmAssemblyEnv,
                              ExpertAssemblyPolicy, LearnedAssemblyPolicy)

    g = torch.Generator().manual_seed(seed + 17)
    if args.policy == "expert":
        env = DualArmAssemblyEnv(AssemblyConfig(instruction=args.instruction, occlusion=args.occlusion, seed=seed))
        policy = ExpertAssemblyPolicy(env, mode_source="master" if args.mode == "master" else "independent",
                                      generator=g)
    else:
        from cma_vpa.models import CloneConfig
        from cma_vpa.training.common import build_clone, load_checkpoint

        ck = load_checkpoint(args.checkpoint)
        ccfg = CloneConfig(**ck["config"])
        if ccfg.action_dim != 7:
            raise SystemExit("--task t2 --policy learned needs an arm checkpoint (CloneConfig.arm(), d_a = 7)")
        env = DualArmAssemblyEnv(AssemblyConfig(instruction=args.instruction, occlusion=args.occlusion, seed=seed,
                                                image_size=ccfg.image_size))
        policy = LearnedAssemblyPolicy(env, build_clone(ccfg, ck["clone"]).eval(), generator=g)
    master = AssemblyModeMaster(env, g) if args.mode == "master" else None
    runner = AssemblyRunner(env, policy, master, f_master=args.f_master, f_dec=args.f_dec, latency=args.latency,
                            packet_loss=args.packet_loss, stage_b=not args.no_stage_b, generator=g)
    return env, policy, master, runner


async def episode_t2(args, seed: int) -> dict:
    env, policy, master, runner = build_t2(args, seed)
    ex = runner.executor
    clock = VirtualClock()
    Tc = env.T_c
    stats = {"interventions": [], "active_ticks": 0, "infeasible": 0, "braking_ticks": 0, "downlinks": 0,
             "safety_packets": 0, "decisions": 0, "master_cycles": 0, "decide_ms": []}
    runner._send(0.0)
    flat = runner._flat

    async def master_loop():
        if master is None:
            return
        k = 0
        while not clock.stopped:
            await clock.sleep_until(k / args.f_master, PRIO_MASTER)
            msgs = master.cycle(clock.now)
            stats["downlinks"] += sum(len(m) for m in msgs)
            stats["master_cycles"] += 1
            t_rx = clock.now + args.link_latency
            clock.call_at(t_rx, PRIO_DELIVERY, lambda m=msgs, t=t_rx: policy.on_downlink(m, t))
            k += 1

    async def clone_loop():
        k = 0
        while not clock.stopped:
            await clock.sleep_until(k / args.f_dec, PRIO_CLONE)
            t0 = time.perf_counter()
            out = policy.decide(clock.now)
            stats["decide_ms"].append((time.perf_counter() - t0) * 1e3)
            stamp = clock.now
            if k == 0:
                runner._publish(stamp, out["A"], out["H_val"])
            else:
                clock.call_at(stamp + args.latency, PRIO_DELIVERY,
                              lambda s=stamp, A=out["A"], h=out["H_val"]: runner._publish(s, A, h))
            stats["decisions"] += 1
            k += 1

    async def executor_loop():
        n = 0
        while not clock.stopped:
            await clock.sleep_until(n * Tc, PRIO_EXECUTOR)
            s = n * Tc
            sim = env.sim
            nbr = runner._receive(s)
            cmd = ex.tick(s, flat(sim.p), flat(sim.R), flat(sim.v), flat(sim.w), nbr)
            twist = cmd.twist if not args.no_stage_b else torch.cat([flat(sim.v), flat(sim.w)], -1) + Tc * cmd.u_hat
            _, _, term, trunc, _ = env.step(env.to_action(twist.view(-1, 2, 6), cmd.gripper.view(-1, 2)))
            on = cmd.n_active > 0
            stats["active_ticks"] += int(on.sum())
            stats["infeasible"] += int((~cmd.stage_b_ok).sum())
            stats["braking_ticks"] += int(cmd.braking.sum())
            if bool(on.any()):
                stats["interventions"].append(cmd.sphere_intervention[on])
            runner._send(s + Tc)
            stats["safety_packets"] += sum(len(p) for p in runner.inflight)
            if bool((term | trunc).all()):
                clock.stopped = True
            n += 1

    wall = time.perf_counter()
    tasks = [asyncio.create_task(c()) for c in (master_loop, clone_loop, executor_loop)]
    await asyncio.sleep(0)
    await clock.drive(env.cfg.max_episode_steps * Tc)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    ev = env.evaluate()
    iv = torch.cat(stats.pop("interventions")) if stats["active_ticks"] else torch.zeros(1, dtype=torch.float64)
    ticks = max(1, round(clock.now / Tc) + 1)
    return {"seed": seed, "success": bool(ev["success"][0]), "fail": bool(ev["fail"][0]),
            "assembly_time": float(env.success_time[0]), "collision": bool(env.collided[0]),
            "n_placed": int(env.n_placed[0]), "first_placed": int(env.first_placed[0]),
            "min_clearance": float(env.min_clearance[0]), "min_proximal": float(env.min_proximal_clearance[0]),
            "intervention_p95": float(iv.quantile(0.95)), "infeasible_pct": 100.0 * stats["infeasible"] / (2 * ticks),
            "fallbacks": len(ex.events), "sim_time": clock.now, "wall": time.perf_counter() - wall,
            "tick_ms": ex.latency_stats(), "roles": master.u_g.tolist() if master is not None else None, **stats}


def report_t2(args, results: List[dict]) -> None:
    for r in results:
        t = f"{r['assembly_time']:.2f} s" if math.isfinite(r["assembly_time"]) else \
            f"n/a ({r['n_placed']}/2 placed{', wrong order' if r['fail'] else ''})"
        print(f"\nepisode seed {r['seed']}: {'SUCCESS' if r['success'] else 'FAILURE'}")
        print(f"  assembly time             {t}")
        print(f"  collision (arm-arm)       {r['collision']}")
        print(f"  min sphere clearance      {r['min_clearance']:.3f} m (end effectors), "
              f"{r['min_proximal']:.3f} m (proximal links)")
        print(f"  Stage-B intervention p95  {r['intervention_p95']:.2f} m/s^2 over {r['active_ticks']} active arm-ticks")
        print(f"  infeasible ticks          {r['infeasible_pct']:.3f} %; fallback onsets {r['fallbacks']}; "
              f"braking {r['braking_ticks']} arm-ticks")
        print(f"  loops                     {r['master_cycles']} master cycles, {r['decisions']} clone decisions in "
              f"{r['sim_time']:.2f} s simulated ({r['wall']:.1f} s wall)")
        if r["tick_ms"]:
            print(f"  executor tick (2 arms)    p50 {r['tick_ms']['p50']:.2f} ms, p99 {r['tick_ms']['p99']:.2f} ms")
        if r["decide_ms"]:
            print(f"  clone decision (2 arms)   median {statistics.median(r['decide_ms']):.2f} ms")
        sim = max(r["sim_time"], 1e-9)
        print(f"  traffic                   downlink {r['downlinks'] * DOWNLINK_SIZE / sim:.0f} B/s, safety channel "
              f"{r['safety_packets'] * SAFETY_SIZE / sim:.0f} B/s (one packet per sphere)")
    if len(results) > 1:
        ok = [r for r in results if r["success"]]
        line = f"\nsummary over {len(results)} episodes: success {len(ok)}/{len(results)}"
        if ok:
            line += f", assembly time mean {statistics.mean(r['assembly_time'] for r in ok):.2f} s"
        line += f", collisions {sum(r['collision'] for r in results)}, fallbacks {sum(r['fallbacks'] for r in results)}"
        print(line)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _fmt_causes(c: dict) -> str:
    names = {1: "stage-B", 2: "no-chunk", 4: "stale-odom", 8: "stale-nbr", 16: "nbr-braking"}
    out = []
    for code, n in sorted(c.items()):
        tags = "+".join(v for b, v in names.items() if code & b) or str(code)
        out.append(f"{tags}:{n}")
    return ", ".join(out) or "-"


def main(argv: Optional[List[str]] = None) -> List[dict]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=["t1", "t2"], default="t1")
    ap.add_argument("--instruction", choices=["red_first", "blue_first", "any"], default="red_first",
                    help="T2 assembly order")
    ap.add_argument("--occlusion", choices=["none", "moderate", "severe"], default="none", help="T2 occluders")
    ap.add_argument("--no-stage-b", action="store_true", help="T2: execute the nominal chunks without the barrier")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--wind", choices=["W0", "W1", "W2"], default="W1")
    ap.add_argument("--v-h", type=float, default=5.0, help="horizontal speed limit [m/s]")
    ap.add_argument("--mode", choices=["master", "independent", "fixed"], default="master")
    ap.add_argument("--policy", choices=["expert", "learned"], default="expert")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--stage-a", action="store_true", help="Stage-A flow-time guidance (learned policy)")
    ap.add_argument("--f-master", type=float, default=2.0)
    ap.add_argument("--f-dec", type=float, default=30.0)
    ap.add_argument("--latency", type=float, default=0.015, help="clone inference + transport latency [s]")
    ap.add_argument("--link-latency", type=float, default=0.01, help="downlink latency [s]")
    ap.add_argument("--packet-loss", type=float, default=0.0, help="safety-broadcast loss probability")
    ap.add_argument("--angle-jitter", type=float, default=0.0, help="start-angle jitter [deg] (0 = symmetric)")
    ap.add_argument("--t-max", type=float, default=60.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.policy == "learned" and not args.checkpoint:
        ap.error("--policy learned needs --checkpoint")
    if args.task == "t2":
        if args.mode == "fixed":
            ap.error("--task t2 supports --mode master or independent")
        print(f"CMA-VPA SITL demo, Task T2: two Panda arms, instruction '{args.instruction}', occlusion "
              f"{args.occlusion}, policy {args.policy}, mode {args.mode}{', NO Stage B' if args.no_stage_b else ''}; "
              f"loops master {args.f_master:g} Hz / clone {args.f_dec:g} Hz / executor 100 Hz")
        results = [asyncio.run(episode_t2(args, args.seed + k)) for k in range(args.episodes)]
        report_t2(args, results)
        return results

    print(f"CMA-VPA SITL demo: N_a = {args.agents}, wind {args.wind}, policy {args.policy}, mode {args.mode}; "
          f"loops master {args.f_master:g} Hz / clone {args.f_dec:g} Hz / executor 100 Hz")
    results = []
    for k in range(args.episodes):
        r = asyncio.run(episode(args, args.seed + k))
        results.append(r)
        ms = f"{r['makespan']:.2f} s" if math.isfinite(r["makespan"]) else f"n/a ({r['arrived']}/{args.agents} arrived)"
        print(f"\nepisode seed {r['seed']}: {'SUCCESS' if r['success'] else 'FAILURE'}")
        print(f"  makespan                  {ms}")
        print(f"  minimum separation d_min  {r['d_min']:.3f} m   (collision threshold 0.70 m, D = 1.20 m)")
        print(f"  collisions                {r['collisions']}")
        print(f"  Stage-B interventions     {r['interventions']} agent-ticks")
        print(f"  fallback onsets           {r['fallbacks']} (Prop. 9 certified {r['fallback_certified']}; "
              f"causes {_fmt_causes(r['fallback_causes'])}); braking {r['braking_ticks']} agent-ticks")
        print(f"  loops                     {r['master_cycles']} master cycles, {r['decisions']} clone decisions, "
              f"{round(r['sim_time'] / 0.01) + 1} executor ticks in {r['sim_time']:.2f} s simulated "
              f"({r['wall']:.1f} s wall)")
        if r["tick_ms"]:
            print(f"  executor tick (team)      p50 {r['tick_ms']['p50']:.2f} ms, p99 {r['tick_ms']['p99']:.2f} ms")
        if r["decide_ms"]:
            print(f"  clone decision (team)     median {statistics.median(r['decide_ms']):.2f} ms")
        sim = max(r["sim_time"], 1e-9)
        bus = (r["uplinks"] * UPLINK_SIZE + r["downlinks"] * DOWNLINK_SIZE) / sim
        print(f"  traffic                   latent bus {bus:.0f} B/s, safety channel "
              f"{r['safety_packets'] * SAFETY_SIZE / sim:.0f} B/s")
    if len(results) > 1:
        succ = sum(r["success"] for r in results)
        mk = [r["makespan"] for r in results if math.isfinite(r["makespan"])]
        line = f"\nsummary over {len(results)} episodes: success {succ}/{len(results)}"
        if mk:
            line += f", makespan mean {statistics.mean(mk):.2f} s"
        line += (f", min d_min {min(r['d_min'] for r in results):.3f} m, collisions "
                 f"{sum(r['collisions'] for r in results)}, fallbacks {sum(r['fallbacks'] for r in results)}")
        print(line)
    return results


if __name__ == "__main__":
    main()

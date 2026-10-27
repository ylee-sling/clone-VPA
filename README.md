# Continuous Multi-Agent Vision-Predictor-Action (CMA-VPA)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23679374.svg)](https://doi.org/10.5281/zenodo.23679374)
[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue.svg)](LICENSE)
[![tests](https://github.com/ylee-sling/clone-VPA/actions/workflows/tests.yml/badge.svg)](https://github.com/ylee-sling/clone-VPA/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![Built on VPA](https://img.shields.io/badge/built%20on-Vision--Predictor--Action-6f42c1.svg)](https://github.com/ylee-sling/Vision-Predictor-Action)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-2ea44f.svg)](CITATION.cff)

Official open-source implementation of:

> **Dynamic Latent Cloning with Continuous Barrier Guidance: Multi-Agent Vision-Predictor-Action**  
> **Yeonseok Lee**  
> *SLING AI Inc., Incheon, Republic of Korea*  
> Email: ylee@sling.ai.kr  
> DOI: [https://doi.org/10.5281/zenodo.23679374](https://doi.org/10.5281/zenodo.23679374)  
> Built upon the single-agent foundation: [https://github.com/ylee-sling/Vision-Predictor-Action](https://github.com/ylee-sling/Vision-Predictor-Action)

CMA-VPA extends the single-agent [Vision-Predictor-Action](https://github.com/ylee-sling/Vision-Predictor-Action)
pipeline to teams of robots.

* A **PR-JEPA master** (1–5 Hz, off the critical path) reads a 35-byte quantized latent uplink from every
  agent, **samples one joint mode** and returns a quantized plan per agent over a 36-byte downlink.
* Each **clone** runs a decision chain of exactly **K + 3** network evaluations: ViT encoder, primitive
  selector, ensemble JEPA predictor and K flow-matching steps.
* A decentralized **CBF-DeGG-Flow** safety layer certifies forward invariance of the collision-free set. It
  combines reciprocal high-order barrier shares with explicit margins, flow-time endpoint projection
  (Stage A) and a 100 Hz knot projection (Stage B).
* The barrier never enters a training loss.

| Task | Agents | Action (d_a) | Safety geometry | Entry point |
|---|---|---|---|---|
| **T1** antipodal crossing | 4–16 quadrotors, wind W0 / W1 / W2 | velocity setpoint and yaw rate (4) | one sphere per vehicle; 22-byte broadcasts at 1 cm | `cma_vpa/envs/antipodal_crossing.py` |
| **T2** dual-arm assembly | 2 Franka Panda arms at a shared fixture | end-effector twist (v, ω) and gripper (7) | 5 spheres per hand (Remark 1); twist-space Stage B; one 22-byte packet per sphere at 1 mm | `cma_vpa/envs/dual_arm_assembly.py` |

---

## Installation

```bash
git clone https://github.com/ylee-sling/clone-VPA.git
cd clone-VPA
pip install -e .                  # torch >= 2.3, numpy >= 1.24
pip install -e ".[test]"          # + pytest, scipy (reference QP solver in the tests)
pip install -e ".[maniskill]"     # optional: ManiSkill3 + gymnasium (make_maniskill_env)
pip install -e ".[clip]"          # optional: transformers, for the frozen CLIP text encoder
```

Everything runs on CPU. On a machine without a GPU, `pip install torch --index-url
https://download.pytorch.org/whl/cpu` first, as the CI workflow does. The training scripts take `--device auto`,
which picks CUDA or MPS when available.

## Repository layout

```
clone-VPA/
├── cma_vpa/
│   ├── comm/
│   │   ├── rvq.py               24-stage residual VQ (256 x 64 codebooks, zero codeword, Lemma 4, Prop. 5)
│   │   └── packet.py            Bit-exact uplink (35 B), downlink (36 B), safety (22 B) codecs, CRC-16/CCITT;
│   │                            configurable safety units (Table 1 cm for T1, mm for arm spheres)
│   ├── models/
│   │   ├── encoder.py           Siamese ViT-S/16 E_psi (width 384, depth 6, d = 256), text projection, EMA target
│   │   ├── selector.py          Primitive selector pi_phi^h (433 -> 384 -> 384 -> 6), milestone pointer, goal feature
│   │   ├── predictor.py         Ensemble P_omega (5 heads, 292 -> 512 -> 512 -> 256), sigma filter, H_val
│   │   ├── bridge.py            Plan tokens (Eq. 16) and the double-buffered K/V cache with null plan
│   │   ├── solver.py            CFM vector field v_theta (width 256, depth 4, 2 KV-cached cross-attention layers)
│   │   ├── master.py            Set-transformer J_Omega, Eq. (11) mode sampling, Eq. (12) plan ensemble
│   │   └── clone.py             The K + 3 DECIDE chain with runtime depth audit, ONDOWNLINK, uplinks;
│   │                            CloneConfig.arm() for d_a = 7
│   ├── safety/
│   │   ├── params.py            Table 4 defaults (R_link = 32 m)
│   │   ├── hocbf.py             Braking-consistent HOCBF (Lemma 8, Eqs. 23-25)
│   │   ├── margins.py           Reciprocal shares with margins, inactivity certificates, R_com (Lemmas 12-15)
│   │   ├── qp_solvers.py        Lemma 18 closed form, grouped Hildreth + exact polish, Stage-B routing
│   │   ├── degg_flow.py         Stage A polytope and GUIDEDSOLVE (Algorithm 2, Lemma 17)
│   │   ├── fallback.py          Jerk-optimal braking (Eq. 36), Prop. 9 envelope
│   │   ├── executor.py          100 Hz TICK (Algorithm 1), radius admission, PX4 setpoints
│   │   └── arm_barrier.py       Remark 1: collision-sphere kinematics, drift bound in e_a, twist-space
│   │                            Stage B, twist braking and the 100 Hz arm executor
│   ├── training/
│   │   ├── losses.py            VICReg (Eq. 8), SIGReg, JEPA, CFM (Eq. 17), bus, master and policy losses
│   │   ├── data.py              Expert-episode collection, storage, rendering, samplers
│   │   ├── common.py            Devices, checkpoints, freezing, team batches
│   │   ├── stage1_repr.py       Stage 1: JEPA representation, EMA target, frozen gamma_bar* and statistics
│   │   ├── stage2_bus.py        Stage 2: P_bus (PCA init) and RVQ codebooks (k-means init)
│   │   ├── stage3_master.py     Stage 3: master on decoded quantized uplinks, mode cross-entropy
│   │   ├── stage4_policy.py     Stage 4: selector + v_theta on master-predicted quantized plans (dropout 0.1)
│   │   └── stage5_dagger.py     Stage 5: closed-loop DAgger with privileged expert relabelling
│   └── envs/
│       ├── quadrotor_sim.py     Disturbed double integrator, zero-order hold at 10 ms, W0/W1/W2 wind (Dryden)
│       ├── render.py            Procedural RGB-D cameras: T1 front camera, z-buffered sphere renderer for T2
│       ├── expert.py            Privileged team expert with shared passing handedness, labelling rules
│       ├── antipodal_crossing.py  Task T1 and its multi-rate closed-loop runner
│       ├── dual_arm_sim.py      Panda kinematics (modified DH, Jacobian, DLS IK) and the twist-controlled
│       │                        dual-arm simulator (100 Hz, zero-order hold)
│       └── dual_arm_assembly.py Task T2 with a ManiSkill3-style API, expert, role-order master, runner,
│                                pd_ee_delta_pose conversion
├── tests/                       145 tests (one opt-in torch.compile test, CMA_VPA_TEST_COMPILE=1)
├── run_sitl_demo.py             asyncio master / clone / executor loops on a virtual clock (T1 and T2)
├── push_to_github.sh            one-shot publishing script (see "Releasing to GitHub")
├── .github/workflows/tests.yml  CI: CPU PyTorch on Python 3.10 and 3.12
├── CITATION.cff                 citation metadata ("Cite this repository" on GitHub)
├── LICENSE                      GPL-3.0-only, the license of the single-agent VPA
└── pyproject.toml
```

## Quick start

```bash
python -m pytest                          # 145 tests (144 run, 1 opt-in), about 2 minutes on a 2-core CPU

# Task T1: antipodal crossing
python run_sitl_demo.py                                   # N_a = 8, wind W1, master joint mode
python run_sitl_demo.py --mode independent                # every vehicle chooses its passing side alone
python run_sitl_demo.py --agents 16 --wind W2 --episodes 3

# Task T2: dual-arm assembly
python run_sitl_demo.py --task t2                         # "insert the red part first, then the blue part"
python run_sitl_demo.py --task t2 --instruction any --occlusion severe --episodes 3
python run_sitl_demo.py --task t2 --instruction any --mode independent   # every arm chooses its role alone
```

Task T2 can also be used directly. It follows the ManiSkill3 multi-agent conventions (batched torch tensors,
a `Dict` action space keyed by `panda-0` and `panda-1`):

```python
import torch
from cma_vpa.envs import (AGENT_UIDS, AssemblyConfig, AssemblyModeMaster, AssemblyRunner, DualArmAssemblyEnv,
                          ExpertAssemblyPolicy)

# gymnasium-style stepping with batched tensors
env = DualArmAssemblyEnv(AssemblyConfig(num_envs=4, instruction="any", occlusion="moderate", obs_mode="rgbd"))
obs, info = env.reset(seed=0)
print(obs["sensor_data"]["panda-0"]["rgbd"].shape)        # torch.Size([4, 8, 128, 160]): wrist + third-person RGB-D
action = {uid: torch.zeros(4, 7) for uid in AGENT_UIDS}  # normalized base-frame twist (6) + gripper (1)
obs, reward, terminated, truncated, info = env.step(action)

# the privileged expert behind the master / clone message path and the arm executor
env = DualArmAssemblyEnv(AssemblyConfig(num_envs=4, instruction="any"))   # state observations
g = torch.Generator().manual_seed(0)
result = AssemblyRunner(env, ExpertAssemblyPolicy(env, generator=g), AssemblyModeMaster(env, g), generator=g).run()
print(result.success, result.assembly_time, result.min_clearance)
```

The runner steps the environment at 100 Hz and `env.step` builds the configured observation every tick, so
runner environments should keep the default `state_dict` observations. A learned clone renders its own
views at the 30 Hz decision rate.

### Training pipeline (Section 4.8)

Each stage loads the previous checkpoint, freezes what the paper freezes and saves a new checkpoint.
The barrier never enters a loss; it acts only inside the Stage-5 rollouts.

```bash
python -m cma_vpa.training.data          --out t1.pt --episodes 40
python -m cma_vpa.training.stage1_repr   --data t1.pt --steps 20000 --out ckpt/stage1.pt
python -m cma_vpa.training.stage2_bus    --data t1.pt --init ckpt/stage1.pt --out ckpt/stage2.pt
python -m cma_vpa.training.stage3_master --data t1.pt --init ckpt/stage2.pt --out ckpt/stage3.pt
python -m cma_vpa.training.stage4_policy --data t1.pt --init ckpt/stage3.pt --out ckpt/stage4.pt
python -m cma_vpa.training.stage5_dagger --data t1.pt --init ckpt/stage4.pt --out ckpt/stage5.pt
python run_sitl_demo.py --policy learned --checkpoint ckpt/stage5.pt --stage-a
```

`--tiny` switches every stage to a small model with the same structure for quick experiments.

The stage scripts collect and label Task T1 data. For Task T2 the pieces are in place:

* `AssemblyRunner(..., record=True)` records the plant state, the expert chunks and the primitives at every
  decision.
* `env.render_agents()` renders the eight-channel views.
* `order_label` and `role_labels` give the mode labels.
* `CloneConfig.arm()` builds the d_a = 7 clone, and `LearnedAssemblyPolicy` runs it.

Adapting `training/data.py` to these records is the remaining step for a trained T2 policy (see Status).

## What is verified

### Task T1 and the shared stack

| Statement | Check | Result |
|---|---|---|
| Table 3 | Analytic parameter count per module | Exact for every row; total **18,669,514** (plus 704 for the two 32-dimensional embedding tables) |
| Prop. 4(a), Table 1 | Packet sizes | 35 / 36 / 22 bytes; CRC detects every single-bit error |
| Lemma 4 | RVQ residual norms at scales 1e-4 to 1e3 | Non-increasing exactly; error ≤ ‖x‖; ‖x̂‖ ≤ 2‖x‖ |
| Prop. 5 | Gaussian floor | Relative MSE ≥ 2^-6 (18.06 dB) |
| Prop. 8 | Forward-hook audit over K in {1, 2, 3, 5} and H in {4, 16} | Exactly `[E_psi, pi_phi_h, P_omega] + [v_theta] * K`; bridge K/V never computed inside a decision |
| Prop. 4(c) | Two clones receive the same downlink | Bit-identical K/V |
| Eq. 11 | Mode sampling | `u_g` sampled (reproducible per generator); `u_M` is the deterministic argmax |
| Master | Permutations, team-size changes, identical agents | Equivariant and invariant; absent slots have no influence |
| Prop. 7 / Table 10 | Two-agent head-on crossing, L = 20 m | (a) Standoff at x = ±0.616 m, d_min = 1.231 m, lateral exactly 0. (b) Drift to y = ∓3.33 m, relative offset exactly 0. (c) Relative lateral acceleration exactly −4 h v_lat, offset up to 4.55 m, d_min = 3.108 m, both goals reached at **5.24 s**, which matches the paper |
| Table 9 | Barrier, margins, Theorem 1 | Reproduced in `tests/test_safety.py`, including λ_min = 1.261, α1''(0) = −79.86 and R_com = 29.864 m |
| Section 4.8 | Five steps per stage | No NaN; frozen modules bit-identical and without gradients; checkpoint reload deterministic; no barrier call in any loss |

### Task T2 (`tests/test_dual_arm.py`)

| Statement | Check | Result |
|---|---|---|
| Panda kinematics | FK at the ready pose; Jacobian against finite differences; IK at every task pose | TCP at (0.3069, 0, 0.4869) m and flange at z = 0.5903 m, as on the robot; Jacobian within 1e-5; IK residual below 1e-6 inside the joint limits |
| Remark 1, velocity | ċ = v + ω × (c − p) against finite differences of the rigid motion | Linear in the twist (J_c ξ within 1e-12; finite differences within 1e-5) |
| Remark 1, drift | c̈ = v̇ + ω̇ × r + ω × (ω × r) against central differences | Matches; ‖ω × (ω × r)‖ ≤ ‖ω‖²‖r‖, with equality for ω ⊥ r |
| Assumption 1 for spheres | `arm_safety_params()` | ē_a = e_track + ω_max² r_max = 0.557 m/s², u_∞ = a_lin + √2 a_ang r_max, v̄ = v_max + ω_max r_max; `check_params` reports nothing; λ_b ≥ λ_min(D) for the nominal and the largest D |
| Eq. 34 with d = 6 | Twist-space Stage B against SciPy SLSQP, at least 4 active shares | Same feasibility verdict; solutions agree within 1e-6 |
| Lemma 18 | 100,000 random sphere-pair states in C_HO | Ill-posed shares only within 1 cm of D; every active share more than 2 cm beyond D is feasible with the closed form |
| Theorem 1, closed loop | Two hands driven head-on at 0.4 m/s, with and without spin | Every sphere pair stays at least D apart with Stage B, and the gripper passes through; without Stage B the hands collide |
| d_a = 7 clone | `CloneConfig.arm()` decision | Chunk (H, 7), chain of K + 3 evaluations, attenuation on the twist only |
| Sphere packets | 22-byte packets at 1 mm | Position and velocity within 0.5 mm; radius never under-reported; T1 centimetre packets unchanged |
| ManiSkill3-style API | Spaces, `state_dict` / `state` / `rgbd` observations, action normalization, `pd_ee_delta_pose` conversion | Shapes and round trips exact |
| Joint mode | Batched instructions red first / blue first / any | Every episode assembles in the instructed (or sampled) order without contact |
| Incompatible roles | Fixed roles contrary to the master | Wrong order fails the predicate; two followers deadlock; two leaders stand off without contact; without Stage B they collide |

## SITL results

These numbers come from the privileged expert driving the real master/clone message path and the real
safety layer. They do not evaluate a trained network.

### Task T1 (wind W1 unless noted)

| Setting | Success | Makespan | d_min | Stage-B interventions | Fallbacks |
|---|---|---|---|---|---|
| N_a = 8, master joint mode, symmetric start | 1/1 | 8.69 s | 1.922 m | 0 | 0 |
| N_a = 8, master joint mode, ±4° jitter | 4/4 | 8.68 s | ≥ 1.677 m | 0–15 agent-ticks | 0 |
| N_a = 8, W2, 5 % safety-packet loss | 2/2 | 8.69 s | ≥ 1.945 m | 0 | 0 |
| N_a = 16, W2, master joint mode | 3/3 | 8.84 s | ≥ 1.511 m | 0 | 0 |
| N_a = 8, **independent** sides | 1/4 | 20.07 s | ≥ 0.977 m | 677–1882 per episode | 24–147 per episode, none Prop. 9-certified |

With independent sides the team deadlocks (Proposition 7(b) at team scale). It recovers only through
repeated braking cascades. During those cascades separations fall below D = 1.2 m but stay above the
0.70 m collision threshold, which is the regime that Proposition 9 does not certify.

### Task T2 (two Panda arms, 40 s limit, tracking disturbance e_track = 0.3 m/s²)

The privileged expert does not use the rendered views, so occlusion has no effect on these runs; it
matters only for a learned clone.

**SITL demo** (`run_sitl_demo.py --task t2`: asynchronous loops on the virtual clock, one environment)

| Setting | Success | Assembly time | Min clearance, hands / links | Stage-B p95 | Fallbacks | Executor tick p50 / p99 |
|---|---|---|---|---|---|---|
| Red part first, master joint mode | 1/1 | 14.05 s | 0.160 m / 0.245 m | 0 (never active) | 0 | 2.30 / 5.78 ms |
| Any order, severe occlusion, 3 episodes | 3/3 | 13.94 s | ≥ 0.160 m / ≥ 0.241 m | 0 | 0 | ≤ 2.41 / ≤ 6.77 ms |
| Any order, **independent** roles, 4 episodes | 3/4 | 13.94 s | ≥ 0.159 m / ≥ 0.232 m | 0 | 0 | ≤ 2.44 / ≤ 6.00 ms |

In the failed independent episode both arms chose to follow, so each waited at its staging pose for the
other. The links carry about 145–149 B/s of downlinks and 22,016 B/s of safety packets (ten per tick).

**Role-order study** (`AssemblyRunner`, batched environments, privileged expert)

| Roles | Envs | Outcome | Collisions | Min clearance, hands / links | Stage-B p95 |
|---|---|---|---|---|---|
| Master joint mode, any order, moderate occlusion | 2 × 8 | 16/16 assembled, each in the order u_g that the master sampled; mean 13.86 s and 13.90 s | 0 | 0.159 m / 0.233 m | 0 |
| Independent roles, any order | 16 | 10/16 assembled: exactly the 10 environments whose roles differ | 0 | 0.079 m / 0.165 m | 9.45 m/s² |
| Fixed: roles contrary to "red first" | 1 | Both parts placed in the wrong order; predicate false | 0 | 0.160 m / 0.245 m | – |
| Fixed: two followers | 1 | Deadlock; nothing placed | 0 | 0.186 m / 0.250 m | – |
| Fixed: two leaders | 1 | Standoff at the fixture; nothing placed | 0 | 0.080 m / 0.166 m | 9.46 m/s² |
| Fixed: two leaders, Stage B off | 1 | Contact at the fixture | 1 | −0.107 m / −0.114 m (penetration) | off |

Without a shared mode the arms fail in the ways Proposition 7 predicts:

* Two leaders meet at the fixture, and the barrier holds them in a standoff without contact.
* Two followers wait for each other until the 40 s limit.
* Roles that contradict the instruction assemble in the wrong order.

The master's joint mode removes all three failures.

## Implementation notes

The following choices fill places where the paper leaves details open.

### Shared stack and Task T1

* **Downlink byte budget.** Table 1 fixes 36 bytes. A separate status byte would make the message 37
  bytes, so the status flags occupy the upper nibble of the mode-token byte; the 16-row mode table of
  Table 3 needs only 4 bits. The "plan uncertainty and validity" byte holds σ̄_M in its upper nibble
  (steps of 0.125, rounded up) and the validity in its lower nibble (0.25 s units; 0 means the null plan).
* **Wire conventions.** Fields are little-endian and use CRC-16/CCITT-FALSE. Uplink σ̄ is sent in steps
  of 1/64, rounded up, and the radius is rounded up to whole centimetres, so neither value is ever
  under-reported. Timestamps are µs mod 2^32 and are unwrapped against the receiver clock. Sequence
  numbers are filtered with RFC 1982 serial arithmetic.
* **Goal feature and proprioception.** Aero-VPA [26] is not part of this repository, so two inputs are
  defined here:
  * `g_t = (tanh(r/l) n, tanh(r/l))` is computed from the heading-frame goal offset.
  * `S_t ∈ R^13` holds velocity, last setpoint, altitude error, sin/cos yaw, σ̄ and the executor flags.
  * These dimensions reproduce the 433-wide selector input and the 721-wide e_t of Table 3.
* **Master mission tokens.** Agent token i cross-attends to (c̃_text, z_g^(i,m_i)). The milestone tokens
  move with the agents, so the master stays permutation-equivariant without identity embeddings. The
  master has 4.37 M parameters; the paper reports 4.40 M without a breakdown.
* **Stages 1–3.**
  * Stage 1: the ensemble heads use Poisson bootstrap weights so that their disagreement stays epistemic.
  * Stage 2: the bus reconstruction is the tied map P_busᵀ, a PCA-type objective that cannot collapse.
  * Stage 3: the text projection P_c is trained together with the master, because the master is the
    first module that consumes c̃.
  * Stage 3: plan targets use the frozen encoder by default (`--target-encoder ema` uses the stored
    Stage-1 EMA copy instead).
* **Stage 4 conditioning.** The master's global mode is set to the episode's handedness label, so that
  the plan matches the mode that the expert chunk realizes. The local mode is the master's own argmax,
  as at deployment.
* **Expert.** The privileged centralized expert is a roundabout planner with a shared handedness. It
  stands in for the SCP planner [82], and its two modes are exact mirror images (Lemma 3).
* **Simulator.** Wind enters as a drag-type tracking error with slow integral compensation. It is
  clipped to 0.95 ē_a, so Assumption 1 holds by construction. Frames come from a procedural RGB-D
  renderer instead of Isaac Sim.

### Task T2

* **Collision spheres (Remark 1).** Each hand carries five spheres rigidly attached to the TCP: the tool
  tip, the two sides of the hand, the flange and the wrist. Physical radii are 3.5–6 cm, plus a 1 cm
  safety margin. Their
  centers obey ċ = v + ω × r with r = c − p, so every barrier quantity of Section 4.6 is evaluated per
  sphere pair between the two arms.
* **Disturbance and speed bounds.** The centrifugal term ω × (ω × r) is bounded by ω_max² r_max and
  added to the disturbance bound: ē_a = e_track + ω_max² r_max = 0.557 m/s². The input bound is
  u_∞ = a_lin + √2 a_ang r_max and the speed bound is v̄ = v_max + ω_max r_max.
* **Stage B in twist space.**
  * A share gᵀc̈ ≥ ζ of sphere k becomes the row [g; r_k × g] on the twist rate u = (v̇, ω̇).
  * Eq. 34 stays a 6-variable QP. Its box is the acceleration box intersected with the per-axis speed
    limits, which makes it asymmetric; Lemma 18 takes general bounds.
  * The projection uses the metric normalized by the acceleration bounds, so that the linear and angular
    channels are commensurable.
  * One active row is solved in closed form. More rows use grouped Hildreth with exact polish and an exact
    finisher.
  * The gripper channel passes through unchanged (Section 4.5).
* **Barrier constants for arms.** `arm_safety_params()` uses a_max = 3 m/s², ς = 0.4 m/s and λ_b = 6.
  * The arm disturbance is a large fraction of a_max, and with ς = 0.25 some well-posed shares were
    infeasible up to 12.6 cm beyond D.
  * With ς = 0.4 the ill-posed layer shrinks to 1 cm, and every share more than 2 cm beyond D is
    feasible.
* **Standoff distance and bin spacing.**
  * At small ρ the margin term 4ρ⁺ē_a dominates λ_b α₁, so the barrier holds static sphere pairs about
    7 cm beyond D.
  * The two part bins are therefore 0.30 m apart (y = ±0.15 m, jitter 1.5 cm), so simultaneous grasps
    stay outside that standoff.
* **Safety packets.** There is one 22-byte packet per sphere, with `agent_id = 8·arm + sphere`, at 1 mm
  and 1 mm/s resolution (`ARM_UNITS`). T1 keeps the Table 1 centimetre units. Two arms with five spheres
  at 100 Hz give about 22 kB/s.
* **Fallback.** Eq. 36 brakes the whole twist with one common duration,
  T_b = 3/2 max(‖v₀‖/a_brk,lin, ‖ω₀‖/a_brk,ang), then holds the pose. The last gripper command is held
  while braking.
* **Modes.**
  * The global mode is the assembly order: 0 is red first, 1 is blue first. The master samples it when
    the instruction is "any".
  * The local mode of each arm is its role, 0 leader or 1 follower, carried in the downlink's mode token.
  * The follower enters the shared zone only after the leader has placed its part and left the zone.
* **Clone inputs.**
  * `CloneConfig.arm()` takes eight input channels, wrist RGB-D and third-person RGB-D, and has
    d_a = 7. The attenuation α acts on the six twist channels only.
  * `S_t ∈ R^13` holds the joints (7), the gripper opening (1), the TCP position relative to the fixture
    in the base frame (3), σ̄ (1) and the braking flag (1). This keeps the Table 3 widths.
  * `g_t ∈ R^3` is the base-frame offset to the current goal.
* **Simulator.**
  * The end effectors follow twist-rate dynamics under zero-order hold at 10 ms, and the joints follow
    through damped least-squares IK every tick.
  * Kinematics use modified DH (Craig) with the datasheet joint limits.
  * The tracking-error disturbance is scaled so that every sphere sees at most e_track.
  * Grasped parts attach kinematically; there is no contact dynamics.
  * The proximal links are checked for collision but are not covered by the barrier (Remark 1).
* **ManiSkill3.**
  * `twist_to_pd_ee_delta_pose` and `pd_ee_delta_pose_to_twist` convert to and from the 7-dimensional
    `pd_ee_delta_pose` action (0.1 m and 0.1 rad per step; opposite gripper sign).
  * `make_maniskill_env` creates `TwoRobotStackCube-v1` with that controller at 100 Hz. The tests do not
    exercise it, because ManiSkill3 is not a dependency.

## Status

* **Section 7 hardware and simulator experiments.** The paper pre-registers Isaac Sim / Pegasus, PX4 SITL
  and HITL, Jetson latency and the ManiSkill3 / SAPIEN arm experiments. They are not part of this code.
  The lightweight simulators here stand in for them.
* **Stage A.** Guidance runs in eager PyTorch at about 125–150 ms per decision, so the 30 Hz clone budget
  needs a native QP port.
* **Task T2 training.** The stage scripts are written for T1 data. The T2 environment, labels, recording
  and the d_a = 7 clone are in place, but the stage adapters for T2 are not.

## Publishing with `push_to_github.sh`

`push_to_github.sh` publishes this tree to an empty GitHub repository in one step. Run it from the
repository root:

```bash
./push_to_github.sh                                                      # HTTPS (token or `gh auth login`)
REMOTE_URL=git@github.com:ylee-sling/clone-VPA.git ./push_to_github.sh   # SSH key instead
FORCE_PUSH=1 ./push_to_github.sh                                         # replace an unrelated remote main
```

The script makes six steps:

1. It runs `git init` if needed and selects `main`.
2. It sets a repository-local identity only if git has none.
3. It points `origin` at `https://github.com/ylee-sling/clone-VPA.git`.
4. It completes `.gitignore` (checkpoints, data caches, `.pytest_cache`, `__pycache__`).
5. It stages everything and commits with the message
   `feat: complete CMA-VPA implementation with Task T1, Task T2, and CBF-DeGG-Flow`.
6. It pushes to `main`.

Re-running it with no changes commits nothing. If the remote `main` already holds commits that are not in
the local history, for example a README created on GitHub, the script stops and explains how to merge
them, or how to overwrite them with `FORCE_PUSH=1` (which uses `--force-with-lease`). `BRANCH`,
`GIT_USER_NAME` and `GIT_USER_EMAIL` can be overridden the same way. The script always commits with the
release message, so later changes should be committed with ordinary `git commit` and `git push`.

## Citation

If you use this code, please cite the paper. GitHub reads [`CITATION.cff`](CITATION.cff) and offers the
same entry under "Cite this repository".

```bibtex
@misc{lee2026cmavpa,
  title     = {Dynamic Latent Cloning with Continuous Barrier Guidance: Multi-Agent Vision-Predictor-Action},
  author    = {Lee, Yeonseok},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23679374},
  url       = {https://doi.org/10.5281/zenodo.23679374}
}
```

The single-agent foundation:

```bibtex
@misc{lee_vpa,
  title        = {Vision-Predictor-Action (VPA)},
  author       = {Lee, Yeonseok},
  howpublished = {\url{https://github.com/ylee-sling/Vision-Predictor-Action}},
  note         = {Single-agent foundation of CMA-VPA}
}
```

CMA-VPA also builds on the following work:

* MA-JEPA, centralized joint-embedding world models for multi-agent dynamics (Kaplowitz et al., 2026).
* DeGG-Flow, decoupled generative guidance (Lin et al., 2026).
* Reciprocal safety barrier certificates (Wang et al., 2017).
* High-order control barrier functions (Xiao and Belta, 2022).

The paper lists the full references.

## License

The code is licensed under **GPL-3.0-only**, the same license as the single-agent
[Vision-Predictor-Action](https://github.com/ylee-sling/Vision-Predictor-Action); see [`LICENSE`](LICENSE).
Every source file starts with the SPDX identifier `GPL-3.0-only` and the notice
`Copyright (C) 2026 SLING AI Inc.`

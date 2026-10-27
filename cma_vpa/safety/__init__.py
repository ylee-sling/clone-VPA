# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""CBF-DeGG-Flow safety engine (Section 4.6 and 4.7 of the CMA-VPA paper).

Modules
-------
params      Table 4 defaults (:class:`SafetyParams`).
hocbf       Braking-consistent HOCBF of relative degree two (Lemma 8, Eq. 23-25).
margins     Reciprocal shares with sampled-data and estimation margins, inactivity
            certificates and the communication radius (Eq. 27-31, Lemmas 12-15).
qp_solvers  Lemma 18 closed form, grouped Hildreth with exact polish (PDAS,
            Goldfarb-Idnani), Stage-B projection (Eq. 34).
degg_flow   Stage A: polyhedron P_t and endpoint-projection guidance (Eq. 32-33,
            Lemmas 16-17, Algorithm 2 GUIDEDSOLVE).
fallback    Jerk-optimal braking (Eq. 36) and its envelope (Prop. 9, Eq. 41).
executor    100 Hz executor (Algorithm 1 TICK): blending, Stage B, fallback,
            radius admission, PX4 output and safety broadcast.
arm_barrier Arms (Remark 1, Task T2): end-effector collision spheres, sphere-pair
            shares, twist-space Stage B, twist braking and the arm executor.
"""

from .params import DEFAULT_PARAMS, SafetyParams
from .hocbf import (HOCBFTerms, alpha1, alpha1_prime, alpha1_second, alpha_terms_b,
                    alpha_terms_rho, closing_speed_limit, hocbf_terms, lambda_min,
                    pairwise_relative, share_slack, spec_barrier)
from .margins import (ShareSet, check_params, communication_radius, estimation_margin,
                      inactivity_certificates, one_tick_decrease_bound, one_tick_reachable_box,
                      reciprocal_shares, reciprocal_shares_reference)
from .qp_solvers import (BoxHalfspaceResult, HildrethResult, StageBResult, enumerate_small_qp,
                         greedy_row_coloring, hildreth, project_box_halfspace,
                         single_constraint_feasible, solve_stage_b_qp, stage_b_fixed)
from .degg_flow import (ChunkMap, GuidedSolveResult, Polyhedron, StageAPolytope,
                        build_stage_a_polytope, constant_velocity_reference, default_nu_schedule,
                        guided_solve, shift_reference)
from .fallback import (BrakingFallback, braking_duration, braking_profile, fallback_envelope,
                       is_fallback_certified)
from .executor import (Chunk, ChunkBuffer, Executor, ExecutorCommand, ExecutorConfig, NeighborTable,
                       SafetyBroadcast, neighbor_tables_from_broadcasts, px4_offboard_control_mode,
                       px4_trajectory_setpoint, requested_radius)
from .arm_barrier import (PANDA_SPHERES, ArmCommand, ArmExecutor, ArmExecutorConfig, ArmLimits, SphereSet,
                          SphereState, SphereTable, arm_safety_params, arm_shares, centrifugal_drift, drift_bound,
                          solve_twist_qp, sphere_acceleration, sphere_jacobian, sphere_kinematics, twist_box,
                          twist_rows)

__all__ = [
    "DEFAULT_PARAMS", "SafetyParams",
    "HOCBFTerms", "alpha1", "alpha1_prime", "alpha1_second", "alpha_terms_b", "alpha_terms_rho",
    "closing_speed_limit", "hocbf_terms", "lambda_min", "pairwise_relative", "share_slack",
    "spec_barrier",
    "ShareSet", "check_params", "communication_radius", "estimation_margin",
    "inactivity_certificates", "one_tick_decrease_bound", "one_tick_reachable_box",
    "reciprocal_shares", "reciprocal_shares_reference",
    "BoxHalfspaceResult", "HildrethResult", "StageBResult", "enumerate_small_qp",
    "greedy_row_coloring", "hildreth", "project_box_halfspace", "single_constraint_feasible",
    "solve_stage_b_qp", "stage_b_fixed",
    "ChunkMap", "GuidedSolveResult", "Polyhedron", "StageAPolytope", "build_stage_a_polytope",
    "constant_velocity_reference", "default_nu_schedule", "guided_solve", "shift_reference",
    "BrakingFallback", "braking_duration", "braking_profile", "fallback_envelope",
    "is_fallback_certified",
    "Chunk", "ChunkBuffer", "Executor", "ExecutorCommand", "ExecutorConfig", "NeighborTable",
    "SafetyBroadcast", "neighbor_tables_from_broadcasts", "px4_offboard_control_mode",
    "px4_trajectory_setpoint", "requested_radius",
    "PANDA_SPHERES", "ArmCommand", "ArmExecutor", "ArmExecutorConfig", "ArmLimits", "SphereSet", "SphereState",
    "SphereTable", "arm_safety_params", "arm_shares", "centrifugal_drift", "drift_bound", "solve_twist_qp",
    "sphere_acceleration", "sphere_jacobian", "sphere_kinematics", "twist_box", "twist_rows",
]

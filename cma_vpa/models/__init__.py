# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Neural modules of the master and the clones (Sections 3.4, 4.2-4.5, Table 3)."""

from .encoder import (ClipTextEmbedder, EMATarget, HashedTextEmbedder, SiameseViTEncoder, TextProjection,
                      latent_scale, momentum_schedule)
from .selector import PRIMITIVES, MilestoneTracker, PrimitiveSelector, goal_feature
from .predictor import EnsembleLinear, EnsemblePredictor, PredictorOutput, UncertaintyFilter
from .solver import CachedCrossAttention, FlowSolver, FlowVectorField, NoisePersistence, PlanKV, SinusoidalTime
from .bridge import BridgeCache, PlanTokenDecoder
from .master import MasterConfig, MasterOutput, MasterPlanner, SetLayer, handedness_of_mode
from .clone import TABLE3_BUDGET, BusProjection, ChunkSlots, CloneConfig, ClonePolicy, DecisionOutput

__all__ = [
    "ClipTextEmbedder", "EMATarget", "HashedTextEmbedder", "SiameseViTEncoder", "TextProjection",
    "latent_scale", "momentum_schedule",
    "PRIMITIVES", "MilestoneTracker", "PrimitiveSelector", "goal_feature",
    "EnsembleLinear", "EnsemblePredictor", "PredictorOutput", "UncertaintyFilter",
    "CachedCrossAttention", "FlowSolver", "FlowVectorField", "NoisePersistence", "PlanKV", "SinusoidalTime",
    "BridgeCache", "PlanTokenDecoder",
    "MasterConfig", "MasterOutput", "MasterPlanner", "SetLayer", "handedness_of_mode",
    "TABLE3_BUDGET", "BusProjection", "ChunkSlots", "CloneConfig", "ClonePolicy", "DecisionOutput",
]

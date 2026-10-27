# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Staged training pipeline (Section 4.8): losses, data and the five stage scripts.

Run the stages in order (each loads the previous checkpoint)::

    python -m cma_vpa.training.data          --out t1.pt --episodes 40
    python -m cma_vpa.training.stage1_repr   --data t1.pt --out ckpt/stage1.pt
    python -m cma_vpa.training.stage2_bus    --data t1.pt --init ckpt/stage1.pt --out ckpt/stage2.pt
    python -m cma_vpa.training.stage3_master --data t1.pt --init ckpt/stage2.pt --out ckpt/stage3.pt
    python -m cma_vpa.training.stage4_policy --data t1.pt --init ckpt/stage3.pt --out ckpt/stage4.pt
    python -m cma_vpa.training.stage5_dagger --data t1.pt --init ckpt/stage4.pt --out ckpt/stage5.pt
"""

from .losses import (bootstrap_weights, bus_loss, cfm_loss, jepa_loss, master_loss, policy_loss, sigreg,
                     vicreg_covariance, vicreg_variance)

__all__ = ["bootstrap_weights", "bus_loss", "cfm_loss", "jepa_loss", "master_loss", "policy_loss", "sigreg",
           "vicreg_covariance", "vicreg_variance"]

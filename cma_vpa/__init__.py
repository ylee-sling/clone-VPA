# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""CMA-VPA: Continuous Multi-Agent Vision-Predictor-Action.

Reference implementation of "Dynamic Latent Cloning with Continuous Barrier
Guidance: Multi-Agent Vision-Predictor-Action" (Y. Lee, SLING AI Inc.).

Subpackages
-----------
safety
    Braking-consistent HOCBF, reciprocal margins, inactivity certificates and
    real-time QP solvers of CBF-DeGG-Flow (Section 4.6 of the paper).
"""

__version__ = "0.1.0"

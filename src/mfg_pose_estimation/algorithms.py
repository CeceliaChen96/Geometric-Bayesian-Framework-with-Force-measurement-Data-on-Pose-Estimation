"""Posterior-update and refinement algorithms extracted from the notebook."""

from .posterior_core import (
    Pose,
    MFGParameters,
    single_pass_mfg_posterior_update,
    batch_residual_norm,
    batch_whitened_residual_norm,
    relaxed_pose_update,
    build_numeric_rot_only_candidate,
    multi_pass_mfg_batch_refinement,
    pose_cost_whitened,
)

__all__ = [
    "Pose",
    "MFGParameters",
    "single_pass_mfg_posterior_update",
    "batch_residual_norm",
    "batch_whitened_residual_norm",
    "relaxed_pose_update",
    "build_numeric_rot_only_candidate",
    "multi_pass_mfg_batch_refinement",
    "pose_cost_whitened",
]

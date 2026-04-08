"""Matrix-Fisher / MFG helper functions extracted from the notebook."""

from .posterior_core import (
    Pose,
    MFGParameters,
    rotation_mode_from_F,
    recover_pose_from_theta,
    nu_from_rotation,
    split_jacobian_blocks,
)
from .sampling_core import (
    proper_svd,
    mf_diag_to_bingham_params,
    build_acg_shape,
    sample_acg,
    log_target_bingham,
    log_proposal_acg,
    sample_mf_so3_independence_mh,
    canonicalize_quaternion,
)

__all__ = [
    "Pose",
    "MFGParameters",
    "rotation_mode_from_F",
    "recover_pose_from_theta",
    "nu_from_rotation",
    "split_jacobian_blocks",
    "proper_svd",
    "mf_diag_to_bingham_params",
    "build_acg_shape",
    "sample_acg",
    "log_target_bingham",
    "log_proposal_acg",
    "sample_mf_so3_independence_mh",
    "canonicalize_quaternion",
]

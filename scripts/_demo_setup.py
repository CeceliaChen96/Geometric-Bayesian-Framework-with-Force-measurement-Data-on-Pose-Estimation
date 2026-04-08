"""Shared helpers for the runnable demo scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def locate_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    candidates = [start, *start.parents]
    for base in candidates:
        if (base / "src" / "mfg_pose_estimation").exists():
            return base
    script_dir = Path(__file__).resolve().parent
    for base in [script_dir, *script_dir.parents]:
        if (base / "src" / "mfg_pose_estimation").exists():
            return base
    raise RuntimeError("Could not locate project root containing src/mfg_pose_estimation")


PROJECT_ROOT = locate_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mfg_pose_estimation.models import (  # noqa: E402
    Pose,
    MFGParameters,
    SuperquadricFieldParameters,
    StiffnessParameters,
)
from mfg_pose_estimation.geometry import make_rotation_y, make_rotation_z, log_so3  # noqa: E402
from mfg_pose_estimation.wrench_model import measurement_model_y  # noqa: E402
from mfg_pose_estimation.posterior_core import sample_gaussian_noise  # noqa: E402



def build_demo_problem(seed: int = 7, with_noise: bool = True) -> dict[str, Any]:
    np.random.seed(seed)

    field_params = SuperquadricFieldParameters(
        a1=0.04,
        a2=0.03,
        a3=0.05,
        eps1=1.0,
        eps2=1.0,
        sdf_eps=1e-8,
    )

    stiffness_params = StiffnessParameters(
        k_min=0.1,
        k_max=1.0,
        d0=0.05,
    )

    contact_points_A = np.array([
        [0.0,   0.0,   0.0],
        [0.008, 0.0,   0.0],
        [0.0,   0.008, 0.0],
    ])

    sensor_poses_WA = []
    num_meas = 3
    for k in range(num_meas):
        angle = np.deg2rad(-6 + 4 * k)
        R_A = make_rotation_z(angle)
        p_A = np.array([
            -0.004 + 0.002 * k,
             0.002 * np.sin(0.5 * k),
             0.001 * k,
        ])
        sensor_poses_WA.append(Pose(R=R_A, p=p_A))

    R_true = make_rotation_z(np.deg2rad(10.0)) @ make_rotation_y(np.deg2rad(-5.0))
    p_true = np.array([0.008, -0.004, 0.010])
    X_WB_true = Pose(R=R_true, p=p_true)

    Sigma_w = np.diag([0.15**2, 0.15**2, 0.15**2, 0.35**2, 0.35**2, 0.35**2])

    measurements_clean = np.asarray([
        measurement_model_y(X_WA_k, X_WB_true, contact_points_A, field_params, stiffness_params)
        for X_WA_k in sensor_poses_WA
    ])

    if with_noise:
        measurements_Y = np.asarray([
            yk + sample_gaussian_noise(Sigma_w) for yk in measurements_clean
        ])
    else:
        measurements_Y = measurements_clean.copy()

    theta0 = MFGParameters(
        F=np.eye(3),
        mu=np.zeros(3),
        Lambda=60.0 * np.eye(3),
        Gamma=np.zeros((3, 3)),
    )
    X_WB0 = Pose(R=np.eye(3), p=np.zeros(3))

    return {
        "field_params": field_params,
        "stiffness_params": stiffness_params,
        "contact_points_A": contact_points_A,
        "sensor_poses_WA": sensor_poses_WA,
        "X_WB_true": X_WB_true,
        "Sigma_w": Sigma_w,
        "measurements_Y": measurements_Y,
        "measurements_clean": measurements_clean,
        "theta0": theta0,
        "X_WB0": X_WB0,
    }



def pose_errors(X_est: Pose, X_true: Pose) -> dict[str, float]:
    rot_err = float(np.linalg.norm(log_so3(X_est.R.T @ X_true.R)))
    trans_err = float(np.linalg.norm(X_est.p - X_true.p))
    return {
        "rotation_geodesic_error": rot_err,
        "translation_error": trans_err,
    }



def ensure_output_dir(path: str | Path | None) -> Path:
    if path is None:
        path = PROJECT_ROOT / "results" / "demo_runs"
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out



def to_serializable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Pose):
        return {"R": x.R.tolist(), "p": x.p.tolist()}
    if isinstance(x, dict):
        return {str(k): to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]
    return x



def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(to_serializable(payload), indent=2, ensure_ascii=False))

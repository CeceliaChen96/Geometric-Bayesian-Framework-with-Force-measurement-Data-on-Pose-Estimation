"""Run a lightweight Algorithm 1 demo using the extracted package.

This script is the practical replacement for the old import-only placeholder.
It builds a small synthetic problem, optionally checks the Jacobian once,
runs the single-pass posterior update, and writes a summary JSON under
results/demo_runs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _demo_setup import PROJECT_ROOT, build_demo_problem, ensure_output_dir, pose_errors, save_json
from mfg_pose_estimation.algorithms import single_pass_mfg_posterior_update
from mfg_pose_estimation.distributions import recover_pose_from_theta
from mfg_pose_estimation.jacobians import check_algorithm3_jacobian
from mfg_pose_estimation.wrench_model import measurement_model_y



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the package-based Algorithm 1 demo.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic data.")
    parser.add_argument("--h-rot", type=float, default=1e-6, help="Rotation finite-difference step.")
    parser.add_argument("--rot-gain", type=float, default=0.02, help="Algorithm 1 rotation gain.")
    parser.add_argument(
        "--alpha-candidates",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0],
        help="Inner line-search candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "demo_runs",
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--skip-jacobian-check",
        action="store_true",
        help="Skip the one-shot Jacobian diagnostic.",
    )
    parser.add_argument(
        "--noise-free",
        action="store_true",
        help="Use clean synthetic measurements instead of noisy ones.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    problem = build_demo_problem(seed=args.seed, with_noise=not args.noise_free)
    output_dir = ensure_output_dir(args.output_dir)

    jacobian_rel_error = None
    if not args.skip_jacobian_check:
        y0 = measurement_model_y(
            problem["sensor_poses_WA"][0],
            problem["X_WB0"],
            problem["contact_points_A"],
            problem["field_params"],
            problem["stiffness_params"],
        )
        _, _, _, err_rel, _ = check_algorithm3_jacobian(
            y0,
            problem["sensor_poses_WA"][0],
            problem["X_WB0"],
            problem["contact_points_A"],
            problem["field_params"],
            problem["stiffness_params"],
            detailed=False,
        )
        jacobian_rel_error = float(err_rel)

    theta_alg1 = single_pass_mfg_posterior_update(
        theta_prior=problem["theta0"],
        X_WB_bar=problem["X_WB0"],
        measurements_Y=problem["measurements_Y"],
        sensor_poses_WA=problem["sensor_poses_WA"],
        Sigma_w=problem["Sigma_w"],
        contact_points_A=problem["contact_points_A"],
        field_params=problem["field_params"],
        stiffness_params=problem["stiffness_params"],
        h_rot=args.h_rot,
        alpha_candidates=args.alpha_candidates,
        rot_gain=args.rot_gain,
        use_line_search=True,
        verbose_inner=False,
    )

    X_alg1 = recover_pose_from_theta(theta_alg1)
    errors = pose_errors(X_alg1, problem["X_WB_true"])

    summary = {
        "seed": args.seed,
        "noise_free": bool(args.noise_free),
        "jacobian_relative_error": jacobian_rel_error,
        "rotation_geodesic_error": errors["rotation_geodesic_error"],
        "translation_error": errors["translation_error"],
        "X_WB_true": problem["X_WB_true"],
        "X_alg1": X_alg1,
        "theta_alg1": {
            "F": theta_alg1.F,
            "mu": theta_alg1.mu,
            "Lambda": theta_alg1.Lambda,
            "Gamma": theta_alg1.Gamma,
        },
    }

    save_path = output_dir / "alg1_demo_summary.json"
    save_json(save_path, summary)

    print("[Alg1] Demo completed.")
    print(f"[Alg1] Output directory: {output_dir}")
    if jacobian_rel_error is not None:
        print(f"[Alg1] Jacobian relative error: {jacobian_rel_error:.6e}")
    print(f"[Alg1] Rotation geodesic error: {errors['rotation_geodesic_error']:.6e}")
    print(f"[Alg1] Translation error:       {errors['translation_error']:.6e}")
    print(f"[Alg1] Saved summary JSON to:  {save_path}")


if __name__ == "__main__":
    main()

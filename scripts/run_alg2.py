"""Run a lightweight safeguarded Algorithm 2 refinement demo.

This is the practical replacement for the old import-only placeholder.
It builds a small synthetic problem, runs the multi-pass refinement, and saves
both a summary JSON and a history figure under results/demo_runs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import warnings

import matplotlib.pyplot as plt

from _demo_setup import PROJECT_ROOT, build_demo_problem, ensure_output_dir, pose_errors, save_json
from mfg_pose_estimation.algorithms import multi_pass_mfg_batch_refinement
from mfg_pose_estimation.plotting import plot_refinement_history



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the package-based Algorithm 2 demo.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic data.")
    parser.add_argument("--Tmax", type=int, default=1, help="Number of safeguarded outer iterations.")
    parser.add_argument("--h-rot", type=float, default=1e-6, help="Rotation finite-difference step.")
    parser.add_argument("--h-rot-numgrad", type=float, default=1e-5, help="Numeric rotation-gradient step.")
    parser.add_argument("--rot-gain", type=float, default=0.08, help="Relaxed rotation update gain.")
    parser.add_argument("--pos-gain", type=float, default=0.05, help="Relaxed translation update gain.")
    parser.add_argument(
        "--alpha-candidates",
        type=float,
        nargs="+",
        default=[1.0, 2.0],
        help="Inner Algorithm 1 line-search candidates.",
    )
    parser.add_argument(
        "--outer-alpha-candidates",
        type=float,
        nargs="+",
        default=[1.0],
        help="Outer safeguarded line-search candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "demo_runs",
        help="Where to write the summary JSON and figure.",
    )
    parser.add_argument(
        "--noise-free",
        action="store_true",
        help="Use clean synthetic measurements instead of noisy ones.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the refinement internals from the package.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    problem = build_demo_problem(seed=args.seed, with_noise=not args.noise_free)
    output_dir = ensure_output_dir(args.output_dir)

    theta_star, X_star, history = multi_pass_mfg_batch_refinement(
        theta0=problem["theta0"],
        X_WB0=problem["X_WB0"],
        measurements_Y=problem["measurements_Y"],
        sensor_poses_WA=problem["sensor_poses_WA"],
        Sigma_w=problem["Sigma_w"],
        contact_points_A=problem["contact_points_A"],
        field_params=problem["field_params"],
        stiffness_params=problem["stiffness_params"],
        Tmax=args.Tmax,
        h_rot=args.h_rot,
        use_line_search=True,
        rot_gain=args.rot_gain,
        pos_gain=args.pos_gain,
        alpha_candidates=args.alpha_candidates,
        outer_alpha_candidates=args.outer_alpha_candidates,
        use_numeric_rot_candidate=True,
        h_rot_numgrad=args.h_rot_numgrad,
        verbose=args.verbose,
        verbose_inner=False,
        return_history=True,
        X_WB_true=problem["X_WB_true"],
    )

    errors = pose_errors(X_star, problem["X_WB_true"])

    warnings.filterwarnings(
        "ignore",
        message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
        category=UserWarning,
    )

    plot_refinement_history(
        history,
        X_WB_true=problem["X_WB_true"],
        figsize=(10, 10),
        suptitle="Package-based safeguarded Algorithm 2 demo",
    )
    fig_path = output_dir / "alg2_demo_history.png"
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close("all")

    summary = {
        "seed": args.seed,
        "noise_free": bool(args.noise_free),
        "Tmax": args.Tmax,
        "rotation_geodesic_error": errors["rotation_geodesic_error"],
        "translation_error": errors["translation_error"],
        "final_iteration_count": history.get("iter", []),
        "accepted": history.get("accepted", []),
        "X_WB_true": problem["X_WB_true"],
        "X_star": X_star,
        "theta_star": {
            "F": theta_star.F,
            "mu": theta_star.mu,
            "Lambda": theta_star.Lambda,
            "Gamma": theta_star.Gamma,
        },
    }

    save_path = output_dir / "alg2_demo_summary.json"
    save_json(save_path, summary)

    print("[Alg2] Demo completed.")
    print(f"[Alg2] Output directory: {output_dir}")
    print(f"[Alg2] Rotation geodesic error: {errors['rotation_geodesic_error']:.6e}")
    print(f"[Alg2] Translation error:       {errors['translation_error']:.6e}")
    print(f"[Alg2] Saved history figure to: {fig_path}")
    print(f"[Alg2] Saved summary JSON to:   {save_path}")


if __name__ == "__main__":
    main()

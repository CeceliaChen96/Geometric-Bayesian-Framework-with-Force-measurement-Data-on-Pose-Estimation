"""Run lightweight package smoke demos through Python entry points."""

from __future__ import annotations

import argparse
import os
import sys
import subprocess

from common import REPO_ROOT


def run_command(command: list[str]) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("MPLBACKEND", "Agg")
    print("+", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight Algorithm 1/2 package demos.")
    parser.add_argument("--skip-alg1", action="store_true", help="Do not run Algorithm 1 demo.")
    parser.add_argument("--skip-alg2", action="store_true", help="Do not run Algorithm 2 demo.")
    parser.add_argument("--alg2-Tmax", type=int, default=1, help="Outer iterations for Algorithm 2 demo.")
    parser.add_argument("--with-noise", action="store_true", help="Use noisy synthetic demo measurements.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise_flag = [] if args.with_noise else ["--noise-free"]
    if not args.skip_alg1:
        run_command(
            [
                sys.executable,
                "scripts/run_alg1.py",
                "--skip-jacobian-check",
                *noise_flag,
            ]
        )
    if not args.skip_alg2:
        run_command(
            [
                sys.executable,
                "scripts/run_alg2.py",
                "--Tmax",
                str(args.alg2_Tmax),
                *noise_flag,
            ]
        )
    print("Smoke demos completed. Outputs are under results/demo_runs/.")


if __name__ == "__main__":
    main()

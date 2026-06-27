#!/usr/bin/env python3
"""Compute frozen-chart drift diagnostics from saved posterior traces.

The script reads existing trace JSON files from the companion experiment code
repository and writes CSV/JSON/LaTeX summaries inside this paper repository.
It does not modify or rerun the external experiment code.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = Path(
    "/Users/cecelia/Documents/AAA_NTU/AAA_Research/Done/"
    "Bayesian_MFG/mfg_posterior_project/mfg_github/notebooks/"
    "Posterior Estimation"
)

SEEDS = (42, 43, 44)
TRACE_PATTERN = "new_exp2CB_mild_beta1_T15_seed{seed}_posterior_trace_records.json"

# This is the setting used in the saved traces:
# sparse-baseline K_eff=10, beta=1, c_kappa=1, c_Lambda=1.
KAPPA0 = 60.0
EPS = 1e-12


def hat(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < 1e-12:
        return np.eye(3) + K + 0.5 * K @ K
    return (
        np.eye(3)
        + (math.sin(theta) / theta) * K
        + ((1.0 - math.cos(theta)) / (theta * theta)) * (K @ K)
    )


def proper_svd(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proper SVD F = U S V^T with U,V in SO(3)."""
    U0, sigma, Vt0 = np.linalg.svd(np.asarray(F, dtype=float))
    V0 = Vt0.T

    sign_u = 1.0 if np.linalg.det(U0) > 0.0 else -1.0
    sign_v = 1.0 if np.linalg.det(V0) > 0.0 else -1.0

    Du = np.diag([1.0, 1.0, sign_u])
    Dv = np.diag([1.0, 1.0, sign_v])

    U = U0 @ Du
    V = V0 @ Dv
    S = Du @ np.diag(sigma) @ Dv
    return U, S, V


def nu_f(F: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Matrix-Fisher chart coordinate nu_F(R)."""
    U, S, V = proper_svd(F)
    Q = U.T @ R @ V
    return vee(Q @ S - S @ Q.T)


def chart_jacobian(F: np.ndarray, R: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Numerical right-perturbation Jacobian of nu_F at R."""
    J = np.zeros((3, 3), dtype=float)
    basis = np.eye(3)
    for i in range(3):
        R_plus = R @ so3_exp(h * basis[i])
        R_minus = R @ so3_exp(-h * basis[i])
        J[:, i] = (nu_f(F, R_plus) - nu_f(F, R_minus)) / (2.0 * h)
    return J


def safe_cond(A: np.ndarray) -> float:
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] <= 1e-15:
        return float("inf")
    return float(s[0] / s[-1])


def metric_row(seed: int, rec: dict, F_prior: np.ndarray) -> dict:
    R_nom = np.asarray(rec["nominal_R"], dtype=float)
    F_post = np.asarray(rec["F_matrix"], dtype=float)

    J_nu = chart_jacobian(F_prior, R_nom)
    J_post = chart_jacobian(F_post, R_nom)
    delta_J = J_post - J_nu

    rel_delta = float(np.linalg.norm(delta_J, ord="fro") / (np.linalg.norm(J_nu, ord="fro") + EPS))
    abs_delta = float(np.linalg.norm(delta_J, ord="fro"))
    amp_delta = float(np.linalg.norm(np.linalg.solve(J_nu, delta_J), ord=2))
    delta_nu = float(np.linalg.norm(nu_f(F_post, R_nom) - nu_f(F_prior, R_nom)))

    return {
        "seed": int(seed),
        "state_id": int(rec["state_id"]),
        "outer_t": int(rec["outer_t"]),
        "source": str(rec["source"]),
        "branch": "" if rec.get("branch") is None else str(rec.get("branch")),
        "rho": float(rec["rho"]),
        "s_rot": float(rec["s_rot"]),
        "delta_J_abs_F": abs_delta,
        "delta_J_rel_F": rel_delta,
        "eta_J_2": amp_delta,
        "delta_nu": delta_nu,
        "cond_J_nu": safe_cond(J_nu),
        "cond_J_post": safe_cond(J_post),
    }


def load_seed_records(seed: int) -> list[dict]:
    path = TRACE_ROOT / TRACE_PATTERN.format(seed=seed)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def fmt_sci(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return r"\(\infty\)"
    if x == 0.0:
        return r"\(0\)"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10.0**exp)
    return rf"\({mant:.{digits}f}{'{'}\times{'}'}10^{{{exp}}}\)"


def fmt_fixed(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return r"\(\infty\)"
    return rf"\({x:.{digits}f}\)"


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def make_table(summary_rows: list[dict]) -> str:
    metrics = {
        "rho": mean_std([r["rho_final"] for r in summary_rows]),
        "s_rot": mean_std([r["s_rot_final"] for r in summary_rows]),
        "delta_J_rel_F": mean_std([r["delta_J_rel_F_final"] for r in summary_rows]),
        "eta_J_2": mean_std([r["eta_J_2_final"] for r in summary_rows]),
        "eta_J_2_max": mean_std([r["eta_J_2_max"] for r in summary_rows]),
        "cond_J_nu": mean_std([r["cond_J_nu_final"] for r in summary_rows]),
        "cond_J_post": mean_std([r["cond_J_post_final"] for r in summary_rows]),
    }

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Frozen-chart drift diagnostic for the sparse-baseline setting.}",
        r"\label{tab:supp_chart_drift_diagnostic}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{c c c c c c c c c}",
        r"\toprule",
        (
            r"Seed & Final state & \(\rho\) & \(s_{\rm rot}\) & "
            r"\(\delta_J\) & \(\eta_J\) & \(\max_t\eta_J\) & "
            r"\(\kappa(J_\nu)\) & \(\kappa(J_{\nu,\rm post})\) \\"
        ),
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(
            " & ".join(
                [
                    str(row["seed"]),
                    str(row["state_id_final"]),
                    fmt_fixed(row["rho_final"], 3),
                    fmt_sci(row["s_rot_final"], 2),
                    fmt_fixed(row["delta_J_rel_F_final"], 3),
                    fmt_fixed(row["eta_J_2_final"], 3),
                    fmt_fixed(row["eta_J_2_max"], 3),
                    fmt_fixed(row["cond_J_nu_final"], 3),
                    fmt_fixed(row["cond_J_post_final"], 3),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"Mean \(\pm\) std & -- & "
                + rf"\({metrics['rho'][0]:.3f}\pm{metrics['rho'][1]:.3f}\) & "
                + rf"\({metrics['s_rot'][0]:.3f}\pm{metrics['s_rot'][1]:.3f}\) & "
                + rf"\({metrics['delta_J_rel_F'][0]:.3f}\pm{metrics['delta_J_rel_F'][1]:.3f}\) & "
                + rf"\({metrics['eta_J_2'][0]:.3f}\pm{metrics['eta_J_2'][1]:.3f}\) & "
                + rf"\({metrics['eta_J_2_max'][0]:.3f}\pm{metrics['eta_J_2_max'][1]:.3f}\) & "
                + rf"\({metrics['cond_J_nu'][0]:.3f}\pm{metrics['cond_J_nu'][1]:.3f}\) & "
                + rf"\({metrics['cond_J_post'][0]:.3f}\pm{metrics['cond_J_post'][1]:.3f}\) \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5ex}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize",
            (
                r"\emph{Note:} The diagnostic uses the saved Algorithm~2 posterior "
                r"traces for the sparse-baseline setting with \(K_{\rm eff}=10\), "
                r"\((\beta,c_\kappa,c_\Lambda)=(1,1,1)\), and seeds "
                r"\(42,43,44\). Here "
                r"\(\delta_J:=\|J_{\nu,\rm post}-J_\nu\|_F/\|J_\nu\|_F\) "
                r"and "
                r"\(\eta_J:=\|J_\nu^{-1}(J_{\nu,\rm post}-J_\nu)\|_2\). "
                r"Values of \(\eta_J\) above one indicate that the sufficient "
                r"small-drift condition used to compare the self-consistent and "
                r"frozen-chart coupling reconstructions is not satisfied along "
                r"these traces; the frozen chart is therefore used as the stable "
                r"computational rule rather than forcing an in-pass chart refresh."
            ),
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for seed in SEEDS:
        records = load_seed_records(seed)
        F_prior = KAPPA0 * np.asarray(records[0]["nominal_R"], dtype=float)
        rows = [metric_row(seed, rec, F_prior) for rec in records]
        detail_rows.extend(rows)

        final = rows[-1]
        summary_rows.append(
            {
                "seed": int(seed),
                "state_id_final": final["state_id"],
                "rho_final": final["rho"],
                "s_rot_final": final["s_rot"],
                "delta_J_rel_F_final": final["delta_J_rel_F"],
                "eta_J_2_final": final["eta_J_2"],
                "eta_J_2_max": max(r["eta_J_2"] for r in rows),
                "cond_J_nu_final": final["cond_J_nu"],
                "cond_J_post_final": final["cond_J_post"],
                "delta_nu_final": final["delta_nu"],
                "n_trace_states": len(rows),
            }
        )

    out_dir = PAPER_ROOT / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / "chart_drift_diagnostic_records.csv"
    with detail_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_csv = out_dir / "chart_drift_diagnostic_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = out_dir / "chart_drift_diagnostic_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2))

    table_path = out_dir / "chart_drift_diagnostic_table.tex"
    table_path.write_text(make_table(summary_rows))

    print(f"Wrote {detail_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

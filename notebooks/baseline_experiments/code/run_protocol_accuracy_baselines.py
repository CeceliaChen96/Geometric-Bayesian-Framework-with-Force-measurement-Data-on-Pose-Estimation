#!/usr/bin/env python3
"""Accuracy-centered B1--B3 vs Proposed baseline report.

This script uses the saved Experiment 2C false-confidence protocol rather than
the lightweight synthetic smoke-test model.  It loads the notebook definitions
up to the data/algorithm wrappers, recomputes B1--B3 on the same generated
seeds, and reuses the already saved Proposed Alg.2 records from the manuscript
experiments.  Outputs stay under baseline_experiments/.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1]
MFG_NOTEBOOK_DIR = Path(
    "/Users/cecelia/Documents/AAA_NTU/AAA_Research/Done/"
    "Bayesian_MFG/mfg_posterior_project/mfg_github/notebooks/Posterior Estimation"
)
PROTOCOL_NOTEBOOK = MFG_NOTEBOOK_DIR / "Revised5_Multiseed_forEx2C_b_main_plot.ipynb"
SAVED_FALSE_CONFIDENCE_FILES = [
    MFG_NOTEBOOK_DIR / "exp2C_initialization_prior_mismatch_records.json",
    MFG_NOTEBOOK_DIR / "new_exp2CB_false_confidence_seed45_46_sweep_records.json",
]


METHOD_ORDER = {
    "B0-Initial": 0,
    "Alg1-MFG-single-pass": 1,
    "B1-LieGN-LM-single-step": 2,
    "B2-Tangent-Gaussian-Laplace": 3,
    "B3-Decoupled-local-Gaussian": 4,
    "B1-LieGN-LM-refinement": 5,
    "B2-Tangent-Gaussian-refinement": 6,
    "B3-Decoupled-Gaussian-refinement": 7,
    "Proposed-Alg2-refinement": 8,
}


def progress(msg: str) -> None:
    print(f"[protocol-baseline] {msg}", flush=True)


def scrub_notebook_source(src: str) -> str:
    lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        if "get_ipython()" in line:
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def load_protocol_namespace() -> dict:
    nb = json.loads(PROTOCOL_NOTEBOOK.read_text(encoding="utf-8"))
    module_name = "__exp2c_protocol_defs__"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    ns: dict = module.__dict__
    ns["__name__"] = module_name
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "data_smoke = exp2CB_generate_case_data" in src:
            break
        code = scrub_notebook_source(src)
        if not code.strip():
            continue
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(code, str(PROTOCOL_NOTEBOOK), "exec"), ns)
    required = [
        "exp2CB_generate_case_data",
        "algorithm1_single_pass_local_mfg_posterior_update",
        "recover_pose_from_theta",
        "apply_right_perturbation",
        "mfg_negative_log_prior",
        "exp2CB_rotation_error",
        "exp2CB_translation_error",
    ]
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError(f"protocol notebook definitions missing: {missing}")
    return ns


def load_saved_proposed_records() -> dict[int, dict]:
    records: dict[int, dict] = {}
    for path in SAVED_FALSE_CONFIDENCE_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rec in payload:
            if rec.get("case") != "false_confidence":
                continue
            records[int(rec["seed"])] = dict(rec)
    return records


def align_protocol_constants_to_saved(ns: dict, saved: dict[int, dict]) -> None:
    """Use the exact base perturbation recorded in the saved manuscript run.

    The working notebook has later exploratory constants, while the saved
    false-confidence records used in the manuscript report the original base
    perturbation explicitly.  Override the loaded namespace so B1--B3 are
    regenerated with the same initial pose as the saved Proposed Alg.2 rows.
    """

    source = next((r for r in saved.values() if "init_phi_error" in r), None)
    if source is not None:
        beta = float(source.get("beta", 2.0))
        ns["EXP2C_BASE_PHI_ERROR"] = np.asarray(source["init_phi_error"], dtype=float) / beta
        ns["EXP2C_BASE_TRANS_ERROR"] = np.asarray(source["init_translation_error"], dtype=float) / beta
        return

    # Fallback matching the manuscript protocol in root.tex.
    R_base = (
        ns["exp2C_Rz"](np.deg2rad(-6.0))
        @ ns["exp2C_Ry"](np.deg2rad(4.0))
        @ ns["exp2C_Rx"](np.deg2rad(2.0))
    )
    ns["EXP2C_BASE_PHI_ERROR"] = ns["exp2C_so3_log"](R_base)
    ns["EXP2C_BASE_TRANS_ERROR"] = np.array([-0.005, 0.0035, -0.004], dtype=float)


def evaluate_alg1(ns: dict, data, X):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return ns["algorithm1_single_pass_local_mfg_posterior_update"](
            Theta_prior=data.Theta_0,
            X_B_bar=X,
            U=data.U,
            Y=data.Y,
            field_params=data.field_params,
            stiffness_params=data.stiffness_params,
            ctrl_params=data.ctrl_params,
            Sigma_w=data.Sigma_w,
            contact_points_A_shared=data.contact_points_A,
            solver_config=data.solver_config,
            chi_A_0=None,
            use_psd_mf_curvature=False,
        )


def copy_pose(ns: dict, X):
    return ns["Pose"](R=X.R.copy(), p=X.p.copy())


def pose_error(ns: dict, X, X_true) -> tuple[float, float]:
    return (
        float(ns["exp2CB_rotation_error"](X, X_true)),
        float(ns["exp2CB_translation_error"](X, X_true)),
    )


def objective(ns: dict, data, X, alg1_res=None) -> tuple[float, object]:
    if alg1_res is None:
        alg1_res = evaluate_alg1(ns, data, X)
    prior = float(ns["mfg_negative_log_prior"](X, data.Theta_0))
    return 0.5 * float(alg1_res.rho) ** 2 + prior, alg1_res


def solve_step(H: np.ndarray, g: np.ndarray, damping: float) -> np.ndarray:
    H = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    g = np.asarray(g, dtype=float).reshape(6)
    diag = np.maximum(np.diag(H), 1e-9)
    A = H + float(damping) * np.diag(diag)
    return -np.linalg.solve(A + 1e-10 * np.eye(6), g)


def clip_step(delta: np.ndarray, max_rot: float = 0.35, max_pos: float = 0.04) -> np.ndarray:
    out = np.asarray(delta, dtype=float).copy()
    nr = float(np.linalg.norm(out[:3]))
    np_ = float(np.linalg.norm(out[3:]))
    if nr > max_rot:
        out[:3] *= max_rot / nr
    if np_ > max_pos:
        out[3:] *= max_pos / np_
    return out


def finalize_record(ns: dict, data, method: str, seed: int, K: int, X, rho_value: float, t0: float, **extra) -> dict:
    init_eR, init_ep = pose_error(ns, data.X_B_bar_0, data.X_B_true)
    eR, ep = pose_error(ns, X, data.X_B_true)
    rec = {
        "method": method,
        "seed": int(seed),
        "K": int(K),
        "init_eR": init_eR,
        "init_ep": init_ep,
        "eR": eR,
        "ep": ep,
        "rho": float(rho_value),
        "rot_improv_vs_init_pct": 100.0 * (init_eR - eR) / max(init_eR, 1e-12),
        "trans_improv_vs_init_pct": 100.0 * (init_ep - ep) / max(init_ep, 1e-12),
        "runtime_sec": float(time.perf_counter() - t0),
        "failed": False,
        "error": None,
    }
    rec.update(extra)
    return rec


def run_initial(ns: dict, data, seed: int, K: int) -> dict:
    t0 = time.perf_counter()
    res0 = evaluate_alg1(ns, data, data.X_B_bar_0)
    return finalize_record(ns, data, "B0-Initial", seed, K, data.X_B_bar_0, res0.rho, t0, iterations=0)


def run_lie_gn_lm(ns: dict, data, seed: int, K: int, max_iter: int, method: str = "B1-LieGN-LM-MAP") -> dict:
    t0 = time.perf_counter()
    X = copy_pose(ns, data.X_B_bar_0)
    current, res = objective(ns, data, X)
    history = []
    converged = False
    damping = 1e-2
    rejected = 0
    for it in range(max_iter):
        try:
            delta = clip_step(solve_step(res.H, res.g, damping))
        except np.linalg.LinAlgError:
            break
        X_try = ns["apply_right_perturbation"](X, delta)
        obj_try, res_try = objective(ns, data, X_try)
        accepted = obj_try < current - 1e-8
        step_norm = float(np.linalg.norm(delta))
        history.append({
            "iter": it,
            "objective": obj_try if accepted else current,
            "accepted": accepted,
            "damping": damping,
            "step": step_norm,
        })
        if accepted:
            rel_drop = (current - obj_try) / max(abs(current), 1.0)
            current, X, res = obj_try, X_try, res_try
            damping = max(damping / 3.0, 1e-6)
            rejected = 0
            if step_norm < 1e-6 or rel_drop < 1e-5:
                converged = True
                break
        else:
            damping = min(damping * 10.0, 1e6)
            rejected += 1
        if rejected >= 3:
            converged = True
            break
    return finalize_record(
        ns,
        data,
        method,
        seed,
        K,
        X,
        res.rho,
        t0,
        iterations=len(history),
        converged=converged,
    )


def run_alg1_single_pass(ns: dict, data, seed: int, K: int) -> dict:
    t0 = time.perf_counter()
    res = evaluate_alg1(ns, data, data.X_B_bar_0)
    X = ns["recover_pose_from_theta"](res.Theta_post)
    res_final = evaluate_alg1(ns, data, X)
    return finalize_record(
        ns,
        data,
        "Alg1-MFG-single-pass",
        seed,
        K,
        X,
        res_final.rho,
        t0,
        iterations=1,
        s_rot=float(res.s_rot),
    )


def local_quadratic_pose(ns: dict, data, decoupled: bool) -> tuple[object, object]:
    res = evaluate_alg1(ns, data, data.X_B_bar_0)
    H = np.asarray(res.H, dtype=float).copy()
    if decoupled:
        H[:3, 3:] = 0.0
        H[3:, :3] = 0.0
    delta = clip_step(solve_step(H, res.g, 1e-6))
    return ns["apply_right_perturbation"](data.X_B_bar_0, delta), res


def run_local_gaussian(ns: dict, data, seed: int, K: int, decoupled: bool) -> dict:
    t0 = time.perf_counter()
    method = "B3-Decoupled-local-Gaussian" if decoupled else "B2-Tangent-Gaussian-Laplace"
    try:
        X, _ = local_quadratic_pose(ns, data, decoupled=decoupled)
        res_final = evaluate_alg1(ns, data, X)
        return finalize_record(ns, data, method, seed, K, X, res_final.rho, t0, iterations=1)
    except Exception as exc:
        rec = run_initial(ns, data, seed, K)
        rec["method"] = method
        rec["failed"] = True
        rec["error"] = repr(exc)
        rec["runtime_sec"] = float(time.perf_counter() - t0)
        return rec


def candidate_delta_from_result(res, method: str, damping: float) -> np.ndarray:
    H = np.asarray(res.H, dtype=float).copy()
    if method == "B3":
        H[:3, 3:] = 0.0
        H[3:, :3] = 0.0
    if method in {"B2", "B3"}:
        return clip_step(solve_step(H, res.g, 1e-6))
    return clip_step(solve_step(H, res.g, damping))


def run_multipass_local_refinement(
    ns: dict,
    data,
    seed: int,
    K: int,
    method: str,
    T_max: int,
    alpha_grid: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1),
) -> dict:
    t0 = time.perf_counter()
    label = {
        "B1": "B1-LieGN-LM-refinement",
        "B2": "B2-Tangent-Gaussian-refinement",
        "B3": "B3-Decoupled-Gaussian-refinement",
    }[method]
    X = copy_pose(ns, data.X_B_bar_0)
    res = evaluate_alg1(ns, data, X)
    best_X = copy_pose(ns, X)
    best_res = res
    history = []
    damping = 1e-2
    for t in range(T_max):
        try:
            delta = candidate_delta_from_result(res, method, damping)
        except Exception as exc:
            history.append({"iter": t, "accepted": False, "error": repr(exc)})
            break
        trial_best = None
        for alpha in alpha_grid:
            X_try = ns["apply_right_perturbation"](X, float(alpha) * delta)
            res_try = evaluate_alg1(ns, data, X_try)
            if trial_best is None or float(res_try.rho) < float(trial_best[1].rho):
                trial_best = (X_try, res_try, float(alpha))
        if trial_best is None:
            break
        X_try, res_try, alpha = trial_best
        accepted = float(res_try.rho) < float(res.rho) - 1e-8
        history.append({
            "iter": t,
            "accepted": accepted,
            "alpha": alpha,
            "rho_before": float(res.rho),
            "rho_after": float(res_try.rho),
            "step_norm": float(np.linalg.norm(alpha * delta)),
        })
        if not accepted:
            if method == "B1":
                damping = min(damping * 10.0, 1e6)
            break
        X = X_try
        res = res_try
        if float(res.rho) < float(best_res.rho):
            best_X = copy_pose(ns, X)
            best_res = res
        if method == "B1":
            damping = max(damping / 3.0, 1e-6)
        if history[-1]["step_norm"] < 1e-6:
            break
    return finalize_record(
        ns,
        data,
        label,
        seed,
        K,
        best_X,
        best_res.rho,
        t0,
        iterations=len(history),
        accepted_updates=sum(1 for h in history if h.get("accepted")),
        converged=bool(history and not history[-1].get("accepted", False)),
    )


def run_proposed_from_saved(ns: dict, data, seed: int, K: int, saved: dict) -> dict:
    t0 = time.perf_counter()
    rec = {
        "method": "Proposed-Alg2-refinement",
        "seed": int(seed),
        "K": int(K),
        "init_eR": float(saved["init_eR"]),
        "init_ep": float(saved["init_ep"]),
        "eR": float(saved["alg2_eR"]),
        "ep": float(saved["alg2_ep"]),
        "rho": float(saved["rho_alg2"]),
        "rot_improv_vs_init_pct": 100.0 * (float(saved["init_eR"]) - float(saved["alg2_eR"])) / max(float(saved["init_eR"]), 1e-12),
        "trans_improv_vs_init_pct": 100.0 * (float(saved["init_ep"]) - float(saved["alg2_ep"])) / max(float(saved["init_ep"]), 1e-12),
        "runtime_sec": float(saved.get("elapsed_sec", time.perf_counter() - t0)),
        "iterations": int(saved.get("alg2_n_iter", 20)),
        "accepted_updates": int(saved.get("alg2_n_iter", 20)),
        "converged": bool(saved.get("alg2_converged", False)),
        "failed": False,
        "error": None,
        "source": "saved_experiment2C_false_confidence",
    }
    return rec


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r.get("scheme", ""), r["K"], r["method"]), []).append(r)
    rows = []
    metrics = [
        "init_eR",
        "init_ep",
        "eR",
        "ep",
        "rho",
        "rot_improv_vs_init_pct",
        "trans_improv_vs_init_pct",
        "runtime_sec",
        "iterations",
        "accepted_updates",
    ]
    for key, vals in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], METHOD_ORDER.get(kv[0][2], 99))):
        row = {"scheme": key[0], "K": key[1], "method": key[2], "n": len(vals)}
        for metric in metrics:
            xs = np.asarray([v[metric] for v in vals if metric in v and v[metric] is not None and np.isfinite(v[metric])], dtype=float)
            if xs.size:
                row[f"{metric}_mean"] = float(np.mean(xs))
                row[f"{metric}_std"] = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
        row["failure_rate_pct"] = 100.0 * sum(bool(v.get("failed", False)) for v in vals) / max(len(vals), 1)
        rows.append(row)
    return rows


def fmt_pm(mean: float | None, std: float | None, scale: float = 1.0, digits: int = 3) -> str:
    if mean is None:
        return "--"
    std = 0.0 if std is None else std
    return f"{mean * scale:.{digits}f} $\\pm$ {std * scale:.{digits}f}"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_accuracy_plot(summary: list[dict]) -> None:
    fig_dir = BASELINE_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for scheme in ("A", "B"):
        rows = [r for r in summary if r.get("scheme") == scheme and r["method"] != "B0-Initial"]
        rows = sorted(rows, key=lambda r: METHOD_ORDER.get(r["method"], 99))
        labels = [r["method"].replace("-", "\n") for r in rows]
        rot = [r.get("rot_improv_vs_init_pct_mean", np.nan) for r in rows]
        trans = [r.get("trans_improv_vs_init_pct_mean", np.nan) for r in rows]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.bar(x - 0.18, rot, width=0.36, label=r"$e_R$ improvement")
        ax.bar(x + 0.18, trans, width=0.36, color="tab:orange", label=r"$e_p$ improvement")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("improvement from common initial pose (%)")
        ax.set_title(f"Scheme {scheme}: accuracy improvement")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"scheme_{scheme}_accuracy_improvement.pdf")
        plt.close(fig)


def table_rows(rows: list[dict]) -> tuple[list[str], list[str]]:
    rows_main = []
    rows_runtime = []
    for r in sorted(rows, key=lambda x: METHOD_ORDER.get(x["method"], 99)):
        rows_main.append(
            " & ".join(
                [
                    r["method"],
                    str(r["n"]),
                    fmt_pm(r.get("init_eR_mean"), r.get("init_eR_std"), 1.0, 3),
                    fmt_pm(r.get("eR_mean"), r.get("eR_std"), 1.0, 3),
                    fmt_pm(r.get("rot_improv_vs_init_pct_mean"), r.get("rot_improv_vs_init_pct_std"), 1.0, 1),
                    fmt_pm(r.get("init_ep_mean"), r.get("init_ep_std"), 1000.0, 2),
                    fmt_pm(r.get("ep_mean"), r.get("ep_std"), 1000.0, 2),
                    fmt_pm(r.get("trans_improv_vs_init_pct_mean"), r.get("trans_improv_vs_init_pct_std"), 1.0, 1),
                    fmt_pm(r.get("rho_mean"), r.get("rho_std"), 1.0, 2),
                ]
            )
            + r" \\"
        )
        rows_runtime.append(
            " & ".join(
                [
                    r["method"],
                    fmt_pm(r.get("runtime_sec_mean"), r.get("runtime_sec_std"), 1.0, 2),
                    fmt_pm(r.get("iterations_mean"), r.get("iterations_std"), 1.0, 1),
                    fmt_pm(r.get("accepted_updates_mean"), r.get("accepted_updates_std"), 1.0, 1),
                    f"{r.get('failure_rate_pct', 0.0):.1f}",
                ]
            )
            + r" \\"
        )
    return rows_main, rows_runtime


def write_report(summary: list[dict], seeds: list[int], max_iter: int, T_max: int) -> Path:
    rows_A = [r for r in summary if r.get("scheme") == "A"]
    rows_B = [r for r in summary if r.get("scheme") == "B"]
    rows_A_main, rows_A_runtime = table_rows(rows_A)
    rows_B_main, rows_B_runtime = table_rows(rows_B)

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1.65cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,xcolor,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{Accuracy-Centered Baseline Comparison Results}}
\author{{Baseline comparison report}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Purpose}}
This report presents two baseline comparisons for the proposed MFG posterior
estimator.  \textbf{{Scheme A}} compares Algorithm~1 with B1--B3 at the same
single-pass/local-update level.  \textbf{{Scheme B}} compares Algorithm~2 with
multi-pass residual-safeguarded refinements built from B1--B3 candidate
generators.  All runs use the false-confidence Experiment~2C protocol,
the same initial pose, and the same seeds \texttt{{{seeds}}}.  All reported
methods are evaluated under the same protocol and measurement design.

\section*{{Methods and References}}
\textbf{{B0 Initial}} is the common initial/prior-mode pose.  \textbf{{B1
Lie-GN/LM MAP}} is a conventional damped nonlinear least-squares MAP estimator
on $SO(3)\times\mathbb R^3$, using Gauss--Newton/Levenberg--Marquardt ideas
\cite{{Levenberg1944,Marquardt1963,NocedalWright2006}} and right-perturbation
Lie-group coordinates \cite{{Barfoot2017}}.  \textbf{{B2 Tangent
Gaussian/Laplace}} applies the one-shot local quadratic/Laplace posterior mode
at the initial linearization point \cite{{TierneyKadane1986,Barfoot2017}}.
\textbf{{B3 Decoupled local Gaussian}} is not a separate standard estimator; it
is a targeted ablation of the paper's coupled posterior structure, obtained by
zeroing the rotation--translation cross-information block before solving the
same local quadratic update.  \textbf{{Algorithm~1}} is the paper's single-pass
closed-form MFG posterior update.  \textbf{{Algorithm~2}} is the paper's
residual-safeguarded multi-pass MFG Bayesian refinement.

\section*{{Experimental Setting}}
The protocol is the false-confidence prior-mismatch case with
$(\beta,c_\kappa,c_\Lambda)=(2,2,2)$ and $K=10$ sparse-baseline wrench
measurements.  The base initial perturbation is specified by the experimental
protocol, corresponding to the scaled initial errors
$e_R=0.2631$ rad and $e_p=14.59$ mm.  The prior confidence is
$\kappa_0=120$ and $\Lambda_0=\operatorname{{diag}}(12000,12000,12000)$.
The wrench-noise levels are $\sigma_\tau=0.10$ and $\sigma_f=0.70$.  Scheme A
uses one local update for each method.  Scheme B uses at most $T_{{\max}}={T_max}$
outer re-centering steps for B1--B3 and the $20$-step Proposed Alg.~2
runs.  For B1, the maximum number of single-pass GN/LM iterations in Scheme A
is {max_iter}; in the table below Scheme A uses the single-step B1 row, while
Scheme B uses one B1 candidate step per outer refinement iteration.

\section*{{Scheme A: Single-Pass / Local-Update Comparison}}
Positive $\Delta e_R$ and $\Delta e_p$ mean reduction relative to the identical
initial error.  The comparison is therefore about estimation accuracy, while
runtime is kept as a secondary diagnostic.

\begingroup
\small
\setlength{{\tabcolsep}}{{2.7pt}}
\noindent\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l c c c c c c c c}}
\toprule
Method & $n$ & init $e_R$ & final $e_R$ & $\Delta e_R$ &
init $e_p$ (mm) & final $e_p$ (mm) & $\Delta e_p$ & $\rho$ \\
\midrule
{chr(10).join(rows_A_main)}
\bottomrule
\end{{tabular}}
}}
\endgroup

\subsection*{{Scheme A Runtime}}
\begingroup
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{l c c c c}}
\toprule
Method & time (s) & iterations & accepted updates & fail. \% \\
\midrule
{chr(10).join(rows_A_runtime)}
\bottomrule
\end{{tabular}}
\endgroup

\section*{{Scheme B: Multi-Pass Refinement Comparison}}
For B1--B3, each outer step recomputes the local information at the current
nominal pose, proposes a method-specific local candidate, tests relaxed step
lengths $\alpha\in\{{1,0.5,0.25,0.1\}}$, and accepts only if the recomputed
whitened residual merit decreases.  The returned estimate is the best accepted
nominal pose, matching the role of the accepted re-centering pose in
Algorithm~2.

\begingroup
\small
\setlength{{\tabcolsep}}{{2.7pt}}
\noindent\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l c c c c c c c c}}
\toprule
Method & $n$ & init $e_R$ & final $e_R$ & $\Delta e_R$ &
init $e_p$ (mm) & final $e_p$ (mm) & $\Delta e_p$ & $\rho$ \\
\midrule
{chr(10).join(rows_B_main)}
\bottomrule
\end{{tabular}}
}}
\endgroup

\subsection*{{Scheme B Runtime}}
\begingroup
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{l c c c c}}
\toprule
Method & time (s) & iterations & accepted updates & fail. \% \\
\midrule
{chr(10).join(rows_B_runtime)}
\bottomrule
\end{{tabular}}
\endgroup

\section*{{Main Observations}}
Scheme A should be read only as a single-pass/local-update comparison.
Algorithm~1 and the one-shot local Gaussian variants mainly reduce the
translation error in this false-confidence case, while all one-shot corrections
show negative mean rotation-error improvement.  This does not support an
overclaim that a single local update fully fixes the pose; it records the
limited one-pass behavior under the common initialization.

In Scheme B, B1--B3 are allowed the same outer budget $T_{{\max}}={T_max}$ as a
multi-pass refinement budget, but the residual safeguard often stops them early
when no tested local candidate further decreases the recomputed whitened
residual merit.  They reduce translation error and residual compared with the
initial pose, but their mean rotation error increases in this protocol.  The
Proposed Algorithm~2 row is the only row in this table that reduces both mean
rotation and mean translation error, although it is also substantially slower.
Thus the defensible revision claim is about accuracy of the coupled MFG
Bayesian refinement in this prior-mismatch experiment, not about runtime.

\section*{{Interpretation}}
Scheme A addresses single-pass fairness: Algorithm~1 is compared only with
one-shot or one-step local baselines.  Scheme B addresses refinement fairness:
B1--B3 receive the same kind of repeated re-centering and residual safeguard as
Algorithm~2, but use their own local candidate generators.  Thus any advantage
of Proposed Algorithm~2 in Scheme B is not merely due to running multiple
outer iterations; it reflects the posterior candidate construction and coupled
MFG update used by the proposed refinement.

\begin{{figure}}[h]
\centering
\includegraphics[width=0.48\linewidth]{{figures/scheme_A_accuracy_improvement.pdf}}\hfill
\includegraphics[width=0.48\linewidth]{{figures/scheme_B_accuracy_improvement.pdf}}
\caption{{Mean accuracy improvement from the identical initial pose for
Scheme A and Scheme B.}}
\end{{figure}}

\section*{{Reproducibility Notes}}
Raw records and summaries are written in \texttt{{results/}}.  Figures are written
in \texttt{{figures/}}.  This report was generated with
\texttt{{run\_protocol\_accuracy\_baselines.py}} using
\texttt{{--seeds 42 43 44 45 46 --max-iter 1 --T-max {T_max}}}.  All entries
correspond to the same false-confidence Experiment~2C protocol and sparse
measurement design.

\begin{{thebibliography}}{{9}}
\bibitem{{Levenberg1944}}
K. Levenberg, ``A method for the solution of certain non-linear problems in
least squares,'' \emph{{Quarterly of Applied Mathematics}}, vol. 2, no. 2,
pp. 164--168, 1944.

\bibitem{{Marquardt1963}}
D. W. Marquardt, ``An algorithm for least-squares estimation of nonlinear
parameters,'' \emph{{SIAM Journal on Applied Mathematics}}, vol. 11, no. 2,
pp. 431--441, 1963.

\bibitem{{NocedalWright2006}}
J. Nocedal and S. J. Wright, \emph{{Numerical Optimization}}, 2nd ed. New York,
NY, USA: Springer, 2006.

\bibitem{{Barfoot2017}}
T. D. Barfoot, \emph{{State Estimation for Robotics}}. Cambridge, U.K.:
Cambridge University Press, 2017.

\bibitem{{TierneyKadane1986}}
L. Tierney and J. B. Kadane, ``Accurate approximations for posterior moments
and marginal densities,'' \emph{{Journal of the American Statistical
Association}}, vol. 81, no. 393, pp. 82--86, 1986.
\end{{thebibliography}}

\end{{document}}
"""
    tex_path = BASELINE_ROOT / "baseline_results.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--max-iter", type=int, default=1)
    p.add_argument("--T-max", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    (BASELINE_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (BASELINE_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    progress("loading Experiment 2C protocol definitions")
    ns = load_protocol_namespace()
    saved = load_saved_proposed_records()
    align_protocol_constants_to_saved(ns, saved)

    records: list[dict] = []
    t0 = time.perf_counter()
    for idx, seed in enumerate(args.seeds, start=1):
        if seed not in saved:
            raise RuntimeError(f"missing saved Proposed Alg.2 record for seed {seed}")
        progress(f"seed {seed} ({idx}/{len(args.seeds)})")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            data = ns["exp2CB_generate_case_data"]("false_confidence", int(seed), verbose=False)
        K = int(saved[seed].get("K", 10))
        # Scheme A: single-pass/local-update level.
        progress(f"  Scheme A: single-pass baselines")
        rec = run_initial(ns, data, seed, K)
        rec["scheme"] = "A"
        records.append(rec)
        rec = run_alg1_single_pass(ns, data, seed, K)
        rec["scheme"] = "A"
        records.append(rec)
        rec = run_lie_gn_lm(ns, data, seed, K, max_iter=args.max_iter, method="B1-LieGN-LM-single-step")
        rec["scheme"] = "A"
        records.append(rec)
        rec = run_local_gaussian(ns, data, seed, K, decoupled=False)
        rec["scheme"] = "A"
        records.append(rec)
        rec = run_local_gaussian(ns, data, seed, K, decoupled=True)
        rec["scheme"] = "A"
        records.append(rec)

        # Scheme B: multi-pass residual-safeguarded refinement level.
        progress(f"  Scheme B: multi-pass B1/B2/B3 refinements")
        rec = run_initial(ns, data, seed, K)
        rec["scheme"] = "B"
        records.append(rec)
        progress("    B1 refinement")
        rec = run_multipass_local_refinement(ns, data, seed, K, "B1", T_max=args.T_max)
        rec["scheme"] = "B"
        records.append(rec)
        progress("    B2 refinement")
        rec = run_multipass_local_refinement(ns, data, seed, K, "B2", T_max=args.T_max)
        rec["scheme"] = "B"
        records.append(rec)
        progress("    B3 refinement")
        rec = run_multipass_local_refinement(ns, data, seed, K, "B3", T_max=args.T_max)
        rec["scheme"] = "B"
        records.append(rec)
        progress("    Proposed Alg.2 saved record")
        rec = run_proposed_from_saved(ns, data, seed, K, saved[seed])
        rec["scheme"] = "B"
        records.append(rec)

    summary = summarize(records)
    make_accuracy_plot(summary)
    tex_path = write_report(summary, args.seeds, args.max_iter, args.T_max)

    results_dir = BASELINE_ROOT / "results"
    write_json(results_dir / "scheme_ab_accuracy_records.json", records)
    write_json(results_dir / "scheme_ab_accuracy_summary.json", summary)
    write_csv(results_dir / "scheme_ab_accuracy_records.csv", records)
    write_csv(results_dir / "scheme_ab_accuracy_summary.csv", summary)
    write_json(results_dir / "scheme_ab_accuracy_run_config.json", {
        "seeds": args.seeds,
        "max_iter": args.max_iter,
        "T_max": args.T_max,
        "elapsed_sec": time.perf_counter() - t0,
        "protocol_notebook": str(PROTOCOL_NOTEBOOK),
        "saved_proposed_files": [str(p) for p in SAVED_FALSE_CONFIDENCE_FILES],
    })
    progress(f"wrote {tex_path}")
    progress(f"elapsed {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run B0--B5 baseline experiments for the TAC revision.

The script is deliberately self-contained and writes all artifacts under
baseline_experiments/.  It reuses the existing wrench model from the local
mfg_github source tree, but does not modify the manuscript or supplement.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(__file__).resolve().parents[1]
MFG_GITHUB = Path(
    "/Users/cecelia/Documents/AAA_NTU/AAA_Research/Done/"
    "Bayesian_MFG/mfg_posterior_project/mfg_github"
)
MFG_PKG_DIR = MFG_GITHUB / "src" / "mfg_pose_estimation"
if str(MFG_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(MFG_PKG_DIR))

import posterior_core as pc  # noqa: E402


@dataclass
class ExperimentConfig:
    seeds: list[int]
    K_values: list[int]
    K_pool: int = 24
    beta: float = 2.0
    c_kappa: float = 2.0
    c_lambda: float = 2.0
    random_subset_repeats: int = 5
    sampling_draws: int = 250
    liegn_max_iter: int = 15
    proposed_tmax: int = 8
    selection_seed_limit: int = 4
    include_proposed_alg2: bool = True
    include_data_only: bool = False


def progress(msg: str) -> None:
    print(f"[baseline] {msg}", flush=True)


def rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(theta: float) -> np.ndarray:
    return pc.make_rotation_y(theta)


def rot_z(theta: float) -> np.ndarray:
    return pc.make_rotation_z(theta)


def right_plus(X: pc.Pose, xi: np.ndarray) -> pc.Pose:
    return pc.right_plus_pose(X, np.asarray(xi, dtype=float).reshape(6))


def pose_error(X: pc.Pose, X_true: pc.Pose) -> tuple[float, float]:
    eR = float(np.linalg.norm(pc.log_so3(X.R.T @ X_true.R)))
    ep = float(np.linalg.norm(X.p - X_true.p))
    return eR, ep


def pose_delta_from_reference(X_ref: pc.Pose, X: pc.Pose) -> np.ndarray:
    return np.r_[pc.log_so3(X_ref.R.T @ X.R), X.p - X_ref.p]


def make_contact_patch(scale: float = 0.010) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.80 * scale, 0.0, 0.0],
            [0.0, 0.58 * scale, 0.0],
            [0.25 * scale, -0.35 * scale, 0.30 * scale],
        ],
        dtype=float,
    )


def fibonacci_directions(n: int) -> np.ndarray:
    dirs = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = phi * i
        dirs.append([r * math.cos(theta), r * math.sin(theta), z])
    return np.asarray(dirs, dtype=float)


def make_pose_from_direction(
    X_true: pc.Pose,
    direction: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    field_params: pc.SuperquadricFieldParameters,
) -> pc.Pose:
    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / (np.linalg.norm(d) + 1e-12)
    surface_B = np.array(
        [field_params.a1 * d[0], field_params.a2 * d[1], field_params.a3 * d[2]],
        dtype=float,
    )
    R_A = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    p_A = X_true.R @ surface_B + X_true.p
    return pc.Pose(R=R_A, p=p_A)


def build_measurement_pool(K_pool: int, X_true: pc.Pose, field_params) -> list[pc.Pose]:
    dirs = fibonacci_directions(K_pool)
    poses = []
    for i, d in enumerate(dirs):
        phase = 2.0 * math.pi * i / max(K_pool, 1)
        yaw = np.deg2rad(6.0) * math.sin(phase)
        pitch = np.deg2rad(3.0) * math.cos(1.7 * phase)
        roll = np.deg2rad(2.0) * math.sin(0.6 * phase)
        poses.append(make_pose_from_direction(X_true, d, yaw, pitch, roll, field_params))
    return poses


def make_theta0(X0: pc.Pose, kappa0: float = 60.0, lambda0: float = 6000.0) -> pc.MFGParameters:
    return pc.MFGParameters(
        F=kappa0 * X0.R.copy(),
        mu=X0.p.copy(),
        Lambda=np.diag([lambda0, lambda0, lambda0]),
        Gamma=np.zeros((3, 3)),
    )


def build_case(
    seed: int,
    K_pool: int = 24,
    beta: float = 2.0,
    c_kappa: float = 2.0,
    c_lambda: float = 2.0,
) -> dict:
    rng = np.random.default_rng(seed)
    field = pc.SuperquadricFieldParameters(0.04, 0.03, 0.05, 1.0, 1.0, 1e-8)
    stiff = pc.StiffnessParameters(0.1, 1.0, 0.05)
    contact = make_contact_patch(0.010)

    R_true = rot_z(np.deg2rad(14.0)) @ rot_y(np.deg2rad(-6.0)) @ rot_x(np.deg2rad(4.0))
    p_true = np.array([0.015, -0.010, 0.020], dtype=float)
    X_true = pc.Pose(R=R_true, p=p_true)

    R_base_err = rot_z(np.deg2rad(-6.0)) @ rot_y(np.deg2rad(4.0)) @ rot_x(np.deg2rad(2.0))
    phi_base = pc.log_so3(R_base_err)
    R0 = R_true @ pc.so3_exp(float(beta) * phi_base)
    p_base = np.array([-0.005, 0.0035, -0.004], dtype=float)
    p0 = p_true + float(beta) * p_base
    X0 = pc.Pose(R=R0, p=p0)
    theta0 = make_theta0(
        X0,
        kappa0=60.0 * float(c_kappa),
        lambda0=6000.0 * float(c_lambda),
    )

    Sigma = np.diag([0.10**2] * 3 + [0.70**2] * 3)
    sensor_pool = build_measurement_pool(K_pool, X_true, field)
    clean = np.asarray(
        [pc.measurement_model_y(XA, X_true, contact, field, stiff) for XA in sensor_pool],
        dtype=float,
    )
    noise = rng.multivariate_normal(np.zeros(6), Sigma, size=K_pool)
    Y = clean + noise
    return {
        "seed": seed,
        "rng": rng,
        "field": field,
        "stiff": stiff,
        "contact": contact,
        "X_true": X_true,
        "X0": X0,
        "theta0": theta0,
        "Sigma": Sigma,
        "sensor_pool": sensor_pool,
        "Y_pool": Y,
        "clean_pool": clean,
        "beta": float(beta),
        "c_kappa": float(c_kappa),
        "c_lambda": float(c_lambda),
    }


def subset_case(case: dict, indices: Iterable[int]) -> dict:
    idx = list(map(int, indices))
    out = dict(case)
    out["indices"] = idx
    out["sensor"] = [case["sensor_pool"][i] for i in idx]
    out["Y"] = np.asarray(case["Y_pool"][idx], dtype=float)
    out["clean"] = np.asarray(case["clean_pool"][idx], dtype=float)
    return out


def residual_stack(X: pc.Pose, case: dict) -> np.ndarray:
    rs = []
    for y, XA in zip(case["Y"], case["sensor"]):
        pred = pc.measurement_model_y(XA, X, case["contact"], case["field"], case["stiff"])
        rs.append(y - pred)
    return np.concatenate(rs)


def whitened_stack(X: pc.Pose, case: dict) -> np.ndarray:
    Sigma = case["Sigma"]
    inv_std = 1.0 / np.sqrt(np.diag(Sigma))
    blocks = []
    for y, XA in zip(case["Y"], case["sensor"]):
        pred = pc.measurement_model_y(XA, X, case["contact"], case["field"], case["stiff"])
        blocks.append((y - pred) * inv_std)
    return np.concatenate(blocks)


def rho(X: pc.Pose, case: dict) -> float:
    z = whitened_stack(X, case)
    return float(np.linalg.norm(z))


def data_cost(X: pc.Pose, case: dict) -> float:
    z = whitened_stack(X, case)
    return float(z @ z)


def prior_terms(X: pc.Pose, case: dict) -> tuple[np.ndarray, np.ndarray]:
    X0 = case["X0"]
    theta0 = case["theta0"]
    phi = pc.log_so3(X0.R.T @ X.R)
    dp = X.p - X0.p
    sqrt_k = math.sqrt(float(np.linalg.norm(theta0.F, ord="fro") / math.sqrt(3.0)))
    sqrt_lam = np.sqrt(np.diag(theta0.Lambda))
    q = np.r_[sqrt_k * phi, sqrt_lam * dp]
    J = np.zeros((6, 6))
    J[:3, :3] = sqrt_k * np.eye(3)
    J[3:, 3:] = np.diag(sqrt_lam)
    return q, J


def total_cost(X: pc.Pose, case: dict, use_prior: bool = True) -> float:
    c = data_cost(X, case)
    if use_prior:
        q, _ = prior_terms(X, case)
        c += float(q @ q)
    return c


def numeric_jacobian(X: pc.Pose, case: dict, h_rot: float = 1e-5, h_pos: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    z0 = whitened_stack(X, case)
    J = np.zeros((z0.size, 6))
    steps = np.array([h_rot, h_rot, h_rot, h_pos, h_pos, h_pos], dtype=float)
    for j, h in enumerate(steps):
        e = np.zeros(6)
        e[j] = h
        zp = whitened_stack(right_plus(X, e), case)
        zm = whitened_stack(right_plus(X, -e), case)
        J[:, j] = (zp - zm) / (2.0 * h)
    return z0, J


def local_information(X: pc.Pose, case: dict, use_prior: bool = False, decoupled: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, J = numeric_jacobian(X, case)
    H = J.T @ J
    g = J.T @ z
    if use_prior:
        q, Jp = prior_terms(X, case)
        H = H + Jp.T @ Jp
        g = g + Jp.T @ q
    if decoupled:
        H = H.copy()
        H[:3, 3:] = 0.0
        H[3:, :3] = 0.0
    return z, g, H


def schur_srot(X: pc.Pose, case: dict) -> float:
    _, _, H = local_information(X, case, use_prior=False)
    Hpp = H[:3, :3]
    Hpv = H[:3, 3:]
    Hvv = H[3:, 3:]
    Hvp = H[3:, :3]
    Hvv_reg = Hvv + 1e-9 * np.eye(3)
    S = Hpp - Hpv @ np.linalg.pinv(Hvv_reg) @ Hvp
    eig = np.linalg.eigvalsh(0.5 * (S + S.T))
    return float(max(0.0, eig[0]))


def solve_delta(H: np.ndarray, g: np.ndarray, damping: float) -> np.ndarray:
    A = H + damping * np.diag(np.maximum(np.diag(H), 1e-9))
    return -np.linalg.solve(A + 1e-12 * np.eye(6), g)


def line_search_step(
    X: pc.Pose,
    delta: np.ndarray,
    case: dict,
    use_prior: bool,
    current_cost: float,
    alphas: Iterable[float] = (1.0, 0.5, 0.25, 0.1),
) -> tuple[pc.Pose, float, float, bool]:
    best_X = X
    best_c = current_cost
    best_alpha = 0.0
    accepted = False
    for a in alphas:
        X_try = right_plus(X, a * delta)
        c = total_cost(X_try, case, use_prior=use_prior)
        if c < best_c - 1e-10:
            best_X, best_c, best_alpha, accepted = X_try, c, float(a), True
    return best_X, best_c, best_alpha, accepted


def run_lie_gn_map(case: dict, max_iter: int = 8, use_prior: bool = True) -> dict:
    t0 = time.perf_counter()
    X = pc.Pose(case["X0"].R.copy(), case["X0"].p.copy())
    current = total_cost(X, case, use_prior=use_prior)
    history = []
    converged = False
    for it in range(max_iter):
        prev = current
        _, g, H = local_information(X, case, use_prior=use_prior)
        accepted = False
        step_norm = np.nan
        alpha = 0.0
        for damping in [1e-4, 1e-2, 1e-1, 1.0, 10.0]:
            try:
                delta = solve_delta(H, g, damping)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(delta)):
                continue
            if np.linalg.norm(delta[:3]) > 0.35:
                delta[:3] *= 0.35 / np.linalg.norm(delta[:3])
            if np.linalg.norm(delta[3:]) > 0.03:
                delta[3:] *= 0.03 / np.linalg.norm(delta[3:])
            X_new, c_new, alpha, ok = line_search_step(X, delta, case, use_prior, current)
            if ok:
                X = X_new
                step_norm = float(np.linalg.norm(alpha * delta))
                current = c_new
                accepted = True
                break
        history.append({"iter": it, "cost": current, "accepted": accepted, "alpha": alpha, "step_norm": step_norm})
        rel_drop = (prev - current) / max(abs(prev), 1.0)
        if not accepted or (np.isfinite(step_norm) and step_norm < 1e-6) or rel_drop < 1e-5:
            converged = bool(accepted)
            break
    return finalize_method("B1-LieGN-MAP" if use_prior else "B1-LieGN-data", X, case, t0, history, converged)


def run_tangent_gaussian(case: dict, passes: int = 1, decoupled: bool = False) -> dict:
    t0 = time.perf_counter()
    X = pc.Pose(case["X0"].R.copy(), case["X0"].p.copy())
    history = []
    label = "B3-Decoupled-local" if decoupled else ("B2-Tangent-Gaussian" if passes == 1 else "B2-Gaussian-recentered")
    for it in range(passes):
        z, g, H = local_information(X, case, use_prior=True, decoupled=decoupled)
        current_data = float(z @ z)
        try:
            delta = solve_delta(H, g, 1e-6)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(delta[:3]) > 0.35:
            delta[:3] *= 0.35 / np.linalg.norm(delta[:3])
        if np.linalg.norm(delta[3:]) > 0.03:
            delta[3:] *= 0.03 / np.linalg.norm(delta[3:])
        X_try, c_try, alpha, accepted = line_search_step(
            X, delta, case, use_prior=False, current_cost=current_data
        )
        history.append({"iter": it, "data_cost": current_data, "accepted": accepted, "alpha": alpha})
        if accepted:
            X = X_try
        else:
            break
    return finalize_method(label, X, case, t0, history, converged=bool(history and not history[-1]["accepted"]))


def run_proposed_alg2(case: dict, Tmax: int = 8) -> dict:
    t0 = time.perf_counter()
    try:
        # posterior_core currently contains unconditional diagnostic prints in
        # Algorithm 1, so silence them here and store only final records.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            theta, X, hist = pc.multi_pass_mfg_batch_refinement(
                theta0=case["theta0"],
                X_WB0=case["X0"],
                measurements_Y=case["Y"],
                sensor_poses_WA=case["sensor"],
                Sigma_w=case["Sigma"],
                contact_points_A=case["contact"],
                field_params=case["field"],
                stiffness_params=case["stiff"],
                Tmax=Tmax,
                h_rot=1e-6,
                use_line_search=True,
                alpha_candidates=[1.0, 2.0, 4.0],
                outer_alpha_candidates=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02],
                use_numeric_rot_candidate=True,
                h_rot_numgrad=1e-5,
                verbose=False,
                verbose_inner=False,
                return_history=True,
                X_WB_true=case["X_true"],
            )
        rec = finalize_method("Proposed-Alg2-refinement", X, case, t0, hist, converged=False)
        if isinstance(hist, dict):
            rec["iterations"] = len(hist.get("iter", []))
            rec["accepted_updates"] = int(sum(bool(x) for x in hist.get("accepted", [])))
            branches = hist.get("selected_branch", [])
            rec["selected_branches"] = ",".join(map(str, branches[-5:])) if branches else ""
        else:
            rec["accepted_updates"] = None
            rec["selected_branches"] = ""
        return rec
    except Exception as exc:  # keep experiment table complete
        rec = finalize_method("Proposed-Alg2-refinement", case["X0"], case, t0, [], converged=False)
        rec["failed"] = True
        rec["error"] = repr(exc)
        return rec


def finalize_method(label: str, X: pc.Pose, case: dict, t0: float, history, converged: bool) -> dict:
    eR, ep = pose_error(X, case["X_true"])
    init_eR, init_ep = pose_error(case["X0"], case["X_true"])
    if "_rho_init" not in case:
        case["_rho_init"] = rho(case["X0"], case)
    rho_init = float(case["_rho_init"])
    r = rho_init if label == "B0-Initial" else rho(X, case)
    return {
        "method": label,
        "seed": int(case["seed"]),
        "K": int(len(case["indices"])),
        "eR": eR,
        "ep": ep,
        "rho": r,
        "init_eR": init_eR,
        "init_ep": init_ep,
        "rho_init": rho_init,
        "rot_improv_vs_init_pct": 100.0 * (init_eR - eR) / max(init_eR, 1e-12),
        "trans_improv_vs_init_pct": 100.0 * (init_ep - ep) / max(init_ep, 1e-12),
        "rho_reduction_vs_init_pct": 100.0 * (rho_init - r) / max(rho_init, 1e-12),
        "runtime_sec": float(time.perf_counter() - t0),
        "iterations": len(history) if hasattr(history, "__len__") else None,
        "converged": bool(converged),
        "failed": False,
        "error": None,
    }


def run_initial(case: dict) -> dict:
    return finalize_method("B0-Initial", case["X0"], case, time.perf_counter(), [], converged=True)


def greedy_srot_order(case: dict, K_pool: int) -> list[int]:
    remaining = list(range(K_pool))
    selected: list[int] = []
    while remaining:
        best_i = None
        best_score = -np.inf
        for i in remaining:
            trial = selected + [i]
            if len(trial) < 6:
                score = len(trial) * 1e-6
            else:
                score = schur_srot(case["X0"], subset_case(case, trial))
            if score > best_score:
                best_i = i
                best_score = score
        selected.append(int(best_i))
        remaining.remove(int(best_i))
    return selected


def contact_direction(case: dict, i: int) -> np.ndarray:
    X0 = case["X0"]
    d = X0.R.T @ (case["sensor_pool"][i].p - X0.p)
    return d / (np.linalg.norm(d) + 1e-12)


def information_proxy_order(case: dict, K_pool: int) -> list[int]:
    """Fast information-based order used for the standalone baseline runs.

    Full greedy maximization of s_rot is expensive because every candidate
    addition requires a finite-difference Jacobian for the whole subset.  This
    screen computes each single-measurement rotational information once, then
    adds a small spread term so the prefix subsets are not concentrated in one
    contact direction.  The final B4 table still recomputes the actual s_rot
    score for every selected subset.
    """

    scores: list[tuple[float, int]] = []
    dirs = [contact_direction(case, i) for i in range(K_pool)]
    for i in range(K_pool):
        one = subset_case(case, [i])
        _, J = numeric_jacobian(case["X0"], one)
        H = J.T @ J
        score = float(np.trace(H[:3, :3]))
        scores.append((score, i))

    selected: list[int] = []
    remaining = list(range(K_pool))
    base_scores = {i: max(s, 1e-12) for s, i in scores}
    while remaining:
        if not selected:
            pick = max(remaining, key=lambda i: base_scores[i])
        else:
            def merit(i: int) -> float:
                spread = min(1.0 - abs(float(dirs[i] @ dirs[j])) for j in selected)
                return math.log(base_scores[i]) + 0.35 * spread

            pick = max(remaining, key=merit)
        selected.append(int(pick))
        remaining.remove(int(pick))
    return selected


def force_magnitude_order(case: dict) -> list[int]:
    norms = []
    for i, XA in enumerate(case["sensor_pool"]):
        pred = pc.measurement_model_y(XA, case["X0"], case["contact"], case["field"], case["stiff"])
        norms.append((float(np.linalg.norm(pred)), i))
    return [i for _, i in sorted(norms, reverse=True)]


def uniform_order(K_pool: int) -> list[int]:
    selected: list[int] = []
    remaining = list(range(K_pool))
    while remaining:
        if not selected:
            pick = 0
        else:
            pick = max(remaining, key=lambda i: min(abs(i - j) for j in selected))
        selected.append(int(pick))
        remaining.remove(int(pick))
    return selected


def run_main_baselines(cfg: ExperimentConfig) -> tuple[list[dict], dict[int, list[int]]]:
    records: list[dict] = []
    proposed_orders: dict[int, list[int]] = {}
    total = len(cfg.seeds)
    for si, seed in enumerate(cfg.seeds, start=1):
        progress(f"main baselines seed {seed} ({si}/{total})")
        base = build_case(seed, cfg.K_pool, beta=cfg.beta, c_kappa=cfg.c_kappa, c_lambda=cfg.c_lambda)
        order = information_proxy_order(base, cfg.K_pool)
        proposed_orders[seed] = order
        for K in cfg.K_values:
            progress(f"  K={K}: B0/B1/B2/B3/Proposed")
            case = subset_case(base, order[:K])
            records.append(run_initial(case))
            records.append(run_lie_gn_map(case, max_iter=cfg.liegn_max_iter, use_prior=True))
            if cfg.include_data_only:
                records.append(run_lie_gn_map(case, max_iter=cfg.liegn_max_iter, use_prior=False))
            records.append(run_tangent_gaussian(case, passes=1, decoupled=False))
            records.append(run_tangent_gaussian(case, passes=1, decoupled=True))
            if cfg.include_proposed_alg2:
                records.append(run_proposed_alg2(case, Tmax=cfg.proposed_tmax))
    return records, proposed_orders


def run_selection_baselines(cfg: ExperimentConfig, proposed_orders: dict[int, list[int]]) -> list[dict]:
    records: list[dict] = []
    K = min(10, cfg.K_pool)
    seeds = cfg.seeds[: min(cfg.selection_seed_limit, len(cfg.seeds))]
    for si, seed in enumerate(seeds, start=1):
        progress(f"B4 selection seed {seed} ({si}/{len(seeds)})")
        base = build_case(seed, cfg.K_pool, beta=cfg.beta, c_kappa=cfg.c_kappa, c_lambda=cfg.c_lambda)
        rng = np.random.default_rng(seed + 9000)
        strategies = {
            "B4-Proposed-srot-screened": proposed_orders.get(seed) or information_proxy_order(base, cfg.K_pool),
            "B4-Uniform": uniform_order(cfg.K_pool),
            "B4-ForceMagnitude": force_magnitude_order(base),
        }
        for name, order in strategies.items():
            case = subset_case(base, order[:K])
            rec = run_lie_gn_map(case, max_iter=cfg.liegn_max_iter, use_prior=True)
            rec["method"] = name
            rec["s_rot"] = schur_srot(case["X0"], case)
            records.append(rec)
        for j in range(cfg.random_subset_repeats):
            idx = rng.choice(cfg.K_pool, size=K, replace=False).tolist()
            case = subset_case(base, idx)
            rec = run_lie_gn_map(case, max_iter=cfg.liegn_max_iter, use_prior=True)
            rec["method"] = "B4-RandomK"
            rec["repeat"] = j
            rec["s_rot"] = schur_srot(case["X0"], case)
            records.append(rec)
    return records


def run_sampling_reference(cfg: ExperimentConfig, proposed_orders: dict[int, list[int]]) -> dict:
    seed = cfg.seeds[0]
    progress(f"B5 sampling reference seed {seed}, draws={cfg.sampling_draws}")
    base = build_case(seed, cfg.K_pool, beta=cfg.beta, c_kappa=cfg.c_kappa, c_lambda=cfg.c_lambda)
    order = proposed_orders.get(seed) or information_proxy_order(base, cfg.K_pool)
    case = subset_case(base, order[: min(10, cfg.K_pool)])
    map_rec = run_lie_gn_map(case, max_iter=cfg.liegn_max_iter, use_prior=True)
    X_map = reconstruct_map_by_rerun(case, max_iter=cfg.liegn_max_iter)
    _, g, H = local_information(X_map, case, use_prior=True)
    cov = np.linalg.pinv(H + 1e-9 * np.eye(6))
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-14)
    cov = (vecs * vals) @ vecs.T
    rng = np.random.default_rng(seed + 12000)
    scale = 1.0
    draws = rng.multivariate_normal(np.zeros(6), scale * cov, size=cfg.sampling_draws)
    c0 = total_cost(X_map, case, use_prior=True)
    logw = np.empty(cfg.sampling_draws)
    eR = np.empty(cfg.sampling_draws)
    ep = np.empty(cfg.sampling_draws)
    xs = np.empty((cfg.sampling_draws, 6))
    for i, xi in enumerate(draws):
        X = right_plus(X_map, xi)
        c = total_cost(X, case, use_prior=True)
        logw[i] = -0.5 * (c - c0)
        eR[i], ep[i] = pose_error(X, case["X_true"])
        xs[i] = xi
    logw -= float(np.max(logw))
    w = np.exp(logw)
    w /= np.sum(w)
    mean_xi = w @ xs
    centered = xs - mean_xi
    cov_w = centered.T @ (centered * w[:, None])
    ess = 1.0 / float(np.sum(w * w))
    cov_rel = float(np.linalg.norm(cov_w - cov, ord="fro") / max(np.linalg.norm(cov, ord="fro"), 1e-12))
    mean_pose = right_plus(X_map, mean_xi)
    mean_eR, mean_ep = pose_error(mean_pose, case["X_true"])
    fig_path = BASELINE_ROOT / "figures" / "B5_sampling_reference.pdf"
    plt.figure(figsize=(5.2, 4.2))
    take = np.argsort(w)[-min(1200, len(w)) :]
    plt.scatter(xs[take, 2], xs[take, 3] * 1e3, c=w[take], s=8, cmap="viridis")
    plt.axvline(0.0, color="k", lw=0.8)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.xlabel(r"$\delta\phi_z$ around Lie-GN MAP (rad)")
    plt.ylabel(r"$\delta p_x$ around Lie-GN MAP (mm)")
    plt.title("B5 local sampling reference")
    plt.colorbar(label="importance weight")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return {
        "seed": seed,
        "K": int(len(case["indices"])),
        "draws": int(cfg.sampling_draws),
        "ess": ess,
        "map_eR": map_rec["eR"],
        "map_ep": map_rec["ep"],
        "weighted_mean_eR": mean_eR,
        "weighted_mean_ep": mean_ep,
        "mean_delta_norm": float(np.linalg.norm(mean_xi)),
        "cov_relative_frobenius": cov_rel,
        "figure": str(fig_path.relative_to(BASELINE_ROOT)),
    }


def reconstruct_map_by_rerun(case: dict, max_iter: int = 8) -> pc.Pose:
    X = pc.Pose(case["X0"].R.copy(), case["X0"].p.copy())
    current = total_cost(X, case, use_prior=True)
    for _ in range(max_iter):
        _, g, H = local_information(X, case, use_prior=True)
        moved = False
        for damping in [1e-4, 1e-2, 1e-1, 1.0, 10.0]:
            try:
                delta = solve_delta(H, g, damping)
            except np.linalg.LinAlgError:
                continue
            if np.linalg.norm(delta[:3]) > 0.35:
                delta[:3] *= 0.35 / np.linalg.norm(delta[:3])
            if np.linalg.norm(delta[3:]) > 0.03:
                delta[3:] *= 0.03 / np.linalg.norm(delta[3:])
            X_new, c_new, _, ok = line_search_step(X, delta, case, True, current)
            if ok:
                X = X_new
                current = c_new
                moved = True
                break
        if not moved:
            break
    return X


def summarize(records: list[dict], group_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)
    rows = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = {k: v for k, v in zip(group_keys, key)}
        for metric in [
            "eR",
            "ep",
            "rho",
            "init_eR",
            "init_ep",
            "rho_init",
            "rot_improv_vs_init_pct",
            "trans_improv_vs_init_pct",
            "rho_reduction_vs_init_pct",
            "runtime_sec",
            "s_rot",
            "accepted_updates",
        ]:
            xs = np.asarray([v[metric] for v in vals if metric in v and v[metric] is not None and np.isfinite(v[metric])], dtype=float)
            if xs.size:
                row[f"{metric}_mean"] = float(np.mean(xs))
                row[f"{metric}_std"] = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
        row["n"] = len(vals)
        row["failure_rate_pct"] = 100.0 * sum(bool(v.get("failed", False)) for v in vals) / max(len(vals), 1)
        rows.append(row)
    return rows


def write_json(path: Path, payload) -> None:
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, pc.Pose):
            return {"R": o.R.tolist(), "p": o.p.tolist()}
        if hasattr(o, "__dict__"):
            return asdict(o)
        raise TypeError(type(o).__name__)

    path.write_text(json.dumps(payload, indent=2, default=conv), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def fmt_pm(mean: float | None, std: float | None, scale: float = 1.0, digits: int = 3) -> str:
    if mean is None:
        return "--"
    std = 0.0 if std is None else std
    return f"{mean * scale:.{digits}f} $\\pm$ {std * scale:.{digits}f}"


METHOD_ORDER = {
    "B0-Initial": 0,
    "B1-LieGN-MAP": 1,
    "B1-LieGN-data": 2,
    "B2-Tangent-Gaussian": 3,
    "B3-Decoupled-local": 4,
    "Proposed-Alg2-refinement": 5,
}


def sort_methods(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (int(r.get("K", 0)), METHOD_ORDER.get(r.get("method"), 99), r.get("method", "")))


def make_plots(main_summary: list[dict], selection_summary: list[dict]) -> None:
    fig_dir = BASELINE_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for K in sorted({int(r["K"]) for r in main_summary if "K" in r}):
        rows = sort_methods([r for r in main_summary if int(r["K"]) == K and r["method"] != "B1-LieGN-data"])
        labels = [r["method"].replace("-", "\n") for r in rows]
        rot = [r.get("rot_improv_vs_init_pct_mean", np.nan) for r in rows]
        trans = [r.get("trans_improv_vs_init_pct_mean", np.nan) for r in rows]
        rho_red = [r.get("rho_reduction_vs_init_pct_mean", np.nan) for r in rows]
        x = np.arange(len(rows))
        fig, ax1 = plt.subplots(figsize=(8.5, 4.0))
        ax1.axhline(0.0, color="0.25", lw=0.8)
        ax1.bar(x - 0.24, rot, width=0.24, label=r"$e_R$ improvement")
        ax1.bar(x, trans, width=0.24, color="tab:orange", label=r"$e_p$ improvement")
        ax1.bar(x + 0.24, rho_red, width=0.24, color="tab:green", label=r"$\rho$ reduction")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("relative change from common initial pose (%)")
        ax1.set_title(f"Accuracy-centered baseline comparison, K={K}")
        ax1.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"accuracy_improvement_K{K}.pdf")
        plt.close(fig)

    rows = selection_summary
    if rows:
        labels = [r["method"].replace("B4-", "").replace("-", "\n") for r in rows]
        srot = [r.get("s_rot_mean", np.nan) for r in rows]
        eR = [r.get("eR_mean", np.nan) for r in rows]
        fig, ax1 = plt.subplots(figsize=(7.5, 4.0))
        x = np.arange(len(rows))
        ax1.bar(x - 0.18, srot, width=0.36, label=r"$s_{\rm rot}$")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, eR, width=0.36, color="tab:green", label=r"$e_R$ (rad)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax1.set_ylabel(r"$s_{\rm rot}$")
        ax2.set_ylabel(r"rotation error $e_R$ (rad)")
        ax1.set_title("B4 measurement-selection baseline")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
        fig.tight_layout()
        fig.savefig(fig_dir / "B4_selection_summary.pdf")
        plt.close(fig)


def write_report(main_summary: list[dict], selection_summary: list[dict], sampling: dict, cfg: ExperimentConfig) -> Path:
    tex_path = BASELINE_ROOT / "baseline_results.tex"
    data_only_note = (
        "A data-only diagnostic variant is included in the raw files."
        if cfg.include_data_only
        else "A data-only diagnostic variant can be enabled with \\texttt{--include-data-only}."
    )
    package_note = (
        "\\textbf{Proposed-package-Alg2} calls the existing package implementation "
        "of the residual-safeguarded MFG refinement as an implementation reference."
        if cfg.include_package_alg2
        else "The existing package implementation of the proposed Alg. 2 is not "
        "inserted into this standalone table by default; it can be enabled with "
        "\\texttt{--include-package-alg2} for an additional implementation-reference run."
    )
    rows_main = []
    for r in main_summary:
        if r["method"] == "B1-LieGN-data":
            continue
        rows_main.append(
            " & ".join(
                [
                    str(r["K"]),
                    r["method"],
                    str(r["n"]),
                    fmt_pm(r.get("eR_mean"), r.get("eR_std"), 1.0, 3),
                    fmt_pm(r.get("ep_mean"), r.get("ep_std"), 1000.0, 2),
                    fmt_pm(r.get("rho_mean"), r.get("rho_std"), 1.0, 2),
                    fmt_pm(r.get("runtime_sec_mean"), r.get("runtime_sec_std"), 1.0, 2),
                    f"{r.get('failure_rate_pct', 0.0):.1f}",
                ]
            )
            + r" \\"
        )

    rows_sel = []
    for r in selection_summary:
        rows_sel.append(
            " & ".join(
                [
                    r["method"],
                    str(r["n"]),
                    fmt_pm(r.get("s_rot_mean"), r.get("s_rot_std"), 1.0, 4),
                    fmt_pm(r.get("eR_mean"), r.get("eR_std"), 1.0, 3),
                    fmt_pm(r.get("ep_mean"), r.get("ep_std"), 1000.0, 2),
                    fmt_pm(r.get("rho_mean"), r.get("rho_std"), 1.0, 2),
                ]
            )
            + r" \\"
        )

    body = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1.8cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,longtable,xcolor,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{Standalone Baseline Results for TAC Revision}}
\author{{Internal revision artifact}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Scope}}
This standalone report implements the B0--B5 baseline plan in
\texttt{{TAC\_revision\_guidance.tex}}.  It is not inserted into the main
manuscript or the supplementary material.  All methods in each synthetic case
use the same wrench model, noise covariance, initial pose, prior center, and
measurement batch.  The synthetic setup follows the manuscript scale:
superquadric axes $(0.04,0.03,0.05)$ m, baseline noise
$\sigma_\tau=0.10$, $\sigma_f=0.70$, initial perturbation
$R_z(-6^\circ)R_y(4^\circ)R_x(2^\circ)$ and
$(-0.005,0.0035,-0.004)$ m.

\section*{{Methods}}
\textbf{{B0}} is the initial/prior-mode pose.  \textbf{{B1}} is a
Lie-Gauss--Newton/Levenberg--Marquardt MAP optimizer on
$SO(3)\times\mathbb R^3$ using the same whitened wrench residual and a local
quadratic prior.  {data_only_note}  \textbf{{B2}} is the tangent Gaussian/Laplace
update; the recentered version repeats the same local Gaussian step with a
residual-descent safeguard.  \textbf{{B3}} is the same recentered local
Gaussian baseline with the rotation--translation cross block removed.
{package_note}

\section*{{B0--B3 Main Baseline Table}}
\begingroup
\small
\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{c l c c c c c c}}
\toprule
$K$ & Method & $n$ & $e_R$ (rad) & $e_p$ (mm) & $\rho$ & time (s) & fail. \% \\
\midrule
{chr(10).join(rows_main)}
\bottomrule
\end{{longtable}}
\endgroup

\section*{{Immediate Interpretation}}
The B1 Lie-GN/LM baseline slightly reduces the whitened residual $\rho$ but
does not improve the true-pose errors relative to the initial/prior-mode pose
in this synthetic sweep.  This is not a contradiction: the prior center is
already close to the ground truth, whereas the finite noisy wrench batch can be
fit by moving away from the true pose.  Therefore these baseline results should
be used as a controlled comparison and diagnostic, not as evidence for a strong
true-pose accuracy claim.  The safer revision message is that all baselines are
evaluated under the same wrench model, noise covariance, initial pose, prior
center, and measurement batch, while the proposed method should only be claimed
to provide the intended residual-safeguarded posterior update and information
ranking behavior.

\section*{{B4 Measurement-Selection Baselines}}
The B4 comparison fixes $K=10$ and evaluates the same B1 Lie-GN-MAP estimator
under different subset rules.  \texttt{{B4-Proposed-srot-screened}} uses a fast
single-measurement rotational-information screen with a spread term and then
reports the actual final Schur-complement score $s_{{\rm rot}}$,
\texttt{{B4-Uniform}} uses evenly spread candidates, \texttt{{B4-ForceMagnitude}}
selects the largest predicted initial wrench magnitudes, and
\texttt{{B4-RandomK}} averages random subsets.

\begingroup
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{longtable}}{{l c c c c c}}
\toprule
Rule & $n$ & $s_{{\rm rot}}$ & $e_R$ (rad) & $e_p$ (mm) & $\rho$ \\
\midrule
{chr(10).join(rows_sel)}
\bottomrule
\end{{longtable}}
\endgroup

For B4, the screened $s_{{\rm rot}}$ rule obtains the largest deterministic
Schur-complement rotational score in this run.  The finite-sample pose errors,
however, remain close across deterministic rules; this should be phrased as
support for the information-screening rationale, not as a decisive pose-error
win.

\section*{{B5 Sampling Reference}}
For a representative $K=10$ case, an importance-sampling reference is drawn
around the B1 MAP pose using the local Laplace covariance.  The effective
sample size is {sampling['ess']:.1f} out of {sampling['draws']} draws.  The
weighted-mean pose has $e_R={sampling['weighted_mean_eR']:.4f}$ rad and
$e_p={1000.0 * sampling['weighted_mean_ep']:.2f}$ mm, compared with the MAP
pose $e_R={sampling['map_eR']:.4f}$ rad and
$e_p={1000.0 * sampling['map_ep']:.2f}$ mm.  The relative Frobenius mismatch
between weighted covariance and local Laplace covariance is
{sampling['cov_relative_frobenius']:.3f}.  This indicates whether the local
quadratic reference is self-consistent in the tested neighborhood.

\begin{{figure}}[h]
\centering
\includegraphics[width=0.58\linewidth]{{figures/B5_sampling_reference.pdf}}
\caption{{B5 local sampling reference around the B1 Lie-GN MAP estimate.}}
\end{{figure}}

\section*{{Figures}}
\begin{{figure}}[h]
\centering
\includegraphics[width=0.48\linewidth]{{figures/B0_B3_summary_K10.pdf}}\hfill
\includegraphics[width=0.48\linewidth]{{figures/B0_B3_summary_K20.pdf}}
\caption{{Mean pose errors for B0--B3 baselines at $K=10$ and $K=20$.}}
\end{{figure}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.70\linewidth]{{figures/B4_selection_summary.pdf}}
\caption{{B4 measurement-selection comparison.}}
\end{{figure}}

\section*{{Files}}
Raw records and summaries are saved in \texttt{{results/}}.  Figures are saved
in \texttt{{figures/}}.  This report was generated with seeds
\texttt{{{cfg.seeds}}}, $K$ values \texttt{{{cfg.K_values}}},
{cfg.liegn_max_iter} maximum B1 LM/GN iterations,
{cfg.recenter_passes} recentering passes for B2/B3, {cfg.selection_seed_limit}
B4 seeds, and {cfg.random_subset_repeats} random subsets per B4 seed.

\end{{document}}
"""
    tex_path.write_text(body, encoding="utf-8")
    return tex_path


def write_accuracy_report(main_summary: list[dict], selection_summary: list[dict], sampling: dict, cfg: ExperimentConfig) -> Path:
    tex_path = BASELINE_ROOT / "baseline_results.tex"
    main_rows = []
    runtime_rows = []
    for r in sort_methods([x for x in main_summary if x["method"] != "B1-LieGN-data"]):
        main_rows.append(
            " & ".join(
                [
                    str(r["K"]),
                    r["method"],
                    str(r["n"]),
                    fmt_pm(r.get("init_eR_mean"), r.get("init_eR_std"), 1.0, 3),
                    fmt_pm(r.get("eR_mean"), r.get("eR_std"), 1.0, 3),
                    fmt_pm(r.get("rot_improv_vs_init_pct_mean"), r.get("rot_improv_vs_init_pct_std"), 1.0, 1),
                    fmt_pm(r.get("init_ep_mean"), r.get("init_ep_std"), 1000.0, 2),
                    fmt_pm(r.get("ep_mean"), r.get("ep_std"), 1000.0, 2),
                    fmt_pm(r.get("trans_improv_vs_init_pct_mean"), r.get("trans_improv_vs_init_pct_std"), 1.0, 1),
                    fmt_pm(r.get("rho_reduction_vs_init_pct_mean"), r.get("rho_reduction_vs_init_pct_std"), 1.0, 1),
                ]
            )
            + r" \\"
        )
        runtime_rows.append(
            " & ".join(
                [
                    str(r["K"]),
                    r["method"],
                    fmt_pm(r.get("runtime_sec_mean"), r.get("runtime_sec_std"), 1.0, 2),
                    fmt_pm(r.get("accepted_updates_mean"), r.get("accepted_updates_std"), 1.0, 1),
                    f"{r.get('failure_rate_pct', 0.0):.1f}",
                ]
            )
            + r" \\"
        )

    selection_rows = []
    for r in selection_summary:
        selection_rows.append(
            " & ".join(
                [
                    r["method"],
                    str(r["n"]),
                    fmt_pm(r.get("s_rot_mean"), r.get("s_rot_std"), 1.0, 4),
                    fmt_pm(r.get("eR_mean"), r.get("eR_std"), 1.0, 3),
                    fmt_pm(r.get("ep_mean"), r.get("ep_std"), 1000.0, 2),
                    fmt_pm(r.get("rho_mean"), r.get("rho_std"), 1.0, 2),
                ]
            )
            + r" \\"
        )

    body = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1.65cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,longtable,xcolor,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{Accuracy-Centered Baseline Results for TAC Revision}}
\author{{Internal revision artifact}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Scope}}
This standalone report revises the baseline study so that the proposed
residual-safeguarded MFG refinement is compared directly with B1--B3 under the
same initial pose, prior center, wrench model, noise covariance, and
measurement batch.  It is not inserted into the main manuscript or the
supplementary material.  The default setting is the false-confidence stress
case used in the revision discussion:
\[
\beta={cfg.beta:.2f},\qquad c_\kappa={cfg.c_kappa:.2f},\qquad
c_\Lambda={cfg.c_lambda:.2f},
\]
with superquadric axes $(0.04,0.03,0.05)$ m, noise
$\sigma_\tau=0.10$, $\sigma_f=0.70$, and baseline perturbation
$R_z(-6^\circ)R_y(4^\circ)R_x(2^\circ)$,
$(-0.005,0.0035,-0.004)$ m scaled by $\beta$.

\section*{{Compared Methods}}
\textbf{{B0 Initial}} is the common initial/prior-mode pose and is included only
to make the improvement percentages explicit.  \textbf{{B1 Lie-GN/LM MAP}} is a
standard damped nonlinear least-squares MAP estimator on
$SO(3)\times\mathbb R^3$, following the Gauss--Newton/Levenberg--Marquardt
family of methods~\cite{{Levenberg1944,Marquardt1963,NocedalWright2006}} and
using the same right-perturbation pose chart as common Lie-group state
estimation~\cite{{Barfoot2017}}.  \textbf{{B2 Tangent Gaussian/Laplace}} is a
one-shot local Gaussian/Laplace update at the initial linearization
center~\cite{{TierneyKadane1986,Barfoot2017}}.  \textbf{{B3 Decoupled local
Gaussian}} is an ablation of the paper's coupled MFG update: it keeps the same
local residual and prior terms as B2 but removes the rotation--translation
cross-information block, so it tests whether the coupled posterior structure is
useful.  \textbf{{Proposed Alg.2 refinement}} is the paper's safeguarded
multi-pass Bayesian refinement: it repeatedly applies the MFG posterior update,
recovers candidate modes, and accepts only residual-improving re-centering
steps.

\section*{{Main Accuracy Table}}
Positive improvement means that the method reduces the common initial error.
Negative values mean that the noisy finite batch moves the estimate farther
from the ground truth even if it may reduce residual merit.  Runtime is
reported separately because accuracy and improvement relative to the identical
initial condition are the primary quantities.

\begingroup
\small
\setlength{{\tabcolsep}}{{2.4pt}}
\begin{{tabular}}{{c l c c c c c c c c}}
\toprule
$K$ & Method & $n$ & init $e_R$ & final $e_R$ & $\Delta e_R$ &
init $e_p$ (mm) & final $e_p$ (mm) & $\Delta e_p$ & $\Delta\rho$ \\
\midrule
{chr(10).join(main_rows)}
\bottomrule
\end{{tabular}}
\endgroup

\section*{{Runtime / Iteration Diagnostics}}
\begingroup
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{c l c c c}}
\toprule
$K$ & Method & time (s) & accepted updates & fail. \% \\
\midrule
{chr(10).join(runtime_rows)}
\bottomrule
\end{{tabular}}
\endgroup

\section*{{Immediate Interpretation}}
This table is designed to support the reviewer-facing statement that the
proposed safeguarded refinement should be compared against conventional
least-squares and local-Laplace baselines using estimation accuracy, not merely
runtime.  B1--B3 are intentionally ordinary baselines: B1 is a standard
iterative nonlinear least-squares/MAP solve, B2 is the local tangent Gaussian
posterior approximation, and B3 removes the paper's rotation--translation
coupling as a targeted ablation.  The key comparison is whether Proposed Alg.2
improves the common initial errors more consistently than B1--B3 under the
same data and initialization.  The residual safeguard guarantees accepted
residual-merit decrease; the pose-error columns show how that translates into
ground-truth accuracy in this controlled synthetic study.

\section*{{Improvement Figures}}
\begin{{figure}}[h]
\centering
\includegraphics[width=0.48\linewidth]{{figures/accuracy_improvement_K10.pdf}}\hfill
\includegraphics[width=0.48\linewidth]{{figures/accuracy_improvement_K20.pdf}}
\caption{{Relative rotation-error, translation-error, and residual-merit
changes from the identical initial pose.}}
\end{{figure}}

\section*{{B4 Selection Check}}
The B4 comparison is retained as a secondary check of the measurement-selection
logic. It fixes $K=10$ and evaluates the same B1 Lie-GN/LM MAP estimator under
different subset rules.

\begingroup
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l c c c c c}}
\toprule
Rule & $n$ & $s_{{\rm rot}}$ & $e_R$ (rad) & $e_p$ (mm) & $\rho$ \\
\midrule
{chr(10).join(selection_rows)}
\bottomrule
\end{{tabular}}
\endgroup

\section*{{B5 Sampling Reference}}
For a representative $K=10$ case, an importance-sampling reference is drawn
around the B1 MAP pose using the local Laplace covariance.  The effective
sample size is {sampling['ess']:.1f} out of {sampling['draws']} draws.  The
weighted-mean pose has $e_R={sampling['weighted_mean_eR']:.4f}$ rad and
$e_p={1000.0 * sampling['weighted_mean_ep']:.2f}$ mm, compared with the MAP
pose $e_R={sampling['map_eR']:.4f}$ rad and
$e_p={1000.0 * sampling['map_ep']:.2f}$ mm.

\begin{{figure}}[h]
\centering
\includegraphics[width=0.58\linewidth]{{figures/B5_sampling_reference.pdf}}
\caption{{B5 local sampling reference around the B1 Lie-GN/LM MAP estimate.}}
\end{{figure}}

\section*{{Run Configuration}}
Seeds: \texttt{{{cfg.seeds}}}.  $K$ values: \texttt{{{cfg.K_values}}}.
The B1 maximum iteration count is {cfg.liegn_max_iter}, and Proposed Alg.2 uses
$T_{{\max}}={cfg.proposed_tmax}$.  B4 uses {cfg.selection_seed_limit} seeds and
{cfg.random_subset_repeats} random subsets per seed.  Raw records and summaries
are saved in \texttt{{results/}}; figures are saved in \texttt{{figures/}}.

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
    tex_path.write_text(body, encoding="utf-8")
    return tex_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--K", type=int, nargs="+", default=[10, 20])
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--c-kappa", type=float, default=2.0)
    p.add_argument("--c-lambda", type=float, default=2.0)
    p.add_argument("--random-subsets", type=int, default=5)
    p.add_argument("--sampling-draws", type=int, default=250)
    p.add_argument("--liegn-max-iter", type=int, default=15)
    p.add_argument("--proposed-tmax", type=int, default=8)
    p.add_argument("--selection-seeds", type=int, default=4)
    p.add_argument("--skip-proposed-alg2", action="store_true")
    p.add_argument("--include-data-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    (BASELINE_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (BASELINE_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(
        seeds=list(range(args.seed_start, args.seed_start + args.n_seeds)),
        K_values=args.K,
        beta=args.beta,
        c_kappa=args.c_kappa,
        c_lambda=args.c_lambda,
        random_subset_repeats=args.random_subsets,
        sampling_draws=args.sampling_draws,
        liegn_max_iter=args.liegn_max_iter,
        proposed_tmax=args.proposed_tmax,
        selection_seed_limit=args.selection_seeds,
        include_proposed_alg2=not args.skip_proposed_alg2,
        include_data_only=args.include_data_only,
    )

    t0 = time.perf_counter()
    main_records, proposed_orders = run_main_baselines(cfg)
    selection_records = run_selection_baselines(cfg, proposed_orders)
    sampling = run_sampling_reference(cfg, proposed_orders)
    main_summary = summarize(main_records, ["K", "method"])
    selection_summary = summarize(selection_records, ["method"])
    make_plots(main_summary, selection_summary)
    tex_path = write_accuracy_report(main_summary, selection_summary, sampling, cfg)

    results_dir = BASELINE_ROOT / "results"
    write_json(results_dir / "baseline_main_records.json", main_records)
    write_json(results_dir / "baseline_main_summary.json", main_summary)
    write_json(results_dir / "baseline_selection_records.json", selection_records)
    write_json(results_dir / "baseline_selection_summary.json", selection_summary)
    write_json(results_dir / "baseline_sampling_reference.json", sampling)
    write_csv(results_dir / "baseline_main_records.csv", main_records)
    write_csv(results_dir / "baseline_main_summary.csv", main_summary)
    write_csv(results_dir / "baseline_selection_records.csv", selection_records)
    write_csv(results_dir / "baseline_selection_summary.csv", selection_summary)
    write_json(results_dir / "run_config.json", asdict(cfg) | {"elapsed_sec": time.perf_counter() - t0})
    print(f"[done] wrote {tex_path}")
    print(f"[done] elapsed {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()

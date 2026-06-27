"""Notebook-extracted core functions from Sampling.ipynb.

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as SciRot
from matplotlib.patches import Ellipse


def canonicalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = canonicalize_quaternion(q)
    q0, q1, q2, q3 = q

    R = np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3,  2*(q1*q2 - q0*q3),              2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),              q0*q0 - q1*q1 + q2*q2 - q3*q3,  2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),              2*(q2*q3 + q0*q1),              q0*q0 - q1*q1 - q2*q2 + q3*q3]
    ], dtype=float)
    return R


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=float)
    return R


def geodesic_distance_so3(R1: np.ndarray, R2: np.ndarray = None) -> float:
    """
        d(R1, R2) = arccos((trace(R1^T R2)-1)/2)
    """
    if R2 is None:
        R2 = np.eye(3)
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))


def proper_svd(F: np.ndarray):
    
    U0, sigma, Vt0 = np.linalg.svd(F)
    V0 = Vt0.T


    sign_u = 1.0 if np.linalg.det(U0) > 0 else -1.0
    sign_v = 1.0 if np.linalg.det(V0) > 0 else -1.0

    Du = np.diag([1.0, 1.0, sign_u])
    Dv = np.diag([1.0, 1.0, sign_v])


    U = U0 @ Du
    V = V0 @ Dv

    
    Sigma = np.diag(sigma)
    S = Du @ Sigma @ Dv
    s = np.diag(S).copy()

    
    tol = 1e-8
    if np.linalg.det(U) < 1 - tol or np.linalg.det(V) < 1 - tol:
        raise ValueError("proper SVD failed: U or V is not numerically in SO(3).")

    
    err = np.linalg.norm(F - U @ S @ V.T, ord='fro')
    if err > 1e-8 * (1.0 + np.linalg.norm(F, ord='fro')):
        raise ValueError(f"proper SVD reconstruction failed, error = {err:.3e}")

    return U, S, V, s


def mf_diag_to_bingham_params(s: np.ndarray):

    s1, s2, s3 = s
    a = np.array([
        s1 + s2 + s3,
        s1 - s2 - s3,
        -s1 + s2 - s3,
        -s1 - s2 + s3
    ], dtype=float)

    z = a - np.max(a)
    return a, z


def build_acg_shape(z: np.ndarray, c: float = 1.0):

    z = np.asarray(z, dtype=float)
    kappa = c - z
    if np.any(kappa <= 0):
        raise ValueError("ACG shape construction failed: some kappa_j <= 0. Try a larger c.")

    K = np.diag(kappa)
    Sigma_g = np.diag(1.0 / kappa)
    return K, Sigma_g, kappa


def sample_acg(Sigma_g: np.ndarray, rng: np.random.Generator) -> np.ndarray:

    y = rng.multivariate_normal(mean=np.zeros(4), cov=Sigma_g)
    q = y / np.linalg.norm(y)
    q = canonicalize_quaternion(q)
    return q


def log_target_bingham(q: np.ndarray, z: np.ndarray) -> float:
    
    q = canonicalize_quaternion(q)
    return float(np.dot(z, q * q))


def log_proposal_acg(q: np.ndarray, K: np.ndarray) -> float:
    
    q = canonicalize_quaternion(q)
    val = float(q @ K @ q)
    return float(-2.0 * np.log(val))


def sample_mf_so3_independence_mh(
    F: np.ndarray,
    N: int = 2000,
    burnin: int = 300,
    thin: int = 1,
    c: float = 1.0,
    seed: int = 42,
):
    
    rng = np.random.default_rng(seed)

    # ---------------------------
    # Step 1: proper SVD
    # ---------------------------
    U, S, V, s = proper_svd(F)

    # ---------------------------
    # Step 2: Bingham 参数
    # ---------------------------
    a, z = mf_diag_to_bingham_params(s)

    # ---------------------------
    # Step 3: ACG proposal
    # ---------------------------
    K, Sigma_g, kappa = build_acg_shape(z, c=c)

    # ---------------------------
    # Step 4: 初始化 q^(0) ~ g(q)
    # ---------------------------
    q_current = sample_acg(Sigma_g, rng)
    log_pi_current = log_target_bingham(q_current, z)
    log_g_current = log_proposal_acg(q_current, K)

   
    chain_q = [q_current.copy()]
    chain_log_pi = [log_pi_current]
    accepted_flags = []

    # 用于保存最终样本
    stored_q = []
    stored_R0 = []
    stored_R = []
    stored_iter = []

    t = 0
    while len(stored_R) < N:
        # ---------------------------
        # Step 5: 提议 q' ~ g(q)
        # ---------------------------
        q_prop = sample_acg(Sigma_g, rng)
        log_pi_prop = log_target_bingham(q_prop, z)
        log_g_prop = log_proposal_acg(q_prop, K)

        # ---------------------------
        # Step 6: log-domain acceptance ratio   
        # Delta = log pi(q') - log pi(q^k) + log g(q^k) - log g(q')
        # 接受条件: log u <= min(0, Delta)
        # ---------------------------
        Delta = log_pi_prop - log_pi_current + log_g_current - log_g_prop
        u = rng.random()

        if np.log(u) <= min(0.0, Delta):
            q_current = q_prop
            log_pi_current = log_pi_prop
            log_g_current = log_g_prop
            accepted = 1
        else:
            accepted = 0

        t += 1
        accepted_flags.append(accepted)
        chain_q.append(q_current.copy())
        chain_log_pi.append(log_pi_current)

        # ---------------------------
        # Step 7: burn-in + thinning
        # ---------------------------
        if t > burnin and ((t - burnin) % thin == 0):
            R0 = quat_to_rot(q_current)
            R = U @ R0 @ V.T

            stored_q.append(q_current.copy())
            stored_R0.append(R0.copy())
            stored_R.append(R.copy())
            stored_iter.append(t)

    results = {
        "F": F,
        "U": U,
        "S": S,
        "V": V,
        "s": s,
        "a": a,
        "z": z,
        "K": K,
        "Sigma_g": Sigma_g,
        "kappa": kappa,
        "chain_q": np.array(chain_q),
        "chain_log_pi": np.array(chain_log_pi),
        "accepted_flags": np.array(accepted_flags),
        "stored_q": np.array(stored_q),
        "stored_R0": np.array(stored_R0),
        "stored_R": np.array(stored_R),
        "stored_iter": np.array(stored_iter),
        "acceptance_rate": float(np.mean(accepted_flags)) if len(accepted_flags) > 0 else 0.0,
        "burnin": burnin,
        "thin": thin,
        "N": N,
        "c": c,
        "seed": seed,
    }
    return results


def plot_unit_sphere(ax, alpha=0.08):
    
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, antialiased=True, alpha=alpha)


def visualize_mf_sampling(results: dict, max_scatter_points: int = 1500):
    
    chain_q = results["chain_q"]
    accepted_flags = results["accepted_flags"]
    stored_R0 = results["stored_R0"]
    burnin = results["burnin"]

    # ---------------------------
    # 图 1: 四元数 trace + acceptance
    # ---------------------------
    fig1, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    iters_chain = np.arange(chain_q.shape[0])
    axs[0].plot(iters_chain, chain_q[:, 0], label=r"$q_0$")
    axs[0].plot(iters_chain, chain_q[:, 1], label=r"$q_1$")
    axs[0].plot(iters_chain, chain_q[:, 2], label=r"$q_2$")
    axs[0].plot(iters_chain, chain_q[:, 3], label=r"$q_3$")
    axs[0].axvline(burnin, linestyle="--", linewidth=1.5, label="burn-in end")
    axs[0].set_title("Quaternion trace of the MH chain")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Quaternion components")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    running_acc = np.cumsum(accepted_flags) / np.arange(1, len(accepted_flags) + 1)
    axs[1].plot(np.arange(1, len(accepted_flags) + 1), running_acc)
    axs[1].axvline(burnin, linestyle="--", linewidth=1.5, label="burn-in end")
    axs[1].set_title("Running acceptance rate")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Acceptance rate")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # ---------------------------
    # 图 2: 标准化样本的测地角分布
    # ---------------------------
    geodesic_angles = np.array([geodesic_distance_so3(R0, np.eye(3)) for R0 in stored_R0])

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(geodesic_angles, bins=40, density=True)
    ax2.set_title(r"Histogram of geodesic distances $d(R_0, I)$")
    ax2.set_xlabel(r"Geodesic angle (radians)")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---------------------------
    # 图 3: 将 R0 e3 投到球面上
    # e3 = [0,0,1]^T，因此 R0 e3 就是 R0 的第三列
    # ---------------------------
    pts = stored_R0[:, :, 2]
    if pts.shape[0] > max_scatter_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_scatter_points).astype(int)
        pts = pts[idx]

    fig3 = plt.figure(figsize=(8, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    plot_unit_sphere(ax3, alpha=0.08)
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, alpha=0.8)

    
    # 画出参考点 e3
    ax3.scatter([0], [0], [1], s=300, marker="^", label=r"$mode direction:e_3$")
    ax3.set_title(r"Scatter of $R_0 e_3$ on the unit sphere")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend()

    # 保持球面比例一致
    ax3.set_box_aspect((1, 1, 1))
    lim = 1.1
    ax3.set_xlim([-lim, lim])
    ax3.set_ylim([-lim, lim])
    ax3.set_zlim([-lim, lim])

    plt.tight_layout()

    # ---------------------------
    # 图 4: log-target trace
    # ---------------------------
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(np.arange(len(results["chain_log_pi"])), results["chain_log_pi"])
    ax4.axvline(burnin, linestyle="--", linewidth=1.5, label="burn-in end")
    ax4.set_title(r"Trace of unnormalized log-target $\widetilde{\log \pi}(q)$")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel(r"$\widetilde{\log \pi}(q)$")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def apply_rotations_to_vector(Rs: np.ndarray, v: np.ndarray) -> np.ndarray:
    
    v = np.asarray(v, dtype=float)
    return np.einsum("nij,j->ni", Rs, v)


def rough_ess_1d(x: np.ndarray, max_lag: int = 200) -> float:
    
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return float(n)

    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-14:
        return float(n)

    # 自相关函数
    acf = []
    for lag in range(1, min(max_lag, n - 1) + 1):
        c = np.dot(x[:-lag], x[lag:]) / (n - lag)
        rho = c / var
        # 遇到非正自相关就截断
        if rho <= 0:
            break
        acf.append(rho)

    tau_int = 1.0 + 2.0 * np.sum(acf)
    ess = n / tau_int
    return float(max(1.0, min(n, ess)))


def acceptance_quality_label(acc_rate: float) -> str:
    
    if acc_rate < 0.05:
        return "poor"
    elif acc_rate < 0.15:
        return "marginal"
    elif acc_rate < 0.30:
        return "usable"
    elif acc_rate < 0.60:
        return "good"
    else:
        return "very good"


def plot_sphere_scatter(ax, pts: np.ndarray, title: str,
                        mode_point: np.ndarray = None,
                        max_points: int = 1200,
                        marker_size: int = 10):
    
    plot_unit_sphere(ax, alpha=0.08)

    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts = pts[idx]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=marker_size, alpha=0.75)

    if mode_point is not None:
        mode_point = np.asarray(mode_point, dtype=float)
    
        # 单位方向
        p = mode_point / np.linalg.norm(mode_point)
    
        # 把标记沿这个方向稍微移到球面外面，避免被散点或球面挡住
        p_out = 1.12 * p
        p_text = 1.20 * p
    
        # 画一条引导线
        ax.plot([p[0], p_out[0]],
                [p[1], p_out[1]],
                [p[2], p_out[2]],
                color="darkorange", linewidth=2)
    
        # 在球面外侧画三角形标记
        ax.scatter([p_out[0]], [p_out[1]], [p_out[2]],
                   s=320,
                   marker="^",
                   color="orange",
                   edgecolors="black",
                   linewidths=1.5,
                   depthshade=False)
    
        # 在图中直接标注文字，不放图例
        ax.text(p_text[0], p_text[1], p_text[2],
                "mode direction",
                fontsize=11,
                color="darkorange")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    lim = 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])


def visualize_single_case_v2(results: dict, max_scatter_points: int = 1200):
    chain_q = results["chain_q"]
    accepted_flags = results["accepted_flags"]
    stored_R0 = results["stored_R0"]
    stored_R = results["stored_R"]
    burnin = results["burnin"]
    U = results["U"]
    V = results["V"]
    s = results["s"]
    acc_rate = results["acceptance_rate"]

    # mode
    R_mode = U @ V.T

    # 球面上的向量散点
    e3 = np.array([0.0, 0.0, 1.0])
    pts_R0_e3 = apply_rotations_to_vector(stored_R0, e3)
    pts_R_e3 = apply_rotations_to_vector(stored_R, e3)
    mode_e3 = R_mode @ e3

    # 距离统计
    d_R0_I = np.array([geodesic_distance_so3(R0, np.eye(3)) for R0 in stored_R0])
    d_R_mode = np.array([geodesic_distance_so3(R, R_mode) for R in stored_R])

    ess_angle = rough_ess_1d(d_R_mode, max_lag=200)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    # -------------------------
    # (1) quaternion trace
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    iters_chain = np.arange(chain_q.shape[0])
    ax1.plot(iters_chain, chain_q[:, 0], label=r"$q_0$")
    ax1.plot(iters_chain, chain_q[:, 1], label=r"$q_1$")
    ax1.plot(iters_chain, chain_q[:, 2], label=r"$q_2$")
    ax1.plot(iters_chain, chain_q[:, 3], label=r"$q_3$")
    ax1.axvline(burnin, linestyle="--", linewidth=1.5, label="burn-in end")
    ax1.set_title(f"Quaternion trace\ns = ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # -------------------------
    # (2) running acceptance
    # -------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    running_acc = np.cumsum(accepted_flags) / np.arange(1, len(accepted_flags) + 1)
    ax2.plot(np.arange(1, len(accepted_flags) + 1), running_acc)
    ax2.axvline(burnin, linestyle="--", linewidth=1.5, label="burn-in end")
    ax2.set_title(f"Running acceptance rate\nfinal = {acc_rate:.3f} ({acceptance_quality_label(acc_rate)})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Acceptance rate")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # -------------------------
    # (3) d(R0, I)
    # -------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(d_R0_I, bins=35, density=True)
    ax3.set_title(r"Histogram of $d(R_0, I)$")
    ax3.set_xlabel("Geodesic angle (rad)")
    ax3.set_ylabel("Density")
    ax3.grid(True, alpha=0.3)

    # -------------------------
    # (4) R0 e3
    # -------------------------
    ax4 = fig.add_subplot(gs[1, 0], projection="3d")
    plot_sphere_scatter(ax4, pts_R0_e3,
                        title=r"Scatter of $R_0 e_3$ on $S^2$",
                        mode_point=np.array([0.0, 0.0, 1.0]),
                        max_points=max_scatter_points)

    # -------------------------
    # (5) R e3
    # -------------------------
    ax5 = fig.add_subplot(gs[1, 1], projection="3d")
    plot_sphere_scatter(ax5, pts_R_e3,
                        title=r"Scatter of recovered $R e_3$ on $S^2$",
                        mode_point=mode_e3,
                        max_points=max_scatter_points)

    # -------------------------
    # (6) d(R, R_mode)
    # -------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(d_R_mode, bins=35, density=True)
    ax6.set_title(rf"Histogram of $d(R,R_{{mode}})$" + f"\nESS ≈ {ess_angle:.1f}")
    ax6.set_xlabel("Geodesic angle (rad)")
    ax6.set_ylabel("Density")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_recovered_axes(results: dict, max_scatter_points: int = 1000):
    
    stored_R = results["stored_R"]
    U = results["U"]
    V = results["V"]

    R_mode = U @ V.T
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    pts1 = apply_rotations_to_vector(stored_R, e1)
    pts2 = apply_rotations_to_vector(stored_R, e2)
    pts3 = apply_rotations_to_vector(stored_R, e3)

    mode1 = R_mode @ e1
    mode2 = R_mode @ e2
    mode3 = R_mode @ e3

    fig = plt.figure(figsize=(15, 4.8))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    plot_sphere_scatter(ax1, pts1, title=r"Recovered $R e_1$", mode_point=mode1,
                        max_points=max_scatter_points)
    plot_sphere_scatter(ax2, pts2, title=r"Recovered $R e_2$", mode_point=mode2,
                        max_points=max_scatter_points)
    plot_sphere_scatter(ax3, pts3, title=r"Recovered $R e_3$", mode_point=mode3,
                        max_points=max_scatter_points)

    plt.tight_layout()
    plt.show()


def run_family_experiment(
    s_list=None,
    N: int = 1500,
    burnin: int = 300,
    thin: int = 1,
    c: float = 1.0,
    seed: int = 123,
    randomize_mode: bool = False,
):
    
    rng = np.random.default_rng(seed)

    if s_list is None:
        s_list = [
            [15.0, 12.0, 10.0],   # 极集中
            [12.0, 8.0, 3.0],     # 中等集中，带各向异性
            [8.0, 4.0, 1.0],      # 更分散
            [6.0, 6.0, 6.0],      # 近似各向同性集中
            [3.0, 2.0, 1.0],      # 比较分散
            [1.0, 0.5, 0.2],      # 很分散
        ]

    family_results = []

    # 如果不随机 mode，就固定一组 U,V，方便只比较 S 的变化
    if not randomize_mode:
        U0 = random_rotation(rng)
        V0 = random_rotation(rng)

    for i, svals in enumerate(s_list):
        svals = np.asarray(svals, dtype=float)

        if randomize_mode:
            U = random_rotation(rng)
            V = random_rotation(rng)
        else:
            U = U0
            V = V0

        F = U @ np.diag(svals) @ V.T

        results = sample_mf_so3_independence_mh(
            F=F,
            N=N,
            burnin=burnin,
            thin=thin,
            c=c,
            seed=seed + 100 * (i + 1),
        )

        results["label"] = f"s=({svals[0]:.1f},{svals[1]:.1f},{svals[2]:.1f})"
        family_results.append(results)

    return family_results


def plot_family_results(family_results, vector=np.array([0.0, 0.0, 1.0]), max_points: int = 900):
    
    m = len(family_results)
    fig = plt.figure(figsize=(4.3 * m, 8))
    gs = fig.add_gridspec(2, m)

    for i, res in enumerate(family_results):
        U = res["U"]
        V = res["V"]
        s = res["s"]
        acc = res["acceptance_rate"]
        R_mode = U @ V.T

        pts = apply_rotations_to_vector(res["stored_R"], vector)
        mode_pt = R_mode @ vector

        d_mode = np.array([geodesic_distance_so3(R, R_mode) for R in res["stored_R"]])
        ess = rough_ess_1d(d_mode, max_lag=200)

        # 上排：球面散点
        ax_top = fig.add_subplot(gs[0, i], projection="3d")
        plot_sphere_scatter(
            ax_top,
            pts,
            title=res.get("label", f"case {i+1}") + f"\nacc={acc:.3f}",
            mode_point=mode_pt,
            max_points=max_points,
        )

        # 下排：到 mode 的测地距离直方图
        ax_bottom = fig.add_subplot(gs[1, i])
        ax_bottom.hist(d_mode, bins=28, density=True)
        ax_bottom.set_title(
            rf"$d(R,R_{{mode}})$"
            + "\n"
            + f"ESS≈{ess:.0f}, {acceptance_quality_label(acc)}"
        )
        ax_bottom.set_xlabel("Angle (rad)")
        ax_bottom.set_ylabel("Density")
        ax_bottom.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_family_summary(family_results):
    
    print("=" * 90)
    print("Family experiment summary")
    print("=" * 90)
    for i, res in enumerate(family_results):
        U = res["U"]
        V = res["V"]
        R_mode = U @ V.T
        d_mode = np.array([geodesic_distance_so3(R, R_mode) for R in res["stored_R"]])
        ess = rough_ess_1d(d_mode, max_lag=200)

        print(f"[Case {i+1}] {res.get('label', '')}")
        print(f"  proper-SVD singular values s = {res['s']}")
        print(f"  acceptance rate            = {res['acceptance_rate']:.4f} "
              f"({acceptance_quality_label(res['acceptance_rate'])})")
        print(f"  mean d(R, R_mode)         = {d_mode.mean():.4f} rad")
        print(f"  std  d(R, R_mode)         = {d_mode.std():.4f} rad")
        print(f"  rough ESS (angle series)  = {ess:.1f}")
        print("-" * 90)


def main_family_demo():
    """
    This demo will accomplish the following:

    1) Draw a more complete single-case plot (including the restored R) for a single function F.

    2) With a fixed mode, only change the singular value s and observe the change in concentration.

    3) Randomly select different functions F and observe the distribution changes of the restored R.
    """

    

# --------------------------------------------------------
    # Part A: 单个案例
    # --------------------------------------------------------
    print("\n[Part A] Single-case visualization")
    rng = np.random.default_rng(2025)
    U = random_rotation(rng)
    V = random_rotation(rng)
    F_single = U @ np.diag([10.0, 6.0, 2.0]) @ V.T

    global res_single
    res_single = sample_mf_so3_independence_mh(
        F=F_single,
        N=2000,
        burnin=400,
        thin=1,
        c=1.0,
        seed=2025
    )

    print(f"Single-case acceptance rate = {res_single['acceptance_rate']:.4f} "
              f"({acceptance_quality_label(res_single['acceptance_rate'])})")

    visualize_single_case_v2(res_single)
    visualize_recovered_axes(res_single)

    # --------------------------------------------------------
    # Part B: 固定 mode，只改变 s
    # --------------------------------------------------------
    print("\n[Part B] Family comparison with fixed mode")
    s_list_fixed_mode = [
        [15.0, 12.0, 10.0],
        [12.0, 8.0, 3.0],
        [8.0, 4.0, 1.0],
        [6.0, 6.0, 6.0],
        [3.0, 2.0, 1.0],
        [1.0, 0.5, 0.2],
    ]
    global family_fixed
    family_fixed = run_family_experiment(
        s_list=s_list_fixed_mode,
        N=1400,
        burnin=300,
        thin=1,
        c=1.0,
        seed=101,
        randomize_mode=False,
    )

    print_family_summary(family_fixed)
    plot_family_results(family_fixed, vector=np.array([0.0, 0.0, 1.0]), max_points=900)

    # --------------------------------------------------------
    # Part C: 随机 F
    # --------------------------------------------------------
    print("\n[Part C] Family comparison with random F")
    s_list_random = [
        [10.0, 7.0, 2.0],
        [8.0, 8.0, 8.0],
        [6.0, 3.0, 1.0],
        [4.0, 2.0, 0.5],
    ]
    global family_random
    family_random = run_family_experiment(
        s_list=s_list_random,
        N=1400,
        burnin=300,
        thin=1,
        c=1.0,
        seed=202,
        randomize_mode=True,
    )

    print_family_summary(family_random)
    plot_family_results(family_random, vector=np.array([0.0, 0.0, 1.0]), max_points=900)


def inspect_structure(obj, prefix="root", max_depth=3, depth=0, visited=None):
    if visited is None:
        visited = set()

    oid = id(obj)
    if oid in visited:
        return
    visited.add(oid)

    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{prefix}: ...")
        return

    if isinstance(obj, dict):
        print(f"{indent}{prefix}: dict, keys={list(obj.keys())}")
        for k, v in obj.items():
            inspect_structure(v, prefix=f"{prefix}.{k}", max_depth=max_depth, depth=depth+1, visited=visited)

    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{prefix}: {type(obj).__name__}, len={len(obj)}")
        for i, v in enumerate(obj[:3]):   # 只看前3个，避免刷屏
            inspect_structure(v, prefix=f"{prefix}[{i}]", max_depth=max_depth, depth=depth+1, visited=visited)

    elif isinstance(obj, np.ndarray):
        print(f"{indent}{prefix}: ndarray, shape={obj.shape}, dtype={obj.dtype}")

    else:
        print(f"{indent}{prefix}: {type(obj).__name__}, value={repr(obj)[:120]}")


def rotation_matrix_to_rotvec_batch(Rs):
    return SciRot.from_matrix(Rs).as_rotvec()


def relative_rotvecs_from_reference(R_samples, R_ref):
    """
    R_samples: (N,3,3)
    R_ref    : (3,3)
    return   : (N,3) relative rotation vectors log(R_ref^T R)^vee
    """
    R_rel = np.einsum("ij,njk->nik", R_ref.T, R_samples)
    return rotation_matrix_to_rotvec_batch(R_rel)


def plot_family_piball_partB(family_fixed, max_points=700, use_common_reference=True, seed=0):
    rng = np.random.default_rng(seed)
    n_cases = len(family_fixed)

    if use_common_reference:
        R_ref_common = family_fixed[0]["stored_R0"][0]
    else:
        R_ref_common = None

    ncols = 3
    nrows = int(np.ceil(n_cases / ncols))

    fig = plt.figure(figsize=(16, 10))

    # 预留右侧 colorbar 和顶部总标题空间
    gs = fig.add_gridspec(
        nrows=nrows, ncols=ncols,
        left=0.05, right=0.88, bottom=0.07, top=0.88,
        wspace=0.18, hspace=0.28
    )

    # pi-ball wireframe
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    X = np.pi * np.outer(np.cos(u), np.sin(v))
    Y = np.pi * np.outer(np.sin(u), np.sin(v))
    Z = np.pi * np.outer(np.ones_like(u), np.cos(v))

    scatter_handle = None
    lim = np.pi

    for i, case in enumerate(family_fixed):
        R_samples = case["stored_R"]
        R0_stack  = case["stored_R0"]
        s_vals    = case["s"]
        acc       = case.get("acceptance_rate", None)

        R_ref = R_ref_common if use_common_reference else R0_stack[0]

        if len(R_samples) > max_points:
            idx = rng.choice(len(R_samples), size=max_points, replace=False)
            R_plot = R_samples[idx]
        else:
            R_plot = R_samples

        rv = relative_rotvecs_from_reference(R_plot, R_ref)
        ang = np.linalg.norm(rv, axis=1)

        ax = fig.add_subplot(gs[i // ncols, i % ncols], projection="3d")

        ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, alpha=0.08)

        scatter_handle = ax.scatter(
            rv[:, 0], rv[:, 1], rv[:, 2],
            c=ang, s=10, alpha=0.82, cmap="viridis"
        )

        # 原点 / mode
        ax.scatter([0], [0], [0], marker="*", s=120)

        s_text = ", ".join(f"{v:.1f}" for v in np.asarray(s_vals).ravel())
        if acc is not None:
            title_text = f"s = [{s_text}]\nacc = {acc:.3f}"
        else:
            title_text = f"s = [{s_text}]"

        ax.set_title(title_text, pad=8, fontsize=12)

        # ax.set_xlabel(r"$r_1$", labelpad=2)
        # ax.set_ylabel(r"$r_2$", labelpad=2)
        # ax.set_zlabel(r"$r_3$", labelpad=2)

        # ax.set_xlim([-lim, lim])
        # ax.set_ylim([-lim, lim])
        # ax.set_zlim([-lim, lim])
        # ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel(r"$r_1$", labelpad=4)
        ax.set_ylabel(r"$r_2$", labelpad=4)
        ax.set_zlabel(r"$r_3$", labelpad=4)

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        ax.set_box_aspect([1, 1, 1])

        tick_vals = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.set_zticks(tick_vals)

        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_zticklabels(tick_labels, fontsize=9)

        ax.view_init(elev=28, azim=-60)



        # 视角可稍微统一得更舒服
        ax.view_init(elev=28, azim=-60)

    # 单独放右侧 colorbar
    cax = fig.add_axes([0.90, 0.22, 0.018, 0.56])
    #cbar = fig.colorbar(scatter_handle, cax=cax)
    #cbar.set_label("relative rotation angle (rad)", rotation=90, labelpad=10)

    cbar = fig.colorbar(scatter_handle, cax=cax)
    cbar.set_label("relative rotation angle", rotation=90, labelpad=10)

    cb_ticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    cb_labels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(cb_labels)



    # fig.suptitle(
    #     r"$\pi$-ball comparison for Part B (fixed mode, varying $F$)$",
    #     fontsize=18,
    #     y=0.95
    # )

    fig.suptitle(
    r"$\pi$-ball comparison for Part B (fixed mode, varying $F$)",
    fontsize=18,
    y=0.95
)

    plt.show()


def plot_partB_angle_boxplot(family_fixed, use_common_reference=True):
    if use_common_reference:
        R_ref_common = family_fixed[0]["stored_R0"][0]

    angle_data = []
    labels = []

    for case in family_fixed:
        R_samples = case["stored_R"]
        R_ref = R_ref_common if use_common_reference else case["stored_R0"][0]

        rv = relative_rotvecs_from_reference(R_samples, R_ref)
        ang = np.linalg.norm(rv, axis=1)
        angle_data.append(ang)

        s_vals = case["s"]
        labels.append("[" + ", ".join(f"{v:.1f}" for v in s_vals) + "]")

    plt.figure(figsize=(9, 5))
    plt.boxplot(angle_data, tick_labels=labels, showfliers=False)

    plt.ylabel(r"relative rotation angle")
    plt.xlabel(r"singular values $s$")
    plt.title(r"Angle spread across Part B cases")

    # y-axis ticks in multiples of pi
    # tick_vals = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    # tick_labels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]

    tick_vals = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi]
    tick_labels = [r"$0$", r"$\pi/6$", r"$\pi/3$", r"$\pi/2$", r"$2\pi/3$", r"$5\pi/6$", r"$\pi$"]
    plt.yticks(tick_vals, tick_labels)

    plt.ylim(0, np.pi)
    plt.tight_layout()
    plt.show()


def cartesian_to_lon_lat(points: np.ndarray):
    """
    Convert the spherical points (N,3) to latitude and longitude:
        lon in [-pi, pi]
        lat in [-pi/2, pi/2]
    """
    pts = np.asarray(points, dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    return lon, lat


def spherical_density_histogram(points: np.ndarray, n_lon: int = 72, n_lat: int = 36):
    """
    Create a two-dimensional density histogram for spherical points with area correction.
    Returns:
        density: shape [n_lat, n_lon]
        lon_edges
        lat_edges
    """
    lon, lat = cartesian_to_lon_lat(points)

    lon_edges = np.linspace(-np.pi, np.pi, n_lon + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)

    counts, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])

    # 面积修正：球面经纬网每个 bin 的面积与 sin(lat2)-sin(lat1) 成正比
    area = np.zeros((n_lat, n_lon))
    for i in range(n_lat):
        lat1 = lat_edges[i]
        lat2 = lat_edges[i + 1]
        band_area = (np.sin(lat2) - np.sin(lat1)) * (2 * np.pi / n_lon)
        area[i, :] = band_area

    density = counts / np.maximum(area, 1e-12)
    density = density / np.sum(density)  # 归一化成相对密度
    return density, lon_edges, lat_edges


def plot_single_axis_heatmap(ax, points: np.ndarray, mode_point: np.ndarray,
                             title: str, n_lon: int = 72, n_lat: int = 36,
                             cmap: str = "viridis"):
    
    density, lon_edges, lat_edges = spherical_density_histogram(points, n_lon=n_lon, n_lat=n_lat)

    im = ax.pcolormesh(
        lon_edges, lat_edges, density,
        shading="auto",
        cmap=cmap
    )

    # mode direction 也转成 lon-lat 标出来
    mode_lon, mode_lat = cartesian_to_lon_lat(mode_point.reshape(1, 3))
    ax.scatter(mode_lon, mode_lat, marker="*", s=120, color="red", label="mode direction")

    ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi / 2, np.pi / 2])

    ax.set_xticks(
        [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    )
    ax.set_yticks(
        [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],
        [r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"]
    )

    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    return im


def visualize_axis_density_heatmaps(results: dict, n_lon: int = 72, n_lat: int = 36,
                                    cmap: str = "viridis"):
    
    stored_R = results["stored_R"]
    U = results["U"]
    V = results["V"]
    s = results["s"]

    R_mode = U @ V.T
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    pts1 = apply_rotations_to_vector(stored_R, e1)
    pts2 = apply_rotations_to_vector(stored_R, e2)
    pts3 = apply_rotations_to_vector(stored_R, e3)

    mode1 = R_mode @ e1
    mode2 = R_mode @ e2
    mode3 = R_mode @ e3

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    im1 = plot_single_axis_heatmap(
        axes[0], pts1, mode1,
        title=rf"Density heatmap of $R e_1$, $s=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f})$",
        n_lon=n_lon, n_lat=n_lat, cmap=cmap
    )
    im2 = plot_single_axis_heatmap(
        axes[1], pts2, mode2,
        title=rf"Density heatmap of $R e_2$",
        n_lon=n_lon, n_lat=n_lat, cmap=cmap
    )
    im3 = plot_single_axis_heatmap(
        axes[2], pts3, mode3,
        title=rf"Density heatmap of $R e_3$",
        n_lon=n_lon, n_lat=n_lat, cmap=cmap
    )


def compare_re3_density_heatmaps_across_family(family_results, n_lon: int = 72, n_lat: int = 36,
                                               cmap: str = "viridis"):
    
    m = len(family_results)
    fig, axes = plt.subplots(1, m, figsize=(4.5 * m, 4.2))

    if m == 1:
        axes = [axes]
    global ims
    ims = []
    e3 = np.array([0.0, 0.0, 1.0])

    for ax, res in zip(axes, family_results):
        stored_R = res["stored_R"]
        U = res["U"]
        V = res["V"]
        s = res["s"]
        acc = res["acceptance_rate"]

        R_mode = U @ V.T
        pts3 = apply_rotations_to_vector(stored_R, e3)
        mode3 = R_mode @ e3

        im = plot_single_axis_heatmap(
            ax, pts3, mode3,
            title=f"s=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f})\nacc={acc:.3f}",
            n_lon=n_lon, n_lat=n_lat, cmap=cmap
        )
        ims.append(im)

    # 共用 colorbar
    from matplotlib.lines import Line2D

    legend_handle = Line2D(
        [0], [0],
        marker='*',
        color='w',
        markerfacecolor='red',
        markersize=14,
        linestyle='None',
        label='mode direction'
    )

    fig.legend(
        handles=[legend_handle],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=True
    )

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("relative density")



    
    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    
    from matplotlib.lines import Line2D

    legend_handle = Line2D(
        [0], [0],
        marker='*',
        color='w',
        markerfacecolor='red',
        markersize=14,
        linestyle='None',
        label='mode direction'
    )

    fig.legend(
        handles=[legend_handle],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=True
    )

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("relative density")


    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def hat(v: np.ndarray) -> np.ndarray:
    """
    hat map: R^3 -> so(3)
    """
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])


def vee(M: np.ndarray) -> np.ndarray:
    """
    vee map: so(3) -> R^3
    """
    return np.array([
        M[2, 1],
        M[0, 2],
        M[1, 0]
    ], dtype=float)


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """
    数值上将一个接近 SO(3) 的矩阵投影回 SO(3)。
    """
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt
    return R_proj


def log_map_so3(R: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Calculate the log-map on SO(3) relative to the identity matrix:

    xi = log(R)^vee in R^3

    Return the rotation vector xi, satisfying ||xi|| = rotation angle.

    This implementation is robust enough for the current Matrix-Fisher sample.

    Numerical strategies:

    - Use a first-order approximation for small angles

    - Use the standard formula for general angles

    - Recover the rotation axis using diagonal elements when approaching pi
    """
    R = project_to_so3(R)

    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # 小角度近似
    if theta < 1e-7:
        return vee(0.5 * (R - R.T))

    # 接近 pi 时的稳定处理
    if np.pi - theta < 1e-5:
        # 从对角元恢复轴方向
        A = (R + np.eye(3)) / 2.0
        axis = np.zeros(3)

        axis[0] = np.sqrt(max(A[0, 0], 0.0))
        axis[1] = np.sqrt(max(A[1, 1], 0.0))
        axis[2] = np.sqrt(max(A[2, 2], 0.0))

        # 用非对角元决定符号
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]

        norm_axis = np.linalg.norm(axis)
        if norm_axis < eps:
            # fallback
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / norm_axis

        return theta * axis

    # 一般情形
    Omega = (R - R.T) / (2.0 * np.sin(theta))
    return theta * vee(Omega)


def tangent_vectors_about_mode(Rs: np.ndarray, R_mode: np.ndarray) -> np.ndarray:
    """
    对一批旋转样本 Rs，计算相对于 mode 的 log-map:
        xi_i = log(R_mode^T R_i)^vee

    输入:
        Rs: shape [N, 3, 3]
        R_mode: shape [3, 3]

    输出:
        Xi: shape [N, 3]
    """
    Xi = []
    for R in Rs:
        R_rel = R_mode.T @ R
        xi = log_map_so3(R_rel)
        Xi.append(xi)
    return np.asarray(Xi, dtype=float)


def covariance_ellipse_params(X2: np.ndarray, n_std: float = 2.0):
    """
    For a 2D point cloud X2 (N x 2), compute the covariance ellipse parameters:
    ellipse center, width, height, rotation angle.
    """
    mu = np.mean(X2, axis=0)
    C = np.cov(X2.T)

    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width = 2.0 * n_std * np.sqrt(max(eigvals[0], 0.0))
    height = 2.0 * n_std * np.sqrt(max(eigvals[1], 0.0))
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return mu, width, height, angle, C, eigvals, eigvecs


def plot_tangent_space_3d(Xi: np.ndarray, title: str = None, max_points: int = 2500):
    """
    图 1：log-map 之后的 3D tangent-space scatter
    """
    Xi_plot = Xi.copy()
    if Xi_plot.shape[0] > max_points:
        idx = np.linspace(0, Xi_plot.shape[0] - 1, max_points).astype(int)
        Xi_plot = Xi_plot[idx]

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Xi_plot[:, 0], Xi_plot[:, 1], Xi_plot[:, 2], s=10, alpha=0.75)
    ax.scatter([0], [0], [0], s=100, marker="*", label=r"mode ($\xi=0$)")
    ax.legend()

    ax.set_xlabel(r"$\xi_1$")
    ax.set_ylabel(r"$\xi_2$")
    ax.set_zlabel(r"$\xi_3$")

    if title is None:
        title = r"3D tangent-space scatter of $\xi_i=\log(R_{\mathrm{mode}}^\top R_i)^\vee$"
    ax.set_title(title)

    # 自动设定坐标范围
    lim = np.max(np.abs(Xi_plot)) * 1.1
    lim = max(lim, 0.5)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect((1, 1, 1))

    # lim = np.pi
    # ax.set_xlim([-lim, lim])
    # ax.set_ylim([-lim, lim])
    # ax.set_zlim([-lim, lim])

    plt.tight_layout()
    plt.show()


def plot_tangent_pairwise(Xi: np.ndarray, n_std: float = 2.0, max_points: int = 2500):
    """
    Figure 2: Pairwise 2D tangent-space scatter + covariance ellipse
    Visualizes anisotropy most intuitively.
    """
    Xi_plot = Xi.copy()
    if Xi_plot.shape[0] > max_points:
        idx = np.linspace(0, Xi_plot.shape[0] - 1, max_points).astype(int)
        Xi_plot = Xi_plot[idx]

    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [
        (r"$\xi_1$", r"$\xi_2$"),
        (r"$\xi_1$", r"$\xi_3$"),
        (r"$\xi_2$", r"$\xi_3$")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, (i, j), (lx, ly) in zip(axes, pairs, labels):
        X2 = Xi_plot[:, [i, j]]

        ax.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.55)
        ax.scatter([0], [0], s=80, marker="*", label="mode")

        mu, width, height, angle, C, eigvals, eigvecs = covariance_ellipse_params(X2, n_std=n_std)

        # 画经验均值点
        ax.scatter([mu[0]], [mu[1]], s=40, marker="x", label="empirical mean")

        # 画协方差椭圆
        ell = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            linewidth=2.0
        )
        ax.add_patch(ell)

        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_title(
            f"{lx} vs {ly}\n"
            f"eig std ≈ ({np.sqrt(eigvals[0]):.3f}, {np.sqrt(eigvals[1]):.3f})"
        )
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend()

    fig.suptitle(r"Pairwise tangent-space scatter with covariance ellipses", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()


def summarize_tangent_cloud(Xi: np.ndarray):
    """
    Output the statistical summary of the tangent cloud, used to explain anisotropy.
    """
    mu = np.mean(Xi, axis=0)
    C = np.cov(Xi.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    print("=" * 72)
    print("Tangent-space summary around R_mode")
    print("=" * 72)
    print("Empirical mean of xi:")
    print(mu)
    print("\nEmpirical covariance of xi:")
    print(C)
    print("\nEigenvalues of covariance (descending):")
    print(eigvals)
    print("\nCorresponding principal directions (columns):")
    print(eigvecs)
    print("=" * 72)


def visualize_tangent_space_for_results(results: dict, max_points: int = 2500):
    """
    Visualize the tangent space for a given set of sampling results (i.e., the output of sample_mf_so3_independence_mh).
    Directly generate log-map / tangent-space visualizations.
    """
    Rs = results["stored_R"]
    U = results["U"]
    V = results["V"]
    s = results["s"]

    R_mode = U @ V.T
    Xi = tangent_vectors_about_mode(Rs, R_mode)

    print(f"\nTangent-space visualization for s = ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f})")
    summarize_tangent_cloud(Xi)

    plot_tangent_space_3d(
        Xi,
        title=rf"3D tangent-space scatter around $R_{{\mathrm{{mode}}}}$, "
              rf"$s=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f})$",
        max_points=max_points
    )
    plot_tangent_pairwise(Xi, n_std=2.0, max_points=max_points)

    return Xi


def compare_tangent_space_across_family(family_results, max_points: int = 1800):
    """
    Compare the tangent space across a family of results:
    For each case, plot a xi1-xi2 plane (with covariance ellipse),
    making it easy to directly compare changes in anisotropy / concentration.
    """
    m = len(family_results)
    fig, axes = plt.subplots(1, m, figsize=(4.3 * m, 4.6))

    if m == 1:
        axes = [axes]

    for ax, res in zip(axes, family_results):
        Rs = res["stored_R"]
        U = res["U"]
        V = res["V"]
        s = res["s"]
        R_mode = U @ V.T

        Xi = tangent_vectors_about_mode(Rs, R_mode)
        if Xi.shape[0] > max_points:
            idx = np.linspace(0, Xi.shape[0] - 1, max_points).astype(int)
            Xi = Xi[idx]

        X2 = Xi[:, [0, 1]]
        ax.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.5)
        ax.scatter([0], [0], s=70, marker="*", label="mode")

        mu, width, height, angle, C, eigvals, eigvecs = covariance_ellipse_params(X2, n_std=2.0)
        ell = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            linewidth=2.0
        )
        ax.add_patch(ell)

        ax.set_title(
            f"s=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f})\n"
            f"std≈({np.sqrt(eigvals[0]):.2f},{np.sqrt(eigvals[1]):.2f})"
        )
        ax.set_xlabel(r"$\xi_1$")
        ax.set_ylabel(r"$\xi_2$")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend()

    fig.suptitle(r"Tangent-space comparison across different Matrix--Fisher parameters", y=1.03, fontsize=14)
    plt.tight_layout()
    plt.show()


def gaussian_kde_1d(samples: np.ndarray, grid: np.ndarray, bandwidth: float = None):
    """
    A simple 1D Gaussian KDE implementation that does not depend on scipy.
    samples: shape [N]
    grid: shape [M]
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return np.zeros_like(grid)

    std = np.std(x, ddof=1)
    if bandwidth is None:
        # Silverman's rule of thumb
        bandwidth = 1.06 * std * (n ** (-1 / 5))
        bandwidth = max(bandwidth, 1e-3)

    diff = grid[:, None] - x[None, :]
    vals = np.exp(-0.5 * (diff / bandwidth) ** 2) / (np.sqrt(2 * np.pi) * bandwidth)
    density = np.mean(vals, axis=1)
    return density


def plot_family_angle_density_curves(
    family_results,
    use_mode_distance: bool = True,
    num_grid: int = 300,
    bandwidth: float = None,
    title: str = None
):
    """
    Merge the angle distance distributions of each case in family_results into a single smooth density curve.

    use_mode_distance=True:
        Plot d(R, R_mode)
    use_mode_distance=False:
        Plot d(R0, I)
    """
    all_angles = []

    for res in family_results:
        if use_mode_distance:
            U = res["U"]
            V = res["V"]
            R_mode = U @ V.T
            angles = np.array([geodesic_distance_so3(R, R_mode) for R in res["stored_R"]])
        else:
            angles = np.array([geodesic_distance_so3(R0, np.eye(3)) for R0 in res["stored_R0"]])
        all_angles.append(angles)

    max_angle = max(np.max(a) for a in all_angles)
    grid = np.linspace(0.0, max_angle * 1.05, num_grid)

    plt.figure(figsize=(8.5, 5.5))

    for res, angles in zip(family_results, all_angles):
        s = res["s"]
        label = f"s=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f})"
        dens = gaussian_kde_1d(angles, grid, bandwidth=bandwidth)
        plt.plot(grid, dens, linewidth=2, label=label)

    xlabel = r"$\theta = d(R,R_{\mathrm{mode}})$ (rad)" if use_mode_distance else r"$\theta = d(R_0,I)$ (rad)"
    if title is None:
        title = "Smoothed angle-density comparison across cases"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



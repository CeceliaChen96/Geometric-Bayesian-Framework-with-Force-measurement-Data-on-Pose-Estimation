"""Notebook-extracted core functions from clear_mfg_post.ipynb.

This file is an initial migration artifact. The function bodies are copied from the
notebook with minimal editing so the original logic stays intact. Higher-level
modules such as geometry.py, distributions.py, sdf.py, and algorithms.py re-export
selected symbols from here.

Source notebook:
- notebooks/posterior/clear_mfg_post.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional


@dataclass
class Pose:
    """
    位姿 X = (R, p)
    R: 3x3 旋转矩阵
    p: 3 维平移向量
    """
    R: np.ndarray   # shape (3,3)
    p: np.ndarray   # shape (3,)


@dataclass
class MFGParameters:
    """
    Matrix-Fisher-Gaussian 分布参数
    Theta = {F, mu, Lambda, Gamma}
    
    F      : Matrix Fisher 参数矩阵, shape (3,3)
    mu     : Gaussian 平移均值, shape (3,)
    Lambda : 平移精度矩阵, shape (3,3)
    Gamma  : 旋转-平移耦合矩阵, shape (3,3)
    """
    F: np.ndarray
    mu: np.ndarray
    Lambda: np.ndarray
    Gamma: np.ndarray


@dataclass
class SuperquadricFieldParameters:
    """
    超二次隐式场参数
    
    F_3d(x,y,z) = (f(x,y))^(eps2/eps1) + |z/a3|^(2/eps1) - 1
    f(x,y)      = |x/a1|^(2/eps2) + |y/a2|^(2/eps2)
    """
    a1: float
    a2: float
    a3: float
    eps1: float
    eps2: float
    sdf_eps: float = 1e-8  # 用于 normalized signed field 的分母稳定项


@dataclass
class StiffnessParameters:
    """
    光滑刚度函数参数
    
    k(d) = k_min + ((1 - tanh(d/d0))/2) * k_max
    """
    k_min: float
    k_max: float
    d0: float


def ensure_vector(x: np.ndarray, dim: int) -> np.ndarray:
    """
    保证输入是指定长度的一维向量
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != dim:
        raise ValueError(f"向量维度错误，期望 {dim}，实际 {x.shape[0]}")
    return x


def ensure_matrix(x: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    保证输入是指定大小的矩阵
    """
    x = np.asarray(x, dtype=float)
    if x.shape != shape:
        raise ValueError(f"矩阵维度错误，期望 {shape}，实际 {x.shape}")
    return x


def hat(v: np.ndarray) -> np.ndarray:
    """
    向量 -> 反对称矩阵
    对于 v = [v1,v2,v3]^T,
    hat(v) = [[ 0, -v3,  v2],
              [ v3,  0, -v1],
              [-v2, v1,   0]]
    """
    v = ensure_vector(v, 3)
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])


def vee(M: np.ndarray) -> np.ndarray:
    """
    反对称矩阵 -> 向量
    """
    M = ensure_matrix(M, (3, 3))
    return np.array([
        M[2, 1],
        M[0, 2],
        M[1, 0]
    ])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """
    SO(3) 指数映射
    R = exp(hat(phi))
    使用 Rodrigues 公式
    """
    phi = ensure_vector(phi, 3)
    theta = np.linalg.norm(phi)

    if theta < 1e-12:
        return np.eye(3) + hat(phi)

    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta ** 2)
    Phi = hat(phi)
    return np.eye(3) + A * Phi + B * (Phi @ Phi)


def log_so3(R: np.ndarray) -> np.ndarray:
    """
    SO(3) 对数映射
    phi = log(R)^\vee
    """
    R = ensure_matrix(R, (3, 3))
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-12:
        return vee(R - R.T) / 2.0

    return vee((theta / (2.0 * np.sin(theta))) * (R - R.T))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """
    用 SVD 将一个 3x3 矩阵投影到 SO(3)
    """
    M = ensure_matrix(M, (3, 3))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return R


def right_plus_pose(X: Pose, xi: np.ndarray) -> Pose:
    """
    右扰动:
        X ⊕ xi = (R exp(hat(phi)), p + v)
    其中 xi = [phi; v] ∈ R^6
    """
    xi = ensure_vector(xi, 6)
    phi = xi[:3]
    v = xi[3:]
    R_new = X.R @ exp_so3(phi)
    p_new = X.p + v
    return Pose(R=R_new, p=p_new)


def pose_inverse(X: Pose) -> Pose:
    """
    位姿逆:
        X^{-1} = (R^T, -R^T p)
    """
    R_inv = X.R.T
    p_inv = -R_inv @ X.p
    return Pose(R=R_inv, p=p_inv)


def compose_pose(X1: Pose, X2: Pose) -> Pose:
    """
    位姿复合:
        X1 * X2 = (R1 R2, R1 p2 + p1)
    """
    R = X1.R @ X2.R
    p = X1.R @ X2.p + X1.p
    return Pose(R=R, p=p)


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3,)
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """
    SO(3) 指数映射
    """
    w = np.asarray(w, dtype=float).reshape(3,)
    theta = np.linalg.norm(w)
    W = skew(w)

    if theta < 1e-12:
        return np.eye(3) + W

    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta**2)
    return np.eye(3) + A * W + B * (W @ W)


def rotation_mode_from_F(F: np.ndarray) -> np.ndarray:
    """
    根据 Proposition VI.1:
        F = U S V^T
        R_hat = U diag(1,1,det(UV^T)) V^T
    从 F 中恢复 MAP 旋转
    """
    F = ensure_matrix(F, (3, 3))

    # print("进入 rotation_mode_from_F")
    # print("F =\n", F)
    # print("F finite? ->", np.isfinite(F).all())
    # print("max |F| =", np.max(np.abs(F)))

    if not np.isfinite(F).all():
        raise ValueError("F 中出现了 NaN 或 Inf，无法做 SVD。")

    U, _, Vt = np.linalg.svd(F)
    R_hat = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return R_hat


def recover_pose_from_theta(theta: MFGParameters) -> Pose:
    """
    根据 Proposition VI.1 从后验参数中恢复 MAP pose:
        R_hat = mode(F_post)
        p_hat = mu_post
    """
    R_hat = rotation_mode_from_F(theta.F)
    p_hat = ensure_vector(theta.mu, 3)
    return Pose(R=R_hat, p=p_hat.copy())


def nu_from_rotation(R: np.ndarray) -> np.ndarray:
    """
    计算 nu_R = (R^T F - F^T R)^vee 的这种标准构造，
    但在当前代码中，Algorithm 1 里需要的是固定 nominal pose 对应的 nu_{Rbar}。
    
    由于论文中 Algorithm 1 用到了:
        eta <- Lambda (mu + Gamma nu_{Rbar})
    这里单独给一个通用接口
    """
    R = ensure_matrix(R, (3, 3))
    return log_so3(R)


def adjoint_wrench_transform(R_AB: np.ndarray, p_AB: np.ndarray) -> np.ndarray:
    """
    wrench 从 frame {B} 变换到 frame {A} 的矩阵:
    
        w_A = [ R_AB      p_AB^x R_AB ] w_B
              [  0            R_AB    ]
    
    其中 w = [tau; f]
    """
    R_AB = ensure_matrix(R_AB, (3, 3))
    p_AB = ensure_vector(p_AB, 3)

    T = np.block([
        [R_AB,              hat(p_AB) @ R_AB],
        [np.zeros((3, 3)),  R_AB]
    ])
    return T


def split_jacobian_blocks(J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 6x6 Jacobian 分成 [J_phi, J_v]
    """
    J = ensure_matrix(J, (6, 6))
    return J[:, :3], J[:, 3:]


def skew_cross_matrix(v: np.ndarray) -> np.ndarray:

    return hat(v)


def smooth_abs_power(x: float, power: float, eps: float = 1e-12) -> float:
    """
    平滑版 |x|^power
    用 (x^2 + eps)^(power/2) 近似，避免在 x=0 处数值不稳定。
    
    说明：
    论文里是严格的 |x|^p。
    这里为了数值稳定，代码实现时采用一个轻微平滑版本。
    """
    return (x * x + eps) ** (0.5 * power)


def superquadric_f_xy(c: np.ndarray, params: SuperquadricFieldParameters) -> float:
    """
    二维中间函数:
        f(x,y) = |x/a1|^(2/eps2) + |y/a2|^(2/eps2)
    """
    c = ensure_vector(c, 3)
    x, y, _ = c
    px = 2.0 / params.eps2
    py = 2.0 / params.eps2

    term_x = smooth_abs_power(x / params.a1, px)
    term_y = smooth_abs_power(y / params.a2, py)
    return term_x + term_y


def superquadric_F3d(c: np.ndarray, params: SuperquadricFieldParameters) -> float:
    """
    超二次隐式场:
        F_3d(x,y,z) = (f(x,y))^(eps2/eps1) + |z/a3|^(2/eps1) - 1
    """
    c = ensure_vector(c, 3)
    _, _, z = c

    fxy = superquadric_f_xy(c, params)
    power_xy = params.eps2 / params.eps1
    power_z = 2.0 / params.eps1

    term_xy = (fxy + 1e-16) ** power_xy
    term_z = smooth_abs_power(z / params.a3, power_z)

    return term_xy + term_z - 1.0


def numerical_gradient_scalar_field(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-6
) -> np.ndarray:
    """
    用中心差分计算标量场的梯度
    """
    x = ensure_vector(x, 3)
    grad = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        grad[i] = (func(x + h * e) - func(x - h * e)) / (2.0 * h)
    return grad


def numerical_hessian_scalar_field(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-4
) -> np.ndarray:
    """
    用中心差分计算标量场的 Hessian
    """
    x = ensure_vector(x, 3)
    H = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            ei = np.zeros(3)
            ej = np.zeros(3)
            ei[i] = 1.0
            ej[j] = 1.0

            f_pp = func(x + h * ei + h * ej)
            f_pm = func(x + h * ei - h * ej)
            f_mp = func(x - h * ei + h * ej)
            f_mm = func(x - h * ei - h * ej)

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)

    return H


def normalized_signed_field(c: np.ndarray, params: SuperquadricFieldParameters) -> float:
    """
    归一化 signed field:
        F_tilde(c) = F_3d(c) / (||grad F_3d(c)|| + eps)
    """
    c = ensure_vector(c, 3)

    def F_func(x):
        return superquadric_F3d(x, params)

    F_val = F_func(c)
    grad_F = numerical_gradient_scalar_field(F_func, c, h=1e-6)
    denom = np.linalg.norm(grad_F) + params.sdf_eps
    return F_val / denom


def normalized_signed_field_gradient(
    c: np.ndarray,
    params: SuperquadricFieldParameters
) -> np.ndarray:
    """
    计算 g_B = grad_c F_tilde(c)
    
    这里直接对 F_tilde 再做一次数值梯度，
    这样符合论文中对 normalized field 的定义，
    也避免手推复杂高阶导数。
    """
    c = ensure_vector(c, 3)

    def F_tilde_func(x):
        return normalized_signed_field(x, params)

    return numerical_gradient_scalar_field(F_tilde_func, c, h=1e-6)


def normalized_signed_field_hessian(
    c: np.ndarray,
    params: SuperquadricFieldParameters
) -> np.ndarray:
    """
    计算 Hessian_c F_tilde(c)
    
    对应 Algorithm 3 里的
        ∇^2_{c_B} F_tilde(c_B)
    """
    c = ensure_vector(c, 3)

    def F_tilde_func(x):
        return normalized_signed_field(x, params)

    return numerical_hessian_scalar_field(F_tilde_func, c, h=1e-4)


def stiffness_k(d: float, stiffness_params: StiffnessParameters) -> float:
    d = float(d)

    k_min = float(stiffness_params.k_min)
    k_max = float(stiffness_params.k_max)
    d0 = float(stiffness_params.d0)

    if d0 <= 0.0:
        raise ValueError("stiffness_params.d0 必须大于 0")

    u = d / d0
    return k_min + 0.5 * (1.0 - np.tanh(u)) * k_max


def stiffness_k_prime(d: float, stiffness_params: StiffnessParameters) -> float:
    d = float(d)

    k_max = float(stiffness_params.k_max)
    d0 = float(stiffness_params.d0)

    if d0 <= 0.0:
        raise ValueError("stiffness_params.d0 必须大于 0")

    u = d / d0
    #sech2 = 1.0 / np.cosh(u) ** 2
    t = np.tanh(u)
    sech2 = 1.0 - t * t
    return -0.5 * k_max * sech2 / d0


def stiffness_k_double_prime(d: float, stiffness_params: StiffnessParameters) -> float:
    d = float(d)

    k_max = float(stiffness_params.k_max)
    d0 = float(stiffness_params.d0)

    if d0 <= 0.0:
        raise ValueError("stiffness_params.d0 必须大于 0")

    u = d / d0
    #sech2 = 1.0 / np.cosh(u) ** 2
    t = np.tanh(u)
    sech2 = 1.0 - t * t
    return k_max * sech2 * np.tanh(u) / (d0 ** 2)


def alpha_from_F_tilde(F_tilde: float, stiffness_params: StiffnessParameters) -> float:
    F_tilde = float(F_tilde)

    k_val = stiffness_k(F_tilde, stiffness_params)
    k_prime_val = stiffness_k_prime(F_tilde, stiffness_params)

    alpha = k_val * F_tilde + 0.5 * k_prime_val * (F_tilde ** 2)
    return alpha


def kappa_from_F_tilde(F_tilde: float, stiffness_params: StiffnessParameters) -> float:
    F_tilde = float(F_tilde)

    k_val = stiffness_k(F_tilde, stiffness_params)
    k_prime_val = stiffness_k_prime(F_tilde, stiffness_params)
    k_double_prime_val = stiffness_k_double_prime(F_tilde, stiffness_params)

    kappa = (
        k_val
        + 2.0 * k_prime_val * F_tilde
        + 0.5 * k_double_prime_val * (F_tilde ** 2)
    )
    return kappa


def alpha_and_kappa_from_F_tilde(
    F_tilde: float,
    stiffness_params: StiffnessParameters
) -> tuple[float, float]:
    alpha = alpha_from_F_tilde(F_tilde, stiffness_params)
    kappa = kappa_from_F_tilde(F_tilde, stiffness_params)
    return alpha, kappa


def local_stiffness_matrix(
    c_B: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
):
    c_B = ensure_vector(c_B, 3)

    F_tilde = normalized_signed_field(c_B, field_params)
    g_B = normalized_signed_field_gradient(c_B, field_params)
    H_B = normalized_signed_field_hessian(c_B, field_params)

    alpha, kappa = alpha_and_kappa_from_F_tilde(F_tilde, stiffness_params)

    # Exact theory:
    #   K_i = kappa * g g^T + alpha * H
    K_i = kappa * np.outer(g_B, g_B) + alpha * H_B

    if (
        (not np.isfinite(F_tilde))
        or (not np.isfinite(g_B).all())
        or (not np.isfinite(H_B).all())
        or (not np.isfinite(alpha))
        or (not np.isfinite(kappa))
        or (not np.isfinite(K_i).all())
    ):
        print("数值异常!")
        print("c_B =", c_B)
        print("F_tilde =", F_tilde)
        print("g_B =", g_B)
        print("H_B =\n", H_B)
        print("alpha =", alpha)
        print("kappa =", kappa)
        print("K_i =\n", K_i)
        raise ValueError("local_stiffness_matrix 中出现 NaN/Inf")

    return g_B, F_tilde, alpha, K_i


def contact_point_in_frame_B(
    r_A_i: np.ndarray,
    X_WA: Pose,
    X_WB: Pose
) -> np.ndarray:
    """
    将 frame {A} 中的接触点 r_A^(i) 转换到 frame {B} 中：
    
        c_B^(i) = R_WB^T ( R_WA r_A^(i) + p_WA - p_WB )
    """
    r_A_i = ensure_vector(r_A_i, 3)
    R_WA = ensure_matrix(X_WA.R, (3, 3))
    p_WA = ensure_vector(X_WA.p, 3)
    R_WB = ensure_matrix(X_WB.R, (3, 3))
    p_WB = ensure_vector(X_WB.p, 3)

    return R_WB.T @ (R_WA @ r_A_i + p_WA - p_WB)


def single_contact_force_in_B(
    c_B_i: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    单个接触点在 frame {B} 中产生的接触力
    
    根据 Proposition III.2:
        f_B^(i) = - alpha^(i) g_B^(i)
    
    返回:
        f_B_i   : 3维接触力
        alpha_i : 标量系数
        g_B_i   : grad F_tilde(c_B_i)
    """
    g_B_i, F_tilde_i, alpha_i, _ = local_stiffness_matrix(
        c_B_i, field_params, stiffness_params
    )
    f_B_i = -alpha_i * g_B_i
    return f_B_i, alpha_i, g_B_i


def single_contact_wrench_in_B(
    c_B_i: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
) -> np.ndarray:
    """
    单个接触点在 frame {B} 中关于原点 O_B 的 wrench:
    
        tau_B^(i) = c_B^(i) x f_B^(i)
        w_i = [tau_B^(i); f_B^(i)]
    """
    c_B_i = ensure_vector(c_B_i, 3)
    f_B_i, _, _ = single_contact_force_in_B(
        c_B_i, field_params, stiffness_params
    )
    tau_B_i = np.cross(c_B_i, f_B_i)
    return np.concatenate([tau_B_i, f_B_i])


def interaction_wrench_in_B(
    X_WA: Pose,
    X_WB: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
) -> np.ndarray:
    """
    所有接触点总和得到的交互 wrench，表达在 frame {B} 中：
    
        w_int,B = sum_i [ c_B^(i) x f_B^(i) ; f_B^(i) ]
    """
    contact_points_A = np.asarray(contact_points_A, dtype=float)
    if contact_points_A.ndim != 2 or contact_points_A.shape[1] != 3:
        raise ValueError("contact_points_A 应为 shape (N,3)")

    w_B = np.zeros(6)

    for i in range(contact_points_A.shape[0]):
        r_A_i = contact_points_A[i]
        c_B_i = contact_point_in_frame_B(r_A_i, X_WA, X_WB)
        w_B += single_contact_wrench_in_B(
            c_B_i, field_params, stiffness_params
        )

    return w_B


def interaction_wrench_in_A(
    X_WA: Pose,
    X_WB: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
) -> np.ndarray:
    """
    将 frame {B} 中的总 wrench 变换到 frame {A} 中：
    
    若
        R_AB = R_WA^T R_WB
        p_AB = R_WA^T (p_WB - p_WA)
    则
        w_A = T(R_AB, p_AB) w_B
    """
    R_AB = X_WA.R.T @ X_WB.R
    p_AB = X_WA.R.T @ (X_WB.p - X_WA.p)

    w_B = interaction_wrench_in_B(
        X_WA, X_WB, contact_points_A, field_params, stiffness_params
    )
    T_AB = adjoint_wrench_transform(R_AB, p_AB)
    return T_AB @ w_B


def measurement_model_y(
    X_WA: Pose,
    X_WB: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
) -> np.ndarray:
    """
    观测模型中的无噪声部分：
        y_k = w_A^int(X_A, X_B) + eps_k
    这里返回的就是
        w_A^int(X_A, X_B)
    """
    return interaction_wrench_in_A(
        X_WA, X_WB, contact_points_A, field_params, stiffness_params
    )


def nominal_AB_transform(X_WA: Pose, X_WB_bar: Pose):
    """
    Return nominal transform quantities:
        R_AB = R_WA^T R_WB_bar
        p_AB = R_WA^T (p_WB_bar - p_WA)
    """
    R_WA = ensure_matrix(X_WA.R, (3, 3))
    p_WA = ensure_vector(X_WA.p, 3)

    R_WB = ensure_matrix(X_WB_bar.R, (3, 3))
    p_WB = ensure_vector(X_WB_bar.p, 3)

    R_AB = R_WA.T @ R_WB
    p_AB = R_WA.T @ (p_WB - p_WA)
    return R_AB, p_AB


def transform_derivative_term_JT(
    X_WA: Pose,
    X_WB_bar: Pose,
    tau_B_bar: np.ndarray,
    f_B_bar: np.ndarray
) -> np.ndarray:
    """
    Compute J_T = (dT/dxi)|_{X_B_bar} * w_int,B_bar, shape (6,6)

    where
        J_T = [ J_T_phi  J_T_v ]

    with
        J_T_phi = [ -R_AB tau_B^x - p_AB^x R_AB f_B^x
                    -R_AB f_B^x                         ]

        J_T_v   = [ -(R_AB f_B)^x R_WA^T
                     0                                 ]
    """
    tau_B_bar = ensure_vector(tau_B_bar, 3)
    f_B_bar = ensure_vector(f_B_bar, 3)

    R_WA = ensure_matrix(X_WA.R, (3, 3))
    R_AB, p_AB = nominal_AB_transform(X_WA, X_WB_bar)

    J_T_phi = np.vstack([
        -R_AB @ hat(tau_B_bar) - hat(p_AB) @ R_AB @ hat(f_B_bar),
        -R_AB @ hat(f_B_bar)
    ])

    J_T_v = np.vstack([
        -hat(R_AB @ f_B_bar) @ R_WA.T,
        np.zeros((3, 3))
    ])

    J_T = np.hstack([J_T_phi, J_T_v])
    return J_T


def interaction_wrench_jacobian_B(
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
):
    """
    Compute B-frame interaction wrench Jacobian:

        J_wB = d w_int,B / d xi  ∈ R^{6x6}

    under right perturbation
        X_B = X_WB_bar ⊕ xi = (R_WB_bar Exp(phi^), p_WB_bar + v).

    Returns
    -------
    J_wB : np.ndarray, shape (6,6)
        Wrench Jacobian in frame {B}.

    w_B_bar : np.ndarray, shape (6,)
        Nominal interaction wrench in frame {B},
        stacked as [tau_B_bar; f_B_bar].

    Notes
    -----
    This function assumes your helper

        local_stiffness_matrix(c_B_i, field_params, stiffness_params)

    has already been updated to the NEW theory and returns

        g_B_i, F_tilde_i, alpha_i, K_i

    with
        K_i = kappa_i * g_B_i g_B_i^T + alpha_i * Hessian(tildeF)(c_B_i).
    """
    contact_points_A = np.asarray(contact_points_A, dtype=float)
    if contact_points_A.ndim != 2 or contact_points_A.shape[1] != 3:
        raise ValueError("contact_points_A 应为 shape (N,3)")

    R_WA = ensure_matrix(X_WA.R, (3, 3))
    p_WA = ensure_vector(X_WA.p, 3)

    R_WB = ensure_matrix(X_WB_bar.R, (3, 3))
    p_WB = ensure_vector(X_WB_bar.p, 3)

    # Initialize Jacobian blocks
    J_phi = np.zeros((6, 3))
    J_v = np.zeros((6, 3))

    # Initialize nominal wrench accumulation in frame {B}
    tau_B_bar = np.zeros(3)
    f_B_bar = np.zeros(3)

    for i in range(contact_points_A.shape[0]):
        r_A_i = ensure_vector(contact_points_A[i], 3)

        # c_B^(i) = R_WB^T ( R_WA r_A^(i) + p_WA - p_WB )
        c_B_i = R_WB.T @ (R_WA @ r_A_i + p_WA - p_WB)

        # IMPORTANT:
        # local_stiffness_matrix(...) must already be updated to the NEW exact K_i
        g_B_i, F_tilde_i, alpha_i, K_i = local_stiffness_matrix(
            c_B_i, field_params, stiffness_params
        )

        # f_B^(i) = - alpha^(i) g_B^(i)
        f_B_i = -alpha_i * g_B_i

        c_cross = hat(c_B_i)
        f_cross = hat(f_B_i)

        # Rotational block:
        # d w_int,B / d phi
        # [ - f^x c^x - c^x K c^x
        #   - K c^x                 ]
        J_phi_i = np.vstack([
            -f_cross @ c_cross - c_cross @ K_i @ c_cross,
            -K_i @ c_cross
        ])
        J_phi += J_phi_i

        # Translational block:
        # d w_int,B / d v
        # [ (f^x + c^x K) R_WB^T
        #    K R_WB^T             ]
        J_v_i = np.vstack([
            (f_cross + c_cross @ K_i) @ R_WB.T,
            K_i @ R_WB.T
        ])
        J_v += J_v_i

        # Accumulate nominal wrench in frame {B}
        tau_B_bar += np.cross(c_B_i, f_B_i)
        f_B_bar += f_B_i

    J_wB = np.hstack([J_phi, J_v])
    w_B_bar = np.hstack([tau_B_bar, f_B_bar])

    return J_wB, w_B_bar


def interaction_wrench_jacobian_A(
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
) -> np.ndarray:
    """
    Compute A-frame interaction wrench Jacobian:

        J_wA = d w_int,A / d xi
             = T(R_AB, p_AB) J_wB + J_T

    Returns
    -------
    J_wA : np.ndarray, shape (6,6)
        Wrench Jacobian in frame {A}.
    """
    J_wB, w_B_bar = interaction_wrench_jacobian_B(
        X_WA, X_WB_bar, contact_points_A, field_params, stiffness_params
    )

    R_AB, p_AB = nominal_AB_transform(X_WA, X_WB_bar)
    T_AB = adjoint_wrench_transform(R_AB, p_AB)

    tau_B_bar = w_B_bar[:3]
    f_B_bar = w_B_bar[3:]

    J_T = transform_derivative_term_JT(
        X_WA, X_WB_bar, tau_B_bar, f_B_bar
    )

    J_wA = T_AB @ J_wB + J_T
    return J_wA


def interaction_residual_jacobian(
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters
) -> np.ndarray:
    """
    Compute the residual Jacobian:

        J_k = d r / d xi
            = - d w_int,A / d xi

    where
        r(X_B) = y - w_int,A(X_WA, X_B).

    This matches the NEW Algorithm 3 / Eq. (V.29).
    """
    J_wA = interaction_wrench_jacobian_A(
        X_WA, X_WB_bar, contact_points_A, field_params, stiffness_params
    )
    J_k = -J_wA
    return J_k


def interaction_wrench_B(
    X_WA: Pose,
    X_WB: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
) -> np.ndarray:
    """
    Compute the interaction wrench expressed in frame {B}:

        w_int,B = [tau_B; f_B] in R^6

    where
        f_B   = sum_i f_B^(i)
        tau_B = sum_i c_B^(i) x f_B^(i)

    with
        c_B^(i) = R_WB^T ( R_WA r_A^(i) + p_WA - p_WB ).
    """
    contact_points_A = np.asarray(contact_points_A, dtype=float)
    if contact_points_A.ndim != 2 or contact_points_A.shape[1] != 3:
        raise ValueError("contact_points_A 应为 shape (N,3)")

    R_WA = ensure_matrix(X_WA.R, (3, 3))
    p_WA = ensure_vector(X_WA.p, 3)

    R_WB = ensure_matrix(X_WB.R, (3, 3))
    p_WB = ensure_vector(X_WB.p, 3)

    tau_B = np.zeros(3)
    f_B = np.zeros(3)

    for i in range(contact_points_A.shape[0]):
        r_A_i = ensure_vector(contact_points_A[i], 3)

        # c_B^(i) = R_WB^T ( R_WA r_A^(i) + p_WA - p_WB )
        c_B_i = R_WB.T @ (R_WA @ r_A_i + p_WA - p_WB)

        # local_stiffness_matrix returns:
        #   g_B_i, F_tilde_i, alpha_i, K_i
        g_B_i, F_tilde_i, alpha_i, K_i = local_stiffness_matrix(
            c_B_i, field_params, stiffness_params
        )

        # f_B^(i) = - alpha^(i) g_B^(i)
        f_B_i = -alpha_i * g_B_i

        # tau_B^(i) = c_B^(i) x f_B^(i)
        tau_B_i = np.cross(c_B_i, f_B_i)

        tau_B += tau_B_i
        f_B += f_B_i

    return np.hstack([tau_B, f_B])


def interaction_wrench_A(
    X_WA: Pose,
    X_WB: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
) -> np.ndarray:
    """
    Compute the interaction wrench expressed in frame {A}:

        w_int,A = T(R_AB, p_AB) w_int,B

    where
        R_AB = R_WA^T R_WB
        p_AB = R_WA^T (p_WB - p_WA).
    """
    w_B = interaction_wrench_B(
        X_WA, X_WB, contact_points_A, field_params, stiffness_params
    )

    R_WA = ensure_matrix(X_WA.R, (3, 3))
    p_WA = ensure_vector(X_WA.p, 3)

    R_WB = ensure_matrix(X_WB.R, (3, 3))
    p_WB = ensure_vector(X_WB.p, 3)

    R_AB = R_WA.T @ R_WB
    p_AB = R_WA.T @ (p_WB - p_WA)

    T_AB = adjoint_wrench_transform(R_AB, p_AB)
    w_A = T_AB @ w_B
    return w_A


def numerical_wrench_jacobian_B(
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    h_rot: float = 1e-6,
    h_pos: float = 1e-6,
) -> np.ndarray:
    """
    Numerical B-frame wrench Jacobian:

        J_wB = d w_int,B / d xi

    using central finite differences under right perturbation.
    """
    J_num = np.zeros((6, 6))

    # rotational block
    for j in range(3):
        d = np.zeros(6)
        d[j] = h_rot

        X_plus = right_plus_pose(X_WB_bar, d)
        X_minus = right_plus_pose(X_WB_bar, -d)

        w_plus = interaction_wrench_B(
            X_WA, X_plus, contact_points_A, field_params, stiffness_params
        )
        w_minus = interaction_wrench_B(
            X_WA, X_minus, contact_points_A, field_params, stiffness_params
        )

        J_num[:, j] = (w_plus - w_minus) / (2.0 * h_rot)

    # translational block
    for j in range(3):
        d = np.zeros(6)
        d[3 + j] = h_pos

        X_plus = right_plus_pose(X_WB_bar, d)
        X_minus = right_plus_pose(X_WB_bar, -d)

        w_plus = interaction_wrench_B(
            X_WA, X_plus, contact_points_A, field_params, stiffness_params
        )
        w_minus = interaction_wrench_B(
            X_WA, X_minus, contact_points_A, field_params, stiffness_params
        )

        J_num[:, 3 + j] = (w_plus - w_minus) / (2.0 * h_pos)

    return J_num


def numerical_wrench_jacobian_A(
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    h_rot: float = 1e-6,
    h_pos: float = 1e-6,
) -> np.ndarray:
    """
    Numerical A-frame wrench Jacobian:

        J_wA = d w_int,A / d xi

    using central finite differences under right perturbation.
    """
    J_num = np.zeros((6, 6))

    # rotational block
    for j in range(3):
        d = np.zeros(6)
        d[j] = h_rot

        X_plus = right_plus_pose(X_WB_bar, d)
        X_minus = right_plus_pose(X_WB_bar, -d)

        w_plus = interaction_wrench_A(
            X_WA, X_plus, contact_points_A, field_params, stiffness_params
        )
        w_minus = interaction_wrench_A(
            X_WA, X_minus, contact_points_A, field_params, stiffness_params
        )

        J_num[:, j] = (w_plus - w_minus) / (2.0 * h_rot)

    # translational block
    for j in range(3):
        d = np.zeros(6)
        d[3 + j] = h_pos

        X_plus = right_plus_pose(X_WB_bar, d)
        X_minus = right_plus_pose(X_WB_bar, -d)

        w_plus = interaction_wrench_A(
            X_WA, X_plus, contact_points_A, field_params, stiffness_params
        )
        w_minus = interaction_wrench_A(
            X_WA, X_minus, contact_points_A, field_params, stiffness_params
        )

        J_num[:, 3 + j] = (w_plus - w_minus) / (2.0 * h_pos)

    return J_num


def numerical_residual_jacobian(
    y_k: np.ndarray,
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    h_rot: float = 1e-6,
    h_pos: float = 1e-6,
) -> np.ndarray:
    """
    Numerical residual Jacobian:

        r(X_B) = y_k - w_int,A(X_WA, X_B)
        J_k = d r / d xi

    using central finite differences under right perturbation.
    """
    y_k = ensure_vector(y_k, 6)
    J_num = np.zeros((6, 6))

    # rotational block
    for j in range(3):
        d = np.zeros(6)
        d[j] = h_rot

        X_plus = right_plus_pose(X_WB_bar, d)
        X_minus = right_plus_pose(X_WB_bar, -d)

        r_plus = y_k - interaction_wrench_A(
            X_WA, X_plus, contact_points_A, field_params, stiffness_params
        )
        r_minus = y_k - interaction_wrench_A(
            X_WA, X_minus, contact_points_A, field_params, stiffness_params
        )

        J_num[:, j] = (r_plus - r_minus) / (2.0 * h_rot)

    # translational block
    for j in range(3):
        d = np.zeros(6)
        d[3 + j] = h_pos

        X_plus = right_plus_pose(X_WB_bar, d)
        X_minus = right_plus_pose(X_WB_bar, -d)

        r_plus = y_k - interaction_wrench_A(
            X_WA, X_plus, contact_points_A, field_params, stiffness_params
        )
        r_minus = y_k - interaction_wrench_A(
            X_WA, X_minus, contact_points_A, field_params, stiffness_params
        )

        J_num[:, 3 + j] = (r_plus - r_minus) / (2.0 * h_pos)

    return J_num


def _print_jacobian_check(tag: str, J_ana: np.ndarray, J_num: np.ndarray):
    err_abs = np.linalg.norm(J_ana - J_num)
    err_rel = err_abs / (np.linalg.norm(J_num) + 1e-12)

    print(f"===== {tag} Jacobian Check =====")
    print("||J_ana - J_num||      =", err_abs)
    print("relative error         =", err_rel)
    print("||J_ana||              =", np.linalg.norm(J_ana))
    print("||J_num||              =", np.linalg.norm(J_num))

    return err_abs, err_rel


def detailed_jacobian_diagnostics(
    J_ana: np.ndarray,
    J_num: np.ndarray,
    tag: str = "Jacobian"
):
    """
    More detailed diagnostics:
      - rotational block error
      - translational block error
      - sign test
      - column-wise error

    Assumes xi = [phi, v], so:
      first 3 cols  -> rotational block
      last  3 cols  -> translational block
    """
    J_ana = np.asarray(J_ana, dtype=float)
    J_num = np.asarray(J_num, dtype=float)

    if J_ana.shape != (6, 6) or J_num.shape != (6, 6):
        raise ValueError("J_ana 和 J_num 都必须是 shape (6,6)")

    J_ana_phi = J_ana[:, :3]
    J_ana_v = J_ana[:, 3:]

    J_num_phi = J_num[:, :3]
    J_num_v = J_num[:, 3:]

    phi_abs = np.linalg.norm(J_ana_phi - J_num_phi)
    phi_rel = phi_abs / (np.linalg.norm(J_num_phi) + 1e-12)

    v_abs = np.linalg.norm(J_ana_v - J_num_v)
    v_rel = v_abs / (np.linalg.norm(J_num_v) + 1e-12)

    print(f"\n=== detailed diagnostics: {tag} ===")
    print("=== block check ===")
    print("phi block error =", phi_abs)
    print("phi block rel   =", phi_rel)
    print("v block error   =", v_abs)
    print("v block rel     =", v_rel)

    print("\n=== sign test ===")
    print("||J_ana - J_num|| =", np.linalg.norm(J_ana - J_num))
    print("||J_ana + J_num|| =", np.linalg.norm(J_ana + J_num))

    print("\n=== column-wise error ===")
    for j in range(6):
        col_abs = np.linalg.norm(J_ana[:, j] - J_num[:, j])
        col_rel = col_abs / (np.linalg.norm(J_num[:, j]) + 1e-12)
        print(f"col {j}: abs = {col_abs:.6e}, rel = {col_rel:.6e}")

    return {
        "phi_abs": phi_abs,
        "phi_rel": phi_rel,
        "v_abs": v_abs,
        "v_rel": v_rel,
        "sign_minus": np.linalg.norm(J_ana - J_num),
        "sign_plus": np.linalg.norm(J_ana + J_num),
    }


def check_wrench_jacobian_B(
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    h_rot: float = 1e-6,
    h_pos: float = 1e-6,
    detailed: bool = True,
):
    """
    Check:
        analytic J_wB  vs  numerical J_wB
    """
    J_ana, w_B_bar = interaction_wrench_jacobian_B(
        X_WA, X_WB_bar, contact_points_A, field_params, stiffness_params
    )

    J_num = numerical_wrench_jacobian_B(
        X_WA, X_WB_bar, contact_points_A, field_params, stiffness_params,
        h_rot=h_rot, h_pos=h_pos
    )

    err_abs, err_rel = _print_jacobian_check("B-frame wrench", J_ana, J_num)

    diag = None
    if detailed:
        diag = detailed_jacobian_diagnostics(J_ana, J_num, tag="B-frame wrench")

    return J_ana, J_num, err_abs, err_rel, diag


def check_wrench_jacobian_A(
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    h_rot: float = 1e-6,
    h_pos: float = 1e-6,
    detailed: bool = True,
):
    """
    Check:
        analytic J_wA  vs  numerical J_wA
    """
    J_ana = interaction_wrench_jacobian_A(
        X_WA, X_WB_bar, contact_points_A, field_params, stiffness_params
    )

    J_num = numerical_wrench_jacobian_A(
        X_WA, X_WB_bar, contact_points_A, field_params, stiffness_params,
        h_rot=h_rot, h_pos=h_pos
    )

    err_abs, err_rel = _print_jacobian_check("A-frame wrench", J_ana, J_num)

    diag = None
    if detailed:
        diag = detailed_jacobian_diagnostics(J_ana, J_num, tag="A-frame wrench")

    return J_ana, J_num, err_abs, err_rel, diag


def check_algorithm3_jacobian(
    y_k: np.ndarray,
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    h_rot: float = 1e-6,
    h_pos: float = 1e-6,
    detailed: bool = True,
):
    """
    Check the NEW Algorithm 3 output:

        analytic residual Jacobian  vs  numerical residual Jacobian
    """
    J_ana = interaction_residual_jacobian(
        X_WA, X_WB_bar, contact_points_A, field_params, stiffness_params
    )

    J_num = numerical_residual_jacobian(
        y_k, X_WA, X_WB_bar, contact_points_A, field_params, stiffness_params,
        h_rot=h_rot, h_pos=h_pos
    )

    err_abs, err_rel = _print_jacobian_check("Residual", J_ana, J_num)

    diag = None
    if detailed:
        diag = detailed_jacobian_diagnostics(J_ana, J_num, tag="Residual")

    return J_ana, J_num, err_abs, err_rel, diag


def numerical_rotation_jacobian_A(
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
    h_rot: float = 1e-5,
) -> np.ndarray:
    """
    Numerical rotational block of the A-frame wrench Jacobian:

        d w_int,A / d phi
    """
    J_phi = np.zeros((6, 3))
    basis = np.eye(3)

    for j in range(3):
        ej = basis[:, j]

        R_plus = X_WB_bar.R @ so3_exp(h_rot * ej)
        R_minus = X_WB_bar.R @ so3_exp(-h_rot * ej)

        X_plus = Pose(R=R_plus, p=X_WB_bar.p.copy())
        X_minus = Pose(R=R_minus, p=X_WB_bar.p.copy())

        w_plus = interaction_wrench_A(
            X_WA, X_plus, contact_points_A, field_params, stiffness_params
        )
        w_minus = interaction_wrench_A(
            X_WA, X_minus, contact_points_A, field_params, stiffness_params
        )

        J_phi[:, j] = (w_plus - w_minus) / (2.0 * h_rot)

    return J_phi


def numerical_rotation_residual_jacobian(
    y_k: np.ndarray,
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
    h_rot: float = 1e-5,
) -> np.ndarray:
    """
    Numerical rotational block of the residual Jacobian:

        d r / d phi
    """
    y_k = ensure_vector(y_k, 6)
    J_phi = np.zeros((6, 3))
    basis = np.eye(3)

    for j in range(3):
        ej = basis[:, j]

        R_plus = X_WB_bar.R @ so3_exp(h_rot * ej)
        R_minus = X_WB_bar.R @ so3_exp(-h_rot * ej)

        X_plus = Pose(R=R_plus, p=X_WB_bar.p.copy())
        X_minus = Pose(R=R_minus, p=X_WB_bar.p.copy())

        r_plus = y_k - interaction_wrench_A(
            X_WA, X_plus, contact_points_A, field_params, stiffness_params
        )
        r_minus = y_k - interaction_wrench_A(
            X_WA, X_minus, contact_points_A, field_params, stiffness_params
        )

        J_phi[:, j] = (r_plus - r_minus) / (2.0 * h_rot)

    return J_phi


def hybrid_residual_jacobian_v2(
    y_k: np.ndarray,
    X_WA: Pose,
    X_WB_bar: Pose,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    h_rot: float = 1e-5
) -> np.ndarray:
    """
    Hybrid residual Jacobian (V2) for Algorithm 1.

    This function returns the residual Jacobian
        J_k = d r / d xi
    in a hybrid form:
        - rotational block J_{k,phi}: numerical central difference
        - translational block J_{k,v}: analytic residual Jacobian block

    Specifically,
        J_k_hybrid = [ J_{k,phi}^{num}   J_{k,v}^{ana} ]

    where
        r(X_B) = y_k - w_int,A(X_WA, X_B)

    Parameters
    ----------
    y_k : np.ndarray, shape (6,)
        Observed wrench at the current step.
    X_WA : Pose
        Known pose of frame {A} in world frame.
    X_WB_bar : Pose
        Nominal pose (linearization point) of frame {B}.
    contact_points_A : np.ndarray, shape (N, 3)
        Contact points expressed in frame {A}.
    field_params : SuperquadricFieldParameters
        Parameters of the normalized signed field.
    stiffness_params : StiffnessParameters
        Parameters of the stiffness profile.
    h_rot : float
        Central-difference step size for the rotational block.

    Returns
    -------
    J_k_hybrid : np.ndarray, shape (6, 6)
        Hybrid residual Jacobian:
            first 3 cols  = numerical d r / d phi
            last  3 cols  = analytic  d r / d v
    """
    y_k = ensure_vector(y_k, 6)

    # Numerical rotational block of the residual Jacobian
    J_k_phi_num = numerical_rotation_residual_jacobian(
        y_k,
        X_WA,
        X_WB_bar,
        contact_points_A,
        field_params,
        stiffness_params,
        h_rot=h_rot
    )

    # Analytic full residual Jacobian
    J_k_ana = interaction_residual_jacobian(
        X_WA,
        X_WB_bar,
        contact_points_A,
        field_params,
        stiffness_params
    )

    # Keep the analytic translational block
    J_k_v_ana = J_k_ana[:, 3:]

    # Hybrid assembly
    J_k_hybrid = np.hstack([J_k_phi_num, J_k_v_ana])

    return J_k_hybrid


def single_pass_mfg_posterior_update(
    theta_prior: MFGParameters,
    X_WB_bar: Pose,
    measurements_Y: np.ndarray,
    sensor_poses_WA: List[Pose],
    Sigma_w: np.ndarray,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
    h_rot: float = 1e-6,
    alpha_candidates: Optional[List[float]] = None,
    rot_gain: float = 0.02,
    use_line_search: bool = True,
    verbose_inner=False
) -> MFGParameters:
    """
    对应 Algorithm 1:
    Single-Pass MFG Posterior Update Under a Fixed Nominal Pose
    
    输入:
        theta_prior     : 先验参数 Θ = {F, mu, Lambda, Gamma}
        X_WB_bar        : 固定 nominal pose \bar X_B = (\bar R_B, \bar p_B)
        measurements_Y  : 观测集合 Y = {y_k}_{k=1}^K, shape (K,6)
        sensor_poses_WA : 每个时刻 k 的 X_A,k, 长度为 K 的 Pose 列表
        Sigma_w         : 观测噪声协方差, shape (6,6)
        contact_points_A: frame {A} 中的接触点, shape (N,3)
        field_params    : superquadric field 参数
        stiffness_params: 刚度函数参数
        h_rot           : 数值扰动大小，用于计算旋转部分的数值 Jacobian
        alpha_candidates : 可选的 line search 步长候选列表
        use_line_search  : 是否启用 line search 来选择旋转更新的步长
    输出:
        后验参数 theta_post = {F_post, mu_post, Lambda_post, Gamma_post
    """
        # 这里的实现与之前的版本类似，但在旋转更新部分引入了数值 Jacobian 和可选的 line search 来提升稳定性。
        # 其他部分的结构和之前保持一致。
        # 具体细节请参考之前的版本，以及上面定义的 hybrid_interaction_wrench_jacobian_v2 函数。

    """
    
    说明:
    这段代码严格对应Algorithm 1 的主要结构。
    
    关于 eta / nu_{Rbar}:
    使用
        eta <- Lambda (mu + Gamma nu_{Rbar})
    以及
        mu_post <- Lambda_post^{-1} eta - Gamma_post nu_{Rbar}
    """
    F = ensure_matrix(theta_prior.F, (3, 3))
    mu = ensure_vector(theta_prior.mu, 3)
    Lambda = ensure_matrix(theta_prior.Lambda, (3, 3))
    Gamma = ensure_matrix(theta_prior.Gamma, (3, 3))

    measurements_Y = np.asarray(measurements_Y, dtype=float)
    if measurements_Y.ndim != 2 or measurements_Y.shape[1] != 6:
        raise ValueError("measurements_Y 应为 shape (K,6)")

    K = measurements_Y.shape[0]
    if len(sensor_poses_WA) != K:
        raise ValueError("sensor_poses_WA 的长度必须与 measurements_Y 的行数一致")

    Sigma_w = ensure_matrix(Sigma_w, (6, 6))
    Sigma_w_inv = np.linalg.inv(Sigma_w)

    # 初始化 posterior 参数
    F_post = F.copy()
    Lambda_post = Lambda.copy()
    Gamma_post = Gamma.copy()

    # 根据 Algorithm 1 中的记号，构造 eta
    nu_Rbar = nu_from_rotation(X_WB_bar.R)
    eta = Lambda @ (mu + Gamma @ nu_Rbar)

    # 遍历全部观测
    for k in range(K):
        X_WA_k = sensor_poses_WA[k]
        y_k = ensure_vector(measurements_Y[k], 6)

        # residual:
        # r_k <- y_k - w_A^int(X_A,k, Xbar_B)
        w_pred_k = measurement_model_y(
            X_WA_k,
            X_WB_bar,
            contact_points_A,
            field_params,
            stiffness_params
        )
        r_k = y_k - w_pred_k

        # Step8: Jacobian J_k, 再拆成 J_{k,phi}, J_{k,v}
        J_k = interaction_residual_jacobian( ### 这里直接用解析的 Jacobian 来计算，不需要传入 y_k，因为 interaction_residual_jacobian 内部会计算 r_k 来得到 J_k）
            X_WA_k,
            X_WB_bar,
            contact_points_A,
            field_params,
            stiffness_params,
        )
        J_k_phi, J_k_v = split_jacobian_blocks(J_k)



        # Step 8' : 用Hybrid 版本
        # J_k = hybrid_residual_jacobian_v2(
        #     y_k,
        #     X_WA_k,
        #     X_WB_bar,
        #     contact_points_A,
        #     field_params,
        #     stiffness_params,
        #     h_rot=h_rot
        # )
        # J_k_phi, J_k_v = split_jacobian_blocks(J_k)

        temp_rot = J_k_phi.T @ Sigma_w_inv @ r_k

        temp_rot_norm = np.linalg.norm(temp_rot)
        
        if verbose_inner:
            print(f"[k={k}] ||temp_rot|| = {temp_rot_norm:.6e}")

        max_rot_step = 1.0
        if temp_rot_norm > max_rot_step:
            temp_rot = temp_rot * (max_rot_step / temp_rot_norm)

        #F_post = F_post + 0.5 * rot_gain * X_WB_bar.R @ hat(temp_rot)
        ##### 因为反号的原因， 改一下符号
        F_post = F_post - 0.5 * rot_gain * X_WB_bar.R @ hat(temp_rot)         
        
        # Step 10:
        # Lambda_post <- Lambda_post + J_{k,v}^T Sigma_w^{-1} J_{k,v}
        Lambda_post = Lambda_post + J_k_v.T @ Sigma_w_inv @ J_k_v

        # Step 11:
        # Gamma_post <- Gamma_post - J_{k,phi}^T Sigma_w^{-1} J_{k,v}
        Gamma_post = Gamma_post - J_k_phi.T @ Sigma_w_inv @ J_k_v

        # Step 12:
        # eta <- eta + J_{k,v}^T Sigma_w^{-1} (r_k + J_{k,v} \bar p_B)
        eta = eta + J_k_v.T @ Sigma_w_inv @ (r_k + J_k_v @ X_WB_bar.p)
        
    # Step 14:
    #mu_post = np.linalg.solve(Lambda_post, eta) #- Gamma_post @ nu_Rbar    先不用 Gamma 项，看看数值稳定性
    ### 一次测验的结果，p_gain =15.0 的时候，单轮测试的结果非常好，误差在1cm以内，且数值稳定性也不错。
    #p_gain = 15.0
    #mu_post = p_gain * p_raw
    #p_raw = np.linalg.solve(Lambda_post, eta)


    # 平移增益，先用 line search 结果给出的经验值
    p_raw = np.linalg.solve(Lambda_post, eta)
    R_post_hat = rotation_mode_from_F(F_post)

    def pose_cost(X_test):
        Sigma_w_inv_local = np.linalg.inv(Sigma_w)
        total = 0.0
        for kk, X_WA_kk in enumerate(sensor_poses_WA):
            y_kk = measurements_Y[kk]
            w_pred_kk = measurement_model_y(
                X_WA_kk,
                X_test,
                contact_points_A,
                field_params,
                stiffness_params
            )
            r_kk = y_kk - w_pred_kk
            total += float(r_kk.T @ Sigma_w_inv_local @ r_kk)
        return total
    


#### 增强诊断版本，接下来我们需要整段替换下面这个部分。 
    # if alpha_candidates is None:
    #     alpha_candidates = [1, 2, 4, 6, 8, 10, 12, 15, 18, 20]
    # best_alpha = 1.0
    # best_cost = np.inf
    # best_p = p_raw.copy()

    # for alpha in alpha_candidates:
    #     #p_test = alpha * p_raw 仅仅对于单轮测试没问题
    #     p_test = X_WB_bar.p + alpha * (p_raw - X_WB_bar.p) ## 适用多伦测试的线性插值版本
    #     X_test = Pose(R=R_post_hat.copy(), p=p_test.copy())
    #     # c = pose_cost(
    #     #     X_test, measurements_Y, sensor_poses_WA, Sigma_w,
    #     #     contact_points_A, field_params, stiffness_params
    #     # )
    #     c = pose_cost(X_test)
    #     if c < best_cost:
    #         best_cost = c
    #         best_alpha = alpha
    #         best_p = p_test.copy()
    # mu_post = best_p ### 最终的平移后验参数 step 14
###### 到这里为止。  把旧版删除掉，替换成下面这个增强诊断版本的线性插值版本。

#### 改动开始
    if alpha_candidates is None:
        alpha_candidates = [1, 2, 4, 6, 8, 10, 12, 15, 18, 20]

    # ------------------------------------------------------------
    # Translation diagnostic helper
    # ------------------------------------------------------------
    def point_cost_for_p(p_test):
        X_test = Pose(R=R_post_hat.copy(), p=p_test.copy())
        return pose_cost(X_test)

    def best_on_ray(p_target):
        """
        Ray starting from nominal p_bar = X_WB_bar.p toward p_target:
            p(alpha) = p_bar + alpha * (p_target - p_bar)
        """
        best_alpha_ray = 1.0
        best_cost_ray = np.inf
        best_p_ray = p_target.copy()

        for alpha in alpha_candidates:
            p_test = X_WB_bar.p + alpha * (p_target - X_WB_bar.p)
            c = point_cost_for_p(p_test)
            if c < best_cost_ray:
                best_cost_ray = c
                best_alpha_ray = alpha
                best_p_ray = p_test.copy()

        return best_alpha_ray, best_cost_ray, best_p_ray

    # ------------------------------------------------------------
    # Current empirical branch (this remains the main branch)
    # ------------------------------------------------------------
    best_alpha, best_cost, best_p = best_on_ray(p_raw)

    # ------------------------------------------------------------
    # Coupling-aware diagnostic candidates (do NOT change main output)
    # ------------------------------------------------------------
    mu_theory_bar = p_raw - Gamma_post @ nu_Rbar
    nu_Rmode = nu_from_rotation(R_post_hat)
    p_mode_coupled = mu_theory_bar + Gamma_post @ nu_Rmode

    # consistency check: should reconstruct p_raw at R_bar
    p_bar_reconstructed = mu_theory_bar + Gamma_post @ nu_Rbar
    gap_reconstruct_bar = np.linalg.norm(p_bar_reconstructed - p_raw)

    # point costs at fixed R_post_hat
    cost_nominal_p = point_cost_for_p(X_WB_bar.p)
    cost_p_raw = point_cost_for_p(p_raw)
    cost_mu_theory_bar = point_cost_for_p(mu_theory_bar)
    cost_p_mode_coupled = point_cost_for_p(p_mode_coupled)
    cost_best_p = point_cost_for_p(best_p)

    # ray diagnostics
    alpha_mu_bar, cost_mu_bar_ray, best_p_mu_bar = best_on_ray(mu_theory_bar)
    alpha_mode, cost_mode_ray, best_p_mode = best_on_ray(p_mode_coupled)

    if verbose_inner:
        print("\n===== Alg1 translation line-search diagnostic =====")
        print("R branch fixed at R_post_hat")
        print("R_post_hat =")
        print(R_post_hat)

        print("\n--- coupling magnitudes ---")
        print("||Gamma_post @ nu_Rbar||  =", np.linalg.norm(Gamma_post @ nu_Rbar))
        print("||Gamma_post @ nu_Rmode|| =", np.linalg.norm(Gamma_post @ nu_Rmode))
        print("||nu_Rbar||               =", np.linalg.norm(nu_Rbar))
        print("||nu_Rmode||              =", np.linalg.norm(nu_Rmode))
        print("||p_bar_reconstructed - p_raw|| =", gap_reconstruct_bar)

        print("\n--- candidate points at fixed R_post_hat ---")
        print("X_WB_bar.p      =", X_WB_bar.p)
        print("p_raw           =", p_raw)
        print("mu_theory_bar   =", mu_theory_bar)
        print("p_mode_coupled  =", p_mode_coupled)
        print("best_p(empirical line search) =", best_p)

        print("\n--- point costs (fixed R_post_hat) ---")
        print("cost(nominal p)        =", cost_nominal_p)
        print("cost(p_raw)            =", cost_p_raw)
        print("cost(mu_theory_bar)    =", cost_mu_theory_bar)
        print("cost(p_mode_coupled)   =", cost_p_mode_coupled)
        print("cost(best_p empirical) =", cost_best_p)

        print("\n--- ray comparison from nominal p_bar ---")
        print("[raw ray]")
        print("  best_alpha_raw   =", best_alpha)
        print("  best_cost_raw    =", best_cost)
        print("  best_p_raw_ray   =", best_p)

        print("[mu_theory_bar ray]")
        print("  best_alpha_mu    =", alpha_mu_bar)
        print("  best_cost_mu     =", cost_mu_bar_ray)
        print("  best_p_mu_ray    =", best_p_mu_bar)

        print("[mode-coupled ray]")
        print("  best_alpha_mode  =", alpha_mode)
        print("  best_cost_mode   =", cost_mode_ray)
        print("  best_p_mode_ray  =", best_p_mode)


###### 第一次修改加上的代码
    # ============================================================
    # NEW: blockwise rotation / translation coordination diagnostic
    # (diagnostic only; does NOT change main output)
    # Add this BEFORE: mu_post = best_p
    # ============================================================

    def point_cost_for_pose(R_test, p_test):
        X_test = Pose(R=R_test.copy(), p=p_test.copy())
        return pose_cost(X_test)

    def best_on_rot_ray(p_fixed):
        """
        Interpolate rotation from R_bar to R_post_hat, with p fixed:
            R(alpha) = R_bar Exp(alpha * log(R_bar^T R_post_hat))
        """
        dphi = log_so3(X_WB_bar.R.T @ R_post_hat)

        best_alpha_rot = 0.0
        best_cost_rot = np.inf
        best_R_rot = X_WB_bar.R.copy()

        for alpha in alpha_candidates:
            R_test = X_WB_bar.R @ so3_exp(alpha * dphi)
            c = point_cost_for_pose(R_test, p_fixed)
            if c < best_cost_rot:
                best_cost_rot = c
                best_alpha_rot = alpha
                best_R_rot = R_test.copy()

        return best_alpha_rot, best_cost_rot, best_R_rot, dphi

    def best_on_trans_ray(R_fixed):
        """
        Interpolate translation from p_bar to best_p, with R fixed:
            p(alpha) = p_bar + alpha * (best_p - p_bar)
        """
        dp = best_p - X_WB_bar.p

        best_alpha_trans = 0.0
        best_cost_trans = np.inf
        best_p_trans = X_WB_bar.p.copy()

        for alpha in alpha_candidates:
            p_test = X_WB_bar.p + alpha * dp
            c = point_cost_for_pose(R_fixed, p_test)
            if c < best_cost_trans:
                best_cost_trans = c
                best_alpha_trans = alpha
                best_p_trans = p_test.copy()

        return best_alpha_trans, best_cost_trans, best_p_trans, dp

    # ------------------------------------------------------------
    # Four key poses from nominal pose
    # ------------------------------------------------------------
    X_nominal = Pose(R=X_WB_bar.R.copy(), p=X_WB_bar.p.copy())
    X_rot_only = Pose(R=R_post_hat.copy(), p=X_WB_bar.p.copy())
    X_trans_only = Pose(R=X_WB_bar.R.copy(), p=best_p.copy())
    X_full = Pose(R=R_post_hat.copy(), p=best_p.copy())

    cost_nominal_full = point_cost_for_pose(X_nominal.R, X_nominal.p)
    cost_rot_only = point_cost_for_pose(X_rot_only.R, X_rot_only.p)
    cost_trans_only = point_cost_for_pose(X_trans_only.R, X_trans_only.p)
    cost_full = point_cost_for_pose(X_full.R, X_full.p)

    # ------------------------------------------------------------
    # Rotation ray at two fixed translations
    #   A) p fixed at p_bar
    #   B) p fixed at best_p
    # ------------------------------------------------------------
    alpha_rot_at_pbar, cost_rot_at_pbar, R_best_at_pbar, dphi_alg1 = best_on_rot_ray(X_WB_bar.p)
    alpha_rot_at_bestp, cost_rot_at_bestp, R_best_at_bestp, _ = best_on_rot_ray(best_p)

    # ------------------------------------------------------------
    # Translation ray at two fixed rotations
    #   A) R fixed at R_bar
    #   B) R fixed at R_post_hat
    # ------------------------------------------------------------
    alpha_trans_at_Rbar, cost_trans_at_Rbar, p_best_at_Rbar, dp_alg1 = best_on_trans_ray(X_WB_bar.R)
    alpha_trans_at_Rmode, cost_trans_at_Rmode, p_best_at_Rmode, _ = best_on_trans_ray(R_post_hat)

    if verbose_inner:
        print("\n===== Alg1 rotation / translation coordination diagnostic =====")

        print("\n--- four key poses from nominal ---")
        print("cost(nominal)      =", cost_nominal_full)
        print("cost(rot_only)     =", cost_rot_only)
        print("cost(trans_only)   =", cost_trans_only)
        print("cost(full)         =", cost_full)

        print("\n--- effective step sizes ---")
        print("||dphi_alg1||      =", np.linalg.norm(dphi_alg1))
        print("||dp_alg1||        =", np.linalg.norm(dp_alg1))

        print("\n--- rotation ray comparison ---")
        print("[rotation ray with p fixed at p_bar]")
        print("  best_alpha_rot_at_pbar   =", alpha_rot_at_pbar)
        print("  best_cost_rot_at_pbar    =", cost_rot_at_pbar)

        print("[rotation ray with p fixed at best_p]")
        print("  best_alpha_rot_at_bestp  =", alpha_rot_at_bestp)
        print("  best_cost_rot_at_bestp   =", cost_rot_at_bestp)

        print("\n--- translation ray comparison ---")
        print("[translation ray with R fixed at R_bar]")
        print("  best_alpha_trans_at_Rbar  =", alpha_trans_at_Rbar)
        print("  best_cost_trans_at_Rbar   =", cost_trans_at_Rbar)

        print("[translation ray with R fixed at R_post_hat]")
        print("  best_alpha_trans_at_Rmode =", alpha_trans_at_Rmode)
        print("  best_cost_trans_at_Rmode  =", cost_trans_at_Rmode)

        print("\n--- interpretation hints ---")
        print("If cost(rot_only) gets worse but cost(trans_only) gets much better,")
        print("then translation is currently doing most of the work.")
        print("If rotation ray improves only when p is fixed at best_p,")
        print("then rotation and translation are strongly coordinated.")
        print("If translation ray is much better at R_bar than at R_post_hat,")
        print("then the new rotation is hurting the translation branch.")

### 第二次修改地方
    # ============================================================
    # NEW: fine rotation alpha scan at fixed translations
    # (diagnostic only; does NOT change main output)
    # Add this AFTER the current coordination diagnostic block
    # and BEFORE: mu_post = best_p
    # ============================================================

    rot_alpha_fine = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    def scan_rot_alpha_with_fixed_p(p_fixed):
        dphi = log_so3(X_WB_bar.R.T @ R_post_hat)

        alpha_list = []
        cost_list = []

        best_alpha_local = None
        best_cost_local = np.inf

        for alpha in rot_alpha_fine:
            R_test = X_WB_bar.R @ so3_exp(alpha * dphi)
            c = point_cost_for_pose(R_test, p_fixed)

            alpha_list.append(alpha)
            cost_list.append(c)

            if c < best_cost_local:
                best_cost_local = c
                best_alpha_local = alpha

        return np.array(alpha_list), np.array(cost_list), best_alpha_local, best_cost_local

    rot_scan_alpha_pbar, rot_scan_cost_pbar, best_alpha_scan_pbar, best_cost_scan_pbar = \
        scan_rot_alpha_with_fixed_p(X_WB_bar.p)

    rot_scan_alpha_bestp, rot_scan_cost_bestp, best_alpha_scan_bestp, best_cost_scan_bestp = \
        scan_rot_alpha_with_fixed_p(best_p)

    if verbose_inner:
        print("\n===== Alg1 fine rotation-alpha scan =====")

        print("\n[fixed p = p_bar]")
        for a, c in zip(rot_scan_alpha_pbar, rot_scan_cost_pbar):
            print(f"  alpha={a:>4.2f}, cost={c}")
        print("  best_alpha_scan_pbar =", best_alpha_scan_pbar)
        print("  best_cost_scan_pbar  =", best_cost_scan_pbar)

        print("\n[fixed p = best_p]")
        for a, c in zip(rot_scan_alpha_bestp, rot_scan_cost_bestp):
            print(f"  alpha={a:>4.2f}, cost={c}")
        print("  best_alpha_scan_bestp =", best_alpha_scan_bestp)
        print("  best_cost_scan_bestp  =", best_cost_scan_bestp)



    # ============================================================
    # NEW: rotation safeguard at fixed best_p
    # This DOES change the main output rotation branch.
    # Add this AFTER the fine rotation-alpha scan block
    # and BEFORE: mu_post = best_p
    # ============================================================

    # keep a copy of the raw rotation mode before safeguard
    R_post_hat_raw = R_post_hat.copy()

    # use a fine alpha list for the actual safeguard
    rot_alpha_safe_candidates = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    dphi_post_raw = log_so3(X_WB_bar.R.T @ R_post_hat_raw)

    best_alpha_rot_safe = 0.0
    best_cost_rot_safe = np.inf
    R_post_hat_safe = X_WB_bar.R.copy()

    for alpha_rot in rot_alpha_safe_candidates:
        R_test = X_WB_bar.R @ so3_exp(alpha_rot * dphi_post_raw)
        X_test = Pose(R=R_test.copy(), p=best_p.copy())
        c = pose_cost(X_test)

        if c < best_cost_rot_safe:
            best_cost_rot_safe = c
            best_alpha_rot_safe = alpha_rot
            R_post_hat_safe = R_test.copy()

    # overwrite the recovered rotation mode with safeguarded one
    R_post_hat = R_post_hat_safe

    # ------------------------------------------------------------
    # IMPORTANT:
    # also rewrite F_post so that its mode becomes the safeguarded
    # R_post_hat; otherwise recover_pose_from_theta(theta_post)
    # will still recover the OLD unsafe rotation.
    # ------------------------------------------------------------
    U_F, s_F, Vt_F = np.linalg.svd(F_post)
    V_F = Vt_F.T
    D_F = np.diag([1.0, 1.0, np.linalg.det(U_F @ Vt_F)])

    # Want mode(F_post_new) = R_post_hat
    # If old mode was U_F D_F Vt_F, then choose
    # U_new = R_post_hat V_F D_F
    U_new = R_post_hat @ V_F @ D_F
    F_post = U_new @ np.diag(s_F) @ Vt_F

    if verbose_inner:
        print("\n===== Alg1 rotation safeguard at fixed best_p =====")
        print("raw R_post_hat =")
        print(R_post_hat_raw)
        print("safe R_post_hat =")
        print(R_post_hat)

        print("||dphi_post_raw||        =", np.linalg.norm(dphi_post_raw))
        print("chosen alpha_rot_safe    =", best_alpha_rot_safe)
        print("best cost at fixed best_p =", best_cost_rot_safe)

        X_raw_full = Pose(R=R_post_hat_raw.copy(), p=best_p.copy())
        X_safe_full = Pose(R=R_post_hat.copy(), p=best_p.copy())
        print("cost(raw full)           =", pose_cost(X_raw_full))
        print("cost(safe full)          =", pose_cost(X_safe_full))

        # optional consistency check
        R_check = rotation_mode_from_F(F_post)
        gap_mode = np.linalg.norm(log_so3(R_check.T @ R_post_hat))
        print("mode(F_post) vs safe R gap =", gap_mode)


    # keep the current main branch unchanged
    mu_post = best_p

#### 以上是改动的版本



    print("||eta|| =", np.linalg.norm(eta))
    print("||solve(Lambda_post, eta)|| =", np.linalg.norm(np.linalg.solve(Lambda_post, eta)))
    print("||Gamma_post @ nu_Rbar|| =", np.linalg.norm(Gamma_post @ nu_Rbar))

    ### 诊断数值结果，看看线性解 p_raw 与 line search 优化后的 best_p 的差异，以及它们与 nominal pose 的关系 ###
    print("p_raw =", p_raw)
    print("||p_raw|| =", np.linalg.norm(p_raw))
    print("alpha_candidates =", alpha_candidates)
    print("best_alpha =", best_alpha)
    print("best_p =", best_p)
    print("||best_p - X_WB_bar.p|| =", np.linalg.norm(best_p - X_WB_bar.p))



    theta_post = MFGParameters(
        F=F_post,
        mu=mu_post,
        Lambda=Lambda_post,
        Gamma=Gamma_post
    )
    return theta_post


def batch_residual_norm(
    X_WB: Pose,
    measurements_Y: np.ndarray,
    sensor_poses_WA,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
) -> float:
    """
    Compute the batch residual norm:

        ||r||_2 = sqrt( sum_k || y_k - w_int,A(X_WA_k, X_WB) ||^2 )
    """
    measurements_Y = np.asarray(measurements_Y, dtype=float)
    if measurements_Y.ndim != 2 or measurements_Y.shape[1] != 6:
        raise ValueError("measurements_Y 应为 shape (K,6)")

    sq = 0.0
    K = measurements_Y.shape[0]

    for k in range(K):
        y_k = ensure_vector(measurements_Y[k], 6)
        X_WA_k = sensor_poses_WA[k]

        w_pred = interaction_wrench_A(
            X_WA_k,
            X_WB,
            contact_points_A,
            field_params,
            stiffness_params,
        )

        r_k = y_k - w_pred
        sq += float(r_k @ r_k)

    return np.sqrt(sq)


def batch_whitened_residual_norm(
    X_WB: Pose,
    measurements_Y: np.ndarray,
    sensor_poses_WA,
    Sigma_w: np.ndarray,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
) -> float:
    """
    Compute the whitened batch residual norm:

        sqrt( sum_k r_k^T Sigma_w^{-1} r_k )

    where
        r_k = y_k - w_int,A(X_WA_k, X_WB).
    """
    measurements_Y = np.asarray(measurements_Y, dtype=float)
    Sigma_w = ensure_matrix(Sigma_w, (6, 6))

    if measurements_Y.ndim != 2 or measurements_Y.shape[1] != 6:
        raise ValueError("measurements_Y 应为 shape (K,6)")

    Sigma_w_inv = np.linalg.inv(Sigma_w)

    quad = 0.0
    K = measurements_Y.shape[0]

    for k in range(K):
        y_k = ensure_vector(measurements_Y[k], 6)
        X_WA_k = sensor_poses_WA[k]

        w_pred = interaction_wrench_A(
            X_WA_k,
            X_WB,
            contact_points_A,
            field_params,
            stiffness_params,
        )

        r_k = y_k - w_pred
        quad += float(r_k @ Sigma_w_inv @ r_k)

    return np.sqrt(quad)


def relaxed_pose_update(
    X_t: Pose,
    X_candidate: Pose,
    rot_gain: float,
    pos_gain: float,
) -> Pose:
    """
    Outer relaxed interpolation from current pose X_t to candidate pose X_candidate.

    R_next = R_t @ Exp(rot_gain * Log(R_t^T R_candidate))
    p_next = p_t + pos_gain * (p_candidate - p_t)

    rot_gain, pos_gain should lie in [0, 1].
    """
    if not (0.0 <= rot_gain <= 1.0):
        raise ValueError("rot_gain 应满足 0 <= rot_gain <= 1")
    if not (0.0 <= pos_gain <= 1.0):
        raise ValueError("pos_gain 应满足 0 <= pos_gain <= 1")

    dR_vec = log_so3(X_t.R.T @ X_candidate.R)
    R_next = X_t.R @ so3_exp(rot_gain * dR_vec)
    p_next = X_t.p + pos_gain * (X_candidate.p - X_t.p)

    return Pose(R=R_next, p=p_next.copy())


def build_numeric_rot_only_candidate(
    X_t: Pose,
    R_mode: np.ndarray,
    measurements_Y: np.ndarray,
    sensor_poses_WA: List[Pose],
    Sigma_w: np.ndarray,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    h_rot_numgrad: float = 1e-5,
    rot_goal_step_candidates: Optional[List[float]] = None,
    verbose: bool = False,
):
    """
    Build a rotation-only candidate at fixed p = X_t.p.

    Compare two fixed-p directions:
      1) mode direction from recovered R_mode
      2) negative numerical gradient direction of 0.5 * ||r_white||^2

    Return:
      X_rot_goal: best rotation-only goal pose
      diag: diagnostics for logging / history
    """

    def white_of_phi(phi: np.ndarray) -> float:
        X_eval = Pose(R=X_t.R @ so3_exp(phi), p=X_t.p.copy())
        return batch_whitened_residual_norm(
            X_eval,
            measurements_Y,
            sensor_poses_WA,
            Sigma_w,
            contact_points_A,
            field_params,
            stiffness_params,
        )

    def obj_of_phi(phi: np.ndarray) -> float:
        w = white_of_phi(phi)
        return 0.5 * (w ** 2)

    current_white = white_of_phi(np.zeros(3))

    # mode direction from current R_t to recovered R_mode
    dphi_mode = log_so3(X_t.R.T @ R_mode)
    mode_step = np.linalg.norm(dphi_mode)

    # numerical gradient at fixed p
    g_num = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        fp = obj_of_phi(h_rot_numgrad * e)
        fm = obj_of_phi(-h_rot_numgrad * e)
        g_num[i] = (fp - fm) / (2.0 * h_rot_numgrad)

    numgrad_norm = np.linalg.norm(g_num)

    # choose scan step candidates (radians)
    if rot_goal_step_candidates is None:
        base = max(mode_step, 2e-2)  # at least about 1.15 degree
        rot_goal_step_candidates = sorted({
            float(np.clip(c * base, 1e-4, 3e-1))
            for c in [0.5, 1.0, 1.5, 2.0]
        })
        if mode_step > 1e-14:
            rot_goal_step_candidates = sorted(
                set(rot_goal_step_candidates + [float(np.clip(mode_step, 1e-4, 3e-1))])
            )

    directions = []
    if mode_step > 1e-14:
        directions.append(("mode", dphi_mode / mode_step))
    if numgrad_norm > 1e-14:
        directions.append(("numgrad", -g_num / numgrad_norm))

    best = {
        "source": "identity",
        "white": current_white,
        "step": 0.0,
        "phi": np.zeros(3),
    }

    scan_log = []

    for name, u in directions:
        for step in rot_goal_step_candidates:
            phi = step * u
            w = white_of_phi(phi)
            scan_log.append((name, step, w))

            if w < best["white"]:
                best = {
                    "source": name,
                    "white": w,
                    "step": step,
                    "phi": phi.copy(),
                }

    X_rot_goal = Pose(
        R=X_t.R @ so3_exp(best["phi"]),
        p=X_t.p.copy(),
    )

    diag = {
        "current_white": current_white,
        "mode_step": mode_step,
        "numgrad_norm": numgrad_norm,
        "best_source": best["source"],
        "best_step": best["step"],
        "best_white": best["white"],
        "scan_log": scan_log,
        "g_num": g_num.copy(),
        "dphi_mode": dphi_mode.copy(),
    }

    if verbose:
        print("[rot-candidate builder]")
        print(f"  current_white = {current_white:.6e}")
        print(f"  mode_step     = {mode_step:.6e}")
        print(f"  ||g_num||     = {numgrad_norm:.6e}")
        print(f"  best_source   = {best['source']}")
        print(f"  best_step     = {best['step']:.6e}")
        print(f"  best_white    = {best['white']:.6e}")
        for name, step, w in scan_log:
            print(f"    [{name}] step={step:.6e}, white={w:.6e}")

    return X_rot_goal, diag


def multi_pass_mfg_batch_refinement(
    theta0: MFGParameters,
    X_WB0: Pose,
    measurements_Y: np.ndarray,
    sensor_poses_WA: List[Pose],
    Sigma_w: np.ndarray,
    contact_points_A: np.ndarray,
    field_params: SuperquadricFieldParameters,
    stiffness_params: StiffnessParameters,
    Tmax: int = 10,
    h_rot: float = 1e-6,
    use_line_search: bool = True,
    rot_gain: float = 1.0,                    # 保留接口兼容；outer 不再额外用它缩步
    pos_gain: Optional[float] = None,         # 保留接口兼容；outer 不再额外用它缩步
    eps_R: float = 1e-6,
    eps_p: float = 1e-6,
    alpha_candidates: Optional[List[float]] = None,         # 给 Algorithm 1 内部用
    outer_alpha_candidates: Optional[List[float]] = None,   # outer safeguarded search
    use_numeric_rot_candidate: bool = True,
    h_rot_numgrad: float = 1e-5,
    rot_goal_step_candidates: Optional[List[float]] = None,
    accept_tol: float = 1e-12,
    verbose: bool = True,
    verbose_inner: bool = False,
    return_history: bool = True,
    X_WB_true: Optional[Pose] = None,
):
    """
    Safeguarded Algorithm 2:
    fixed prior + block-coordinate outer refinement

    Core ideas:
    1) run one single-pass posterior update at current nominal X_t;
    2) build three goal poses: full / trans_only / rot_only;
    3) for each branch, run safeguarded relaxed outer search;
    4) accept the branch-step with the best decreased whitened residual.
    """

    if Tmax <= 0:
        raise ValueError("Tmax 必须为正整数")
    if eps_R < 0 or eps_p < 0:
        raise ValueError("eps_R, eps_p 必须非负")
    if accept_tol < 0:
        raise ValueError("accept_tol 必须非负")

    # 兼容旧接口
    if pos_gain is None:
        pos_gain = 1.0

    if outer_alpha_candidates is None:
        outer_alpha_candidates = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]

    # 只保留 (0,1) 内的 relaxed alpha；alpha=1 的 full goal 单独先试
    relaxed_alphas = sorted(
        {float(a) for a in outer_alpha_candidates if 0.0 < float(a) < 1.0},
        reverse=True
    )

    # ------------------------------------------------
    # fixed prior
    # ------------------------------------------------
    theta_prior_fixed = MFGParameters(
        F=theta0.F.copy(),
        mu=theta0.mu.copy(),
        Lambda=theta0.Lambda.copy(),
        Gamma=theta0.Gamma.copy(),
    )

    # ------------------------------------------------
    # current nominal pose
    # ------------------------------------------------
    X_t = Pose(R=X_WB0.R.copy(), p=X_WB0.p.copy())

    current_res_norm = batch_residual_norm(
        X_t,
        measurements_Y,
        sensor_poses_WA,
        contact_points_A,
        field_params,
        stiffness_params,
    )

    current_res_norm_white = batch_whitened_residual_norm(
        X_t,
        measurements_Y,
        sensor_poses_WA,
        Sigma_w,
        contact_points_A,
        field_params,
        stiffness_params,
    )

    # ------------------------------------------------
    # best iterate init
    # ------------------------------------------------
    best_X = Pose(R=X_t.R.copy(), p=X_t.p.copy())
    best_res_norm = current_res_norm
    best_res_norm_white = current_res_norm_white
    best_iter = -1

    best_theta = single_pass_mfg_posterior_update(
        theta_prior=theta_prior_fixed,
        X_WB_bar=best_X,
        measurements_Y=measurements_Y,
        sensor_poses_WA=sensor_poses_WA,
        Sigma_w=Sigma_w,
        contact_points_A=contact_points_A,
        field_params=field_params,
        stiffness_params=stiffness_params,
        h_rot=h_rot,
        alpha_candidates=alpha_candidates,
        use_line_search=use_line_search,
        verbose_inner=False,
    )

    # ------------------------------------------------
    # history
    # ------------------------------------------------
    history = {
        "iter": [],
        "accepted": [],
        "chosen_alpha": [],
        "delta_R": [],
        "delta_p": [],
        "res_norm": [],
        "res_norm_white": [],
        "rot_err": [],
        "pos_err": [],
        "p_x": [],
        "p_y": [],
        "p_z": [],
        "cand_p_x": [],
        "cand_p_y": [],
        "cand_p_z": [],
        "best_res_norm_white": [],
        "candidate_rot_step": [],         # compatibility: full-goal dR
        "relaxed_rot_step": [],           # accepted dR
        "candidate_pos_step": [],         # compatibility: full-goal dp
        "relaxed_pos_step": [],           # accepted dp
        "current_res_norm_white_before": [],
        "trial_res_norm_white_after": [],
        "X_t_p": [],
        "X_candidate_p": [],
        "X_next_p": [],
        "candidate_res_norm_white": [],   # compatibility: full-goal white
        "candidate_res_norm": [],         # compatibility: full-goal residual
        "full_candidate_accepted": [],

        "selected_branch": [],

        "cand_full_res_norm": [],
        "cand_full_res_norm_white": [],
        "cand_full_rot_step": [],
        "cand_full_pos_step": [],

        "cand_rot_res_norm": [],
        "cand_rot_res_norm_white": [],
        "cand_rot_step": [],
        "cand_rot_pos_step": [],

        "cand_trans_res_norm": [],
        "cand_trans_res_norm_white": [],
        "cand_trans_rot_step": [],
        "cand_trans_pos_step": [],

        "searched_best_branch": [],
        "searched_best_white": [],

        "rot_candidate_source": [],
        "rot_candidate_goal_step": [],
    }

    if verbose:
        print("===== Safeguarded Algorithm 2 (block-coordinate version) =====")
        print(f"initial residual norm      = {current_res_norm:.6e}")
        print(f"initial whitened res norm  = {current_res_norm_white:.6e}")
        print("[note] outer 步长只由 outer_alpha_candidates 控制；")
        print("       rot_gain / pos_gain 仅保留接口兼容，不再参与 outer 双重缩步。")

    # ========================================================
    # outer loop
    # ========================================================
    for t in range(Tmax):
        # --------------------------------------------
        # Step 1: single-pass posterior update (Algorithm 1)
        # --------------------------------------------
        theta_post_t = single_pass_mfg_posterior_update(
            theta_prior=theta_prior_fixed,
            X_WB_bar=X_t,
            measurements_Y=measurements_Y,
            sensor_poses_WA=sensor_poses_WA,
            Sigma_w=Sigma_w,
            contact_points_A=contact_points_A,
            field_params=field_params,
            stiffness_params=stiffness_params,
            h_rot=h_rot,
            alpha_candidates=alpha_candidates,
            use_line_search=use_line_search,
            verbose_inner=verbose_inner,
        )

        # --------------------------------------------
        # Step 2: build three branch goal poses
        # --------------------------------------------
        X_candidate_full = recover_pose_from_theta(theta_post_t)

        # translation-only: keep current R, only move p
        X_candidate_trans = Pose(
            R=X_t.R.copy(),
            p=X_candidate_full.p.copy(),
        )

        # rotation-only: keep current p, only move R
        if use_numeric_rot_candidate:
            X_candidate_rot, rot_builder_diag = build_numeric_rot_only_candidate(
                X_t=X_t,
                R_mode=X_candidate_full.R,
                measurements_Y=measurements_Y,
                sensor_poses_WA=sensor_poses_WA,
                Sigma_w=Sigma_w,
                contact_points_A=contact_points_A,
                field_params=field_params,
                stiffness_params=stiffness_params,
                h_rot_numgrad=h_rot_numgrad,
                rot_goal_step_candidates=rot_goal_step_candidates,
                verbose=False,
            )
        else:
            X_candidate_rot = Pose(
                R=X_candidate_full.R.copy(),
                p=X_t.p.copy(),
            )
            rot_builder_diag = None

        def eval_pose(X_eval: Pose):
            res = batch_residual_norm(
                X_eval,
                measurements_Y,
                sensor_poses_WA,
                contact_points_A,
                field_params,
                stiffness_params,
            )
            res_white = batch_whitened_residual_norm(
                X_eval,
                measurements_Y,
                sensor_poses_WA,
                Sigma_w,
                contact_points_A,
                field_params,
                stiffness_params,
            )
            dR = np.linalg.norm(log_so3(X_t.R.T @ X_eval.R))
            dp = np.linalg.norm(X_eval.p - X_t.p)
            return res, res_white, dR, dp

        cand_full_res, cand_full_white, cand_full_dR, cand_full_dp = eval_pose(X_candidate_full)
        cand_trans_res, cand_trans_white, cand_trans_dR, cand_trans_dp = eval_pose(X_candidate_trans)
        cand_rot_res, cand_rot_white, cand_rot_dR, cand_rot_dp = eval_pose(X_candidate_rot)

        if verbose:
            print(f"\n[outer iter {t}]")
            print(f"  current white residual = {current_res_norm_white:.6e}")
            print(f"  [full]  white={cand_full_white:.6e},  dR={cand_full_dR:.6e}, dp={cand_full_dp:.6e}")
            print(f"  [trans] white={cand_trans_white:.6e}, dR={cand_trans_dR:.6e}, dp={cand_trans_dp:.6e}")
            print(f"  [rot]   white={cand_rot_white:.6e},   dR={cand_rot_dR:.6e}, dp={cand_rot_dp:.6e}")
            print(f"  X_t.p              = {X_t.p}")
            print(f"  X_candidate_full.p = {X_candidate_full.p}")
            print(f"  theta_post_t.mu    = {theta_post_t.mu}")
            if rot_builder_diag is not None:
                print("  [rot-candidate builder]")
                print(f"    current_white = {rot_builder_diag['current_white']:.6e}")
                print(f"    mode_step     = {rot_builder_diag['mode_step']:.6e}")
                print(f"    ||g_num||     = {rot_builder_diag['numgrad_norm']:.6e}")
                print(f"    best_source   = {rot_builder_diag['best_source']}")
                print(f"    best_step     = {rot_builder_diag['best_step']:.6e}")
                print(f"    best_white    = {rot_builder_diag['best_white']:.6e}")

        # Keep old compatibility fields tied to the FULL recovered candidate
        candidate_rot_step = cand_full_dR
        candidate_pos_step = cand_full_dp
        cand_res_norm = cand_full_res
        cand_res_norm_white = cand_full_white

        # Save current white BEFORE any acceptance
        current_white_before = current_res_norm_white

        # --------------------------------------------
        # Step 3: safeguarded branch search
        # --------------------------------------------
        def safeguarded_branch_search(branch_name: str, X_goal: Pose):
            goal_res, goal_white, goal_dR, goal_dp = eval_pose(X_goal)

            out = {
                "branch": branch_name,
                "goal_X": X_goal,
                "goal_res": goal_res,
                "goal_white": goal_white,
                "goal_dR": goal_dR,
                "goal_dp": goal_dp,

                "accepted": False,
                "full_candidate_accepted": False,
                "chosen_alpha": np.nan,

                "X_accept": None,
                "res_accept": None,
                "white_accept": None,
                "dR_accept": None,
                "dp_accept": None,

                "searched_best_white": goal_white,
                "searched_best_X": X_goal,
                "searched_best_dR": goal_dR,
                "searched_best_dp": goal_dp,
            }

            # First try the goal pose itself (alpha = 1)
            if goal_white < current_res_norm_white - accept_tol:
                out["accepted"] = True
                out["full_candidate_accepted"] = True
                out["chosen_alpha"] = 1.0
                out["X_accept"] = X_goal
                out["res_accept"] = goal_res
                out["white_accept"] = goal_white
                out["dR_accept"] = goal_dR
                out["dp_accept"] = goal_dp

            # Then try relaxed alphas
            for alpha in relaxed_alphas:
                X_trial = relaxed_pose_update(
                    X_t=X_t,
                    X_candidate=X_goal,
                    rot_gain=alpha,
                    pos_gain=alpha,
                )

                trial_res, trial_white, trial_dR, trial_dp = eval_pose(X_trial)

                if trial_white < out["searched_best_white"]:
                    out["searched_best_white"] = trial_white
                    out["searched_best_X"] = X_trial
                    out["searched_best_dR"] = trial_dR
                    out["searched_best_dp"] = trial_dp

                if verbose:
                    print(
                        f"    [{branch_name}] alpha={alpha:.3f}, "
                        f"trial_white={trial_white:.6e}, "
                        f"dR={trial_dR:.6e}, dp={trial_dp:.6e}"
                    )

                if trial_white < current_res_norm_white - accept_tol:
                    if (not out["accepted"]) or (trial_white < out["white_accept"]):
                        out["accepted"] = True
                        out["full_candidate_accepted"] = False
                        out["chosen_alpha"] = alpha
                        out["X_accept"] = X_trial
                        out["res_accept"] = trial_res
                        out["white_accept"] = trial_white
                        out["dR_accept"] = trial_dR
                        out["dp_accept"] = trial_dp

            return out

        trial_full = safeguarded_branch_search("full", X_candidate_full)
        trial_trans = safeguarded_branch_search("trans_only", X_candidate_trans)
        trial_rot = safeguarded_branch_search("rot_only", X_candidate_rot)

        all_trials = [trial_full, trial_trans, trial_rot]
        accepted_trials = [tr for tr in all_trials if tr["accepted"]]
        searched_best_trial = min(all_trials, key=lambda tr: tr["searched_best_white"])

        accepted = len(accepted_trials) > 0
        chosen_alpha = np.nan
        full_candidate_accepted = False
        selected_branch = None

        best_trial_X = None
        best_trial_res_norm = None
        best_trial_res_norm_white = None
        best_trial_delta_R = None
        best_trial_delta_p = None
        chosen_goal_X = None

        if accepted:
            chosen_trial = min(accepted_trials, key=lambda tr: tr["white_accept"])

            selected_branch = chosen_trial["branch"]
            chosen_alpha = chosen_trial["chosen_alpha"]
            full_candidate_accepted = chosen_trial["full_candidate_accepted"]

            best_trial_X = chosen_trial["X_accept"]
            best_trial_res_norm = chosen_trial["res_accept"]
            best_trial_res_norm_white = chosen_trial["white_accept"]
            best_trial_delta_R = chosen_trial["dR_accept"]
            best_trial_delta_p = chosen_trial["dp_accept"]
            chosen_goal_X = chosen_trial["goal_X"]

            if verbose:
                print(
                    f"  choose branch = {selected_branch}, "
                    f"alpha = {chosen_alpha}, "
                    f"white = {best_trial_res_norm_white:.6e}"
                )
        else:
            if verbose:
                print("  no branch gives acceptable decrease.")

        # --------------------------------------------
        # If no acceptable outer step, stop
        # --------------------------------------------
        if not accepted:
            if verbose:
                print(f"[outer iter {t}] no acceptable outer step found.")
                print(f"    current whitened res norm = {current_res_norm_white:.6e}")
                print(f"    best searched branch      = {searched_best_trial['branch']}")
                print(f"    best searched white       = {searched_best_trial['searched_best_white']:.6e}")
                print("    stop outer refinement and return best iterate.")

            history["iter"].append(t)
            history["accepted"].append(False)
            history["chosen_alpha"].append(np.nan)
            history["delta_R"].append(0.0)
            history["delta_p"].append(0.0)
            history["res_norm"].append(current_res_norm)
            history["res_norm_white"].append(current_res_norm_white)
            history["best_res_norm_white"].append(best_res_norm_white)

            # compatibility fields: keep them tied to FULL candidate
            history["candidate_rot_step"].append(cand_full_dR)
            history["relaxed_rot_step"].append(0.0)
            history["candidate_pos_step"].append(cand_full_dp)
            history["relaxed_pos_step"].append(0.0)
            history["current_res_norm_white_before"].append(current_res_norm_white)
            history["trial_res_norm_white_after"].append(searched_best_trial["searched_best_white"])
            history["candidate_res_norm_white"].append(cand_full_white)
            history["candidate_res_norm"].append(cand_full_res)
            history["full_candidate_accepted"].append(False)

            history["X_t_p"].append(X_t.p.copy())
            history["X_candidate_p"].append(X_candidate_full.p.copy())
            history["X_next_p"].append(X_t.p.copy())

            history["p_x"].append(X_t.p[0])
            history["p_y"].append(X_t.p[1])
            history["p_z"].append(X_t.p[2])

            history["cand_p_x"].append(X_candidate_full.p[0])
            history["cand_p_y"].append(X_candidate_full.p[1])
            history["cand_p_z"].append(X_candidate_full.p[2])

            history["selected_branch"].append("none")

            history["cand_full_res_norm"].append(cand_full_res)
            history["cand_full_res_norm_white"].append(cand_full_white)
            history["cand_full_rot_step"].append(cand_full_dR)
            history["cand_full_pos_step"].append(cand_full_dp)

            history["cand_trans_res_norm"].append(cand_trans_res)
            history["cand_trans_res_norm_white"].append(cand_trans_white)
            history["cand_trans_rot_step"].append(cand_trans_dR)
            history["cand_trans_pos_step"].append(cand_trans_dp)

            history["cand_rot_res_norm"].append(cand_rot_res)
            history["cand_rot_res_norm_white"].append(cand_rot_white)
            history["cand_rot_step"].append(cand_rot_dR)
            history["cand_rot_pos_step"].append(cand_rot_dp)

            history["searched_best_branch"].append(searched_best_trial["branch"])
            history["searched_best_white"].append(searched_best_trial["searched_best_white"])
            history["rot_candidate_source"].append(
                "none" if rot_builder_diag is None else rot_builder_diag["best_source"]
            )
            history["rot_candidate_goal_step"].append(
                0.0 if rot_builder_diag is None else rot_builder_diag["best_step"]
            )

            if X_WB_true is not None:
                rot_err = np.linalg.norm(log_so3(X_WB_true.R.T @ X_t.R))
                pos_err = np.linalg.norm(X_WB_true.p - X_t.p)
            else:
                rot_err = np.nan
                pos_err = np.nan

            history["rot_err"].append(rot_err)
            history["pos_err"].append(pos_err)

            if return_history:
                return best_theta, best_X, history
            return best_theta, best_X

        # --------------------------------------------
        # Accept this iteration
        # --------------------------------------------
        X_candidate = chosen_goal_X   # for compatibility with old plotting / logging
        X_next = best_trial_X
        delta_R = best_trial_delta_R
        delta_p = best_trial_delta_p
        res_norm = best_trial_res_norm
        res_norm_white = best_trial_res_norm_white

        if X_WB_true is not None:
            rot_err = np.linalg.norm(log_so3(X_WB_true.R.T @ X_next.R))
            pos_err = np.linalg.norm(X_WB_true.p - X_next.p)
        else:
            rot_err = np.nan
            pos_err = np.nan

        # update current baseline
        current_res_norm = res_norm
        current_res_norm_white = res_norm_white

        # update best iterate
        if res_norm_white < best_res_norm_white - accept_tol:
            best_res_norm_white = res_norm_white
            best_res_norm = res_norm
            best_X = Pose(R=X_next.R.copy(), p=X_next.p.copy())
            best_iter = t

            best_theta = single_pass_mfg_posterior_update(
                theta_prior=theta_prior_fixed,
                X_WB_bar=best_X,
                measurements_Y=measurements_Y,
                sensor_poses_WA=sensor_poses_WA,
                Sigma_w=Sigma_w,
                contact_points_A=contact_points_A,
                field_params=field_params,
                stiffness_params=stiffness_params,
                h_rot=h_rot,
                alpha_candidates=alpha_candidates,
                use_line_search=use_line_search,
                verbose_inner=False,
            )

        # record history
        history["iter"].append(t)
        history["accepted"].append(True)
        history["chosen_alpha"].append(chosen_alpha)
        history["delta_R"].append(delta_R)
        history["delta_p"].append(delta_p)
        history["res_norm"].append(res_norm)
        history["res_norm_white"].append(res_norm_white)
        history["rot_err"].append(rot_err)
        history["pos_err"].append(pos_err)

        history["p_x"].append(X_next.p[0])
        history["p_y"].append(X_next.p[1])
        history["p_z"].append(X_next.p[2])

        history["cand_p_x"].append(X_candidate.p[0])
        history["cand_p_y"].append(X_candidate.p[1])
        history["cand_p_z"].append(X_candidate.p[2])

        history["best_res_norm_white"].append(best_res_norm_white)

        # compatibility fields: keep them tied to FULL candidate
        history["candidate_rot_step"].append(cand_full_dR)
        history["relaxed_rot_step"].append(delta_R)
        history["candidate_pos_step"].append(cand_full_dp)
        history["relaxed_pos_step"].append(delta_p)

        history["current_res_norm_white_before"].append(current_white_before)
        history["trial_res_norm_white_after"].append(res_norm_white)

        history["candidate_res_norm_white"].append(cand_full_white)
        history["candidate_res_norm"].append(cand_full_res)
        history["full_candidate_accepted"].append(full_candidate_accepted)

        history["X_t_p"].append(X_t.p.copy())
        history["X_candidate_p"].append(X_candidate.p.copy())
        history["X_next_p"].append(X_next.p.copy())

        history["selected_branch"].append(selected_branch)

        history["cand_full_res_norm"].append(cand_full_res)
        history["cand_full_res_norm_white"].append(cand_full_white)
        history["cand_full_rot_step"].append(cand_full_dR)
        history["cand_full_pos_step"].append(cand_full_dp)

        history["cand_trans_res_norm"].append(cand_trans_res)
        history["cand_trans_res_norm_white"].append(cand_trans_white)
        history["cand_trans_rot_step"].append(cand_trans_dR)
        history["cand_trans_pos_step"].append(cand_trans_dp)

        history["cand_rot_res_norm"].append(cand_rot_res)
        history["cand_rot_res_norm_white"].append(cand_rot_white)
        history["cand_rot_step"].append(cand_rot_dR)
        history["cand_rot_pos_step"].append(cand_rot_dp)

        history["searched_best_branch"].append(selected_branch)
        history["searched_best_white"].append(res_norm_white)
        history["rot_candidate_source"].append(
            "none" if rot_builder_diag is None else rot_builder_diag["best_source"]
        )
        history["rot_candidate_goal_step"].append(
            0.0 if rot_builder_diag is None else rot_builder_diag["best_step"]
        )

        if verbose:
            print(f"[outer iter {t}] accepted")
            print(f"    selected_branch      = {selected_branch}")
            print(f"    chosen_alpha         = {chosen_alpha}")
            print(f"    delta_R              = {delta_R:.6e}")
            print(f"    delta_p              = {delta_p:.6e}")
            print(f"    X_t.p                = {X_t.p}")
            print(f"    X_candidate.p        = {X_candidate.p}")
            print(f"    X_next.p             = {X_next.p}")
            print(f"    residual norm        = {res_norm:.6e}")
            print(f"    whitened res norm    = {res_norm_white:.6e}")
            if X_WB_true is not None:
                print(f"    rot_err              = {rot_err:.6e}")
                print(f"    pos_err              = {pos_err:.6e}")
            print(f"    best iter so far     = {best_iter}")
            print(f"    best white residual  = {best_res_norm_white:.6e}")

        # convergence / anti-stall criterion
        white_improve = current_white_before - res_norm_white
        if (delta_R < eps_R) and (delta_p < eps_p) and (white_improve <= 1e-10):
            if verbose:
                print("满足收敛条件，停止外迭代。")
                print(f"return best iterate = {best_iter}")
            if return_history:
                return best_theta, best_X, history
            return best_theta, best_X

        # update nominal pose
        X_t = Pose(R=X_next.R.copy(), p=X_next.p.copy())

    if verbose:
        print("达到最大迭代次数，返回 best iterate。")
        print(f"best_iter                = {best_iter}")
        print(f"best residual norm       = {best_res_norm:.6e}")
        print(f"best whitened res norm   = {best_res_norm_white:.6e}")

    if return_history:
        return best_theta, best_X, history
    return best_theta, best_X


def make_rotation_z(theta: float) -> np.ndarray:
    """
    绕 z 轴旋转矩阵
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def make_rotation_y(theta: float) -> np.ndarray:
    """
    绕 y 轴旋转矩阵
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])


def sample_gaussian_noise(cov: np.ndarray) -> np.ndarray:
    """
    从 N(0, cov) 采样 6 维噪声
    """
    cov = ensure_matrix(cov, (6, 6))
    return np.random.multivariate_normal(mean=np.zeros(6), cov=cov)


def rotation_objective_fixed_p(
    phi: np.ndarray,
    X_center: Pose,
    p_fixed: np.ndarray,
    measurements_Y: np.ndarray,
    sensor_poses_WA,
    Sigma_w: np.ndarray,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
):
    """
    Scalar objective:
        f(phi) = 0.5 * || r_white(phi) ||^2
    with
        X(phi) = ( R_center * Exp(phi^), p_fixed ).

    Returns
    -------
    f : float
        0.5 * whitened_residual_norm^2
    white : float
        whitened residual norm
    X_eval : Pose
        evaluated pose
    """
    X_eval = Pose(
        R=X_center.R @ so3_exp(phi),
        p=p_fixed.copy(),
    )
    white = batch_whitened_residual_norm(
        X_eval,
        measurements_Y,
        sensor_poses_WA,
        Sigma_w,
        contact_points_A,
        field_params,
        stiffness_params,
    )
    f = 0.5 * (white ** 2)
    return f, white, X_eval


def numeric_rotation_hessian_fixed_p(
    X_center: Pose,
    p_fixed: np.ndarray,
    measurements_Y: np.ndarray,
    sensor_poses_WA,
    Sigma_w: np.ndarray,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
    h_phi: float = 1e-3,
):
    """
    Numerical gradient and 3x3 Hessian of
        f(phi) = 0.5 * ||r_white(phi)||^2
    at phi = 0, with p fixed.

    Uses central finite differences.
    """
    e = np.eye(3)

    def f_of(phi):
        f, _, _ = rotation_objective_fixed_p(
            phi=phi,
            X_center=X_center,
            p_fixed=p_fixed,
            measurements_Y=measurements_Y,
            sensor_poses_WA=sensor_poses_WA,
            Sigma_w=Sigma_w,
            contact_points_A=contact_points_A,
            field_params=field_params,
            stiffness_params=stiffness_params,
        )
        return f

    def white_of(phi):
        _, w, _ = rotation_objective_fixed_p(
            phi=phi,
            X_center=X_center,
            p_fixed=p_fixed,
            measurements_Y=measurements_Y,
            sensor_poses_WA=sensor_poses_WA,
            Sigma_w=Sigma_w,
            contact_points_A=contact_points_A,
            field_params=field_params,
            stiffness_params=stiffness_params,
        )
        return w

    f0 = f_of(np.zeros(3))
    white0 = white_of(np.zeros(3))

    # gradient
    g = np.zeros(3)
    for i in range(3):
        fp = f_of(h_phi * e[i])
        fm = f_of(-h_phi * e[i])
        g[i] = (fp - fm) / (2.0 * h_phi)

    # Hessian
    H = np.zeros((3, 3))

    # diagonal
    for i in range(3):
        fp = f_of(h_phi * e[i])
        fm = f_of(-h_phi * e[i])
        H[i, i] = (fp - 2.0 * f0 + fm) / (h_phi ** 2)

    # off-diagonal
    for i in range(3):
        for j in range(i + 1, 3):
            f_pp = f_of(h_phi * e[i] + h_phi * e[j])
            f_pm = f_of(h_phi * e[i] - h_phi * e[j])
            f_mp = f_of(-h_phi * e[i] + h_phi * e[j])
            f_mm = f_of(-h_phi * e[i] - h_phi * e[j])
            Hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_phi ** 2)
            H[i, j] = Hij
            H[j, i] = Hij

    # symmetrize for safety
    H = 0.5 * (H + H.T)

    evals, evecs = np.linalg.eigh(H)   # ascending order

    diag = {
        "f0": f0,
        "white0": white0,
        "grad": g,
        "grad_norm": np.linalg.norm(g),
        "H": H,
        "eigvals": evals,
        "eigvecs": evecs,
        "h_phi": h_phi,
    }
    return diag


def directional_quadratic_diagnostics(
    u: np.ndarray,
    X_center: Pose,
    p_fixed: np.ndarray,
    measurements_Y: np.ndarray,
    sensor_poses_WA,
    Sigma_w: np.ndarray,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
    h_dir: float = 1e-3,
):
    """
    For a unit direction u in R^3, compute
      - first directional derivative
      - second directional derivative
    of f(phi)=0.5||r_white(phi)||^2 at phi=0.
    """
    u = np.asarray(u, dtype=float)
    nu = np.linalg.norm(u)
    if nu < 1e-15:
        return {"dir_deriv": np.nan, "dir_curv": np.nan}
    u = u / nu

    def f_of(alpha):
        f, _, _ = rotation_objective_fixed_p(
            phi=alpha * u,
            X_center=X_center,
            p_fixed=p_fixed,
            measurements_Y=measurements_Y,
            sensor_poses_WA=sensor_poses_WA,
            Sigma_w=Sigma_w,
            contact_points_A=contact_points_A,
            field_params=field_params,
            stiffness_params=stiffness_params,
        )
        return f

    f0 = f_of(0.0)
    fp = f_of(h_dir)
    fm = f_of(-h_dir)

    dir_deriv = (fp - fm) / (2.0 * h_dir)
    dir_curv = (fp - 2.0 * f0 + fm) / (h_dir ** 2)

    return {
        "dir_deriv": dir_deriv,
        "dir_curv": dir_curv,
    }


def rotation_information_report(
    X_center: Pose,
    p_fixed: np.ndarray,
    measurements_Y: np.ndarray,
    sensor_poses_WA,
    Sigma_w: np.ndarray,
    contact_points_A: np.ndarray,
    field_params,
    stiffness_params,
    h_phi: float = 1e-3,
    X_WB_true: Pose = None,
    R_mode: np.ndarray = None,
    verbose: bool = True,
):
    """
    Main diagnostic entry.

    Returns a dict containing:
      - grad, Hessian, eigvals, eigvecs
      - condition metrics
      - directional curvatures along:
          * weakest eigvec
          * strongest eigvec
          * mode direction (if given)
          * truth direction (if given)
          * -grad direction
    """
    diag = numeric_rotation_hessian_fixed_p(
        X_center=X_center,
        p_fixed=p_fixed,
        measurements_Y=measurements_Y,
        sensor_poses_WA=sensor_poses_WA,
        Sigma_w=Sigma_w,
        contact_points_A=contact_points_A,
        field_params=field_params,
        stiffness_params=stiffness_params,
        h_phi=h_phi,
    )

    H = diag["H"]
    g = diag["grad"]
    evals = diag["eigvals"]
    evecs = diag["eigvecs"]

    lam_min = evals[0]
    lam_mid = evals[1]
    lam_max = evals[2]

    eps = 1e-14
    cond_abs = np.inf if abs(lam_min) < eps else abs(lam_max) / max(abs(lam_min), eps)
    anisotropy = abs(lam_min) / max(abs(lam_max), eps)
    trace_H = np.trace(H)

    report = {
        **diag,
        "lam_min": lam_min,
        "lam_mid": lam_mid,
        "lam_max": lam_max,
        "cond_abs": cond_abs,
        "anisotropy": anisotropy,
        "trace_H": trace_H,
        "dir_info": {},
    }

    # weakest / strongest eigen-directions
    report["dir_info"]["eig_weak"] = directional_quadratic_diagnostics(
        evecs[:, 0], X_center, p_fixed,
        measurements_Y, sensor_poses_WA, Sigma_w,
        contact_points_A, field_params, stiffness_params, h_dir=h_phi
    )
    report["dir_info"]["eig_strong"] = directional_quadratic_diagnostics(
        evecs[:, 2], X_center, p_fixed,
        measurements_Y, sensor_poses_WA, Sigma_w,
        contact_points_A, field_params, stiffness_params, h_dir=h_phi
    )

    # -grad direction
    if np.linalg.norm(g) > 1e-15:
        u_ng = -g / np.linalg.norm(g)
        report["dir_info"]["neg_grad"] = directional_quadratic_diagnostics(
            u_ng, X_center, p_fixed,
            measurements_Y, sensor_poses_WA, Sigma_w,
            contact_points_A, field_params, stiffness_params, h_dir=h_phi
        )
        report["neg_grad_dir"] = u_ng.copy()
    else:
        report["dir_info"]["neg_grad"] = {"dir_deriv": np.nan, "dir_curv": np.nan}
        report["neg_grad_dir"] = np.zeros(3)

    # mode direction
    if R_mode is not None:
        dphi_mode = log_so3(X_center.R.T @ R_mode)
        report["dphi_mode"] = dphi_mode.copy()
        report["mode_step"] = np.linalg.norm(dphi_mode)
        if np.linalg.norm(dphi_mode) > 1e-15:
            u_mode = dphi_mode / np.linalg.norm(dphi_mode)
            report["dir_info"]["mode"] = directional_quadratic_diagnostics(
                u_mode, X_center, p_fixed,
                measurements_Y, sensor_poses_WA, Sigma_w,
                contact_points_A, field_params, stiffness_params, h_dir=h_phi
            )
        else:
            report["dir_info"]["mode"] = {"dir_deriv": np.nan, "dir_curv": np.nan}
    else:
        report["dphi_mode"] = None
        report["mode_step"] = np.nan

    # truth direction
    if X_WB_true is not None:
        dphi_truth = log_so3(X_center.R.T @ X_WB_true.R)
        report["dphi_truth"] = dphi_truth.copy()
        report["truth_step"] = np.linalg.norm(dphi_truth)
        if np.linalg.norm(dphi_truth) > 1e-15:
            u_truth = dphi_truth / np.linalg.norm(dphi_truth)
            report["dir_info"]["truth"] = directional_quadratic_diagnostics(
                u_truth, X_center, p_fixed,
                measurements_Y, sensor_poses_WA, Sigma_w,
                contact_points_A, field_params, stiffness_params, h_dir=h_phi
            )
        else:
            report["dir_info"]["truth"] = {"dir_deriv": np.nan, "dir_curv": np.nan}
    else:
        report["dphi_truth"] = None
        report["truth_step"] = np.nan

    if verbose:
        print("\n===== Rotation Information / Hessian Diagnostic =====")
        print(f"white0            = {report['white0']:.6e}")
        print(f"f0                = {report['f0']:.6e}")
        print(f"||grad||          = {report['grad_norm']:.6e}")
        print("grad              =", report["grad"])
        print("Hessian H =")
        print(report["H"])
        print(f"eigvals(H)        = {report['eigvals']}")
        print(f"trace(H)          = {report['trace_H']:.6e}")
        print(f"lam_min           = {report['lam_min']:.6e}")
        print(f"lam_mid           = {report['lam_mid']:.6e}")
        print(f"lam_max           = {report['lam_max']:.6e}")
        print(f"|lam_min|/|lam_max| = {report['anisotropy']:.6e}")
        print(f"cond_abs          = {report['cond_abs']:.6e}")

        print("\n--- directional curvature / derivative ---")
        for k, v in report["dir_info"].items():
            print(f"[{k}] dir_deriv = {v['dir_deriv']:.6e}, dir_curv = {v['dir_curv']:.6e}")

        if report["dphi_mode"] is not None:
            print(f"\nmode_step         = {report['mode_step']:.6e}")
            print("dphi_mode         =", report["dphi_mode"])

        if report["dphi_truth"] is not None:
            print(f"truth_step        = {report['truth_step']:.6e}")
            print("dphi_truth        =", report["dphi_truth"])

    return report


def safe_cond(A, tol=1e-14):
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] < tol:
        return np.inf
    return s[0] / s[-1]


def sym(A):
    return 0.5 * (A + A.T)


def safe_solve(A, b, eps=1e-8):
    return np.linalg.solve(A + eps * np.eye(A.shape[0]), b)


def angle_between(u, v, eps=1e-12):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return np.nan
    c = np.dot(u, v) / (nu * nv)
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)


def pose_cost_whitened(X_test):
    Sigma_w_inv_local = np.linalg.inv(Sigma_w)
    total = 0.0
    for kk, X_WA_kk in enumerate(sensor_poses_WA):
        y_kk = measurements_Y[kk]
        w_pred_kk = measurement_model_y(
            X_WA_kk,
            X_test,
            contact_points_A,
            field_params,
            stiffness_params
        )
        r_kk = y_kk - w_pred_kk
        total += float(r_kk.T @ Sigma_w_inv_local @ r_kk)
    return total


def hat3(w):
    return np.array([
        [0.0,   -w[2],  w[1]],
        [w[2],   0.0,  -w[0]],
        [-w[1],  w[0],  0.0]
    ])


def exp_so3_local(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3) + hat3(phi)
    Kmat = hat3(phi / theta)
    return (
        np.eye(3)
        + np.sin(theta) * Kmat
        + (1.0 - np.cos(theta)) * (Kmat @ Kmat)
    )


def right_retract_pose_local(X_bar, xi):
    phi = xi[:3]
    v = xi[3:]
    R_new = X_bar.R @ exp_so3_local(phi)
    p_new = X_bar.p + v
    return Pose(R=R_new, p=p_new)


def plot_refinement_history(
    history: dict,
    X_WB_true: Pose = None,
    figsize=(14, 14),
    suptitle="Safeguarded Algorithm 2 Diagnostics",
):
    """
    Visualize history returned by safeguarded multi_pass_mfg_batch_refinement.

    Expected keys in history:
        iter
        accepted
        chosen_alpha
        delta_R
        delta_p
        res_norm
        res_norm_white
        rot_err
        pos_err
        p_x, p_y, p_z
        cand_p_x, cand_p_y, cand_p_z
        best_res_norm_white
        candidate_rot_step
        relaxed_rot_step
        candidate_pos_step
        relaxed_pos_step
        X_t_p, X_candidate_p, X_next_p
    """

    iters = np.asarray(history["iter"], dtype=float)

    accepted = np.asarray(history["accepted"], dtype=bool)
    chosen_alpha = np.asarray(history["chosen_alpha"], dtype=float)

    delta_R = np.asarray(history["delta_R"], dtype=float)
    delta_p = np.asarray(history["delta_p"], dtype=float)

    res_norm = np.asarray(history["res_norm"], dtype=float)
    res_norm_white = np.asarray(history["res_norm_white"], dtype=float)
    best_res_norm_white = np.asarray(history["best_res_norm_white"], dtype=float)

    rot_err = np.asarray(history["rot_err"], dtype=float)
    pos_err = np.asarray(history["pos_err"], dtype=float)

    p_x = np.asarray(history["p_x"], dtype=float)
    p_y = np.asarray(history["p_y"], dtype=float)
    p_z = np.asarray(history["p_z"], dtype=float)

    cand_p_x = np.asarray(history["cand_p_x"], dtype=float)
    cand_p_y = np.asarray(history["cand_p_y"], dtype=float)
    cand_p_z = np.asarray(history["cand_p_z"], dtype=float)

    candidate_rot_step = np.asarray(history["candidate_rot_step"], dtype=float)
    relaxed_rot_step = np.asarray(history["relaxed_rot_step"], dtype=float)
    candidate_pos_step = np.asarray(history["candidate_pos_step"], dtype=float)
    relaxed_pos_step = np.asarray(history["relaxed_pos_step"], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.ravel()

    # =========================================================
    # 1) Error vs outer iteration
    # =========================================================
    ax = axes[0]
    if np.isfinite(rot_err).any():
        ax.plot(iters, rot_err, marker="o", label="rotation error")
    if np.isfinite(pos_err).any():
        ax.plot(iters, pos_err, marker="s", label="position error")

    # 标记 rejected steps
    rej_idx = np.where(~accepted)[0]
    if len(rej_idx) > 0:
        y_ref = np.nanmax(rot_err) if np.isfinite(rot_err).any() else 0.0
        ax.scatter(iters[rej_idx], np.full_like(iters[rej_idx], y_ref),
                   marker="x", s=80, label="rejected step")

    ax.set_title("Error vs Outer Iteration")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("error")
    ax.grid(True)
    ax.legend()

    # =========================================================
    # 2) Residual norms
    # =========================================================
    ax = axes[1]
    ax.plot(iters, res_norm, marker="o", label="residual norm")
    ax.plot(iters, res_norm_white, marker="s", label="whitened residual norm")
    ax.plot(iters, best_res_norm_white, marker="^", label="best-so-far whitened")

    if len(rej_idx) > 0:
        y_rej = np.interp(iters[rej_idx], iters, res_norm_white)
        ax.scatter(iters[rej_idx], y_rej, marker="x", s=80, label="rejected step")

    ax.set_title("Residual Norms vs Outer Iteration")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("norm")
    ax.grid(True)
    ax.legend()

    # =========================================================
    # 3) Accepted / rejected + chosen alpha
    # =========================================================
    ax = axes[2]
    accepted_num = accepted.astype(float)
    ax.step(iters, accepted_num, where="mid", label="accepted (1=yes, 0=no)")
    ax.plot(iters, chosen_alpha, marker="o", label="chosen alpha")
    ax.set_title("Acceptance and Chosen Alpha")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("value")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True)
    ax.legend()

    # =========================================================
    # 4) Relaxed update sizes
    # =========================================================
    ax = axes[3]
    ax.plot(iters, delta_R, marker="o", label="delta_R")
    ax.plot(iters, delta_p, marker="s", label="delta_p")

    if len(rej_idx) > 0:
        ax.scatter(iters[rej_idx], delta_R[rej_idx], marker="x", s=80, label="rejected step")

    ax.set_title("Relaxed Update Sizes")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("step size")
    ax.grid(True)
    ax.legend()

    # =========================================================
    # 5) Translation components
    # =========================================================
    ax = axes[4]
    ax.plot(iters, p_x, marker="o", label="p_x")
    ax.plot(iters, p_y, marker="s", label="p_y")
    ax.plot(iters, p_z, marker="^", label="p_z")

    if X_WB_true is not None:
        ax.axhline(X_WB_true.p[0], linestyle="--", label="p_x true")
        ax.axhline(X_WB_true.p[1], linestyle="--", label="p_y true")
        ax.axhline(X_WB_true.p[2], linestyle="--", label="p_z true")

    if len(rej_idx) > 0:
        ax.scatter(iters[rej_idx], p_x[rej_idx], marker="x", s=80, label="rejected step")

    ax.set_title("Translation Components vs Outer Iteration")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("translation component")
    ax.grid(True)
    ax.legend()

    # =========================================================
    # 6) Candidate vs relaxed outer steps
    # =========================================================
    ax = axes[5]
    ax.plot(iters, candidate_rot_step, marker="o", label="candidate rot step")
    ax.plot(iters, relaxed_rot_step, marker="s", label="relaxed rot step")
    ax.plot(iters, candidate_pos_step, marker="^", label="candidate pos step")
    ax.plot(iters, relaxed_pos_step, marker="d", label="relaxed pos step")

    if len(rej_idx) > 0:
        ax.scatter(iters[rej_idx], candidate_rot_step[rej_idx], marker="x", s=80, label="rejected step")

    ax.set_title("Candidate vs Relaxed Outer Steps")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("step size")
    ax.grid(True)
    ax.legend()

    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



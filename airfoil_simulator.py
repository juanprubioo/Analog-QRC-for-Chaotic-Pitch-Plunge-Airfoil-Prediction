
"""
airfoil_simulator.py

Single-file Python simulator for the conceptual 2-DOF aeroelastic airfoil
model used in:
- Liu et al., AIAA Journal 2025 (NG-RC paper)
- Liu et al., Acta Mechanica Sinica 2021 (reference with explicit coefficients)

Implemented features
--------------------
1) 6-state first-order model:
   x = [alpha, alpha_dot, xi, xi_dot, w1, w2]
2) Cubic structural nonlinearity:
   G(xi) = xi + gamma3 * xi**3
   M(alpha) = alpha + beta3 * alpha**3
3) Optional free-play pitch nonlinearity from the reference paper
4) Fixed-step RK4 integration
5) Downsampling to create observations
6) Additive synthetic noise:
   Y = X + noise_level * std(X) * epsilon_noise
7) Ready-made presets matching the cases used in the NG-RC paper

Notes
-----
- Time variable is the nondimensional time tau.
- The simulator is faithful to the state-space form and appendix coefficients
  given in the 2021 reference.
- For ML datasets, the default generation path follows the 2025 NG-RC paper:
  dt = 0.005, Delta_t = 0.05, sampling rate M = 10.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Literal
import json
import math
import argparse
import numpy as np


Array = np.ndarray
NoiseKind = Literal["gaussian", "student_t", "none"]
NonlinearityKind = Literal["cubic", "freeplay"]


@dataclass
class AirfoilParams:
    # Main physical parameters
    mu: float = 100.0
    a_h: float = -0.5
    x_alpha: float = 0.25
    r_alpha: float = 0.5
    zeta_xi: float = 0.0
    zeta_alpha: float = 0.0
    omega_bar: float = 0.2
    U_star: float = 6.2851 * 2.05  # Example chaotic case by default

    # Cubic nonlinearity parameters
    gamma3: float = 10.0
    beta3: float = 80.0

    # Wagner constants
    psi1: float = 0.165
    psi2: float = 0.335
    eps1: float = 0.0455
    eps2: float = 0.3

    # Free-play parameters (optional)
    # Symmetric free-play rational approximation reported in the 2021 reference
    freeplay_delta_deg: float = 0.25

    # Integration and dataset defaults
    dt: float = 0.005
    sample_every: int = 10

    # Model choice
    nonlinearity: NonlinearityKind = "cubic"


def cubic_G(xi: float, p: AirfoilParams) -> float:
    return xi + p.gamma3 * xi**3


def cubic_M(alpha: float, p: AirfoilParams) -> float:
    return alpha + p.beta3 * alpha**3


def freeplay_M_rational(alpha: float, p: AirfoilParams) -> float:
    """
    Rational approximation for the symmetric free-play nonlinearity
    given in the 2021 reference for delta = 0.25 deg.
    """
    num = 0.00002747 - 0.01702 * alpha - 11.94 * alpha**2 + 5462.0 * alpha**3
    den = 0.4556 - 12.36 * alpha + 5732.0 * alpha**2
    return num / den


def structural_terms(alpha: float, xi: float, p: AirfoilParams) -> Tuple[float, float]:
    # Plunge nonlinearity
    G = cubic_G(xi, p)

    # Pitch nonlinearity
    if p.nonlinearity == "cubic":
        M = cubic_M(alpha, p)
    elif p.nonlinearity == "freeplay":
        M = freeplay_M_rational(alpha, p)
    else:
        raise ValueError(f"Unsupported nonlinearity: {p.nonlinearity}")

    return G, M


def compute_coefficients(p: AirfoilParams) -> Dict[str, float]:
    """
    Coefficients from the Appendix of the 2021 reference.
    """
    mu = p.mu
    ah = p.a_h
    xa = p.x_alpha
    ra = p.r_alpha
    zxi = p.zeta_xi
    za = p.zeta_alpha
    ob = p.omega_bar
    Us = p.U_star
    psi1 = p.psi1
    psi2 = p.psi2
    eps1 = p.eps1
    eps2 = p.eps2

    e0 = 1.0 + 1.0 / mu
    d0 = (xa * mu - ah) / (mu * ra**2)
    e1 = xa - ah / mu
    d1 = 1.0 + (1.0 + 8.0 * ah**2) / (8.0 * mu * ra**2)
    f0 = 1.0 / (e0 * d1 - e1 * d0)
    f1 = 0.5 - ah

    e2 = 2.0 * (zxi * ob / Us + 1.0 / mu)
    d2 = -(1.0 + 2.0 * ah) / (mu * ra**2)
    e3 = 2.0 * (1.0 - ah) / mu
    d3 = 2.0 * za / Us - ah * (1.0 - 2.0 * ah) / (mu * ra**2)
    e4 = 2.0 / mu
    d4 = -(1.0 + 2.0 * ah) / (mu * ra**2)
    e5 = -e4
    e6 = -e4
    d5 = -d4
    d6 = -d4
    e7 = (ob / Us) ** 2
    d7 = (1.0 / Us) ** 2

    # Main linear coefficients
    n21 = f0 * (d0 * e4 - e0 * d4)
    n22 = f0 * (d0 * e3 - e0 * d3)
    n24 = f0 * (d0 * e2 - e0 * d2)
    n25 = f0 * (d0 * e5 - e0 * d5)
    n26 = f0 * (d0 * e6 - e0 * d6)
    m2 = -f0 * e0 * d7
    g2 = f0 * d0 * e7

    n41 = -f0 * (d1 * e4 - e1 * d4)
    n42 = -f0 * (d1 * e3 - e1 * d3)
    n44 = -f0 * (d1 * e2 - e1 * d2)
    n45 = -f0 * (d1 * e5 - e1 * d5)
    n46 = -f0 * (d1 * e6 - e1 * d6)
    m4 = f0 * e1 * d7
    g4 = -f0 * d1 * e7

    n51 = (f0 * (e1 * d4 - d1 * e4) + f0 * f1 * (d0 * e4 - e0 * d4)) * psi1
    n52 = (f0 * (e1 * d3 - d1 * e3) + f0 * f1 * (d0 * e3 - e0 * d3) + 1.0) * psi1
    n54 = (f0 * (e1 * d2 - d1 * e2) + f0 * f1 * (d0 * e2 - e0 * d2)) * psi1
    n55 = -eps1 + (f0 * (e1 * d5 - d1 * e5) + f0 * f1 * (d0 * e5 - e0 * d5)) * psi1
    n56 = (f0 * (e1 * d6 - d1 * e6) + f0 * f1 * (d0 * e6 - e0 * d6)) * psi1
    m5 = psi1 * f0 * (e1 * d7 - f1 * e0 * d7)
    g5 = psi1 * f0 * (f1 * d0 * e7 - d1 * e7)

    n61 = (f0 * (e1 * d4 - d1 * e4) + f0 * f1 * (d0 * e4 - e0 * d4)) * psi2
    n62 = (f0 * (e1 * d3 - d1 * e3) + f0 * f1 * (d0 * e3 - e0 * d3) + 1.0) * psi2
    n64 = (f0 * (e1 * d2 - d1 * e2) + f0 * f1 * (d0 * e2 - e0 * d2)) * psi2
    n65 = (f0 * (e1 * d5 - d1 * e5) + f0 * f1 * (d0 * e5 - e0 * d5)) * psi2
    n66 = -eps2 + (f0 * (e1 * d6 - d1 * e6) + f0 * f1 * (d0 * e6 - e0 * d6)) * psi2
    m6 = psi2 * f0 * (e1 * d7 - f1 * e0 * d7)
    g6 = psi2 * f0 * (f1 * d0 * e7 - d1 * e7)

    return {
        "e0": e0, "d0": d0, "e1": e1, "d1": d1, "f0": f0, "f1": f1,
        "e2": e2, "d2": d2, "e3": e3, "d3": d3, "e4": e4, "d4": d4,
        "e5": e5, "e6": e6, "d5": d5, "d6": d6, "e7": e7, "d7": d7,
        "n21": n21, "n22": n22, "n24": n24, "n25": n25, "n26": n26, "m2": m2, "g2": g2,
        "n41": n41, "n42": n42, "n44": n44, "n45": n45, "n46": n46, "m4": m4, "g4": g4,
        "n51": n51, "n52": n52, "n54": n54, "n55": n55, "n56": n56, "m5": m5, "g5": g5,
        "n61": n61, "n62": n62, "n64": n64, "n65": n65, "n66": n66, "m6": m6, "g6": g6,
    }


def airfoil_rhs(tau: float, x: Array, p: AirfoilParams, c: Optional[Dict[str, float]] = None) -> Array:
    """
    6-state first-order ODE system:
        x = [alpha, alpha_dot, xi, xi_dot, w1, w2]
    """
    if c is None:
        c = compute_coefficients(p)

    x1, x2, x3, x4, x5, x6 = x
    G, M = structural_terms(alpha=x1, xi=x3, p=p)

    dx1 = x2
    dx2 = c["n21"] * x1 + c["n22"] * x2 + c["n24"] * x4 + c["n25"] * x5 + c["n26"] * x6 + c["g2"] * G + c["m2"] * M
    dx3 = x4
    dx4 = c["n41"] * x1 + c["n42"] * x2 + c["n44"] * x4 + c["n45"] * x5 + c["n46"] * x6 + c["g4"] * G + c["m4"] * M
    dx5 = c["n51"] * x1 + c["n52"] * x2 + c["n54"] * x4 + c["n55"] * x5 + c["n56"] * x6 + c["g5"] * G + c["m5"] * M
    dx6 = c["n61"] * x1 + c["n62"] * x2 + c["n64"] * x4 + c["n65"] * x5 + c["n66"] * x6 + c["g6"] * G + c["m6"] * M

    return np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=float)


def rk4_step(fun, tau: float, x: Array, h: float, *args, **kwargs) -> Array:
    k1 = fun(tau, x, *args, **kwargs)
    k2 = fun(tau + 0.5 * h, x + 0.5 * h * k1, *args, **kwargs)
    k3 = fun(tau + 0.5 * h, x + 0.5 * h * k2, *args, **kwargs)
    k4 = fun(tau + h, x + h * k3, *args, **kwargs)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    p: AirfoilParams,
    t_final: float,
    x0: Array,
) -> Tuple[Array, Array]:
    """
    Integrate the system with fixed-step RK4.

    Returns
    -------
    tau : (N,)
    X   : (N, 6)
    """
    c = compute_coefficients(p)
    dt = p.dt
    n_steps = int(round(t_final / dt))

    tau = np.linspace(0.0, n_steps * dt, n_steps + 1)
    X = np.zeros((n_steps + 1, 6), dtype=float)
    X[0] = np.asarray(x0, dtype=float)

    for i in range(n_steps):
        X[i + 1] = rk4_step(airfoil_rhs, tau[i], X[i], dt, p, c)

    return tau, X


def downsample(tau: Array, X: Array, sample_every: int) -> Tuple[Array, Array]:
    idx = np.arange(0, len(tau), sample_every, dtype=int)
    return tau[idx], X[idx]


def add_noise(
    X: Array,
    noise_level: float = 0.40,
    noise_kind: NoiseKind = "gaussian",
    student_df: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, Array]:
    """
    Create synthetic observations according to:
        Y = X + noise_level * std(X) * epsilon_noise

    Parameters
    ----------
    noise_level : float
        Use 0.40 for 40% noise.
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    std = X.std(axis=0, ddof=0)

    if noise_kind == "none":
        eps = np.zeros_like(X)
    elif noise_kind == "gaussian":
        eps = rng.normal(loc=0.0, scale=1.0, size=X.shape)
    elif noise_kind == "student_t":
        eps = rng.standard_t(df=student_df, size=X.shape)
    else:
        raise ValueError(f"Unsupported noise_kind: {noise_kind}")

    noise = noise_level * std[None, :] * eps
    Y = X + noise
    return Y, noise


def save_dataset(
    out_path: str,
    params: AirfoilParams,
    tau_full: Array,
    X_full: Array,
    tau_sampled: Array,
    X_sampled: Array,
    Y_sampled: Array,
    noise_sampled: Array,
    metadata: Dict,
) -> None:
    np.savez(
        out_path,
        tau_full=tau_full,
        X_full=X_full,
        tau_sampled=tau_sampled,
        X_sampled=X_sampled,
        Y_sampled=Y_sampled,
        noise_sampled=noise_sampled,
        params_json=json.dumps(asdict(params)),
        metadata_json=json.dumps(metadata),
    )


def preset_case(case_name: str) -> Tuple[AirfoilParams, Array, float]:
    """
    Presets aligned with the NG-RC paper scenarios.

    Returns
    -------
    params, x0, t_final
    """
    case = case_name.upper()

    if case == "I":
        # Damped oscillations
        U_L = 6.2851
        p = AirfoilParams(
            mu=100.0, a_h=-0.5, x_alpha=0.25, r_alpha=0.5,
            zeta_xi=0.0, zeta_alpha=0.0,
            omega_bar=0.2, beta3=80.0, gamma3=10.0,
            U_star=0.25 * U_L,
            nonlinearity="cubic",
            dt=0.005, sample_every=10
        )
        x0 = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        t_final = 1000.0

    elif case == "II":
        # Single-periodic LCO
        U_L = 6.2851
        p = AirfoilParams(
            mu=100.0, a_h=-0.5, x_alpha=0.25, r_alpha=0.5,
            zeta_xi=0.0, zeta_alpha=0.0,
            omega_bar=0.2, beta3=80.0, gamma3=10.0,
            U_star=1.50 * U_L,
            nonlinearity="cubic",
            dt=0.005, sample_every=10
        )
        x0 = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        t_final = 3500.0

    elif case == "III":
        # Multi-periodic oscillations
        U_L = 6.2851
        p = AirfoilParams(
            mu=100.0, a_h=-0.5, x_alpha=0.25, r_alpha=0.5,
            zeta_xi=0.0, zeta_alpha=0.0,
            omega_bar=0.2, beta3=80.0, gamma3=10.0,
            U_star=2.45 * U_L,
            nonlinearity="cubic",
            dt=0.005, sample_every=10
        )
        x0 = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        t_final = 3500.0

    elif case == "IV":
        # Chaotic oscillations
        U_L = 6.2851
        p = AirfoilParams(
            mu=100.0, a_h=-0.5, x_alpha=0.25, r_alpha=0.5,
            zeta_xi=0.0, zeta_alpha=0.0,
            omega_bar=0.2, beta3=80.0, gamma3=10.0,
            U_star=2.05 * U_L,
            nonlinearity="cubic",
            dt=0.005, sample_every=10
        )
        x0 = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        t_final = 3500.0

    elif case == "V":
        # Amplitude-modulated oscillations
        U_L = 2.951
        p = AirfoilParams(
            mu=100.0, a_h=-0.5, x_alpha=0.10, r_alpha=0.5,
            zeta_xi=0.0, zeta_alpha=0.0,
            omega_bar=1.2, beta3=10.0, gamma3=10.0,
            U_star=1.10 * U_L,
            nonlinearity="cubic",
            dt=0.005, sample_every=10
        )
        x0 = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        t_final = 4000.0

    else:
        raise ValueError("case_name must be one of: I, II, III, IV, V")

    return p, x0, t_final


def generate_case_dataset(
    case_name: str = "IV",
    noise_level: float = 0.40,
    noise_kind: NoiseKind = "gaussian",
    student_df: int = 5,
    seed: int = 12345,
    out_path: Optional[str] = None,
):
    p, x0, t_final = preset_case(case_name)
    tau_full, X_full = simulate(p, t_final=t_final, x0=x0)
    tau_sampled, X_sampled = downsample(tau_full, X_full, p.sample_every)
    rng = np.random.default_rng(seed)
    Y_sampled, noise_sampled = add_noise(
        X_sampled,
        noise_level=noise_level,
        noise_kind=noise_kind,
        student_df=student_df,
        rng=rng,
    )

    metadata = {
        "case_name": case_name,
        "noise_level_fraction": noise_level,
        "noise_kind": noise_kind,
        "student_df": student_df,
        "seed": seed,
        "dt": p.dt,
        "sample_every": p.sample_every,
        "sampling_interval": p.dt * p.sample_every,
        "state_order": ["alpha", "alpha_dot", "xi", "xi_dot", "w1", "w2"],
    }

    if out_path is not None:
        save_dataset(
            out_path=out_path,
            params=p,
            tau_full=tau_full,
            X_full=X_full,
            tau_sampled=tau_sampled,
            X_sampled=X_sampled,
            Y_sampled=Y_sampled,
            noise_sampled=noise_sampled,
            metadata=metadata,
        )

    return p, tau_full, X_full, tau_sampled, X_sampled, Y_sampled, noise_sampled, metadata


def print_summary(p: AirfoilParams, metadata: Dict, X_sampled: Array, Y_sampled: Array) -> None:
    print("=== Airfoil simulator summary ===")
    print(json.dumps(asdict(p), indent=2))
    print("=== Dataset metadata ===")
    print(json.dumps(metadata, indent=2))
    print(f"Sampled clean shape : {X_sampled.shape}")
    print(f"Sampled noisy shape : {Y_sampled.shape}")
    print(f"Std clean states    : {X_sampled.std(axis=0)}")
    print(f"Std noisy states    : {Y_sampled.std(axis=0)}")


def main():
    parser = argparse.ArgumentParser(description="Single-file aeroelastic airfoil simulator")
    parser.add_argument("--case", type=str, default="IV", help="I, II, III, IV, or V")
    parser.add_argument("--noise-level", type=float, default=0.40, help="Use 0.40 for 40%% noise")
    parser.add_argument("--noise-kind", type=str, default="gaussian", choices=["gaussian", "student_t", "none"])
    parser.add_argument("--student-df", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=str, default="airfoil_case_dataset.npz")
    args = parser.parse_args()

    p, tau_full, X_full, tau_sampled, X_sampled, Y_sampled, noise_sampled, metadata = generate_case_dataset(
        case_name=args.case,
        noise_level=args.noise_level,
        noise_kind=args.noise_kind,
        student_df=args.student_df,
        seed=args.seed,
        out_path=args.out,
    )

    print_summary(p, metadata, X_sampled, Y_sampled)
    print(f"Saved dataset to: {args.out}")


if __name__ == "__main__":
    main()

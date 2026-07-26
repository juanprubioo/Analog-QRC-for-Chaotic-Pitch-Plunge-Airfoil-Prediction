"""
airfoil_state_estimation.py

State-estimation / smoothing helpers for the conceptual aeroelastic airfoil model.

Purpose
-------
This module provides two levels of preprocessing for noisy aeroelastic state
observations Y:

1) quick_state_estimation:
   A lightweight smoothing stage based on Savitzky-Golay filtering (or moving
   average fallback). This is useful for fast experiments and as an initial
   guess for the optimization-based smoother.

2) fixed_interval_smoother:
   A practical direct-transcription fixed-interval smoother inspired by the
   formulation in Liu et al. (Acta Mechanica Sinica, 2021). The full paper uses
   a Runge-Kutta-based optimization with dynamic consistency and closeness to
   measurements. Here we implement a computationally lighter but still model-
   constrained version that optimizes the state trajectory X directly with two
   residual blocks:

     - observation residuals:  sqrt(lambda_obs) * (X_hat - Y)
     - dynamics residuals:     sqrt(lambda_dyn) * (X_hat[j+1] - Phi(X_hat[j]))

   where Phi is the sampled-time state transition obtained by repeatedly
   applying the RK4 integrator of the aeroelastic model over one sampling
   interval.

Notes
-----
- The smoother assumes that the observations contain all 6 state components,
  which matches the fully observable synthetic-data setting used in the current
  scripts.
- For large datasets, fixed-interval smoothing on the whole trajectory can be
  expensive. The companion `airfoil_qrc_v6.py` therefore applies it on a compact
  prefix segment sized to the requested train/test windows.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


# -----------------------------------------------------------------------------
# Dynamic import of the aeroelastic simulator
# -----------------------------------------------------------------------------
def load_airfoil_module():
    here = Path(__file__).resolve().parent
    candidate = here / "airfoil_simulator.py"
    if not candidate.exists():
        raise FileNotFoundError(
            "airfoil_simulator.py was not found in the same directory as this script."
        )

    spec = importlib.util.spec_from_file_location("airfoil_simulator", candidate)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import spec for airfoil_simulator.py")

    import sys

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


airfoil = load_airfoil_module()


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _as_params(params: Any):
    if isinstance(params, airfoil.AirfoilParams):
        return params
    if isinstance(params, dict):
        return airfoil.AirfoilParams(**params)
    raise TypeError("params must be an AirfoilParams instance or a dict")


def sample_step_map(x: np.ndarray, params: Any, dt_sample: Optional[float] = None) -> np.ndarray:
    """
    Advance one sampled-time step using repeated RK4 substeps from airfoil_simulator.py.
    """
    p = _as_params(params)
    if dt_sample is None:
        dt_sample = p.dt * p.sample_every
    n_sub = max(1, int(round(dt_sample / p.dt)))

    xk = np.asarray(x, dtype=float).copy()
    # rk4_step in airfoil_simulator.py has signature:
    #   rk4_step(fun, tau, x, h, *args, **kwargs)
    # so we must pass an explicit time argument. The airfoil RHS is autonomous,
    # therefore a dummy tau that advances with the substep index is sufficient.
    c = airfoil.compute_coefficients(p)
    tau = 0.0
    for _ in range(n_sub):
        xk = airfoil.rk4_step(airfoil.airfoil_rhs, tau, xk, p.dt, p, c)
        tau += p.dt
    return xk


def _safe_odd_window(n: int, requested: int) -> int:
    n = int(n)
    requested = int(max(3, requested))
    requested = min(requested, n if n % 2 == 1 else n - 1)
    if requested < 3:
        requested = 3 if n >= 3 else n
    if requested % 2 == 0:
        requested -= 1
    return max(1, requested)


# -----------------------------------------------------------------------------
# Quick smoothing
# -----------------------------------------------------------------------------
def quick_state_estimation(
    Y_obs: np.ndarray,
    method: str = "savgol",
    window_length: int = 11,
    polyorder: int = 3,
    moving_average_window: int = 7,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fast denoising / initial guess stage.
    """
    Y = np.asarray(Y_obs, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y_obs must be a 2D array of shape (T, D)")

    T, D = Y.shape
    X0 = np.empty_like(Y)

    if method == "savgol":
        win = _safe_odd_window(T, window_length)
        po = min(polyorder, max(1, win - 1))
        if win < 3:
            X0 = Y.copy()
        else:
            for d in range(D):
                X0[:, d] = savgol_filter(Y[:, d], window_length=win, polyorder=po, mode="interp")
        info = {"method": "savgol", "window_length": int(win), "polyorder": int(po)}
    elif method == "moving_average":
        win = max(1, int(moving_average_window))
        kernel = np.ones(win, dtype=float) / float(win)
        for d in range(D):
            X0[:, d] = np.convolve(Y[:, d], kernel, mode="same")
        info = {"method": "moving_average", "window_length": int(win)}
    elif method == "identity":
        X0 = Y.copy()
        info = {"method": "identity"}
    else:
        raise ValueError(f"Unsupported quick smoothing method: {method}")

    X0 = np.nan_to_num(X0, nan=0.0, posinf=0.0, neginf=0.0)
    return X0, info


# -----------------------------------------------------------------------------
# Fixed-interval smoother (practical direct-transcription variant)
# -----------------------------------------------------------------------------
def _smoother_residual_vector(
    z_flat: np.ndarray,
    Y_obs: np.ndarray,
    params: Any,
    dt_sample: float,
    lambda_obs: float,
    lambda_dyn: float,
) -> np.ndarray:
    X = z_flat.reshape(Y_obs.shape)
    T, D = X.shape

    obs_res = np.sqrt(lambda_obs) * (X - Y_obs)

    dyn_res = np.empty((T - 1, D), dtype=float)
    for j in range(T - 1):
        x_pred = sample_step_map(X[j], params=params, dt_sample=dt_sample)
        dyn_res[j] = np.sqrt(lambda_dyn) * (X[j + 1] - x_pred)

    return np.concatenate([obs_res.ravel(), dyn_res.ravel()])



def fixed_interval_smoother(
    Y_obs: np.ndarray,
    params: Any,
    dt_sample: Optional[float] = None,
    lambda_obs: float = 1.0,
    lambda_dyn: float = 5.0,
    max_nfev: int = 50,
    initial_method: str = "savgol",
    initial_window_length: int = 11,
    initial_polyorder: int = 3,
    verbose: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Practical fixed-interval smoothing with model-consistency residuals.

    This is not a line-for-line reproduction of the full optimization in the
    reference paper, but it preserves the key idea: estimate the whole state
    trajectory by balancing closeness to the measurements and fidelity to the
    governing dynamics.
    """
    Y = np.asarray(Y_obs, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y_obs must be a 2D array of shape (T, D)")

    p = _as_params(params)
    if dt_sample is None:
        dt_sample = p.dt * p.sample_every

    X0, quick_info = quick_state_estimation(
        Y,
        method=initial_method,
        window_length=initial_window_length,
        polyorder=initial_polyorder,
    )

    # Soft bounds around data to stabilize optimization.
    std = np.std(Y, axis=0, ddof=1)
    std = np.where(std > 1e-12, std, 1.0)
    margin = 6.0 * std
    lo = (Y - margin).ravel()
    hi = (Y + margin).ravel()

    result = least_squares(
        fun=_smoother_residual_vector,
        x0=X0.ravel(),
        bounds=(lo, hi),
        method="trf",
        max_nfev=int(max_nfev),
        verbose=int(verbose),
        kwargs={
            "Y_obs": Y,
            "params": p,
            "dt_sample": float(dt_sample),
            "lambda_obs": float(lambda_obs),
            "lambda_dyn": float(lambda_dyn),
        },
    )

    X_hat = result.x.reshape(Y.shape)
    X_hat = np.nan_to_num(X_hat, nan=0.0, posinf=0.0, neginf=0.0)

    info: Dict[str, Any] = {
        "method": "fixed_interval",
        "lambda_obs": float(lambda_obs),
        "lambda_dyn": float(lambda_dyn),
        "dt_sample": float(dt_sample),
        "max_nfev": int(max_nfev),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "quick_initialization": quick_info,
    }
    return X_hat, info


# -----------------------------------------------------------------------------
# High-level wrapper
# -----------------------------------------------------------------------------
def estimate_states_from_observations(
    Y_obs: np.ndarray,
    params: Any,
    method: str = "quick",
    dt_sample: Optional[float] = None,
    quick_window_length: int = 11,
    quick_polyorder: int = 3,
    lambda_obs: float = 1.0,
    lambda_dyn: float = 5.0,
    max_nfev: int = 50,
    X_true: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Main entry point used by the QRC script.
    """
    if method == "quick":
        X_hat, info = quick_state_estimation(
            Y_obs,
            method="savgol",
            window_length=quick_window_length,
            polyorder=quick_polyorder,
        )
    elif method == "fixed_interval":
        X_hat, info = fixed_interval_smoother(
            Y_obs,
            params=params,
            dt_sample=dt_sample,
            lambda_obs=lambda_obs,
            lambda_dyn=lambda_dyn,
            max_nfev=max_nfev,
            initial_method="savgol",
            initial_window_length=quick_window_length,
            initial_polyorder=quick_polyorder,
            verbose=0,
        )
    else:
        raise ValueError(f"Unsupported state-estimation method: {method}")

    if X_true is not None:
        X_true = np.asarray(X_true, dtype=float)
        mse = float(np.mean((X_hat - X_true) ** 2))
        rmse = float(np.sqrt(mse))
        info["mse_vs_truth"] = mse
        info["rmse_vs_truth"] = rmse

    return X_hat, info


if __name__ == "__main__":
    # Minimal smoke test when run directly.
    p, X_full, tau_full, tau_s, X_s, Y_s, noise_s, meta = airfoil.generate_case_dataset(
        case_name="IV",
        noise_level=0.4,
        noise_kind="gaussian",
        seed=123,
        out_path=None,
    )
    Y_seg = Y_s[:80]
    X_seg = X_s[:80]
    X_quick, info_quick = estimate_states_from_observations(Y_seg, p, method="quick", X_true=X_seg)
    X_fix, info_fix = estimate_states_from_observations(Y_seg, p, method="fixed_interval", max_nfev=20, X_true=X_seg)
    print({"quick": info_quick, "fixed_interval": info_fix})

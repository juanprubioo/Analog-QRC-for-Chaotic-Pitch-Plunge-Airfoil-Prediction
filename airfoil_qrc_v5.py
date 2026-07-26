"""
airfoil_qrc_v4.py

Second-version pipeline for the conceptual aeroelastic airfoil QRC study.

What is new in this version
---------------------------
1) Automatic plots:
   - time history
   - PSD (Welch)
   - phase portraits
2) Classical NG-RC baseline in the same script
3) Two QRC encodings:
   - local detuning encoding (closest to the inspected Aquila notebook)
   - global pulse encoding variant (closer to the timeseries treatment in Kornjaca et al.)

Design philosophy
-----------------
- The local encoding follows as closely as possible the structure that appears in the
  inspected `QRC Demo Aquila Submission.ipynb`: fixed chain geometry, constant global
  Rabi drive, constant global detuning offset, local feature-dependent detuning, Z and ZZ readout.
- The global encoding is implemented as a practical variant for timeseries windows:
  the flattened input window is mapped to a piecewise-linear global detuning waveform.
  Because the exact `QRC Demo Timeseries.ipynb` content was not accessible in full during
  construction, the helper `_apply_piecewise_global_detuning` tries the most likely Bloqade
  APIs and raises an explicit error if the local Bloqade version differs.
- This is an emulation-first script. Hardware submission is intentionally left out here.

Typical usage
-------------
# Local-detuning QRC + NG-RC baseline + plots
python airfoil_qrc_v4.py --case IV --encoding local --window-length 2 \
    --state-indices 0 1 2 3 4 5 --nshots 200 --make-plots 1 \
    --out-prefix caseIV_local

# Global-pulse QRC variant + baseline + plots
python airfoil_qrc_v4.py --case IV --encoding global --window-length 2 \
    --state-indices 0 1 2 3 4 5 --nshots 200 --make-plots 1 \
    --out-prefix caseIV_global
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import gc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler


# -----------------------------------------------------------------------------
# Dynamic import of the previously created aeroelastic simulator
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
# Lazy Bloqade import so that non-quantum parts remain importable if Bloqade is missing
# -----------------------------------------------------------------------------
def get_bloqade_objects():
    try:
        import bloqade
        from bloqade.analog.ir import Chain
        return bloqade, Chain
    except Exception as exc:
        raise ImportError(
            "Bloqade is required for QRC emulation. The repository requirements pin "
            "bloqade==0.30.0 and compatible Braket dependencies."
        ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STATE_NAMES = ["alpha", "alpha_dot", "xi", "xi_dot", "w1", "w2"]


@dataclass
class QRCConfig:
    atom_number: int
    encoding: str = "local"  # 'local' or 'global'
    lattice_spacing: float = 10.0
    encoding_scale: float = 9.0
    rabi_frequency: float = 6.283
    total_time: float = 4.0
    time_steps: int = 8
    readouts: str = "Z"
    pulse_bias: float = 0.0


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def load_or_generate_dataset(
    input_npz: Optional[str],
    case: str,
    noise_level: float,
    noise_kind: str,
    student_df: int,
    seed: int,
) -> Dict[str, Any]:
    if input_npz is not None:
        data = np.load(input_npz, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        params = json.loads(str(data["params_json"])) if "params_json" in data else {}
        return {
            "source": "loaded",
            "tau_sampled": data["tau_sampled"],
            "X_sampled": data["X_sampled"],
            "Y_sampled": data["Y_sampled"],
            "noise_sampled": data["noise_sampled"],
            "metadata": metadata,
            "params": params,
        }

    p, _, _, tau_sampled, X_sampled, Y_sampled, noise_sampled, metadata = airfoil.generate_case_dataset(
        case_name=case,
        noise_level=noise_level,
        noise_kind=noise_kind,
        student_df=student_df,
        seed=seed,
        out_path=None,
    )
    return {
        "source": "generated",
        "tau_sampled": tau_sampled,
        "X_sampled": X_sampled,
        "Y_sampled": Y_sampled,
        "noise_sampled": noise_sampled,
        "metadata": metadata,
        "params": asdict(p),
    }


def select_state_columns(X: np.ndarray, state_indices: Sequence[int]) -> np.ndarray:
    return X[:, list(state_indices)]


def build_window_dataset(
    X_input: np.ndarray,
    X_target: np.ndarray,
    window_length: int = 2,
    predict_delta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input windows are taken from X_input, while targets are built from X_target.
    This allows noisy inputs with clean targets in synthetic-data experiments.
    """
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    if len(X_input) != len(X_target):
        raise ValueError("X_input and X_target must have the same number of timesteps")

    n_total, d = X_input.shape
    if n_total <= window_length:
        raise ValueError("Not enough timesteps for the requested window_length")

    windows, targets, current = [], [], []
    for n in range(window_length - 1, n_total - 1):
        w = X_input[n - window_length + 1 : n + 1].reshape(-1)
        x_n = X_target[n]
        x_next = X_target[n + 1]
        y = (x_next - x_n) if predict_delta else x_next
        windows.append(w)
        targets.append(y)
        current.append(x_n)

    return np.asarray(windows), np.asarray(targets), np.asarray(current)


def temporal_split(
    windows: np.ndarray,
    targets: np.ndarray,
    current: np.ndarray,
    train_fraction: float = 0.7,
) -> Dict[str, np.ndarray]:
    n = len(windows)
    n_train = int(np.floor(train_fraction * n))
    n_train = max(1, min(n - 1, n_train))
    return {
        "X_train": windows[:n_train],
        "Y_train": targets[:n_train],
        "C_train": current[:n_train],
        "X_test": windows[n_train:],
        "Y_test": targets[n_train:],
        "C_test": current[n_train:],
        "n_train": n_train,
    }


def estimate_bifurcation(U_tags: np.ndarray, alpha_values: np.ndarray, threshold: Optional[float] = None):
    U_unique = np.unique(U_tags)
    rms_values = []
    for U in U_unique:
        mask = U_tags == U
        rms_values.append(np.sqrt(np.mean(alpha_values[mask] ** 2)))
    rms_values = np.asarray(rms_values)

    if threshold is None:
        baseline = np.median(rms_values[: max(2, len(rms_values) // 4)])
        threshold = 3.0 * baseline

    idx = np.where(rms_values > threshold)[0]
    U_bif = U_unique[idx[0]] if len(idx) > 0 else np.nan
    return U_bif, U_unique, rms_values, threshold


# -----------------------------------------------------------------------------
# Classical baseline: NG-RC-like polynomial lift
# -----------------------------------------------------------------------------
def build_ngrc_model(degree: int = 2, alpha: float = None) -> Pipeline:
    # Fair NG-RC baseline. Keeps the nonlinear (degree-2) feature map that
    # defines NG-RC, but selects the ridge penalty by cross-validation on the
    # training set. The previous fixed alpha=1e-4 left 45 poly features vs ~44
    # training samples essentially unregularized -> overfit -> skill = -8.9,
    # an unfair (rigged-weak) baseline. CV picks alpha from the data instead.
    # NOTE: any alpha passed by the notebook cells is intentionally ignored;
    # the model cross-validates. Pass an explicit float only to force a value.
    if alpha is None:
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), fit_intercept=False)
    else:
        ridge = Ridge(alpha=alpha, fit_intercept=False)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("ridge", ridge),
        ]
    )


def _safe_clip(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(arr, lo), hi)


def autonomous_rollout_classical(
    model: Any,
    initial_window: np.ndarray,
    steps: int,
    state_dim: int,
    predict_delta: bool = True,
    input_lo: Optional[np.ndarray] = None,
    input_hi: Optional[np.ndarray] = None,
    delta_lo: Optional[np.ndarray] = None,
    delta_hi: Optional[np.ndarray] = None,
    state_lo: Optional[np.ndarray] = None,
    state_hi: Optional[np.ndarray] = None,
) -> np.ndarray:
    history = initial_window.copy().reshape(-1, state_dim)
    preds = []
    for _ in range(steps):
        x_flat = history.reshape(-1)
        x_flat = np.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
        if input_lo is not None and input_hi is not None:
            x_flat = _safe_clip(x_flat, input_lo, input_hi)
        x_flat = x_flat.reshape(1, -1)

        y_hat = model.predict(x_flat)[0]
        y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        if delta_lo is not None and delta_hi is not None:
            y_hat = _safe_clip(y_hat, delta_lo, delta_hi)

        x_next = history[-1] + y_hat if predict_delta else y_hat
        x_next = np.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0)
        if state_lo is not None and state_hi is not None:
            x_next = _safe_clip(x_next, state_lo, state_hi)

        preds.append(x_next)
        history = np.vstack([history[1:], x_next])
    return np.asarray(preds)


# -----------------------------------------------------------------------------
# QRC helpers
# -----------------------------------------------------------------------------
def qrc_scaler_fit_transform(X_train: np.ndarray, X_test: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return np.clip(X_train_s, 0.0, 1.0), np.clip(X_test_s, 0.0, 1.0), scaler


def build_local_task(qrc: QRCConfig, x_scaled: np.ndarray):
    _, Chain = get_bloqade_objects()
    geom = Chain(qrc.atom_number, lattice_spacing=qrc.lattice_spacing)
    dt = qrc.total_time / qrc.time_steps

    program = (
        geom.rydberg.rabi.amplitude.uniform.constant(duration="run_time", value=qrc.rabi_frequency)
        .detuning.uniform.constant(duration="run_time", value=qrc.encoding_scale / 2.0)
        .scale(list(x_scaled)).constant(duration="run_time", value=-qrc.encoding_scale)
    )
    return program.batch_assign(run_time=np.arange(1, qrc.time_steps + 1, 1) * dt)


def _apply_piecewise_global_detuning(program_builder: Any, durations: List[float], values: List[float]):
    """
    Best-effort helper for Bloqade piecewise-linear global detuning.
    It tries a few likely call signatures. If none works, it raises a clear error.
    """
    errors = []
    candidate_calls = [
        lambda b: b.piecewise_linear(durations, values),
        lambda b: b.piecewise_linear(durations=durations, values=values),
        lambda b: b.piecewise_linear(duration=durations, value=values),
    ]
    for fn in candidate_calls:
        try:
            return fn(program_builder)
        except Exception as exc:
            errors.append(repr(exc))

    raise RuntimeError(
        "No supported Bloqade piecewise-linear global detuning API matched. "
        "Please adjust _apply_piecewise_global_detuning() to your Bloqade version. "
        f"Tried signatures errors: {errors}"
    )


def build_global_task_reports(qrc: QRCConfig, x_scaled: np.ndarray, nshots: int):
    """
    Global-pulse encoding variant for timeseries.

    Strategy:
    - map the flattened window to a piecewise-linear global detuning waveform
    - emulate several probe times by truncating the waveform prefix at fractions of the total sequence
    - this avoids relying on batch assignment against a variable-length pulse schedule
    """
    bloqade, Chain = get_bloqade_objects()
    geom = Chain(qrc.atom_number, lattice_spacing=qrc.lattice_spacing)

    seq = np.asarray(x_scaled, dtype=float)
    if len(seq) < 2:
        seq = np.concatenate([seq, seq])

    segment_duration = qrc.total_time / len(seq)
    durations_full = [segment_duration] * (len(seq) - 1)
    values_full = list(qrc.pulse_bias + qrc.encoding_scale * (2.0 * seq - 1.0))

    reports = []
    for probe_idx in range(1, qrc.time_steps + 1):
        prefix_len = max(2, int(np.ceil(len(seq) * probe_idx / qrc.time_steps)))
        durations = durations_full[: prefix_len - 1]
        values = values_full[:prefix_len]

        base = geom.rydberg.rabi.amplitude.uniform.constant(
            duration=float(np.sum(durations)), value=qrc.rabi_frequency
        )
        program = _apply_piecewise_global_detuning(base.detuning.uniform, durations, values)
        report = program.bloqade.python().run(shots=nshots, rtol=1e-8, atol=1e-8).report()
        reports.append(report)
    return reports


def _coerce_bitstrings_to_array(bits: Any, natoms: int) -> np.ndarray:
    """Normalize Bloqade bitstring containers to a numeric (shots, natoms) array."""
    if isinstance(bits, np.ndarray):
        arr = bits
    elif isinstance(bits, (list, tuple)):
        if len(bits) == 0:
            return np.zeros((0, natoms), dtype=float)
        # Some Bloqade report objects wrap a single timestep as [array(...)]
        if len(bits) == 1 and isinstance(bits[0], np.ndarray):
            arr = bits[0]
        # Common case: list of strings like ['0101', '1110', ...]
        elif isinstance(bits[0], str):
            arr = np.asarray([[1.0 if ch == '1' else 0.0 for ch in s.strip()] for s in bits], dtype=float)
        else:
            arr = np.asarray(bits, dtype=float)
    else:
        arr = np.asarray(bits, dtype=float)

    if arr.ndim == 1:
        # Single bitstring encoded as length-natoms vector
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        # Wrapped single timestep
        arr = arr[0]

    if arr.shape[-1] != natoms:
        raise ValueError(f"Bitstring array has incompatible shape {arr.shape}; expected last dim {natoms}.")
    return arr.astype(float, copy=False)


def process_results_from_bitstring_list(qrc: QRCConfig, bitstrings_by_time: List[Any]) -> np.ndarray:
    embedding: List[float] = []
    natoms = qrc.atom_number
    for bits in bitstrings_by_time:
        bits_arr = _coerce_bitstrings_to_array(bits, natoms)
        if bits_arr.shape[0] == 0:
            # Fallback to zeros if no shots are present
            for _ in range(natoms):
                embedding.append(0.0)
            if qrc.readouts == "ZZ":
                for i in range(natoms):
                    for j in range(i + 1, natoms):
                        embedding.append(0.0)
            continue
        ar1 = -1.0 + 2.0 * bits_arr
        nshots = ar1.shape[0]
        for i in range(natoms):
            embedding.append(np.sum(ar1[:, i]) / nshots)
        if qrc.readouts == "ZZ":
            for i in range(natoms):
                for j in range(i + 1, natoms):
                    embedding.append(np.sum(ar1[:, i] * ar1[:, j]) / nshots)
    return np.asarray(embedding, dtype=float)


def process_local_report(qrc: QRCConfig, report: Any) -> np.ndarray:
    bitstrings_by_time = [report.bitstrings()[t] for t in range(qrc.time_steps)]
    return process_results_from_bitstring_list(qrc, bitstrings_by_time)


def emulate_qrc_embeddings(qrc: QRCConfig, X_scaled: np.ndarray, nshots: int) -> np.ndarray:
    embeddings = []
    for k in range(X_scaled.shape[0]):
        x = X_scaled[k]
        if qrc.encoding == "local":
            report = build_local_task(qrc, x).bloqade.python().run(shots=nshots, rtol=1e-8, atol=1e-8).report()
            emb = process_local_report(qrc, report)
            del report
        elif qrc.encoding == "global":
            reports = build_global_task_reports(qrc, x, nshots=nshots)
            bitstrings_by_time = [rep.bitstrings() for rep in reports]
            emb = process_results_from_bitstring_list(qrc, bitstrings_by_time)
            del reports, bitstrings_by_time
        else:
            raise ValueError(f"Unsupported encoding: {qrc.encoding}")
        embeddings.append(np.asarray(emb, dtype=np.float32))
        if (k + 1) % 5 == 0:
            gc.collect()
    return np.asarray(embeddings, dtype=np.float32)


def autonomous_rollout_qrc(
    qrc: QRCConfig,
    model: Any,
    scaler: MinMaxScaler,
    initial_window: np.ndarray,
    steps: int,
    state_dim: int,
    nshots: int,
    predict_delta: bool = True,
) -> np.ndarray:
    history = initial_window.copy().reshape(-1, state_dim)
    preds = []
    for _ in range(steps):
        x_flat = history.reshape(1, -1)
        x_scaled = np.clip(scaler.transform(x_flat), 0.0, 1.0)
        emb = emulate_qrc_embeddings(qrc, x_scaled, nshots=nshots)
        y_hat = model.predict(emb)[0]
        x_next = history[-1] + y_hat if predict_delta else y_hat
        preds.append(x_next)
        history = np.vstack([history[1:], x_next])
    return np.asarray(preds)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_time_history(
    tau: np.ndarray,
    true_states: np.ndarray,
    qrc_states: np.ndarray,
    ngrc_states: np.ndarray,
    state_names: Sequence[str],
    out_path: Path,
):
    n_states = true_states.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(tau, true_states[:, i], label="Ground truth")
        ax.plot(tau, qrc_states[:, i], label="QRC rollout", linestyle="--")
        ax.plot(tau, ngrc_states[:, i], label="NG-RC rollout", linestyle=":")
        ax.set_ylabel(state_names[i])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_psd_comparison(
    true_states: np.ndarray,
    qrc_states: np.ndarray,
    ngrc_states: np.ndarray,
    fs: float,
    state_names: Sequence[str],
    out_path: Path,
):
    n_states = true_states.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        f_true, p_true = welch(true_states[:, i], fs=fs, nperseg=min(256, len(true_states)))
        f_qrc, p_qrc = welch(qrc_states[:, i], fs=fs, nperseg=min(256, len(qrc_states)))
        f_ng, p_ng = welch(ngrc_states[:, i], fs=fs, nperseg=min(256, len(ngrc_states)))
        ax.semilogy(f_true, p_true, label="Ground truth")
        ax.semilogy(f_qrc, p_qrc, label="QRC rollout", linestyle="--")
        ax.semilogy(f_ng, p_ng, label="NG-RC rollout", linestyle=":")
        ax.set_ylabel(f"PSD {state_names[i]}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frequency")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def choose_phase_pairs(state_indices: Sequence[int]) -> List[Tuple[int, int]]:
    idx = list(state_indices)
    pairs = []
    if 0 in idx and 1 in idx:
        pairs.append((idx.index(0), idx.index(1)))
    if 2 in idx and 3 in idx:
        pairs.append((idx.index(2), idx.index(3)))
    if not pairs:
        if len(idx) >= 2:
            pairs.append((0, 1))
    return pairs[:2]


def plot_phase_portraits(
    true_states: np.ndarray,
    qrc_states: np.ndarray,
    ngrc_states: np.ndarray,
    state_names: Sequence[str],
    phase_pairs: List[Tuple[int, int]],
    out_path: Path,
):
    n_pairs = len(phase_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4))
    if n_pairs == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, phase_pairs):
        ax.plot(true_states[:, i], true_states[:, j], label="Ground truth")
        ax.plot(qrc_states[:, i], qrc_states[:, j], label="QRC rollout", linestyle="--")
        ax.plot(ngrc_states[:, i], ngrc_states[:, j], label="NG-RC rollout", linestyle=":")
        ax.set_xlabel(state_names[i])
        ax.set_ylabel(state_names[j])
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Aeroelastic QRC v2 with plots and NG-RC baseline")
    parser.add_argument("--input-npz", type=str, default=None)
    parser.add_argument("--case", type=str, default="IV", choices=["I", "II", "III", "IV", "V"])
    parser.add_argument("--noise-level", type=float, default=0.40)
    parser.add_argument("--noise-kind", type=str, default="gaussian", choices=["gaussian", "student_t", "none"])
    parser.add_argument("--student-df", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-noisy-input", type=int, default=0)
    parser.add_argument("--state-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--window-length", type=int, default=2)
    parser.add_argument("--predict-delta", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--encoding", type=str, default="local", choices=["local", "global"])
    parser.add_argument("--qrc-atoms", type=int, default=None, help="If omitted, use flattened input dimension")
    parser.add_argument("--nshots", type=int, default=200)
    parser.add_argument("--qrc-time-steps", type=int, default=8)
    parser.add_argument("--qrc-total-time", type=float, default=4.0)
    parser.add_argument("--qrc-encoding-scale", type=float, default=9.0)
    parser.add_argument("--qrc-rabi-frequency", type=float, default=6.283)
    parser.add_argument("--qrc-readouts", type=str, default="Z", choices=["Z", "ZZ"])
    parser.add_argument("--ngrc-degree", type=int, default=2)
    parser.add_argument("--ngrc-alpha", type=float, default=1e-3)
    parser.add_argument("--readout-alpha", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=80)
    parser.add_argument("--max-test-samples", type=int, default=40)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--make-plots", type=int, default=1)
    parser.add_argument("--out-prefix", type=str, default="airfoil_qrc_v3")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    ds = load_or_generate_dataset(
        input_npz=args.input_npz,
        case=args.case,
        noise_level=args.noise_level,
        noise_kind=args.noise_kind,
        student_df=args.student_df,
        seed=args.seed,
    )

    X_clean = select_state_columns(ds["X_sampled"], args.state_indices)
    X_noisy = select_state_columns(ds["Y_sampled"], args.state_indices)
    X_input = X_noisy if args.use_noisy_input else X_clean
    X_target = X_clean

    windows, targets, current = build_window_dataset(
        X_input=X_input,
        X_target=X_target,
        window_length=args.window_length,
        predict_delta=bool(args.predict_delta),
    )
    split = temporal_split(windows, targets, current, train_fraction=args.train_fraction)

    # Optional downselection to keep Bloqade emulation tractable on a laptop/desktop.
    def _downselect(X, Y, C, stride, max_samples):
        Xs = X[::max(1, stride)]
        Ys = Y[::max(1, stride)]
        Cs = C[::max(1, stride)]
        if max_samples is not None and len(Xs) > max_samples:
            Xs = Xs[:max_samples]
            Ys = Ys[:max_samples]
            Cs = Cs[:max_samples]
        return Xs, Ys, Cs

    split["X_train"], split["Y_train"], split["C_train"] = _downselect(
        split["X_train"], split["Y_train"], split["C_train"], args.train_stride, args.max_train_samples
    )
    split["X_test"], split["Y_test"], split["C_test"] = _downselect(
        split["X_test"], split["Y_test"], split["C_test"], args.test_stride, args.max_test_samples
    )

    state_dim = len(args.state_indices)
    input_dim = split["X_train"].shape[1]
    atom_number = args.qrc_atoms if args.qrc_atoms is not None else input_dim

    if args.encoding == "local" and atom_number != input_dim:
        raise ValueError(
            "For local encoding, qrc-atoms must equal the flattened input dimension. "
            f"Got atom_number={atom_number}, input_dim={input_dim}."
        )

    qrc = QRCConfig(
        atom_number=atom_number,
        encoding=args.encoding,
        total_time=args.qrc_total_time,
        time_steps=args.qrc_time_steps,
        encoding_scale=args.qrc_encoding_scale,
        rabi_frequency=args.qrc_rabi_frequency,
        readouts=args.qrc_readouts,
    )

    # ---------------------------
    # NG-RC baseline
    # ---------------------------
    ngrc = build_ngrc_model(degree=args.ngrc_degree, alpha=args.ngrc_alpha)
    X_train_finite = np.nan_to_num(split["X_train"], nan=0.0, posinf=0.0, neginf=0.0)
    Y_train_finite = np.nan_to_num(split["Y_train"], nan=0.0, posinf=0.0, neginf=0.0)
    X_test_finite = np.nan_to_num(split["X_test"], nan=0.0, posinf=0.0, neginf=0.0)
    ngrc.fit(X_train_finite, Y_train_finite)
    Yhat_ngrc = ngrc.predict(X_test_finite)

    # Robust clipping envelopes for autoregressive rollout of the classical baseline.
    x_train_lo = np.nanmin(X_train_finite, axis=0)
    x_train_hi = np.nanmax(X_train_finite, axis=0)
    x_pad = 0.25 * np.maximum(1.0, x_train_hi - x_train_lo)
    x_train_lo = x_train_lo - x_pad
    x_train_hi = x_train_hi + x_pad

    y_train_lo = np.nanmin(Y_train_finite, axis=0)
    y_train_hi = np.nanmax(Y_train_finite, axis=0)
    y_pad = 0.50 * np.maximum(1.0, y_train_hi - y_train_lo)
    y_train_lo = y_train_lo - y_pad
    y_train_hi = y_train_hi + y_pad

    c_train_lo = np.nanmin(split["C_train"], axis=0)
    c_train_hi = np.nanmax(split["C_train"], axis=0)
    c_pad = 0.50 * np.maximum(1.0, c_train_hi - c_train_lo)
    c_train_lo = c_train_lo - c_pad
    c_train_hi = c_train_hi + c_pad

    # ---------------------------
    # QRC embeddings + readout
    # ---------------------------
    X_train_s, X_test_s, qrc_scaler = qrc_scaler_fit_transform(split["X_train"], split["X_test"])
    E_train = emulate_qrc_embeddings(qrc, X_train_s, nshots=args.nshots)
    E_test = emulate_qrc_embeddings(qrc, X_test_s, nshots=args.nshots)

    qrc_readout = Ridge(alpha=args.readout_alpha, fit_intercept=True)
    qrc_readout.fit(E_train, split["Y_train"])
    Yhat_qrc = qrc_readout.predict(E_test)

    # ---------------------------
    # One-step metrics
    # ---------------------------
    mse_qrc_total = mean_squared_error(split["Y_test"], Yhat_qrc)
    mse_ngrc_total = mean_squared_error(split["Y_test"], Yhat_ngrc)

    # alpha one-step metric if alpha is included among selected states
    alpha_local_index = None
    if 0 in args.state_indices:
        alpha_local_index = list(args.state_indices).index(0)
        mse_qrc_alpha = mean_squared_error(split["Y_test"][:, alpha_local_index], Yhat_qrc[:, alpha_local_index])
        mse_ngrc_alpha = mean_squared_error(split["Y_test"][:, alpha_local_index], Yhat_ngrc[:, alpha_local_index])
    else:
        mse_qrc_alpha = np.nan
        mse_ngrc_alpha = np.nan

    # ---------------------------
    # Autonomous rollout on the test horizon
    # ---------------------------
    n_roll = len(split["X_test"])
    initial_window = split["X_test"][0]

    rollout_qrc = autonomous_rollout_qrc(
        qrc=qrc,
        model=qrc_readout,
        scaler=qrc_scaler,
        initial_window=initial_window,
        steps=n_roll,
        state_dim=state_dim,
        nshots=args.nshots,
        predict_delta=bool(args.predict_delta),
    )
    rollout_ngrc = autonomous_rollout_classical(
        model=ngrc,
        initial_window=initial_window,
        steps=n_roll,
        state_dim=state_dim,
        predict_delta=bool(args.predict_delta),
        input_lo=x_train_lo,
        input_hi=x_train_hi,
        delta_lo=y_train_lo,
        delta_hi=y_train_hi,
        state_lo=c_train_lo,
        state_hi=c_train_hi,
    )

    true_rollout = X_target[args.window_length + split["n_train"] : args.window_length + split["n_train"] + n_roll]
    tau_rollout = ds["tau_sampled"][args.window_length + split["n_train"] : args.window_length + split["n_train"] + n_roll]

    mse_rollout_qrc = mean_squared_error(true_rollout, rollout_qrc)
    mse_rollout_ngrc = mean_squared_error(true_rollout, rollout_ngrc)

    # bifurcation estimate only if alpha available and U_tags can be formed from metadata
    # here, for a single-case simulation, U_tags is constant and not informative.
    # keep fields for future multi-case sweeps.
    results = {
        "case": args.case,
        "encoding": args.encoding,
        "state_indices": list(args.state_indices),
        "window_length": args.window_length,
        "predict_delta": bool(args.predict_delta),
        "nshots": args.nshots,
        "qrc_atom_number": atom_number,
        "qrc_embedding_dim": int(E_train.shape[1]),
        "mse_one_step_qrc_total": float(mse_qrc_total),
        "mse_one_step_ngrc_total": float(mse_ngrc_total),
        "mse_one_step_qrc_alpha": float(mse_qrc_alpha),
        "mse_one_step_ngrc_alpha": float(mse_ngrc_alpha),
        "mse_rollout_qrc_total": float(mse_rollout_qrc),
        "mse_rollout_ngrc_total": float(mse_rollout_ngrc),
    }

    out_prefix = Path(args.out_prefix)
    out_npz = out_prefix.with_suffix(".npz")
    np.savez(
        out_npz,
        tau_rollout=tau_rollout,
        true_rollout=true_rollout,
        rollout_qrc=rollout_qrc,
        rollout_ngrc=rollout_ngrc,
        Y_test=split["Y_test"],
        Yhat_qrc=Yhat_qrc,
        Yhat_ngrc=Yhat_ngrc,
        E_train=E_train,
        E_test=E_test,
        metadata_json=json.dumps(ds["metadata"]),
        params_json=json.dumps(ds["params"]),
        results_json=json.dumps(results),
    )

    if args.make_plots:
        state_names = [STATE_NAMES[i] for i in args.state_indices]
        dt_sample = float(np.median(np.diff(ds["tau_sampled"])))
        fs = 1.0 / dt_sample if dt_sample > 0 else 1.0
        plot_time_history(
            tau=tau_rollout,
            true_states=true_rollout,
            qrc_states=rollout_qrc,
            ngrc_states=rollout_ngrc,
            state_names=state_names,
            out_path=out_prefix.with_name(out_prefix.name + "_time_history.png"),
        )
        plot_psd_comparison(
            true_states=true_rollout,
            qrc_states=rollout_qrc,
            ngrc_states=rollout_ngrc,
            fs=fs,
            state_names=state_names,
            out_path=out_prefix.with_name(out_prefix.name + "_psd.png"),
        )
        phase_pairs = choose_phase_pairs(args.state_indices)
        plot_phase_portraits(
            true_states=true_rollout,
            qrc_states=rollout_qrc,
            ngrc_states=rollout_ngrc,
            state_names=state_names,
            phase_pairs=phase_pairs,
            out_path=out_prefix.with_name(out_prefix.name + "_phase.png"),
        )

    print(json.dumps(results, indent=2))
    print(f"Saved results to {out_npz}")
    if args.make_plots:
        print(f"Saved plots to {out_prefix.with_name(out_prefix.name + '_time_history.png')}")
        print(f"Saved plots to {out_prefix.with_name(out_prefix.name + '_psd.png')}")
        print(f"Saved plots to {out_prefix.with_name(out_prefix.name + '_phase.png')}")


if __name__ == "__main__":
    main()

"""hw_common.py — Módulo común de la corrida de hardware Aquila (diseño PRERREGISTRADO, congelado).

Diseño: 8 átomos, cadena 1D a=10 µm, encoding local Δ_i = 4.5 − 9·x_i rad/µs,
Ω = 6.283 rad/µs, readout Z, time_steps=1, probes {0.5, 0.70, 1.8} µs,
66 anchors (semilla 1234, ventanas denoised de data_cache.npz, window=2,
MinMax ajustado SOLO en los 44 anchors de train), 100 shots/tarea.

Desviación documentada (restricción de Aquila) + ENMIENDA E1 (confirmada por Juan):
el detuning local y Ω deben iniciar y terminar en 0 → perfil rampa-meseta-rampa.
E1: rampa mínima legal RAMP_US = 0.05 µs (límites: slew Ω ≤ 250 rad/µs², paso ≥ 50 ns)
y COMPENSACIÓN DE ÁREA: tiempo programado = probe + RAMP_US, de modo que el tiempo
efectivo (área del trapecio / Ω_max = t_prog − t_ramp) sea exactamente el probe físico
prerregistrado. Programados {0.55, 0.75, 1.85} → efectivos {0.5, 0.70, 1.8} µs.
Validación mock: skill nodo +0.068 vs emu100 +0.047; ρ(ramp↔emu500)=0.955 ≈ línea
base de muestreo. La rampa v15 (min(0.2, tt/4)) destruía el nodo (contraste −0.011).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------- Diseño congelado ----------------
CASE = "IV"
SEED = 1234
PROBES_US = [0.5, 0.7, 1.8]   # probes FÍSICOS (tiempo efectivo)
RAMP_US = 0.05                # E1: rampa mínima legal
AMENDMENT = "E1_ramp0.05_area_comp"
N_ATOMS = 8
SPACING_UM = 10.0
ENCODING_SCALE = 9.0        # rad/us  -> Delta_i = ENC/2 - ENC*x_i = 4.5 - 9*x_i
RABI = 6.283                # rad/us
TIME_STEPS = 1
READOUT = "Z"
NSHOTS = 100
STATE_INDICES = [0, 1, 2, 3]
WINDOW = 2
N_TRAIN, N_TEST = 44, 22
BURN_IN_TU = 500.0
WIN_STRIDE_TU = 5.0
EVAL_FRACS = [1.0, 2.0, 3.0]
AQUILA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"

ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "aquila_hw_tasks"      # manifiestos por tarea (ARN + metadatos)
RESULT_DIR = ROOT / "aquila_hw_results"  # shots crudos por tarea
LOG_PATH = ROOT / "aquila_hw_submit.log"


# ---------------- Datos: anchors y ventanas (idéntico a run_generic/verify) ----------------
def _char_time_tu(x, dt, max_lag_tu=150.0):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    ml = min(n - 1, int(round(max_lag_tu / dt)))
    ac = np.correlate(x, x, "full")[n - 1: n - 1 + ml]
    ac = ac / ac[0]
    return float(np.where(ac < 1.0 / np.e)[0][0] * dt)

def load_protocol(data_npz="data_cache.npz"):
    d = np.load(ROOT / data_npz)
    XC = d["XC"]; dt = float(d["dt"])
    i0 = int(round(BURN_IN_TU / dt))
    char_tu = _char_time_tu(XC[i0:i0 + min(int(round(600 / dt)), len(XC) - i0), 0], dt)
    assert abs(char_tu - 10.3) < 0.5, f"tau_c inesperado: {char_tu}"
    spacing = max(1, int(round(char_tu / dt)))
    wstride = max(1, int(round(WIN_STRIDE_TU / dt)))
    H_eval = sorted(set(max(1, int(round(f * char_tu / dt))) for f in EVAL_FRACS))
    H_max = max(H_eval)
    n0 = N_TRAIN + N_TEST
    base = i0 + (WINDOW - 1) * wstride
    anchors = base + np.arange(n0) * spacing
    anchors = anchors[(anchors + H_max < len(XC)) & (anchors - (WINDOW - 1) * wstride >= 0)]
    assert len(anchors) == 66, f"anchors={len(anchors)} (esperado 66)"
    Xin = d[f"dn_{SEED}"][:, STATE_INDICES]
    W = np.asarray([Xin[a - (WINDOW - 1 - np.arange(WINDOW)) * wstride].reshape(-1) for a in anchors])
    tr = np.arange(0, len(anchors) - N_TEST)
    te = np.arange(len(anchors) - N_TEST, len(anchors))
    Wsc = np.clip(MinMaxScaler((0.0, 1.0)).fit(W[tr]).transform(W), 0.0, 1.0)
    return dict(XC=XC, dt=dt, char_tu=char_tu, anchors=anchors, H_eval=H_eval,
                tr=tr, te=te, W=W, Wsc=Wsc,
                Xc_sel=XC[:, STATE_INDICES])


# ---------------- Programa AHS nativo (ENMIENDA E1: rampa 0.05 + compensación de área) ----------------
def build_native_ahs(x_scaled, total_time_us):
    """total_time_us es el probe FÍSICO (tiempo efectivo). El tiempo programado es
    total_time_us + RAMP_US, con rampas de RAMP_US en Ω y detuning local, de modo que
    el área del trapecio dividida por el valor de meseta = total_time_us."""
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.ahs.local_detuning import LocalDetuning
    from braket.ahs.field import Field
    from braket.ahs.pattern import Pattern
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.timings.time_series import TimeSeries

    x_scaled = np.asarray(x_scaled, dtype=float)
    assert x_scaled.shape == (N_ATOMS,)
    pattern_values = np.clip(x_scaled, 0.0, 1.0).tolist()

    time_ramp = RAMP_US * 1e-6
    time_max = (float(total_time_us) + RAMP_US) * 1e-6   # compensación de área
    assert time_max > 2.0 * time_ramp

    omega_max = RABI * 1e6
    delta_global = (ENCODING_SCALE / 2.0) * 1e6
    delta_local = -abs(ENCODING_SCALE * 1e6)

    register = AtomArrangement()
    for k in range(N_ATOMS):
        register.add((k * SPACING_UM * 1e-6, 0.0))

    omega = TimeSeries.from_lists(
        times=[0.0, time_ramp, time_max - time_ramp, time_max],
        values=[0.0, omega_max, omega_max, 0.0])
    phase = TimeSeries.from_lists(times=[0.0, time_max], values=[0.0, 0.0])
    detuning = TimeSeries.from_lists(times=[0.0, time_max], values=[delta_global, delta_global])
    local_ts = TimeSeries.from_lists(
        times=[0.0, time_ramp, time_max - time_ramp, time_max],
        values=[0.0, delta_local, delta_local, 0.0])

    drive = DrivingField(amplitude=omega, phase=phase, detuning=detuning)
    local_shift = LocalDetuning(magnitude=Field(time_series=local_ts, pattern=Pattern(pattern_values)))
    return AnalogHamiltonianSimulation(register=register, hamiltonian=drive + local_shift)


# ---------------- Parsing ----------------
# CORRECCIÓN C7 (bug heredado de v15): bloqade codifica ground=1 en bitstrings y el
# emulador computa <Z_i> = mean(-1 + 2*bit) -> ground=+1. v15 usaba 1-post (ground=0),
# invirtiendo el signo de los embeddings de hardware respecto al emulador y por tanto
# el signo de rho(hw<->emu). Aquí bit = post (ground=1, Rydberg=0), igual que bloqade.
# Verificación empírica: con Omega=0 ambos backends reportan 1 (ground) por átomo.
def ahs_result_to_rydberg_bits(result, natoms=N_ATOMS):
    """Bits en convención bloqade (1=ground, 0=Rydberg). Descarta shots con
    status != Success o pre_sequence imperfecta (átomo no cargado)."""
    rows, n_total, n_badstatus, n_badfill = [], 0, 0, 0
    for shot in result.measurements:
        n_total += 1
        if "Success" not in str(getattr(shot, "status", "")):
            n_badstatus += 1
            continue
        pre = np.asarray(shot.pre_sequence, dtype=int)
        post = np.asarray(shot.post_sequence, dtype=int)
        if pre.shape[0] != natoms or post.shape[0] != natoms or not np.all(pre == 1):
            n_badfill += 1
            continue
        rows.append(post)
    bits = np.asarray(rows, dtype=int) if rows else np.zeros((0, natoms), dtype=int)
    stats = dict(n_total=n_total, n_used=len(rows), n_badstatus=n_badstatus, n_badfill=n_badfill)
    return bits, stats


def bits_to_embedding(bits):
    """Readout Z: <Z_i> por átomo, convención ar1 = -1 + 2*bit (idéntica al emulador)."""
    if bits.shape[0] == 0:
        return np.zeros(N_ATOMS)
    ar1 = -1.0 + 2.0 * bits.astype(float)
    return ar1.mean(axis=0)


# ---------------- Persistencia ----------------
def task_tag(tt, anchor_idx):
    return f"aquila_{CASE}_s{SEED}_tt{str(tt).replace('.', 'p')}_a{anchor_idx:02d}"

def save_manifest(tag, payload):
    TASK_DIR.mkdir(exist_ok=True, parents=True)
    with open(TASK_DIR / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_manifest(tag):
    p = TASK_DIR / f"{tag}.json"
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))

def save_raw_result(tag, arn, result, mode):
    """Shots crudos a disco inmediatamente: pre/post sequences + status por shot."""
    RESULT_DIR.mkdir(exist_ok=True, parents=True)
    pre, post, status = [], [], []
    for shot in result.measurements:
        status.append(str(getattr(shot, "status", "")))
        pre.append(np.asarray(shot.pre_sequence, dtype=int))
        post.append(np.asarray(shot.post_sequence, dtype=int))
    np.savez(RESULT_DIR / f"{tag}_raw.npz",
             arn=np.array(arn), mode=np.array(mode),
             pre=np.asarray(pre), post=np.asarray(post),
             status=np.array(status))

def load_raw_bits(tag):
    """Reconstruye bits (convención bloqade: 1=ground) desde los shots crudos en disco.
    Los .npz guardan pre/post SIN transformar, así que la corrección C7 vive solo aquí."""
    d = np.load(RESULT_DIR / f"{tag}_raw.npz")
    rows = []
    for s, pre, post in zip(d["status"], d["pre"], d["post"]):
        if "Success" not in str(s):
            continue
        if not np.all(pre == 1):
            continue
        rows.append(post)
    bits = np.asarray(rows, dtype=int) if rows else np.zeros((0, N_ATOMS), dtype=int)
    return bits

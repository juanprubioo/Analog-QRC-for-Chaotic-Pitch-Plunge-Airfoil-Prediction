"""exp_ramp_scan.py — Experimento GRATIS (no toca el diseño congelado ni el path de envío).

Pregunta: ¿el nodo de anti-resonancia a 0.70 µs sobrevive con la rampa mínima legal de
Aquila (0.05 µs), en vez de la rampa conservadora de v15 (min(0.2, tt/4))?

Corre el simulador AHS local con waveform rampa-meseta-rampa parametrizada por RAMP_US
y computa la skill del protocolo estándar. Resultados a exp_ramp/<ramp>/tt<tt>.npy.

Uso: python3 exp_ramp_scan.py <ramp_us> <tt_us> [max_seconds]
"""
import sys, time
from pathlib import Path
import numpy as np
import warnings; warnings.filterwarnings("ignore")

import hw_common as hw
from braket.devices import LocalSimulator
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.ahs.local_detuning import LocalDetuning
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.timings.time_series import TimeSeries

RAMP_US = float(sys.argv[1])
TT = float(sys.argv[2])
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9

def build(x_scaled, tt_us, ramp_us):
    time_max = tt_us * 1e-6
    tr = min(ramp_us * 1e-6, time_max / 2.0 * 0.999)
    omega_max = hw.RABI * 1e6
    dg = (hw.ENCODING_SCALE / 2.0) * 1e6
    dl = -abs(hw.ENCODING_SCALE * 1e6)
    reg = AtomArrangement()
    for k in range(hw.N_ATOMS):
        reg.add((k * hw.SPACING_UM * 1e-6, 0.0))
    om = TimeSeries.from_lists(times=[0.0, tr, max(tr, time_max - tr), time_max],
                               values=[0.0, omega_max, omega_max, 0.0])
    ph = TimeSeries.from_lists(times=[0.0, time_max], values=[0.0, 0.0])
    de = TimeSeries.from_lists(times=[0.0, time_max], values=[dg, dg])
    lts = TimeSeries.from_lists(times=[0.0, tr, max(tr, time_max - tr), time_max],
                                values=[0.0, dl, dl, 0.0])
    drive = DrivingField(amplitude=om, phase=ph, detuning=de)
    lsh = LocalDetuning(magnitude=Field(time_series=lts, pattern=Pattern(np.clip(x_scaled, 0, 1).tolist())))
    return AnalogHamiltonianSimulation(register=reg, hamiltonian=drive + lsh)

proto = hw.load_protocol()
Wsc = proto["Wsc"]
outdir = Path(f"exp_ramp/r{str(RAMP_US).replace('.','p')}")
outdir.mkdir(parents=True, exist_ok=True)
fp = outdir / f"tt{str(TT).replace('.','p')}.npy"
E = np.load(fp) if fp.exists() else np.full((66, hw.N_ATOMS), np.nan)

sim = LocalSimulator("braket_ahs")
t0 = time.time()
for idx in range(66):
    if not np.isnan(E[idx]).any():
        continue
    if time.time() - t0 > BUDGET:
        print(f"BUDGET: {(~np.isnan(E).any(axis=1)).sum()}/66 listos", flush=True)
        sys.exit(0)
    res = sim.run(build(Wsc[idx], TT, RAMP_US), shots=hw.NSHOTS).result()
    bits, _ = hw.ahs_result_to_rydberg_bits(res)
    E[idx] = hw.bits_to_embedding(bits)
    np.save(fp, E)
print(f"DONE ramp={RAMP_US} tt={TT}: 66/66 en {time.time()-t0:.0f}s", flush=True)

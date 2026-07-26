"""step1_emu100_baseline.py — Paso 1 del protocolo de hardware (prerregistrado).

Computa, para los 3 probes {0.5, 0.70, 1.8} µs, semilla 1234, 66 anchors:
  (i)  skills emu-100 y emu-500 (protocolo estándar 44/22, H={1,2,3}tau_c,
       targets delta sobre estados limpios, readout RidgeCV, pooled + CI bootstrap 3000)
  (ii) rho_probe = corr(emb_100, emb_500) aplanado sobre anchors x atomos
       (línea base de consistencia: puro efecto de muestreo 100 vs 500 shots)

Salida: step1_baseline.csv + impresión de tabla.
Uso: python3 step1_emu100_baseline.py
"""
import json, pickle, subprocess, sys
import numpy as np
import pandas as pd

PROBES = [0.5, 0.7, 1.8]
SEED = 1234
KEY = lambda tt, ns: ("IV", tt, SEED, 8, 1, "Z", ns, (0, 1, 2, 3), 2, 10.0, 66)

cache = pickle.load(open("emb_cache.pkl", "rb"))

rows = []
for tt in PROBES:
    e100 = np.asarray(cache[KEY(tt, 100)], float)
    e500 = np.asarray(cache[KEY(tt, 500)], float)
    assert e100.shape == e500.shape == (66, 8), (tt, e100.shape, e500.shape)
    rho = float(np.corrcoef(e100.ravel(), e500.ravel())[0, 1])
    rows.append({"probe_us": tt, "rho_emu100_emu500": rho})
rho_df = pd.DataFrame(rows)

def skills(nshots):
    cfg = {"state_indices": [0, 1, 2, 3], "window": 2, "time_steps": 1,
           "readout": "Z", "tts": PROBES, "seeds": [SEED], "nshots": nshots}
    out = subprocess.run([sys.executable, "analyze_generic.py", json.dumps(cfg)],
                         capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if l.strip() and l.lstrip()[0].isdigit()]
    recs = []
    for l in lines:
        v = l.split()
        recs.append({"probe_us": float(v[0]), f"skill_emu{nshots}": float(v[3]),
                     f"ci_lo_{nshots}": float(v[4]), f"ci_hi_{nshots}": float(v[5])})
    return pd.DataFrame(recs)

df = skills(100).merge(skills(500), on="probe_us").merge(rho_df, on="probe_us")
c100 = df.loc[df.probe_us == 0.5, "skill_emu100"].iloc[0] - df.loc[df.probe_us == 0.7, "skill_emu100"].iloc[0]
c500 = df.loc[df.probe_us == 0.5, "skill_emu500"].iloc[0] - df.loc[df.probe_us == 0.7, "skill_emu500"].iloc[0]

pd.set_option("display.float_format", lambda v: f"{v:+.3f}")
print(df.to_string(index=False))
print(f"\nContraste S(0.5)-S(0.70):  emu100 = {c100:+.3f}   emu500 = {c500:+.3f}")
df.to_csv("step1_baseline.csv", index=False)
print("Guardado: step1_baseline.csv")

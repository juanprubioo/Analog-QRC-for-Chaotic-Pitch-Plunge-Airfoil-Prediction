#!/usr/bin/env python3
"""export_totaltime_csv.py — Regenera qrc_totaltime_IV_v2.csv (Tabla 2 del manuscrito).

Coloca este archivo en emulator/ y ejecuta:   python3 export_totaltime_csv.py

Qué hace
--------
Provee el contexto que `cell_4_4_v2.py` esperaba del notebook y lo ejecuta tal cual.
Los 30 embeddings del barrido (5 total_time x 6 semillas) ya están en
results/qrc_totaltime_emb_IV.pkl, así que NO se emula nada: si faltara alguna clave
el script aborta en vez de recalcular, para que el CSV no pueda salir de una
configuración distinta de la que produjo la tabla publicada.

Salida
------
results/qrc_totaltime_IV_v2.csv  y una verificación contra la Tabla 2 del manuscrito.
"""
import sys, json, hashlib, pickle, importlib.util
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent          # emulator/
REPO = HERE.parent
ROOT = HERE / "results"                          # donde vive el .pkl y saldrá el .csv

# ---------- núcleo compartido del proyecto ----------
def _imp(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

q5 = _imp("q5_export", REPO / "airfoil_qrc_v5.py")
load_or_generate_dataset = q5.load_or_generate_dataset
build_ngrc_model = q5.build_ngrc_model

# ---------- constantes del estudio (idénticas a verify.py y hw_common.py) ----------
CASE, SEED = "IV", 1234
NOISE_LEVEL, NOISE_KIND, STUDENT_DF = 0.40, "gaussian", 5
USE_NOISY_INPUT, PREDICT_DELTA = True, True
QRC_LATTICE_SPACING, QRC_ENCODING_SCALE, QRC_RABI_FREQUENCY = 10.0, 9.0, 6.283
_chaotic = True
DYNAMICS = "auto"

# ---------- trayectoria limpia ----------
cache_npz = REPO / "data_cache.npz"
if cache_npz.exists():
    _d = np.load(cache_npz); XC = _d["XC"]; dt = float(_d["dt"])
else:
    clean = load_or_generate_dataset(input_npz=None, case=CASE, noise_level=0.0,
                                     noise_kind="none", student_df=STUDENT_DF, seed=SEED)
    XC = np.asarray(clean["X_sampled"], float)
    dt = float(clean["metadata"]["sampling_interval"])

def _char_time_tu(x, dt_, max_lag_tu=150.0):
    x = np.asarray(x, float) - np.mean(x)
    ac = np.correlate(x, x, "full")[len(x)-1:len(x)-1+int(round(max_lag_tu/dt_))]
    ac = ac / ac[0]
    idx = np.where(ac < 1/np.e)[0]
    return float(idx[0] * dt_) if len(idx) else None

# ---------- el emulador no debe invocarse: todo está cacheado ----------
class QRCConfig:
    def __init__(self, **kw): self.__dict__.update(kw)

def emulate_qrc_embeddings(*a, **k):
    raise RuntimeError(
        "Falta una clave en qrc_totaltime_emb_IV.pkl. Este exportador no emula: "
        "el CSV debe salir exactamente de los embeddings que produjeron la Tabla 2.")

# ---------- ejecutar el script original sin tocarlo ----------
src = (HERE / "cell_4_4_v2.py").read_text(encoding="utf-8")
src = src.replace("import matplotlib.pyplot as plt", "import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt")
exec(compile(src, "cell_4_4_v2.py", "exec"), globals())

# ---------- verificación contra la Tabla 2 del manuscrito ----------
REF = {1.0: 0.698, 1.5: -0.197, 2.0: 0.658, 2.5: -0.314, 3.0: 0.131}
out = ROOT / f"qrc_totaltime_{CASE}_v2.csv"
print(f"\n=== Verificación contra la Tabla 2 del manuscrito ===\n(escrito en {out})")
ok = True
for _, r in df.iterrows():                                   # noqa: F821  (df lo define cell_4_4_v2)
    tt = float(r["total_time"]); got = float(r["skill_qrc"]); ref = REF.get(tt)
    good = ref is not None and abs(got - ref) < 0.002
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  t_tot={tt:>4} us   obtenido {got:+.3f}   manuscrito {ref:+.3f}")
print("\nOK: el CSV reproduce la Tabla 2." if ok else
      "\nFALLO: revisar antes de commitear. No edites el CSV a mano.")
sys.exit(0 if ok else 1)

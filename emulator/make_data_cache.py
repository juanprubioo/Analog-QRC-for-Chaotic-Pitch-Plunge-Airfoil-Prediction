"""make_data_cache.py — Regenera data_cache.npz (datos limpios + denoised por semilla).

Requiere en el mismo directorio: airfoil_simulator.py, airfoil_qrc_v5.py.
Salida: data_cache.npz con XC (trayectoria limpia Caso IV), dt, y dn_<seed>
(observaciones ruidosas al 40% suavizadas con Savitzky-Golay w=31, p=3),
exactamente como las consume run_generic.py / analyze_generic.py.
"""
import warnings, sys, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import savgol_filter

SEEDS = [1234, 1335, 1436, 1537, 1638, 1739]  # SEED + {0,101,202,303,404,505}
CASE = "IV"; NOISE_LEVEL = 0.40; NOISE_KIND = "gaussian"; STUDENT_DF = 5

spec = importlib.util.spec_from_file_location("q5", "airfoil_qrc_v5.py")
q5 = importlib.util.module_from_spec(spec); sys.modules["q5"] = q5
spec.loader.exec_module(q5)

out = {}
clean = q5.load_or_generate_dataset(input_npz=None, case=CASE, noise_level=0.0,
                                    noise_kind="none", student_df=STUDENT_DF, seed=SEEDS[0])
out["XC"] = np.asarray(clean["X_sampled"], float)
out["dt"] = np.array(float(clean["metadata"]["sampling_interval"]))

for seed in SEEDS:
    ds = q5.load_or_generate_dataset(input_npz=None, case=CASE, noise_level=NOISE_LEVEL,
                                     noise_kind=NOISE_KIND, student_df=STUDENT_DF, seed=seed)
    src = np.asarray(ds["Y_sampled"], float)
    X = np.empty_like(src)
    for d in range(src.shape[1]):
        X[:, d] = savgol_filter(src[:, d], 31, 3, mode="interp")
    out[f"dn_{seed}"] = X
    print(f"seed {seed} listo")

np.savez("data_cache.npz", **out)
print("Guardado data_cache.npz:", {k: getattr(v, "shape", v) for k, v in out.items()})

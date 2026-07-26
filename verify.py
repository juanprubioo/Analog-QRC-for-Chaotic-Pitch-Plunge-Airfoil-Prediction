"""verify.py — Smoke test del protocolo de evaluación.

Recomputa desde cero los tres números clásicos de referencia del protocolo 4.4
(Caso IV, 66 anchors, train 44 / test 22, 3 semillas, horizontes {1,2,3}*tau_c):

    NG-RC deg2 alpha=1e-4 (roto)  -> -8.912
    NG-RC deg2 RidgeCV            -> +0.405
    NG-RC deg1 RidgeCV (lineal)   -> +0.706

Si alguien modifica build_ngrc_model, el suavizado, o la construcción de
anchors/ventanas/targets, este test lo detecta. No usa el emulador cuántico:
corre en ~2-3 min (regenera los datasets) o en segundos si existe data_cache.npz.

Uso:  python3 verify.py          (tolerancia por defecto 0.02)
      python3 verify.py 0.005    (tolerancia estricta)
"""
import warnings, sys, os, importlib.util
warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import savgol_filter

TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02

REFS = {
    "NG-RC deg2 alpha=1e-4 (roto)": (-8.912, dict(degree=2, alpha=1e-4)),
    "NG-RC deg2 RidgeCV (justo)":   (+0.405, dict(degree=2)),
    "NG-RC deg1 RidgeCV (lineal)":  (+0.706, dict(degree=1)),
}

# --- imports del proyecto ---
def dyn_import(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

q5 = dyn_import("q5_verify", "airfoil_qrc_v5.py")

# --- datos: cache si existe, si no regenerar ---
CASE = "IV"; SEEDS = [1234, 1335, 1436]
NOISE_LEVEL = 0.40; NOISE_KIND = "gaussian"; STUDENT_DF = 5

if os.path.exists("data_cache.npz"):
    _d = np.load("data_cache.npz")
    XC = _d["XC"]; dt = float(_d["dt"])
    denoised = {s: _d[f"dn_{s}"] for s in SEEDS}
else:
    clean = q5.load_or_generate_dataset(input_npz=None, case=CASE, noise_level=0.0,
                                        noise_kind="none", student_df=STUDENT_DF, seed=SEEDS[0])
    XC = np.asarray(clean["X_sampled"], float)
    dt = float(clean["metadata"]["sampling_interval"])
    denoised = {}
    for s in SEEDS:
        ds = q5.load_or_generate_dataset(input_npz=None, case=CASE, noise_level=NOISE_LEVEL,
                                         noise_kind=NOISE_KIND, student_df=STUDENT_DF, seed=s)
        src = np.asarray(ds["Y_sampled"], float); X = np.empty_like(src)
        for d in range(src.shape[1]):
            X[:, d] = savgol_filter(src[:, d], 31, 3, mode="interp")
        denoised[s] = X

# --- protocolo 4.4: anchors, ventanas, targets ---
STATE_IDX = [0, 1, 2, 3]; WINDOW = 2
BURN_IN_TU = 500.0; WIN_STRIDE_TU = 5.0; EVAL_FRACS = [1.0, 2.0, 3.0]
N_TRAIN, N_TEST = 44, 22

i0 = int(round(BURN_IN_TU / dt))
x = XC[i0:i0 + int(round(600/dt)), 0]; x = x - x.mean()
ac = np.correlate(x, x, "full")[len(x)-1:len(x)-1+int(round(150/dt))]; ac = ac/ac[0]
char_tu = float(np.where(ac < 1/np.e)[0][0] * dt)
assert abs(char_tu - 10.3) < 0.5, f"tau_c inesperado: {char_tu:.2f} (esperado ~10.3)"

spacing = int(round(char_tu/dt)); wstride = int(round(WIN_STRIDE_TU/dt))
H_eval = sorted(set(int(round(f*char_tu/dt)) for f in EVAL_FRACS)); H_max = max(H_eval)
anchors = (i0 + (WINDOW-1)*wstride) + np.arange(N_TRAIN + N_TEST) * spacing
anchors = anchors[anchors + H_max < len(XC)]
tr = np.arange(0, len(anchors) - N_TEST)
te = np.arange(len(anchors) - N_TEST, len(anchors))
Xc_sel = XC[:, STATE_IDX]

def windows_of(seed):
    Xin = denoised[seed][:, STATE_IDX]
    return np.asarray([Xin[a - (WINDOW-1-np.arange(WINDOW))*wstride].reshape(-1) for a in anchors])

W_by_seed = {s: windows_of(s) for s in SEEDS}

def pooled_skill(**model_kwargs):
    e_pool, p_pool = [], []
    for s in SEEDS:
        W = W_by_seed[s]
        for H in H_eval:
            Y = Xc_sel[anchors + H] - Xc_sel[anchors]
            m = q5.build_ngrc_model(**model_kwargs).fit(W[tr], Y[tr])
            e_pool.append(((m.predict(W[te]) - Y[te])**2).mean(1))
            p_pool.append((Y[te]**2).mean(1))
    E = np.concatenate(e_pool); P = np.concatenate(p_pool)
    return 1 - E.mean()/P.mean()

# --- ejecutar y comparar ---
print(f"tau_c = {char_tu:.2f} tu | anchors = {len(anchors)} (train {len(tr)} / test {len(te)}) | tol = {TOL}")
failed = False
for name, (ref, kw) in REFS.items():
    got = pooled_skill(**kw)
    ok = abs(got - ref) < TOL
    failed |= not ok
    print(f"  {'PASS' if ok else 'FAIL':4s}  {name:32s} obtenido {got:+.3f}  esperado {ref:+.3f}")

if failed:
    print("\nVERIFY FAILED: el protocolo o build_ngrc_model cambiaron respecto a la referencia.")
    sys.exit(1)
print("\nVERIFY OK: protocolo de evaluación consistente con la referencia.")

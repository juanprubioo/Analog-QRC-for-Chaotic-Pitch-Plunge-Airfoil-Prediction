"""Analiza configuraciones cacheadas por run_generic.py.
Uso: python3 analyze_generic.py <config_json>
Replica el protocolo 4.4 (mismos anchors/targets/pooling) con readout QRC RidgeCV y baselines CV.
"""
import warnings, pickle, sys, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, RidgeCV
import importlib.util

def dyn_import(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

qrc5 = dyn_import("airfoil_qrc_v5_local", "airfoil_qrc_v5.py")
load_or_generate_dataset = qrc5.load_or_generate_dataset
build_ngrc_model = qrc5.build_ngrc_model

CASE = "IV"; SEED = 1234
NOISE_LEVEL = 0.40; NOISE_KIND = "gaussian"; STUDENT_DF = 5
TT_WIN_STRIDE_TU = 5.0; TT_BURN_IN_TU = 500.0
TT_EVAL_FRACS = [1.0, 2.0, 3.0]
BASE_N_TRAIN = 44; BASE_N_TEST = 22

cfg = json.loads(sys.argv[1])
state_indices = cfg["state_indices"]; window = cfg["window"]
atoms = window * len(state_indices)
ts = cfg["time_steps"]; readout = cfg["readout"]
nshots = cfg.get("nshots", 500)
spacing_um = cfg.get("lattice_spacing", 10.0)
n_extra = cfg.get("n_anchor_extra", 0)
readout_mode = cfg.get("readout_mode", "cv")  # "cv" | "fixed"
train_sizes = cfg.get("train_sizes", None)     # para curvas de aprendizaje
z_only = cfg.get("z_only_from_zz", False)      # extraer subset Z de un embedding ZZ

def _char_time_tu(x, dt, max_lag_tu=150.0):
    x = np.asarray(x, float) - np.mean(x); n = len(x); ml = min(n - 1, int(round(max_lag_tu / dt)))
    ac = np.correlate(x, x, "full")[n - 1: n - 1 + ml]
    if ac[0] <= 0: return None
    ac = ac / ac[0]; b = np.where(ac < 1.0 / np.e)[0]
    return float(b[0] * dt) if len(b) else None

_data = np.load("data_cache.npz")
XC = _data["XC"]
dt = float(_data["dt"])
i0 = int(round(TT_BURN_IN_TU / dt))
char_tu = _char_time_tu(XC[i0:i0 + min(int(round(600/dt)), len(XC)-i0), 0], dt) or 10.0
spacing_raw = max(1, int(round(char_tu / dt)))
win_stride_raw = max(1, int(round(TT_WIN_STRIDE_TU / dt)))
H_eval = sorted(set(max(1, int(round(fr * char_tu / dt))) for fr in TT_EVAL_FRACS))
H_max = max(H_eval)

n0 = BASE_N_TRAIN + BASE_N_TEST + n_extra
base = i0 + (window - 1) * win_stride_raw
anchors = base + np.arange(n0) * spacing_raw
keep = (anchors + H_max < len(XC)) & (anchors - (window - 1) * win_stride_raw >= 0)
anchors = anchors[keep]
n_test = BASE_N_TEST
tr_all = np.arange(0, len(anchors) - n_test)
te_idx = np.arange(len(anchors) - n_test, len(anchors))
Xc_sel = XC[:, state_indices]
seed_list = cfg["seeds"]

_dn = {}
def _denoised(seed):
    return _data[f"dn_{seed}"]

def _W(seed):
    Xin = _denoised(seed)[:, state_indices]
    return np.asarray([Xin[a - (window - 1 - np.arange(window)) * win_stride_raw].reshape(-1) for a in anchors])

cache = pickle.load(open("emb_cache.pkl", "rb"))
def key(tt, seed):
    return (CASE, tt, int(seed), atoms, ts, readout, nshots,
            tuple(state_indices), window, spacing_um, len(anchors))

def z_subset(E):
    """Extrae las columnas Z de un embedding ZZ: por probe-time, primeras `atoms` de (atoms + C(atoms,2))."""
    per = atoms + atoms*(atoms-1)//2
    cols = []
    for t in range(ts):
        cols.extend(range(t*per, t*per + atoms))
    return E[:, cols]

_rng = np.random.default_rng(0)
def _boot(em, ep, B=3000):
    n = len(em); out = np.empty(B)
    for b in range(B):
        idx = _rng.integers(0, n, n); pp = ep[idx].mean()
        out[b] = 1 - em[idx].mean()/pp if pp > 1e-30 else np.nan
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)

def make_readout():
    if readout_mode == "fixed":
        return Ridge(alpha=1e-4, fit_intercept=True)
    return RidgeCV(alphas=np.logspace(-6, 3, 19), fit_intercept=True)

def eval_tt(tt, n_train):
    tr_idx = tr_all[-n_train:]  # los mas cercanos al test
    eq, en2, en1, ep = [], [], [], []
    psq, psbest = [], []
    for seed in seed_list:
        E = cache[key(tt, seed)]
        if z_only: E = z_subset(E)
        W = _W(seed)
        sq, s2, s1, sp_ = [], [], [], []
        for H in H_eval:
            Y = (Xc_sel[anchors + H] - Xc_sel[anchors])
            Yq = make_readout().fit(E[tr_idx], Y[tr_idx]).predict(E[te_idx])
            Yn2 = build_ngrc_model(degree=2).fit(W[tr_idx], Y[tr_idx]).predict(W[te_idx])
            Yn1 = build_ngrc_model(degree=1).fit(W[tr_idx], Y[tr_idx]).predict(W[te_idx])
            sq.append(((Yq - Y[te_idx])**2).mean(1))
            s2.append(((Yn2 - Y[te_idx])**2).mean(1))
            s1.append(((Yn1 - Y[te_idx])**2).mean(1))
            sp_.append((Y[te_idx]**2).mean(1))
        sq, s2, s1, sp_ = map(np.concatenate, (sq, s2, s1, sp_))
        eq.append(sq); en2.append(s2); en1.append(s1); ep.append(sp_)
        psq.append(1 - sq.mean()/sp_.mean())
        psbest.append(max(1 - s2.mean()/sp_.mean(), 1 - s1.mean()/sp_.mean()))
    EQ, EN2, EN1, EP = map(np.concatenate, (eq, en2, en1, ep))
    mp = EP.mean()
    lo, hi = _boot(EQ, EP)
    dim = (cache[key(tt, seed_list[0])].shape[1] if not z_only
           else z_subset(cache[key(tt, seed_list[0])]).shape[1])
    return {"total_time": tt, "n_train": n_train, "emb_dim": dim,
            "skill_qrc": 1-EQ.mean()/mp, "ci_lo": lo, "ci_hi": hi,
            "qrc_seed_min": min(psq), "cls_seed_max": max(psbest),
            "skill_ngrc": 1-EN2.mean()/mp, "skill_ngrc_lin": 1-EN1.mean()/mp,
            "ok_dim": n_train > dim, "robust": min(psq) > max(psbest)}

if __name__ == "__main__":
    sizes = train_sizes or [len(tr_all)]
    rows = [eval_tt(tt, n) for tt in cfg["tts"] for n in sizes]
    df = pd.DataFrame(rows)
    df["skill_ngrc_best"] = df[["skill_ngrc", "skill_ngrc_lin"]].max(axis=1)
    df["beats_classical"] = df["ci_lo"] > df["skill_ngrc_best"]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    tag = f"atoms={atoms} ts={ts} ro={readout}{' (Z-subset)' if z_only else ''} sp={spacing_um} readout_mode={readout_mode}"
    print(f"=== {tag} | anchors={len(anchors)} test={n_test} ===")
    print(df.to_string(index=False))

# =============================================================================
# CELL F (4.4) v2: TOTAL_TIME drill-down — Camino B, protocolo congelado
# Cambios vs v1 (justificados en la sesión de verificación):
#   1. Readout QRC: RidgeCV (simetría con los clásicos; el alpha=1e-4 fijo estaba
#      sub-regularizado frente al ruido de 500 shots y distorsionaba los valles).
#   2. Menú clásico ampliado: deg1 CV, deg2 CV, deg3 CV y control RFF de dimensión
#      igualada al embedding (features tanh aleatorias + RidgeCV, promedio de 5
#      proyecciones). beats_classical se evalúa contra el MÁXIMO de los cuatro.
#   3. Seis semillas (criterio de robustez per-seed con más poder).
# Nota de encuadre: el barrido de total_time es un corte 1D de la estructura
# temporal de los probes (ver mapa de pares); se reporta como caracterización.
# =============================================================================
import json, pickle, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import RidgeCV

# ---- fixed reservoir config (the 8-atom Z baseline) ----
TT_STATE_INDICES = [0, 1, 2, 3]
TT_WINDOW        = 2
TT_TIME_STEPS    = 2
TT_READOUT       = "Z"
TT_NSHOTS        = 500
# ---- the axis we drill ----
TT_VALUES        = [1.0, 1.5, 2.0, 2.5, 3.0]   # total_time (us); all <= 4 us (Aquila-realizable)
# ---- evaluation (v2: 6 seeds) ----
TT_SEEDS         = [SEED, SEED + 101, SEED + 202, SEED + 303, SEED + 404, SEED + 505]
TT_N_TRAIN       = 44
TT_N_TEST        = 22
TT_WIN_STRIDE_TU = 5.0
TT_BURN_IN_TU    = 500.0
TT_EVAL_FRACS    = [1.0, 2.0, 3.0]      # multiples of decorrelation time
TT_BOOTSTRAP     = 3000
TT_MAX_EMU_RUNS  = 4000                 # subido: 6 semillas x 5 tt puede requerir ~3960 en frío
TT_ALPHAS        = np.logspace(-6, 3, 19)
TT_RFF_REPS      = 5                    # proyecciones RFF promediadas

_chaotic = (DYNAMICS == "chaotic") or (DYNAMICS == "auto" and str(CASE).upper() == "IV")

def _char_time_tu(x, dt, max_lag_tu=150.0):
    x = np.asarray(x, float) - np.mean(x); n = len(x); ml = min(n - 1, int(round(max_lag_tu / dt)))
    ac = np.correlate(x, x, "full")[n - 1: n - 1 + ml]
    if ac[0] <= 0: return None
    ac = ac / ac[0]; b = np.where(ac < 1.0 / np.e)[0]
    return float(b[0] * dt) if len(b) else None

_clean = load_or_generate_dataset(input_npz=None, case=CASE, noise_level=0.0, noise_kind="none",
                                  student_df=STUDENT_DF, seed=TT_SEEDS[0])
XC = np.asarray(_clean["X_sampled"], float)
dt = float(_clean.get("metadata", {}).get("sampling_interval",
                                          np.median(np.diff(np.asarray(_clean["tau_sampled"], float)))))
i0 = int(round(TT_BURN_IN_TU / dt))
char_tu = (_char_time_tu(XC[i0:i0 + min(int(round(600/dt)), len(XC)-i0), 0], dt) or 10.0) if _chaotic else 75.0
spacing_raw = max(1, int(round(char_tu / dt)))
win_stride_raw = max(1, int(round(TT_WIN_STRIDE_TU / dt)))
H_eval = sorted(set(max(1, int(round(fr * char_tu / dt))) for fr in TT_EVAL_FRACS))
H_max = max(H_eval)

n0 = TT_N_TRAIN + TT_N_TEST
base = i0 + (TT_WINDOW - 1) * win_stride_raw
anchors = base + np.arange(n0) * spacing_raw
keep = (anchors + H_max < len(XC)) & (anchors - (TT_WINDOW - 1) * win_stride_raw >= 0)
anchors = anchors[keep]; n_anchor = len(anchors)
n_test = min(TT_N_TEST, max(3, n_anchor // 3))
tr_idx = np.arange(0, n_anchor - n_test); te_idx = np.arange(n_anchor - n_test, n_anchor)

Xc_sel = XC[:, TT_STATE_INDICES]
seed_list = TT_SEEDS if USE_NOISY_INPUT else ["clean"]

cache_path = ROOT / f"qrc_totaltime_emb_{CASE}.pkl"
cache = pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}
def _ek(tt, seed):
    return hashlib.md5(json.dumps([CASE, "TT", tt, TT_STATE_INDICES, TT_WINDOW, TT_TIME_STEPS, TT_READOUT,
        TT_NSHOTS, int(seed), round(char_tu,3), int(spacing_raw), TT_WIN_STRIDE_TU, int(n_anchor)],
        sort_keys=True).encode()).hexdigest()

planned = sum(n_anchor * TT_TIME_STEPS for tt in TT_VALUES for seed in seed_list if _ek(tt, seed) not in cache)
print(f"chaotic={_chaotic}, tau_c~{char_tu:.1f} tu | total_time values {TT_VALUES} | seeds {len(seed_list)}")
print(f"anchors {n_anchor} (train {len(tr_idx)}/test {len(te_idx)}) | eval horizons {[round(h*dt,1) for h in H_eval]} tu")
print(f"pooled test points per total_time: {len(seed_list)*len(te_idx)*len(H_eval)}")
print(f"Planned emulator runs: ~{planned}  (cap {TT_MAX_EMU_RUNS})")
if planned > TT_MAX_EMU_RUNS:
    raise RuntimeError(f"Refusing ~{planned} runs (> {TT_MAX_EMU_RUNS}); fewer TT_VALUES/TT_SEEDS or lower TT_N_TRAIN/TEST.")

_dn = {}
def _denoised(seed):
    if seed in _dn: return _dn[seed]
    if USE_NOISY_INPUT:
        ds = load_or_generate_dataset(input_npz=None, case=CASE, noise_level=NOISE_LEVEL,
                                      noise_kind=NOISE_KIND, student_df=STUDENT_DF, seed=int(seed))
        src = np.asarray(ds["Y_sampled"], float); X = np.empty_like(src)
        _w = 31 if len(src) >= 31 else (len(src)//2*2 - 1)
        for d_ in range(src.shape[1]): X[:, d_] = savgol_filter(src[:, d_], max(5, _w), 3, mode="interp")
    else:
        X = XC
    _dn[seed] = X; return X

def _W(seed):
    Xin = _denoised(seed)[:, TT_STATE_INDICES]
    return np.asarray([Xin[a - (TT_WINDOW - 1 - np.arange(TT_WINDOW)) * win_stride_raw].reshape(-1) for a in anchors])

_rng = np.random.default_rng(0)
def _boot(em, ep, B):
    n = len(em); out = np.empty(B)
    for b in range(B):
        idx = _rng.integers(0, n, n); pp = ep[idx].mean()
        out[b] = 1 - em[idx].mean()/pp if pp > 1e-30 else np.nan
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)

# ---- v2: control RFF de dimensión igualada (features tanh aleatorias + RidgeCV) ----
class _RFF:
    def __init__(self, dim, in_dim, rng, scale=1.0):
        self.G = rng.normal(0, scale/np.sqrt(in_dim), (in_dim, dim))
        self.b = rng.uniform(-np.pi, np.pi, dim)
    def transform(self, X):
        return np.tanh(X @ self.G + self.b)

def _ridgecv():
    return RidgeCV(alphas=TT_ALPHAS, fit_intercept=True)

W_by_seed = {s: _W(s) for s in seed_list}
EMB_DIM = TT_WINDOW * len(TT_STATE_INDICES) * TT_TIME_STEPS if TT_READOUT == "Z" else None
rows = []
for tt in TT_VALUES:
    qrc = QRCConfig(atom_number=TT_WINDOW*len(TT_STATE_INDICES), encoding="local",
                    lattice_spacing=QRC_LATTICE_SPACING, encoding_scale=QRC_ENCODING_SCALE,
                    rabi_frequency=QRC_RABI_FREQUENCY, total_time=tt, time_steps=TT_TIME_STEPS, readouts=TT_READOUT)
    eq_pool, ep_pool = [], []
    ecls_pool = {m: [] for m in ("d1", "d2", "d3", "rff")}
    psq, pscls_best = [], []
    for seed in seed_list:
        W = W_by_seed[seed]; ek = _ek(tt, seed)
        if ek in cache:
            E = cache[ek]
        else:
            Wsc = np.clip(MinMaxScaler((0.0, 1.0)).fit(W[tr_idx]).transform(W), 0.0, 1.0)
            E = emulate_qrc_embeddings(qrc, Wsc, nshots=TT_NSHOTS)
            cache[ek] = E; pickle.dump(cache, open(cache_path, "wb"))
        rff_dim = E.shape[1]  # dimensión igualada al embedding QRC real
        seq, sep = [], []
        secls = {m: [] for m in ecls_pool}
        for H in H_eval:
            Y = (Xc_sel[anchors + H] - Xc_sel[anchors]) if PREDICT_DELTA else Xc_sel[anchors + H]
            # v2: readout QRC con RidgeCV (simetría)
            Yq  = _ridgecv().fit(E[tr_idx], Y[tr_idx]).predict(E[te_idx])
            seq.append(((Yq-Y[te_idx])**2).mean(1))
            # menú clásico: NG-RC deg1/deg2/deg3 (todos CV) sobre la MISMA ventana
            for m, deg in (("d1",1), ("d2",2), ("d3",3)):
                Yc = build_ngrc_model(degree=deg).fit(W[tr_idx], Y[tr_idx]).predict(W[te_idx])
                secls[m].append(((Yc-Y[te_idx])**2).mean(1))
            # control RFF de dimensión igualada, promedio de TT_RFF_REPS proyecciones
            sc = StandardScaler().fit(W[tr_idx])
            Wtr, Wte = sc.transform(W[tr_idx]), sc.transform(W[te_idx])
            mses = []
            for r in range(TT_RFF_REPS):
                rff = _RFF(rff_dim, W.shape[1], np.random.default_rng(100 + r))
                Yr = _ridgecv().fit(rff.transform(Wtr), Y[tr_idx]).predict(rff.transform(Wte))
                mses.append(((Yr - Y[te_idx])**2).mean(1))
            secls["rff"].append(np.mean(mses, axis=0))
            sep.append((Y[te_idx]**2).mean(1))
        seq = np.concatenate(seq); sep = np.concatenate(sep)
        eq_pool.append(seq); ep_pool.append(sep)
        for m in ecls_pool: ecls_pool[m].append(np.concatenate(secls[m]))
        psq.append(1 - seq.mean()/sep.mean())
        pscls_best.append(max(1 - np.concatenate(secls[m]).mean()/sep.mean() for m in secls))
    EQ = np.concatenate(eq_pool); EP = np.concatenate(ep_pool); mp = EP.mean()
    skq = 1 - EQ.mean()/mp
    sk_cls = {m: 1 - np.concatenate(ecls_pool[m]).mean()/mp for m in ecls_pool}
    skn_best = max(sk_cls.values())
    lo, hi = _boot(EQ, EP, TT_BOOTSTRAP)
    rows.append({"total_time": tt, "skill_qrc": skq, "ci_lo": lo, "ci_hi": hi,
                 "qrc_seed_min": min(psq), "qrc_seed_max": max(psq),
                 "skill_ngrc_lin": sk_cls["d1"], "skill_ngrc": sk_cls["d2"],
                 "skill_ngrc_d3": sk_cls["d3"], "skill_rff": sk_cls["rff"],
                 "skill_cls_best": skn_best,
                 "beats_persist_sig": lo > 0,
                 "beats_classical": lo > skn_best,
                 "beats_classical_robust": min(psq) > max(pscls_best)})

df = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print("\n=== total_time drill-down v2 (8 atoms, Z; readout CV; pooled 95% CI; 6 seeds) ===")
print(df.to_string(index=False))
df.to_csv(ROOT / f"qrc_totaltime_{CASE}_v2.csv", index=False)

# --------------------------- VERDICT (Camino B) ------------------------------
print("\n=== VERDICT (Camino B: characterization, NOT advantage) ===")
qrc_best = df.sort_values("skill_qrc", ascending=False).iloc[0]
cls_best = df["skill_cls_best"].max()
print(f"Classical menu (total_time-independent): lin={df['skill_ngrc_lin'].iloc[0]:.3f}, "
      f"deg2={df['skill_ngrc'].iloc[0]:.3f}, deg3={df['skill_ngrc_d3'].iloc[0]:.3f}, "
      f"RFF(dim-matched)={df['skill_rff'].iloc[0]:.3f} -> best={cls_best:.3f}")
print(f"QRC best: {qrc_best['skill_qrc']:.3f} at total_time={qrc_best['total_time']} us "
      f"(CI [{qrc_best['ci_lo']:.3f}, {qrc_best['ci_hi']:.3f}]).")
print("Note: the total_time sweep is a 1D slice of the probe-time structure "
      "(information decay + narrow anti-resonances + multi-probe complementarity); "
      "report as characterization.")
if df["beats_classical"].any():
    w = df[df["beats_classical"]]
    print(f"CAUTION: QRC CI-lower-bound exceeds the best classical at {list(w['total_time'])} us.")
    print("  Before any claim: check beats_classical_robust, add seeds, and account for "
          "multiplicity across every configuration explored in the project.")
else:
    print("At 8 atoms the QRC does NOT exceed the best fair classical baseline. "
          "This is the expected outcome at this scale (cf. Kornjaca et al.) and is the "
          "declared no-advantage result of the paper.")

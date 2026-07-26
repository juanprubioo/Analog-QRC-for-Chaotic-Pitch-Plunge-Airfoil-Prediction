"""lcurve_fixedtest.py — Learning curve with FIXED test block and anti-overlap gap.

Design (recommendation 4):
  - Test: 22 anchors at fixed absolute position, at the end of usable trajectory.
  - Gap = H_max between last train anchor and first test anchor
    (train targets no longer penetrate the test epoch).
  - Train: grows backward in time; evaluated with n most recent.
  - Embeddings with ZZ readout (dim 72) -> subset Z (dim 16) extracted for free.
  - Classic menu: linear CV, deg2 CV, deg3 CV, RFF of matched dimension (to Z and ZZ).

Usage:
  python3 lcurve_fixedtest.py run <budget_s>   # computes embeddings (incremental)
  python3 lcurve_fixedtest.py analyze          # table + figure lcurve_fixedtest.png
"""
import warnings, pickle, sys, os, time
warnings.filterwarnings("ignore")
import numpy as np
import importlib.util

def dyn_import(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

q5 = dyn_import("airfoil_qrc_v5_lc", "airfoil_qrc_v5.py")

# ---------------- configuration ----------------
CASE = "IV"
SEEDS = [1234, 1335, 1436]          # expandable to 6 for more power
STATE_IDX = [0, 1, 2, 3]; WINDOW = 2; ATOMS = 8
TS = 2; READOUT = "ZZ"; NSHOTS = 500; SPACING_UM = 10.0
TT = 2.0                             # peak of base sweep
N_TEST = 22
TRAIN_SIZES = [44, 80, 120, 178, 250]
BURN_IN_TU = 500.0; WIN_STRIDE_TU = 5.0; EVAL_FRACS = [1.0, 2.0, 3.0]
RABI = 6.283; ENC = 9.0
CACHE = "emb_cache.pkl"
SCHEME = "IV-LCFIX"                  # marker for anchor geometry (fixed test + gap)

_d = np.load("data_cache.npz")
XC = _d["XC"]; dt = float(_d["dt"])

i0 = int(round(BURN_IN_TU / dt))
x = XC[i0:i0 + int(round(600/dt)), 0]; x = x - x.mean()
ac = np.correlate(x, x, "full")[len(x)-1:len(x)-1+int(round(150/dt))]; ac = ac/ac[0]
char_tu = float(np.where(ac < 1/np.e)[0][0] * dt)
spacing = int(round(char_tu/dt)); wstride = int(round(WIN_STRIDE_TU/dt))
H_eval = sorted(set(int(round(f*char_tu/dt)) for f in EVAL_FRACS)); H_max = max(H_eval)

# ---- geometry: fixed test at end, gap, train backward ----
last_valid = len(XC) - 1 - H_max
test_anchors = last_valid - np.arange(N_TEST)[::-1] * spacing          # fixed, absolute
gap = H_max                                                             # anti-overlap
n_train_max = max(TRAIN_SIZES)
first_train_end = test_anchors[0] - gap
train_anchors_all = first_train_end - np.arange(n_train_max)[::-1] * spacing
assert train_anchors_all[0] - (WINDOW-1)*wstride >= 0, "train exceeds start"
assert train_anchors_all[0] >= i0, f"train starts before burn-in ({train_anchors_all[0]*dt:.0f} tu)"
anchors = np.concatenate([train_anchors_all, test_anchors])             # temporal order
tr_all = np.arange(0, n_train_max)
te_idx = np.arange(n_train_max, n_train_max + N_TEST)
Xc_sel = XC[:, STATE_IDX]

def W_of(seed):
    Xin = _d[f"dn_{seed}"][:, STATE_IDX]
    return np.asarray([Xin[a - (WINDOW-1-np.arange(WINDOW))*wstride].reshape(-1) for a in anchors])

def key(seed):
    return (SCHEME, TT, int(seed), ATOMS, TS, READOUT, NSHOTS,
            tuple(STATE_IDX), WINDOW, SPACING_UM, len(anchors))

# ---------------- run ----------------
def do_run(budget):
    from sklearn.preprocessing import MinMaxScaler
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    t0 = time.time()
    for seed in SEEDS:
        k = key(seed)
        if k in cache: continue
        qrc = q5.QRCConfig(atom_number=ATOMS, encoding="local", lattice_spacing=SPACING_UM,
                           encoding_scale=ENC, rabi_frequency=RABI, total_time=TT,
                           time_steps=TS, readouts=READOUT)
        W = W_of(seed)
        # scaler fitted on maximum train (all curves use same physical scaling)
        Wsc = np.clip(MinMaxScaler((0.0, 1.0)).fit(W[tr_all]).transform(W), 0.0, 1.0)
        pk = ("PARTIAL",) + k
        parts = cache.get(pk, []); done = sum(len(p) for p in parts); BLOCK = 15
        while done < len(Wsc):
            if time.time() - t0 > budget:
                cache[pk] = parts; pickle.dump(cache, open(CACHE, "wb"))
                print(f"PARTIAL {done}/{len(Wsc)} seed={seed}", flush=True); return
            E_blk = q5.emulate_qrc_embeddings(qrc, Wsc[done:done+BLOCK], nshots=NSHOTS)
            parts.append(E_blk); done += len(E_blk)
            cache[pk] = parts; pickle.dump(cache, open(CACHE, "wb"))
        cache[key(seed)] = np.concatenate(parts, axis=0); cache.pop(pk, None)
        pickle.dump(cache, open(CACHE, "wb"))
        print(f"[{time.time()-t0:6.1f}s] seed={seed} E={cache[key(seed)].shape}", flush=True)
    print("DONE", flush=True)

# ---------------- analyze ----------------
def z_subset(E):
    per = ATOMS + ATOMS*(ATOMS-1)//2
    cols = []
    for t in range(TS): cols.extend(range(t*per, t*per + ATOMS))
    return E[:, cols]

def do_analyze():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    cache = pickle.load(open(CACHE, "rb"))
    def ridgecv(): return RidgeCV(alphas=np.logspace(-6, 3, 19), fit_intercept=True)
    class RFF:
        def __init__(self, dim, in_dim, rng, scale=1.0):
            self.G = rng.normal(0, scale/np.sqrt(in_dim), (in_dim, dim))
            self.b = rng.uniform(-np.pi, np.pi, dim)
        def transform(self, X): return np.tanh(X @ self.G + self.b)

    rows = []
    for n_train in TRAIN_SIZES:
        tr = tr_all[-n_train:]                      # n most recent (before gap)
        pools = {m: [] for m in ("qrc_z","qrc_zz","d1","d2","d3","rff_z","rff_zz","persist")}
        for seed in SEEDS:
            E = cache[key(seed)]; Ez = z_subset(E); W = W_of(seed)
            for H in H_eval:
                Y = Xc_sel[anchors + H] - Xc_sel[anchors]
                pools["qrc_zz"].append(((ridgecv().fit(E[tr],Y[tr]).predict(E[te_idx])-Y[te_idx])**2).mean(1))
                pools["qrc_z"].append(((ridgecv().fit(Ez[tr],Y[tr]).predict(Ez[te_idx])-Y[te_idx])**2).mean(1))
                for m, deg in (("d1",1),("d2",2),("d3",3)):
                    Yc = q5.build_ngrc_model(degree=deg).fit(W[tr],Y[tr]).predict(W[te_idx])
                    pools[m].append(((Yc-Y[te_idx])**2).mean(1))
                sc = StandardScaler().fit(W[tr]); Wtr, Wte = sc.transform(W[tr]), sc.transform(W[te_idx])
                for m, dim in (("rff_z", Ez.shape[1]), ("rff_zz", E.shape[1])):
                    ms = []
                    for r in range(5):
                        rff = RFF(dim, W.shape[1], np.random.default_rng(100+r))
                        ms.append(((ridgecv().fit(rff.transform(Wtr),Y[tr]).predict(rff.transform(Wte))-Y[te_idx])**2).mean(1))
                    pools[m].append(np.mean(ms, axis=0))
                pools["persist"].append((Y[te_idx]**2).mean(1))
        P = {m: np.concatenate(v) for m, v in pools.items()}
        mp = P["persist"].mean()
        row = {"n_train": n_train}
        for m in ("qrc_z","qrc_zz","d1","d2","d3","rff_z","rff_zz"):
            row[m] = 1 - P[m].mean()/mp
        row["ok_dim_zz"] = n_train > 72
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(f"=== Learning curve, FIXED TEST (22 final anchors) + gap {H_max*dt:.0f} tu | tt={TT} ===")
    print(df.to_string(index=False))
    df.to_csv("lcurve_fixedtest.csv", index=False)

    # ---- Bootstrap CIs for QRC curves (valid n only) ----
    rng = np.random.default_rng(0)
    def boot_ci(em, ep, B=3000):
        n = len(em); out = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, n); pp = ep[idx].mean()
            out[b] = 1 - em[idx].mean()/pp if pp > 1e-30 else np.nan
        return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)

    valid_sizes = [n for n in TRAIN_SIZES if n > 72]   # ok_dim for ZZ/RFF-72
    ci = {m: {} for m in ("qrc_z", "qrc_zz")}
    for n_train in valid_sizes:
        tr = tr_all[-n_train:]
        for m in ci:
            em, ep = [], []
            for seed in SEEDS:
                E = cache[key(seed)]
                Em = z_subset(E) if m == "qrc_z" else E
                for H in H_eval:
                    Y = Xc_sel[anchors + H] - Xc_sel[anchors]
                    Yq = ridgecv().fit(Em[tr], Y[tr]).predict(Em[te_idx])
                    em.append(((Yq - Y[te_idx])**2).mean(1)); ep.append((Y[te_idx]**2).mean(1))
            ci[m][n_train] = boot_ci(np.concatenate(em), np.concatenate(ep))

    # ---- figure: ONLY valid region (n >= 80); n=44 in table with its ok_dim flag ----
    dfv = df[df["n_train"].isin(valid_sizes)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {"qrc_z":("o-","#2a78d6","QRC Z (dim 16)"), "qrc_zz":("o--","#173f6e","QRC ZZ (dim 72)"),
              "d1":("s-","#1baf7a","NG-RC linear CV"), "d2":("s--","#0e6e4b","NG-RC deg2 CV"),
              "d3":("s:","#7a7a2a","NG-RC deg3 CV"),
              "rff_z":("^-","#d64a2a","RFF dim 16"), "rff_zz":("^--","#7a2a17","RFF dim 72")}
    for m,(st,c,lb) in styles.items():
        ax.plot(dfv["n_train"], dfv[m], st, color=c, label=lb, ms=5)
    for m, c in (("qrc_z", "#2a78d6"), ("qrc_zz", "#173f6e")):
        lo = [ci[m][n][0] for n in valid_sizes]; hi = [ci[m][n][1] for n in valid_sizes]
        ax.fill_between(valid_sizes, lo, hi, color=c, alpha=0.12)
    ax.set_xlabel("training samples"); ax.set_ylabel("skill vs persistence (fixed test)")
    ax.set_title(f"Learning curve with fixed test and gap ({H_max*dt:.0f} tu)\n"
                 f"chaotic case, tt={TT} µs, {len(SEEDS)} seeds; only n with ok\\_dim (bands: 95% CI QRC)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("lcurve_fixedtest.png", dpi=300)
    print("figure: lcurve_fixedtest.png (valid region only; n=44 in table with ok_dim)")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if mode == "run":
        do_run(float(sys.argv[2]) if len(sys.argv) > 2 else 1e9)
    else:
        do_analyze()

"""fig_scaling.py — Regenerates scaling comparison across atoms (8/12/16).

CORRECTION relative to session summary: the informally reported triplet
(0.665/0.700/0.798) mixed tt=2.0, tt=1.0 and tt=2.0. The canonical comparison
of this script is at FIXED tt = 2.0 µs (only value computed for 16 atoms),
also reporting tt=1.0 for 8 and 12 atoms as secondary reference.

Configurations (classic always receives SAME window as feeds QRC):
  8 atoms   = 4 states (α, α̇, ξ, ξ̇) × window 2      -> window dim 8
  12 atoms  = 6 states (+w1, w2)     × window 2      -> window dim 12
  16 atoms  = 4 states               × window 4      -> window dim 16
QRC readout: RidgeCV. Classics: NG-RC linear and deg2 (RidgeCV). 3 seeds.

Requires: airfoil_qrc_v5.py, airfoil_simulator.py, data_cache.npz, emb_cache.pkl.
Output: fig_scaling.png and tab_scaling.csv. Deterministic given cache.
"""
import warnings, pickle, sys, csv
warnings.filterwarnings("ignore")
import numpy as np
import importlib.util
from sklearn.linear_model import RidgeCV
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dyn_import(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

q5 = dyn_import("airfoil_qrc_v5_sc", "airfoil_qrc_v5.py")
build_ngrc_model = q5.build_ngrc_model

SEEDS = [1234, 1335, 1436]
CONFIGS = [
    # (atoms, state_indices, window, available tts in cache)
    (8,  (0, 1, 2, 3),       2, [1.0, 2.0]),
    (12, (0, 1, 2, 3, 4, 5), 2, [1.0, 2.0]),
    (16, (0, 1, 2, 3),       4, [2.0]),
]

_d = np.load("data_cache.npz"); XC = _d["XC"]; dt = float(_d["dt"])
i0 = int(round(500.0/dt)); ws = int(round(5.0/dt))
x = XC[i0:i0+int(round(600/dt)), 0]; x = x - x.mean()
ac = np.correlate(x, x, "full")[len(x)-1:len(x)-1+int(round(150/dt))]; ac = ac/ac[0]
char_tu = float(np.where(ac < 1/np.e)[0][0]*dt)
sp = int(round(char_tu/dt))
H_eval = sorted(set(int(round(f*char_tu/dt)) for f in [1, 2, 3])); H_max = max(H_eval)

cache = pickle.load(open("emb_cache.pkl", "rb"))

def protocol(window):
    base = i0 + (window - 1) * ws
    anchors = base + np.arange(66) * sp
    anchors = anchors[(anchors + H_max < len(XC)) & (anchors - (window-1)*ws >= 0)]
    tr = np.arange(0, len(anchors)-22); te = np.arange(len(anchors)-22, len(anchors))
    return anchors, tr, te

def eval_config(atoms, sidx, window, tt):
    anchors, tr, te = protocol(window)
    Xs = XC[:, list(sidx)]
    k = lambda s: ("IV", tt, int(s), atoms, 2, "Z", 500, tuple(sidx), window, 10.0, len(anchors))
    eq, e1, e2, ep = [], [], [], []
    for s in SEEDS:
        E = cache[k(s)]
        Xin = _d[f"dn_{s}"][:, list(sidx)]
        W = np.asarray([Xin[a - (window-1-np.arange(window))*ws].reshape(-1) for a in anchors])
        for H in H_eval:
            Y = Xs[anchors+H] - Xs[anchors]
            m = RidgeCV(alphas=np.logspace(-6, 3, 19), fit_intercept=True).fit(E[tr], Y[tr])
            eq.append(((m.predict(E[te]) - Y[te])**2).mean(1))
            for pool, deg in ((e1, 1), (e2, 2)):
                Yc = build_ngrc_model(degree=deg).fit(W[tr], Y[tr]).predict(W[te])
                pool.append(((Yc - Y[te])**2).mean(1))
            ep.append((Y[te]**2).mean(1))
    mp = np.concatenate(ep).mean()
    return (1 - np.concatenate(eq).mean()/mp,
            1 - np.concatenate(e1).mean()/mp,
            1 - np.concatenate(e2).mean()/mp)

rows = []
for atoms, sidx, window, tts in CONFIGS:
    for tt in tts:
        sq, s1, s2 = eval_config(atoms, sidx, window, tt)
        rows.append({"atoms": atoms, "window_dim": window*len(sidx), "tt": tt,
                     "skill_qrc": sq, "skill_lin": s1, "skill_deg2": s2,
                     "gap": sq - max(s1, s2)})

with open("tab_scaling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

print(" atoms  dim  tt   | QRC(CV) |  lin CV | deg2 CV |   gap")
for r in rows:
    print(f"  {r['atoms']:3d}  {r['window_dim']:3d}  {r['tt']:.1f}  |  {r['skill_qrc']:+.3f} | "
          f"{r['skill_lin']:+.3f} |  {r['skill_deg2']:+.3f} | {r['gap']:+.3f}")

# ---- figure: canonical comparison at tt = 2.0 ----
r20 = [r for r in rows if r["tt"] == 2.0]
labels = [f"{r['atoms']} at.\n(win. {r['window_dim']}d)" for r in r20]
fig, ax = plt.subplots(figsize=(7, 4.6))
xpos = np.arange(len(r20))
ax.plot(xpos - 0.06, [r["skill_qrc"] for r in r20], "o", color="#2a78d6", ms=9,
        label="QRC (readout CV), tt = 2.0 µs")
ax.plot(xpos + 0.06, [max(r["skill_lin"], r["skill_deg2"]) for r in r20], "s", color="#1baf7a",
        ms=9, label="best NG-RC CV (same window)")
r10 = {r["atoms"]: r for r in rows if r["tt"] == 1.0}
ax.plot([x - 0.06 for x, r in zip(xpos, r20) if r["atoms"] in r10],
        [r10[r["atoms"]]["skill_qrc"] for r in r20 if r["atoms"] in r10],
        "o", color="#2a78d6", ms=6, alpha=0.35, label="QRC, tt = 1.0 µs (ref.)")
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel("skill vs persistence")
ax.set_title("Atom scaling at fixed tt = 2.0 µs:\ngap vs classic does not close")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig_scaling.png", dpi=300)
print("\nfigure: fig_scaling.png  |  data: tab_scaling.csv")
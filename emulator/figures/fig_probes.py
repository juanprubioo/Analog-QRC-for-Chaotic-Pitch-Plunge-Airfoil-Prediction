"""fig_probes.py — Regenera la figura de estructura temporal de probes (Fig. probes del paper).

Panel (a): skill de probe único vs t (0.3–2.9 µs, paso 0.1), banda del nodo sombreada.
Panel (b): zoom del nodo (0.5–1.0 µs, incluye 0.65/0.75/0.85) con el diagnóstico
           R²(W←E) de reconstrucción lineal de la entrada desde el embedding.
Panel (c): mapa de pares (t1, t2) sintetizado concatenando embeddings de probe único
           (válido porque cada probe es una evolución independiente desde t=0).

Requiere en el directorio: airfoil_qrc_v5.py, airfoil_simulator.py,
data_cache.npz y emb_cache.pkl (con los chunks ts=1 de probes 0.3–2.9 y 0.65/0.75/0.85).
Salida: estructura_probes_final.png y fig_probes_data.csv (los números de los paneles a y b).
Determinista dado el caché: reproduce la figura del manuscrito exactamente.
"""
import warnings, pickle, sys
warnings.filterwarnings("ignore")
import numpy as np
import importlib.util
from sklearn.linear_model import RidgeCV, LinearRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dyn_import(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

q5 = dyn_import("airfoil_qrc_v5_fig", "airfoil_qrc_v5.py")

# ---- protocolo (idéntico a verify.py / celda 4.4) ----
SEEDS = [1234, 1335, 1436]
_d = np.load("data_cache.npz"); XC = _d["XC"]; dt = float(_d["dt"])
i0 = int(round(500.0/dt)); ws = int(round(5.0/dt))
x = XC[i0:i0+int(round(600/dt)), 0]; x = x - x.mean()
ac = np.correlate(x, x, "full")[len(x)-1:len(x)-1+int(round(150/dt))]; ac = ac/ac[0]
char_tu = float(np.where(ac < 1/np.e)[0][0]*dt)
sp = int(round(char_tu/dt))
H_eval = sorted(set(int(round(f*char_tu/dt)) for f in [1, 2, 3])); H_max = max(H_eval)
anchors = (i0 + ws) + np.arange(66)*sp
anchors = anchors[anchors + H_max < len(XC)]
tr = np.arange(0, len(anchors)-22); te = np.arange(len(anchors)-22, len(anchors))
Xs = XC[:, [0, 1, 2, 3]]

def W_of(seed):
    Xin = _d[f"dn_{seed}"][:, [0, 1, 2, 3]]
    return np.asarray([Xin[a - (1-np.arange(2))*ws].reshape(-1) for a in anchors])

cache = pickle.load(open("emb_cache.pkl", "rb"))
def key(tt, seed):
    return ("IV", tt, int(seed), 8, 1, "Z", 500, (0, 1, 2, 3), 2, 10.0, 66)

def skill_of(E_by_seed):
    e_p, p_p = [], []
    for s in SEEDS:
        E = E_by_seed[s]
        for H in H_eval:
            Y = Xs[anchors+H] - Xs[anchors]
            m = RidgeCV(alphas=np.logspace(-6, 3, 19), fit_intercept=True).fit(E[tr], Y[tr])
            e_p.append(((m.predict(E[te]) - Y[te])**2).mean(1))
            p_p.append((Y[te]**2).mean(1))
    return 1 - np.concatenate(e_p).mean()/np.concatenate(p_p).mean()

def input_info(t):
    """R² medio de reconstruir la ventana W desde el embedding E (fit en train, eval en test)."""
    r2s = []
    for s in SEEDS:
        E = cache[key(t, s)]; W = W_of(s)
        m = LinearRegression().fit(E[tr], W[tr]); pred = m.predict(E[te])
        ss_res = ((pred - W[te])**2).sum(0); ss_tot = ((W[te] - W[te].mean(0))**2).sum(0)
        r2s.append(np.mean(1 - ss_res/ss_tot))
    return float(np.mean(r2s))

# ---- panel (a): probes únicos 0.3–2.9 ----
probes_a = [round(0.3 + 0.1*i, 1) for i in range(27)]
sk_a = [skill_of({s: cache[key(t, s)] for s in SEEDS}) for t in probes_a]

# ---- panel (b): zoom con R² ----
probes_b = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
sk_b = [skill_of({s: cache[key(t, s)] for s in SEEDS}) for t in probes_b]
r2_b = [input_info(t) for t in probes_b]

# ---- panel (c): mapa de pares sintetizado ----
n = len(probes_a)
M = np.full((n, n), np.nan)
for i, t1 in enumerate(probes_a):
    for j, t2 in enumerate(probes_a):
        if j > i:
            M[i, j] = skill_of({s: np.concatenate([cache[key(t1, s)], cache[key(t2, s)]], axis=1)
                                for s in SEEDS})
np.save("pair_skill_matrix_full.npy", M)

# ---- CSV con los números de (a) y (b) ----
import csv
with open("fig_probes_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["panel", "t_us", "skill", "r2_input"])
    for t, s in zip(probes_a, sk_a): w.writerow(["a", t, f"{s:.4f}", ""])
    for t, s, r in zip(probes_b, sk_b, r2_b): w.writerow(["b", t, f"{s:.4f}", f"{r:.4f}"])

# ---- figura ----
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
ax = axes[0]
ax.plot(probes_a, sk_a, "o-", color="#2a78d6", ms=4)
ax.axhline(0, color="#999", ls=":")
ax.axvspan(0.65, 0.80, color="#d64a2a", alpha=0.12)
ax.set_xlabel("probe time t [µs]"); ax.set_ylabel("skill (single probe)")
ax.set_title("(a) Information decay with anti-resonance (~0.7 µs)")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(probes_b, sk_b, "o-", color="#2a78d6", label="prediction skill")
ax.plot(probes_b, r2_b, "s--", color="#d64a2a", label=r"linear $R^2(W \leftarrow E)$")
ax.axhline(0, color="#999", ls=":")
ax.set_xlabel("probe time t [µs]")
ax.set_title("(b) Node close-up: embedding loses input dependence")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
im = ax.imshow(M, origin="lower", cmap="RdBu_r", vmin=-0.7, vmax=0.7,
               extent=[probes_a[0]-0.05, probes_a[-1]+0.05, probes_a[0]-0.05, probes_a[-1]+0.05])
ax.set_xlabel("probe t$_2$ [µs]"); ax.set_ylabel("probe t$_1$ [µs]")
ax.set_title("(c) Two-probe skill map (synthesized)")
plt.colorbar(im, ax=ax, label="skill")
fig.tight_layout()
fig.savefig("estructura_probes_final.png", dpi=300)
print("Regenerada: estructura_probes_final.png  |  datos: fig_probes_data.csv")
print(f"chequeos: skill(0.5)={sk_b[0]:+.3f} (ref +0.643) | skill(0.70)={sk_b[3]:+.3f} (ref -0.086) | "
      f"R2(0.70)={r2_b[3]:+.3f} (ref -0.351)")

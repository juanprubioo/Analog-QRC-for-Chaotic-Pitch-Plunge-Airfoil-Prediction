"""fig_parity.py — Regenera la evidencia de degeneración de paridad (Tabla parity del paper).

Computa, desde emb_cache.pkl, las correlaciones por átomo de la lectura Z (probe único,
t = 0.6 µs) con la entrada codificada x_i (canal de signo) y con |x_i - 0.5| (canal de
paridad), para a = 30 µm (V ≈ 0) y a = 10 µm (V ≈ Ω). Produce además una figura de
dispersión que muestra el plegamiento: sin interacciones, x y 1−x colapsan a la misma
lectura.

Requiere: airfoil_qrc_v5.py, airfoil_simulator.py, data_cache.npz, emb_cache.pkl
(chunks ts=1, t=0.6, espaciados 10.0 y 30.0).
Salida: fig_parity.png y tab_parity.csv.
Determinista dado el caché.

Valores de referencia (protocolo de 3 semillas; deben reproducirse exactamente):
  a=30 µm: |corr(E,x)| = 0.126,  |corr(E,|x-0.5|)| = 0.976
  a=10 µm: |corr(E,x)| = 0.908,  |corr(E,|x-0.5|)| = 0.242
(Nota: el diagnóstico exploratorio de la sesión usó solo la semilla 1234 y dio
0.092/0.911; los valores canónicos del paper son los de 3 semillas de arriba.)
"""
import warnings, pickle, csv
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [1234, 1335, 1436]
T_PROBE = 0.6
SPACINGS = [(30.0, r"$a=30\,\mu$m ($V\approx 0$)"), (10.0, r"$a=10\,\mu$m ($V\approx\Omega$)")]

_d = np.load("data_cache.npz"); XC = _d["XC"]; dt = float(_d["dt"])
i0 = int(round(500.0/dt)); ws = int(round(5.0/dt))
x = XC[i0:i0+int(round(600/dt)), 0]; x = x - x.mean()
ac = np.correlate(x, x, "full")[len(x)-1:len(x)-1+int(round(150/dt))]; ac = ac/ac[0]
char_tu = float(np.where(ac < 1/np.e)[0][0]*dt)
sp = int(round(char_tu/dt))
H_max = int(round(3*char_tu/dt))
anchors = (i0 + ws) + np.arange(66)*sp
anchors = anchors[anchors + H_max < len(XC)]
tr = np.arange(0, len(anchors)-22)

cache = pickle.load(open("emb_cache.pkl", "rb"))
def key(spum, seed):
    return ("IV", T_PROBE, int(seed), 8, 1, "Z", 500, (0, 1, 2, 3), 2, spum, 66)

def Wsc_of(seed):
    Xin = _d[f"dn_{seed}"][:, [0, 1, 2, 3]]
    W = np.asarray([Xin[a - (1-np.arange(2))*ws].reshape(-1) for a in anchors])
    return np.clip(MinMaxScaler((0, 1)).fit(W[tr]).transform(W), 0, 1)

# ---- tabla: correlaciones medias por átomo, promediadas sobre semillas ----
rows = []
scatter = {}
for spum, label in SPACINGS:
    cx_all, cax_all = [], []
    for s in SEEDS:
        E = cache[key(spum, s)]; Wsc = Wsc_of(s)
        cx_all += [abs(np.corrcoef(E[:, i], Wsc[:, i])[0, 1]) for i in range(8)]
        cax_all += [abs(np.corrcoef(E[:, i], np.abs(Wsc[:, i]-0.5))[0, 1]) for i in range(8)]
    rows.append({"spacing_um": spum, "corr_sign": np.mean(cx_all), "corr_parity": np.mean(cax_all)})
    # para la figura: todos los pares (x_i, E_i) de la semilla 1234
    E = cache[key(spum, 1234)]; Wsc = Wsc_of(1234)
    scatter[spum] = (Wsc[:, :8].ravel(), E.ravel())

with open("tab_parity.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["spacing_um", "abs_corr_E_x", "abs_corr_E_absx05"])
    for r in rows: w.writerow([r["spacing_um"], f"{r['corr_sign']:.3f}", f"{r['corr_parity']:.3f}"])

print("Tabla de paridad (probe único t=0.6 µs, media por átomo sobre 3 semillas):")
print(f"{'espaciado':>12} | {'|corr(E,x)|':>12} | {'|corr(E,|x-0.5|)|':>18}")
for r in rows:
    print(f"{r['spacing_um']:>10.1f} µm | {r['corr_sign']:>12.3f} | {r['corr_parity']:>18.3f}")

# ---- figura de dispersión: el plegamiento de paridad ----
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for ax, (spum, label) in zip(axes, SPACINGS):
    xs, es = scatter[spum]
    ax.scatter(xs, es, s=6, alpha=0.35,
               color="#d64a2a" if spum == 30.0 else "#2a78d6")
    ax.set_xlabel(r"encoded input $x_i$"); ax.set_title(label)
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r"readout $\langle n_i \rangle$")
axes[0].annotate("x and 1−x fold onto\nthe same readout\n(parity degeneracy)",
                 xy=(1, 1), xycoords="axes fraction", ha="center", fontsize=8)
fig.suptitle("Single-atom readout vs. encoded input, probe at 0.6 µs", y=1.0)
fig.tight_layout()
fig.savefig("fig_parity.png", dpi=300)
print("\nfigura: fig_parity.png  |  datos: tab_parity.csv")

"""fig_pipeline.py — Figura 1 del paper: esquema del pipeline QRC aeroelástico.

Cinco etapas: (a) ventana aeroelástica denoised -> (b) codificación en detunings
locales -> (c) evolución de átomos neutros -> (d) embedding Z/ZZ -> (e) readout ridge.

Usa datos REALES: traza del Caso IV con ruido y denoised (data_cache.npz), una
ventana real escalada, y un embedding real del caché (8 átomos, ts=2, tt=2.0,
seed 1234) para el panel (d).

Requiere: data_cache.npz, emb_cache.pkl.
Salida: fig_pipeline.png y fig_pipeline.pdf (para el .tex).
"""
import warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from sklearn.preprocessing import MinMaxScaler

# ---------- datos reales ----------
_d = np.load("data_cache.npz")
XC = _d["XC"]; DN = _d["dn_1234"]; dt = float(_d["dt"])
cache = pickle.load(open("emb_cache.pkl", "rb"))
E_all = cache[("IV", 2.0, 1234, 8, 2, "Z", 500, (0, 1, 2, 3), 2, 10.0, 66)]

i0 = int(round(500.0/dt)); ws = int(round(5.0/dt)); sp = 206
anchors = (i0 + ws) + np.arange(66)*sp
a = anchors[10]                       # un anchor representativo
seg = slice(a - 3*ws, a + ws)         # tramo de la serie alrededor del anchor
tau = np.arange(seg.start, seg.stop) * dt

Xin = DN[:, [0, 1, 2, 3]]
W = np.asarray([Xin[x - (1 - np.arange(2))*ws].reshape(-1) for x in anchors])
tr = np.arange(0, 44)
Wsc = np.clip(MinMaxScaler((0, 1)).fit(W[tr]).transform(W), 0, 1)
x_enc = Wsc[10]                       # la ventana escalada del anchor -> 8 detunings
E_vec = E_all[10]                     # su embedding real (16 dims)

BLUE = "#2a78d6"; RED = "#d64a2a"; GREEN = "#1baf7a"; GREY = "#555555"

fig = plt.figure(figsize=(15.5, 3.9))
gs = fig.add_gridspec(1, 5, wspace=0.55, left=0.035, right=0.985, top=0.80, bottom=0.16)

# ---------- (a) ventana denoised ----------
ax = fig.add_subplot(gs[0])
noisy = XC[seg, 0] + 0.40*XC[:, 0].std()*np.random.default_rng(3).normal(size=seg.stop-seg.start)
ax.plot(tau, noisy, ".", ms=1.6, color="#bbbbbb", label="noisy obs.")
ax.plot(tau, DN[seg, 0], color=BLUE, lw=1.6, label="denoised")
for k, tt_ in enumerate([a - ws, a]):
    ax.axvline(tt_*dt, color=RED, ls="--", lw=1.1)
    ax.plot(tt_*dt, DN[tt_, 0], "o", color=RED, ms=6, zorder=5)
ax.annotate(r"$\Delta_w = 5$ tu", xy=((a - ws/2)*dt, ax.get_ylim()[1]*0.92),
            ha="center", fontsize=8, color=RED)
ax.set_xlabel(r"$\tau$ [tu]", fontsize=8); ax.set_yticks([])
ax.set_title("(a) Denoised input window\n" r"$\mathbf{w}\in\mathbb{R}^{8}$: 4 states $\times$ 2 samples",
             fontsize=9)
ax.legend(fontsize=6.5, loc="lower left"); ax.tick_params(labelsize=7)

# ---------- (b) codificacion en detunings locales ----------
ax = fig.add_subplot(gs[1]); ax.set_xlim(-0.6, 7.6); ax.set_ylim(-1.6, 2.1); ax.axis("off")
cmap = plt.get_cmap("coolwarm")
for i in range(8):
    delta = 4.5 - 9.0*x_enc[i]
    c = cmap((delta + 4.5)/9.0)
    ax.add_patch(Circle((i, 0), 0.30, facecolor=c, edgecolor="k", lw=0.8, zorder=3))
    ax.annotate(f"{delta:+.1f}", xy=(i, -0.75), ha="center", fontsize=6.2, color=GREY)
    if i < 7:
        ax.plot([i+0.32, i+0.68], [0, 0], color=GREY, lw=1.0, zorder=1)
ax.annotate(r"$a=10\,\mu$m", xy=(3.5, 0.55), ha="center", fontsize=7.5, color=GREY)
ax.annotate(r"$\Delta_i = \Delta_{\rm enc}/2 - \Delta_{\rm enc}\,x_i$" "\n"
            r"$\Delta_{\rm enc}=9$ rad/$\mu$s", xy=(3.5, 1.45), ha="center", fontsize=8)
ax.set_title("(b) Local-detuning encoding\n8 atoms, one feature per site", fontsize=9)

# ---------- (c) evolucion cuantica ----------
ax = fig.add_subplot(gs[2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.annotate(r"$H=\sum_i\left[\frac{\Omega}{2}\sigma_x^{(i)}-\Delta_i n_i\right]+\sum_{i<j}V_{ij}n_in_j$",
            xy=(0.5, 0.86), ha="center", fontsize=8.6)
tline = np.linspace(0.08, 0.92, 100)
ax.plot(tline, 0.52 + 0*tline, color="k", lw=1.0)
ax.fill_between([0.08, 0.92], 0.52, 0.68, color=BLUE, alpha=0.18)
ax.annotate(r"$\Omega = 2\pi$ rad/$\mu$s (quench)", xy=(0.5, 0.60), ha="center", fontsize=7.5, color=BLUE)
for frac, lab in [(0.5, r"$t_1$"), (0.92, r"$t_2$")]:
    ax.axvline(frac, ymin=0.18, ymax=0.66, color=RED, ls="--", lw=1.2)
    ax.annotate(lab, xy=(frac, 0.12), ha="center", fontsize=8.5, color=RED)
ax.annotate("projective\nmeasurement", xy=(0.71, 0.30), ha="center", fontsize=7, color=RED)
ax.annotate(r"$V=C_6/a^6=5.42$ rad/$\mu$s", xy=(0.5, 0.74), ha="center", fontsize=7.5, color=GREY)
ax.set_title("(c) Neutral-atom evolution\nindependent run per probe time", fontsize=9)

# ---------- (d) embedding Z ----------
ax = fig.add_subplot(gs[3])
colors = [BLUE]*8 + ["#173f6e"]*8
ax.bar(np.arange(16), E_vec, color=colors, width=0.8)
ax.axhline(0, color="k", lw=0.7)
ax.set_xticks([3.5, 11.5]); ax.set_xticklabels([r"$\langle Z_i\rangle$ at $t_1$",
                                                 r"$\langle Z_i\rangle$ at $t_2$"], fontsize=7.5)
ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=7)
ax.set_title("(d) $Z$ / $ZZ$ embedding\n" r"$E\in\mathbb{R}^{N_aN_t}$ (500 shots)", fontsize=9)

# ---------- (e) readout ridge ----------
ax = fig.add_subplot(gs[4]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.10, 0.42), 0.80, 0.30, boxstyle="round,pad=0.03",
             facecolor="#eef4fb", edgecolor=BLUE, lw=1.2))
ax.annotate(r"$\hat{\mathbf{y}} = \mathbf{W}_{\rm ridge}\,E$", xy=(0.5, 0.57), ha="center", fontsize=10)
ax.annotate("RidgeCV, fitted on training anchors only", xy=(0.5, 0.33), ha="center", fontsize=7.2, color=GREY)
ax.annotate(r"target: $\mathbf{y}=\mathbf{x}_{\rm clean}(\tau+H)-\mathbf{x}_{\rm clean}(\tau)$"
            "\n" r"$H\in\{1,2,3\}\,\tau_c$", xy=(0.5, 0.12), ha="center", fontsize=7.8)
ax.set_title("(e) Ridge readout\nfinite-horizon increment prediction", fontsize=9)

# ---------- flechas entre paneles (posiciones exactas desde los ejes) ----------
fig.canvas.draw()
axes_list = fig.get_axes()
bounds = sorted([(ax_.get_position().x0, ax_.get_position().x1) for ax_ in axes_list])
for k in range(4):
    gap_l = bounds[k][1]; gap_r = bounds[k+1][0]
    xm = 0.5*(gap_l + gap_r); half = 0.42*(gap_r - gap_l)
    fig.patches.append(FancyArrowPatch((xm - half, 0.47), (xm + half, 0.47),
                       transform=fig.transFigure, arrowstyle="-|>",
                       mutation_scale=16, lw=1.6, color=GREY))

fig.savefig("fig_pipeline.png", dpi=200)
fig.savefig("fig_pipeline.pdf")
print("fig_pipeline.png / fig_pipeline.pdf generadas")

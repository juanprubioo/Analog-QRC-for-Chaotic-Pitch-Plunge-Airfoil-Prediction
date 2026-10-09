"""fig_pipeline_qmi.py — Figure 1 of the QMI manuscript: QRC pipeline and classical branch.

Five stages on the quantum branch, (a) denoised input window -> (b) local-detuning encoding
-> (c) neutral-atom evolution with two independent probes -> (d) Z embedding -> (e) ridge
readout, plus the classical branch that receives the identical window.

Uses real data at the preregistered operating point (8 atoms, Z readout, t_tot = 1.0 us,
probes 0.5 and 1.0 us, seed 1234): the denoised trace from data_cache.npz and the cached
Bloqade embedding from emb_cache.pkl for one representative anchor.

Run from emulator/ (needs ../data_cache.npz or data_cache.npz, emb_cache.pkl):
    python3 figures/fig_pipeline_qmi.py
Outputs: figures/fig_pipeline.png and figures/fig_pipeline.pdf
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
EMU = HERE.parent
dc = EMU / "data_cache.npz"
if not dc.exists():
    dc = EMU.parent / "data_cache.npz"
_d = np.load(dc)
XC, DN, dt = _d["XC"], _d["dn_1234"], float(_d["dt"])
cache = pickle.load(open(EMU / "emb_cache.pkl", "rb"))
E_all = np.asarray(cache[("IV", 1.0, 1234, 8, 2, "Z", 500, (0, 1, 2, 3), 2, 10.0, 66)], float)

i0, ws, sp = int(round(500.0 / dt)), int(round(5.0 / dt)), int(round(10.30 / dt))
anchors = (i0 + ws) + np.arange(66) * sp
K = 10                                  # representative training anchor
a = anchors[K]
seg = slice(a - 3 * ws, a + ws)
tau = np.arange(seg.start, seg.stop) * dt
W = np.asarray([DN[x - (1 - np.arange(2)) * ws][:, [0, 1, 2, 3]].reshape(-1) for x in anchors])
Wsc = np.clip(MinMaxScaler((0, 1)).fit(W[:44]).transform(W), 0, 1)
x_enc, E_vec = Wsc[K], E_all[K]

BLUE, DARK, RED, GREY, GREEN = "#2a78d6", "#173f6e", "#d64a2a", "#666666", "#1b8a5a"
plt.rcParams.update({"font.size": 6.5})
# sized to the sn-jnl text width (372 pt = 5.15 in) so fonts print at their nominal size
fig = plt.figure(figsize=(5.15, 4.1))
gs = fig.add_gridspec(2, 3, wspace=0.55, hspace=0.95, left=0.03, right=0.985, top=0.90, bottom=0.08)
TFS = 7

# (a) input window
ax = fig.add_subplot(gs[0, 0])
noisy = XC[seg, 0] + 0.40 * XC[:, 0].std() * np.random.default_rng(3).normal(size=seg.stop - seg.start)
ax.plot(tau, noisy, ".", ms=1.2, color="#bdbdbd", label="noisy")
ax.plot(tau, DN[seg, 0], color=BLUE, lw=1.0, label="denoised")
for t_ in (a - ws, a):
    ax.axvline(t_ * dt, color=RED, ls="--", lw=0.7)
    ax.plot(t_ * dt, DN[t_, 0], "o", color=RED, ms=3, zorder=5)
ax.set_xlabel(r"$\tau$ (tu)", fontsize=6.5, labelpad=1); ax.set_yticks([]); ax.tick_params(labelsize=6, pad=1)
ax.set_title("(a) Input window\n4 states $\\times$ 2 times", fontsize=TFS)
ax.legend(fontsize=5.5, loc="lower left", frameon=False, handlelength=1.0, borderpad=0.1)

# (b) local-detuning encoding
ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(-0.7, 7.7); ax.set_ylim(0, 1); ax.axis("off")
cmap = plt.get_cmap("coolwarm")
dets = 4.5 - 9.0 * x_enc
ax.plot([0, 7], [0.52, 0.52], color=GREY, lw=0.6, zorder=1)
ax.scatter(np.arange(8), np.full(8, 0.52), s=34, c=[cmap((d + 4.5) / 9.0) for d in dets],
           edgecolors="k", linewidths=0.4, zorder=3)
ax.text(3.5, 0.88, r"$\Delta_i=4.5-9\,x_i$", ha="center", fontsize=6.8)
ax.text(3.5, 0.66, r"$a=10\,\mu$m", ha="center", fontsize=6, color=GREY)
ax.text(3.5, 0.34, "blue $\\Delta_i<0$, red $\\Delta_i>0$", ha="center", fontsize=5.8, color=GREY)
ax.text(3.5, 0.05, "sign-definite variant:\n$\\Delta_i=-0.5-9\\,x_i$ (Sec. 4.3)", ha="center", fontsize=5.8, color=GREY)
ax.set_title("(b) Local-detuning\nencoding (rad/$\\mu$s)", fontsize=TFS)

# (c) evolution and probes
ax = fig.add_subplot(gs[0, 2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.92, r"$H=\Sigma_i\left[\frac{\Omega}{2}\sigma_x^{(i)}-\Delta_i n_i\right]$", ha="center", fontsize=6.6)
ax.text(0.5, 0.77, r"$+\,\Sigma_{i<j}\,V_{ij}\,n_in_j$", ha="center", fontsize=6.6)
ax.fill_between([0.06, 0.94], 0.42, 0.58, color=BLUE, alpha=0.18, lw=0)
ax.plot([0.06, 0.94], [0.42, 0.42], color="k", lw=0.6)
for fx, lab in [(0.50, r"$t_1$"), (0.94, r"$t_2$")]:
    ax.plot([fx, fx], [0.33, 0.60], color=RED, ls="--", lw=0.8)
    ax.text(fx, 0.23, lab, ha="center", fontsize=6.5, color=RED)
ax.text(0.03, 0.24, r"$\Omega=2\pi$" "\n" r"$V=5.42$" "\n" r"(rad/$\mu$s)", ha="left", va="center", fontsize=5.6, color=DARK)
ax.text(0.5, 0.04, r"$t_1=0.5$, $t_2=1.0\,\mu$s", ha="center", fontsize=6, color=RED)
ax.set_title("(c) Quench from $|g\\ldots g\\rangle$,\none run per probe time", fontsize=TFS)

# (d) Z embedding
ax = fig.add_subplot(gs[1, 2])
ax.bar(np.arange(16), E_vec, color=[BLUE] * 8 + [DARK] * 8, width=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks([3.5, 11.5]); ax.set_xticklabels([r"$\langle Z_i\rangle(t_1)$", r"$\langle Z_i\rangle(t_2)$"], fontsize=6)
ax.set_ylim(-1, 1); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=6, pad=1)
ax.set_title("(d) Embedding $E\\in\\mathbb{R}^{16}$\n(Bloqade, 500 shots)", fontsize=TFS)

# (e) readout
ax = fig.add_subplot(gs[1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.06, 0.55), 0.88, 0.28, boxstyle="round,pad=0.02",
                            facecolor="#eef4fb", edgecolor=BLUE, lw=0.8))
ax.text(0.5, 0.69, r"$\hat{\mathbf{y}}=\mathbf{W}E+\mathbf{b}$", ha="center", va="center", fontsize=7.5)
ax.text(0.5, 0.38, "ridge, penalty chosen by\nCV on training anchors", ha="center", va="center", fontsize=5.8, color=GREY)
ax.text(0.5, 0.08, r"$\mathbf{y}=\mathbf{x}(\tau+H)-\mathbf{x}(\tau)$" "\n" r"$H\in\{1,2,3\}\,\tau_c$",
        ha="center", va="center", fontsize=6)
ax.set_title("(e) Readout:\nincrement prediction", fontsize=TFS)

# (f) classical branch
ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.04, 0.12), 0.92, 0.74, boxstyle="round,pad=0.02",
                            facecolor="#eef7f1", edgecolor=GREEN, lw=0.8))
ax.text(0.5, 0.49, "linear ridge\nNG-RC (degree 2, 3)\nrandom features\n(dimension-matched)\n\nsame window,\ntargets and readout",
        ha="center", va="center", fontsize=5.9, color="#114d32", linespacing=1.3)
ax.set_title("(f) Classical menu", fontsize=TFS, color="#114d32")

# arrows
fig.canvas.draw()
rend = fig.canvas.get_renderer()
def ext(ax_):
    bb = ax_.get_tightbbox(rend).transformed(fig.transFigure.inverted()) if ax_.axison else \
        ax_.get_position()
    return bb.x0, bb.x1, bb.y0, bb.y1
A_, B_, C_, D_, E_, F_ = [ext(x) for x in fig.get_axes()]
def arrow(p0, p1, color=GREY, head=True):
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>" if head else "-",
                                       mutation_scale=8, lw=0.9, color=color, shrinkA=0, shrinkB=0))
y_top = 0.5 * (B_[2] + B_[3]); y_bot = 0.5 * (E_[2] + E_[3])
arrow((A_[1] + 0.008, y_top), (B_[0] - 0.008, y_top))
arrow((B_[1] + 0.008, y_top), (C_[0] - 0.008, y_top))
xc = 0.5 * (C_[0] + C_[1])
arrow((xc, C_[2] - 0.005), (xc, D_[3] + 0.055))
arrow((D_[0] - 0.008, y_bot), (E_[1] + 0.008, y_bot))
xa = 0.5 * (A_[0] + A_[1])
arrow((xa, A_[2] - 0.005), (xa, F_[3] + 0.055), color=GREEN)
arrow((F_[1] + 0.008, y_bot), (E_[0] - 0.008, y_bot), color=GREEN)

out = HERE / "fig_pipeline"
fig.savefig(f"{out}.png", dpi=300)
fig.savefig(f"{out}.pdf")
print(f"Saved {out}.png / .pdf  (anchor {K}, t_tot = 1.0 us, seed 1234)")

"""fig_exact_chain.py — Appendix B of the QMI manuscript: exact eight-atom chain diagnostic.

H = sum_i [ (Omega/2) sigma_x^i - Delta_i n_i ] + sum_{i<j} (C6 / r_ij^6) n_i n_j,
psi(0) = all-ground, constant quench (as build_local_task in airfoil_qrc_v5.py).
Encodings: symmetric Delta_i = 4.5 - 9 x_i ; sign-definite Delta_i = -0.5 - 9 x_i (rad/us).
For NSAMPLES inputs x ~ U[0,1]^8 we compute <n_i(t)> exactly (dim 256) and the mean
per-atom channels
    sign   : mean_i |corr(<n_i(t)>, x_i)|
    parity : mean_i |corr(<n_i(t)>, |x_i - 0.5|)|
for V = C6/a^6 at a = 10 um (5.42 rad/us) and for V = 0.

Single-atom limit: for V = 0 each atom evolves independently,
    <n(t)> = (Omega^2 / Omega_eff^2) sin^2(Omega_eff t / 2),  Omega_eff = sqrt(Omega^2 + Delta^2),
which is even in Delta, so the sign channel vanishes identically for the symmetric encoding.
The first zero of the single-atom parity channel is found numerically on a dense uniform
grid of x (it has no closed form).

Run from emulator/figures/:  python3 fig_exact_chain.py
Outputs: fig_exact_chain.png / .pdf, exact_chain_channels.csv, exact_chain_nodes.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMEGA, ENC, C6, A_UM, N = 6.283, 9.0, 5.42e6, 10.0, 8
NSAMPLES = 200
TS = np.round(np.arange(0.02, 3.0001, 0.005), 4)
OFFSETS = {"symmetric": ENC / 2.0, "sign-definite": -0.5}
DIM = 2 ** N
states = np.arange(DIM)
NMAT = ((states[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
XOP = np.zeros((DIM, DIM))
for i in range(N):
    XOP[states, states ^ (1 << i)] += OMEGA / 2.0


def vdiag(V):
    v = np.zeros(DIM)
    for i in range(N):
        for j in range(i + 1, N):
            v += V / (j - i) ** 6 * NMAT[:, i] * NMAT[:, j]
    return v


def channels(X, offset, V):
    vd = vdiag(V)
    Nt = np.empty((len(X), len(TS), N))
    for k, x in enumerate(X):
        w, U = np.linalg.eigh(XOP + np.diag(-(NMAT @ (offset - ENC * x)) + vd))
        c0 = U[0, :]
        psi = U @ (np.exp(-1j * np.outer(w, TS)) * c0[:, None])     # (DIM, T)
        Nt[k] = (np.abs(psi) ** 2).T @ NMAT
    sgn = np.array([np.mean([abs(np.corrcoef(Nt[:, t, i], X[:, i])[0, 1]) for i in range(N)]) for t in range(len(TS))])
    par = np.array([np.mean([abs(np.corrcoef(Nt[:, t, i], np.abs(X[:, i] - 0.5))[0, 1]) for i in range(N)]) for t in range(len(TS))])
    return sgn, par


def first_min(y, tmin, tmax):
    m = (TS >= tmin) & (TS <= tmax)
    i = np.argmin(np.where(m, y, np.inf))
    return TS[i], y[i]


X = np.random.default_rng(7).uniform(0, 1, (NSAMPLES, N))
V10 = C6 / A_UM ** 6
res = {}
for enc, off in OFFSETS.items():
    for V in (0.0, V10):
        res[(enc, V)] = channels(X, off, V)

# single-atom analytic limit (dense grid in x)
xg = np.linspace(0, 1, 20001)
D = OFFSETS["symmetric"] - ENC * xg
W = np.sqrt(OMEGA ** 2 + D ** 2)
tf = np.arange(0.5, 1.2, 0.0005)
c1 = np.array([abs(np.corrcoef((OMEGA ** 2 / W ** 2) * np.sin(W * t / 2) ** 2, np.abs(xg - 0.5))[0, 1]) for t in tf])
t_single = tf[np.argmin(c1)]

nodes = pd.DataFrame([
    dict(level="single atom (analytic, V=0)", channel="parity", t_node_us=t_single, value=c1.min()),
    dict(level="exact chain, V=0", channel="parity", t_node_us=first_min(res[("symmetric", 0.0)][1], 0.5, 1.2)[0],
         value=first_min(res[("symmetric", 0.0)][1], 0.5, 1.2)[1]),
    dict(level=f"exact chain, V={V10:.2f}", channel="sign", t_node_us=first_min(res[("symmetric", V10)][0], 0.5, 1.0)[0],
         value=first_min(res[("symmetric", V10)][0], 0.5, 1.0)[1]),
])
print(nodes.to_string(index=False))
print(f"symmetric, V=0: mean sign channel over t = {res[('symmetric', 0.0)][0].mean():.3f}")
nodes.to_csv("exact_chain_nodes.csv", index=False)
pd.DataFrame({"t_us": TS, **{f"{e}|V={v:.2f}|{ch}": res[(e, v)][k]
              for (e, v) in res for k, ch in enumerate(("sign", "parity"))}}).to_csv("exact_chain_channels.csv", index=False)

# ---------------- figure ----------------
BLUE, RED, GREY = "#2a78d6", "#d64a2a", "#777777"
fig, axes = plt.subplots(1, 2, figsize=(5.15, 2.75), sharey=True)
for ax, V, title in [(axes[0], 0.0, r"(a) $V=0$ (non-interacting)"),
                     (axes[1], V10, rf"(b) $V={V10:.2f}$ rad/$\mu$s ($a=10\,\mu$m)")]:
    s, p = res[("symmetric", V)]
    sd, _ = res[("sign-definite", V)]
    ax.plot(TS, s, color=BLUE, lw=1.4, label="symmetric, sign channel")
    ax.plot(TS, p, color=RED, lw=1.2, ls="--", label="symmetric, parity channel")
    ax.plot(TS, sd, color=GREY, lw=1.0, ls=":", label="sign-definite, sign channel")
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel(r"probe time $t$ ($\mu$s)", fontsize=7)
    ax.tick_params(labelsize=6.5); ax.set_xlim(0, 3); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].axvline(t_single, color=RED, lw=0.8, alpha=0.6)
axes[0].annotate(f"single-atom\nparity node\n{t_single:.3f} $\\mu$s", xy=(t_single, 0.5), xytext=(0.08, 0.42),
                 fontsize=6, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
tn = nodes.iloc[2].t_node_us
axes[1].axvspan(0.65, 0.75, color="k", alpha=0.08, lw=0)
axes[1].annotate(f"sign-channel node {tn:.3f} $\\mu$s\n(shaded: emulator anti-resonance)",
                 xy=(tn, nodes.iloc[2].value), xytext=(0.95, 0.86), fontsize=6,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5),
                 arrowprops=dict(arrowstyle="->", color="k", lw=0.6))
axes[0].set_ylabel("mean per-atom |correlation|", fontsize=7)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.3, frameon=False, handlelength=1.8, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("fig_exact_chain.png", dpi=300)
fig.savefig("fig_exact_chain.pdf")
print("Saved fig_exact_chain.png/.pdf, exact_chain_channels.csv, exact_chain_nodes.csv")

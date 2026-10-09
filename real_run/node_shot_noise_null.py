"""node_shot_noise_null.py — Shot-noise null distribution for the hardware probes.

Question tested (Reviewer 1, AIAA J 2026-07-J067734): is the hardware node skill
(+0.202 at 0.70 us) distinguishable from noise-free dynamics sampled with the SAME
per-anchor shot counts that survived post-selection on Aquila?

Method
  1. Exact 8-atom evolution (2^8 = 256) of the preregistered program (hw_common.py):
     Delta_i = 4.5 - 9 x_i rad/us, Omega = 6.283 rad/us, a = 10 um, C6 = 5.42e6 rad/us um^6,
     amendment E1 waveform (0.05 us linear ramps on Omega and local detuning,
     programmed time = probe + 0.05 us). Constant-quench profile computed as a check.
  2. For each probe, K replicates: multinomial bitstring sampling per anchor with
     n_shots = 100, = hardware post-selected count, or = 500; <Z_i> embeddings
     (ground = +1, Bloqade convention); skill via the preregistered readout
     (RidgeCV, 44/22, H = {1,2,3} tau_c, pooled).
  3. Optional: add single-site detection errors to the sampled bits.
  4. Report percentile of the hardware skill in the matched-shot distribution.

Run from real_run/ (needs hw_common.py, data_cache.npz, emb_cache.pkl,
hw_embeddings_real.npz):
    python3 node_shot_noise_null.py            # K = 300
    python3 node_shot_noise_null.py 1000       # K = 1000
Outputs: node_null_probs.npz (cache), node_null_skills.csv, node_null_summary.csv,
         fig_hw_reemulation.png / .pdf
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import RidgeCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hw_common as hw

K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROOT = Path(__file__).resolve().parent
PROBES = [0.5, 0.7, 1.8]
TAGS = {0.5: "0p5", 0.7: "0p7", 1.8: "1p8"}
C6 = 5.42e6           # rad/us * um^6 (Braket/Bloqade value for Rb 70S)
N = hw.N_ATOMS
DIM = 2 ** N

# ---------------- protocol (identical to analyze_hardware.py) ----------------
proto = hw.load_protocol()
A, tr, te = proto["anchors"], proto["tr"], proto["te"]
Xc, H_eval, Wsc = proto["Xc_sel"], proto["H_eval"], proto["Wsc"]


def skill_of(E):
    v = ~np.isnan(E).any(axis=1)
    trv, tev = tr[v[tr]], te[v[te]]
    e_pool, p_pool = [], []
    for H in H_eval:
        Y = Xc[A + H] - Xc[A]
        m = RidgeCV(alphas=np.logspace(-6, 3, 19), fit_intercept=True).fit(E[trv], Y[trv])
        e_pool.append(((m.predict(E[tev]) - Y[tev]) ** 2).mean(1))
        p_pool.append((Y[tev] ** 2).mean(1))
    return 1.0 - np.concatenate(e_pool).mean() / np.concatenate(p_pool).mean()


# ---------------- exact dynamics ----------------
states = np.arange(DIM)
NMAT = ((states[:, None] >> np.arange(N)[None, :]) & 1).astype(float)   # n_i: 1 = Rydberg
ZSIGN = 1.0 - 2.0 * NMAT                                                 # ground = +1
VDIAG = np.zeros(DIM)
for i in range(N):
    for j in range(i + 1, N):
        VDIAG += C6 / ((j - i) * hw.SPACING_UM) ** 6 * NMAT[:, i] * NMAT[:, j]
XOP = np.zeros((DIM, DIM))
for i in range(N):
    XOP[states, states ^ (1 << i)] += hw.RABI / 2.0
PSI0 = np.zeros(DIM, complex)
PSI0[0] = 1.0
DG = hw.ENCODING_SCALE / 2.0


def hamiltonian(x, s=1.0):
    """s scales Omega and the local detuning together (E1 ramps); global detuning fixed."""
    delta = DG - hw.ENCODING_SCALE * x * s
    return s * XOP + np.diag(-(NMAT @ delta) + VDIAG)


def propagate(Hm, dt, psi):
    w, U = eigh(Hm)
    return U @ (np.exp(-1j * w * dt) * (U.conj().T @ psi))


def probs_const(x, t):
    return np.abs(propagate(hamiltonian(x), t, PSI0)) ** 2


def probs_e1(x, t, ramp=hw.RAMP_US, nr=20):
    T = t + ramp
    dt = ramp / nr
    psi = PSI0.copy()
    for k in range(nr):
        psi = propagate(hamiltonian(x, (k + 0.5) / nr), dt, psi)
    psi = propagate(hamiltonian(x), T - 2 * ramp, psi)
    for k in range(nr):
        psi = propagate(hamiltonian(x, 1 - (k + 0.5) / nr), dt, psi)
    return np.abs(psi) ** 2


cache_path = ROOT / "node_null_probs.npz"
if cache_path.exists():
    P = dict(np.load(cache_path))
else:
    P = {}
    for tt in PROBES:
        P[f"e1_{TAGS[tt]}"] = np.array([probs_e1(Wsc[k], tt) for k in range(len(A))])
        P[f"const_{TAGS[tt]}"] = np.array([probs_const(Wsc[k], tt) for k in range(len(A))])
    np.savez(cache_path, **P)

# ---------------- consistency checks ----------------
emb = pickle.load(open(ROOT / "emb_cache.pkl", "rb"))
KEY = lambda tt, ns: ("IV", tt, hw.SEED, 8, 1, "Z", ns, (0, 1, 2, 3), 2, 10.0, 66)
hwz = np.load(ROOT / "hw_embeddings_real.npz")
print("Checks (exact = infinite shots):")
for tt in PROBES:
    Ee1 = P[f"e1_{TAGS[tt]}"] @ ZSIGN
    Ec = P[f"const_{TAGS[tt]}"] @ ZSIGN
    e500 = np.asarray(emb[KEY(tt, 500)], float)
    print(f"  t={tt}: corr(E1,const)={np.corrcoef(Ee1.ravel(), Ec.ravel())[0,1]:.4f}  "
          f"corr(E1,Bloqade emu500)={np.corrcoef(Ee1.ravel(), e500.ravel())[0,1]:.4f}  "
          f"skill exact E1={skill_of(Ee1):+.3f} const={skill_of(Ec):+.3f}")

# ---------------- null distributions ----------------
rng = np.random.default_rng(20261007)


def sample_embedding(Pr, shots, eps_g2r=0.0, eps_r2g=0.0):
    E = np.full((len(A), N), np.nan)
    for k in range(len(A)):
        if shots[k] == 0:
            continue
        p = Pr[k] / Pr[k].sum()
        if eps_g2r == 0.0 and eps_r2g == 0.0:
            E[k] = (rng.multinomial(shots[k], p) @ ZSIGN) / shots[k]
        else:
            n = NMAT[rng.choice(DIM, size=shots[k], p=p)].astype(int)
            u = rng.random(n.shape)
            flip = np.where(n == 0, u < eps_g2r, u < eps_r2g)
            E[k] = (1 - 2 * np.where(flip, 1 - n, n)).mean(0)
    return E


rows, summ = [], []
for tt in PROBES:
    tag = TAGS[tt]
    Pr = P[f"e1_{tag}"]
    neff = hwz[f"neff_tt{tag}"].astype(int)
    s_hw = skill_of(hwz[f"hw_tt{tag}"])
    for lab, shots in [("100", np.full(len(A), 100)), ("hw_matched", neff), ("500", np.full(len(A), 500))]:
        S = np.array([skill_of(sample_embedding(Pr, shots)) for _ in range(K)])
        rows += [dict(probe_us=tt, shots=lab, rep=r, skill=s) for r, s in enumerate(S)]
        summ.append(dict(probe_us=tt, shots=lab, K=K, mean=S.mean(), sd=S.std(ddof=1),
                         p2p5=np.percentile(S, 2.5), p97p5=np.percentile(S, 97.5),
                         max=S.max(), skill_hw=s_hw,
                         frac_null_ge_hw=float((S >= s_hw).mean())))
        print(f"t={tt} shots={lab:>10}: mean={S.mean():+.3f} sd={S.std(ddof=1):.3f} "
              f"95%=[{np.percentile(S,2.5):+.3f},{np.percentile(S,97.5):+.3f}] max={S.max():+.3f}")
    print(f"  hardware={s_hw:+.3f}")

# detection-error-only alternative at the node (matched shots)
neff = hwz["neff_tt0p7"].astype(int)
for eg, er in [(0.01, 0.05), (0.01, 0.10), (0.03, 0.15)]:
    S = np.array([skill_of(sample_embedding(P["e1_0p7"], neff, eg, er)) for _ in range(max(100, K // 2))])
    summ.append(dict(probe_us=0.7, shots=f"hw_matched+det({eg},{er})", K=len(S), mean=S.mean(),
                     sd=S.std(ddof=1), p2p5=np.percentile(S, 2.5), p97p5=np.percentile(S, 97.5),
                     max=S.max(), skill_hw=skill_of(hwz["hw_tt0p7"]),
                     frac_null_ge_hw=float((S >= skill_of(hwz["hw_tt0p7"])).mean())))
    print(f"node, detection errors g->r={eg}, r->g={er}: 95%=[{np.percentile(S,2.5):+.3f},"
          f"{np.percentile(S,97.5):+.3f}] max={S.max():+.3f}")

pd.DataFrame(rows).to_csv(ROOT / "node_null_skills.csv", index=False)
pd.DataFrame(summ).to_csv(ROOT / "node_null_summary.csv", index=False)

# ---------------- figure ----------------
df = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, tt in zip(axes, PROBES):
    S = df[(df.probe_us == tt) & (df.shots == "hw_matched")].skill.values
    s_hw = skill_of(hwz[f"hw_tt{TAGS[tt]}"])
    ax.hist(S, bins=30, color="0.75", edgecolor="0.4", lw=0.5,
            label="noise-free, matched shots")
    ax.axvline(s_hw, color="C3", lw=2, label="Aquila")
    ax.axvline(skill_of(P[f"e1_{TAGS[tt]}"] @ ZSIGN), color="k", ls="--", lw=1,
               label="noise-free, infinite shots")
    ax.set_title(f"probe {tt:.2f} $\\mu$s", fontsize=10)
    ax.set_xlabel("skill vs. persistence")
axes[0].set_ylabel("replicates")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "fig_hw_reemulation.png", dpi=300)
fig.savefig(ROOT / "fig_hw_reemulation.pdf")
print("Saved node_null_*.csv and fig_hw_reemulation.png/.pdf")

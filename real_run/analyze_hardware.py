"""analyze_hardware.py — Preregistered analysis of the run (table + figure Sec. V.E).

Usage: python3 analyze_hardware.py [mock|real]
Reads raw shots from aquila_hw_results/, builds embeddings <Z_i> per anchor and probe,
and computes the preregistered metrics:
  (a) skill vs persistence per probe (standard protocol 44/22, H={1,2,3}τ_c,
      targets delta over clean states, RidgeCV, pooled + bootstrap CI 3000, bootstrap seed 0)
  (b) primary contrast S(0.5) − S(0.70) and its sign
  (c) ρ_probe = corr(emb_hw, emb_emu500) and corr(emb_hw, emb_emu100), flattened anchors×atoms
  (d) decomposition hw↔emu100 (decoherence+imperfections) vs emu100↔emu500 (sampling)
Outputs: hw_table_<mode>.csv, hw_skill_vs_probe_<mode>.png, hw_embeddings_<mode>.npz
"""
import sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

import hw_common as hw

MODE = sys.argv[1] if len(sys.argv) > 1 else "mock"
hw.set_mode(MODE)

proto = hw.load_protocol()
anchors, tr, te = proto["anchors"], proto["tr"], proto["te"]
Xc_sel, H_eval = proto["Xc_sel"], proto["H_eval"]
n_anchor = len(anchors)

cache = pickle.load(open(hw.ROOT / "emb_cache.pkl", "rb"))
KEY = lambda tt, ns: ("IV", tt, hw.SEED, 8, 1, "Z", ns, (0, 1, 2, 3), 2, 10.0, 66)

# ---------- hardware embeddings from raw shots ----------
E_hw, n_eff = {}, {}
missing = []
for tt in hw.PROBES_US:
    rows, effs = [], []
    for idx in range(n_anchor):
        tag = hw.task_tag(tt, idx)
        try:
            bits = hw.load_raw_bits(tag)
        except FileNotFoundError:
            missing.append(tag)
            rows.append(np.full(hw.N_ATOMS, np.nan)); effs.append(0)
            continue
        rows.append(hw.bits_to_embedding(bits)); effs.append(bits.shape[0])
    E_hw[tt] = np.asarray(rows, float)
    n_eff[tt] = np.asarray(effs, int)

if missing:
    print(f"WARNING: {len(missing)} tasks without result (computed with available anchors, effective n declared):")
    for t in missing[:10]:
        print("  -", t)

def make_readout():
    return RidgeCV(alphas=np.logspace(-6, 3, 19), fit_intercept=True)

_rng = np.random.default_rng(0)
def boot_ci(em, ep, B=3000):
    n = len(em); out = np.empty(B)
    for b in range(B):
        i = _rng.integers(0, n, n); pp = ep[i].mean()
        out[b] = 1 - em[i].mean() / pp if pp > 1e-30 else np.nan
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)

def skill_of(E):
    """Standard protocol on embedding E (66, 8); anchors with NaN are excluded (effective n)."""
    valid = ~np.isnan(E).any(axis=1)
    trv = np.array([i for i in tr if valid[i]])
    tev = np.array([i for i in te if valid[i]])
    e_pool, p_pool = [], []
    for H in H_eval:
        Y = Xc_sel[anchors + H] - Xc_sel[anchors]
        m = make_readout().fit(E[trv], Y[trv])
        e_pool.append(((m.predict(E[tev]) - Y[tev]) ** 2).mean(1))
        p_pool.append((Y[tev] ** 2).mean(1))
    EQ, EP = np.concatenate(e_pool), np.concatenate(p_pool)
    lo, hi = boot_ci(EQ, EP)
    return 1 - EQ.mean() / EP.mean(), lo, hi, len(trv), len(tev)

rows = []
for tt in hw.PROBES_US:
    e100 = np.asarray(cache[KEY(tt, 100)], float)
    e500 = np.asarray(cache[KEY(tt, 500)], float)
    Eh = E_hw[tt]
    valid = ~np.isnan(Eh).any(axis=1)
    s_hw = skill_of(Eh)
    s100 = skill_of(e100)
    s500 = skill_of(e500)
    rho_hw_500 = float(np.corrcoef(Eh[valid].ravel(), e500[valid].ravel())[0, 1])
    rho_hw_100 = float(np.corrcoef(Eh[valid].ravel(), e100[valid].ravel())[0, 1])
    rho_100_500 = float(np.corrcoef(e100.ravel(), e500.ravel())[0, 1])
    rows.append(dict(probe_us=tt,
                     skill_hw=s_hw[0], hw_lo=s_hw[1], hw_hi=s_hw[2],
                     skill_emu100=s100[0], e100_lo=s100[1], e100_hi=s100[2],
                     skill_emu500=s500[0], e500_lo=s500[1], e500_hi=s500[2],
                     rho_hw_emu500=rho_hw_500, rho_hw_emu100=rho_hw_100,
                     rho_emu100_emu500=rho_100_500,
                     n_anchor_eff=int(valid.sum()),
                     n_shots_eff_median=int(np.median(n_eff[tt][valid])) if valid.any() else 0))

df = pd.DataFrame(rows)
c_hw = df.loc[df.probe_us == 0.5, "skill_hw"].iloc[0] - df.loc[df.probe_us == 0.7, "skill_hw"].iloc[0]
c100 = df.loc[df.probe_us == 0.5, "skill_emu100"].iloc[0] - df.loc[df.probe_us == 0.7, "skill_emu100"].iloc[0]
c500 = df.loc[df.probe_us == 0.5, "skill_emu500"].iloc[0] - df.loc[df.probe_us == 0.7, "skill_emu500"].iloc[0]

pd.set_option("display.float_format", lambda v: f"{v:+.3f}")
print(f"\n=== Table V.E ({MODE}) — seed {hw.SEED}, 66 anchors, 44/22, RidgeCV, pooled H={{1,2,3}}τ_c ===")
print(df.to_string(index=False))
print(f"\nPreregistered contrast S(0.5)−S(0.70):  {MODE}={c_hw:+.3f}  emu100={c100:+.3f}  emu500={c500:+.3f}")
print(f"Primary criterion (positive sign in {MODE}): {'MET' if c_hw > 0 else 'NOT MET'}")

df.to_csv(hw.ROOT / f"hw_table_{MODE}.csv", index=False)
np.savez(hw.ROOT / f"hw_embeddings_{MODE}.npz",
         **{f"hw_tt{str(tt).replace('.','p')}": E_hw[tt] for tt in hw.PROBES_US},
         **{f"neff_tt{str(tt).replace('.','p')}": n_eff[tt] for tt in hw.PROBES_US})

fig, ax = plt.subplots(figsize=(6.0, 4.2))
x = df["probe_us"].values
for col, lo, hi, lab, mk in [("skill_hw", "hw_lo", "hw_hi", "hardware" if MODE=="real" else f"hardware ({MODE})", "o"),
                             ("skill_emu100", "e100_lo", "e100_hi", "emulator, 100 shots", "s"),
                             ("skill_emu500", "e500_lo", "e500_hi", "emulator, 500 shots", "^")]:
    y = df[col].values
    yerr = np.vstack([y - df[lo].values, df[hi].values - y])
    ax.errorbar(x, y, yerr=yerr, marker=mk, capsize=3, label=lab)
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Probe time $t_t$ (µs)")
ax.set_ylabel("Skill vs. persistence")
# no title: the figure caption in the paper covers it (AIAA style)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(hw.ROOT / f"hw_skill_vs_probe_{MODE}.png", dpi=300)
fig.savefig(hw.ROOT / f"hw_skill_vs_probe_{MODE}.pdf")
print(f"\nSaved: hw_table_{MODE}.csv, hw_embeddings_{MODE}.npz, hw_skill_vs_probe_{MODE}.png")

# --- pendiente b: regresion sin intercepto hw sobre emu500 (columna b, Tabla V.E) ---
print("\nSlope b (hardware embeddings regressed on emu500, no intercept):")
for tt in hw.PROBES_US:
    _m = ~np.isnan(E_hw[tt]).any(1)
    _Em = np.asarray(cache[KEY(tt, 500)], float)[_m].ravel()
    _Eh = E_hw[tt][_m].ravel()
    print(f"  b(tt={tt}) = {float(_Eh @ _Em / (_Em @ _Em)):.2f}")

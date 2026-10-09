#!/usr/bin/env python3
"""signdef_prediction.py — Prediction skill with the sign-definite local-detuning encoding.

Tests whether the parity-degeneracy design rule (Sec. 4 of the QMI manuscript) translates
into prediction skill, under the frozen protocol of cell_4_4_v2.py (Table 2):
8 atoms, Z readout, two independent probes (t_tot/2, t_tot), 66 anchors (44/22),
H = {1,2,3} tau_c, six noise seeds, RidgeCV readout, classical menu
{linear, NG-RC deg2, deg3, dimension-matched RFF}, pooled bootstrap CI and per-seed criterion.

Encodings (rad/us, x_i in [0,1] after MinMax on the training anchors):
    symmetric      Delta_i = 4.5 - 9 x_i   (crosses zero; the published reservoir)
    sign-definite  Delta_i = -0.5 - 9 x_i  (same 9 rad/us span, never crosses zero)
Grid: t_tot in {1.0,1.5,2.0,2.5,3.0} us, lattice spacing a in {10, 30} um.

Dynamics: exact 8-atom evolution (2^8 = 256, all-pair C6/r^6 with C6 = 5.42e6 rad/us um^6,
Omega = 6.283 rad/us, all-ground start, constant quench), identical Hamiltonian and
conventions to build_local_task() in airfoil_qrc_v5.py (Bloqade bitstring 1 = ground,
Z = +1 for ground). Each probe is an independent evolution, as in Bloqade batch_assign.
Shot noise: R independent 500-shot multinomial realizations per configuration; the SAME
random stream is used for both encodings (paired comparison). Infinite-shot values are
reported as well. Validation: the symmetric a = 10 um configuration is checked against the
cached Bloqade embeddings that produce Table 2.

Run from emulator/ (needs ../data_cache.npz, ../airfoil_qrc_v5.py,
results/qrc_totaltime_emb_IV.pkl, cell_4_4_v2.py, export_totaltime_csv.py):
    python3 signdef_prediction.py          # R = 10 shot realizations
    python3 signdef_prediction.py 30
Outputs: results/signdef_prediction.csv, results/signdef_probs.npz (cache)
"""
import sys, io, contextlib, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.preprocessing import MinMaxScaler, StandardScaler

R_SHOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NSHOTS = 500
HERE = Path(__file__).resolve().parent

# ---------------- frozen protocol context (from the Table 2 exporter) ----------------
src = (HERE / "export_totaltime_csv.py").read_text(encoding="utf-8")
src = src.split("# ---------- verificación contra la Tabla 2")[0]
G = {"__file__": str(HERE / "export_totaltime_csv.py"), "__name__": "signdef_ctx"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "export_totaltime_csv.py", "exec"), G)
A, tr, te = G["anchors"], G["tr_idx"], G["te_idx"]
H_eval, Xc = G["H_eval"], G["Xc_sel"]
W_by_seed, SEEDS = G["W_by_seed"], G["seed_list"]
ridge, ngrc, RFF = G["_ridgecv"], G["build_ngrc_model"], G["_RFF"]
cache_tbl2, ek = G["cache"], G["_ek"]
TT_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
SPACINGS = [10.0, 30.0]
ENCODINGS = {"symmetric": 4.5, "sign-definite": -0.5}   # global offset; local part = -9 x_i
RABI, ENC_SCALE, C6, N = 6.283, 9.0, 5.42e6, 8
DIM = 2 ** N

# ---------------- exact dynamics ----------------
states = np.arange(DIM)
NMAT = ((states[:, None] >> np.arange(N)[None, :]) & 1).astype(float)  # 1 = Rydberg
ZSIGN = 1.0 - 2.0 * NMAT                                                # ground = +1
XOP = np.zeros((DIM, DIM))
for i in range(N):
    XOP[states, states ^ (1 << i)] += RABI / 2.0


def vdiag(a):
    v = np.zeros(DIM)
    for i in range(N):
        for j in range(i + 1, N):
            v += C6 / ((j - i) * a) ** 6 * NMAT[:, i] * NMAT[:, j]
    return v


VD = {a: vdiag(a) for a in SPACINGS}


def probs(x, offset, a, times):
    delta = offset - ENC_SCALE * x
    w, U = eigh(XOP + np.diag(-(NMAT @ delta) + VD[a]))
    c0 = U[0, :].conj()                          # <k|psi0>, psi0 = all-ground (index 0)
    return np.array([np.abs(U @ (np.exp(-1j * w * t) * c0)) ** 2 for t in times])


def wsc(seed):
    W = W_by_seed[seed]
    return np.clip(MinMaxScaler((0.0, 1.0)).fit(W[tr]).transform(W), 0.0, 1.0)


ALL_T = sorted({t for tt in TT_VALUES for t in (tt / 2, tt)})
cache_p = HERE / "results" / "signdef_probs.npz"
if cache_p.exists():
    P = dict(np.load(cache_p))
else:
    P = {}
    for enc, off in ENCODINGS.items():
        for a in SPACINGS:
            for s in SEEDS:
                X = wsc(s)
                P[f"{enc}|{a}|{s}"] = np.array([probs(X[k], off, a, ALL_T) for k in range(len(A))])
    np.savez(cache_p, **P)
TIDX = {t: i for i, t in enumerate(ALL_T)}


def embedding(Pk, tt, rng=None):
    """Pk: (n_anchor, n_times, DIM). Two independent probes (tt/2, tt); Z readout."""
    out = []
    for t in (tt / 2, tt):
        pr = Pk[:, TIDX[t], :]
        if rng is None:
            out.append(pr @ ZSIGN)
        else:
            cnt = np.array([rng.multinomial(NSHOTS, p / p.sum()) for p in pr])
            out.append(cnt @ ZSIGN / NSHOTS)
    return np.hstack(out)


# ---------------- readout and classical menu (identical to cell_4_4_v2.py) ----------------
def qrc_errors(E_by_seed):
    eq, ep, per_seed = [], [], []
    for s in SEEDS:
        E = E_by_seed[s]; se, sp = [], []
        for H in H_eval:
            Y = Xc[A + H] - Xc[A]
            se.append(((ridge().fit(E[tr], Y[tr]).predict(E[te]) - Y[te]) ** 2).mean(1))
            sp.append((Y[te] ** 2).mean(1))
        se, sp = np.concatenate(se), np.concatenate(sp)
        eq.append(se); ep.append(sp); per_seed.append(1 - se.mean() / sp.mean())
    return np.concatenate(eq), np.concatenate(ep), np.array(per_seed)


def classical():
    el, eall, ep = [], {m: [] for m in ("d1", "d2", "d3", "rff")}, []
    best_seed = []
    for s in SEEDS:
        W = W_by_seed[s]; sec = {m: [] for m in eall}; sp = []
        for H in H_eval:
            Y = Xc[A + H] - Xc[A]
            for m, d in (("d1", 1), ("d2", 2), ("d3", 3)):
                sec[m].append(((ngrc(degree=d).fit(W[tr], Y[tr]).predict(W[te]) - Y[te]) ** 2).mean(1))
            sc = StandardScaler().fit(W[tr]); Wtr, Wte = sc.transform(W[tr]), sc.transform(W[te])
            ms = []
            for r in range(5):
                f = RFF(16, W.shape[1], np.random.default_rng(100 + r))
                ms.append(((ridge().fit(f.transform(Wtr), Y[tr]).predict(f.transform(Wte)) - Y[te]) ** 2).mean(1))
            sec["rff"].append(np.mean(ms, axis=0)); sp.append((Y[te] ** 2).mean(1))
        sp = np.concatenate(sp)
        for m in eall: eall[m].append(np.concatenate(sec[m]))
        best_seed.append(max(1 - np.concatenate(sec[m]).mean() / sp.mean() for m in sec))
        ep.append(sp)
    ep = np.concatenate(ep)
    sk = {m: 1 - np.concatenate(eall[m]).mean() / ep.mean() for m in eall}
    return sk, np.concatenate(eall["d1"]), ep, np.array(best_seed)


S = lambda e, p: 1 - e.mean() / p.mean()


def boot(e, p, ref=None, B=3000, seed=0):
    rng = np.random.default_rng(seed); n = len(p); out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        out[b] = S(e[i], p[i]) - (S(ref[i], p[i]) if ref is not None else 0.0)
    return np.percentile(out, [2.5, 97.5])


cls, e_lin, ep_ref, cls_best_seed = classical()
cls_best = max(cls.values())
print(f"Classical menu: lin={cls['d1']:.3f} deg2={cls['d2']:.3f} deg3={cls['d3']:.3f} "
      f"RFF16={cls['rff']:.3f} | best per seed max = {cls_best_seed.max():.3f}")

# ---------------- validation against the Table 2 Bloqade cache ----------------
Ebl = {s: np.asarray(cache_tbl2[ek(1.0, s)], float) for s in SEEDS}
Eex = {s: embedding(P[f"symmetric|10.0|{s}"], 1.0) for s in SEEDS}
rho = np.corrcoef(np.vstack([Ebl[s] for s in SEEDS]).ravel(), np.vstack([Eex[s] for s in SEEDS]).ravel())[0, 1]
print(f"Validation, symmetric a=10 t_tot=1.0: corr(exact, Bloqade 500-shot cache) = {rho:.4f}; "
      f"skill Bloqade cache = {S(*qrc_errors(Ebl)[:2]):.3f}, exact infinite-shot = {S(*qrc_errors(Eex)[:2]):.3f}")

# ---------------- main grid ----------------
rows = []
for a in SPACINGS:
    for tt in TT_VALUES:
        res = {}
        for enc in ENCODINGS:
            Pk = {s: P[f"{enc}|{a}|{s}"] for s in SEEDS}
            eq_inf, ep_inf, ps_inf = qrc_errors({s: embedding(Pk[s], tt) for s in SEEDS})
            reps = []
            for r in range(R_SHOTS):  # identical stream for both encodings -> paired
                rng = np.random.default_rng(1000 * r + int(tt * 10) + int(a))
                reps.append(qrc_errors({s: embedding(Pk[s], tt, rng) for s in SEEDS}))
            res[enc] = (eq_inf, ep_inf, ps_inf, reps)
        for enc in ENCODINGS:
            eq_inf, ep_inf, ps_inf, reps = res[enc]
            sk_r = np.array([S(r[0], r[1]) for r in reps])
            # representative 500-shot realization = median skill; CI and per-seed from it
            k = int(np.argsort(sk_r)[len(sk_r) // 2])
            eq, ep, ps = reps[k]
            lo, hi = boot(eq, ep)
            dlo, dhi = boot(eq, ep, ref=e_lin)
            robust_frac = np.mean([r[2].min() > cls_best_seed.max() for r in reps])
            row = dict(encoding=enc, a_um=a, t_tot_us=tt,
                       skill_inf=S(eq_inf, ep_inf), skill_500_median=float(np.median(sk_r)),
                       skill_500_min=sk_r.min(), skill_500_max=sk_r.max(),
                       ci_lo=lo, ci_hi=hi, diff_vs_lin=S(eq, ep) - cls["d1"], diff_lo=dlo, diff_hi=dhi,
                       seed_min=ps.min(), seed_max=ps.max(),
                       beats_classical=lo > cls_best, beats_classical_robust=ps.min() > cls_best_seed.max(),
                       frac_reps_robust=robust_frac)
            if enc == "sign-definite":
                d = np.array([S(rs[0], rs[1]) - S(rr[0], rr[1]) for rs, rr in zip(reps, res["symmetric"][3])])
                row.update(gain_vs_symmetric_median=float(np.median(d)), gain_min=d.min(), gain_max=d.max(),
                           gain_inf=S(eq_inf, ep_inf) - S(res["symmetric"][0], res["symmetric"][1]))
            rows.append(row)
        print(f"a={a:>4} tt={tt}: " + " | ".join(
            f"{r['encoding'][:4]} inf={r['skill_inf']:+.3f} 500med={r['skill_500_median']:+.3f} "
            f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]" for r in rows[-2:]), flush=True)

df = pd.DataFrame(rows)
df.to_csv(HERE / "results" / "signdef_prediction.csv", index=False)
pd.set_option("display.float_format", lambda v: f"{v:+.3f}"); pd.set_option("display.width", 250)
print("\n", df[["encoding", "a_um", "t_tot_us", "skill_inf", "skill_500_median", "ci_lo", "ci_hi",
                "seed_min", "beats_classical", "beats_classical_robust", "frac_reps_robust",
                "gain_vs_symmetric_median", "gain_min", "gain_max"]].to_string(index=False))
print(f"\nClassical best (pooled) = {cls_best:.3f}; best classical per seed (max over seeds) = {cls_best_seed.max():.3f}")
print("Saved results/signdef_prediction.csv")

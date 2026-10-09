#!/usr/bin/env python3
"""make_supplement.py — Regenerates every number, table and figure of the Supplementary
Information of the QMI manuscript from the cached embeddings and hardware records in this
repository. Nothing is emulated here.

Sections (S#) and sources
  S1  protocol-design record: fixed vs cross-validated readouts (3 seeds)
      -> emulator/emb_cache.pkl, data_cache.npz
  S2  degree-3 NG-RC regularization checks (6 seeds, base geometry)
      -> emulator/results/qrc_totaltime_emb_IV.pkl context (cell_4_4_v2.py)
  S3  per-state and per-horizon skill, representative predictions (Reviewer 2, point 3)
      -> same context, t_tot = 1.0 us
  S4  12-atom target-definition check (3 seeds) -> emulator/emb_cache.pkl
  S5  hardware post-selection check -> real_run/hw_embeddings_real.npz, real_run/emb_cache.pkl
  S6  waveform amendment record -> real_run/mock_validation/*
  S7  sign-definite encoding, full grid -> emulator/results/signdef_prediction.csv

Run from the repository root:  python3 supplement/make_supplement.py
Outputs in supplement/: S*.csv, figS_predictions.png/.pdf, supplement_numbers.txt
"""
import io, os, sys, json, pickle, contextlib, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EMU, HW, OUT = ROOT / "emulator", ROOT / "real_run", ROOT / "supplement"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


def imp(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


q5 = imp("q5_supp", ROOT / "airfoil_qrc_v5.py")
S = lambda e, p: 1 - e.mean() / p.mean()
rng0 = np.random.default_rng(0)


def boot(e, p, B=3000):
    n = len(p); out = np.empty(B)
    for b in range(B):
        i = rng0.integers(0, n, n); out[b] = S(e[i], p[i])
    return np.percentile(out, [2.5, 97.5])


# ---------------------------------------------------------------- generic protocol (3 seeds)
D = np.load(ROOT / "data_cache.npz"); XC, dt = D["XC"], float(D["dt"])
i0 = int(round(500.0 / dt))
x_ = XC[i0:i0 + int(round(600 / dt)), 0]; x_ = x_ - x_.mean()
ac = np.correlate(x_, x_, "full")[len(x_) - 1:len(x_) - 1 + int(round(150 / dt))]; ac /= ac[0]
TAU_C = float(np.where(ac < 1 / np.e)[0][0] * dt)
SP, WS = int(round(TAU_C / dt)), int(round(5.0 / dt))
H_EVAL = sorted({int(round(f * TAU_C / dt)) for f in (1.0, 2.0, 3.0)})
EMB = pickle.load(open(EMU / "emb_cache.pkl", "rb"))
SEEDS3 = [1234, 1335, 1436]


def protocol(state_idx, window=2):
    base = i0 + (window - 1) * WS
    A = base + np.arange(66) * SP
    A = A[(A + max(H_EVAL) < len(XC))]
    tr, te = np.arange(0, len(A) - 22), np.arange(len(A) - 22, len(A))
    def Wof(s):
        Xin = D[f"dn_{s}"][:, state_idx]
        return np.asarray([Xin[a - (window - 1 - np.arange(window)) * WS].reshape(-1) for a in A])
    return A, tr, te, Wof


def ekey(tt, s, atoms, ts, ro, sidx, n):
    return ("IV", tt, int(s), atoms, ts, ro, 500, tuple(sidx), 2, 10.0, n)


# ================================================================ S1
say("== S1 protocol-design record (3 seeds, 8 atoms, two-probe Z) ==")
A, tr, te, Wof = protocol([0, 1, 2, 3])
Xc4 = XC[:, [0, 1, 2, 3]]
rows = []
def pooled(fn, seeds=SEEDS3):
    e, p = [], []
    for s in seeds:
        for H in H_EVAL:
            Y = Xc4[A + H] - Xc4[A]
            e.append(((fn(s, Y) - Y[te]) ** 2).mean(1)); p.append((Y[te] ** 2).mean(1))
    e, p = np.concatenate(e), np.concatenate(p)
    return S(e, p), boot(e, p)
for name, kw in [("NG-RC deg 2, fixed alpha=1e-4", dict(degree=2, alpha=1e-4)),
                 ("NG-RC deg 2, CV", dict(degree=2)), ("linear ridge, CV", dict(degree=1))]:
    sk, ci = pooled(lambda s, Y, kw=kw: q5.build_ngrc_model(**kw).fit(Wof(s)[tr], Y[tr]).predict(Wof(s)[te]))
    rows.append(dict(model=name, t_tot_us=np.nan, skill=sk, ci_lo=ci[0], ci_hi=ci[1])); say(f"  {name}: {sk:+.3f}")
for tt in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for mode, mk in [("fixed alpha=1e-4", lambda: Ridge(alpha=1e-4)), ("CV", lambda: RidgeCV(alphas=np.logspace(-6, 3, 19)))]:
        sk, ci = pooled(lambda s, Y, mk=mk: mk().fit(np.asarray(EMB[ekey(tt, s, 8, 2, "Z", [0, 1, 2, 3], len(A))])[tr], Y[tr])
                        .predict(np.asarray(EMB[ekey(tt, s, 8, 2, "Z", [0, 1, 2, 3], len(A))])[te]))
        rows.append(dict(model=f"QRC readout, {mode}", t_tot_us=tt, skill=sk, ci_lo=ci[0], ci_hi=ci[1]))
        say(f"  QRC tt={tt} readout {mode}: {sk:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
pd.DataFrame(rows).to_csv(OUT / "S1_protocol_record.csv", index=False)

# ================================================================ 6-seed context (Table 2)
src = (EMU / "export_totaltime_csv.py").read_text(encoding="utf-8").split("# ---------- verificación contra la Tabla 2")[0]
G = {"__file__": str(EMU / "export_totaltime_csv.py"), "__name__": "supp_ctx"}
cwd = os.getcwd(); os.chdir(EMU)
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "export_totaltime_csv.py", "exec"), G)
os.chdir(cwd)
A6, tr6, te6, H6, Xc6 = G["anchors"], G["tr_idx"], G["te_idx"], G["H_eval"], G["Xc_sel"]
W6, SEEDS6, ridge = G["W_by_seed"], G["seed_list"], G["_ridgecv"]
E6 = {s: np.asarray(G["cache"][G["_ek"](1.0, s)], float) for s in SEEDS6}

# ================================================================ S2
say("== S2 degree-3 NG-RC regularization (6 seeds, base geometry) ==")
def deg3(alphas, standardize_after=False):
    steps = [("scaler", StandardScaler()), ("poly", PolynomialFeatures(degree=3, include_bias=True))]
    if standardize_after:
        steps.append(("post", StandardScaler()))
    steps.append(("ridge", RidgeCV(alphas=alphas, fit_intercept=standardize_after)))
    return Pipeline(steps)
rows = []
for name, al, post in [("default grid 1e-3..1e3", np.logspace(-3, 3, 13), False),
                       ("grid extended 5 decades up, 1e-3..1e8", np.logspace(-3, 8, 23), False),
                       ("grid extended 5 decades down, 1e-8..1e3", np.logspace(-8, 3, 23), False),
                       ("standardized polynomial features", np.logspace(-3, 3, 13), True)]:
    e, p, chosen = [], [], []
    for s in SEEDS6:
        for H in H6:
            Y = Xc6[A6 + H] - Xc6[A6]
            m = deg3(al, post).fit(W6[s][tr6], Y[tr6]); chosen.append(m.named_steps["ridge"].alpha_)
            e.append(((m.predict(W6[s][te6]) - Y[te6]) ** 2).mean(1)); p.append((Y[te6] ** 2).mean(1))
    e, p = np.concatenate(e), np.concatenate(p)
    lo_ = np.min(chosen); hi_ = np.max(chosen)
    rows.append(dict(variant=name, skill=S(e, p), alpha_median=np.median(chosen), alpha_min=lo_, alpha_max=hi_,
                     frac_alpha_at_grid_edge=np.mean([(c <= al[0] * 1.0001) or (c >= al[-1] * 0.9999) for c in chosen])))
    say(f"  {name}: skill {S(e, p):+.3f}, alpha median {np.median(chosen):.3g} (range {lo_:.3g}-{hi_:.3g})")
pd.DataFrame(rows).to_csv(OUT / "S2_deg3_regularization.csv", index=False)

# ================================================================ S3
say("== S3 per-state and per-horizon skill, t_tot = 1.0 us (6 seeds) ==")
NAMES = ["alpha", "alpha_dot", "xi", "xi_dot"]
err = {m: np.zeros((len(SEEDS6), len(H6), len(te6), 4)) for m in ("qrc", "lin")}
pers = np.zeros((len(SEEDS6), len(H6), len(te6), 4)); preds = {}
for si, s in enumerate(SEEDS6):
    for hi, H in enumerate(H6):
        Y = Xc6[A6 + H] - Xc6[A6]
        yq = ridge().fit(E6[s][tr6], Y[tr6]).predict(E6[s][te6])
        yl = q5.build_ngrc_model(degree=1).fit(W6[s][tr6], Y[tr6]).predict(W6[s][te6])
        err["qrc"][si, hi], err["lin"][si, hi], pers[si, hi] = (yq - Y[te6]) ** 2, (yl - Y[te6]) ** 2, Y[te6] ** 2
        preds[(s, hi)] = (Y[te6], yq, yl)
rows = []
for c in range(4):
    for hi in [None, 0, 1, 2]:
        sl = slice(None) if hi is None else hi
        pq = pers[:, sl, :, c].ravel()
        rows.append(dict(state=NAMES[c], horizon="pooled" if hi is None else f"{hi + 1}tau_c",
                         skill_qrc=S(err["qrc"][:, sl, :, c].ravel(), pq), skill_lin=S(err["lin"][:, sl, :, c].ravel(), pq),
                         share_of_persistence_mse=pq.mean() / pers[:, sl].mean() / 4))
df3 = pd.DataFrame(rows); df3.to_csv(OUT / "S3_per_state_skill.csv", index=False)
say(df3[df3.horizon == "pooled"].round(3).to_string(index=False))
comp_avg = {m: np.mean([S(err[m][..., c].ravel(), pers[..., c].ravel()) for c in range(4)]) for m in err}
say(f"  component-averaged skill: QRC {comp_avg['qrc']:+.3f}, linear {comp_avg['lin']:+.3f}; "
    f"pooled-MSE skill (preregistered): QRC {S(err['qrc'].mean(-1).ravel(), pers.mean(-1).ravel()):+.3f}, "
    f"linear {S(err['lin'].mean(-1).ravel(), pers.mean(-1).ravel()):+.3f}")

# figure: representative predictions, seed 1234, pitch and plunge, H = tau_c and 3 tau_c
tte = (A6[te6] - A6[te6][0]) * dt
fig, axes = plt.subplots(2, 2, figsize=(5.15, 3.8), sharex=True)
for r, c in enumerate([0, 2]):
    for k, hi in enumerate([0, 2]):
        Y, yq, yl = preds[(1234, hi)]; ax = axes[r, k]
        ax.plot(tte, Y[:, c], "k-", lw=0.9, label="true increment")
        ax.plot(tte, yq[:, c], "o", ms=2.6, color="#2a78d6", label="QRC")
        ax.plot(tte, yl[:, c], "s", ms=2.2, mfc="none", mec="#d64a2a", mew=0.7, label="linear ridge")
        ax.axhline(0, color="0.6", lw=0.5)
        sq = S(err["qrc"][0, hi, :, c], pers[0, hi, :, c]); sl_ = S(err["lin"][0, hi, :, c], pers[0, hi, :, c])
        lab = ["pitch $\\alpha$", "plunge $\\xi$"][r]
        ax.set_title(lab + ", $H=" + ("" if hi == 0 else str(hi + 1)) + r"\tau_c$" + f"  (S: QRC {sq:.2f}, lin. {sl_:.2f})", fontsize=6.6)
        ax.tick_params(labelsize=6)
        if k == 0: ax.set_ylabel("increment", fontsize=6.5)
        if r == 1: ax.set_xlabel("time since first test anchor (tu)", fontsize=6.5)
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=6.2, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "figS_predictions.png", dpi=300); fig.savefig(OUT / "figS_predictions.pdf")

# ================================================================ S4
say("== S4 12-atom target-definition check (3 seeds) ==")
A12, tr12, te12, Wof12 = protocol([0, 1, 2, 3, 4, 5])
rows = []
for tt in [1.0, 2.0]:
    for tgt, cols in [("six states", [0, 1, 2, 3, 4, 5]), ("four mechanical states", [0, 1, 2, 3])]:
        Xt = XC[:, cols]; eq, el, ep = [], [], []
        for s in SEEDS3:
            E = np.asarray(EMB[ekey(tt, s, 12, 2, "Z", [0, 1, 2, 3, 4, 5], len(A12))]); W = Wof12(s)
            for H in H_EVAL:
                Y = Xt[A12 + H] - Xt[A12]
                eq.append(((RidgeCV(alphas=np.logspace(-6, 3, 19)).fit(E[tr12], Y[tr12]).predict(E[te12]) - Y[te12]) ** 2).mean(1))
                best = [((q5.build_ngrc_model(degree=d).fit(W[tr12], Y[tr12]).predict(W[te12]) - Y[te12]) ** 2).mean(1) for d in (1, 2)]
                el.append(best); ep.append((Y[te12] ** 2).mean(1))
        eq, ep = np.concatenate(eq), np.concatenate(ep)
        scl = max(S(np.concatenate([b[d] for b in el]), ep) for d in (0, 1))
        rows.append(dict(t_tot_us=tt, target=tgt, skill_qrc=S(eq, ep), skill_ngrc_best=scl, gap=S(eq, ep) - scl))
        say(f"  tt={tt} target={tgt}: QRC {S(eq, ep):+.3f}, best NG-RC {scl:+.3f}, gap {S(eq, ep) - scl:+.3f}")
pd.DataFrame(rows).to_csv(OUT / "S4_12atom_target.csv", index=False)

# ================================================================ S5
say("== S5 hardware post-selection check (node probe, 0.70 us) ==")
hwz = np.load(HW / "hw_embeddings_real.npz"); EH = pickle.load(open(HW / "emb_cache.pkl", "rb"))
rows = []
for tag, tt in [("0p5", 0.5), ("0p7", 0.7), ("1p8", 1.8)]:
    Eh, n = hwz[f"hw_tt{tag}"], hwz[f"neff_tt{tag}"].astype(float)
    Ee = np.asarray(EH[("IV", tt, 1234, 8, 1, "Z", 500, (0, 1, 2, 3), 2, 10.0, 66)], float)
    ok = n > 0; d = (Eh - Ee)[ok]; surv = n[ok] / 100.0
    u = d.mean(0); u /= np.linalg.norm(u); proj = d @ u
    r, pv = stats.pearsonr(surv, proj); med = np.median(surv)
    hi_, lo_ = proj[surv > med].mean(), proj[surv <= med].mean()
    rows.append(dict(probe_us=tt, n_anchors=int(ok.sum()), survival_mean=surv.mean(), survival_min=surv.min(),
                     survival_max=surv.max(), r_survival_projection=r, p_value=pv,
                     proj_above_median=hi_, proj_below_median=lo_, mean_proj=proj.mean()))
    say(f"  t={tt}: r={r:+.3f} (p={pv:.2f}); projection above/below median survival {hi_:+.3f}/{lo_:+.3f}; "
        f"survival {surv.mean():.3f} [{surv.min():.2f},{surv.max():.2f}]")
pd.DataFrame(rows).to_csv(OUT / "S5_postselection.csv", index=False)

# ================================================================ S6
say("== S6 waveform amendment record (noise-free Braket AHS simulator, 100 shots) ==")
rows = []
for lab, f in [("E1: 0.05 us ramps + area compensation", "hw_table_mock.csv"),
               ("v15: ramps min(0.2 us, t/4)", "hw_table_mock_v15ramp.csv")]:
    t = pd.read_csv(HW / "mock_validation" / f)
    for _, r in t.iterrows():
        rows.append(dict(waveform=lab, probe_us=r.probe_us, skill=r.skill_hw, rho_vs_emu500=r.rho_hw_emu500,
                         rho_sampling_baseline=r.rho_emu100_emu500))
    c = t.set_index("probe_us").skill_hw
    say(f"  {lab}: skill 0.5/0.70/1.8 = {c[0.5]:+.3f}/{c[0.7]:+.3f}/{c[1.8]:+.3f}; contrast {c[0.5] - c[0.7]:+.3f}; "
        f"rho = {', '.join(f'{x:.3f}' for x in t.rho_hw_emu500)}")
mz = np.load(HW / "mock_validation" / "hw_embeddings_mock.npz")
for tag, tt in [("0p5", 0.5), ("0p7", 0.7), ("1p8", 1.8)]:
    key = [k for k in mz.files if k.startswith("hw_") and tag in k]
    if key:
        Em = mz[key[0]]; Ee = np.asarray(EH[("IV", tt, 1234, 8, 1, "Z", 500, (0, 1, 2, 3), 2, 10.0, 66)], float)
        b = float((Em * Ee).sum() / (Ee * Ee).sum())
        for r in rows:
            if r["probe_us"] == tt and r["waveform"].startswith("E1"): r["slope_b"] = b
        say(f"  E1 slope b (no intercept, vs emu500) at {tt} us = {b:.3f}")
pd.DataFrame(rows).to_csv(OUT / "S6_waveform_amendment.csv", index=False)

# ================================================================ S7
say("== S7 sign-definite grid ==")
g = pd.read_csv(EMU / "results" / "signdef_prediction.csv")
g7 = g[["encoding", "a_um", "t_tot_us", "skill_inf", "skill_500_median", "ci_lo", "ci_hi", "seed_min"]]
g7.to_csv(OUT / "S7_signdef_grid.csv", index=False)
say(g7.round(3).to_string(index=False))

(OUT / "supplement_numbers.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("Saved supplement/S*.csv, figS_predictions.png/.pdf, supplement_numbers.txt")

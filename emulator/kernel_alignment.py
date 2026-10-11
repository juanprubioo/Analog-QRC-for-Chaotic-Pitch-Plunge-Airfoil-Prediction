#!/usr/bin/env python3
"""kernel_alignment.py — "Structured versus random": kernel-target alignment and effective
dimension of the QRC-16 embedding and the dimension-matched RFF-16 control.

All quantities are computed on the 44 TRAINING anchors of the base geometry (six seeds), so
they describe the feature maps as the readout sees them, with no access to the test block.

Feature maps (all 16-dimensional except the linear window):
  QRC symmetric      cached Bloqade embeddings of Table 2, t_tot in {1.0, ..., 3.0} us
  QRC sign-definite  cached Bloqade embeddings (results/signdef_bloqade_emb.pkl), t_tot = 1.0 us
  RFF-16             tanh(W_std G + b), exactly as in cell_4_4_v2.py, five projections
  linear window      the 8-dimensional standardized input window (reference)
Each feature is standardized on the training anchors before forming kernels.

Metrics
  centered kernel-target alignment (Cortes, Mohri & Rostamizadeh 2012)
      A(K, L) = <K_c, L_c>_F / (||K_c||_F ||L_c||_F),  K = Phi Phi^T,  L = Y Y^T,
      K_c = H K H, H = I - 11^T/n; Y = training targets (four state increments at the three
      horizons, concatenated), in physical units so that the weighting matches the pooled
      MSE of the preregistered score. A sensitivity variant with standardized targets is
      also reported.
  participation ratio of the feature covariance spectrum, PR = (sum l)^2 / sum l^2
      (1 <= PR <= d; d for isotropic features).
  in-sample R^2 of the targets on the features (ordinary least squares, training anchors).

Run from emulator/:  python3 kernel_alignment.py
Outputs: results/kernel_alignment.csv (per seed), results/kernel_alignment_summary.csv
"""
import io, pickle, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
src = (HERE / "export_totaltime_csv.py").read_text(encoding="utf-8").split("# ---------- verificación contra la Tabla 2")[0]
G = {"__file__": str(HERE / "export_totaltime_csv.py"), "__name__": "kta_ctx"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "export_totaltime_csv.py", "exec"), G)
A, tr, te, H_eval, Xc = G["anchors"], G["tr_idx"], G["te_idx"], G["H_eval"], G["Xc_sel"]
W_by_seed, SEEDS, RFF, cache, ek = G["W_by_seed"], G["seed_list"], G["_RFF"], G["cache"], G["_ek"]
sd_path = HERE / "results" / "signdef_bloqade_emb.pkl"
SD = pickle.load(open(sd_path, "rb")) if sd_path.exists() else {}

Y_all = np.hstack([Xc[A + H] - Xc[A] for H in H_eval])        # (66, 12) physical units
n = len(tr)
Hc = np.eye(n) - 1.0 / n


def kta(Phi, Y):
    K = Hc @ (Phi @ Phi.T) @ Hc
    L = Hc @ (Y @ Y.T) @ Hc
    return float((K * L).sum() / (np.linalg.norm(K) * np.linalg.norm(L)))


def pr(Phi):
    lam = np.clip(np.linalg.eigvalsh(np.cov(Phi, rowvar=False)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def r2(Phi, Y):
    X = np.hstack([np.ones((len(Phi), 1)), Phi])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    return float(1 - (res ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def metrics(F, Y):
    Phi = StandardScaler().fit_transform(F[tr])
    Ytr = Y[tr]
    Ys = StandardScaler().fit_transform(Ytr)
    return dict(dim=F.shape[1], kta=kta(Phi, Ytr), kta_std_targets=kta(Phi, Ys), pr=pr(Phi), r2_train=r2(Phi, Ytr))


rows = []
for s in SEEDS:
    W = W_by_seed[s]
    rows.append(dict(seed=s, features="linear window (8)", **metrics(W, Y_all)))
    sc = StandardScaler().fit(W[tr]); Ws = sc.transform(W)
    rff = [metrics(RFF(16, W.shape[1], np.random.default_rng(100 + r)).transform(Ws), Y_all) for r in range(5)]
    rows.append(dict(seed=s, features="RFF-16 (mean of 5)", **{k: np.mean([m[k] for m in rff]) for k in rff[0]}))
    for tt in [1.0, 1.5, 2.0, 2.5, 3.0]:
        rows.append(dict(seed=s, features=f"QRC-16 symmetric, t_tot={tt}", **metrics(np.asarray(cache[ek(tt, s)], float), Y_all)))
    for a in (10.0, 30.0):
        k = ("sign-definite", a, 1.0, int(s), 500)
        if k in SD:
            rows.append(dict(seed=s, features=f"QRC-16 sign-definite a={a:.0f}, t_tot=1.0", **metrics(np.asarray(SD[k], float), Y_all)))
df = pd.DataFrame(rows)
df.to_csv(HERE / "results" / "kernel_alignment.csv", index=False)

# pooled test skill of each QRC configuration (from Table 2 / signdef check) for context
skill = {f"QRC-16 symmetric, t_tot={tt}": v for tt, v in zip([1.0, 1.5, 2.0, 2.5, 3.0], [0.698, -0.197, 0.658, -0.314, 0.131])}
skill.update({"QRC-16 sign-definite a=10, t_tot=1.0": 0.768, "QRC-16 sign-definite a=30, t_tot=1.0": 0.739,
              "RFF-16 (mean of 5)": 0.580, "linear window (8)": 0.714})
summ = df.groupby("features", sort=False).agg(**{f"{m}_{s}": (m, f) for m in ("kta", "kta_std_targets", "pr", "r2_train")
                                               for s, f in (("mean", "mean"), ("sd", "std"))}).reset_index()
summ["test_skill"] = summ.features.map(skill)
# paired QRC(1.0) - RFF differences across seeds
q = df[df.features == "QRC-16 symmetric, t_tot=1.0"].set_index("seed")
rf = df[df.features == "RFF-16 (mean of 5)"].set_index("seed")
for m in ("kta", "pr", "r2_train"):
    d = q[m] - rf[m]
    print(f"QRC(1.0) - RFF, {m}: mean {d.mean():+.3f}, min {d.min():+.3f}, max {d.max():+.3f}, positive in {int((d > 0).sum())}/6 seeds")
sym = summ[summ.features.str.startswith("QRC-16 symmetric")]
r_kta = np.corrcoef(sym.kta_mean, sym.test_skill)[0, 1]
print(f"Across the five symmetric total times: corr(KTA, test skill) = {r_kta:+.3f}")
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(summ[["features", "kta_mean", "kta_sd", "kta_std_targets_mean", "pr_mean", "pr_sd", "r2_train_mean", "test_skill"]].to_string(index=False))
summ.to_csv(HERE / "results" / "kernel_alignment_summary.csv", index=False)
print("Saved results/kernel_alignment.csv, results/kernel_alignment_summary.csv")

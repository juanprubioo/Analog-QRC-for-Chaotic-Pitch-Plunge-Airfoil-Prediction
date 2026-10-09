#!/usr/bin/env python3
"""signdef_shot_passrate.py — Shot-noise sensitivity of the sign-definite result.

Reuses the exact-evolution machinery of signdef_prediction.py (and its probability cache
results/signdef_probs.npz) to draw K independent 500-shot realizations of the
sign-definite configurations at t_tot = 1.0 us (a = 10 and 30 um), and reports the
distribution of pooled skill and how often each preregistered acceptance criterion passes:
    pooled   : lower bound of the 95% bootstrap CI > best classical pooled skill (0.714)
    per-seed : worst QRC seed > best classical seed (0.754)
It also checks that the Bloqade embeddings (results/signdef_bloqade_emb.pkl) equal the exact
embeddings up to shot noise (normalized residuals ~ N(0,1)).

Run from emulator/:  python3 signdef_shot_passrate.py [K=50]
Output: results/signdef_shot_passrate.csv
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

K = int(sys.argv[1]) if len(sys.argv) > 1 else 50
HERE = Path(__file__).resolve().parent
_src = (HERE / "signdef_prediction.py").read_text(encoding="utf-8")
_src = _src.split("cls, e_lin, ep_ref, cls_best_seed = classical()")[0]
_ns = {"__file__": str(HERE / "signdef_prediction.py"), "__name__": "signdef_passrate"}
_argv, sys.argv = sys.argv, ["signdef_prediction.py", "1"]
exec(compile(_src, "signdef_prediction.py", "exec"), _ns)
sys.argv = _argv
for k in ("classical", "qrc_errors", "embedding", "boot", "S", "P", "SEEDS"):
    globals()[k] = _ns[k]

cls, _, _, cls_best_seed = classical()
best, best_seed = max(cls.values()), cls_best_seed.max()
bq_path = HERE / "results" / "signdef_bloqade_emb.pkl"
bq = pickle.load(open(bq_path, "rb")) if bq_path.exists() else {}

rows = []
for enc, a in [("sign-definite", 10.0), ("sign-definite", 30.0)]:
    Pk = {s: P[f"{enc}|{a}|{s}"] for s in SEEDS}
    sk, lo, sm = [], [], []
    for r in range(K):
        rng = np.random.default_rng(777 + r)
        eq, ep, ps = qrc_errors({s: embedding(Pk[s], 1.0, rng) for s in SEEDS})
        sk.append(S(eq, ep)); lo.append(boot(eq, ep, B=1000, seed=r)[0]); sm.append(ps.min())
    sk, lo, sm = map(np.array, (sk, lo, sm))
    row = dict(encoding=enc, a_um=a, K=K, skill_mean=sk.mean(), skill_sd=sk.std(ddof=1),
               skill_min=sk.min(), skill_max=sk.max(), pass_pooled=np.mean(lo > best),
               pass_per_seed=np.mean(sm > best_seed), pass_both=np.mean((lo > best) & (sm > best_seed)))
    key0 = (enc, a, 1.0, int(SEEDS[0]), 500)
    if key0 in bq:
        Eb = np.vstack([bq[(enc, a, 1.0, int(s), 500)] for s in SEEDS])
        Ex = np.vstack([embedding(Pk[s], 1.0) for s in SEEDS])
        z = (Eb - Ex) / np.sqrt(np.clip(1 - Ex ** 2, 1e-6, None) / 500)
        row.update(bloqade_corr=np.corrcoef(Eb.ravel(), Ex.ravel())[0, 1], bloqade_z_mean=z.mean(), bloqade_z_sd=z.std())
    rows.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

pd.DataFrame(rows).to_csv(HERE / "results" / "signdef_shot_passrate.csv", index=False)
print(f"best classical pooled = {best:.3f}, best classical seed = {best_seed:.3f}")
print("Saved results/signdef_shot_passrate.csv")

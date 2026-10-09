"""paired_diff.py — Paired skill difference S_lin - S_QRC at the best operating point
(t_tot = 1.0 us, 8 atoms, Z, base geometry, six seeds) and the per-horizon breakdown
(Table "horizon" of the manuscript), both from the SAME cached Bloqade embeddings that
produce Table 2 (results/qrc_totaltime_emb_IV.pkl).

Skill aggregation (identical to cell_4_4_v2.py): squared errors are averaged over the
predicted state components for each test point, then pooled over test anchors, horizons
and seeds; S = 1 - mean(MSE_model)/mean(MSE_persistence). Because model and persistence
share the denominator, the paired point difference equals the difference of the two
pooled skills exactly. CI: paired bootstrap over the pooled test points (3000 draws).

Run from emulator/:  python3 paired_diff.py   -> results/paired_diff.csv"""
import sys, pickle, importlib.util
from pathlib import Path
import numpy as np
HERE = Path.cwd()
src = (HERE/"export_totaltime_csv.py").read_text()
# reuse the exporter context but stop before its verification/exit
src = src.split("# ---------- verificación contra la Tabla 2")[0]
src = src.replace('src = (HERE / "cell_4_4_v2.py")', 'src = (HERE / "cell_4_4_v2.py")')
g = {"__file__": str(HERE/"export_totaltime_csv.py"), "__name__": "x"}
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "export", "exec"), g)
anchors, tr, te, H_eval, Xc_sel = g["anchors"], g["tr_idx"], g["te_idx"], g["H_eval"], g["Xc_sel"]
W_by_seed, cache, _ek, seeds = g["W_by_seed"], g["cache"], g["_ek"], g["seed_list"]
ridge, ngrc = g["_ridgecv"], g["build_ngrc_model"]
TT = 1.0
eq, el, ep, sid, hid = [], [], [], [], []
for s in seeds:
    W = W_by_seed[s]; E = cache[_ek(TT, s)]
    for h, H in enumerate(H_eval):
        Y = Xc_sel[anchors+H]-Xc_sel[anchors]
        eq.append(((ridge().fit(E[tr],Y[tr]).predict(E[te])-Y[te])**2).mean(1))
        el.append(((ngrc(degree=1).fit(W[tr],Y[tr]).predict(W[te])-Y[te])**2).mean(1))
        ep.append((Y[te]**2).mean(1)); sid += [s]*len(te); hid += [h]*len(te)
eq, el, ep = map(np.concatenate, (eq, el, ep)); sid = np.array(sid); hid = np.array(hid)
S = lambda e, p: 1 - e.mean()/p.mean()
d_point = S(el, ep) - S(eq, ep)
print(f"n test points pooled = {len(ep)}")
print(f"S_QRC = {S(eq,ep):.4f}  S_lin = {S(el,ep):.4f}  point diff S_lin-S_QRC = {d_point:+.4f}")
rng = np.random.default_rng(0); B = 3000; n = len(ep); bd = np.empty(B)
for b in range(B):
    i = rng.integers(0, n, n); bd[b] = S(el[i], ep[i]) - S(eq[i], ep[i])
print(f"paired bootstrap (pooled ratio): mean={bd.mean():+.4f} median={np.median(bd):+.4f} "
      f"95% CI [{np.percentile(bd,2.5):+.4f}, {np.percentile(bd,97.5):+.4f}]")
# alternative aggregations that could have produced 0.021
ps = [S(el[sid==s],ep[sid==s]) - S(eq[sid==s],ep[sid==s]) for s in seeds]
print(f"mean of per-seed differences   = {np.mean(ps):+.4f}  (per seed: {np.round(ps,3)})")
ph = [S(el[(sid==s)&(hid==h)],ep[(sid==s)&(hid==h)]) - S(eq[(sid==s)&(hid==h)],ep[(sid==s)&(hid==h)]) for s in seeds for h in range(3)]
print(f"mean of per-seed-horizon diffs = {np.mean(ph):+.4f}")
pp = 1 - eq/ep; pl = 1 - el/ep
print(f"mean of per-point skill diffs  = {np.mean(pl-pp):+.4f}")
ph2 = [S(el[hid==h],ep[hid==h]) - S(eq[hid==h],ep[hid==h]) for h in range(3)]
print(f"mean of per-horizon diffs      = {np.mean(ph2):+.4f}  (per H: {np.round(ph2,3)})")
import pandas as pd
rows=[dict(scope="pooled", S_qrc=S(eq,ep), S_lin=S(el,ep), diff=d_point,
           diff_lo=np.percentile(bd,2.5), diff_hi=np.percentile(bd,97.5))]
print("--- per horizon (95% bootstrap CI)")
for h in range(3):
    m = hid==h; e0, e1, p = eq[m], el[m], ep[m]; nn=len(p); b0=[]; b1=[]
    for _ in range(B):
        i=rng.integers(0,nn,nn); b0.append(S(e0[i],p[i])); b1.append(S(e1[i],p[i]))
    rows.append(dict(scope=f"H={h+1}tau_c", S_qrc=S(e0,p), S_qrc_lo=np.percentile(b0,2.5), S_qrc_hi=np.percentile(b0,97.5),
                     S_lin=S(e1,p), S_lin_lo=np.percentile(b1,2.5), S_lin_hi=np.percentile(b1,97.5), diff=S(e1,p)-S(e0,p)))
    print(f"H={h+1}tau_c: QRC {S(e0,p):.3f} [{np.percentile(b0,2.5):.3f},{np.percentile(b0,97.5):.3f}]  "
          f"lin {S(e1,p):.3f} [{np.percentile(b1,2.5):.3f},{np.percentile(b1,97.5):.3f}]")
pd.DataFrame(rows).to_csv(HERE/"results"/"paired_diff.csv", index=False)
print("Saved results/paired_diff.csv")

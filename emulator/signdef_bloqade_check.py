#!/usr/bin/env python3
"""signdef_bloqade_check.py — Bloqade confirmation of the sign-definite operating points.

signdef_prediction.py uses exact 8-atom evolution with multinomial shot sampling. This
script re-emulates the configurations quoted in the manuscript with Bloqade (500 shots),
so that every number in the text comes from the same emulator as Table 2:
    sign-definite, a = 10 um, t_tot = 1.0 us
    sign-definite, a = 30 um, t_tot = 1.0 us
    symmetric,     a = 30 um, t_tot = 1.0 us
(symmetric, a = 10 um, t_tot = 1.0 us is the Table 2 cache and is not re-run).

The program is build_local_task() from airfoil_qrc_v5.py with the global detuning offset
changed from encoding_scale/2 = 4.5 to -0.5 rad/us for the sign-definite encoding.
Embeddings are cached in results/signdef_bloqade_emb.pkl (resumable).

Run from emulator/:  python3 signdef_bloqade_check.py      (~3200 Bloqade runs)
"""
import io, contextlib, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
src = (HERE / "export_totaltime_csv.py").read_text(encoding="utf-8")
src = src.split("# ---------- verificación contra la Tabla 2")[0]
G = {"__file__": str(HERE / "export_totaltime_csv.py"), "__name__": "signdef_bq"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "export_totaltime_csv.py", "exec"), G)
q5 = G["q5"]
A, tr, te, H_eval, Xc = G["anchors"], G["tr_idx"], G["te_idx"], G["H_eval"], G["Xc_sel"]
W_by_seed, SEEDS, ridge, ngrc = G["W_by_seed"], G["seed_list"], G["_ridgecv"], G["build_ngrc_model"]
cache_tbl2, ek = G["cache"], G["_ek"]
NSHOTS, TT = 500, 1.0
CONFIGS = [("sign-definite", 10.0), ("sign-definite", 30.0), ("symmetric", 30.0)]
OFFSET = {"symmetric": 4.5, "sign-definite": -0.5}

_, Chain = q5.get_bloqade_objects()


def run_bloqade(x, offset, a):
    dt = TT / 2
    prog = (Chain(8, lattice_spacing=a).rydberg.rabi.amplitude.uniform
            .constant(duration="run_time", value=6.283)
            .detuning.uniform.constant(duration="run_time", value=offset)
            .scale(list(x)).constant(duration="run_time", value=-9.0))
    rep = (prog.batch_assign(run_time=np.arange(1, 3) * dt)
           .bloqade.python().run(shots=NSHOTS, rtol=1e-8, atol=1e-8).report())
    cfg = q5.QRCConfig(atom_number=8, readouts="Z", time_steps=2)
    return q5.process_local_report(cfg, rep)


cache_path = HERE / "results" / "signdef_bloqade_emb.pkl"
cache = pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}
for enc, a in CONFIGS:
    for s in SEEDS:
        key = (enc, a, TT, int(s), NSHOTS)
        if key in cache:
            continue
        W = W_by_seed[s]
        Wsc = np.clip(MinMaxScaler((0.0, 1.0)).fit(W[tr]).transform(W), 0.0, 1.0)
        cache[key] = np.array([run_bloqade(Wsc[k], OFFSET[enc], a) for k in range(len(A))], np.float32)
        pickle.dump(cache, open(cache_path, "wb"))
        print(f"  done {enc} a={a} seed={s}", flush=True)

S = lambda e, p: 1 - e.mean() / p.mean()


def errors(E_by_seed):
    eq, ep, ps = [], [], []
    for s in SEEDS:
        E = E_by_seed[s]; se, sp = [], []
        for H in H_eval:
            Y = Xc[A + H] - Xc[A]
            se.append(((ridge().fit(E[tr], Y[tr]).predict(E[te]) - Y[te]) ** 2).mean(1))
            sp.append((Y[te] ** 2).mean(1))
        se, sp = np.concatenate(se), np.concatenate(sp)
        eq.append(se); ep.append(sp); ps.append(S(se, sp))
    return np.concatenate(eq), np.concatenate(ep), np.array(ps)


el = []
for s in SEEDS:
    W = W_by_seed[s]
    for H in H_eval:
        Y = Xc[A + H] - Xc[A]
        el.append(((ngrc(degree=1).fit(W[tr], Y[tr]).predict(W[te]) - Y[te]) ** 2).mean(1))
el = np.concatenate(el)


def boot(e, p, ref=None, B=3000):
    rng = np.random.default_rng(0); n = len(p); out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        out[b] = S(e[i], p[i]) - (S(ref[i], p[i]) if ref is not None else 0.0)
    return np.percentile(out, [2.5, 97.5])


rows = []
sym10 = errors({s: np.asarray(cache_tbl2[ek(TT, s)], float) for s in SEEDS})
for (enc, a), (eq, ep, ps) in [(("symmetric", 10.0), sym10)] + [
        ((enc, a), errors({s: cache[(enc, a, TT, int(s), NSHOTS)] for s in SEEDS})) for enc, a in CONFIGS]:
    lo, hi = boot(eq, ep); dlo, dhi = boot(eq, ep, el)
    row = dict(encoding=enc, a_um=a, t_tot_us=TT, skill=S(eq, ep), ci_lo=lo, ci_hi=hi,
               diff_vs_lin=S(eq, ep) - S(el, ep), diff_lo=dlo, diff_hi=dhi,
               seed_min=ps.min(), seed_max=ps.max())
    if enc == "sign-definite":
        glo, ghi = boot(eq, ep, sym10[0])
        row.update(gain_vs_sym10=S(eq, ep) - S(sym10[0], sym10[1]), gain_lo=glo, gain_hi=ghi)
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda v: f"{v:+.3f}"); pd.set_option("display.width", 250)
print(df.to_string(index=False))
df.to_csv(HERE / "results" / "signdef_bloqade_check.csv", index=False)
print("Saved results/signdef_bloqade_check.csv")

# Emulator Pipeline

Local, exact-physics reproduction of the QRC results using the Bloqade
analog emulator (`bloqade.python()` — free, runs on your machine, no cloud
access). Nothing here touches real hardware or `quera.mock`.

## 1. Setup

This folder's scripts import `airfoil_qrc_v5.py` and `airfoil_simulator.py`
via a path-relative dynamic import, so every file they need must sit in the
**same directory**. Copy the shared files from the repository root first:

```bash
cd emulator
cp ../airfoil_simulator.py ../airfoil_qrc_v5.py ../airfoil_state_estimation.py ../verify.py ../data_cache.npz .
```

Dependencies per script:

| Script | Needs in this directory | Needs installed |
|---|---|---|
| `make_data_cache.py` | `airfoil_qrc_v5.py`, `airfoil_simulator.py` | numpy, scipy |
| `verify.py` | the two above (+ `data_cache.npz`, optional — speeds it up) | numpy, scipy, scikit-learn |
| `run_generic.py` | the two above + `data_cache.npz` (required) | + bloqade-analog |
| `analyze_generic.py` | the two above + `data_cache.npz` + `emb_cache.pkl` | numpy, scipy, scikit-learn, pandas |

```bash
pip install bloqade-analog numpy scipy scikit-learn pandas
python3 -c "import bloqade.analog; import sklearn; print('OK')"
```

`airfoil_state_estimation.py` isn't imported by these scripts directly, but is
used by `airfoil_aquila_verification_local_ld_native_v15.ipynb`.

## 2. First-time order of operations

### 2.1 — `data_cache.npz`

Already provided at the repository root (copy it in, see Setup above). If you
need to regenerate it from scratch:

```bash
python3 make_data_cache.py     # ~2-3 min
```

This simulates the clean Case IV trajectory (`XC`, 70001×6) and, for each seed
in `[1234, 1335, 1436, 1537, 1638, 1739]`, generates 40%-noise Gaussian
observations smoothed with a Savitzky-Golay filter (window 31, order 3).
Everything is stored in `data_cache.npz` under keys `XC`, `dt`, `dn_<seed>`.
The cache exists so every experiment reads exactly the same input data instead
of re-simulating it (~1 min) on every run. To add seeds, extend `SEEDS` at the
top of the script and rerun (deterministic per seed).

### 2.2 — `verify.py`

```bash
python3 verify.py
```

Expected output:

```
tau_c = 10.30 tu | anchors = 66 (train 44 / test 22) | tol = 0.02
  PASS  NG-RC deg2 alpha=1e-4 (broken)     obtained -8.912  expected -8.912
  PASS  NG-RC deg2 RidgeCV (fair)          obtained +0.405  expected +0.405
  PASS  NG-RC deg1 RidgeCV (linear)        obtained +0.706  expected +0.706
VERIFY OK
```

Run this at the start of every results session, and after any edit to
`airfoil_qrc_v5.py` or the protocol. Exits with code 1 on failure, so it can
gate CI: `python3 verify.py || exit 1`. If it fails, do not proceed — a
failure on the τ_c ≈ 10.3 assertion points to the simulator/burn-in; a
failure on the skill numbers points to the evaluation protocol or the model.

### 2.3 — `emb_cache.pkl`

Provided in this folder (~4600 emulations, ~1.5 h of compute, 69 chunks). If
starting from scratch, `run_generic.py` creates it empty and fills it in.

Inventory of the delivered cache — format
`(atoms, time_steps, readout, spacing µm, n_anchors)`:

```
(8,  2, Z,  8.5, 66)   3 chunks   — spacing probe V>Ω
(8,  2, Z, 10.0, 66)  15 chunks   — base sweep tt={1.0,1.5,2.0,2.5,3.0} × 3 seeds
(8,  2, Z, 12.0, 66)   3 chunks   — spacing probe V<Ω
(8,  2, ZZ,10.0,200)   3 chunks   — extended anchors, tt=2.0
(8,  3, Z, 10.0, 66)  15 chunks   — ts=3, full tt sweep
(8,  5, Z, 10.0, 66)  12 chunks   — ts=5, tt={1.0,1.5,2.0} (6 seeds at tt=1.5)
(12, 2, Z, 10.0, 66)  15 chunks   — 12 atoms, full tt sweep
(16, 2, Z, 10.0, 66)   3 chunks   — 16 atoms, tt=2.0
```

A "chunk" = one (tt, seed) combination = 66 (or 200) embeddings.

## 3. Using `run_generic.py`

```bash
python3 run_generic.py '<config_json>' <time_budget_s>
```

The time budget is optional (default: unlimited). When it runs out, the
script saves progress and exits cleanly with `TIME BUDGET REACHED` or
`PARTIAL saved k/66`; the next invocation with the **same JSON** resumes
exactly where it left off.

Config fields:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `state_indices` | list[int] | yes | Airfoil states used as input (0=α, 1=α̇, 2=ξ, 3=ξ̇, 4=w1, 5=w2) |
| `window` | int | yes | Temporal window depth (# past instants, stride 5 tu) |
| `time_steps` | int | yes | # probe times, at `k·(total_time/time_steps)`, k=1..ts |
| `readout` | "Z" \| "ZZ" | yes | Measured observables. Dim: Z → atoms×ts; ZZ → (atoms + C(atoms,2))×ts |
| `tts` | list[float] | yes | `total_time` values in µs to compute |
| `seeds` | list[int] | yes | Noise seeds (must exist as `dn_<seed>` in `data_cache.npz`) |
| `nshots` | int | no (500) | Shots per emulation |
| `lattice_spacing` | float | no (10.0) | Chain spacing in µm (sets the van der Waals V) |
| `n_anchor_extra` | int | no (0) | Extra anchors appended → extends train, **shifts the test set** (see §5) |

The number of atoms is not declared directly: it's `window × len(state_indices)`.

Fixed constants inside the script (edit there if the protocol changes):
Ω=6.283 rad/µs, encoding scale 9.0, burn-in 500 tu, window stride 5 tu,
horizons {1,2,3}×τ_c, base train/test 44/22, "local" encoding.

Example (base sweep — a no-op if the delivered cache is present, which is
the proof the cache is working):

```bash
python3 run_generic.py '{"state_indices":[0,1,2,3],"window":2,"time_steps":2,"readout":"Z","tts":[1.0,1.5,2.0,2.5,3.0],"seeds":[1234,1335,1436]}' 400
```

## 4. Using `analyze_generic.py`

```bash
python3 analyze_generic.py '<config_json>'
```

Same fields as `run_generic.py` (must match exactly, cache keys are
field-sensitive), plus: `readout_mode` ("cv" default, or "fixed"),
`train_sizes` (learning-curve evaluation), `z_only_from_zz` (extract Z from a
cached ZZ embedding without re-emulating).

Output columns: `skill_qrc`/`ci_lo`/`ci_hi` (skill vs. persistence, pooled,
bootstrap 95% CI), `skill_ngrc`/`skill_ngrc_lin` (classical NG-RC baselines
on the same window), `skill_ngrc_best`/`beats_classical`, `qrc_seed_min` /
`cls_seed_max` / `robust` (the per-seed criterion — worst QRC seed must beat
the best classical seed; **this is the filter that matters**), `ok_dim`
(`n_train > dim(embedding)` — if False, the row is invalid).

## 5. Cache-invalidation rules

| Change | `data_cache`? | `emb_cache`? |
|---|---|---|
| New tt, seed (existing in data), atoms, ts, readout, spacing | nothing | added automatically (new key) |
| New noise seed | regenerate (add to `SEEDS`) | nothing |
| Savitzky-Golay window, noise level/type, case | **regenerate** | **invalidate all** |
| Burn-in, window stride, horizons, base train/test | nothing | **invalidate all** (anchors change; the key doesn't distinguish) |
| `n_anchor_extra` | nothing | new key, but **the test set shifts** — skills are not comparable to the 66-anchor baseline |

"Invalidate all" means rename `emb_cache.pkl` to `emb_cache_v1.pkl` (don't
delete — keep it for traceability) and let it regenerate.

## 6. Contents of this folder

- `make_data_cache.py`, `run_generic.py`, `analyze_generic.py` — pipeline scripts (see above).
- `cell_4_4_v2.py` — total-time drill-down, frozen protocol, extended classical baselines.
- `toy_interacting_model.py` — appendix: exact unitary dynamics of the 8-atom chain, used as a physical cross-check of the probe-node predictions.
- `airfoil_aquila_verification_local_ld_native_v15.ipynb` — local verification notebook (pre-hardware).
- `emb_cache.pkl` — delivered embedding cache (emu-500).
- `figures/` — one script + its output figure/table per manuscript figure: `fig_parity.*`, `fig_pipeline.*`, `fig_probes.*`, `fig_scaling.*`, `lcurve_fixedtest.*`.
- `results/` — consolidated tables and auxiliary caches (`resultados_consolidados.csv`, `pair_skill_matrix_full.npy`, `qrc_horizon_ms_emb_IV.pkl`, `qrc_totaltime_emb_IV.pkl`).
- `docs/GUIA_IMPLEMENTACION_es.md` — original (Spanish) implementation notes, kept for reference.

## 7. Manuscript convention

Every number that goes into the paper should be regenerable from: (1) the
repo commit, (2) an exact config JSON, (3) a backed-up `emb_cache.pkl`.

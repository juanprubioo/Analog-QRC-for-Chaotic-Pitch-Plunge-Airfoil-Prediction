# Real-Hardware Run — AWS Braket Aquila

The frozen, preregistered protocol that was submitted to AWS Braket's
neutral-atom QPU **Aquila** (198 tasks, single run, design frozen before
submission), plus the mock-hardware validation that preceded it and the
analysis that compares hardware, 100-shot emulation, and 500-shot emulation.

## 1. Setup

```bash
pip install amazon-braket-sdk numpy scipy scikit-learn pandas matplotlib
```

Copy the shared core files from the repository root into this folder first
(the scripts import them via path-relative dynamic import, so they must live
alongside `hw_common.py`):

```bash
cd real_run
cp ../airfoil_simulator.py ../airfoil_qrc_v5.py ../verify.py ../data_cache.npz .
```

## 2. Protocol status (as run)

- **Step 0** — `verify.py`: 3/3 PASS; cache verified (emu-500 chunks, 3 probes, seed 1234).
- **Step 1** — emu-100 reference computed and cached (`nshots=100` keys) — `step1_baseline.csv`.
- **Step 2** — pipeline fixes: C7 (Bloqade bit convention, ground=1 — v15 had flipped the sign of ρ), `"aquila"` prefix, per-task try/except + log, ARNs persisted to disk immediately, safe resumption, fail-fast (abort after 3 initial failures).
- **Step 3** — mock end-to-end, 198/198, using Braket's local AHS simulator (same program and parsing as the real path): table + figure generated; primary criterion **MET**.
- **Amendment E1** (confirmed) — minimum legal ramp 0.05 µs + area compensation. Programmed times {0.55, 0.75, 1.85} µs → effective {0.5, 0.70, 1.8} µs. Justification: ρ(mock↔emu500) = 0.971/0.953/0.944 vs. a sampling-only baseline of 0.968/0.952/0.938. The v15 ramp (`min(0.2, tt/4)`) destroyed the node (contrast −0.011). See `hw_common.py` and `exp_ramp_scan.py` / `exp_ramp/`.

## 3. Design (frozen, preregistered — see `hw_common.py`)

8 atoms, 1D chain, spacing a = 10 µm, local encoding
Δᵢ = 4.5 − 9·xᵢ rad/µs, Ω = 6.283 rad/µs, readout Z, `time_steps=1`,
probes {0.5, 0.70, 1.8} µs, 66 anchors (seed 1234, denoised windows from
`data_cache.npz`, `window=2`, MinMax scaler fit only on the 44 train
anchors), 100 shots/task. Ramp-plateau-ramp waveform required by Aquila
(detuning and Ω must start/end at 0); see Amendment E1 above for the exact
ramp/timing compensation.

## 4. Run sequence (Python ≥ 3.10)

```bash
python3 verify.py                          # must give 3 PASS
python3 submit_aquila.py budget            # review the cost estimate, check the current AWS rate
python3 submit_aquila.py real "AUTORIZO EL ENVÍO A AQUILA"   # exact phrase required, AWS creds must be configured
python3 fetch_aquila.py                    # poll/download; raw shots backed up immediately
python3 analyze_hardware.py real           # table + figure (Sec. V.E)
```

`submit_aquila.py` modes:

```
python3 submit_aquila.py mock    # Braket's local AHS simulator (free): same program,
                                  # same result schema, same parsing as Aquila.
python3 submit_aquila.py budget  # prints the cost estimate only; sends nothing.
python3 submit_aquila.py real "AUTORIZO EL ENVÍO A AQUILA"
                                  # real submission. Requires the EXACT phrase above
                                  # and AWS credentials already configured in the environment.
```

Guarantees: try/except per task with logging to `aquila_hw_submit_real.log`
(a failed task doesn't abort the batch); the ARN and metadata (probe, anchor,
config) are persisted immediately to `aquila_hw_tasks/<tag>.json`; safe
resumption — tasks with an existing `submitted`/`done` manifest are not
resent; `submit_aquila.py` never reads, requests, or prints credentials.

**Resuming**: if anything is interrupted, rerun the same command — anything
already submitted is not resent (manifests in `aquila_hw_tasks/`). The ARNs
live in those JSON files — do not delete them.

**Contingencies**: a rejected/failed task → delete only its `'failed'` `.json`
manifest and relaunch; partial data loss → `analyze_hardware.py` computes
with the anchors available and reports the effective n.

## 5. Aquila availability

Aquila serves tasks in scheduled availability windows (see the Braket
console / Braket Direct). All 198 tasks queue up; `fetch_aquila.py` can be
left running or relaunched as needed.

## 6. Cost

198 tasks × ($0.30/task + $0.01/shot × 100) = $257.40 (nominal). Verify the
current rate on the AWS console before authorizing.

## 7. Contents of this folder

- `submit_aquila.py`, `fetch_aquila.py`, `hw_common.py`, `analyze_hardware.py`, `exp_ramp_scan.py`, `step1_emu100_baseline.py` — pipeline scripts (see above).
- `aquila_hw_tasks/` — 198 task manifests (ARNs + metadata) from the real run.
- `aquila_hw_results/` — 198 raw-shot backups (`*_raw.npz`) from the real run.
- `exp_ramp/` — free, non-frozen ramp-scan check (`exp_ramp_scan.py` output) supporting Amendment E1.
- `emb_cache_emu100.pkl` — emu-100 embedding cache used as the step-1 baseline (distinct from `emulator/emb_cache.pkl`, which is emu-500).
- `hw_embeddings_real.npz`, `hw_table_real.csv`, `hw_skill_vs_probe_real.png`/`.pdf` — outputs of `analyze_hardware.py real`.
- `aquila_hw_submit_real.log` — submission/fetch log of the actual hardware run.
- `step1_baseline.csv` — output of `step1_emu100_baseline.py`.
- `lanzador.ipynb` — notebook used to drive/launch the submission workflow interactively.
- `mock_validation/` — Step-3 mock end-to-end run: `hw_embeddings_mock*.npz`, `hw_table_mock*.csv`, `hw_skill_vs_probe_mock.png`, `aquila_hw_submit_mock.log`.
- `docs/README_CORRIDA_REAL_es.md` — original (Spanish) working notes, kept for reference.

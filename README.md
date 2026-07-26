# Airfoil State Estimation via Quantum Reservoir Computing on Neutral-Atom Hardware

This repository contains the code and data behind a study that uses a quantum
reservoir computing (QRC) approach to estimate the state of a conceptual
aeroelastic airfoil model. Embeddings are generated in two ways: locally, with
an exact physics emulator (Bloqade), and on real neutral-atom quantum hardware
(AWS Braket **Aquila**). The repository is split into two self-contained
pipelines that share a common physics/model core.

## Repository layout

```
.
├── airfoil_simulator.py          # Aeroelastic airfoil model (shared)
├── airfoil_qrc_v5.py              # QRC embedding pipeline (shared, imports airfoil_simulator.py)
├── airfoil_state_estimation.py    # State-estimation / smoothing helpers (shared)
├── verify.py                      # Smoke test for the classical reference numbers (shared)
├── data_cache.npz                 # Cached clean/noisy airfoil trajectories used by both pipelines
├── emulator/                       # Local emulator pipeline (Bloqade, no hardware)
└── real_run/                      # Real-hardware pipeline (AWS Braket Aquila)
```

### Shared core files (repository root)

`airfoil_simulator.py`, `airfoil_qrc_v5.py`, `airfoil_state_estimation.py`,
`verify.py`, and `data_cache.npz` are used by **both** `emulator/` and
`real_run/`. They are kept at the repository root, on purpose, instead of
being duplicated in each subfolder, to keep a single source of truth for the
physics model and the evaluation protocol.

`airfoil_qrc_v5.py` and the scripts inside `emulator/` and `real_run/` load
`airfoil_simulator.py` (and each other) with a **path-relative dynamic
import**, not a Python package import. That means the interpreter looks for
these files in its *current working directory*, not just anywhere on
`sys.path`. Before running anything inside `emulator/` or `real_run/`,
copy (or symlink) the shared files into that folder first:

```bash
cd emulator   # or: cd real_run
cp ../airfoil_simulator.py ../airfoil_qrc_v5.py ../airfoil_state_estimation.py ../verify.py ../data_cache.npz .
```

See `emulator/README.md` and `real_run/README.md` for the exact set each
pipeline needs and the full run instructions.

## The two pipelines

- **`emulator/`** — reproduces every result using the local, exact Bloqade
  emulator (`bloqade.python()`, free, no cloud access). This is where the
  caches, figures, and tables for the manuscript are generated and verified.
- **`real_run/`** — the frozen, preregistered protocol that was actually
  submitted to AWS Braket Aquila (198 tasks), plus the mock-hardware
  validation run and the analysis that produced the hardware-vs-emulator
  comparison (Sec. V.E of the paper).

## Citing / provenance

Each pipeline folder documents how to regenerate every number and figure from
the manuscript, so results can be traced back to a specific config and cache
file. See the `docs/` subfolders for the original (Spanish) working notes.

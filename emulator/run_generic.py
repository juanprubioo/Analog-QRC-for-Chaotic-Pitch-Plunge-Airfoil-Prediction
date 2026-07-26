"""Runner genérico: computa embeddings QRC para configuraciones arbitrarias y las cachea.
Uso: python3 run_generic.py <config_json> <time_budget_s>
config: {"atoms":?, "state_indices":[..], "window":?, "time_steps":?, "readout":"Z|ZZ",
         "tts":[..], "seeds":[..], "n_anchor_extra":0, "lattice_spacing":10.0, "nshots":500}
"""
import warnings, pickle, time, os, sys, json
warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import importlib.util

def dyn_import(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

qrc5 = dyn_import("airfoil_qrc_v5_local", "airfoil_qrc_v5.py")
QRCConfig = qrc5.QRCConfig
load_or_generate_dataset = qrc5.load_or_generate_dataset
emulate_qrc_embeddings = qrc5.emulate_qrc_embeddings

CASE = "IV"; SEED = 1234
NOISE_LEVEL = 0.40; NOISE_KIND = "gaussian"; STUDENT_DF = 5
RABI = 6.283; ENC = 9.0
TT_WIN_STRIDE_TU = 5.0; TT_BURN_IN_TU = 500.0
TT_EVAL_FRACS = [1.0, 2.0, 3.0]
BASE_N_TRAIN = 44; BASE_N_TEST = 22

cfg = json.loads(sys.argv[1])
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9

state_indices = cfg["state_indices"]; window = cfg["window"]
atoms = window * len(state_indices)
ts = cfg["time_steps"]; readout = cfg["readout"]
nshots = cfg.get("nshots", 500)
spacing_um = cfg.get("lattice_spacing", 10.0)
n_extra = cfg.get("n_anchor_extra", 0)

def _char_time_tu(x, dt, max_lag_tu=150.0):
    x = np.asarray(x, float) - np.mean(x); n = len(x); ml = min(n - 1, int(round(max_lag_tu / dt)))
    ac = np.correlate(x, x, "full")[n - 1: n - 1 + ml]
    if ac[0] <= 0: return None
    ac = ac / ac[0]; b = np.where(ac < 1.0 / np.e)[0]
    return float(b[0] * dt) if len(b) else None

_data = np.load("data_cache.npz")
XC = _data["XC"]
dt = float(_data["dt"])
i0 = int(round(TT_BURN_IN_TU / dt))
char_tu = _char_time_tu(XC[i0:i0 + min(int(round(600/dt)), len(XC)-i0), 0], dt) or 10.0
spacing_raw = max(1, int(round(char_tu / dt)))
win_stride_raw = max(1, int(round(TT_WIN_STRIDE_TU / dt)))
H_eval = sorted(set(max(1, int(round(fr * char_tu / dt))) for fr in TT_EVAL_FRACS))
H_max = max(H_eval)

n0 = BASE_N_TRAIN + BASE_N_TEST + n_extra
base = i0 + (window - 1) * win_stride_raw
anchors = base + np.arange(n0) * spacing_raw
keep = (anchors + H_max < len(XC)) & (anchors - (window - 1) * win_stride_raw >= 0)
anchors = anchors[keep]

# tren/test: los EXTRA se insertan al inicio del train para que el test sea el MISMO bloque final
n_test = BASE_N_TEST

_dn = {}
def _denoised(seed):
    return _data[f"dn_{seed}"]

def _W(seed):
    Xin = _denoised(seed)[:, state_indices]
    return np.asarray([Xin[a - (window - 1 - np.arange(window)) * win_stride_raw].reshape(-1) for a in anchors])

CACHE = "emb_cache.pkl"
cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
def key(tt, seed):
    return (CASE, tt, int(seed), atoms, ts, readout, nshots,
            tuple(state_indices), window, spacing_um, len(anchors))

t0 = time.time()
tr_n = len(anchors) - n_test
for tt in cfg["tts"]:
    q = QRCConfig(atom_number=atoms, encoding="local", lattice_spacing=spacing_um,
                  encoding_scale=ENC, rabi_frequency=RABI, total_time=tt,
                  time_steps=ts, readouts=readout)
    for seed in cfg["seeds"]:
        k = key(tt, seed)
        if k in cache: continue
        if time.time() - t0 > budget:
            print("TIME BUDGET REACHED", flush=True); raise SystemExit(0)
        W = _W(seed)
        Wsc = np.clip(MinMaxScaler((0.0, 1.0)).fit(W[:tr_n]).transform(W), 0.0, 1.0)
        pk = ("PARTIAL",) + k
        parts = cache.get(pk, [])
        done = sum(len(p) for p in parts)
        BLOCK = 15
        while done < len(Wsc):
            if time.time() - t0 > budget:
                cache[pk] = parts; pickle.dump(cache, open(CACHE, "wb"))
                print(f"PARTIAL saved {done}/{len(Wsc)} for {k}", flush=True)
                raise SystemExit(0)
            E_blk = emulate_qrc_embeddings(q, Wsc[done:done+BLOCK], nshots=nshots)
            parts.append(E_blk); done += len(E_blk)
            cache[pk] = parts; pickle.dump(cache, open(CACHE, "wb"))
        E = np.concatenate(parts, axis=0)
        cache.pop(pk, None)
        cache[k] = E
        pickle.dump(cache, open(CACHE, "wb"))
        print(f"[{time.time()-t0:7.1f}s] atoms={atoms} ts={ts} ro={readout} sp={spacing_um} tt={tt} seed={seed} n={len(anchors)} E={E.shape}", flush=True)
print("DONE", flush=True)

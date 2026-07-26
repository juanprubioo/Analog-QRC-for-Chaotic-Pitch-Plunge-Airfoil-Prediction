"""submit_aquila.py — Envío de la corrida ÚNICA de hardware (198 tareas, diseño congelado).

Modos:
  python3 submit_aquila.py mock            # simulador AHS local de Braket (gratis): mismo programa,
                                           # mismo esquema de resultados, mismo parsing que Aquila.
  python3 submit_aquila.py budget          # solo imprime presupuesto; no envía nada.
  python3 submit_aquila.py real "AUTORIZO EL ENVÍO A AQUILA"
                                           # envío real. Requiere la frase EXACTA como argumento
                                           # y credenciales AWS ya configuradas en el entorno.

Garantías:
  - try/except por tarea con log en aquila_hw_submit.log; una tarea fallida no aborta el lote.
  - Persistencia INMEDIATA del ARN + metadatos (probe, anchor, config) en aquila_hw_tasks/<tag>.json.
  - Reanudación segura: tareas con manifiesto existente (status submitted/done) NO se reenvían.
  - En mock, el resultado se guarda crudo de inmediato (el simulador local es síncrono).
  - NUNCA lee, pide ni imprime credenciales.
"""
from __future__ import annotations
import sys, json, time, traceback
from datetime import datetime, timezone

import numpy as np
import hw_common as hw

MODE = sys.argv[1] if len(sys.argv) > 1 else "budget"
PHRASE = sys.argv[2] if len(sys.argv) > 2 else ""
AUTH_PHRASE = "AUTORIZO EL ENVÍO A AQUILA"

assert MODE in ("mock", "budget", "real"), f"modo inválido: {MODE}"

def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(hw.LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------- presupuesto ----------------
proto = hw.load_protocol()
n_anchor = len(proto["anchors"])
n_tasks = n_anchor * len(hw.PROBES_US)
PER_TASK_USD, PER_SHOT_USD = 0.30, 0.01   # tarifa nominal Braket QPU; VERIFICAR en consola AWS antes de enviar
usd = n_tasks * (PER_TASK_USD + PER_SHOT_USD * hw.NSHOTS)
print("=" * 70)
print("CORRIDA ÚNICA AQUILA — diseño prerregistrado congelado")
print(f"  {hw.N_ATOMS} átomos | a={hw.SPACING_UM} µm | Ω={hw.RABI} rad/µs | Δ_i=4.5−9·x_i | readout {hw.READOUT}")
print(f"  probes {hw.PROBES_US} µs | ts={hw.TIME_STEPS} | semilla {hw.SEED} | anchors {n_anchor}")
print(f"  tareas: {n_anchor} × {len(hw.PROBES_US)} = {n_tasks} | shots/tarea: {hw.NSHOTS}")
print(f"  COSTO NOMINAL: {n_tasks} × (${PER_TASK_USD} + ${PER_SHOT_USD}×{hw.NSHOTS}) = ${usd:,.2f}"
      f"  (verificar tarifa vigente en la consola de AWS Braket)")
print("=" * 70)

if MODE == "budget":
    sys.exit(0)

if MODE == "real":
    if PHRASE != AUTH_PHRASE:
        print(f'\nNO SE ENVÍA. El modo real exige la frase exacta como segundo argumento:\n'
              f'  python3 submit_aquila.py real "{AUTH_PHRASE}"')
        sys.exit(1)
    from braket.aws import AwsDevice
    device = AwsDevice(hw.AQUILA_ARN)
    log(f"REAL: dispositivo {device.name}, status {device.status}")
else:
    from braket.devices import LocalSimulator
    device = LocalSimulator("braket_ahs")
    log("MOCK: LocalSimulator('braket_ahs') — mismo programa y parsing que la ruta real, costo $0")

# ---------------- envío ----------------
Wsc = proto["Wsc"]
submitted = skipped = failed = 0
t0 = time.time()

for tt in hw.PROBES_US:
    for idx in range(n_anchor):
        tag = hw.task_tag(tt, idx)
        m = hw.load_manifest(tag)
        if m is not None and m.get("status") in ("submitted", "done"):
            skipped += 1
            continue
        try:
            prog = hw.build_native_ahs(Wsc[idx], total_time_us=tt)
            meta = dict(tag=tag, probe_us=tt, anchor_idx=int(idx), seed=hw.SEED,
                        n_atoms=hw.N_ATOMS, spacing_um=hw.SPACING_UM, rabi=hw.RABI,
                        encoding_scale=hw.ENCODING_SCALE, readout=hw.READOUT,
                        nshots=hw.NSHOTS, mode=MODE,
                        amendment=hw.AMENDMENT, ramp_us=hw.RAMP_US,
                        programmed_time_us=tt + hw.RAMP_US, effective_time_us=tt,
                        submitted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            if MODE == "real":
                prog_d = prog.discretize(device)
                task = device.run(prog_d, shots=hw.NSHOTS, experimental_capabilities="ALL")
                meta.update(arn=task.id, status="submitted")
                hw.save_manifest(tag, meta)          # ARN a disco ANTES de cualquier otra cosa
                log(f"SUBMITTED {tag} -> {task.id}")
            else:
                task = device.run(prog, shots=hw.NSHOTS)
                res = task.result()                   # síncrono en local
                arn = getattr(task, "id", f"local:{tag}")
                hw.save_raw_result(tag, arn, res, mode=MODE)
                meta.update(arn=str(arn), status="done")
                hw.save_manifest(tag, meta)
                log(f"MOCK OK  {tag}  ({time.time()-t0:6.1f}s)")
            submitted += 1
        except Exception:
            failed += 1
            err = traceback.format_exc()
            log(f"FAILED {tag}\n{err}")
            hw.save_manifest(tag, dict(tag=tag, probe_us=tt, anchor_idx=int(idx),
                                       mode=MODE, status="failed", error=err.splitlines()[-1]))
            if MODE == "real" and failed >= 3 and submitted == 0:
                log("ABORT: 3 fallos consecutivos sin ningún envío exitoso — probable rechazo "
                    "de validación del dispositivo. Revisar el error, ajustar lo mínimo "
                    "requerido (contingencia), y relanzar (reanuda).")
                sys.exit(2)

log(f"FIN {MODE}: enviadas {submitted}, saltadas (ya existentes) {skipped}, fallidas {failed}, "
    f"total esperado {n_tasks}, t={time.time()-t0:.0f}s")
if failed:
    log("Reintento SOLO de fallidas: borrar sus .json con status 'failed' y relanzar este script (reanuda).")

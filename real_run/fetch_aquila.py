"""fetch_aquila.py — Recuperación de resultados de la corrida real (por ARN) y respaldo crudo.

Uso: python3 fetch_aquila.py
Recorre aquila_hw_tasks/*.json con status 'submitted', consulta cada tarea por su ARN,
y al completarse guarda los shots crudos en aquila_hw_results/<tag>_raw.npz y marca
el manifiesto como 'done'. Reanudable: lo ya descargado no se vuelve a pedir.
En mock no se necesita (submit_aquila.py ya guarda los resultados).
"""
import json, time, traceback
from datetime import datetime, timezone

import hw_common as hw

def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(hw.LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

from braket.aws import AwsQuantumTask

pending = []
for p in sorted(hw.TASK_DIR.glob("*.json")):
    m = json.load(open(p, encoding="utf-8"))
    if m.get("status") == "submitted":
        pending.append(m)
log(f"Tareas pendientes de descarga: {len(pending)}")

while pending:
    still = []
    for m in pending:
        tag, arn = m["tag"], m["arn"]
        try:
            task = AwsQuantumTask(arn)
            state = task.state()
            if state in ("COMPLETED",):
                res = task.result()
                hw.save_raw_result(tag, arn, res, mode="real")
                m["status"] = "done"
                m["fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                hw.save_manifest(tag, m)
                log(f"DONE   {tag}")
            elif state in ("FAILED", "CANCELLED"):
                m["status"] = f"hw_{state.lower()}"
                hw.save_manifest(tag, m)
                log(f"{state} {tag} — documentar; reintentar solo esta tarea si procede (contingencia)")
            else:
                still.append(m)
        except Exception:
            log(f"FETCH ERROR {tag}\n{traceback.format_exc().splitlines()[-1]}")
            still.append(m)
    if still:
        log(f"{len(still)} tareas aún en cola/ejecución; nueva pasada en 120 s")
        time.sleep(120)
    pending = still

log("Descarga completa.")

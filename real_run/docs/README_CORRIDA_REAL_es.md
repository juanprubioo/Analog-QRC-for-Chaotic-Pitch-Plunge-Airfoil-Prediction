# CORRIDA REAL EN AQUILA — pasos 0–3 COMPLETADOS, pendiente SOLO autorización + envío

## Estado del protocolo
- Paso 0: verify.py 3 PASS; caché verificado (chunks emu-500, 3 probes, semilla 1234).
- Paso 1: referencia emu-100 computada y cacheada (claves nshots=100) — step1_baseline.csv.
- Paso 2: pipeline corregido: C7 (convención de bits bloqade, ground=1; v15 invertía
  el signo de rho), prefijo "aquila", try/except + log por tarea, ARNs a disco
  inmediatos, reanudación segura, fail-fast (3 fallos iniciales -> abort).
- Paso 3: mock end-to-end 198/198 con el simulador AHS local de Braket (mismo programa
  y parsing que la ruta real): tabla + figura generadas; criterio primario CUMPLIDO.
- ENMIENDA E1 (confirmada): rampa mínima legal 0.05 µs + compensación de área.
  Tiempos programados {0.55, 0.75, 1.85} µs -> efectivos {0.5, 0.70, 1.8} µs.
  Justificación: rho(mock↔emu500) = 0.971/0.953/0.944 vs línea base de muestreo
  0.968/0.952/0.938. La rampa v15 destruía el nodo (contraste -0.011).

## Secuencia en tu máquina (Python >= 3.10)
    pip install amazon-braket-sdk numpy scipy scikit-learn pandas matplotlib
    # colocar todos los archivos de este paquete en un directorio, con
    # data_cache.npz y emb_cache.pkl (el emb_cache.pkl DE ESTE PAQUETE: trae emu-100)
    python3 verify.py                       # debe dar 3 PASS
    python3 submit_aquila.py budget         # revisa el resumen y VERIFICA tarifa en consola AWS
    python3 submit_aquila.py real "AUTORIZO EL ENVÍO A AQUILA"
    python3 fetch_aquila.py                 # espera/descarga; respaldo crudo inmediato
    python3 analyze_hardware.py real        # tabla + figura Sec. V.E

Reanudación: si algo se corta, relanzar el mismo comando; lo enviado no se reenvía
(manifiestos en aquila_hw_tasks/). Los ARNs quedan en esos JSON — NO borrar.
Contingencias: tarea rechazada/fallida -> borrar SOLO su .json 'failed' y relanzar;
pérdida parcial -> analyze_hardware.py computa con los anchors disponibles y declara n.

## Ventana de operación de Aquila
Aquila atiende por ventanas de disponibilidad (ver consola Braket / Braket Direct).
Las 198 tareas entran en cola; fetch_aquila.py puede dejarse corriendo o relanzarse.

## Costo
198 tareas x ($0.30/tarea + $0.01/shot x 100) = $257.40 nominal.
Verificar tarifa vigente en la consola antes de autorizar.

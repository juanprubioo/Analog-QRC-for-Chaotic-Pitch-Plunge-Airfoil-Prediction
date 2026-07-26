# Guía de implementación — pipeline de verificación y experimentación QRC

Cómo integrar y usar `make_data_cache.py`, `verify.py`, `run_generic.py`, `analyze_generic.py` y los cachés (`data_cache.npz`, `emb_cache.pkl`) en el proyecto del airfoil.

---

## 1. Estructura de directorio

Todos los archivos deben estar en el **mismo directorio**, porque los scripts hacen import dinámico por ruta relativa:

```
proyecto_qrc/
├── airfoil_simulator.py          # tuyo (sin cambios)
├── airfoil_qrc_v5.py             # tuyo (sin cambios)
├── airfoil_state_estimation.py   # tuyo (no lo usan estos scripts, pero el notebook sí)
├── make_data_cache.py            # nuevo
├── verify.py                     # nuevo
├── run_generic.py                # nuevo
├── analyze_generic.py            # nuevo
├── data_cache.npz                # generado (o entregado)
└── emb_cache.pkl                 # generado (o entregado — ~1.5 h de emulación dentro)
```

Dependencias de cada script:

| Script | Requiere en el directorio | Requiere instalado |
|---|---|---|
| `make_data_cache.py` | `airfoil_qrc_v5.py`, `airfoil_simulator.py` | numpy, scipy |
| `verify.py` | los dos anteriores (+ `data_cache.npz` opcional, acelera) | numpy, scipy, scikit-learn |
| `run_generic.py` | los dos anteriores + `data_cache.npz` (obligatorio) | + bloqade-analog |
| `analyze_generic.py` | los dos anteriores + `data_cache.npz` + `emb_cache.pkl` | numpy, scipy, scikit-learn, pandas |

Nota: `airfoil_qrc_v5.py` a su vez importa `airfoil_simulator.py` desde su propio directorio — por eso todo va junto.

## 2. Instalación del entorno

```bash
pip install bloqade-analog numpy scipy scikit-learn pandas
```

Verificación mínima:

```bash
python3 -c "import bloqade.analog; import sklearn; print('OK')"
```

El emulador que usan los scripts es `.bloqade.python()` (física exacta, gratis, local). Nada aquí toca hardware ni `quera.mock`.

## 3. Orden de operaciones la primera vez

### Paso 3.1 — Generar (o colocar) `data_cache.npz`

Si tienes el archivo entregado, cópialo al directorio y salta al paso 3.2. Si no:

```bash
python3 make_data_cache.py     # ~2-3 min
```

Qué hace: simula el Caso IV limpio (`XC`, 70001×6), y para cada semilla en `[1234, 1335, 1436, 1537, 1638, 1739]` genera las observaciones con 40% de ruido gaussiano y las suaviza con Savitzky-Golay (ventana 31, orden 3). Guarda todo en `data_cache.npz` con claves `XC`, `dt`, `dn_<seed>`.

Por qué existe: cada script consumía ~1 min re-simulando esto en cada invocación; el caché lo baja a segundos y garantiza que **todos** los experimentos usan exactamente los mismos datos de entrada.

Si necesitas más semillas: añádelas a la lista `SEEDS` al inicio del script y re-ejecuta (regenera todo, es determinista por semilla).

### Paso 3.2 — Correr `verify.py`

```bash
python3 verify.py
```

Salida esperada:

```
tau_c = 10.30 tu | anchors = 66 (train 44 / test 22) | tol = 0.02
  PASS  NG-RC deg2 alpha=1e-4 (roto)     obtenido -8.912  esperado -8.912
  PASS  NG-RC deg2 RidgeCV (justo)       obtenido +0.405  esperado +0.405
  PASS  NG-RC deg1 RidgeCV (lineal)      obtenido +0.706  esperado +0.706
VERIFY OK
```

Si algo da FAIL, no sigas: o cambió `build_ngrc_model`, o el suavizado, o la construcción de anchors. El assert de τ_c ≈ 10.3 separa los dos casos: si falla ahí, el problema es el simulador/burn-in; si falla en los skills, es el protocolo de evaluación o el modelo.

Cuándo correrlo: al inicio de cada sesión de resultados, y después de cualquier edición a `airfoil_qrc_v5.py` o al protocolo. Sale con código 1 en fallo, así que sirve en CI: `python3 verify.py || exit 1`.

Cuándo actualizarlo: si cambias el protocolo **deliberadamente** (p. ej. test fijo con gap, recomendación 4), los tres números de referencia cambian de forma legítima. Procedimiento: corre el protocolo nuevo, revisa a mano que los valores son razonables, y actualiza el diccionario `REFS` al inicio de `verify.py` con los nuevos valores y un comentario con la fecha y el motivo.

### Paso 3.3 — Colocar `emb_cache.pkl` (o empezar de cero)

Si tienes el entregado, cópialo: contiene 69 chunks de embeddings (~4600 emulaciones, ~1.5 h de cómputo). Si no está, `run_generic.py` lo crea vacío y lo va llenando.

Inventario de lo que trae el entregado — formato `(átomos, time_steps, readout, espaciado µm, n_anchors)`:

```
(8,  2, Z,  8.5, 66)   3 chunks   — sonda de espaciado V>Ω
(8,  2, Z, 10.0, 66)  15 chunks   — barrido base tt={1.0,1.5,2.0,2.5,3.0} × 3 semillas
(8,  2, Z, 12.0, 66)   3 chunks   — sonda de espaciado V<Ω
(8,  2, ZZ,10.0,200)   3 chunks   — anchors extendidos, tt=2.0
(8,  3, Z, 10.0, 66)  15 chunks   — ts=3, barrido tt completo
(8,  5, Z, 10.0, 66)  12 chunks   — ts=5, tt={1.0,1.5,2.0} (6 semillas en tt=1.5)
(12, 2, Z, 10.0, 66)  15 chunks   — 12 átomos, barrido tt completo
(16, 2, Z, 10.0, 66)   3 chunks   — 16 átomos, tt=2.0
```

Un "chunk" = una combinación (tt, semilla) = 66 (o 200) embeddings.

## 4. Uso de `run_generic.py`

### 4.1 Sintaxis

```bash
python3 run_generic.py '<config_json>' <presupuesto_en_segundos>
```

El presupuesto es opcional (por defecto ilimitado). Al agotarse, el script guarda el progreso y sale limpiamente con `TIME BUDGET REACHED` o `PARTIAL saved k/66`; la siguiente invocación con el **mismo JSON** retoma exactamente donde iba.

### 4.2 Campos del JSON

| Campo | Tipo | Obligatorio | Significado |
|---|---|---|---|
| `state_indices` | lista int | sí | Estados del airfoil usados como entrada (0=α, 1=α̇, 2=ξ, 3=ξ̇, 4=w1, 5=w2) |
| `window` | int | sí | Profundidad temporal de la ventana (nº de instantes pasados, stride 5 tu) |
| `time_steps` | int | sí | Nº de probe-times; caen en `k·(total_time/time_steps)`, k=1..ts |
| `readout` | "Z" \| "ZZ" | sí | Observables medidos. Dim: Z → átomos×ts; ZZ → (átomos + C(átomos,2))×ts |
| `tts` | lista float | sí | Valores de `total_time` en µs a computar |
| `seeds` | lista int | sí | Semillas de ruido (deben existir como `dn_<seed>` en `data_cache.npz`) |
| `nshots` | int | no (500) | Shots por emulación |
| `lattice_spacing` | float | no (10.0) | Espaciado de la cadena en µm (fija V de van der Waals) |
| `n_anchor_extra` | int | no (0) | Anchors adicionales al final → alarga el train, **pero mueve el test** (ver §7) |

El número de átomos **no se declara**: es `window × len(state_indices)`. Ejemplos: `[0,1,2,3]` × window 2 = 8 átomos; `[0..5]` × 2 = 12; `[0,1,2,3]` × 4 = 16.

Constantes fijadas dentro del script (edítalas ahí si el protocolo cambia): Ω=6.283 rad/µs, escala de encoding 9.0, burn-in 500 tu, stride de ventana 5 tu, horizontes {1,2,3}×τ_c, train/test base 44/22, encoding "local".

### 4.3 Ejemplos

Reproducir el barrido base (no hará nada si el caché entregado está presente — esa es la prueba de que el caché funciona):

```bash
python3 run_generic.py '{"state_indices":[0,1,2,3],"window":2,"time_steps":2,"readout":"Z","tts":[1.0,1.5,2.0,2.5,3.0],"seeds":[1234,1335,1436]}' 400
```

Barrido de fase directo (recomendación 2; ts=1 → el único probe cae en total_time):

```bash
python3 run_generic.py '{"state_indices":[0,1,2,3],"window":2,"time_steps":1,"readout":"Z","tts":[2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9],"seeds":[1234,1335,1436,1537,1638,1739]}' 450
```

(60 chunks de 8 átomos ≈ 10–12 min; repite la invocación hasta ver `DONE`.)

### 4.4 Costos orientativos (1 núcleo, emulador exacto)

| Config | s/punto | chunk de 66 |
|---|---|---|
| 8 átomos, ts=2 | ~0.15 | ~10 s |
| 8 átomos, ts=5 | ~0.35 | ~22 s |
| 12 átomos, ts=2 | ~0.42 | ~28 s |
| 16 átomos, ts=2 | ~5.0 | ~5.5 min |

La primera emulación de cada proceso añade ~5 s de compilación. La memoria escala como 2^N: 16 átomos ya roza los 3 GB; no intentes 18+ sin medir antes.

### 4.5 Qué guarda

Cada chunk terminado entra en `emb_cache.pkl` bajo la clave:

```python
(CASE, total_time, seed, atoms, time_steps, readout, nshots,
 tuple(state_indices), window, lattice_spacing, n_anchors)
```

Los chunks a medias se guardan bajo `("PARTIAL",) + clave` en bloques de 15 puntos y se consolidan al terminar. Si un `PARTIAL` queda huérfano (cambiaste la config a medias), bórralo:

```python
import pickle
c = pickle.load(open("emb_cache.pkl","rb"))
for k in [k for k in c if k[0]=="PARTIAL"]: del c[k]
pickle.dump(c, open("emb_cache.pkl","wb"))
```

## 5. Uso de `analyze_generic.py`

### 5.1 Sintaxis

```bash
python3 analyze_generic.py '<config_json>'
```

El JSON usa **los mismos campos** que `run_generic.py` (deben coincidir exactamente para que las claves de caché resuelvan), más tres opcionales propios del análisis:

| Campo extra | Valores | Significado |
|---|---|---|
| `readout_mode` | "cv" (defecto) \| "fixed" | Readout del QRC: RidgeCV (protocolo final) o Ridge α=1e-4 (solo para reproducir números históricos) |
| `train_sizes` | lista int | Curva de aprendizaje: evalúa con los n anchors de train más cercanos al test |
| `z_only_from_zz` | true/false | Extrae el subconjunto Z de un embedding ZZ cacheado (sin re-emular) |

### 5.2 Columnas de salida

| Columna | Significado |
|---|---|
| `skill_qrc`, `ci_lo`, `ci_hi` | Skill vs persistencia pooled (semillas × horizontes) + CI bootstrap 95% (3000 remuestreos de puntos de test) |
| `skill_ngrc`, `skill_ngrc_lin` | NG-RC deg2 CV y deg1 CV sobre la **misma ventana** que alimenta al QRC |
| `skill_ngrc_best`, `beats_classical` | Máximo clásico y si `ci_lo > máximo` |
| `qrc_seed_min`, `cls_seed_max`, `robust` | Criterio per-seed: la peor semilla del QRC debe superar a la mejor semilla clásica. **Este es el filtro que importa** — el pooled CI deja pasar falsos positivos de navaja (lo vimos con ts=5 @ tt=1.5) |
| `ok_dim` | `n_train > dim(embedding)`. Si es False, la fila es inválida para cualquier afirmación |

### 5.3 Limitación conocida

El menú clásico interno es solo deg1/deg2. Para el protocolo final del paper (recomendación 1) hay que añadir deg3 CV y el control RFF de dimensión igualada — el patrón está en `strong_classical.py` de la sesión (clase `RFF`: proyección tanh aleatoria, StandardScaler en train, RidgeCV, promedio de MSE sobre 5 proyecciones). Sin ese control, un `beats_classical=True` no es evidencia (así se desarmó el falso positivo del ZZ).

## 6. Flujos de trabajo completos

### 6.1 Sesión típica

```bash
python3 verify.py || exit 1                      # 1. sanity check
python3 run_generic.py '<config>' 450            # 2. computar (repetir hasta DONE)
python3 analyze_generic.py '<config>'            # 3. analizar
```

### 6.2 Añadir un experimento nuevo (checklist)

1. Define el JSON de config y calcula la dimensión del embedding; verifica `n_train > dim` **antes** de gastar cómputo.
2. Si usas semillas nuevas, añádelas primero a `make_data_cache.py` y regenera.
3. Corre `run_generic.py` en invocaciones con presupuesto (400–450 s va bien) hasta `DONE`.
4. Analiza. Si algo da `beats_classical=True`: (a) revisa `ok_dim` y `robust`; (b) añade el control RFF de dimensión igualada; (c) duplica las semillas; (d) cuenta cuántas configs exploraste antes de encontrarlo (multiplicidad). Solo si sobrevive a los cuatro, repórtalo.
5. Respaldar `emb_cache.pkl` (es el activo caro).

## 7. Reglas de invalidación de cachés

La clave de caché protege contra mezclas **solo** de lo que contiene. Tabla de decisiones:

| Cambio | ¿data_cache? | ¿emb_cache? |
|---|---|---|
| Nuevo tt, seed (existente en data), átomos, ts, readout, espaciado | nada | se añade solo (clave nueva) |
| Semilla nueva de ruido | regenerar (añadir a SEEDS) | nada |
| Ventana Savitzky-Golay, nivel/tipo de ruido, caso | **regenerar** | **invalidar TODO** (los embeddings vienen de ventanas que ya no existen) |
| Burn-in, stride de ventana, horizontes, train/test base | nada | **invalidar TODO** (cambian los anchors; la clave no distingue) |
| `n_anchor_extra` | nada | clave nueva, pero **el test se mueve** a un segmento posterior del atractor: los skills NO son comparables con los de 66 anchors (lo vimos: 0.9x vs 0.7x por dificultad local, no por mérito). Para curvas de aprendizaje comparables, implementa primero el test fijo (recomendación 4) |

"Invalidar todo" = renombrar `emb_cache.pkl` a `emb_cache_v1.pkl` (no borrar: trazabilidad) y dejar que se regenere.

## 8. Solución de problemas

- `KeyError` con una tupla larga en `analyze_generic.py` → el chunk no está en el caché o el JSON no coincide campo a campo con el usado al computar (incluye los opcionales: `nshots`, `lattice_spacing`, `n_anchor_extra`). Inventaría el caché con el snippet del §3.3.
- `KeyError: 'dn_XXXX'` → la semilla no está en `data_cache.npz`; añádela a `make_data_cache.py`.
- Proceso `Killed` en configs grandes → memoria (2^N del emulador). El checkpointing por bloques de 15 preserva el progreso; relanza.
- Números distintos a los de referencia en configs QRC → esperado dentro del ruido de muestreo: los shots del emulador no tienen semilla fija, cada recomputación de un embedding difiere en ~±0.05–0.15 de skill a 500 shots. Los números **clásicos** sí son deterministas — para eso está `verify.py`. Si quieres embeddings reproducibles bit a bit, conserva el `emb_cache.pkl`.

## 9. Convención para el manuscrito

Cada número que entre al paper debe poder regenerarse con: (1) el commit del repo, (2) un JSON de config, (3) `emb_cache.pkl` respaldado. Sugerencia práctica: un archivo `experiments.md` en el repo con una línea por resultado: JSON exacto → tabla de salida → fecha. Eso es lo que un referee (o tú en seis meses) necesita.

"""Stage `evaluate` del pipeline DVC (pilot-vertebral-imaging, Arq.1 eval-only) con el
**venturalitica-sdk**. Segmenta la cohorte EN MEMORIA (TotalSegmentator/GPU vía
`segment.segment_cohort`) y la audita unida a los metadatos de adquisición REALES
(TCIA Spine-Mets-CT-SEG).

**ONLINE, SIN CSV (§3.4):** este stage YA NO lee un `cohort_results.csv` (era PHI-shaped y
fuente de staleness — eliminado). Segmenta y evalúa en un solo paso, manteniendo la cohorte
por-caso en un DataFrame en memoria; el pipeline nunca toca un CSV.

Contrato `metrics.json` (igual que loan, §3.2): rutas fijas relativas + escribe
`{control_id: valor}` (SDK 0.6.10) o `{control_id: {value, power}}` (SDK ≥0.6.11). NO juzga:
el veredicto autoritativo lo pone el motor (Rust) contra el mismo OSCAL (§6.6). Tras cerrar la
sesión `vl.monitor`, promueve el `bom.json` del run a `.venturalitica/bom.json` (lo que lee el
motor) para que el ML-BOM viaje en la evidencia firmada.

Dos fases — AMBAS sobre el MISMO DataFrame por-caso (1 fila por paciente) con power-stats:
- **Art.15 (segmentación/equidad)**: las métricas de segmentación YA SON métricas del **registry
  del SDK (≥0.6.12)** — operan sobre la columna por-caso continua `Dice` (binding `inputs.score:
  Dice`): `mean_score` (Dice global), `max_score` (sonda de fuga), `min_group_score` (peor
  escáner/modelo), `group_score_gap` (brecha por edad), `worst_cell_score` (peor celda compuesta),
  `es_dice`/ESSP (FairSeg ICLR'24, equity-scaled Dice por sexo/edad) y `score_gap` (advisory).
  Ya NO se precomputan escalares: `vl.enforce(data=df, cluster="PatientID", phase="validation")`
  → el SDK les adjunta `power` (IC por **cluster bootstrap a nivel paciente**; el df por-caso
  tiene 1 fila/paciente → `cluster=PatientID` = bootstrap a nivel paciente).
- **Art.10 (datos)**: `vl.enforce(data=df, phase="training", cluster="PatientID")` —
  k-anonimato (sobre [age_group, Manufacturer] generalizado) y completitud.

CAVEAT DE SESGO-POR-TAMAÑO DEL DICE (arXiv 2509.19778): una brecha cruda de Dice por sexo/edad
puede ser artefacto del tamaño anatómico de la estructura, no sesgo del modelo. Los gates de
equidad usan `es_dice` (ESSP) y la brecha cruda `score_gap` es solo advisory; el contraste en una
métrica de borde (NSD) es LÍNEA FUTURA (apuntar `score: NSD` a los MISMOS controles `es_dice`).
"""

import os

os.environ.setdefault("VENTURALITICA_NO_ANALYTICS", "1")

import contextlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import venturalitica as vl

from segment import segment_cohort  # segmentación EN MEMORIA (sin cohort_results.csv, §3.4)

META = "data/trusted_metadata.csv"  # metadatos de adquisición reales (TCIA) — NO es PHI cohorte
OSCAL = "shared_data/policies/assessment_plan.oscal.yaml"
METRICS = "metrics.json"
CLUSTER = "PatientID"  # unidad del cluster bootstrap (jerarquía paciente, power-stats §3.1)
BOM_ROOT = ".venturalitica/bom.json"
RUNS_DIR = Path(".venturalitica/runs")

SUCCESS_DICE = 0.85  # umbral de "segmentación clínicamente usable" → is_successful


def control_order(oscal_path: str) -> dict:
    doc = yaml.safe_load(open(oscal_path))
    reqs = doc["component-definition"]["components"][0]["control-implementations"][0][
        "implemented-requirements"
    ]
    return {r["control-id"]: i for i, r in enumerate(reqs)}


def thickness_bucket(t: float) -> str:
    return "<=1" if t <= 1 else ("1-2" if t <= 2 else ">2")


def build_cohort() -> pd.DataFrame:
    """Cohorte por-caso EN MEMORIA: segmenta (GPU) → DataFrame → une los metadatos de
    adquisición. SIN persistir ningún `cohort_results.csv` (§3.4). Deriva las columnas de
    subgrupo (`is_successful`, `age_group`, `thickness_bucket`) que trocean los controles."""
    co = segment_cohort()  # PatientID, Dice, Jaccard, SpineVol, Confidence — en memoria
    md = pd.read_csv(META)
    co["PatientID"] = co["PatientID"].astype(str)
    md["PatientID"] = md["PatientID"].astype(str)
    df = co.merge(md, on="PatientID", how="inner")
    df["is_successful"] = (df["Dice"] >= SUCCESS_DICE).astype(int)
    df["age_group"] = np.where(df["Age"] >= 65, "elderly", "adult")
    df["thickness_bucket"] = df["SliceThickness"].apply(thickness_bucket)
    return df


def metric_entry(result):
    """Entrada de `metrics.json` para un `ComplianceResult` (contrato §3.2, power-stats):
    objeto `{value, power}` si el SDK (≥0.6.11) adjuntó `power` (IC por cluster bootstrap), o
    escalar `value` con 0.6.10 (back-compat). El núcleo Rust acepta ambas formas."""
    value = float(result.actual_value)
    power = getattr(result, "power", None)
    return {"value": value, "power": power} if power else value


def _promote_bom() -> None:
    """Promueve el bom.json que BOMProbe dejó en .venturalitica/runs/<run>/ a la raíz
    .venturalitica/bom.json (lo que read_bom lee, bom.rs:15). Elige el run con mtime máximo
    (misma heurística que el CLI push). FAIL-LOUD si no hay ningún bom.json que promover."""
    candidates = sorted(
        (p for p in RUNS_DIR.glob("*/bom.json") if p.parent.name != "latest"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("eval: no se generó ningún bom.json en .venturalitica/runs/")
    Path(".venturalitica").mkdir(exist_ok=True)
    shutil.copyfile(candidates[-1], BOM_ROOT)
    print(f"bom → {BOM_ROOT} (desde {candidates[-1]})", file=sys.stderr)


def main():
    df = build_cohort()  # segmenta + une metadatos EN MEMORIA (sin cohort_results.csv, §3.4)

    with contextlib.redirect_stdout(sys.stderr):
        with vl.monitor(name="pilot-vertebral-imaging", label="venth eval"):
            # Art.15 (segmentación/equidad): las métricas de dominio YA SON métricas del registry
            # del SDK (≥0.6.12) sobre la columna por-caso `Dice` (binding `inputs.score: Dice`).
            # `data=df` + `cluster=PatientID` → cada control recibe `power` (IC por cluster
            # bootstrap a nivel paciente; el df tiene 1 fila/paciente → bootstrap por paciente).
            # `dimension`/`Sex`/`age_group` se autoenlazan a sus columnas; `score: Dice` y
            # `dimensions: [..]` viajan como params del control (compilados desde `inputs`).
            model_results = vl.enforce(
                data=df, policy=OSCAL, Sex="Sex", age_group="age_group",
                cluster=CLUSTER, phase="validation", strict=False,
            )
            # Art.10 (datos): k-anonimato + completitud. Mismo df, mismo cluster.
            data_results = vl.enforce(
                data=df, policy=OSCAL, cluster=CLUSTER, phase="training", strict=False,
            )
        _promote_bom()  # tras cerrar la sesión, el bom.json del run ya existe

    order = control_order(OSCAL)
    results = sorted(data_results + model_results, key=lambda r: order.get(r.control_id, 10**6))
    metrics = {r.control_id: metric_entry(r) for r in results}
    json.dump(metrics, open(METRICS, "w"), indent=2)


if __name__ == "__main__":
    main()

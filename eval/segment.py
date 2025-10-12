"""TotalSegmentator v2 vertebra-segmentation inference wrapper.

El **modelo BYO REAL** de pilot-vertebral-imaging. Lee las imágenes de la CACHÉ
(`SEI_TCIA_CACHE`: TCIA Spine-Mets-CT-SEG ya descargado, CT+SEG ground-truth — "las imágenes
copiadas en una carpeta") y corre TotalSegmentator en GPU sobre TODAS. `segment_cohort()`
devuelve la cohorte por-caso EN MEMORIA (sin persistir ningún `cohort_results.csv` — §3.4): el
stage `evaluate` (`eval.py`) la importa y la audita online. NO se ejecuta en el CI hermético
(requiere GPU + caché). El Dice es REAL (TS vs el SEG ground-truth de TCIA), no simulado.

Mirrors the contract of `model_evaluation.py` (which runs the MONAI
`wholeBody_ct_segmentation` SegResNet bundle) so the same audit pipeline
can score both models on the same TCIA Spine-Mets-CT-SEG cohort and emit
a side-by-side comparison.

Cohort columns (in-memory DataFrame): PatientID, Dice, Jaccard, SpineVol, Confidence
— same shape as the per-case results produced by MONAI inference.

TotalSegmentator v2.13 (Wasserthal et al., Feb 2025) is the closest
public model to the SOTA for vertebra segmentation on CT: nnU-Net
backbone, individual labels for the 25 vertebral bodies (C1–S1), Apache
2.0. We restrict to the `total` task with `roi_subset` limited to
`vertebrae_*` so the comparison is apples-to-apples with the MONAI
bundle (which we only consume the 24 vertebra channels of).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.transforms import Compose, Orientation, Spacing

# Métricas de segmentación NATIVAS del SDK (≥0.6.12, extra `[imaging]`): wrappers finos sobre
# `monai.metrics` (DiceMetric / MeanIoU). Reemplazan el bucle de Dice hecho a mano — sobre
# máscaras binarias dan el MISMO número (Dice = 2|A∩B|/(|A|+|B|)), verificado en unit-check.
from venturalitica.assurance.imaging import dice as imaging_dice
from venturalitica.assurance.imaging import iou as imaging_iou

# Local utilities (re-used from MONAI pipeline so Dice is computed
# identically against the TCIA ground-truth SEG).
from dicom_utils import (
    auto_align_orientation,
    find_ct_and_seg_files,
    get_annotated_spine_indices,
    load_dicom_seg_reconstructed,
    load_dicom_volume_robust,
    sort_dicom_files,
)

warnings.filterwarnings("ignore")


# TotalSegmentator vertebra label ids in the `total` task (v2.13).
# Range 26 (S1) → 50 (C1) plus 25 (sacrum bone). Keep aligned with
# totalsegmentator.map_to_binary if upgrading versions.
_VERTEBRA_LABEL_IDS_TS = list(range(25, 51))


# ─── MONAI bundle v0.2.7 ↔ TotalSegmentator v2.13 label mapping ────────────
# Both models share the TotalSegmentator-derived vertebra naming, but the
# integer IDs differ between MONAI's wholeBody_ct_segmentation channel_def
# (TotalSegmentator v1 scheme) and TotalSegmentator v2's class_map.
# The map is BY ANATOMICAL NAME and is the source of truth for filtering
# TS predictions down to the subset that the TCIA SEG annotator labelled
# (the SEG file uses the MONAI scheme — `get_annotated_spine_indices`
# returns MONAI ids).
_MONAI_TO_TS: dict[int, list[int]] = {
    18: [27],  # L5
    19: [28],  # L4
    20: [29],  # L3
    21: [30],  # L2
    22: [31],  # L1
    23: [32],  # T12
    24: [33],  # T11
    25: [34],  # T10
    26: [35],  # T9
    27: [36],  # T8
    28: [37],  # T7
    29: [38],  # T6
    30: [39],  # T5
    31: [40],  # T4
    32: [41],  # T3
    33: [42],  # T2
    34: [43],  # T1
    35: [44],  # C7
    36: [45],  # C6
    37: [46],  # C5
    38: [47],  # C4
    39: [48],  # C3
    40: [49],  # C2
    41: [50],  # C1
    # MONAI's `sacrum` (92) lumps the sacral bone together with the S1
    # vertebral body. TS v2 keeps them separate (25 = sacrum bone,
    # 26 = vertebrae_S1) — to make the masks compatible we take the
    # union of both TS labels whenever the GT marks MONAI's 92.
    92: [25, 26],
}


def _annotated_ts_labels(annotated_monai: list[int]) -> list[int]:
    """Translate the MONAI-scheme indices found in a TCIA SEG into the
    set of TotalSegmentator-scheme indices that cover the same anatomy.
    Used to restrict the TS prediction to the cohort the GT actually
    annotated — fair-comparison prerequisite."""
    out: list[int] = []
    for m in annotated_monai:
        if m in _MONAI_TO_TS:
            out.extend(_MONAI_TO_TS[m])
    return sorted(set(out))


def _dicom_ct_to_nifti(ct_files: list[Path]) -> tuple[Path, MetaTensor]:
    """Save the DICOM CT series as a NIfTI in its **raw** orientation so
    TotalSegmentator (which performs its own RAS-orientation / resampling
    internally and emits output in the exact input grid) can be compared
    against the TCIA SEG ground truth in the same coordinate system.

    Earlier versions of this wrapper pre-transformed the CT to
    RAS + 1.5 mm before saving; the resulting pred and the GT (resampled
    via a different code path) ended up on subtly different grids and
    the Dice collapsed to zero. Letting TotalSegmentator handle
    orientation removes that whole class of alignment bugs.
    """
    raw_meta = load_dicom_volume_robust([str(p) for p in ct_files])

    tmp_dir = Path(tempfile.mkdtemp(prefix="vlts_"))
    nifti_path = tmp_dir / "ct.nii.gz"

    affine = raw_meta.affine.cpu().numpy() if torch.is_tensor(raw_meta.affine) else np.asarray(raw_meta.affine)
    array = raw_meta.cpu().numpy() if torch.is_tensor(raw_meta) else np.asarray(raw_meta)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    nib.save(nib.Nifti1Image(array.astype(np.int16), affine), str(nifti_path))
    return nifti_path, raw_meta


def _run_totalsegmentator(nifti_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, "nib.Nifti1Image"]:
    """Run TotalSegmentator on a CT NIfTI; return (multilabel_mask, affine,
    softmax_probs_or_none).

    We pass `roi_subset=['vertebrae_*']` so only vertebra labels are kept
    — keeps the comparison apples-to-apples with the MONAI bundle's
    spine-only prediction path.
    """
    from totalsegmentator.python_api import totalsegmentator

    # With ml=True TotalSegmentator writes a SINGLE multilabel NIfTI file
    # whose path is taken verbatim from the `output` argument (it ignores
    # the directory-vs-file distinction and appends .nii if missing). So
    # we pass an explicit .nii.gz file path rather than a folder.
    multilabel_path = nifti_path.parent / "ts_out.nii.gz"
    from totalsegmentator.map_to_binary import class_map
    roi_subset = [v for v in class_map["total"].values() if v.startswith("vertebrae_")]

    totalsegmentator(
        input=str(nifti_path),
        output=str(multilabel_path),
        task="total",
        roi_subset=roi_subset,
        ml=True,
        fast=False,
        verbose=False,
        quiet=True,
    )
    # Fall back to any sibling .nii / .nii.gz the API may have produced.
    if not multilabel_path.exists():
        candidates = sorted(nifti_path.parent.glob("ts_out*.ni*"))
        if not candidates:
            raise FileNotFoundError(
                f"TotalSegmentator produced no output near {multilabel_path}"
            )
        multilabel_path = candidates[0]

    img = nib.load(str(multilabel_path))
    mask = np.asarray(img.dataobj).astype(np.int32)
    affine = img.affine
    # Return the loaded nibabel image too — callers need it as the
    # reference grid when resampling the GT into the prediction's frame.
    return mask, affine, None, img


def _compute_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Per-caso Dice + Jaccard de las MISMAS máscaras binarias, ahora vía la métrica
    NATIVA del SDK (`venturalitica.assurance.imaging`, wrapper fino sobre `monai.metrics`),
    en lugar del bucle hecho a mano. `imaging.dice`→`DiceMetric`, `imaging.iou`→`MeanIoU`.

    En máscaras BINARIAS coincide EXACTAMENTE con la fórmula previa (Dice = 2|A∩B|/(|A|+|B|),
    Jaccard = |A∩B|/|A∪B|) — verificado en un unit-check sintético. Beneficio: la columna
    `Dice` por-caso la produce la misma librería (Apache-2.0) que el resto del ecosistema
    MONAI, y deja de ser código a mantener aquí. Caso degenerado (ambas vacías): MONAI emite
    NaN; lo mapeamos a 0.0 para conservar el contrato de la fórmula anterior."""
    pred_b = (pred > 0).astype(np.uint8)
    gt_b = (gt > 0).astype(np.uint8)
    if int(pred_b.sum()) == 0 and int(gt_b.sum()) == 0:
        return 0.0, 0.0
    dice = imaging_dice(pred_b, gt_b)
    jaccard = imaging_iou(pred_b, gt_b)
    dice = 0.0 if np.isnan(dice) else float(dice)
    jaccard = 0.0 if np.isnan(jaccard) else float(jaccard)
    return dice, jaccard


def evaluate_patient(patient_dir: Path) -> dict | None:
    """Run TotalSegmentator on one patient and compute Dice against the
    TCIA SEG ground truth. Returns the metrics row or None on hard failure."""
    pid = patient_dir.name
    print(f"\n  ▶ {pid}")
    try:
        ct_files, seg_files = find_ct_and_seg_files(patient_dir)
    except Exception as exc:
        print(f"      ❌ DICOM walk failed: {exc}")
        return None
    if not ct_files:
        print(f"      ⏭  no CT series — skipping")
        return None

    tmp_root = None
    try:
        nifti_path, nifti_meta = _dicom_ct_to_nifti([Path(f) for f in ct_files])
        tmp_root = nifti_path.parent
        ml_mask, ml_affine, _, ml_img = _run_totalsegmentator(nifti_path)

        # We restrict the TS prediction to the same subset of vertebrae
        # the TCIA SEG annotated (otherwise TS predicts all 25 vertebrae
        # and the unannotated ones inflate the union → Dice collapses).
        # `get_annotated_spine_indices` returns MONAI-scheme ids; we
        # translate to TS ids before filtering.
        annotated_monai = get_annotated_spine_indices(seg_files[0]) if seg_files else None
        if annotated_monai:
            ts_subset = _annotated_ts_labels(annotated_monai) or _VERTEBRA_LABEL_IDS_TS
        else:
            ts_subset = _VERTEBRA_LABEL_IDS_TS
        spine_mask = np.isin(ml_mask, ts_subset).astype(np.uint8)
        spine_vol = int(spine_mask.sum())
        print(
            f"      ↳ predicted spine volume: {spine_vol} voxels  "
            f"(restricted to {len(ts_subset)} TS labels mapped from "
            f"{len(annotated_monai) if annotated_monai else 0} GT-annotated MONAI labels)"
        )

        # Compute Dice against the TCIA SEG ground truth — but the GT
        # arrives in the raw CT grid (post auto_align_orientation
        # heuristic) while the TS prediction lives in TotalSegmentator's
        # output grid (orientation reconciled via the DICOM affine).
        # `auto_align`'s bone-overlap heuristic occasionally picks the
        # wrong flip for sparse-SEG patients, so we use nibabel's
        # affine-aware `resample_from_to` to put the GT exactly on the
        # prediction's grid. That eliminates orientation + spacing drift
        # in one operation.
        dice, jaccard = 0.0, 0.0
        if seg_files:
            from nibabel.processing import resample_from_to

            ct_files_sorted = sort_dicom_files([str(p) for p in ct_files])
            raw_seg_tensor = load_dicom_seg_reconstructed(
                seg_files[0], ct_files_sorted, target_shape=nifti_meta.shape
            )
            raw_seg_tensor = auto_align_orientation(nifti_meta, raw_seg_tensor)
            gt_array = raw_seg_tensor.cpu().numpy() if torch.is_tensor(raw_seg_tensor) else np.asarray(raw_seg_tensor)
            if gt_array.ndim == 4 and gt_array.shape[0] == 1:
                gt_array = gt_array[0]

            # The GT lives in the raw CT grid — wrap it as a nibabel image
            # with the raw CT affine so resample_from_to can project it
            # into the prediction's frame.
            raw_ct_affine = (
                nifti_meta.affine.cpu().numpy()
                if torch.is_tensor(nifti_meta.affine)
                else np.asarray(nifti_meta.affine)
            )
            gt_img = nib.Nifti1Image(gt_array.astype(np.int32), raw_ct_affine)
            gt_in_pred_grid = resample_from_to(gt_img, ml_img, order=0)  # NN — preserve label ids
            gt_resampled = np.asarray(gt_in_pred_grid.dataobj).astype(np.int32)

            # TCIA SEG voxels are stored as a BINARY mask of the annotated
            # vertebrae (the per-level identity lives in the SEG metadata
            # consumed by `get_annotated_spine_indices`, not in the voxel
            # values). We mirror MONAI's evaluator: filter the PREDICTION
            # to the annotated levels (done above) and compare against
            # the binary GT as-is.
            gt_binary = (gt_resampled > 0).astype(np.uint8)
            dice, jaccard = _compute_dice(spine_mask, gt_binary)
            print(
                f"      ↳ Dice={dice:.4f} | Jaccard={jaccard:.4f}  "
                f"(pred {int(spine_mask.sum()):,} vox / gt {int(gt_binary.sum()):,} vox / "
                f"shared shape {spine_mask.shape})"
            )
        else:
            print(f"      ⏭  no SEG — Dice unavailable")

        # TotalSegmentator's python_api does not surface per-voxel softmax,
        # so we report a deterministic Confidence proxy: the fraction of
        # the multilabel volume that received a vertebra label. This is a
        # weaker calibration signal than MONAI's softmax-derived
        # Confidence — flagged in the audit warnings.
        proxy_conf = float(spine_vol) / float(ml_mask.size) if ml_mask.size else 0.0

        return {
            "PatientID": pid,
            "Dice": dice,
            "Jaccard": jaccard,
            "SpineVol": spine_vol,
            "Confidence": proxy_conf,
        }
    except Exception as exc:
        print(f"      ❌ {exc.__class__.__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if tmp_root and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)


# Marca con la que el subproceso de un paciente emite su fila (JSON) en stdout,
# separada del log de progreso (que va a stderr). Ver el bucle de segment_cohort.
_ROW_MARKER = "__SEI_ROW__"


def segment_cohort(data_path: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Segmenta TODA la cohorte en GPU y devuelve el resultado por-caso **EN MEMORIA** (sin
    persistir ningún `cohort_results.csv` — es PHI-shaped + fuente de staleness, §3.4). Cada fila:
    PatientID, Dice (vs el SEG ground-truth de TCIA), Jaccard, SpineVol, Confidence (proxy).
    El stage `evaluate` lo consume directamente → `vl.enforce` → `metrics.json` (el motor nunca
    toca un CSV, corre online)."""
    # Resolución del directorio de la cohorte (orden de precedencia):
    #   1) data_path explícito (uso programático/standalone).
    #   2) SEI_TCIA_CACHE del entorno — es el contrato del pipeline: el stage `evaluate` llama
    #      a segment_cohort() SIN args y la caché TCIA (CT+SEG, no es PHI commiteable) se monta
    #      por env, no en el repo (~13G).
    #   3) fallback <repo>/shared_data/dicom (layout local de depuración).
    if data_path:
        data_dir = Path(data_path)
    elif os.environ.get("SEI_TCIA_CACHE"):
        data_dir = Path(os.environ["SEI_TCIA_CACHE"])
    else:
        data_dir = Path(__file__).parent.parent / "shared_data" / "dicom"
    if not data_dir.exists():
        raise SystemExit(f"DICOM directory not found: {data_dir}")

    # SEI_TCIA_LIMIT acota la cohorte (útil para validación rápida en GPU) cuando el
    # llamante no pasa limit explícito.
    if limit is None and os.environ.get("SEI_TCIA_LIMIT"):
        limit = int(os.environ["SEI_TCIA_LIMIT"])

    patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if limit:
        patient_dirs = patient_dirs[:limit]
    print(f"=== TotalSegmentator v2 vertebra evaluation — {len(patient_dirs)} patients ===")

    rows = []
    for pat_dir in patient_dirs:
        # AISLAMIENTO DE MEMORIA POR PACIENTE. Cada caso maneja volúmenes de ~300M
        # vóxeles y nnU-Net/PyTorch retienen memoria NATIVA (heap C, contexto CUDA,
        # /dev/shm de los workers) que gc.collect()/empty_cache() NO recuperan; sobre
        # la cohorte completa el proceso crece sin límite y el OOM-killer lo mata
        # (exit 137). Ejecutamos cada paciente en un SUBPROCESO fresco (`--one-patient`):
        # al terminar, el kernel recupera TODA su memoria. Cuesta recargar el modelo por
        # paciente, pero es la única forma robusta de barrer la cohorte entera en una
        # máquina con RAM finita. La fila (Dice/Jaccard/…) vuelve por stdout marcada con
        # _ROW_MARKER; el log de progreso del hijo se reenvía por stderr.
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--one-patient", str(pat_dir)],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        row = None
        for line in proc.stdout.splitlines():
            if line.startswith(_ROW_MARKER):
                payload = line[len(_ROW_MARKER):]
                row = json.loads(payload) if payload != "null" else None
                break
        else:
            # El hijo no emitió fila: un fallo en UN paciente (crash/OOM aislado) no
            # tumba el barrido de la cohorte; se registra y se continúa.
            if proc.returncode != 0:
                print(
                    f"      ⚠ paciente {pat_dir.name}: subproceso salió con código "
                    f"{proc.returncode} — descartado del análisis"
                )
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"    mean Dice: {df['Dice'].mean():.4f} | min: {df['Dice'].min():.4f} | max: {df['Dice'].max():.4f}")
    return df


def main(model_path: str | None = None, data_path: str | None = None,
         output_csv: str | None = None, limit: int | None = None) -> None:
    """Útil standalone/depuración: segmenta y, SOLO si se pide explícitamente un `output_csv`,
    vuelca el CSV. El pipeline NO lo usa (el stage `evaluate` llama a `segment_cohort` y mantiene
    la cohorte en memoria — sin CSV intermedio, §3.4)."""
    df = segment_cohort(data_path=data_path, limit=limit)
    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n  ✓ Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    # MODO SUBPROCESO (`--one-patient <dir>`): evalúa UN paciente aislado y emite su
    # fila como JSON marcado en stdout. Lo invoca segment_cohort() por cada paciente
    # para acotar el pico de memoria (ver el bucle). El log de progreso de
    # evaluate_patient se redirige a stderr para no contaminar la línea marcada.
    if "--one-patient" in sys.argv:
        _pdir = Path(sys.argv[sys.argv.index("--one-patient") + 1])
        with contextlib.redirect_stdout(sys.stderr):
            _row = evaluate_patient(_pdir)
        sys.stdout.write(_ROW_MARKER + json.dumps(_row, default=float) + "\n")
        sys.stdout.flush()
        sys.exit(0)

    # Uso standalone/depuración. El pipeline NO ejecuta este `__main__`: el stage `evaluate`
    # importa `segment_cohort` y mantiene la cohorte EN MEMORIA (sin CSV, §3.4). La caché de
    # imágenes va por env (no en el repo, son 13G). El volcado a CSV es OPT-IN explícito
    # (`SEI_SEGMENT_OUT`) — sin él no se persiste ningún `cohort_results.csv`.
    import os

    cache = os.environ.get("SEI_TCIA_CACHE")
    if not cache:
        raise SystemExit(
            "SEI_TCIA_CACHE no definido: carpeta de imágenes TCIA cacheadas "
            "(<PatientID>/ con CT + SEG ground-truth). Es 'engañar al script con las "
            "imágenes copiadas' — sin red en cada ejecución."
        )
    out = os.environ.get("SEI_SEGMENT_OUT")  # opcional: sin él, NO se escribe CSV (online)
    limit = int(os.environ["SEI_TCIA_LIMIT"]) if os.environ.get("SEI_TCIA_LIMIT") else None
    main(data_path=cache, output_csv=out, limit=limit)

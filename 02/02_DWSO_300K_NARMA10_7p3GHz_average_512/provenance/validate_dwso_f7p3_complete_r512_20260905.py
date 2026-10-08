#!/usr/bin/env python3
"""Independently validate the local complete-R512 characterization bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_R256_MSE = 0.0036902752417541564
EXPECTED_R434_MSE = 0.003567052440879413


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    qa = json.loads((root / "summary_qa.json").read_text(encoding="utf-8"))
    if qa.get("status") != "pass":
        raise ValueError("summary_qa status is not pass")
    metrics_path = root / "dwso_300K_f7p3_complete_R512_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["group_id"]) != {
        "historical_R256",
        "combined_completed_R434",
        "final_increment_R78",
        "complete_R512",
    }:
        raise ValueError("metrics group set mismatch")

    metric_checks: list[dict[str, float | int | str]] = []
    for row in metrics.itertuples(index=False):
        pred_path = root / row.group_id / "narma10_predictions.csv"
        pred = pd.read_csv(pred_path)
        test = pred.loc[pred["split"].eq("test")]
        if len(test) != 780:
            raise ValueError(f"{row.group_id}: sealed-test count is {len(test)}")
        y = test["target_y_next"].to_numpy(dtype=float)
        yhat = test["pred_y_next"].to_numpy(dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(yhat).all():
            raise ValueError(f"{row.group_id}: non-finite prediction data")
        mse = float(np.mean((y - yhat) ** 2))
        nmse = mse / float(np.var(y))
        pcc2 = float(np.corrcoef(y, yhat)[0, 1] ** 2)
        differences = {
            "MSE": abs(mse - float(row.test_MSE_dimensionless_squared)),
            "NMSE": abs(nmse - float(row.test_NMSE_dimensionless)),
            "NRMSE": abs(math.sqrt(nmse) - float(row.test_NRMSE_dimensionless)),
            "R2": abs((1.0 - nmse) - float(row.test_R2_dimensionless)),
            "PCC2": abs(pcc2 - float(row.test_PCC2_dimensionless)),
        }
        if max(differences.values()) > 2e-10:
            raise ValueError(f"{row.group_id}: metric mismatch {differences}")
        metric_checks.append(
            {
                "group_id": row.group_id,
                "sealed_test_n": len(test),
                "recomputed_MSE": mse,
                "max_absolute_metric_difference": max(differences.values()),
            }
        )

    by_group = metrics.set_index("group_id")
    r256 = float(by_group.loc["historical_R256", "test_MSE_dimensionless_squared"])
    r434 = float(by_group.loc["combined_completed_R434", "test_MSE_dimensionless_squared"])
    r512 = float(by_group.loc["complete_R512", "test_MSE_dimensionless_squared"])
    if abs(r256 - EXPECTED_R256_MSE) > 5e-10:
        raise ValueError("R256 reproduction mismatch")
    if abs(r434 - EXPECTED_R434_MSE) > 5e-10:
        raise ValueError("R434 reproduction mismatch")

    coverage = pd.read_csv(root / "drive_thermal_seed_coverage_R512.csv")
    seeds = np.sort(coverage["drive_thermal_seed"].astype(int).to_numpy())
    if not np.array_equal(seeds, np.arange(32345, 32857)):
        raise ValueError("coverage is not the contiguous R512 seed range")
    if not coverage["qa_status"].eq("pass").all():
        raise ValueError("coverage contains non-pass raw QA status")

    mean_path = root / "complete_R512" / "complete_R512_physical_state_mean_float32.npy"
    mean = np.load(mean_path, mmap_mode="r", allow_pickle=False)
    if mean.shape != (4000, 1024) or mean.dtype != np.float32:
        raise ValueError(f"R512 mean invalid: {mean.shape}/{mean.dtype}")
    if not np.isfinite(mean).all():
        raise ValueError("R512 mean contains NaN/Inf")
    mean_summary = json.loads(
        (root / "complete_R512" / "analysis_summary.json").read_text(encoding="utf-8")
    )
    if sha256_file(mean_path) != mean_summary["mean_feature_sha256"]:
        raise ValueError("R512 mean SHA256 mismatch")

    image_checks = []
    for name in (
        "dwso_300K_f7p3_physical_state_average_denoising_R512.png",
        "dwso_300K_f7p3_prediction_trace_R256_R434_R512.png",
    ):
        path = root / name
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
            if min(image.size) < 1000 or float(array.std()) < 5.0:
                raise ValueError(f"figure visual-content QA failed: {name}")
            image_checks.append(
                {
                    "path": str(path),
                    "width_px": image.size[0],
                    "height_px": image.size[1],
                    "rgb_std": float(array.std()),
                    "sha256": sha256_file(path),
                }
            )
    for stem in (
        "dwso_300K_f7p3_physical_state_average_denoising_R512",
        "dwso_300K_f7p3_prediction_trace_R256_R434_R512",
    ):
        for suffix in (".pdf", ".svg"):
            path = root / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty publication figure: {path}")

    notebook_path = root / "dwso_300K_f7p3_complete_R512_review.executed.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if not code_cells or any(cell.get("execution_count") is None for cell in code_cells):
        raise ValueError("notebook is not fully executed")
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if error_outputs:
        raise ValueError("executed notebook contains error output")

    artifact_rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"final_validation.json", "artifact_sha256.csv"}:
            continue
        artifact_rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    artifact_manifest = root / "artifact_sha256.csv"
    with artifact_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(artifact_rows[0]))
        writer.writeheader()
        writer.writerows(artifact_rows)

    validation = {
        "status": "pass",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "raw_QA_coverage": {
            "count": len(seeds),
            "min_seed": int(seeds.min()),
            "max_seed": int(seeds.max()),
            "contiguous_unique": True,
            "all_status_pass": True,
        },
        "state_matrix": {
            "shape": list(mean.shape),
            "dtype": str(mean.dtype),
            "finite": True,
            "sha256": sha256_file(mean_path),
        },
        "metric_recomputations": metric_checks,
        "R256_reproduction_pass": True,
        "R434_reproduction_pass": True,
        "R512_test_MSE_dimensionless_squared": r512,
        "R512_test_NMSE_dimensionless": float(
            by_group.loc["complete_R512", "test_NMSE_dimensionless"]
        ),
        "R512_test_NRMSE_dimensionless": float(
            by_group.loc["complete_R512", "test_NRMSE_dimensionless"]
        ),
        "relative_MSE_reduction_vs_R256": (r256 - r512) / r256,
        "relative_MSE_reduction_vs_R434": (r434 - r512) / r434,
        "figure_checks": image_checks,
        "notebook_execution": {
            "path": str(notebook_path),
            "code_cell_count": len(code_cells),
            "error_output_count": 0,
            "pass": True,
        },
        "artifact_manifest": str(artifact_manifest),
        "artifact_count": len(artifact_rows),
    }
    (root / "final_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

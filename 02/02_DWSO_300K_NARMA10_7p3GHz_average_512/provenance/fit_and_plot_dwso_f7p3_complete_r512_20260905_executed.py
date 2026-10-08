#!/usr/bin/env python3
"""Fit and plot the audited complete R=512 DWSO 7.3 GHz ensemble.

The script preserves the established floating-point aggregation order for the
previous R=256 and R=434 results, adds the final 78 strictly QA-passed seeds,
and fits the locked NARMA10 readout after physical-state averaging.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_SHAPE = (4000, 1024)
EXPECTED_INPUT_SHA256 = "76abb2456a37da29a86a756642a09fe81aa4eb175ee1a0f04a520789978bcb13"
EXPECTED_PROTOCOL_SHA256 = "9166ca51b53edeaca1dc178f1c217117bce1e752beba8dffccdf92764cbf7046"
EXPECTED_TARGET_SHA256 = "a1ed0375d41391305742eacffe84d2fcd63a81275e639c50503a45036ccbe319"
EXPECTED_R256_MSE = 0.0036902752417541564
EXPECTED_R434_MSE = 0.003567052440879413


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    paths: list[str] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-script", required=True, type=Path)
    parser.add_argument("--historical-sums", nargs=3, required=True, type=Path)
    parser.add_argument("--historical-audits", nargs=3, required=True, type=Path)
    parser.add_argument("--r178-sums", nargs=3, required=True, type=Path)
    parser.add_argument("--r178-audits", nargs=3, required=True, type=Path)
    parser.add_argument("--increment-sums", nargs=3, required=True, type=Path)
    parser.add_argument("--increment-audits", nargs=3, required=True, type=Path)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    parser.add_argument("--historical-summary-csv", required=True, type=Path)
    parser.add_argument("--historical-samples-csv", required=True, type=Path)
    parser.add_argument("--coverage-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    base_script = args.base_script.resolve()
    base = load_module("r434_characterization_base", base_script)
    historical_counts = [128, 48, 80]
    r178_counts = [48, 14, 116]
    increment_counts = [16, 2, 60]

    def load_parts(paths, audits, counts):
        values: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for path, audit, count in zip(paths, audits, counts):
            part, record = base.validate_sum(path, audit, count)
            values.append(np.asarray(part))
            records.append(record)
        return values, records

    historical_parts, historical_audits = load_parts(
        args.historical_sums, args.historical_audits, historical_counts
    )
    r178_parts, r178_audits = load_parts(args.r178_sums, args.r178_audits, r178_counts)
    increment_parts, increment_audits = load_parts(
        args.increment_sums, args.increment_audits, increment_counts
    )

    # Keep the already validated association used for R=256 and R=434.
    historical_sum = historical_parts[0] + (historical_parts[1] + historical_parts[2])
    r178_sum = (r178_parts[0] + r178_parts[1]) + r178_parts[2]
    r434_sum = historical_sum + r178_sum
    increment_sum = (increment_parts[0] + increment_parts[1]) + increment_parts[2]
    r512_sum = r434_sum + increment_sum
    for name, values in (
        ("R256", historical_sum),
        ("R178", r178_sum),
        ("R434", r434_sum),
        ("increment_R78", increment_sum),
        ("R512", r512_sum),
    ):
        if values.shape != EXPECTED_SHAPE or values.dtype != np.float64:
            raise ValueError(f"{name}: invalid shape/dtype {values.shape}/{values.dtype}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: NaN/Inf detected")

    analyzer = args.analyzer.resolve()
    input_csv = args.input_csv.resolve()
    analyzer_hash = sha256_file(analyzer)
    input_hash = sha256_file(input_csv)
    if analyzer_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(f"analyzer SHA256 mismatch: {analyzer_hash}")
    if input_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(f"input SHA256 mismatch: {input_hash}")
    protocol = base.load_protocol(analyzer)
    labels = protocol.read_narma10_labels(input_csv, 4000)
    effective_symbols = np.where(labels["valid_target"])[0]
    effective_symbols = effective_symbols[(effective_symbols >= 50) & (effective_symbols < 3999)]
    effective_target = np.ascontiguousarray(labels["target_y_next"][effective_symbols])
    target_hash = hashlib.sha256(effective_target.tobytes(order="C")).hexdigest()
    if effective_target.shape != (3949,) or target_hash != EXPECTED_TARGET_SHA256:
        raise ValueError("effective target QA failed")

    groups = [
        ("historical_R256", 256, historical_sum / 256.0, "32345-32600"),
        ("combined_completed_R434", 434, r434_sum / 434.0, "32345-32804 with prior completed-set scope"),
        ("final_increment_R78", 78, increment_sum / 78.0, "78 final QA-pass realizations"),
        ("complete_R512", 512, r512_sum / 512.0, "32345-32856, contiguous"),
    ]
    summaries: list[dict[str, Any]] = []
    traces: dict[str, dict[str, np.ndarray]] = {}
    seed_descriptions: list[str] = []
    for label, count, mean, seed_description in groups:
        summary, trace = base.fit_one(output_dir, label, count, mean, protocol, labels)
        summaries.append(summary)
        traces[label] = trace
        seed_descriptions.append(seed_description)

    metrics_rows: list[dict[str, Any]] = []
    for summary, seed_description in zip(summaries, seed_descriptions):
        test = summary["metrics"]["test"]
        metrics_rows.append(
            {
                "group_id": summary["label"],
                "repeat_count_R": summary["repeat_count_R"],
                "drive_thermal_seeds": seed_description,
                "best_alpha": summary["best_alpha"],
                "test_MSE_dimensionless_squared": float(test["RMSE"]) ** 2,
                "test_NMSE_dimensionless": test["NMSE"],
                "test_NRMSE_dimensionless": test["NRMSE"],
                "test_R2_dimensionless": test["R2"],
                "test_PCC2_dimensionless": test["PCC2"],
                "sealed_test_n": summary["split"]["test_n"],
            }
        )
    metrics_path = output_dir / "dwso_300K_f7p3_complete_R512_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)

    mse = {row["group_id"]: float(row["test_MSE_dimensionless_squared"]) for row in metrics_rows}
    r256_diff = abs(mse["historical_R256"] - EXPECTED_R256_MSE)
    r434_diff = abs(mse["combined_completed_R434"] - EXPECTED_R434_MSE)
    if r256_diff > 5e-10:
        raise ValueError(f"R256 reproduction failed: diff={r256_diff}")
    if r434_diff > 5e-10:
        raise ValueError(f"R434 reproduction failed: diff={r434_diff}")

    summary_rows = [
        row for row in read_csv(args.historical_summary_csv.resolve()) if row["device"] == "DWSO"
    ]
    sample_rows = [
        row for row in read_csv(args.historical_samples_csv.resolve()) if row["device"] == "DWSO"
    ]
    curve_x = [int(row["repeat_count_R"]) for row in summary_rows]
    curve_y = [float(row["mean_test_MSE_dimensionless_squared"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    for repeat in sorted({int(row["repeat_count"]) for row in sample_rows}):
        values = [
            float(row["test_MSE_dimensionless_squared"])
            for row in sample_rows
            if int(row["repeat_count"]) == repeat
        ]
        ax.scatter(
            np.full(len(values), repeat, dtype=float),
            values,
            s=36,
            facecolors="white",
            edgecolors="#2878B5",
            linewidths=1.25,
            alpha=0.76,
            label="Disjoint-group results" if repeat == 1 else None,
            zorder=2,
        )
    ax.plot(
        curve_x + [434, 512],
        curve_y + [mse["combined_completed_R434"], mse["complete_R512"]],
        color="#2878B5",
        marker="o",
        linewidth=2.6,
        markersize=6.5,
        label="Physical-state average before readout",
        zorder=3,
    )
    ax.scatter(
        [78],
        [mse["final_increment_R78"]],
        marker="D",
        s=72,
        facecolors="white",
        edgecolors="#E37A22",
        linewidths=1.8,
        label="Final added cohort only (R=78)",
        zorder=4,
    )
    for repeat, label in ((434, "combined_completed_R434"), (512, "complete_R512")):
        ax.annotate(
            f"R={repeat}\nMSE={mse[label]:.6f}",
            xy=(repeat, mse[label]),
            xytext=(-84, 24 if repeat == 512 else -42),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "#555555"},
            fontsize=9,
        )
    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32, 64, 78, 128, 256, 434, 512]
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.set_yscale("log")
    ax.set_xlabel("Number of independent thermal realizations, R")
    ax.set_ylabel(r"Sealed-test MSE (NARMA10 output$^2$; dimensionless$^2$)")
    ax.set_title("300 K DWSO NARMA10 physical-state averaging at 7.3 GHz", fontsize=16, pad=24)
    ax.text(
        0.5,
        1.01,
        r"scalar $m_z$; 8 phases; 8$\times$8 pooling; N=4000; target $y[n+1]$; sealed test n=780",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#555555",
        fontsize=10,
    )
    ax.grid(True, which="major", color="#D8DEE5", linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color="#EDF0F2", linewidth=0.55)
    ax.legend(frameon=False, loc="best")
    denoise_paths = save_figure(
        fig, output_dir / "dwso_300K_f7p3_physical_state_average_denoising_R512"
    )

    trace256 = traces["historical_R256"]
    trace434 = traces["combined_completed_R434"]
    trace512 = traces["complete_R512"]
    window = slice(0, 240)
    fig, ax = plt.subplots(figsize=(12.2, 5.9), constrained_layout=True)
    ax.plot(
        trace512["symbol"][window],
        trace512["target"][window],
        color="#202020",
        linewidth=1.8,
        label="Target y[n+1]",
    )
    ax.plot(
        trace256["symbol"][window],
        trace256["prediction"][window],
        color="#E37A22",
        linewidth=1.15,
        alpha=0.75,
        label=f"R=256 (MSE={mse['historical_R256']:.6f})",
    )
    ax.plot(
        trace434["symbol"][window],
        trace434["prediction"][window],
        color="#6E9F4B",
        linewidth=1.25,
        alpha=0.9,
        label=f"R=434 (MSE={mse['combined_completed_R434']:.6f})",
    )
    ax.plot(
        trace512["symbol"][window],
        trace512["prediction"][window],
        color="#2878B5",
        linewidth=1.6,
        label=f"R=512 (MSE={mse['complete_R512']:.6f})",
    )
    ax.set_xlabel("Symbol index n")
    ax.set_ylabel("NARMA10 output (dimensionless)")
    ax.set_title("300 K DWSO 7.3 GHz: sealed-test prediction after state averaging", fontsize=15)
    ax.grid(True, color="#E1E5E8", linewidth=0.7)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    trace_paths = save_figure(
        fig, output_dir / "dwso_300K_f7p3_prediction_trace_R256_R434_R512"
    )

    coverage_path = args.coverage_csv.resolve()
    coverage_rows = read_csv(coverage_path)
    coverage_seeds = sorted(int(row["drive_thermal_seed"]) for row in coverage_rows)
    if coverage_seeds != list(range(32345, 32857)):
        raise ValueError("coverage CSV is not the contiguous seed range 32345-32856")
    if any(row.get("qa_status") != "pass" for row in coverage_rows):
        raise ValueError("coverage CSV contains non-pass rows")
    inventory_path = output_dir / "drive_thermal_seed_coverage_R512.csv"
    inventory_path.write_bytes(coverage_path.read_bytes())

    figure_qa = {
        "status": "pass",
        "generated_at_utc": now_utc(),
        "chart_contract": {
            "primary_question": "How does aligned physical-state averaging change sealed-test NARMA10 MSE?",
            "x_axis": "independent thermal realization count R; base-2 log scale; no horizontal jitter",
            "y_axis": "sealed-test MSE, dimensionless squared; log scale",
            "comparison": "historical disjoint groups plus exact R256, R434 and complete R512 means",
        },
        "denoising_figures": denoise_paths,
        "prediction_figures": trace_paths,
        "all_exist_nonempty": all(
            Path(path).is_file() and Path(path).stat().st_size > 0
            for path in denoise_paths + trace_paths
        ),
    }
    write_json(output_dir / "figure_qa.json", figure_qa)

    top_qa = {
        "status": "pass",
        "generated_at_utc": now_utc(),
        "scientific_scope": "complete 300 K DWSO 7.3 GHz NARMA10 thermal ensemble R=512",
        "drive_thermal_seed_range": [32345, 32856],
        "drive_thermal_seed_count": 512,
        "coverage_is_contiguous_unique": True,
        "input_sha256": input_hash,
        "protocol_sha256": analyzer_hash,
        "effective_target_sha256": target_hash,
        "R256_reproduction": {
            "expected_MSE": EXPECTED_R256_MSE,
            "observed_MSE": mse["historical_R256"],
            "absolute_difference": r256_diff,
            "tolerance": 5e-10,
            "pass": True,
        },
        "R434_reproduction": {
            "expected_MSE": EXPECTED_R434_MSE,
            "observed_MSE": mse["combined_completed_R434"],
            "absolute_difference": r434_diff,
            "tolerance": 5e-10,
            "pass": True,
        },
        "feature_protocol": "scalar mz, layer0, 8 phases, non-overlapping 8x8 arithmetic-mean pooling, 1024 nodes",
        "state_average_order": "per-seed OVF -> pooled aligned state -> arithmetic average across R -> one locked NARMA10 readout",
        "target_alignment": "state after symbol n predicts y[n+1]",
        "split_protocol": "washout 50; chronological 60/20/20; purge gap 10; train-only standardization; validation-selected alpha",
        "metrics_csv": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "coverage_csv": str(inventory_path),
        "coverage_sha256": sha256_file(inventory_path),
        "source_aggregate_audits": historical_audits + r178_audits + increment_audits,
        "figure_qa": figure_qa,
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "base_script": str(base_script),
        "base_script_sha256": sha256_file(base_script),
        "command_used": " ".join(sys.argv),
    }
    write_json(output_dir / "summary_qa.json", top_qa)

    paths = [
        "300 K DWSO 7.3 GHz complete R=512 characterization audit paths",
        f"generated_at_utc: {top_qa['generated_at_utc']}",
        "",
        f"locked_input_csv: {input_csv}",
        f"locked_analyzer: {analyzer}",
        f"metrics_csv: {metrics_path}",
        f"summary_qa: {output_dir / 'summary_qa.json'}",
        f"coverage_csv: {inventory_path}",
        "",
        "aggregate_audits:",
        *[str(path.resolve()) for path in args.historical_audits],
        *[str(path.resolve()) for path in args.r178_audits],
        *[str(path.resolve()) for path in args.increment_audits],
    ]
    (output_dir / "audit_paths.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")

    improvement_434_to_512 = (
        mse["combined_completed_R434"] - mse["complete_R512"]
    ) / mse["combined_completed_R434"]
    readme = f"""# 300 K DWSO 7.3 GHz complete R=512 characterization

The complete, contiguous drive-thermal-seed set 32345-32856 passed strict raw-output QA before characterization. Each realization contributes scalar `mz` from layer 0 at 8 phase samples. Every 64x128 state is pooled by non-overlapping 8x8 arithmetic means to 1024 reservoir nodes. Aligned physical states are then averaged across all R=512 realizations before one locked linear-ridge NARMA10 readout is fitted.

Locked task: N=4000, input seed 20260707, state after symbol n predicts y[n+1], washout 50, chronological 60/20/20 split, purge gap 10, training-only standardization, validation-selected alpha, sealed test n=780.

Test results (dimensionless squared MSE):

- R=256: {mse['historical_R256']:.12g}
- R=434: {mse['combined_completed_R434']:.12g}
- R=512: {mse['complete_R512']:.12g}
- Relative MSE change R=434 to R=512: {improvement_434_to_512:+.3%} (positive means improvement)

See `summary_qa.json`, `dwso_300K_f7p3_complete_R512_metrics.csv`, and `audit_paths.txt`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metrics_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

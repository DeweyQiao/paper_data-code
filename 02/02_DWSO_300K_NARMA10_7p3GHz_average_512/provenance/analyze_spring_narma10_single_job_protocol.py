#!/usr/bin/env python3
"""Analyze one completed high-frequency NARMA10 MuMax3 job."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
MC_ANALYZER = HERE / "analyze_spring_mc_single_job_20260630.py"
if not MC_ANALYZER.is_file():
    MC_ANALYZER = ROOT / "remote_scripts" / "analyze_spring_mc_single_job_20260630.py"
ALPHA_GRID = np.logspace(-10, 4, 29)


def load_mc_analyzer():
    spec = importlib.util.spec_from_file_location("spring_mc_analyzer_for_narma10", MC_ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MC analyzer helpers: {MC_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MC = load_mc_analyzer()


def now_utc() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_narma10_labels(path: Path, expected_n: int | None = None) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"n", "u_narma", "drive_u_centered", "y_narma", "target_y_next", "valid_target"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        rows = list(reader)
    if expected_n is not None and len(rows) != expected_n:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected_n}")
    n = np.asarray([int(row["n"]) for row in rows], dtype=np.int64)
    if not np.array_equal(n, np.arange(n.size, dtype=np.int64)):
        raise ValueError(f"{path} has non-contiguous n indices")
    u = np.asarray([float(row["u_narma"]) for row in rows], dtype=np.float64)
    drive_u = np.asarray([float(row["drive_u_centered"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["y_narma"]) for row in rows], dtype=np.float64)
    target = np.asarray(
        [float(row["target_y_next"]) if row["target_y_next"] else np.nan for row in rows],
        dtype=np.float64,
    )
    valid = np.asarray([int(row["valid_target"]) == 1 for row in rows], dtype=bool)
    for name, arr in [("u_narma", u), ("drive_u_centered", drive_u), ("y_narma", y)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{path} contains NaN/Inf in {name}")
    if np.nanmin(u) < 0.0 or np.nanmax(u) > 0.5:
        raise ValueError("NARMA10 input u_narma is outside [0, 0.5]")
    if np.nanmin(drive_u) < -1.0 or np.nanmax(drive_u) > 1.0:
        raise ValueError("drive_u_centered is outside [-1, 1]")
    return {
        "n": n,
        "u_narma": u,
        "drive_u_centered": drive_u,
        "y_narma": y,
        "target_y_next": target,
        "valid_target": valid,
    }


def split_indices(n_rows: int, purge_gap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(0.60 * n_rows)
    val_start = min(n_rows, train_end + purge_gap)
    val_end = int(0.80 * n_rows)
    if val_end - val_start < 20:
        val_start = train_end
    test_start = min(n_rows, val_end + purge_gap)
    if n_rows - test_start < 20:
        test_start = val_end
    train = np.arange(0, train_end, dtype=np.int64)
    val = np.arange(val_start, val_end, dtype=np.int64)
    test = np.arange(test_start, n_rows, dtype=np.int64)
    if train.size < 30 or val.size < 20 or test.size < 20:
        raise ValueError(f"bad split sizes: train={train.size}, val={val.size}, test={test.size}")
    return train, val, test


def centered_kernel(
    kernel: np.ndarray,
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    train_rows: np.ndarray,
) -> np.ndarray:
    base = kernel[np.ix_(rows_a, rows_b)]
    a_mu = kernel[np.ix_(rows_a, train_rows)].mean(axis=1)[:, None]
    mu_b = kernel[np.ix_(train_rows, rows_b)].mean(axis=0)[None, :]
    mu_mu = float(kernel[np.ix_(train_rows, train_rows)].mean())
    return base - a_mu - mu_b + mu_mu


def pcc2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return math.nan
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = float(np.sqrt(np.sum(yt * yt) * np.sum(yp * yp)))
    if denom <= 0.0 or not np.isfinite(denom):
        return math.nan
    value = float(np.sum(yt * yp) / denom)
    return value * value


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0.0:
        return math.nan
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = float(np.var(y_true))
    if var <= 0.0 or not np.isfinite(var):
        return math.nan
    return mse(y_true, y_pred) / var


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    value_nmse = nmse(y_true, y_pred)
    return {
        "NMSE": value_nmse,
        "NRMSE": math.sqrt(value_nmse) if np.isfinite(value_nmse) and value_nmse >= 0.0 else math.nan,
        "R2": r2_score(y_true, y_pred),
        "PCC2": pcc2(y_true, y_pred),
        "RMSE": math.sqrt(mse(y_true, y_pred)),
    }


def prepare_kernel(
    features: np.ndarray,
    train_rows: np.ndarray,
    std_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = features[train_rows].astype(np.float64, copy=False)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    active = np.isfinite(std) & (std > std_floor)
    if int(active.sum()) == 0:
        raise ValueError("all feature columns are constant or non-finite")
    z = (features[:, active].astype(np.float64, copy=False) - mean[active]) / std[active]
    if not np.isfinite(z).all():
        raise ValueError("standardized feature matrix contains NaN or Inf")
    z = z.astype(np.float32, copy=False)
    kernel = (z @ z.T).astype(np.float64, copy=False) / float(active.sum())
    return kernel, {
        "standardization": "train-set mean/std only",
        "std_floor": float(std_floor),
        "active_feature_columns": int(active.sum()),
        "dropped_constant_feature_columns": int(active.size - active.sum()),
        "kernel_shape": [int(kernel.shape[0]), int(kernel.shape[1])],
    }


def predict_from_eigh(
    kernel_eval: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    y_centered: np.ndarray,
    y_mean: float,
    alpha: float,
) -> np.ndarray:
    coeff = eigvecs @ ((eigvecs.T @ y_centered) / (eigvals + alpha))
    return kernel_eval @ coeff + y_mean


def fit_kernel_ridge(
    features: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    std_floor: float,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    kernel, kernel_info = prepare_kernel(features, train_rows, std_floor)
    k_train = centered_kernel(kernel, train_rows, train_rows, train_rows)
    k_val = centered_kernel(kernel, val_rows, train_rows, train_rows)
    k_test = centered_kernel(kernel, test_rows, train_rows, train_rows)
    k_all = centered_kernel(kernel, np.arange(features.shape[0], dtype=np.int64), train_rows, train_rows)

    y_train = target[train_rows]
    y_val = target[val_rows]
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    eigvals, eigvecs = np.linalg.eigh(k_train.astype(np.float64, copy=False))
    eigvals = np.maximum(eigvals, 0.0)

    best_alpha = float(ALPHA_GRID[len(ALPHA_GRID) // 2])
    best_nmse = math.inf
    alpha_rows: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        pred_val = predict_from_eigh(k_val, eigvals, eigvecs, y_centered, y_mean, float(alpha))
        val_metrics = metrics(y_val, pred_val)
        alpha_rows.append({"alpha": float(alpha), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if np.isfinite(val_metrics["NMSE"]) and val_metrics["NMSE"] < best_nmse:
            best_alpha = float(alpha)
            best_nmse = float(val_metrics["NMSE"])

    pred_all = predict_from_eigh(k_all, eigvals, eigvecs, y_centered, y_mean, best_alpha)
    result = {
        "alpha": best_alpha,
        "train": metrics(target[train_rows], pred_all[train_rows]),
        "val": metrics(target[val_rows], pred_all[val_rows]),
        "test": metrics(target[test_rows], pred_all[test_rows]),
        "split": {
            "train_n": int(train_rows.size),
            "val_n": int(val_rows.size),
            "test_n": int(test_rows.size),
            "purge_gap_rows": int(min(val_rows[0] - train_rows[-1] - 1, test_rows[0] - val_rows[-1] - 1)),
        },
        "alpha_grid_metrics": alpha_rows,
    }
    return result, pred_all, kernel_info


def write_predictions(
    path: Path,
    symbols: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    split_label: np.ndarray,
    labels: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "symbol_n",
                "split",
                "u_narma",
                "drive_u_centered",
                "target_y_next",
                "pred_y_next",
                "residual",
            ]
        )
        for row, symbol in enumerate(symbols):
            writer.writerow(
                [
                    row,
                    int(symbol),
                    split_label[row],
                    f"{labels['u_narma'][symbol]:.12g}",
                    f"{labels['drive_u_centered'][symbol]:.12g}",
                    f"{target[row]:.12g}",
                    f"{pred[row]:.12g}",
                    f"{target[row] - pred[row]:.12g}",
                ]
            )


def write_alpha_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(
    out_dir: Path,
    symbols: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    split_label: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(symbols, target, lw=1.3, label="target y[n+1]")
    ax.plot(symbols, pred, lw=1.1, label="prediction")
    for label, color in [("train", "#e8eef5"), ("val", "#fff3cd"), ("test", "#e7f5e8")]:
        mask = split_label == label
        if mask.any():
            ax.axvspan(symbols[mask][0], symbols[mask][-1], color=color, alpha=0.45, lw=0)
    ax.set_xlabel("symbol n")
    ax.set_ylabel("NARMA10 output y[n+1] (dimensionless)")
    ax.set_title(f"{title}: NARMA10 prediction")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "narma10_prediction_trace.png", dpi=220)
    fig.savefig(out_dir / "narma10_prediction_trace.pdf")
    plt.close(fig)

    test_mask = split_label == "test"
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(target[test_mask], pred[test_mask], s=16, alpha=0.8)
    lo = float(min(np.min(target[test_mask]), np.min(pred[test_mask])))
    hi = float(max(np.max(target[test_mask]), np.max(pred[test_mask])))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("target y[n+1]")
    ax.set_ylabel("predicted y[n+1]")
    ax.set_title(f"{title}: test scatter")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "narma10_test_scatter.png", dpi=220)
    fig.savefig(out_dir / "narma10_test_scatter.pdf")
    plt.close(fig)


def analyze_one(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = args.raw_dir.resolve()
    input_csv = args.input_csv.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists and is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    job_metadata: dict[str, Any] = {}
    if args.job_metadata:
        with args.job_metadata.resolve().open("r", encoding="utf-8") as handle:
            job_metadata = json.load(handle)
    expected_symbols = int(job_metadata.get("n_symbols", args.n_symbols))

    labels = read_narma10_labels(input_csv, expected_symbols)
    file_grid = MC.index_readout_files(raw_dir, expected_symbols, args.phase_count)
    phase_indices = None
    if args.phase_indices is not None:
        phase_indices = [int(item.strip()) for item in args.phase_indices.split(",") if item.strip()]
    features, centers_nm, feature_info = MC.build_features(
        file_grid,
        args.feature_mode,
        args.component,
        phase_indices=phase_indices,
    )
    if features.shape[0] != expected_symbols:
        raise ValueError(f"feature row mismatch: {features.shape[0]} != {expected_symbols}")

    valid_symbols = np.where(labels["valid_target"])[0]
    valid_symbols = valid_symbols[valid_symbols >= args.washout]
    valid_symbols = valid_symbols[valid_symbols < expected_symbols - 1]
    if valid_symbols.size < 120:
        raise ValueError(f"too few effective NARMA10 rows after washout: {valid_symbols.size}")
    target = labels["target_y_next"][valid_symbols]
    if not np.isfinite(target).all():
        raise ValueError("target_y_next contains NaN/Inf after filtering")
    if float(np.var(target)) <= 0.0:
        raise ValueError("target_y_next has zero variance after filtering")
    x_eff = features[valid_symbols]
    train_rows, val_rows, test_rows = split_indices(x_eff.shape[0], args.purge_gap)
    result, pred_all, kernel_info = fit_kernel_ridge(
        x_eff,
        target,
        train_rows,
        val_rows,
        test_rows,
        args.std_floor,
    )

    split_label = np.full(valid_symbols.size, "unused", dtype=object)
    split_label[train_rows] = "train"
    split_label[val_rows] = "val"
    split_label[test_rows] = "test"

    write_predictions(out_dir / "narma10_predictions.csv", valid_symbols, target, pred_all, split_label, labels)
    write_alpha_csv(out_dir / "ridge_alpha_validation.csv", result["alpha_grid_metrics"])
    MC.write_center_csv(out_dir / "dw_center_by_symbol_phase.csv", centers_nm)

    title = title_from_metadata(job_metadata, raw_dir.name)
    if not args.no_plots:
        plot_outputs(out_dir, valid_symbols, target, pred_all, split_label, title)

    summary = {
        "job_id": job_metadata.get("job_id", raw_dir.name),
        "analysis_generated_at": now_utc(),
        "raw_dir": str(raw_dir),
        "input_csv": str(input_csv),
        "job_metadata": job_metadata,
        "task": "NARMA10 readout",
        "target_alignment": "reservoir state after symbol n predicts target_y_next = y[n+1]",
        "washout_symbols": int(args.washout),
        "effective_symbols": {
            "start": int(valid_symbols[0]),
            "stop": int(valid_symbols[-1]),
            "count": int(valid_symbols.size),
        },
        "readout": {
            "model": "kernel ridge regression on standardized reservoir features",
            "alpha_selection": "minimize validation NMSE",
            "best_alpha": result["alpha"],
            "feature_mode": args.feature_mode,
            "component_used": args.component,
            "phase_count": args.phase_count,
            "phase_indices": phase_indices if phase_indices is not None else list(range(args.phase_count)),
        },
        "feature_info": feature_info,
        "kernel_info": kernel_info,
        "split": result["split"],
        "metrics": {
            "train": result["train"],
            "val": result["val"],
            "test": result["test"],
        },
        "target_summary": {
            "mean": float(np.mean(target)),
            "std": float(np.std(target)),
            "min": float(np.min(target)),
            "max": float(np.max(target)),
        },
        "input_summary": {
            "u_narma_min": float(np.min(labels["u_narma"])),
            "u_narma_max": float(np.max(labels["u_narma"])),
            "drive_u_centered_min": float(np.min(labels["drive_u_centered"])),
            "drive_u_centered_max": float(np.max(labels["drive_u_centered"])),
        },
        "outputs": {
            "predictions_csv": str(out_dir / "narma10_predictions.csv"),
            "alpha_validation_csv": str(out_dir / "ridge_alpha_validation.csv"),
            "prediction_trace_png": str(out_dir / "narma10_prediction_trace.png") if not args.no_plots else "",
            "test_scatter_png": str(out_dir / "narma10_test_scatter.png") if not args.no_plots else "",
            "dw_center_csv": str(out_dir / "dw_center_by_symbol_phase.csv"),
        },
    }
    with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (out_dir / "narma10_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "job_id",
            "oscillator",
            "frequency_GHz",
            "current_amplitude_A_per_m2",
            "best_alpha",
            "train_NMSE",
            "val_NMSE",
            "test_NMSE",
            "test_NRMSE",
            "test_R2",
            "test_PCC2",
            "effective_rows",
            "active_feature_columns",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "job_id": summary["job_id"],
                "oscillator": job_metadata.get("oscillator", ""),
                "frequency_GHz": job_metadata.get("frequency_GHz", ""),
                "current_amplitude_A_per_m2": job_metadata.get("current_amplitude_A_per_m2", ""),
                "best_alpha": result["alpha"],
                "train_NMSE": result["train"]["NMSE"],
                "val_NMSE": result["val"]["NMSE"],
                "test_NMSE": result["test"]["NMSE"],
                "test_NRMSE": result["test"]["NRMSE"],
                "test_R2": result["test"]["R2"],
                "test_PCC2": result["test"]["PCC2"],
                "effective_rows": valid_symbols.size,
                "active_feature_columns": kernel_info["active_feature_columns"],
            }
        )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return summary


def analyze_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    plan_path = run_dir / "job_plan.tsv"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    rows: list[dict[str, str]]
    with plan_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    summaries: list[dict[str, Any]] = []
    for row in rows:
        raw_dir = Path(row["raw_output_dir"])
        if not raw_dir.is_dir():
            print(f"skip missing raw dir: {raw_dir}", file=sys.stderr)
            continue
        out_dir = run_dir / "analysis_narma10_mz_phase8" / row["job_id"]
        one_args = argparse.Namespace(
            raw_dir=raw_dir,
            input_csv=Path(row["input_csv_local"]),
            job_metadata=Path(row["metadata_local"]),
            out_dir=out_dir,
            feature_mode=args.feature_mode,
            component=args.component,
            phase_count=args.phase_count,
            phase_indices=args.phase_indices,
            n_symbols=args.n_symbols,
            washout=args.washout,
            purge_gap=args.purge_gap,
            std_floor=args.std_floor,
            overwrite=args.overwrite,
        )
        summaries.append(analyze_one(one_args))

    if summaries:
        combined_path = run_dir / "analysis_narma10_mz_phase8" / "narma10_combined_metrics.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "job_id",
            "oscillator",
            "frequency_GHz",
            "current_amplitude_A_per_m2",
            "best_alpha",
            "test_NMSE",
            "test_NRMSE",
            "test_R2",
            "test_PCC2",
            "effective_rows",
            "active_feature_columns",
            "analysis_dir",
        ]
        with combined_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                meta = summary["job_metadata"]
                writer.writerow(
                    {
                        "job_id": summary["job_id"],
                        "oscillator": meta.get("oscillator", ""),
                        "frequency_GHz": meta.get("frequency_GHz", ""),
                        "current_amplitude_A_per_m2": meta.get("current_amplitude_A_per_m2", ""),
                        "best_alpha": summary["readout"]["best_alpha"],
                        "test_NMSE": summary["metrics"]["test"]["NMSE"],
                        "test_NRMSE": summary["metrics"]["test"]["NRMSE"],
                        "test_R2": summary["metrics"]["test"]["R2"],
                        "test_PCC2": summary["metrics"]["test"]["PCC2"],
                        "effective_rows": summary["effective_symbols"]["count"],
                        "active_feature_columns": summary["kernel_info"]["active_feature_columns"],
                        "analysis_dir": str(run_dir / "analysis_narma10_mz_phase8" / str(summary["job_id"])),
                    }
                )
        print(combined_path)
    return 0


def title_from_metadata(job_metadata: dict[str, Any], fallback: str) -> str:
    try:
        oscillator = str(job_metadata["oscillator"])
        freq = float(job_metadata["frequency_GHz"])
        current = float(job_metadata["current_amplitude_A_per_m2"])
    except (KeyError, TypeError, ValueError):
        return fallback
    return f"{oscillator}, f={freq:g} GHz, J={current:.1e} A/m2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_one = sub.add_parser("one", help="Analyze one completed job.")
    p_one.add_argument("--raw-dir", required=True, type=Path)
    p_one.add_argument("--input-csv", required=True, type=Path)
    p_one.add_argument("--job-metadata", type=Path)
    p_one.add_argument("--out-dir", required=True, type=Path)
    p_one.set_defaults(func=lambda args: analyze_one(args) and 0)

    p_run = sub.add_parser("run", help="Analyze every completed job in a run directory.")
    p_run.add_argument("--run-dir", required=True, type=Path)
    p_run.set_defaults(func=analyze_run)

    for p in (p_one, p_run):
        p.add_argument("--feature-mode", choices=["component_full", "mxyz_full", "dw_center"], default="component_full")
        p.add_argument("--component", choices=["mx", "my", "mz"], default="mz")
        p.add_argument("--phase-count", type=int, default=8)
        p.add_argument("--phase-indices", type=str, default=None)
        p.add_argument("--n-symbols", type=int, default=500)
        p.add_argument("--washout", type=int, default=50)
        p.add_argument("--purge-gap", type=int, default=10)
        p.add_argument("--std-floor", type=float, default=1e-7)
        p.add_argument("--no-plots", action="store_true")
        p.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

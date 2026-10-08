#!/usr/bin/env python3
"""Reproduce the NARMA10 linear readout for the R=512 mean DWSO state.

For each thermal realization, a 4000 x 1024 state matrix was extracted from
32,000 scalar-mz OVF files using layer 0, eight phases per symbol, and
non-overlapping 8 x 8 spatial averaging at each phase. The 512 aligned matrices
for thermal seeds 32345--32856 were then averaged elementwise with equal weights
before training the linear ridge readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_SYMBOLS = 4000
N_FEATURES = 1024
WASHOUT = 50
PURGE_GAP = 10
STD_FLOOR = 1.0e-7
ALPHA_GRID = np.logspace(-10.0, 4.0, 29)
EXPECTED_BEST_ALPHA = 0.03162277660168379
EXPECTED_TEST_MSE = 0.0034656418745736477
EXPECTED_TEST_NMSE = 0.22060580621777207
EXPECTED_STATE_SHA256 = "513a97bda97869a8379082c6d5bf5d1acef025d36b4b23ebb717a682d6ee06a9"
EXPECTED_LABELS_SHA256 = "76abb2456a37da29a86a756642a09fe81aa4eb175ee1a0f04a520789978bcb13"
EXPECTED_PREDICTIONS_SHA256 = "ecc96f6443f34a0d6ad1faf14a47c09aa2082699f2b516354936048225eb68d5"
EXPECTED_ALPHA_GRID_SHA256 = "f5ab118d15dfe5abc7acc94033c51e76b1fb9176857f1bdabeceb28ac732a6fb"
# Eigensolver/BLAS implementations can differ slightly for this ill-conditioned
# dual problem. The archived Linux predictions remain the exact reference and
# are checked independently.
MSE_RTOL = 1.0e-5
MSE_ATOL = 5.0e-9


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the locked chronological 60%/20%/20% split with two 10-row purge gaps."""
    train_end = int(0.60 * n_rows)
    val_start = train_end + PURGE_GAP
    val_end = int(0.80 * n_rows)
    test_start = val_end + PURGE_GAP
    train = np.arange(0, train_end, dtype=np.int64)
    validation = np.arange(val_start, val_end, dtype=np.int64)
    test = np.arange(test_start, n_rows, dtype=np.int64)
    require((train.size, validation.size, test.size) == (2369, 780, 780), "Incorrect data-split lengths")
    return train, validation, test


def centered_kernel(
    kernel: np.ndarray,
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    train_rows: np.ndarray,
) -> np.ndarray:
    base = kernel[np.ix_(rows_a, rows_b)]
    a_mean = kernel[np.ix_(rows_a, train_rows)].mean(axis=1)[:, None]
    mean_b = kernel[np.ix_(train_rows, rows_b)].mean(axis=0)[None, :]
    grand_mean = float(kernel[np.ix_(train_rows, train_rows)].mean())
    return base - a_mean - mean_b + grand_mean


def predict_dual(
    kernel_eval: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    centered_target: np.ndarray,
    target_mean: float,
    alpha: float,
) -> np.ndarray:
    coefficients = eigenvectors @ (
        (eigenvectors.T @ centered_target) / (eigenvalues + alpha)
    )
    return kernel_eval @ coefficients + target_mean


def metric_dict(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    mse = float(np.mean(residual * residual))
    variance = float(np.var(target))
    correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "NMSE": mse / variance,
        "NRMSE": math.sqrt(mse / variance),
        "R2": 1.0 - mse / variance,
        "PCC2": correlation * correlation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).resolve().parent
    parser.add_argument("--states", type=Path, default=package / "data" / "reservoir_states_R512_mean_mz_layer0_phase8_pool8x8_float32.npy")
    parser.add_argument("--labels", type=Path, default=package / "data" / "narma10_labels.csv")
    parser.add_argument("--published-predictions", type=Path, default=package / "data" / "narma10_predictions_locked_R512.csv")
    parser.add_argument("--published-validation-grid", type=Path, default=package / "data" / "ridge_alpha_validation_locked_R512.csv")
    parser.add_argument("--output-dir", type=Path, default=package / "outputs")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    for path in (args.states, args.labels, args.published_predictions, args.published_validation_grid):
        if not path.is_file():
            raise FileNotFoundError(path)

    require(sha256_file(args.states) == EXPECTED_STATE_SHA256, "SHA-256 mismatch for the R=512 mean-state file")
    require(sha256_file(args.labels) == EXPECTED_LABELS_SHA256, "SHA-256 mismatch for the NARMA10 label file")
    require(sha256_file(args.published_predictions) == EXPECTED_PREDICTIONS_SHA256, "SHA-256 mismatch for the locked R=512 predictions")
    require(sha256_file(args.published_validation_grid) == EXPECTED_ALPHA_GRID_SHA256, "SHA-256 mismatch for the locked R=512 validation grid")

    states = np.load(args.states, mmap_mode="r", allow_pickle=False)
    require(states.shape == (N_SYMBOLS, N_FEATURES), f"Incorrect reservoir shape: {states.shape}")
    require(states.dtype == np.float32, f"Incorrect reservoir dtype: {states.dtype}")
    require(np.isfinite(states).all(), "The reservoir states contain NaN or Inf")

    labels = pd.read_csv(args.labels)
    required_columns = {"n", "u_narma", "drive_u_centered", "target_y_next", "valid_target"}
    require(required_columns.issubset(labels.columns), f"Missing label columns: {required_columns - set(labels.columns)}")
    require(len(labels) == N_SYMBOLS, f"Incorrect number of label rows: {len(labels)}")
    symbol_n = labels["n"].to_numpy(np.int64)
    require(np.array_equal(symbol_n, np.arange(N_SYMBOLS)), "n is not the consecutive sequence 0..3999")
    u_narma = labels["u_narma"].to_numpy(np.float64)
    drive = labels["drive_u_centered"].to_numpy(np.float64)
    require(np.allclose(drive, 4.0 * u_narma - 1.0, rtol=0.0, atol=1.0e-12), "The input mapping drive=4u-1 is not satisfied")

    valid = labels["valid_target"].to_numpy(bool)
    valid_symbols = np.flatnonzero(valid)
    valid_symbols = valid_symbols[(valid_symbols >= WASHOUT) & (valid_symbols < N_SYMBOLS - 1)]
    require(valid_symbols.shape == (3949,), f"Incorrect number of valid symbols: {valid_symbols.size}")
    target = labels["target_y_next"].to_numpy(np.float64)[valid_symbols]
    require(np.isfinite(target).all() and float(np.var(target)) > 0.0, "The target contains invalid values or has zero variance")

    # Preserve the archived float32 R=512 mean state. Training-set mean and
    # standard deviation are evaluated in float64, matching the locked analysis.
    features = np.asarray(states[valid_symbols], dtype=np.float32)
    train_rows, validation_rows, test_rows = split_indices(features.shape[0])

    # Standardize using training-set statistics only, then convert to float32 to
    # match the locked matrix-multiplication path.
    x_train = features[train_rows].astype(np.float64, copy=False)
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    active = np.isfinite(feature_std) & (feature_std > STD_FLOOR)
    require(int(active.sum()) == N_FEATURES, f"Expected 1024 active state variables, found {int(active.sum())}")
    standardized = (
        (features[:, active].astype(np.float64, copy=False) - feature_mean[active])
        / feature_std[active]
    ).astype(np.float32, copy=False)
    require(np.isfinite(standardized).all(), "The standardized states contain NaN or Inf")

    # The linear kernel is only the dual representation of linear ridge
    # regression; no RBF or polynomial nonlinear kernel is used.
    kernel = (standardized @ standardized.T).astype(np.float64, copy=False) / float(active.sum())
    k_train = centered_kernel(kernel, train_rows, train_rows, train_rows)
    k_validation = centered_kernel(kernel, validation_rows, train_rows, train_rows)
    # The locked implementation predicts all 3949 rows in one operation.
    # Splitting the matrix multiplication changes the final numerical digits, so
    # the 780-row test segment must not be multiplied separately.
    all_rows = np.arange(features.shape[0], dtype=np.int64)
    k_all = centered_kernel(kernel, all_rows, train_rows, train_rows)

    y_train = target[train_rows]
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    eigenvalues, eigenvectors = np.linalg.eigh(k_train)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    validation_grid: list[dict[str, float]] = []
    best_alpha = float(ALPHA_GRID[len(ALPHA_GRID) // 2])
    best_nmse = math.inf
    for alpha in ALPHA_GRID:
        prediction = predict_dual(
            k_validation, eigenvalues, eigenvectors, y_centered, y_mean, float(alpha)
        )
        metrics = metric_dict(target[validation_rows], prediction)
        validation_grid.append({"alpha": float(alpha), **{f"validation_{key}": value for key, value in metrics.items()}})
        if metrics["NMSE"] < best_nmse:
            best_nmse = metrics["NMSE"]
            best_alpha = float(alpha)

    all_prediction = predict_dual(
        k_all, eigenvalues, eigenvectors, y_centered, y_mean, best_alpha
    )
    test_prediction = all_prediction[test_rows]
    test_target = target[test_rows]
    test_metrics = metric_dict(test_target, test_prediction)
    print(
        f"Locally recomputed test MSE={test_metrics['MSE']:.17g}; "
        f"locked value={EXPECTED_TEST_MSE:.17g}; "
        f"difference={abs(test_metrics['MSE'] - EXPECTED_TEST_MSE):.3e}"
    )
    require(
        np.isclose(test_metrics["MSE"], EXPECTED_TEST_MSE, rtol=MSE_RTOL, atol=MSE_ATOL),
        "The test MSE exceeds the tolerance of the locked audit",
    )
    require(best_alpha == EXPECTED_BEST_ALPHA, f"The best alpha changed: {best_alpha}")
    require(
        np.isclose(test_metrics["NMSE"], EXPECTED_TEST_NMSE, rtol=MSE_RTOL, atol=5.0e-7),
        "The test NMSE exceeds the tolerance of the locked audit",
    )

    # These files contain the original predictions and validation sweep exported
    # by the locked Linux/BLAS analysis of the archived R=512 mean state.
    published = pd.read_csv(args.published_predictions)
    published_test = published.loc[published["split"] == "test"]
    require(len(published_test) == 780, "The locked predictions do not contain 780 test rows")
    published_mse = float(np.mean(published_test["residual"].to_numpy(np.float64) ** 2))
    require(
        np.isclose(published_mse, EXPECTED_TEST_MSE, rtol=0.0, atol=2.0e-12),
        "The locked prediction CSV does not reproduce the locked MSE",
    )
    published_grid = pd.read_csv(args.published_validation_grid)
    require(len(published_grid) == ALPHA_GRID.size, f"Expected {ALPHA_GRID.size} locked alpha rows, found {len(published_grid)}")
    require({"alpha", "val_NMSE"}.issubset(published_grid.columns), "The locked validation grid is missing required columns")
    published_best_alpha = float(published_grid.loc[published_grid["val_NMSE"].idxmin(), "alpha"])
    require(
        np.isclose(published_best_alpha, EXPECTED_BEST_ALPHA, rtol=0.0, atol=1.0e-15),
        f"The locked validation grid selects alpha={published_best_alpha}",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_grid).to_csv(args.output_dir / "recomputed_validation_alpha_grid.csv", index=False)
    pd.DataFrame(
        {
            "symbol_n": valid_symbols[test_rows],
            "target_y_next": test_target,
            "prediction_y_next": test_prediction,
            "residual": test_target - test_prediction,
        }
    ).to_csv(args.output_dir / "recomputed_test_predictions.csv", index=False, float_format="%.17g")
    summary = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "300 K DWSO 7.3 GHz NARMA10, physical-state average of 512 thermal realizations",
        "drive_thermal_seed_range_inclusive": [32345, 32856],
        "repeat_count_R": 512,
        "state_shape": [N_SYMBOLS, N_FEATURES],
        "state_dtype": str(states.dtype),
        "state_sha256": EXPECTED_STATE_SHA256,
        "target_alignment": "state after symbol n predicts target_y_next = y[n+1]",
        "feature_protocol": "layer-0 scalar mz; 8 phases/symbol; non-overlapping 8x8 spatial mean; 1024 nodes",
        "physical_state_averaging": "elementwise equal-weight mean of 512 aligned state matrices before readout fitting",
        "readout": "linear ridge regression in dual linear-Gram form; no nonlinear feature map",
        "best_alpha": best_alpha,
        "active_feature_columns": int(active.sum()),
        "split_sizes": {"train": int(train_rows.size), "validation": int(validation_rows.size), "test": int(test_rows.size)},
        "test_metrics": test_metrics,
        "locked_test_MSE": EXPECTED_TEST_MSE,
        "published_prediction_CSV_test_MSE": published_mse,
        "published_validation_grid_best_alpha": published_best_alpha,
        "absolute_MSE_difference": abs(test_metrics["MSE"] - EXPECTED_TEST_MSE),
        "declared_MSE_tolerance": {"rtol": MSE_RTOL, "atol": MSE_ATOL},
        "numerical_note": "The locked result was produced on Linux from the same archived float32 R=512 mean state. Eigensolver/BLAS implementations can shift the final numerical digits; the archived locked prediction CSV and validation grid are therefore checked independently.",
    }
    (args.output_dir / "recomputed_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    view = slice(0, 250)
    x = valid_symbols[test_rows][view]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True, constrained_layout=True)
    ax1.plot(x, test_target[view], color="black", lw=1.5, label="NARMA10 target")
    ax1.plot(x, test_prediction[view], color="#0072B2", lw=1.15, label="R=512 DWSO readout")
    ax1.set_ylabel("NARMA10 output")
    ax1.set_title(
        "300 K DWSO, 7.3 GHz, R=512 physical-state mean\n"
        f"test NMSE = {test_metrics['NMSE']:.4f}; MSE = {test_metrics['MSE']:.4e}"
    )
    ax1.legend(frameon=False, ncol=2)
    ax1.grid(alpha=0.22)
    ax2.plot(x, (test_target - test_prediction)[view], color="#D55E00", lw=1.0)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_xlabel("Input symbol index n")
    ax2.set_ylabel("Residual")
    ax2.grid(alpha=0.22)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"narma10_7p3GHz_R512_state_average.{suffix}", dpi=300 if suffix == "png" else None)
    if args.show:
        plt.show()
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figures and recomputed files saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

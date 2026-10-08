#!/usr/bin/env python3
"""Reproduce Mackey–Glass prediction by the R=256 mean DWSO state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


N_SYMBOLS = 4000
N_FEATURES = 1024
TRAIN = np.arange(500, 2600, dtype=np.int64)
VALIDATION = np.arange(2650, 3300, dtype=np.int64)
TEST = np.arange(3350, 4000, dtype=np.int64)
ALPHA_GRID = np.logspace(-6.0, 5.0, 23)
STD_FLOOR = 1.0e-12
EXPECTED_TEST_MSE = 0.0006465311108484558


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    mse = float(np.mean(residual * residual))
    variance = float(np.mean((target - target.mean()) ** 2))
    correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "NMSE": mse / variance,
        "R2": 1.0 - mse / variance,
        "PCC2": correlation * correlation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).resolve().parent
    parser.add_argument("--states", type=Path, default=package / "data" / "reservoir_states_R256_mean_mz_layer0_phase8_pool8x8.npy")
    parser.add_argument("--labels", type=Path, default=package / "data" / "mackey_glass_input_target_h10.csv")
    parser.add_argument("--output-dir", type=Path, default=package / "outputs")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    for path in (args.states, args.labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    states = np.load(args.states, mmap_mode="r", allow_pickle=False)
    require(states.shape == (N_SYMBOLS, N_FEATURES), f"Incorrect reservoir shape: {states.shape}")
    require(states.dtype == np.float32, f"Incorrect reservoir dtype: {states.dtype}")
    require(np.isfinite(states).all(), "The reservoir states contain NaN or Inf")

    labels = pd.read_csv(args.labels)
    required_columns = {
        "symbol_n",
        "mg_input",
        "drive_u_centered",
        "mg_target_h10",
        "target_symbol_n",
        "prediction_horizon_samples",
        "prediction_horizon_MG_time_units",
    }
    require(required_columns.issubset(labels.columns), f"Missing label columns: {required_columns - set(labels.columns)}")
    require(len(labels) == N_SYMBOLS, f"Incorrect number of label rows: {len(labels)}")
    symbol = labels["symbol_n"].to_numpy(np.int64)
    target_symbol = labels["target_symbol_n"].to_numpy(np.int64)
    require(np.array_equal(symbol, np.arange(N_SYMBOLS)), "symbol_n is not the consecutive sequence 0..3999")
    require(np.array_equal(target_symbol - symbol, np.full(N_SYMBOLS, 10)), "The target is not aligned to n+10")
    require((labels["prediction_horizon_samples"] == 10).all(), "The prediction horizon is not 10 stored samples")
    require((labels["prediction_horizon_MG_time_units"] == 10).all(), "The Mackey–Glass time horizon is not 10")
    target = labels["mg_target_h10"].to_numpy(np.float64)
    require(np.isfinite(target).all(), "The target contains NaN or Inf")

    # All standardization parameters are derived from the training set only; the
    # validation and sealed test data do not contribute to model fitting.
    x_train_raw = np.asarray(states[TRAIN], dtype=np.float64)
    feature_mean = x_train_raw.mean(axis=0)
    feature_std = x_train_raw.std(axis=0)
    active = feature_std > STD_FLOOR
    require(int(active.sum()) == N_FEATURES, f"Expected 1024 active state variables, found {int(active.sum())}")

    def standardized(rows: np.ndarray) -> np.ndarray:
        values = np.asarray(states[rows, :][:, active], dtype=np.float64)
        result = (values - feature_mean[active]) / feature_std[active]
        require(np.isfinite(result).all(), "The standardized states contain NaN or Inf")
        return result

    x_train = standardized(TRAIN)
    x_validation = standardized(VALIDATION)
    x_test = standardized(TEST)

    validation_rows: list[dict[str, float | bool]] = []
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="cholesky")
        model.fit(x_train, target[TRAIN])
        prediction = model.predict(x_validation)
        mse = float(np.mean((target[VALIDATION] - prediction) ** 2))
        validation_rows.append({"alpha": float(alpha), "validation_MSE": mse})
    selected = min(validation_rows, key=lambda row: (float(row["validation_MSE"]), float(row["alpha"])))
    selected_alpha = float(selected["alpha"])
    for row in validation_rows:
        row["selected"] = bool(float(row["alpha"]) == selected_alpha)

    model = Ridge(alpha=selected_alpha, fit_intercept=True, solver="cholesky")
    model.fit(x_train, target[TRAIN])
    test_prediction = model.predict(x_test)
    test_metrics = metrics(target[TEST], test_prediction)
    require(abs(test_metrics["MSE"] - EXPECTED_TEST_MSE) <= 1.0e-12, "The sealed-test MSE differs from the locked value")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(args.output_dir / "recomputed_validation_alpha_grid.csv", index=False)
    pd.DataFrame(
        {
            "symbol_n": symbol[TEST],
            "target_symbol_n": target_symbol[TEST],
            "target": target[TEST],
            "prediction": test_prediction,
            "residual": target[TEST] - test_prediction,
        }
    ).to_csv(args.output_dir / "recomputed_sealed_test_predictions.csv", index=False, float_format="%.17g")
    summary = {
        "status": "PASS",
        "repeat_count": 256,
        "drive_thermal_seed_range_inclusive": [32345, 32600],
        "active_feature_columns": int(active.sum()),
        "selected_alpha": selected_alpha,
        "split_half_open": {
            "washout": [0, 500],
            "train": [500, 2600],
            "purge_1": [2600, 2650],
            "validation": [2650, 3300],
            "purge_2": [3300, 3350],
            "sealed_test": [3350, 4000],
        },
        "sealed_test_metrics": test_metrics,
        "locked_test_MSE": EXPECTED_TEST_MSE,
        "absolute_MSE_difference": abs(test_metrics["MSE"] - EXPECTED_TEST_MSE),
        "plot_window": {
            "split": "sealed_test",
            "symbol_n_inclusive": [int(TEST[0]), int(TEST[-1])],
            "n_points": int(TEST.size),
        },
    }
    (args.output_dir / "recomputed_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Match manuscript Fig. 6h by showing the complete 650-point sealed test
    # segment rather than only its first 250 points.
    x = symbol[TEST]
    plot_target = target[TEST]
    plot_prediction = test_prediction
    plot_residual = plot_target - plot_prediction
    require(x.size == 650, f"Incorrect plotting-window length: {x.size}")
    require((int(x[0]), int(x[-1])) == (3350, 3999), f"Incorrect plotting-index range: {x[0]}..{x[-1]}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True, constrained_layout=True)
    ax1.plot(x, plot_target, color="black", lw=1.5, label="MG target (n+10)")
    ax1.plot(x, plot_prediction, color="#009E73", lw=1.15, label="R=256 DWSO readout")
    ax1.set_ylabel("Normalized MG state")
    ax1.set_title(
        f"300 K DWSO, 7.3 GHz, 256-state mean — complete sealed-test MSE = {test_metrics['MSE']:.4e}"
    )
    ax1.legend(frameon=False, ncol=2)
    ax1.grid(alpha=0.22)
    ax2.plot(x, plot_residual, color="#CC79A7", lw=1.0)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_xlabel("Input symbol index n")
    ax2.set_ylabel("Residual")
    ax2.grid(alpha=0.22)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"mackey_glass_R256_average.{suffix}", dpi=300 if suffix == "png" else None)
    if args.show:
        plt.show()
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figures and recomputed files saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

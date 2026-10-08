#!/usr/bin/env python3
"""Recompute and plot sine-to-square transformation by the 0 K SSO.

The data files contain several noise protocols and three ridge alpha values.
This script selects only the noise-free C2C predictions at alpha=1e-6 used in
Figure 4 and evaluates the MSE on the test set only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_N = 280
TOTAL_N = 400
FRAMES_PER_CYCLE = 40
RIDGE_ALPHA = 1.0e-6
CASES = {
    0.875: {
        "file": "SSO_0K_0p875GHz_predictions_all_protocols.npz",
        "expected_mse": 5.1379426294216446e-11,
        "color": "#0072B2",
    },
    0.880: {
        "file": "SSO_0K_0p880GHz_predictions_all_protocols.npz",
        "expected_mse": 4.5281708578916456e-4,
        "color": "#D55E00",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_case(path: Path) -> dict[str, np.ndarray | float]:
    """Load one frequency case and select its noise-free prediction at alpha=1e-6."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "target",
            "ridge_alphas",
            "clean_train_prediction",
            "clean_test_prediction",
        }
        require(required.issubset(archive.files), f"{path.name} is missing arrays: {required - set(archive.files)}")
        target = np.asarray(archive["target"], dtype=np.float64)
        alphas = np.asarray(archive["ridge_alphas"], dtype=np.float64)
        train_all = np.asarray(archive["clean_train_prediction"], dtype=np.float64)
        test_all = np.asarray(archive["clean_test_prediction"], dtype=np.float64)

    require(target.shape == (TOTAL_N,), f"Incorrect target shape: {target.shape}")
    require(train_all.shape == (alphas.size, TRAIN_N), f"Incorrect training-prediction shape: {train_all.shape}")
    require(test_all.shape == (alphas.size, TOTAL_N - TRAIN_N), f"Incorrect test-prediction shape: {test_all.shape}")
    require(np.isfinite(target).all() and np.isfinite(train_all).all() and np.isfinite(test_all).all(), "NaN or Inf detected")

    matches = np.flatnonzero(np.isclose(alphas, RIDGE_ALPHA, rtol=0.0, atol=1.0e-15))
    require(matches.size == 1, f"Expected one match for alpha={RIDGE_ALPHA:g}, found {matches.size}")
    alpha_index = int(matches[0])
    test_prediction = test_all[alpha_index]
    test_target = target[TRAIN_N:]
    mse = float(np.mean((test_target - test_prediction) ** 2))
    return {
        "target": target,
        "train_prediction": train_all[alpha_index],
        "test_prediction": test_prediction,
        "mse": mse,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it")
    args = parser.parse_args()

    results: dict[float, dict[str, np.ndarray | float]] = {}
    metric_rows: list[dict[str, float | int | str]] = []
    for frequency_ghz, specification in CASES.items():
        result = load_case(args.data_dir / str(specification["file"]))
        mse = float(result["mse"])
        expected = float(specification["expected_mse"])
        require(abs(mse - expected) <= 1.0e-12, f"The {frequency_ghz:g} GHz MSE differs from the locked value")
        results[frequency_ghz] = result
        metric_rows.append(
            {
                "frequency_GHz": frequency_ghz,
                "ridge_alpha": RIDGE_ALPHA,
                "train_n": TRAIN_N,
                "test_n": TOTAL_N - TRAIN_N,
                "test_MSE": mse,
                "locked_MSE": expected,
                "absolute_difference": abs(mse - expected),
                "status": "PASS",
            }
        )
        print(f"SSO {frequency_ghz:.3f} GHz: test MSE = {mse:.17g} [PASS]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_dir / "recomputed_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (args.output_dir / "recomputed_metrics.json").write_text(
        json.dumps(metric_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Input phases follow frame=1..40 in the source filenames, so each cycle
    # begins at sin(2π/40).
    frame_in_cycle = np.arange(TOTAL_N) % FRAMES_PER_CYCLE + 1
    input_sine = np.sin(2.0 * np.pi * frame_in_cycle / FRAMES_PER_CYCLE)
    test_x = np.arange(TRAIN_N, TOTAL_N)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(test_x, input_sine[TRAIN_N:], color="#666666", lw=1.4)
    axes[0].set_ylabel("Sine input")
    axes[0].set_title("Test-window input")
    axes[0].grid(alpha=0.22)

    for axis, frequency_ghz in zip(axes[1:], sorted(results)):
        result = results[frequency_ghz]
        target = np.asarray(result["target"])[TRAIN_N:]
        prediction = np.asarray(result["test_prediction"])
        # Use the same step interpolation for the target and prediction. Mixing
        # a step target with an ordinary line plot would introduce sloped lines
        # between samples and create an apparent half-sample shift at each edge,
        # even though the two discrete series are aligned sample by sample.
        axis.step(
            test_x,
            target,
            where="mid",
            color="black",
            lw=2.4,
            label="Target square wave",
        )
        axis.step(
            test_x,
            prediction,
            where="mid",
            color=str(CASES[frequency_ghz]["color"]),
            lw=1.2,
            label="Reservoir readout",
        )
        axis.set_ylabel("Output")
        axis.set_title(f"SSO {frequency_ghz:.3f} GHz — test MSE = {float(result['mse']):.4e}")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Sample index")

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"sine_to_square_0p875_0p880GHz.{suffix}", dpi=300 if suffix == "png" else None)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Figures and metrics saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

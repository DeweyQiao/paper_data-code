#!/usr/bin/env python3
"""Run the evaluation and plotting scripts for all three datasets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "01_SSO_0K_sine_to_square_0p875_0p880GHz" / "analyze_and_plot.py",
    ROOT / "02_DWSO_300K_NARMA10_7p3GHz_average_512" / "analyze_and_plot.py",
    ROOT / "03_DWSO_300K_Mackey_Glass_7p3GHz_average_256" / "analyze_and_plot.py",
]


def main() -> int:
    for script in SCRIPTS:
        if not script.is_file():
            raise FileNotFoundError(script)
        print(f"\nRunning: {script.parent.name}", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)
    print("\nAll three dataset analyses completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

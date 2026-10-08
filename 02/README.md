# Source data and analysis code for topological magnetic soliton spring oscillators

This directory accompanies the manuscript **Current-controlled task-adaptive reservoir computing with topological magnetic soliton spring oscillators** and contains three representative manuscript-related datasets. The source files were copied and reorganized without overwriting or modifying the original results. Each subdirectory contains the data, parameter documentation, commented Python code, and an `outputs/` directory with the generated figures and independently recomputed metrics.

The root-level `DATA_MANIFEST.csv` records the array shapes, file sizes, and SHA-256 checksums of the core data files. `VALIDATION_SUMMARY.json` summarizes the numerical results from the complete validation run.

## Contents

1. `01_SSO_0K_sine_to_square_0p875_0p880GHz`: Sine-to-square waveform transformation by the 0 K SSO at 0.875 and 0.880 GHz. The protocol is locked to manuscript Figure 4: noise-free C2C data, ridge alpha = 1e-6, the first 280 samples for training, and the final 120 samples for testing.
2. `02_DWSO_300K_NARMA10_7p3GHz_average_512`: NARMA10 reservoir data for the 300 K DWSO at 7.3 GHz, obtained by elementwise averaging 512 thermal realizations with drive-stage thermal seeds 32345–32856. The reservoir matrix contains 4000 × 1024 layer-0 `m_z` states, with eight sampled phases and non-overlapping 8 × 8 spatial averaging.
3. `03_DWSO_300K_Mackey_Glass_7p3GHz_average_256`: Mackey–Glass reservoir data for the 300 K DWSO at 7.3 GHz, obtained by elementwise averaging 256 thermal realizations. The reservoir matrix has shape 4000 × 1024.

## Running the analyses

The required dependencies are available in the current Python environment. From this directory, run:

```powershell
python run_all.py
```

Alternatively, enter any dataset subdirectory and run its analysis separately:

```powershell
python analyze_and_plot.py
```

In a new Python environment, install the required packages first:

```powershell
python -m pip install -r requirements.txt
```

## Numerical reference values

- SSO waveform-transformation test MSE at 0.875 GHz: `5.1379426294216446e-11`
- SSO waveform-transformation test MSE at 0.880 GHz: `4.5281708578916456e-4`
- DWSO NARMA10 test MSE at 300 K and 7.3 GHz with R = 512 averaging: `3.4656418745736477e-3`; test NMSE: `0.22060580621777207`
- DWSO Mackey–Glass sealed-test MSE at 300 K and 7.3 GHz with R = 256 averaging: `6.465311108484558e-4`

## Scope and limitations of the archived data

- Dataset 1 contains the manuscript-locked processed predictions and protocol results, rather than the complete set of raw OVF files. The historical raw-OVF directory for 0.875 GHz is no longer stored on this computer; the corresponding predictions, MSE, and Figure 4 locked table remain available and have passed the existing numerical audit. A spot-check record against raw OVF data is available for 0.880 GHz.
- Dataset 2 provides the archived float32 state matrix formed from a float64 accumulation of 512 symbol- and phase-aligned thermal realizations, rather than the 512 sets of raw OVF files. The accompanying aggregation audit records the source state and metadata checksums and confirms contiguous, unique thermal-seed coverage. MuMax3 was not rerun for this package.
- Dataset 3 is a direct copy of the audited R = 256 mean-state matrix and its associated labels, predictions, and protocol files.
- Successful execution demonstrates reproducibility of the arrays, data splits, and numerical metrics. It does not constitute an independent revalidation of the underlying magnetization dynamics or their physical interpretation.

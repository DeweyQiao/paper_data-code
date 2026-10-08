# 300 K DWSO at 7.3 GHz: NARMA10 with 512-realization state averaging

This dataset uses physical states averaged over 512 independent drive-stage thermal-noise realizations. The thermal seeds are the consecutive integers `32345–32856`, inclusive. Every realization uses the same NARMA10 input sequence. After strict alignment by symbol and phase, the `4000 × 1024` state matrices are averaged elementwise with equal weights before the readout is trained. The 512 predictions or performance metrics are not averaged.

## Locked physical and feature parameters

- Oscillator and temperature: DWSO, 300 K
- Carrier frequency: 7.3 GHz
- Baseline current density: `7.0e12 A/m²`
- Modulation ratio: 0.6; `drive_u_centered = 4*u_narma - 1`
- NARMA10 input random seed: 20260707; number of symbols: 4000
- Fixed relaxation-stage thermal seed: 12345; drive-stage thermal seeds: 32345–32856
- One carrier cycle per symbol and eight sampled readout phases per symbol
- Component and layer: layer-0 scalar `m_z`
- Spatial processing: non-overlapping 8 × 8 averaging at each phase; 1024 state variables in total

## Analysis protocol and results

After the first 50 symbols are discarded as washout, the state after symbol `n` is used to predict `y[n+1]`. The valid symbols are 50–3998, giving 3949 rows. In chronological order, these rows are divided into training, validation, and test sets of 2369, 780, and 780 rows, respectively, with a 10-row purge gap between adjacent sets. Standardization statistics are calculated from the training set only. The readout is linear ridge regression implemented in dual linear-Gram form; it uses no RBF, polynomial, or other nonlinear feature map. `alpha` is selected from `logspace(-10, 4, 29)` by minimizing the validation NMSE.

Locked Linux results:

- Best `alpha = 0.03162277660168379`
- Test MSE: `0.0034656418745736477`
- Test NMSE: `0.22060580621777207`
- Test NRMSE: `0.4696869236180331`
- Test R²: `0.779394193782228`
- Test PCC²: `0.791094624240468`

Compared with R = 256 under the same protocol, the test MSE decreases from `0.0036902752417541564` to `0.0034656418745736477`, an improvement of approximately 6.09%.

Run:

```powershell
python analyze_and_plot.py
```

The script checks file paths, SHA-256 checksums, array shapes, dtypes, NaN/Inf values, the input mapping, target alignment, data splits, and the number of active features. It then reselects `alpha`, recomputes the metrics, and writes the test predictions and figures. Because eigensolver and BLAS implementations can shift the final numerical digits in this ill-conditioned dual problem, the script also verifies the locked Linux prediction CSV and validation grid independently.

## File descriptions

- `data/reservoir_states_R512_mean_mz_layer0_phase8_pool8x8_float32.npy`: The R = 512 mean-state matrix used for the locked fit, with shape `(4000, 1024)` and dtype `float32`. Per-realization states were accumulated in float64 before division by 512; the locked mean-state artifact was then stored as float32.
- `data/narma10_labels.csv`: NARMA10 input, drive mapping, original target, and one-step-ahead target.
- `data/narma10_predictions_locked_R512.csv`: Predictions over the complete valid segment from the locked Linux environment.
- `data/ridge_alpha_validation_locked_R512.csv`: Validation metrics for the 29 candidate `alpha` values.
- `data/dataset_metadata.json`: Parameters, averaging procedure, data shapes, checksums, and provenance.
- `provenance/`: The complete R = 512 QA summary, contiguous seed-coverage table, source-artifact hash manifest, locked analysis summary, and original analysis/validation scripts.
- `outputs/`: Locally recomputed metrics, predictions, validation-curve data, and PNG/PDF figures.

## Scope and limitations of the archived data

This directory provides the R = 512 mean-state matrix, labels, locked predictions, validation sweep, and aggregation audit needed to recompute the reported result. It does not duplicate the 512 per-seed state matrices or the approximately 16.4 million raw OVF files. `provenance/summary_qa_R512.json` and `provenance/drive_thermal_seed_coverage_R512.csv` provide per-seed traceability to the original computation. Successful execution demonstrates reproducibility of the archived data and numerical workflow; it does not mean that MuMax3 was rerun or that the underlying magnetization dynamics were independently revalidated.

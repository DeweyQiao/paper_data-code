# 0 K SSO: sine-to-square transformation at 0.875 and 0.880 GHz

Run `python analyze_and_plot.py`. The script selects the noise-free C2C predictions at `alpha=1e-6`, recomputes the MSE over the final 120-sample test set, and saves PNG, PDF, CSV, and JSON outputs in `outputs/`.

The figure shows the sinusoidal input over the test window, the target square wave, and the reservoir readout at each of the two frequencies.

The other arrays in the data files contain historical results for the C2N/N2N noise protocols and other alpha values. The default script does not mix these results with the locked C2C analysis.

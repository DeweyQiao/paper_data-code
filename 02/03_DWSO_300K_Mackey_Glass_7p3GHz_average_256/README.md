# 300 K DWSO at 7.3 GHz: Mackey–Glass reservoir with 256-realization averaging

Run `python analyze_and_plot.py`. The script loads the 4000 × 1024 state matrix formed by elementwise averaging 256 independent drive-stage thermal realizations, standardizes the states using training-set statistics, and selects the alpha for strictly linear ridge regression on the validation set. It plots the target, prediction, and residual over the complete 650-point sealed test segment (input-symbol indices n = 3350–3999), matching the display length in manuscript Fig. 6h.

The thermal-seed range is 32345–32600, inclusive. The Mackey–Glass delay parameter is τ = 17 and the prediction horizon is H = 10. The expected sealed-test MSE is `0.0006465311108484558`.

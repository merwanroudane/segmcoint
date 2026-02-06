"""
Example: Segmented Cointegration Testing
=========================================

This script demonstrates the use of the segmcoint package for
testing segmented cointegration using both:

    1. Kim (2003) - Inference on Segmented Cointegration
       (Econometric Theory, 19, 620-639)

    2. Martins & Rodrigues (2022) - Tests for Segmented Cointegration
       (Empirical Economics, 63, 567-600)

The example generates data from a known DGP with segmented cointegration
(a cointegration relation that is interrupted by a nonstationary period)
and applies both test procedures.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from segmcoint import (
    kim_test,
    kim_break_estimator,
    mr_test,
    generate_segmented_data,
)


def main():
    # ================================================================
    # 1. Generate data with segmented cointegration
    # ================================================================
    print("=" * 72)
    print("SEGMENTED COINTEGRATION: ILLUSTRATIVE EXAMPLE")
    print("=" * 72)
    print()

    T = 200
    rho = 0.85
    n_break_start = int(0.4 * T)  # = 80
    n_break_end = int(0.6 * T)    # = 120

    print(f"Data generation:")
    print(f"  Sample size:                T = {T}")
    print(f"  AR root (cointegration):    rho = {rho}")
    print(f"  Noncointegration period:    [{n_break_start}, {n_break_end})")
    print(f"  True tau_0:                 {n_break_start/T:.2f}")
    print(f"  True tau_1:                 {n_break_end/T:.2f}")
    print()

    y, X, eps, info = generate_segmented_data(
        T=T, beta=1.0, rho=rho, sigma_v=0.1, sigma_u=0.1,
        n_break_start=n_break_start, n_break_end=n_break_end,
        model="drift", alpha=0.5, seed=42,
    )

    # ================================================================
    # 2. Kim (2003) tests
    # ================================================================
    print("=" * 72)
    print("TEST 1: Kim (2003) - Inference on Segmented Cointegration")
    print("=" * 72)
    print()

    # Run the test for all three model specifications
    for model_spec in ["drift"]:
        print(f"--- Model: {model_spec} ---")
        res_kim = kim_test(
            y, X, model=model_spec, max_ell=0.3, step=3,
            stat_types=("Zp", "Zt", "ADFp", "ADFt"))
        print(res_kim.summary())
        print()

    # ================================================================
    # 3. Kim (2003) break estimator
    # ================================================================
    print("=" * 72)
    print("BREAK DATE ESTIMATION: Kim (2003) Extremum Estimator")
    print("=" * 72)
    print()

    break_est = kim_break_estimator(y, X, model="drift", step=3)
    tau0_hat, tau1_hat = break_est["tau_hat"]
    print(f"  True break fractions:     tau_0 = {info['tau_0']:.3f}, "
          f"tau_1 = {info['tau_1']:.3f}")
    print(f"  Estimated break fractions: tau_0 = {tau0_hat:.3f}, "
          f"tau_1 = {tau1_hat:.3f}")
    print(f"  Lambda_max:                {break_est['Lambda_max']:.4f}")
    print()

    # ================================================================
    # 4. Martins & Rodrigues (2022) tests
    # ================================================================
    print("=" * 72)
    print("TEST 2: Martins & Rodrigues (2022) - Wald-Type Tests")
    print("=" * 72)
    print()

    for model_spec in ["drift"]:
        print(f"--- Model: {model_spec} ---")
        res_mr = mr_test(
            y, X, model=model_spec, max_breaks=4,
            epsilon=0.15, step=3)
        print(res_mr.summary())
        print()

    # ================================================================
    # 5. Comparison under the null (no cointegration)
    # ================================================================
    print("=" * 72)
    print("COMPARISON UNDER H0 (No Cointegration)")
    print("=" * 72)
    print()

    rng = np.random.default_rng(123)
    y_null = np.cumsum(rng.normal(0, 1, T))
    X_null = np.cumsum(rng.normal(0, 1, T))

    print("Kim (2003) under H0:")
    res_null_kim = kim_test(y_null, X_null, model="drift", step=5)
    for s in ["Zt", "Zp"]:
        rej = res_null_kim.significant(s, 0.05)
        print(f"  {s}*: {res_null_kim.infimum_stats[s]:.4f}, "
              f"Reject at 5%: {rej}")
    print()

    print("Martins & Rodrigues (2022) under H0:")
    res_null_mr = mr_test(y_null, X_null, model="drift",
                          max_breaks=4, step=5)
    for m in range(1, 5):
        rej = res_null_mr.significant(m, 0.05)
        print(f"  W({m}): {res_null_mr.W_stats[m]:.4f}, "
              f"Reject at 5%: {rej}")
    print(f"  W_max: {res_null_mr.W_max:.4f}, "
          f"Reject at 5%: {res_null_mr.significant('max', 0.05)}")

    print()
    print("=" * 72)
    print("END OF EXAMPLE")
    print("=" * 72)


if __name__ == "__main__":
    main()

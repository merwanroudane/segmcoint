"""
segmcoint: Tests for Segmented Cointegration
=============================================

A Python library implementing tests for segmented cointegration from:

1. Kim, J.-Y. (2003). Inference on Segmented Cointegration.
   Econometric Theory, 19, 620-639.

2. Martins, L.F. and Rodrigues, P.M.M. (2022). Tests for Segmented
   Cointegration: An Application to US Governments Budgets.
   Empirical Economics, 63, 567-600.

Main Functions
--------------
kim_test :
    Kim (2003) infimum-based tests for segmented cointegration.
    Computes Z*_rho, Z*_t, ADF*_rho, ADF*_t statistics.

mr_test :
    Martins & Rodrigues (2022) Wald-type tests for segmented cointegration.
    Computes W(m*) and W_max statistics.

kim_break_estimator :
    Extremum estimator for the noncointegration period (Kim 2003, Eq. 3.16-3.17).

Utility Functions
-----------------
generate_segmented_data :
    Generate simulated data from a segmented cointegration DGP.

simulate_kim_critical_values :
    Monte Carlo simulation of critical values for Kim (2003) tests.

simulate_mr_critical_values :
    Monte Carlo simulation of critical values for Martins & Rodrigues (2022) tests.

monte_carlo_size_power :
    Size and power analysis for both test procedures.

Example
-------
>>> import numpy as np
>>> from segmcoint import kim_test, mr_test, generate_segmented_data
>>>
>>> # Generate segmented cointegration data
>>> y, X, eps, info = generate_segmented_data(T=200, rho=0.85, seed=42)
>>>
>>> # Kim (2003) tests
>>> res_kim = kim_test(y, X, model='drift')
>>> print(res_kim)
>>>
>>> # Martins & Rodrigues (2022) tests
>>> res_mr = mr_test(y, X, model='drift')
>>> print(res_mr)
"""

__version__ = "1.0.0"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

from .kim2003 import (
    kim_test,
    kim_break_estimator,
    get_critical_value,
    KimTestResult,
)

from .martins_rodrigues2022 import (
    mr_test,
    get_mr_critical_value,
    MRTestResult,
)

from .utils import (
    ols_residuals,
    generate_segmented_data,
)

from .simulation import (
    simulate_kim_critical_values,
    simulate_mr_critical_values,
    monte_carlo_size_power,
)

__all__ = [
    # Kim (2003)
    "kim_test",
    "kim_break_estimator",
    "get_critical_value",
    "KimTestResult",
    # Martins & Rodrigues (2022)
    "mr_test",
    "get_mr_critical_value",
    "MRTestResult",
    # Utilities
    "ols_residuals",
    "generate_segmented_data",
    # Simulation
    "simulate_kim_critical_values",
    "simulate_mr_critical_values",
    "monte_carlo_size_power",
]

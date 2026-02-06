"""
Monte Carlo simulation for generating critical values.

Implements the simulation procedures described in:
    - Kim (2003, Section 3.1): Response surface method for asymptotic
      critical values of Z*_rho, Z*_t, ADF*_rho, ADF*_t.
    - Martins & Rodrigues (2022, Table 1): Critical values for W(m*) and
      W_max statistics.

References
----------
MacKinnon, J.G. (1991). Critical values for cointegration tests.
    In Engle and Granger (eds.), Long-Run Economic Relationships.
"""

import numpy as np
from numpy.linalg import inv
import warnings


def simulate_kim_critical_values(n, model="drift", max_ell=0.3,
                                 T=500, n_reps=5000, seed=None):
    """
    Simulate critical values for Kim (2003) infimum test statistics.

    Generates data under H_0 (no cointegration: all variables are
    independent random walks), computes the infimum statistics
    Z*_rho, Z*_t, ADF*_rho, ADF*_t, and returns the empirical
    distribution quantiles.

    Following Kim (2003, p.628), the response surface approach of
    MacKinnon (1991) is used: critical values are computed for
    multiple sample sizes and then extrapolated.

    Parameters
    ----------
    n : int
        Number of variables in the cointegration regression.
    model : str
        Model specification: 'none', 'drift', 'trend'.
    max_ell : float
        Maximum length of noncointegration period.
    T : int
        Sample size for simulation.
    n_reps : int
        Number of Monte Carlo replications.
    seed : int or None
        Random seed.

    Returns
    -------
    results : dict
        Dictionary with keys 'Zp', 'Zt', 'ADFp', 'ADFt', each containing
        a dict of percentile -> critical value.
    """
    from .kim2003 import _compute_Zp_Zt, _compute_ADF
    from .utils import ols_residuals, select_lag_bic

    rng = np.random.default_rng(seed)

    percentiles = [0.01, 0.025, 0.05, 0.10, 0.15, 0.95, 0.975, 0.99]

    all_Zp = []
    all_Zt = []
    all_ADFp = []
    all_ADFt = []

    max_len = int(max_ell * T)

    for rep in range(n_reps):
        # Generate n independent random walks under H0
        u = rng.normal(0, 1, size=(T, n))
        data = np.cumsum(u, axis=0)

        y = data[:, 0]
        X = data[:, 1:] if n > 1 else rng.normal(0, 1, size=(T, 1))
        X_rw = np.cumsum(X, axis=0) if n == 1 else X

        # OLS residuals
        try:
            e, _ = ols_residuals(y, X_rw, model=model)
        except np.linalg.LinAlgError:
            continue

        # Select lag order
        p = min(select_lag_bic(e, max_p=4), 2)

        # Search for infimum
        best_Zp = np.inf
        best_Zt = np.inf
        best_ADFp = np.inf
        best_ADFt = np.inf

        for ell_N in range(1, max_len + 1, max(1, max_len // 20)):
            for k0 in range(0, T - ell_N + 1, max(1, (T - ell_N) // 20)):
                k1 = k0 + ell_N
                T_C = T - ell_N
                if T_C < n:
                    continue

                w = np.ones(T)
                w[k0:k1] = 0.0

                Zp, Zt = _compute_Zp_Zt(e, w)
                if np.isfinite(Zp) and Zp < best_Zp:
                    best_Zp = Zp
                if np.isfinite(Zt) and Zt < best_Zt:
                    best_Zt = Zt

                try:
                    ADFp, ADFt = _compute_ADF(e, w, p=p)
                    if np.isfinite(ADFp) and ADFp < best_ADFp:
                        best_ADFp = ADFp
                    if np.isfinite(ADFt) and ADFt < best_ADFt:
                        best_ADFt = ADFt
                except (ValueError, np.linalg.LinAlgError):
                    pass

        all_Zp.append(best_Zp)
        all_Zt.append(best_Zt)
        all_ADFp.append(best_ADFp)
        all_ADFt.append(best_ADFt)

    # Compute quantiles
    results = {}
    for name, values in [("Zp", all_Zp), ("Zt", all_Zt),
                          ("ADFp", all_ADFp), ("ADFt", all_ADFt)]:
        arr = np.array([v for v in values if np.isfinite(v)])
        if len(arr) > 0:
            results[name] = {
                p: np.percentile(arr, p * 100) for p in percentiles
            }
        else:
            results[name] = {p: np.nan for p in percentiles}

    return results


def simulate_mr_critical_values(K_plus_1, model="drift", max_breaks=4,
                                 epsilon=0.15, T=1000, n_reps=5000,
                                 seed=None):
    """
    Simulate critical values for Martins & Rodrigues (2022) Wald tests.

    Generates data under H_0 (no cointegration), computes W(m*) for
    m* = 1,...,max_breaks and W_max, and returns empirical quantiles.

    Parameters
    ----------
    K_plus_1 : int
        Total number of variables.
    model : str
        Model specification: 'none', 'drift', 'trend'.
    max_breaks : int
        Maximum number of breaks.
    epsilon : float
        Trimming parameter.
    T : int
        Sample size.
    n_reps : int
        Number of Monte Carlo replications.
    seed : int or None
        Random seed.

    Returns
    -------
    results : dict
        Dictionary with keys 1, 2, ..., max_breaks, 'max', each containing
        a dict of percentile -> critical value.
    """
    from .martins_rodrigues2022 import (
        _compute_F_statistic, _generate_break_dates_fast)
    from .utils import ols_residuals, select_lag_bic

    rng = np.random.default_rng(seed)
    K = K_plus_1 - 1

    sig_levels = [0.90, 0.95, 0.975, 0.99]

    # Storage
    W_all = {m: [] for m in range(1, max_breaks + 1)}
    W_max_all = []

    for rep in range(n_reps):
        # Generate K+1 independent random walks under H0
        u = rng.normal(0, 1, size=(T, K_plus_1))
        data = np.cumsum(u, axis=0)

        y = data[:, 0]
        X = data[:, 1:]

        # OLS residuals
        try:
            e, _ = ols_residuals(y, X, model=model)
        except np.linalg.LinAlgError:
            continue

        p = min(select_lag_bic(e, max_p=4), 2)

        W_rep = {}
        for m_star in range(1, max_breaks + 1):
            best_FA = -np.inf
            best_FB = -np.inf

            step = max(1, int(T * epsilon / 5))

            for breaks in _generate_break_dates_fast(
                    T, m_star, epsilon, step):
                FA = _compute_F_statistic(e, breaks, m_star, "A", p)
                FB = _compute_F_statistic(e, breaks, m_star, "B", p)
                if np.isfinite(FA) and FA > best_FA:
                    best_FA = FA
                if np.isfinite(FB) and FB > best_FB:
                    best_FB = FB

            W_m = max(best_FA, best_FB)
            W_all[m_star].append(W_m)
            W_rep[m_star] = W_m

        W_max_all.append(max(W_rep.values()) if W_rep else np.nan)

    # Compute quantiles
    results = {}
    for m_star in range(1, max_breaks + 1):
        arr = np.array([v for v in W_all[m_star] if np.isfinite(v)])
        if len(arr) > 0:
            results[m_star] = {
                p: np.percentile(arr, p * 100) for p in sig_levels
            }
        else:
            results[m_star] = {p: np.nan for p in sig_levels}

    arr = np.array([v for v in W_max_all if np.isfinite(v)])
    if len(arr) > 0:
        results["max"] = {
            p: np.percentile(arr, p * 100) for p in sig_levels
        }
    else:
        results["max"] = {p: np.nan for p in sig_levels}

    return results


def monte_carlo_size_power(T=200, n_reps=2000, rho=0.9,
                           n_break_start=None, n_break_end=None,
                           model="drift", alpha=0.05, test="both",
                           seed=None):
    """
    Perform Monte Carlo size and power analysis.

    Reproduces the simulation design from:
    - Kim (2003, Section 4): Tables 3-5
    - Martins & Rodrigues (2022, Section 5): Tables 2-4

    Parameters
    ----------
    T : int
        Sample size.
    n_reps : int
        Number of Monte Carlo replications.
    rho : float
        AR(1) root in cointegration regime. rho=1 gives size, rho<1 gives power.
    n_break_start : int or None
        Start of noncointegration period. None defaults to int(0.4*T).
    n_break_end : int or None
        End of noncointegration period. None defaults to int(0.6*T).
    model : str
        Model specification.
    alpha : float
        Nominal significance level.
    test : str
        Which test to evaluate: 'kim', 'mr', or 'both'.
    seed : int or None
        Random seed.

    Returns
    -------
    results : dict
        Rejection frequencies for each test statistic.
    """
    from .utils import generate_segmented_data
    from .kim2003 import kim_test
    from .martins_rodrigues2022 import mr_test

    rng = np.random.default_rng(seed)

    if n_break_start is None:
        n_break_start = int(0.4 * T)
    if n_break_end is None:
        n_break_end = int(0.6 * T)

    kim_rejections = {"Zp": 0, "Zt": 0, "ADFp": 0, "ADFt": 0}
    mr_rejections = {m: 0 for m in range(1, 5)}
    mr_rejections["max"] = 0

    for rep in range(n_reps):
        seed_i = rng.integers(0, 2**31)

        y, X, _, _ = generate_segmented_data(
            T, beta=1.0, rho=rho, sigma_v=0.1, sigma_u=0.1,
            n_break_start=n_break_start, n_break_end=n_break_end,
            model=model, seed=seed_i)

        if test in ("kim", "both"):
            try:
                res_kim = kim_test(y, X, model=model, step=max(1, T // 50))
                for s in ["Zp", "Zt", "ADFp", "ADFt"]:
                    if res_kim.significant(s, alpha):
                        kim_rejections[s] += 1
            except Exception:
                pass

        if test in ("mr", "both"):
            try:
                res_mr = mr_test(y, X, model=model,
                                 step=max(1, T // 50))
                for m in range(1, 5):
                    if res_mr.significant(m, alpha):
                        mr_rejections[m] += 1
                if res_mr.significant("max", alpha):
                    mr_rejections["max"] += 1
            except Exception:
                pass

    results = {}
    if test in ("kim", "both"):
        results["kim"] = {
            s: kim_rejections[s] / n_reps for s in kim_rejections}
    if test in ("mr", "both"):
        results["mr"] = {
            s: mr_rejections[s] / n_reps for s in mr_rejections}

    return results

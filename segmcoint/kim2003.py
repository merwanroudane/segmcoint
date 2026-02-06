"""
Segmented cointegration tests from Kim (2003).

Implements the inference procedures for segmented cointegration proposed in:

    Kim, J.-Y. (2003). Inference on Segmented Cointegration.
    Econometric Theory, 19, 620-639.

This module provides:
    - Phillips-Perron-Ouliaris type tests: Z_rho(C_T) and Z_t(C_T)
    - Augmented Dickey-Fuller type tests: ADF_rho(C_T) and ADF_t(C_T)
    - Infimum test statistics: Z*_rho, Z*_t, ADF*_rho, ADF*_t
    - Extremum estimator for the noncointegration period
    - Critical value tables (Tables 1 and 2 from the paper)

Model Specifications (Kim 2003, Eq. 2.1a-2.1c):
    Case I   (model='none'):  x_{1t} = beta' x_{2t} + eps_t
    Case II  (model='drift'): x_{1t} = alpha + beta' x_{2t} + eps_t
    Case III (model='trend'): x_{1t} = alpha + gamma*t + beta' x_{2t} + eps_t
"""

import numpy as np
import warnings
from numpy.linalg import inv
from .utils import (
    ols_residuals,
    ar1_regression,
    adf_regression,
    newey_west_lrv,
    select_lag_bic,
)

# ============================================================================
# Asymptotic critical values from Kim (2003, Tables 1 and 2)
# Columns: 0.01, 0.025, 0.05, 0.10, 0.15, 0.95, 0.975, 0.99
# ============================================================================

# Table 1: Critical values of Z*_rho(C) and ADF*_rho(C)
# ell_bar(T_N) = 0.3
_CV_Zp_CASE_I = {
    1: [-13.00, -10.16, -8.18, -5.68, -4.57, 1.29, 1.64, 2.12],
    2: [-37.20, -32.30, -27.90, -23.70, -20.82, -2.40, -1.29, -0.56],
    3: [-46.63, -41.05, -36.41, -31.61, -28.74, -7.54, -5.81, -4.11],
    4: [-55.89, -48.97, -44.59, -39.43, -36.36, -12.73, -10.85, -9.22],
    5: [-63.94, -58.40, -52.79, -47.78, -44.59, -17.92, -15.86, -13.75],
    6: [-70.39, -64.30, -59.83, -54.22, -51.02, -22.47, -20.60, -18.42],
}

_CV_Zp_CASE_II = {
    1: [-20.15, -16.73, -13.96, -11.37, -9.59, -0.13, 0.50, 1.21],
    2: [-87.37, -64.30, -50.75, -39.58, -34.02, -7.77, -6.19, -4.79],
    3: [-105.48, -84.12, -65.53, -51.53, -44.82, -13.57, -11.25, -9.21],
    4: [-122.91, -96.68, -78.66, -61.95, -54.24, -18.65, -16.37, -13.77],
    5: [-130.86, -106.23, -87.07, -69.28, -60.56, -23.79, -21.33, -18.94],
    6: [-134.72, -109.84, -89.89, -73.65, -66.98, -28.72, -25.86, -22.50],
}

_CV_Zp_CASE_III = {
    1: [-29.17, -24.90, -21.55, -18.22, -16.03, -2.60, -1.67, -0.75],
    2: [-107.65, -80.24, -62.54, -46.59, -38.44, -8.26, -6.42, -4.39],
    3: [-131.66, -101.33, -79.34, -58.96, -49.43, -14.13, -12.05, -9.81],
    4: [-135.59, -109.94, -87.87, -68.31, -59.07, -19.65, -17.64, -15.20],
    5: [-140.29, -115.59, -96.57, -74.32, -65.16, -24.78, -22.25, -19.61],
    6: [-144.09, -118.83, -98.31, -78.46, -70.02, -28.98, -26.27, -23.71],
}

# Table 2: Critical values of Z*_t(C) and ADF*_t(C)
_CV_Zt_CASE_I = {
    1: [-2.51, -2.20, -1.96, -1.61, -1.43, 1.29, 1.70, 2.17],
    2: [-4.23, -3.95, -3.65, -3.34, -3.15, -0.88, -0.54, -0.23],
    3: [-4.88, -4.49, -4.23, -3.92, -3.73, -1.74, -1.46, -1.15],
    4: [-5.26, -4.89, -4.68, -4.38, -4.19, -2.32, -2.10, -1.84],
    5: [-5.59, -5.33, -5.09, -4.86, -4.70, -2.84, -2.62, -2.33],
    6: [-5.89, -5.68, -5.45, -5.16, -5.00, -3.24, -3.05, -2.81],
}

_CV_Zt_CASE_II = {
    1: [-3.49, -3.12, -2.88, -2.58, -2.37, -0.06, 0.31, 0.66],
    2: [-8.84, -7.43, -6.36, -5.31, -4.71, -1.91, -1.62, -1.28],
    3: [-10.05, -8.68, -7.38, -6.22, -5.47, -2.53, -2.28, -2.01],
    4: [-11.04, -9.51, -8.25, -6.86, -6.05, -3.03, -2.78, -2.51],
    5: [-11.33, -10.03, -8.70, -7.21, -6.35, -3.43, -3.20, -2.93],
    6: [-11.78, -10.14, -8.76, -7.33, -6.56, -3.78, -3.58, -3.37],
}

_CV_Zt_CASE_III = {
    1: [-3.92, -3.66, -3.40, -3.12, -2.94, -0.90, -0.62, -0.30],
    2: [-10.42, -8.78, -7.74, -6.32, -5.52, -1.96, -1.67, -1.27],
    3: [-11.68, -9.98, -8.62, -7.23, -6.20, -2.61, -2.27, -1.91],
    4: [-11.90, -10.37, -9.13, -7.57, -6.62, -3.10, -2.87, -2.61],
    5: [-12.32, -10.80, -9.52, -7.83, -6.86, -3.51, -3.30, -3.05],
    6: [-12.45, -10.98, -9.53, -7.79, -6.91, -3.84, -3.62, -3.38],
}

# Mapping percentiles to column indices
_CV_PERCENTILES = [0.01, 0.025, 0.05, 0.10, 0.15, 0.95, 0.975, 0.99]


def _get_cv_table(stat_type, model):
    """Get the appropriate critical value table."""
    if stat_type in ("Zp", "ADFp"):
        tables = {"none": _CV_Zp_CASE_I, "drift": _CV_Zp_CASE_II,
                   "trend": _CV_Zp_CASE_III}
    elif stat_type in ("Zt", "ADFt"):
        tables = {"none": _CV_Zt_CASE_I, "drift": _CV_Zt_CASE_II,
                   "trend": _CV_Zt_CASE_III}
    else:
        raise ValueError(f"Unknown stat_type: {stat_type}")
    return tables.get(model)


def get_critical_value(n, alpha, stat_type="Zt", model="drift"):
    """
    Look up asymptotic critical value from Kim (2003, Tables 1-2).

    These are for ell_bar(T_N) = 0.3 as reported in the paper.

    Parameters
    ----------
    n : int
        Number of variables in the cointegration regression (1 for
        univariate, 2 for bivariate, etc.).
    alpha : float
        Significance level (one of 0.01, 0.025, 0.05, 0.10, 0.15).
    stat_type : str
        Test statistic type: 'Zp', 'Zt', 'ADFp', 'ADFt'.
    model : str
        Model specification: 'none' (Case I), 'drift' (Case II),
        'trend' (Case III).

    Returns
    -------
    cv : float
        Critical value at the specified significance level.
    """
    table = _get_cv_table(stat_type, model)
    if table is None:
        raise ValueError(f"No table for model={model}")
    if n not in table:
        raise ValueError(f"n={n} not in table. Available: {list(table.keys())}")
    if alpha not in _CV_PERCENTILES:
        raise ValueError(
            f"alpha={alpha} not available. Use one of {_CV_PERCENTILES}")
    idx = _CV_PERCENTILES.index(alpha)
    return table[n][idx]


# ============================================================================
# Phillips-Perron-Ouliaris type statistics: Z_rho(C_T) and Z_t(C_T)
# Kim (2003, Eqs. 3.3 and 3.4)
# ============================================================================

def _compute_Zp_Zt(e, weighted, q=None):
    """
    Compute Z_rho and Z_t statistics for a given segmentation.

    Implements Kim (2003, Eqs. 3.3) and (3.4):
        Z_rho(C_T) = T_C * (rho_hat - 1)
                      - 0.5*(T_C^2 * sigma_rho^2 / s^2) * (lambda^2 - gamma_0)
        Z_t(C_T)   = (gamma_0/lambda^2)^{1/2} * t(C_T)
                      - {(lambda^2 - gamma_0)/(2*lambda)} * {T_C * sigma_rho / s}

    Parameters
    ----------
    e : ndarray
        Full residual series.
    weighted : ndarray
        Weight series w_t(C_T).
    q : int or None
        Bandwidth for long-run variance estimator.

    Returns
    -------
    Z_rho : float
    Z_t : float
    """
    T = len(e)
    T_C = int(np.sum(weighted > 0))

    # AR(1) regression with weights
    rho_hat, s_sq, sigma_rho_sq, _ = ar1_regression(e, weighted=weighted)

    # t-statistic
    if sigma_rho_sq > 0:
        t_stat = (rho_hat - 1.0) / np.sqrt(sigma_rho_sq)
    else:
        t_stat = 0.0

    # Compute residuals for long-run variance: v_hat = w_t * (e_t - rho_hat * e_{t-1})
    v_hat = weighted[1:] * (e[1:] - rho_hat * e[:-1])
    # Only keep observations where weight > 0
    v_active = v_hat[weighted[1:] > 0]

    if len(v_active) < 2:
        return np.nan, np.nan

    lambda_sq, gamma_0 = newey_west_lrv(v_active, q=q)

    if lambda_sq <= 0 or s_sq <= 0:
        return np.nan, np.nan

    # Z_rho (Eq. 3.3)
    Z_rho = T_C * (rho_hat - 1.0) - 0.5 * (
        T_C ** 2 * sigma_rho_sq / s_sq) * (lambda_sq - gamma_0)

    # Z_t (Eq. 3.4)
    lambda_hat = np.sqrt(lambda_sq)
    gamma_0_sqrt = np.sqrt(gamma_0)

    Z_t = (gamma_0_sqrt / lambda_hat) * t_stat - (
        (lambda_sq - gamma_0) / (2.0 * lambda_hat)) * (
        T_C * np.sqrt(sigma_rho_sq) / np.sqrt(s_sq))

    return Z_rho, Z_t


# ============================================================================
# ADF type statistics: ADF_rho(C_T) and ADF_t(C_T)
# Kim (2003, Eqs. 3.6 and 3.7)
# ============================================================================

def _compute_ADF(e, weighted, p=1):
    """
    Compute ADF_rho and ADF_t statistics for a given segmentation.

    Implements Kim (2003, Eqs. 3.6) and (3.7):
        ADF_rho(C_T) = T_C * (lambda_tilde / sigma_epsilon) * (rho_tilde - 1)
        ADF_t(C_T)   = t_tilde_T

    where lambda_tilde/sigma_epsilon = (1 - zeta_1 - ... - zeta_{p-1})^{-1}

    Parameters
    ----------
    e : ndarray
        Full residual series.
    weighted : ndarray
        Weight series w_t(C_T).
    p : int
        Lag order for the augmented regression.

    Returns
    -------
    ADF_rho : float
    ADF_t : float
    """
    T_C = int(np.sum(weighted > 0))

    rho_hat, t_stat, sigma_e, zeta_hat, _ = adf_regression(
        e, p=p, weighted=weighted)

    # lambda_tilde / sigma_epsilon  (Eq. below 3.7)
    if len(zeta_hat) > 0:
        denom = 1.0 - np.sum(zeta_hat)
        if abs(denom) < 1e-10:
            return np.nan, np.nan
        lambda_ratio = 1.0 / denom
    else:
        lambda_ratio = 1.0

    # ADF_rho (Eq. 3.6)
    ADF_rho = T_C * lambda_ratio * (rho_hat - 1.0)

    # ADF_t (Eq. 3.7)
    ADF_t = t_stat

    return ADF_rho, ADF_t


# ============================================================================
# Segmented cointegration test: infimum statistics
# Kim (2003, Eqs. 3.13 and 3.14)
# ============================================================================

def kim_test(y, X, model="drift", max_ell=0.3, step=1,
             q=None, p=None, max_p=12, stat_types=("Zp", "Zt", "ADFp", "ADFt"),
             verbose=False):
    """
    Kim (2003) tests for segmented cointegration.

    Searches over all possible segmentations {N_T} and computes the infimum
    of the test statistics Z_rho(C_T), Z_t(C_T), ADF_rho(C_T), ADF_t(C_T)
    over these segmentations, as described in Kim (2003, Section 3.1).

    The null hypothesis is H_0: rho = 1 for all t (no cointegration).
    The alternative is H_1: segmented cointegration where rho < 1 in C_T
    and rho = 1 in N_T.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    X : array_like, shape (T,) or (T, K)
        Regressor(s).
    model : str
        Deterministic specification: 'none' (Case I), 'drift' (Case II),
        'trend' (Case III).
    max_ell : float
        Upper bound for the length of the noncointegration period as a
        fraction of T. Denoted ell_bar(T_N) in Kim (2003).
        Critical values in Tables 1-2 are for max_ell = 0.3.
    step : int
        Step size for searching over segmentations (in observations).
    q : int or None
        Bandwidth for long-run variance estimator. If None, uses automatic
        selection.
    p : int or None
        Lag order for ADF statistics. If None, selected by BIC.
    max_p : int
        Maximum lag order for BIC selection.
    stat_types : tuple of str
        Which statistics to compute: any subset of
        ('Zp', 'Zt', 'ADFp', 'ADFt').
    verbose : bool
        If True, print progress information.

    Returns
    -------
    results : KimTestResult
        Object containing test statistics, critical values, break dates,
        and other information.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T = len(y)
    n = X.shape[1] + 1  # Number of variables in cointegration regression

    # Step 1: Estimate cointegrating regression on full sample
    e_full, beta_hat = ols_residuals(y, X, model=model)

    # Step 2: Select lag order if not provided
    if p is None:
        p = select_lag_bic(e_full, max_p=max_p)

    # Step 3: Search over all possible segmentations
    max_len = int(max_ell * T)
    min_k0 = 0          # N_T can start at the beginning
    max_k1 = T          # N_T can extend to the end

    results_dict = {s: {"stat": np.inf, "k0": None, "k1": None}
                    for s in stat_types}
    all_stats = {s: [] for s in stat_types}

    n_searched = 0

    for ell_N in range(1, max_len + 1, step):
        for k0 in range(0, T - ell_N + 1, step):
            k1 = k0 + ell_N
            if k1 > T:
                continue

            # Ensure C_T has at least n observations (Assumption 2)
            T_C = T - ell_N
            if T_C < n:
                continue

            # Construct weight vector: w_t = 1 for t in C_T, 0 for t in N_T
            # N_T = {k0+1, ..., k1} (1-indexed), or indices k0 to k1-1 (0-indexed)
            w = np.ones(T)
            w[k0:k1] = 0.0

            n_searched += 1

            # Compute statistics
            if "Zp" in stat_types or "Zt" in stat_types:
                Zp_val, Zt_val = _compute_Zp_Zt(e_full, w, q=q)

                if "Zp" in stat_types and np.isfinite(Zp_val):
                    all_stats["Zp"].append((Zp_val, k0, k1))
                    if Zp_val < results_dict["Zp"]["stat"]:
                        results_dict["Zp"] = {"stat": Zp_val, "k0": k0, "k1": k1}

                if "Zt" in stat_types and np.isfinite(Zt_val):
                    all_stats["Zt"].append((Zt_val, k0, k1))
                    if Zt_val < results_dict["Zt"]["stat"]:
                        results_dict["Zt"] = {"stat": Zt_val, "k0": k0, "k1": k1}

            if "ADFp" in stat_types or "ADFt" in stat_types:
                try:
                    ADFp_val, ADFt_val = _compute_ADF(e_full, w, p=p)
                except (ValueError, np.linalg.LinAlgError):
                    ADFp_val, ADFt_val = np.nan, np.nan

                if "ADFp" in stat_types and np.isfinite(ADFp_val):
                    all_stats["ADFp"].append((ADFp_val, k0, k1))
                    if ADFp_val < results_dict["ADFp"]["stat"]:
                        results_dict["ADFp"] = {
                            "stat": ADFp_val, "k0": k0, "k1": k1}

                if "ADFt" in stat_types and np.isfinite(ADFt_val):
                    all_stats["ADFt"].append((ADFt_val, k0, k1))
                    if ADFt_val < results_dict["ADFt"]["stat"]:
                        results_dict["ADFt"] = {
                            "stat": ADFt_val, "k0": k0, "k1": k1}

    if verbose:
        print(f"Searched {n_searched} segmentations.")

    # Step 4: Compute standard (non-segmented) tests on full sample
    w_full = np.ones(T)
    Zp_full, Zt_full = _compute_Zp_Zt(e_full, w_full, q=q)
    try:
        ADFp_full, ADFt_full = _compute_ADF(e_full, w_full, p=p)
    except (ValueError, np.linalg.LinAlgError):
        ADFp_full, ADFt_full = np.nan, np.nan

    full_sample_stats = {
        "Zp": Zp_full, "Zt": Zt_full,
        "ADFp": ADFp_full, "ADFt": ADFt_full
    }

    # Step 5: Collect critical values
    cvs = {}
    for s in stat_types:
        try:
            cvs[s] = {
                alpha: get_critical_value(n, alpha, stat_type=s, model=model)
                for alpha in [0.01, 0.025, 0.05, 0.10]
            }
        except (ValueError, KeyError):
            cvs[s] = {}

    return KimTestResult(
        stat_types=stat_types,
        infimum_stats={s: results_dict[s]["stat"] for s in stat_types},
        break_k0={s: results_dict[s]["k0"] for s in stat_types},
        break_k1={s: results_dict[s]["k1"] for s in stat_types},
        full_sample_stats=full_sample_stats,
        critical_values=cvs,
        model=model,
        n=n,
        T=T,
        max_ell=max_ell,
        lag_order=p,
        beta_hat=beta_hat,
        residuals=e_full,
    )


# ============================================================================
# Extremum estimator for the noncointegration period
# Kim (2003, Eqs. 3.16 and 3.17)
# ============================================================================

def kim_break_estimator(y, X, model="drift", max_ell=0.3, step=1):
    """
    Extremum estimator for the noncointegration period.

    Implements the estimator from Kim (2003, Eq. 3.16-3.17):
        Lambda_T(tau) = [((tau_1 - tau_0)T]^{-2} * sum_{t in N_T} e_t(C_T)^2
                         / [T_C^{-1} * sum_{t in C_T} e_t(C_T)^2]
        tau_hat = argmax_{tau in T} Lambda_T(tau)

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    X : array_like, shape (T,) or (T, K)
        Regressor(s).
    model : str
        Deterministic specification: 'none', 'drift', 'trend'.
    max_ell : float
        Maximum length of noncointegration period as fraction of T.
    step : int
        Step size for search.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'tau_hat': (tau_0_hat, tau_1_hat) estimated break fractions
        - 'k0_hat': estimated start of noncointegration period (0-indexed)
        - 'k1_hat': estimated end of noncointegration period (0-indexed)
        - 'Lambda_max': maximum value of Lambda_T
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T = len(y)
    n = X.shape[1] + 1

    max_len = int(max_ell * T)

    best_Lambda = -np.inf
    best_k0 = 0
    best_k1 = 0

    for ell_N in range(1, max_len + 1, step):
        for k0 in range(0, T - ell_N + 1, step):
            k1 = k0 + ell_N
            T_C = T - ell_N
            if T_C < n:
                continue

            # Construct weights for estimating beta from C_T only
            w = np.ones(T)
            w[k0:k1] = 0.0

            # Estimate beta using weighted least squares on C_T
            mask = w > 0
            y_c = y[mask]
            X_c = X[mask]
            e_c_all, _ = ols_residuals(y_c, X_c, model=model)

            # Compute full residuals using this beta estimate
            _, beta_c = ols_residuals(y_c, X_c, model=model)
            # Reconstruct residuals for ALL periods using C_T beta
            if model == "none":
                e_all = y - X @ beta_c
            elif model == "drift":
                e_all = y - np.column_stack([np.ones(T), X]) @ beta_c
            elif model == "trend":
                e_all = y - np.column_stack(
                    [np.ones(T), np.arange(1, T + 1), X]) @ beta_c

            # Compute Lambda_T (Eq. 3.16)
            e_N = e_all[k0:k1]  # Residuals in N_T
            e_C = e_all[mask]   # Residuals in C_T

            sum_sq_N = np.sum(e_N ** 2)
            sum_sq_C = np.sum(e_C ** 2)

            if sum_sq_C < 1e-15 or ell_N < 1:
                continue

            Lambda = (ell_N ** (-2) * sum_sq_N) / (T_C ** (-1) * sum_sq_C)

            if Lambda > best_Lambda:
                best_Lambda = Lambda
                best_k0 = k0
                best_k1 = k1

    tau_0_hat = best_k0 / T
    tau_1_hat = best_k1 / T

    return {
        "tau_hat": (tau_0_hat, tau_1_hat),
        "k0_hat": best_k0,
        "k1_hat": best_k1,
        "Lambda_max": best_Lambda,
    }


# ============================================================================
# Result class
# ============================================================================

class KimTestResult:
    """
    Container for Kim (2003) segmented cointegration test results.

    Attributes
    ----------
    stat_types : tuple
        Test statistic types computed.
    infimum_stats : dict
        Infimum statistics: Z*_rho, Z*_t, ADF*_rho, ADF*_t.
    break_k0 : dict
        Start of estimated noncointegration period for each statistic.
    break_k1 : dict
        End of estimated noncointegration period for each statistic.
    full_sample_stats : dict
        Full-sample (non-segmented) test statistics.
    critical_values : dict
        Critical values from Kim (2003, Tables 1-2).
    model : str
        Deterministic specification.
    n : int
        Number of variables.
    T : int
        Sample size.
    max_ell : float
        Maximum segmentation length.
    lag_order : int
        ADF lag order used.
    beta_hat : ndarray
        Estimated cointegrating vector.
    residuals : ndarray
        OLS residuals from full-sample regression.
    """

    def __init__(self, stat_types, infimum_stats, break_k0, break_k1,
                 full_sample_stats, critical_values, model, n, T, max_ell,
                 lag_order, beta_hat, residuals):
        self.stat_types = stat_types
        self.infimum_stats = infimum_stats
        self.break_k0 = break_k0
        self.break_k1 = break_k1
        self.full_sample_stats = full_sample_stats
        self.critical_values = critical_values
        self.model = model
        self.n = n
        self.T = T
        self.max_ell = max_ell
        self.lag_order = lag_order
        self.beta_hat = beta_hat
        self.residuals = residuals

    def significant(self, stat_type="Zt", alpha=0.05):
        """Check if the infimum test rejects H_0 at the given level."""
        cv = self.critical_values.get(stat_type, {}).get(alpha)
        if cv is None:
            return None
        return self.infimum_stats[stat_type] < cv

    def break_dates(self, stat_type="Zt"):
        """Return estimated break dates (0-indexed) for a given statistic."""
        return self.break_k0.get(stat_type), self.break_k1.get(stat_type)

    def break_fractions(self, stat_type="Zt"):
        """Return estimated break fractions tau_0, tau_1."""
        k0 = self.break_k0.get(stat_type)
        k1 = self.break_k1.get(stat_type)
        if k0 is None or k1 is None:
            return None, None
        return k0 / self.T, k1 / self.T

    def summary(self):
        """
        Produce a formatted summary string suitable for publication.

        Returns
        -------
        s : str
        """
        model_labels = {"none": "Case I (no deterministics)",
                        "drift": "Case II (intercept)",
                        "trend": "Case III (intercept + trend)"}

        lines = []
        lines.append("=" * 72)
        lines.append("Kim (2003) Segmented Cointegration Test Results")
        lines.append("=" * 72)
        lines.append(f"Model:            {model_labels.get(self.model, self.model)}")
        lines.append(f"Sample size (T):  {self.T}")
        lines.append(f"Variables (n):    {self.n}")
        lines.append(f"Max ell (T_N):    {self.max_ell:.2f}")
        lines.append(f"ADF lag order:    {self.lag_order}")
        lines.append("")
        lines.append("-" * 72)
        lines.append(f"{'Statistic':<12} {'Inf. Value':>12} {'Full Sample':>12}"
                      f"  {'5% CV':>10} {'Reject H0':>10}"
                      f"  {'tau_0':>8} {'tau_1':>8}")
        lines.append("-" * 72)

        for s in self.stat_types:
            inf_val = self.infimum_stats.get(s, np.nan)
            full_val = self.full_sample_stats.get(s, np.nan)
            cv_05 = self.critical_values.get(s, {}).get(0.05, np.nan)
            reject = self.significant(s, 0.05)
            reject_str = "Yes***" if reject else ("No" if reject is not None else "N/A")
            t0, t1 = self.break_fractions(s)
            t0_str = f"{t0:.3f}" if t0 is not None else "N/A"
            t1_str = f"{t1:.3f}" if t1 is not None else "N/A"

            lines.append(
                f"{s + '*':<12} {inf_val:>12.4f} {full_val:>12.4f}"
                f"  {cv_05:>10.4f} {reject_str:>10}"
                f"  {t0_str:>8} {t1_str:>8}")

        lines.append("-" * 72)
        lines.append("Notes: Infimum statistics are Z*_rho, Z*_t, ADF*_rho, ADF*_t")
        lines.append("       from Kim (2003, Eqs. 3.13-3.14).")
        lines.append("       Critical values from Tables 1-2 for ell_bar(T_N)=0.3.")
        lines.append("       Reject H0 implies segmented cointegration detected.")
        lines.append("=" * 72)
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

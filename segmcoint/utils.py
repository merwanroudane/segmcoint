"""
Utility functions for segmented cointegration tests.

Shared helper routines used by both Kim (2003) and Martins & Rodrigues (2022)
test implementations.

References
----------
Kim, J.-Y. (2003). Inference on Segmented Cointegration.
    Econometric Theory, 19, 620-639.
Martins, L.F. and Rodrigues, P.M.M. (2022). Tests for Segmented
    Cointegration. Empirical Economics, 63, 567-600.
"""

import numpy as np
from numpy.linalg import inv
from scipy import stats
import warnings


def ols_residuals(y, X, model="none"):
    """
    Compute OLS residuals from cointegrating regression.

    Implements the three model specifications from Kim (2003, Eq. 2.1a-2.1c):
        - 'none':  x_{1t} = beta' x_{2t} + epsilon_t           (2.1a)
        - 'drift': x_{1t} = alpha + beta' x_{2t} + epsilon_t   (2.1b)
        - 'trend': x_{1t} = alpha + gamma*t + beta' x_{2t} + epsilon_t  (2.1c)

    Parameters
    ----------
    y : ndarray, shape (T,)
        Dependent variable (x_{1t}).
    X : ndarray, shape (T,) or (T, K)
        Regressor(s) (x_{2t}).
    model : str, one of {'none', 'drift', 'trend'}
        Deterministic specification for the cointegrating regression.

    Returns
    -------
    residuals : ndarray, shape (T,)
        OLS residuals epsilon_hat_t.
    beta_hat : ndarray
        Estimated coefficient vector.
    """
    T = len(y)
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if model == "none":
        Z = X.copy()
    elif model == "drift":
        Z = np.column_stack([np.ones(T), X])
    elif model == "trend":
        Z = np.column_stack([np.ones(T), np.arange(1, T + 1), X])
    else:
        raise ValueError(f"Unknown model: {model}. Use 'none', 'drift', or 'trend'.")

    beta_hat = inv(Z.T @ Z) @ (Z.T @ y)
    residuals = y - Z @ beta_hat
    return residuals, beta_hat


def adf_regression(e, p=1, weighted=None):
    """
    Run augmented Dickey-Fuller regression on residual series.

    Implements the ADF regression:
        Delta e_t = zeta_1 * Delta e_{t-1} + ... + zeta_{p-1} * Delta e_{t-p+1}
                    + w_t(C_T) * (rho - 1) * e_{t-1} + epsilon_t

    as described in Kim (2003, Eq. 3.5).

    Parameters
    ----------
    e : ndarray, shape (T,)
        Residual series from cointegrating regression.
    p : int
        Lag order for augmented terms.
    weighted : ndarray or None, shape (T,)
        Weight series w_t(C_T) from Eq. (3.2). If None, all weights are 1.

    Returns
    -------
    rho_hat : float
        Estimated first-order autoregressive coefficient.
    t_stat : float
        t-statistic for testing rho = 1 (i.e., for rho - 1 = 0).
    sigma_e : float
        Estimated standard deviation of the regression error.
    zeta_hat : ndarray
        Estimated augmented lag coefficients.
    ssr : float
        Sum of squared residuals from the ADF regression.
    """
    e = np.asarray(e, dtype=np.float64).ravel()
    T = len(e)

    if weighted is None:
        weighted = np.ones(T)
    else:
        weighted = np.asarray(weighted, dtype=np.float64).ravel()

    # Construct Delta e_t
    de = np.diff(e)  # length T-1

    # Build the regression matrix
    # Dependent: Delta e_t for t = p+1, ..., T (indices p to T-1 in de)
    n_obs = T - 1 - p
    if n_obs <= 0:
        raise ValueError(f"Insufficient observations: T={T}, p={p}")

    Y = de[p:]  # (n_obs,)

    # Regressor: w_{t}(C_T) * e_{t-1}
    # For period t (1-indexed), e_{t-1} corresponds to e[t-1] in 0-indexed
    # de[i] = e[i+1] - e[i], so for de[p:], t ranges from p+1 to T
    # e_{t-1} for these t's: e[p] to e[T-1]
    e_lag = e[p:-1] if p > 0 else e[:-1]
    w_lag = weighted[p + 1: T]  # w_t for t = p+1, ..., T (0-indexed: p+1 to T)

    # Weighted error correction term
    we_lag = (w_lag * e_lag).reshape(-1, 1)

    # Augmented lags: Delta e_{t-j} for j = 1, ..., p
    if p > 0:
        aug_lags = np.column_stack([de[p - j: T - 1 - j] for j in range(1, p + 1)])
        Z = np.column_stack([aug_lags, we_lag])
    else:
        Z = we_lag

    # OLS estimation
    beta = inv(Z.T @ Z) @ (Z.T @ Y)
    residuals = Y - Z @ beta
    ssr = np.sum(residuals ** 2)
    sigma_e_sq = ssr / (n_obs - Z.shape[1])
    sigma_e = np.sqrt(max(sigma_e_sq, 1e-15))

    # Extract rho - 1 estimate (last regressor)
    gamma_hat = beta[-1]  # rho - 1
    rho_hat = gamma_hat + 1.0

    # Standard error of gamma_hat
    cov_beta = sigma_e_sq * inv(Z.T @ Z)
    se_gamma = np.sqrt(max(cov_beta[-1, -1], 1e-15))
    t_stat = gamma_hat / se_gamma

    # Extract zeta estimates
    zeta_hat = beta[:-1] if p > 0 else np.array([])

    return rho_hat, t_stat, sigma_e, zeta_hat, ssr


def ar1_regression(e, weighted=None):
    """
    Run simple AR(1) regression on residual series.

    Implements:  e_t = rho * e_{t-1} + v_t
    with optional weighting for segmented cointegration.

    Parameters
    ----------
    e : ndarray, shape (T,)
        Residual series.
    weighted : ndarray or None, shape (T,)
        Weight series w_t(C_T).

    Returns
    -------
    rho_hat : float
        Estimated AR(1) coefficient.
    s_sq : float
        Estimated variance of v_t.
    sigma_rho_sq : float
        Estimated variance of rho_hat.
    ssr : float
        Sum of squared residuals.
    """
    e = np.asarray(e, dtype=np.float64).ravel()
    T = len(e)

    if weighted is None:
        weighted = np.ones(T)
    else:
        weighted = np.asarray(weighted, dtype=np.float64).ravel()

    # Build weighted regression: w_t * e_t on w_{t-1} * e_{t-1} (per Kim 2003 p.625)
    # Actually: regress w_t(C_T)*e_t on w_{t-1}(C_T)*e_{t-1}
    y_reg = weighted[1:] * e[1:]
    x_reg = weighted[:-1] * e[:-1]

    # OLS
    denom = np.sum(x_reg ** 2)
    if denom < 1e-15:
        return 1.0, 0.0, 0.0, 0.0

    rho_hat = np.sum(x_reg * y_reg) / denom
    residuals = y_reg - rho_hat * x_reg
    ssr = np.sum(residuals ** 2)
    T_eff = np.sum(weighted[1:] > 0)
    s_sq = ssr / max(T_eff - 1, 1)
    sigma_rho_sq = s_sq / denom

    return rho_hat, s_sq, sigma_rho_sq, ssr


def newey_west_lrv(v, q=None):
    """
    Newey-West long-run variance estimator.

    Computes the long-run variance lambda^2 and gamma_0 as defined in
    Kim (2003, p.626):
        gamma_j = (1/T_C) * sum w_t * v_hat_t * v_hat_{t-j}
        lambda_hat = gamma_0 + sum_{j=1}^{q} (1 - j/(1+q)) * gamma_j

    Parameters
    ----------
    v : ndarray, shape (T,)
        Residual series (v_hat).
    q : int or None
        Bandwidth for Newey-West estimator.
        If None, uses floor(4*(T/100)^{2/9}).

    Returns
    -------
    lambda_sq : float
        Estimated long-run variance.
    gamma_0 : float
        Estimated variance (gamma_0).
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    T = len(v)

    if q is None:
        q = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    # gamma_0
    gamma_0 = np.mean(v ** 2)

    # gamma_j for j = 1, ..., q
    lambda_sq = gamma_0
    for j in range(1, q + 1):
        gamma_j = np.mean(v[j:] * v[:-j])
        w_j = 1.0 - j / (1.0 + q)
        lambda_sq += 2.0 * w_j * gamma_j  # factor 2 for symmetry

    return lambda_sq, gamma_0


def select_lag_aic(e, max_p=12):
    """
    Select lag order for ADF regression using the Akaike Information Criterion.

    Parameters
    ----------
    e : ndarray, shape (T,)
        Residual series.
    max_p : int
        Maximum lag order to consider.

    Returns
    -------
    p_opt : int
        Optimal lag order.
    """
    e = np.asarray(e, dtype=np.float64).ravel()
    T = len(e)
    max_p = min(max_p, T // 4)

    best_aic = np.inf
    p_opt = 0

    for p in range(0, max_p + 1):
        try:
            _, _, sigma_e, _, ssr = adf_regression(e, p=p)
            n_obs = T - 1 - p
            n_params = p + 1
            if n_obs <= n_params:
                continue
            aic = n_obs * np.log(ssr / n_obs) + 2 * n_params
            if aic < best_aic:
                best_aic = aic
                p_opt = p
        except (ValueError, np.linalg.LinAlgError):
            continue

    return p_opt


def select_lag_bic(e, max_p=12):
    """
    Select lag order for ADF regression using the Schwarz (BIC) criterion.

    Parameters
    ----------
    e : ndarray, shape (T,)
        Residual series.
    max_p : int
        Maximum lag order to consider.

    Returns
    -------
    p_opt : int
        Optimal lag order.
    """
    e = np.asarray(e, dtype=np.float64).ravel()
    T = len(e)
    max_p = min(max_p, T // 4)

    best_bic = np.inf
    p_opt = 0

    for p in range(0, max_p + 1):
        try:
            _, _, sigma_e, _, ssr = adf_regression(e, p=p)
            n_obs = T - 1 - p
            n_params = p + 1
            if n_obs <= n_params:
                continue
            bic = n_obs * np.log(ssr / n_obs) + n_params * np.log(n_obs)
            if bic < best_bic:
                best_bic = bic
                p_opt = p
        except (ValueError, np.linalg.LinAlgError):
            continue

    return p_opt


def generate_segmented_data(T, beta=1.0, rho=0.9, sigma_v=0.1,
                            sigma_u=0.1, n_break_start=None,
                            n_break_end=None, model="drift",
                            alpha=0.0, gamma=0.0, seed=None):
    """
    Generate data from a segmented cointegration DGP.

    Implements the model from Kim (2003, Section 4) and Martins &
    Rodrigues (2022, Eqs. 5.1-5.4):

        x_{1t} = alpha + gamma*t + beta * x_{2t} + epsilon_t
        x_{2t} = x_{2,t-1} + u_t
        epsilon_t = rho^{(t)} * epsilon_{t-1} + v_t

    where rho^{(t)} = rho for t in C_T (cointegration period)
    and   rho^{(t)} = 1   for t in N_T (noncointegration period).

    Parameters
    ----------
    T : int
        Sample size.
    beta : float or ndarray
        Cointegrating coefficient(s).
    rho : float
        AR(1) root in the cointegration regime. Must satisfy |rho| < 1.
    sigma_v : float
        Standard deviation of the innovation v_t in the error equation.
    sigma_u : float
        Standard deviation of the innovation u_t in x_{2t}.
    n_break_start : int or None
        Start of the noncointegration period (1-indexed).
        If None, defaults to int(0.4 * T).
    n_break_end : int or None
        End of the noncointegration period (1-indexed).
        If None, defaults to int(0.6 * T).
    model : str
        Deterministic specification: 'none', 'drift', 'trend'.
    alpha : float
        Intercept term.
    gamma : float
        Trend coefficient.
    seed : int or None
        Random seed.

    Returns
    -------
    y : ndarray, shape (T,)
        Dependent variable x_{1t}.
    X : ndarray, shape (T,) or (T, K)
        Regressor(s) x_{2t}.
    eps : ndarray, shape (T,)
        True error process epsilon_t.
    break_info : dict
        Dictionary with 'n_start', 'n_end', 'tau_0', 'tau_1'.
    """
    rng = np.random.default_rng(seed)

    if n_break_start is None:
        n_break_start = int(0.4 * T)
    if n_break_end is None:
        n_break_end = int(0.6 * T)

    beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    K = len(beta)

    # Generate x_{2t} as random walk(s)
    u = rng.normal(0, sigma_u, size=(T, K))
    X = np.cumsum(u, axis=0)

    # Generate epsilon_t with segmented persistence
    v = rng.normal(0, sigma_v, size=T)
    eps = np.zeros(T)
    for t in range(1, T):
        if n_break_start <= t < n_break_end:
            # Noncointegration period: unit root
            eps[t] = eps[t - 1] + v[t]
        else:
            # Cointegration period: stationary
            eps[t] = rho * eps[t - 1] + v[t]

    # Generate y
    trend = np.arange(1, T + 1)
    if model == "none":
        y = X @ beta + eps
    elif model == "drift":
        y = alpha + X @ beta + eps
    elif model == "trend":
        y = alpha + gamma * trend + X @ beta + eps
    else:
        raise ValueError(f"Unknown model: {model}")

    if K == 1:
        X = X.ravel()

    break_info = {
        "n_start": n_break_start,
        "n_end": n_break_end,
        "tau_0": n_break_start / T,
        "tau_1": n_break_end / T,
    }

    return y, X, eps, break_info

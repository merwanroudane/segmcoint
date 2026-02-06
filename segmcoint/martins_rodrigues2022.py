"""
Wald-type tests for segmented cointegration from Martins & Rodrigues (2022).

Implements the residual-based Wald-type tests for segmented cointegration
proposed in:

    Martins, L.F. and Rodrigues, P.M.M. (2022). Tests for Segmented
    Cointegration: An Application to US Governments Budgets.
    Empirical Economics, 63, 567-600.

This module provides:
    - F_A(tau, m*) and F_B(tau, m*) statistics (Eq. 3.2)
    - sup F_A(m*) and sup F_B(m*) (Eq. 3.3)
    - W(m*) combined statistic (Eq. 3.4)
    - W_max double maximum statistic (Eq. 3.5)
    - Critical value table (Table 1 from the paper)
    - Break date estimation (Remark 3)
"""

import numpy as np
import warnings
from numpy.linalg import inv
from itertools import product as iter_product
from .utils import ols_residuals, select_lag_bic


# ============================================================================
# Critical values from Martins & Rodrigues (2022, Table 1)
# Columns: W(1), W(2), W(3), W(4), W_max
# Rows indexed by K+1 (number of variables)
# ============================================================================

# Significance levels: 10%, 5%, 2.5%, 1%
_SIG_LEVELS = [0.10, 0.05, 0.025, 0.01]

# No deterministics
_CV_NO_DET = {
    2: {
        0.10: [8.229, 8.362, 7.168, 6.804, 9.677],
        0.05: [9.367, 9.329, 7.956, 7.790, 11.033],
        0.025: [10.615, 10.334, 8.901, 8.932, 12.499],
        0.01: [12.349, 11.958, 10.574, 11.129, 15.016],
    },
    3: {
        0.10: [7.812, 8.216, 6.872, 6.657, 9.403],
        0.05: [8.915, 9.165, 7.660, 7.612, 10.539],
        0.025: [9.829, 9.942, 8.583, 8.575, 12.085],
        0.01: [11.476, 11.298, 9.972, 10.942, 13.887],
    },
    4: {
        0.10: [7.529, 7.988, 6.664, 6.492, 9.131],
        0.05: [8.542, 8.903, 7.450, 7.495, 10.459],
        0.025: [9.615, 9.851, 8.293, 8.661, 11.795],
        0.01: [11.026, 11.217, 9.967, 10.835, 13.941],
    },
    5: {
        0.10: [7.546, 7.942, 6.516, 6.393, 8.952],
        0.05: [8.448, 8.921, 7.171, 7.530, 9.989],
        0.025: [9.320, 9.734, 7.915, 8.750, 11.180],
        0.01: [10.718, 10.741, 9.175, 10.640, 13.109],
    },
    6: {
        0.10: [7.857, 7.882, 6.553, 6.374, 9.151],
        0.05: [8.772, 8.832, 7.235, 7.397, 10.425],
        0.025: [9.733, 9.875, 8.003, 8.740, 11.803],
        0.01: [11.029, 10.910, 9.127, 11.314, 13.336],
    },
}

# Intercept only
_CV_INTERCEPT = {
    2: {
        0.10: [8.050, 8.279, 7.069, 6.895, 9.536],
        0.05: [9.106, 9.277, 7.764, 7.731, 10.762],
        0.025: [10.308, 10.269, 8.684, 8.930, 12.025],
        0.01: [12.089, 11.428, 10.329, 11.083, 14.670],
    },
    3: {
        0.10: [7.711, 8.036, 6.830, 6.664, 9.214],
        0.05: [8.761, 9.090, 7.645, 7.619, 10.552],
        0.025: [9.661, 10.050, 8.405, 8.602, 11.797],
        0.01: [11.103, 11.470, 9.855, 10.713, 13.973],
    },
    4: {
        0.10: [7.669, 7.913, 6.598, 6.499, 9.093],
        0.05: [8.628, 8.852, 7.281, 7.475, 10.330],
        0.025: [9.721, 9.816, 8.199, 8.738, 11.597],
        0.01: [10.707, 11.419, 9.374, 10.905, 13.785],
    },
    5: {
        0.10: [7.994, 7.928, 6.658, 6.418, 9.194],
        0.05: [8.936, 8.936, 7.364, 7.449, 10.407],
        0.025: [9.876, 9.801, 8.104, 8.683, 11.867],
        0.01: [11.179, 11.204, 9.720, 11.017, 13.893],
    },
    6: {
        0.10: [8.452, 8.023, 6.740, 6.375, 9.591],
        0.05: [9.616, 8.942, 7.450, 7.323, 10.754],
        0.025: [10.667, 9.864, 8.199, 8.511, 11.806],
        0.01: [11.793, 11.028, 9.480, 10.701, 13.379],
    },
}

# Intercept and time trend
_CV_TREND = {
    2: {
        0.10: [8.373, 8.527, 7.279, 7.152, 9.946],
        0.05: [9.810, 9.755, 8.229, 8.311, 11.580],
        0.025: [11.638, 11.092, 9.531, 9.561, 13.631],
        0.01: [15.592, 12.568, 11.884, 12.126, 17.734],
    },
    3: {
        0.10: [7.666, 8.085, 6.863, 6.713, 9.298],
        0.05: [8.756, 9.070, 7.597, 7.792, 10.430],
        0.025: [9.843, 10.110, 8.467, 9.017, 11.671],
        0.01: [11.365, 11.222, 9.612, 11.072, 13.690],
    },
    4: {
        0.10: [7.588, 7.985, 6.659, 6.588, 9.208],
        0.05: [8.589, 9.049, 7.366, 7.688, 10.375],
        0.025: [9.471, 10.000, 8.148, 9.003, 11.628],
        0.01: [10.870, 11.291, 9.423, 11.085, 13.334],
    },
    5: {
        0.10: [7.947, 7.903, 6.646, 6.435, 9.279],
        0.05: [9.000, 8.969, 7.289, 7.528, 10.461],
        0.025: [10.037, 9.785, 8.024, 8.784, 12.084],
        0.01: [11.751, 11.230, 9.500, 11.696, 14.375],
    },
    6: {
        0.10: [8.330, 7.890, 6.641, 6.458, 9.443],
        0.05: [9.412, 8.767, 7.363, 7.454, 10.676],
        0.025: [10.398, 9.755, 8.048, 9.080, 12.028],
        0.01: [11.916, 10.989, 9.111, 11.956, 13.747],
    },
}


def _get_mr_cv_table(model):
    """Get critical value table for M&R (2022) tests."""
    if model == "none":
        return _CV_NO_DET
    elif model == "drift":
        return _CV_INTERCEPT
    elif model == "trend":
        return _CV_TREND
    else:
        raise ValueError(f"Unknown model: {model}")


def get_mr_critical_value(K_plus_1, alpha, m_star, model="drift"):
    """
    Look up critical value from Martins & Rodrigues (2022, Table 1).

    Parameters
    ----------
    K_plus_1 : int
        Total number of variables (K+1, where K is the number of regressors).
    alpha : float
        Significance level: 0.10, 0.05, 0.025, 0.01.
    m_star : int or str
        Number of breaks (1, 2, 3, 4) or 'max' for W_max.
    model : str
        Deterministic specification: 'none', 'drift', 'trend'.

    Returns
    -------
    cv : float
        Critical value.
    """
    table = _get_mr_cv_table(model)
    if K_plus_1 not in table:
        raise ValueError(
            f"K+1={K_plus_1} not available. Use one of {list(table.keys())}")
    if alpha not in table[K_plus_1]:
        raise ValueError(f"alpha={alpha} not available. Use one of {_SIG_LEVELS}")

    vals = table[K_plus_1][alpha]

    if m_star == "max":
        return vals[4]
    elif isinstance(m_star, int) and 1 <= m_star <= 4:
        return vals[m_star - 1]
    else:
        raise ValueError(f"m_star must be 1, 2, 3, 4, or 'max'. Got {m_star}")


# ============================================================================
# Core computation: ADF test regression for a subsample
# ============================================================================

def _subsample_adf_regression(e, t_start, t_end, p_T, include_ec=True):
    """
    ADF regression on a subsample e[t_start:t_end].

    For the subsample, estimate:
        Delta e_t = c + gamma * e_{t-1} + sum_{i=1}^{p_T} pi_i * Delta e_{t-i} + a_t

    or under the null (include_ec=False):
        Delta e_t = sum_{i=1}^{p_T} pi_i * Delta e_{t-i} + a_t

    Parameters
    ----------
    e : ndarray
        Full residual series.
    t_start : int
        Start index of subsample (inclusive, 0-indexed).
    t_end : int
        End index of subsample (exclusive, 0-indexed).
    p_T : int
        Lag order for augmented terms.
    include_ec : bool
        If True, include the error correction term (c + gamma * e_{t-1}).

    Returns
    -------
    ssr : float
        Sum of squared residuals.
    n_obs : int
        Number of observations used.
    """
    sub_e = e[t_start:t_end]
    T_sub = len(sub_e)

    if T_sub <= p_T + 2:
        return np.nan, 0

    de = np.diff(sub_e)  # Delta e_t, length T_sub - 1

    # Effective sample starts at index p_T in de
    n_obs = len(de) - p_T
    if n_obs <= 0:
        return np.nan, 0

    Y = de[p_T:]  # Dependent variable

    # Build regressors
    regressors = []

    # Augmented lags: Delta e_{t-i} for i = 1, ..., p_T
    for i in range(1, p_T + 1):
        regressors.append(de[p_T - i: len(de) - i])

    if include_ec:
        # Intercept
        regressors.append(np.ones(n_obs))
        # e_{t-1}
        e_lag = sub_e[p_T: -1] if p_T > 0 else sub_e[:-1]
        regressors.append(e_lag)

    if len(regressors) == 0:
        ssr = np.sum(Y ** 2)
        return ssr, n_obs

    Z = np.column_stack(regressors)

    if Z.shape[1] >= n_obs:
        return np.nan, 0

    try:
        beta = inv(Z.T @ Z) @ (Z.T @ Y)
        residuals = Y - Z @ beta
        ssr = np.sum(residuals ** 2)
    except np.linalg.LinAlgError:
        return np.nan, 0

    return ssr, n_obs


# ============================================================================
# Compute F_A and F_B statistics
# Martins & Rodrigues (2022, Eq. 3.2)
# ============================================================================

def _compute_F_statistic(e, breaks, m_star, hypothesis, p_T):
    """
    Compute F_A or F_B statistic for given break dates.

    Parameters
    ----------
    e : ndarray
        Full residual series.
    breaks : tuple of int
        Break dates (0-indexed). Length m_star.
    m_star : int
        Number of breaks.
    hypothesis : str
        'A' (first regime is I(1)) or 'B' (first regime is I(0)).
    p_T : int
        Lag order.

    Returns
    -------
    F_stat : float
        Test statistic value.
    """
    T = len(e)

    # Build regime boundaries
    boundaries = [0] + list(breaks) + [T]
    n_regimes = m_star + 1

    # Compute SSR0: restricted SSR under null (no error correction anywhere)
    ssr0_total = 0.0
    for j in range(n_regimes):
        ssr_j, _ = _subsample_adf_regression(
            e, boundaries[j], boundaries[j + 1], p_T, include_ec=False)
        if np.isnan(ssr_j):
            return np.nan
        ssr0_total += ssr_j

    # Compute SSR_k,m*: unrestricted SSR under alternative
    ssr_alt_total = 0.0

    for j in range(n_regimes):
        regime_num = j + 1  # 1-indexed

        if hypothesis == "A":
            # H1A: odd regimes are I(1), even regimes are I(0)
            is_stationary = (regime_num % 2 == 0)
        elif hypothesis == "B":
            # H1B: odd regimes are I(0), even regimes are I(1)
            is_stationary = (regime_num % 2 == 1)
        else:
            raise ValueError(f"hypothesis must be 'A' or 'B', got {hypothesis}")

        ssr_j, _ = _subsample_adf_regression(
            e, boundaries[j], boundaries[j + 1], p_T,
            include_ec=is_stationary)
        if np.isnan(ssr_j):
            return np.nan
        ssr_alt_total += ssr_j

    if ssr_alt_total <= 0:
        return np.nan

    # Compute the F-statistic (Eq. 3.2)
    if hypothesis == "A":
        delta_B = 0
    else:
        delta_B = 1

    if m_star % 2 == 0:
        denom_df = m_star + 2 * delta_B
        numer_df = T - m_star - 2 * delta_B - p_T
    else:
        denom_df = m_star + 1
        numer_df = T - m_star - 1 - p_T

    if numer_df <= 0 or denom_df <= 0:
        return np.nan

    F_stat = (numer_df * (ssr0_total - ssr_alt_total)) / (
        denom_df * ssr_alt_total)

    return F_stat


# ============================================================================
# Generate all possible break date combinations
# ============================================================================

def _generate_break_dates(T, m_star, epsilon):
    """
    Generate all admissible break date partitions.

    Following Martins & Rodrigues (2022, below Eq. 3.3):
        tau_{j+1} - tau_j >= epsilon
        tau_1 >= epsilon
        tau_{m*} <= 1 - epsilon

    Parameters
    ----------
    T : int
        Sample size.
    m_star : int
        Number of breaks.
    epsilon : float
        Trimming parameter.

    Yields
    ------
    breaks : tuple of int
        Break dates (0-indexed).
    """
    min_seg = max(int(np.ceil(epsilon * T)), 2)

    if m_star == 1:
        for t1 in range(min_seg, T - min_seg + 1):
            yield (t1,)
    elif m_star == 2:
        for t1 in range(min_seg, T - 2 * min_seg + 1):
            for t2 in range(t1 + min_seg, T - min_seg + 1):
                yield (t1, t2)
    elif m_star == 3:
        for t1 in range(min_seg, T - 3 * min_seg + 1):
            for t2 in range(t1 + min_seg, T - 2 * min_seg + 1):
                for t3 in range(t2 + min_seg, T - min_seg + 1):
                    yield (t1, t2, t3)
    elif m_star == 4:
        for t1 in range(min_seg, T - 4 * min_seg + 1):
            for t2 in range(t1 + min_seg, T - 3 * min_seg + 1):
                for t3 in range(t2 + min_seg, T - 2 * min_seg + 1):
                    for t4 in range(t3 + min_seg, T - min_seg + 1):
                        yield (t1, t2, t3, t4)
    else:
        raise ValueError(f"m_star must be 1-4, got {m_star}")


def _generate_break_dates_fast(T, m_star, epsilon, step=1):
    """
    Generate break date partitions with optional step for speed.

    Same as _generate_break_dates but with configurable step size.
    """
    min_seg = max(int(np.ceil(epsilon * T)), 2)

    if m_star == 1:
        for t1 in range(min_seg, T - min_seg + 1, step):
            yield (t1,)
    elif m_star == 2:
        for t1 in range(min_seg, T - 2 * min_seg + 1, step):
            for t2 in range(t1 + min_seg, T - min_seg + 1, step):
                yield (t1, t2)
    elif m_star == 3:
        for t1 in range(min_seg, T - 3 * min_seg + 1, step):
            for t2 in range(t1 + min_seg, T - 2 * min_seg + 1, step):
                for t3 in range(t2 + min_seg, T - min_seg + 1, step):
                    yield (t1, t2, t3)
    elif m_star == 4:
        for t1 in range(min_seg, T - 4 * min_seg + 1, step):
            for t2 in range(t1 + min_seg, T - 3 * min_seg + 1, step):
                for t3 in range(t2 + min_seg, T - 2 * min_seg + 1, step):
                    for t4 in range(t3 + min_seg, T - min_seg + 1, step):
                        yield (t1, t2, t3, t4)
    else:
        raise ValueError(f"m_star must be 1-4, got {m_star}")


# ============================================================================
# Main test function
# ============================================================================

def mr_test(y, X, model="drift", max_breaks=4, epsilon=0.15,
            p=None, max_p=12, step=1, verbose=False):
    """
    Martins & Rodrigues (2022) Wald-type tests for segmented cointegration.

    Computes residual-based sup-Wald-type test statistics for detecting
    segmented cointegration with multiple structural breaks.

    The null hypothesis is H_0: no cointegration over the entire sample.
    The alternative allows m breaks with consecutive switches between
    stationarity and nonstationarity.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    X : array_like, shape (T,) or (T, K)
        Regressor(s).
    model : str
        Deterministic specification for the cointegrating regression:
        'none', 'drift', 'trend'.
    max_breaks : int
        Maximum number of breaks to consider (m_bar). Default 4.
    epsilon : float
        Trimming parameter. Default 0.15 as in the paper.
    p : int or None
        Lag order for ADF augmented terms. If None, selected by BIC.
    max_p : int
        Maximum lag order for BIC selection.
    step : int
        Step size for grid search over break dates.
        Use step > 1 for faster computation with large samples.
    verbose : bool
        If True, print progress.

    Returns
    -------
    results : MRTestResult
        Object containing W(m*), W_max statistics, critical values,
        break date estimates, and other information.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T = len(y)
    K = X.shape[1]
    K_plus_1 = K + 1

    # Step 1: Estimate cointegrating regression on full sample
    e, beta_hat = ols_residuals(y, X, model=model)

    # Step 2: Select lag order
    if p is None:
        p = select_lag_bic(e, max_p=max_p)

    # Step 3: Compute test statistics for each m*
    W_stats = {}
    sup_FA_stats = {}
    sup_FB_stats = {}
    best_breaks = {}

    for m_star in range(1, max_breaks + 1):
        if verbose:
            print(f"Computing W({m_star})...")

        best_FA = -np.inf
        best_FA_breaks = None
        best_FB = -np.inf
        best_FB_breaks = None

        for breaks in _generate_break_dates_fast(T, m_star, epsilon, step):
            # F_A: first regime is I(1)
            FA = _compute_F_statistic(e, breaks, m_star, "A", p)
            if np.isfinite(FA) and FA > best_FA:
                best_FA = FA
                best_FA_breaks = breaks

            # F_B: first regime is I(0)
            FB = _compute_F_statistic(e, breaks, m_star, "B", p)
            if np.isfinite(FB) and FB > best_FB:
                best_FB = FB
                best_FB_breaks = breaks

        sup_FA_stats[m_star] = best_FA
        sup_FB_stats[m_star] = best_FB

        # W(m*) = max(sup F_A(m*), sup F_B(m*))  (Eq. 3.4)
        W_m = max(best_FA, best_FB)
        W_stats[m_star] = W_m

        # Determine which hypothesis and breaks correspond to W(m*)
        if best_FA >= best_FB:
            best_breaks[m_star] = {
                "hypothesis": "A",
                "breaks": best_FA_breaks,
                "fractions": tuple(
                    b / T for b in best_FA_breaks) if best_FA_breaks else None,
            }
        else:
            best_breaks[m_star] = {
                "hypothesis": "B",
                "breaks": best_FB_breaks,
                "fractions": tuple(
                    b / T for b in best_FB_breaks) if best_FB_breaks else None,
            }

    # W_max = max_{1<=m<=m_bar} W(m)  (Eq. 3.5)
    W_max = max(W_stats.values()) if W_stats else np.nan
    W_max_m = max(W_stats, key=W_stats.get) if W_stats else None

    # Collect critical values
    cvs = {}
    for m_star in range(1, max_breaks + 1):
        try:
            cvs[m_star] = {
                alpha: get_mr_critical_value(
                    K_plus_1, alpha, m_star, model=model)
                for alpha in _SIG_LEVELS
            }
        except (ValueError, KeyError):
            cvs[m_star] = {}
    try:
        cvs["max"] = {
            alpha: get_mr_critical_value(
                K_plus_1, alpha, "max", model=model)
            for alpha in _SIG_LEVELS
        }
    except (ValueError, KeyError):
        cvs["max"] = {}

    return MRTestResult(
        W_stats=W_stats,
        W_max=W_max,
        W_max_m=W_max_m,
        sup_FA=sup_FA_stats,
        sup_FB=sup_FB_stats,
        best_breaks=best_breaks,
        critical_values=cvs,
        model=model,
        K_plus_1=K_plus_1,
        T=T,
        epsilon=epsilon,
        max_breaks=max_breaks,
        lag_order=p,
        beta_hat=beta_hat,
        residuals=e,
    )


# ============================================================================
# Result class
# ============================================================================

class MRTestResult:
    """
    Container for Martins & Rodrigues (2022) test results.

    Attributes
    ----------
    W_stats : dict
        W(m*) statistics for m* = 1, ..., max_breaks.
    W_max : float
        W_max double maximum statistic.
    W_max_m : int
        Number of breaks corresponding to W_max.
    sup_FA : dict
        sup F_A(m*) statistics.
    sup_FB : dict
        sup F_B(m*) statistics.
    best_breaks : dict
        Best break date information for each m*.
    critical_values : dict
        Critical values from Table 1.
    model : str
        Deterministic specification.
    K_plus_1 : int
        Total number of variables.
    T : int
        Sample size.
    epsilon : float
        Trimming parameter.
    max_breaks : int
        Maximum number of breaks considered.
    lag_order : int
        ADF lag order used.
    beta_hat : ndarray
        Estimated cointegrating vector.
    residuals : ndarray
        Full-sample OLS residuals.
    """

    def __init__(self, W_stats, W_max, W_max_m, sup_FA, sup_FB,
                 best_breaks, critical_values, model, K_plus_1, T,
                 epsilon, max_breaks, lag_order, beta_hat, residuals):
        self.W_stats = W_stats
        self.W_max = W_max
        self.W_max_m = W_max_m
        self.sup_FA = sup_FA
        self.sup_FB = sup_FB
        self.best_breaks = best_breaks
        self.critical_values = critical_values
        self.model = model
        self.K_plus_1 = K_plus_1
        self.T = T
        self.epsilon = epsilon
        self.max_breaks = max_breaks
        self.lag_order = lag_order
        self.beta_hat = beta_hat
        self.residuals = residuals

    def significant(self, m_star="max", alpha=0.05):
        """
        Check if the test rejects H_0 at the given level.

        Parameters
        ----------
        m_star : int or str
            Number of breaks (1-4) or 'max'.
        alpha : float
            Significance level.

        Returns
        -------
        reject : bool or None
        """
        if m_star == "max":
            stat_val = self.W_max
            cv = self.critical_values.get("max", {}).get(alpha)
        else:
            stat_val = self.W_stats.get(m_star)
            cv = self.critical_values.get(m_star, {}).get(alpha)

        if cv is None or stat_val is None:
            return None
        return stat_val > cv

    def estimated_breaks(self, m_star=None):
        """
        Return estimated break dates and fractions.

        Parameters
        ----------
        m_star : int or None
            Number of breaks. If None, uses W_max_m.

        Returns
        -------
        info : dict
        """
        if m_star is None:
            m_star = self.W_max_m
        return self.best_breaks.get(m_star, {})

    def summary(self):
        """
        Produce a formatted summary string suitable for publication.

        Returns
        -------
        s : str
        """
        model_labels = {"none": "No deterministics",
                        "drift": "Intercept only",
                        "trend": "Intercept and time trend"}

        lines = []
        lines.append("=" * 78)
        lines.append("Martins & Rodrigues (2022) Wald-Type Tests for Segmented Cointegration")
        lines.append("=" * 78)
        lines.append(f"Model:            {model_labels.get(self.model, self.model)}")
        lines.append(f"Sample size (T):  {self.T}")
        lines.append(f"Variables (K+1):  {self.K_plus_1}")
        lines.append(f"Trimming (eps):   {self.epsilon:.2f}")
        lines.append(f"Max breaks:       {self.max_breaks}")
        lines.append(f"ADF lag order:    {self.lag_order}")
        lines.append("")

        # W(m*) statistics
        lines.append("-" * 78)
        lines.append(f"{'Test':<10} {'Statistic':>12}  "
                      f"{'10% CV':>10} {'5% CV':>10} {'1% CV':>10}  "
                      f"{'Reject 5%':>10}")
        lines.append("-" * 78)

        for m_star in range(1, self.max_breaks + 1):
            stat = self.W_stats.get(m_star, np.nan)
            cv10 = self.critical_values.get(m_star, {}).get(0.10, np.nan)
            cv05 = self.critical_values.get(m_star, {}).get(0.05, np.nan)
            cv01 = self.critical_values.get(m_star, {}).get(0.01, np.nan)
            rej = self.significant(m_star, 0.05)
            rej_str = "Yes**" if rej else ("No" if rej is not None else "N/A")

            lines.append(
                f"W({m_star})     {stat:>12.4f}  "
                f"{cv10:>10.3f} {cv05:>10.3f} {cv01:>10.3f}  "
                f"{rej_str:>10}")

        # W_max
        stat = self.W_max
        cv10 = self.critical_values.get("max", {}).get(0.10, np.nan)
        cv05 = self.critical_values.get("max", {}).get(0.05, np.nan)
        cv01 = self.critical_values.get("max", {}).get(0.01, np.nan)
        rej = self.significant("max", 0.05)
        rej_str = "Yes**" if rej else ("No" if rej is not None else "N/A")

        lines.append(
            f"W_max     {stat:>12.4f}  "
            f"{cv10:>10.3f} {cv05:>10.3f} {cv01:>10.3f}  "
            f"{rej_str:>10}")

        lines.append("-" * 78)

        # Break date estimates
        lines.append("")
        lines.append("Estimated break dates (for W_max):")
        m_opt = self.W_max_m
        if m_opt is not None:
            info = self.best_breaks.get(m_opt, {})
            hyp = info.get("hypothesis", "N/A")
            brk = info.get("breaks", ())
            frac = info.get("fractions", ())

            r1_label = "I(1)" if hyp == "A" else "I(0)"
            lines.append(f"  Number of breaks:  {m_opt}")
            lines.append(f"  First regime:      {r1_label} (H1{hyp})")
            if brk:
                lines.append(f"  Break dates:       {brk}")
                lines.append(f"  Break fractions:   "
                             f"{tuple(round(f, 4) for f in frac)}")

        lines.append("")
        lines.append("Notes: W(m*) = max(sup F_A(m*), sup F_B(m*)).")
        lines.append("       W_max = max_{1<=m<=m_bar} W(m).")
        lines.append("       Critical values from Martins & Rodrigues (2022, Table 1).")
        lines.append("=" * 78)
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

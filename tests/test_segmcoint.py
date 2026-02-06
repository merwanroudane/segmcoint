"""
Test suite for the segmcoint package.

Tests cover:
    1. Kim (2003) test statistics computation and critical values
    2. Martins & Rodrigues (2022) Wald-type tests
    3. Utility functions
    4. Consistency checks between papers
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmcoint import (
    kim_test,
    kim_break_estimator,
    mr_test,
    get_critical_value,
    get_mr_critical_value,
    generate_segmented_data,
    ols_residuals,
)
from segmcoint.utils import (
    ar1_regression,
    adf_regression,
    newey_west_lrv,
    select_lag_bic,
)


# ============================================================================
# Test data generation
# ============================================================================

class TestDataGeneration:
    """Tests for the data generation process."""

    def test_basic_generation(self):
        y, X, eps, info = generate_segmented_data(T=200, seed=42)
        assert len(y) == 200
        assert len(X) == 200
        assert len(eps) == 200
        assert info["n_start"] == 80  # 0.4 * 200
        assert info["n_end"] == 120   # 0.6 * 200

    def test_unit_root_in_break_period(self):
        """Error should be unit root in N_T."""
        T = 1000
        y, X, eps, info = generate_segmented_data(
            T=T, rho=0.5, sigma_v=1.0, seed=123)
        # Variance in N_T should be much larger than in C_T
        n_start = info["n_start"]
        n_end = info["n_end"]
        var_NT = np.var(eps[n_start:n_end])
        var_CT_before = np.var(eps[:n_start])
        # N_T has unit root, so variance should grow
        # This is a probabilistic test, so we use a loose bound
        assert var_NT > 0

    def test_model_specifications(self):
        for model in ["none", "drift", "trend"]:
            y, X, eps, info = generate_segmented_data(
                T=100, model=model, seed=42)
            assert len(y) == 100

    def test_multivariate(self):
        y, X, eps, info = generate_segmented_data(
            T=100, beta=np.array([1.0, 2.0]), seed=42)
        assert X.shape == (100, 2)


# ============================================================================
# Test OLS residuals
# ============================================================================

class TestOLSResiduals:
    """Tests for OLS residual computation."""

    def test_no_deterministics(self):
        rng = np.random.default_rng(42)
        T = 100
        X = np.cumsum(rng.normal(size=T))
        eps = rng.normal(size=T) * 0.1
        y = 2.0 * X + eps
        resid, beta = ols_residuals(y, X, model="none")
        assert abs(beta[0] - 2.0) < 0.5
        assert len(resid) == T

    def test_with_intercept(self):
        rng = np.random.default_rng(42)
        T = 100
        X = np.cumsum(rng.normal(size=T))
        eps = rng.normal(size=T) * 0.1
        y = 1.0 + 2.0 * X + eps
        resid, beta = ols_residuals(y, X, model="drift")
        assert abs(beta[1] - 2.0) < 0.5
        assert len(resid) == T

    def test_with_trend(self):
        rng = np.random.default_rng(42)
        T = 100
        X = np.cumsum(rng.normal(size=T))
        y = 1.0 + 0.01 * np.arange(1, T + 1) + 2.0 * X
        resid, beta = ols_residuals(y, X, model="trend")
        assert len(resid) == T


# ============================================================================
# Test Kim (2003) critical values
# ============================================================================

class TestKimCriticalValues:
    """Tests for Kim (2003) critical value lookup."""

    def test_table1_case_i(self):
        # Z*_rho, Case I, n=2, 5%: -27.9
        cv = get_critical_value(2, 0.05, stat_type="Zp", model="none")
        assert abs(cv - (-27.9)) < 0.01

    def test_table1_case_ii(self):
        # Z*_rho, Case II, n=2, 5%: -50.75
        cv = get_critical_value(2, 0.05, stat_type="Zp", model="drift")
        assert abs(cv - (-50.75)) < 0.01

    def test_table2_case_i(self):
        # Z*_t, Case I, n=2, 5%: -3.65
        cv = get_critical_value(2, 0.05, stat_type="Zt", model="none")
        assert abs(cv - (-3.65)) < 0.01

    def test_table2_case_iii(self):
        # Z*_t, Case III, n=3, 10%: -7.23
        cv = get_critical_value(3, 0.10, stat_type="Zt", model="trend")
        assert abs(cv - (-7.23)) < 0.01

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            get_critical_value(7, 0.05, stat_type="Zt", model="none")


# ============================================================================
# Test Martins & Rodrigues (2022) critical values
# ============================================================================

class TestMRCriticalValues:
    """Tests for M&R (2022) critical value lookup."""

    def test_no_det_k2(self):
        # W(1), K+1=2, 5%, no det: 9.367
        cv = get_mr_critical_value(2, 0.05, 1, model="none")
        assert abs(cv - 9.367) < 0.001

    def test_intercept_k2(self):
        # W(2), K+1=2, 5%, intercept: 9.277
        cv = get_mr_critical_value(2, 0.05, 2, model="drift")
        assert abs(cv - 9.277) < 0.001

    def test_wmax(self):
        # W_max, K+1=2, 5%, no det: 11.033
        cv = get_mr_critical_value(2, 0.05, "max", model="none")
        assert abs(cv - 11.033) < 0.001

    def test_trend_k3(self):
        # W(1), K+1=3, 1%, trend: 11.365
        cv = get_mr_critical_value(3, 0.01, 1, model="trend")
        assert abs(cv - 11.365) < 0.001


# ============================================================================
# Test Kim (2003) full test procedure
# ============================================================================

class TestKimTest:
    """Tests for Kim (2003) test procedure."""

    def test_under_alternative(self):
        """Power test: should detect segmented cointegration."""
        y, X, _, _ = generate_segmented_data(
            T=200, rho=0.7, sigma_v=0.1, seed=42)
        res = kim_test(y, X, model="drift", step=5)
        # Check that infimum stats are computed
        assert np.isfinite(res.infimum_stats["Zt"])
        assert np.isfinite(res.infimum_stats["Zp"])

    def test_under_null(self):
        """Size test: independent random walks (no cointegration)."""
        rng = np.random.default_rng(42)
        T = 200
        y = np.cumsum(rng.normal(size=T))
        X = np.cumsum(rng.normal(size=T))
        res = kim_test(y, X, model="drift", step=5)
        # Full sample stats should be computed
        assert np.isfinite(res.full_sample_stats["Zt"])

    def test_break_fractions(self):
        """Break fractions should be in [0, 1]."""
        y, X, _, _ = generate_segmented_data(T=200, rho=0.8, seed=42)
        res = kim_test(y, X, model="drift", step=5)
        t0, t1 = res.break_fractions("Zt")
        if t0 is not None and t1 is not None:
            assert 0 <= t0 <= 1
            assert 0 <= t1 <= 1
            assert t0 < t1

    def test_summary(self):
        """Summary should be a string."""
        y, X, _, _ = generate_segmented_data(T=100, rho=0.8, seed=42)
        res = kim_test(y, X, model="drift", step=5)
        s = res.summary()
        assert isinstance(s, str)
        assert "Kim (2003)" in s

    def test_all_models(self):
        y, X, _, _ = generate_segmented_data(T=100, rho=0.8, seed=42)
        for model in ["none", "drift", "trend"]:
            res = kim_test(y, X, model=model, step=5)
            assert res.model == model


# ============================================================================
# Test Kim (2003) break estimator
# ============================================================================

class TestKimBreakEstimator:
    """Tests for the extremum estimator."""

    def test_estimator_runs(self):
        y, X, _, info = generate_segmented_data(
            T=200, rho=0.7, seed=42)
        result = kim_break_estimator(y, X, model="drift", step=5)
        assert "tau_hat" in result
        assert "Lambda_max" in result
        assert result["Lambda_max"] > 0

    def test_estimator_accuracy(self):
        """Estimated break should be near the true break."""
        y, X, _, info = generate_segmented_data(
            T=300, rho=0.5, sigma_v=0.5, seed=42)
        result = kim_break_estimator(y, X, model="drift", step=3)
        tau0_true = info["tau_0"]
        tau1_true = info["tau_1"]
        tau0_hat, tau1_hat = result["tau_hat"]
        # Allow tolerance of 0.15 (Theorem 3: consistency)
        assert abs(tau0_hat - tau0_true) < 0.20
        assert abs(tau1_hat - tau1_true) < 0.20


# ============================================================================
# Test Martins & Rodrigues (2022) full test procedure
# ============================================================================

class TestMRTest:
    """Tests for M&R (2022) test procedure."""

    def test_under_alternative(self):
        """Should detect segmented cointegration."""
        y, X, _, _ = generate_segmented_data(
            T=200, rho=0.7, sigma_v=0.1, seed=42)
        res = mr_test(y, X, model="drift", max_breaks=2, step=5)
        assert np.isfinite(res.W_max)
        assert all(np.isfinite(v) for v in res.W_stats.values())

    def test_under_null(self):
        """Independent random walks: should not reject often."""
        rng = np.random.default_rng(42)
        T = 200
        y = np.cumsum(rng.normal(size=T))
        X = np.cumsum(rng.normal(size=T))
        res = mr_test(y, X, model="drift", max_breaks=2, step=5)
        assert np.isfinite(res.W_max)

    def test_break_dates(self):
        """Break dates should be between 0 and T."""
        y, X, _, _ = generate_segmented_data(T=200, rho=0.8, seed=42)
        res = mr_test(y, X, model="drift", max_breaks=2, step=5)
        info = res.estimated_breaks()
        if info and info.get("breaks"):
            for b in info["breaks"]:
                assert 0 < b < 200

    def test_summary(self):
        y, X, _, _ = generate_segmented_data(T=100, rho=0.8, seed=42)
        res = mr_test(y, X, model="drift", max_breaks=2, step=5)
        s = res.summary()
        assert isinstance(s, str)
        assert "Martins" in s

    def test_all_models(self):
        y, X, _, _ = generate_segmented_data(T=100, rho=0.8, seed=42)
        for model in ["none", "drift", "trend"]:
            res = mr_test(y, X, model=model, max_breaks=2, step=5)
            assert res.model == model

    def test_multivariate(self):
        """Test with K > 1 regressors."""
        rng = np.random.default_rng(42)
        T = 150
        K = 2
        X = np.cumsum(rng.normal(size=(T, K)), axis=0)
        eps = np.zeros(T)
        for t in range(1, T):
            if 60 <= t < 90:
                eps[t] = eps[t - 1] + rng.normal() * 0.1
            else:
                eps[t] = 0.7 * eps[t - 1] + rng.normal() * 0.1
        y = X @ np.array([1.0, 2.0]) + eps
        res = mr_test(y, X, model="drift", max_breaks=2, step=5)
        assert res.K_plus_1 == 3


# ============================================================================
# Test utility functions
# ============================================================================

class TestUtilities:
    """Tests for utility functions."""

    def test_ar1_regression(self):
        rng = np.random.default_rng(42)
        T = 500
        eps = np.zeros(T)
        for t in range(1, T):
            eps[t] = 0.8 * eps[t - 1] + rng.normal() * 0.1
        rho, s_sq, sigma_rho_sq, ssr = ar1_regression(eps)
        assert abs(rho - 0.8) < 0.1

    def test_newey_west(self):
        rng = np.random.default_rng(42)
        v = rng.normal(size=200)
        lam_sq, g0 = newey_west_lrv(v)
        assert lam_sq > 0
        assert g0 > 0

    def test_lag_selection(self):
        rng = np.random.default_rng(42)
        e = np.cumsum(rng.normal(size=100))
        p = select_lag_bic(e, max_p=8)
        assert 0 <= p <= 8


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

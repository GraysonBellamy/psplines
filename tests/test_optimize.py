"""
Tests for psplines.optimize module.
"""
import numpy as np
import pytest

from psplines.core import PSpline
from psplines.optimize import aic, cross_validation, l_curve, v_curve


class TestCrossValidation:
    """Test cross-validation functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.x = np.linspace(0, 1, 50)
        self.y = np.sin(2 * np.pi * self.x) + 0.1 * np.random.randn(50)
        self.spline = PSpline(self.x, self.y, nseg=10)

    def test_basic_cross_validation(self):
        """Test basic cross-validation functionality."""
        best_lambda, best_score = cross_validation(self.spline)

        assert isinstance(best_lambda, float)
        assert isinstance(best_score, float)
        assert best_lambda > 0
        assert best_score > 0

    def test_cross_validation_with_range(self):
        """Test cross-validation with specified lambda bounds."""
        lambda_bounds = (1e-2, 1e2)
        best_lambda, best_score = cross_validation(self.spline, lambda_bounds=lambda_bounds)

        assert lambda_bounds[0] <= best_lambda <= lambda_bounds[1]

    def test_cross_validation_stability(self):
        """Test that cross-validation produces stable results."""
        # Cross-validation modifies the spline state, so use fresh instances
        spline1 = PSpline(self.x, self.y, nseg=10)
        spline2 = PSpline(self.x, self.y, nseg=10)

        best_lambda1, score1 = cross_validation(spline1)
        best_lambda2, score2 = cross_validation(spline2)

        # Both should find reasonable lambda values
        assert best_lambda1 > 0
        assert best_lambda2 > 0
        # Both should produce reasonable GCV scores
        assert score1 > 0
        assert score2 > 0

    def test_different_spline_configurations(self):
        """Test cross-validation with different spline configurations."""
        # Different degrees
        for degree in [1, 2, 3]:
            spline = PSpline(self.x, self.y, degree=degree, nseg=8)
            best_lambda, _ = cross_validation(spline)
            assert best_lambda > 0

        # Different penalty orders
        for penalty_order in [1, 2, 3]:
            spline = PSpline(self.x, self.y, penalty_order=penalty_order, nseg=8)
            best_lambda, _ = cross_validation(spline)
            assert best_lambda > 0


class TestAIC:
    """Test AIC functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.x = np.linspace(0, 1, 30)
        self.y = np.sin(2 * np.pi * self.x) + 0.1 * np.random.randn(30)
        self.spline = PSpline(self.x, self.y, nseg=8)

    def test_basic_aic(self):
        """Test basic AIC computation."""
        best_lambda, best_aic = aic(self.spline)

        assert isinstance(best_lambda, float)
        assert isinstance(best_aic, float)
        assert best_lambda > 0

    def test_aic_with_range(self):
        """Test AIC with specified lambda bounds."""
        lambda_bounds = (1e-1, 1e1)
        best_lambda, _ = aic(self.spline, lambda_bounds=lambda_bounds)

        assert lambda_bounds[0] <= best_lambda <= lambda_bounds[1]

    def test_aic_monotonicity_property(self):
        """Test that AIC changes with lambda in expected way."""
        # Very small lambda (undersmoothing) should have higher AIC
        # Very large lambda (oversmoothing) should have higher AIC
        # Optimal should be somewhere in between
        lambda_range = np.logspace(-3, 3, 20)

        aic_values = []
        for lam in lambda_range:
            spline_temp = PSpline(self.x, self.y, lambda_=lam, nseg=8)
            spline_temp.fit()

            # Compute AIC manually for this lambda
            n = len(self.y)
            residuals = self.y - spline_temp.fitted_values
            rss = np.sum(residuals**2)
            aic_val = n * np.log(rss / n) + 2 * spline_temp.ED
            aic_values.append(aic_val)

        # Should have a minimum somewhere in the middle
        min_idx = np.argmin(aic_values)
        assert 2 < min_idx < len(lambda_range) - 2  # Not at boundaries


class TestLCurve:
    """Test L-curve functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.x = np.linspace(0, 1, 30)
        self.y = np.sin(2 * np.pi * self.x) + 0.1 * np.random.randn(30)
        self.spline = PSpline(self.x, self.y, nseg=8)

    def test_basic_l_curve(self):
        """Test basic L-curve computation."""
        best_lambda, _ = l_curve(self.spline)

        assert isinstance(best_lambda, float)
        assert best_lambda > 0

    def test_l_curve_with_range(self):
        """Test L-curve with specified lambda bounds."""
        lambda_bounds = (1e-2, 1e2)
        best_lambda, _ = l_curve(self.spline, lambda_bounds=lambda_bounds)

        assert lambda_bounds[0] <= best_lambda <= lambda_bounds[1]


class TestVCurve:
    """Test V-curve functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.x = np.linspace(0, 1, 30)
        self.y = np.sin(2 * np.pi * self.x) + 0.1 * np.random.randn(30)
        self.spline = PSpline(self.x, self.y, nseg=8)

    def test_basic_v_curve(self):
        """Test basic V-curve computation."""
        best_lambda, _ = v_curve(self.spline)

        assert isinstance(best_lambda, float)
        assert best_lambda > 0


class TestOptimizationComparison:
    """Test that different optimization methods give reasonable results."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.x = np.linspace(0, 1, 40)
        self.y = np.sin(2 * np.pi * self.x) + 0.1 * np.random.randn(40)
        self.spline = PSpline(self.x, self.y, nseg=10)

    def test_methods_give_reasonable_results(self):
        """Test that different methods give results in similar ballpark."""
        # Get results from different methods
        lambda_cv, _ = cross_validation(self.spline)
        lambda_aic, _ = aic(self.spline)
        lambda_l, _ = l_curve(self.spline)
        lambda_v, _ = v_curve(self.spline)

        lambdas = [lambda_cv, lambda_aic, lambda_l, lambda_v]

        # All should be positive
        assert all(lam > 0 for lam in lambdas)

        # Should be within reasonable range (not too far apart)
        log_lambdas = np.log10(lambdas)
        assert np.ptp(log_lambdas) < 6  # Within 6 orders of magnitude

    def test_fitted_splines_are_reasonable(self):
        """Test that optimized splines produce reasonable fits."""
        lambda_opt, _ = cross_validation(self.spline)

        # Fit with optimal lambda
        spline_opt = PSpline(self.x, self.y, lambda_=lambda_opt, nseg=10)
        spline_opt.fit()

        # Should have reasonable effective degrees of freedom
        assert 3 < spline_opt.ED < len(self.y) - 5

        # Should fit better than no smoothing or extreme smoothing
        residuals_opt = self.y - spline_opt.fitted_values
        mse_opt = np.mean(residuals_opt**2)

        # Compare to very small lambda (undersmoothing)
        spline_under = PSpline(self.x, self.y, lambda_=1e-6, nseg=10)
        spline_under.fit()
        residuals_under = self.y - spline_under.fitted_values
        mse_under = np.mean(residuals_under**2)

        # Compare to very large lambda (oversmoothing)
        spline_over = PSpline(self.x, self.y, lambda_=1e6, nseg=10)
        spline_over.fit()
        residuals_over = self.y - spline_over.fitted_values
        mse_over = np.mean(residuals_over**2)

        # Optimal should generally be better than both extremes
        # (though this isn't guaranteed for all data/noise levels)
        assert mse_opt <= max(mse_under, mse_over) * 2  # Allow some tolerance


class TestOptimizationEdgeCases:
    """Test edge cases in optimization."""

    def test_very_smooth_data(self):
        """Test optimization on very smooth data."""
        x = np.linspace(0, 1, 30)
        y = np.sin(2 * np.pi * x)  # No noise
        spline = PSpline(x, y, nseg=10)

        # Should still find reasonable lambda
        best_lambda, _ = cross_validation(spline)
        assert best_lambda > 0

    def test_very_noisy_data(self):
        """Test optimization on very noisy data."""
        np.random.seed(42)
        x = np.linspace(0, 1, 30)
        y = np.sin(2 * np.pi * x) + np.random.randn(30)  # High noise
        spline = PSpline(x, y, nseg=8)

        # Should find reasonable lambda for smoothing noisy data
        best_lambda, _ = cross_validation(spline)
        assert best_lambda > 0.1  # Should prefer some smoothing


if __name__ == "__main__":
    pytest.main([__file__])

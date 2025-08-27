"""
Comprehensive tests for psplines.core module.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from psplines.core import PSpline


class TestPSplineInitialization:
    """Test PSpline initialization and validation."""

    def test_basic_initialization(self):
        """Test basic valid initialization."""
        x = np.linspace(0, 1, 10)
        y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(10)

        spline = PSpline(x, y)
        assert spline.nseg == 20
        assert spline.degree == 3
        assert spline.lambda_ == 10.0
        assert spline.penalty_order == 2

    def test_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            PSpline([], [])

    def test_mismatched_lengths(self):
        """Test that mismatched x and y lengths raise ValueError."""
        with pytest.raises(ValueError, match="x and y must have the same length"):
            PSpline([1, 2, 3], [1, 2])

    def test_insufficient_data_points(self):
        """Test that insufficient data points raise ValueError."""
        with pytest.raises(ValueError, match="Need at least 2 data points"):
            PSpline([1], [1])

    def test_non_finite_values(self):
        """Test that non-finite values raise ValueError."""
        with pytest.raises(ValueError, match="x contains non-finite values"):
            PSpline([1, np.inf, 3], [1, 2, 3])

        with pytest.raises(ValueError, match="y contains non-finite values"):
            PSpline([1, 2, 3], [1, np.nan, 3])

    def test_non_unique_x_values(self):
        """Test that non-unique x values raise ValueError."""
        with pytest.raises(ValueError, match="x must contain at least 2 unique values"):
            PSpline([1, 1, 1], [1, 2, 3])

    def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        x, y = [1, 2, 3], [1, 2, 3]

        with pytest.raises(ValueError, match="nseg must be positive"):
            PSpline(x, y, nseg=-1)

        with pytest.raises(ValueError, match="degree must be non-negative"):
            PSpline(x, y, degree=-1)

        with pytest.raises(ValueError, match="lambda_ must be positive"):
            PSpline(x, y, lambda_=-1)

        with pytest.raises(ValueError, match="penalty_order must be >= 1"):
            PSpline(x, y, penalty_order=0)

        with pytest.raises(ValueError, match="nseg .* must be greater than degree"):
            PSpline(x, y, nseg=2, degree=3)


class TestPSplineFitting:
    """Test PSpline fitting functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.x = np.linspace(0, 1, 50)
        self.y = np.sin(2 * np.pi * self.x) + 0.1 * np.random.randn(50)
        self.spline = PSpline(self.x, self.y, nseg=15)

    def test_basic_fitting(self):
        """Test basic fitting functionality."""
        result = self.spline.fit()

        assert result is self.spline  # Returns self
        assert self.spline.coef is not None
        assert self.spline.fitted_values is not None
        assert self.spline.B is not None
        assert self.spline.knots is not None
        assert self.spline.ED is not None
        assert self.spline.sigma2 is not None
        assert self.spline.se_coef is not None
        assert self.spline.se_fitted is not None

    def test_fitting_with_domain_specification(self):
        """Test fitting with explicit domain boundaries."""
        self.spline.fit(xl=-0.1, xr=1.1)

        assert self.spline._xl == -0.1
        assert self.spline._xr == 1.1

    def test_invalid_domain_boundaries(self):
        """Test that invalid domain boundaries raise ValueError."""
        spline = PSpline(self.x, self.y)

        with pytest.raises(ValueError, match="xl must be finite"):
            spline.fit(xl=np.inf)

        with pytest.raises(ValueError, match="xl .* must be <= min\\(x\\)"):
            spline.fit(xl=0.5)  # x starts at 0

        with pytest.raises(ValueError, match="xr .* must be >= max\\(x\\)"):
            spline.fit(xr=0.5)  # x goes to 1

        # Note: Testing xl >= xr is impossible with current validation logic since
        # xl must be <= min(x) and xr must be >= max(x), which guarantees xl < xr

    def test_fitting_quality(self):
        """Test that fitting produces reasonable results."""
        self.spline.fit()

        # Check that fitted values are reasonable
        assert self.spline.fitted_values.shape == self.y.shape

        # Check that residuals are small for a smooth function
        residuals = self.y - self.spline.fitted_values
        mse = np.mean(residuals**2)
        assert mse < 1.0  # Should be much smaller for this smooth function

        # Check that effective degrees of freedom is reasonable
        assert 3 < self.spline.ED < len(self.y)

        # Check that sigma2 is positive
        assert self.spline.sigma2 > 0

    def test_different_penalty_orders(self):
        """Test fitting with different penalty orders."""
        for penalty_order in [1, 2, 3]:
            spline = PSpline(self.x, self.y, penalty_order=penalty_order)
            spline.fit()
            assert spline.coef is not None

    def test_different_degrees(self):
        """Test fitting with different spline degrees."""
        for degree in [1, 2, 3, 4]:
            spline = PSpline(self.x, self.y, degree=degree, nseg=15)
            spline.fit()
            assert spline.coef is not None


class TestPSplinePrediction:
    """Test PSpline prediction functionality."""

    def setup_method(self):
        """Set up test data and fitted spline."""
        np.random.seed(42)
        self.x = np.linspace(0, 1, 30)
        self.y = np.sin(2 * np.pi * self.x) + 0.05 * np.random.randn(30)
        self.spline = PSpline(self.x, self.y, nseg=10).fit()
        self.x_new = np.linspace(0.1, 0.9, 20)

    def test_prediction_before_fitting(self):
        """Test that prediction fails before fitting."""
        unfitted_spline = PSpline(self.x, self.y)
        with pytest.raises(RuntimeError, match="Model not fitted"):
            unfitted_spline.predict(self.x_new)

    def test_basic_prediction(self):
        """Test basic prediction functionality."""
        y_pred = self.spline.predict(self.x_new)

        assert y_pred.shape == (20,)
        assert np.all(np.isfinite(y_pred))

    def test_prediction_with_standard_errors(self):
        """Test prediction with analytical standard errors."""
        y_pred, se = self.spline.predict(self.x_new, return_se=True)

        assert y_pred.shape == (20,)
        assert se.shape == (20,)
        assert np.all(se > 0)  # Standard errors should be positive
        assert np.all(np.isfinite(se))

    def test_prediction_input_validation(self):
        """Test prediction input validation."""
        with pytest.raises(ValueError, match="x_new cannot be empty"):
            self.spline.predict([])

        with pytest.raises(ValueError, match="x_new contains non-finite values"):
            self.spline.predict([0.5, np.nan])

        with pytest.raises(ValueError, match="derivative_order must be positive"):
            self.spline.predict(self.x_new, derivative_order=0)

        with pytest.raises(ValueError, match="se_method must be"):
            self.spline.predict(self.x_new, se_method="invalid")

        with pytest.raises(ValueError, match="B_boot must be positive"):
            self.spline.predict(self.x_new, se_method="bootstrap", return_se=True, B_boot=-1)

    def test_prediction_consistency(self):
        """Test that predictions are consistent with fitted values."""
        y_pred = self.spline.predict(self.x)
        assert_allclose(y_pred, self.spline.fitted_values, rtol=1e-10)


class TestPSplineDerivatives:
    """Test PSpline derivative computation."""

    def setup_method(self):
        """Set up test data and fitted spline."""
        np.random.seed(42)
        self.x = np.linspace(0, 1, 30)
        self.y = np.sin(2 * np.pi * self.x) + 0.05 * np.random.randn(30)
        self.spline = PSpline(self.x, self.y, nseg=10).fit()
        self.x_new = np.linspace(0.1, 0.9, 20)

    def test_derivative_before_fitting(self):
        """Test that derivative fails before fitting."""
        unfitted_spline = PSpline(self.x, self.y)
        with pytest.raises(RuntimeError, match="Model not fitted"):
            unfitted_spline.derivative(self.x_new)

    def test_first_derivative(self):
        """Test first derivative computation."""
        dy_dx = self.spline.derivative(self.x_new, deriv_order=1)

        assert dy_dx.shape == (20,)
        assert np.all(np.isfinite(dy_dx))

    def test_second_derivative(self):
        """Test second derivative computation."""
        d2y_dx2 = self.spline.derivative(self.x_new, deriv_order=2)

        assert d2y_dx2.shape == (20,)
        assert np.all(np.isfinite(d2y_dx2))

    def test_derivative_with_standard_errors(self):
        """Test derivative computation with standard errors."""
        dy_dx, se = self.spline.derivative(self.x_new, return_se=True)

        assert dy_dx.shape == (20,)
        assert se.shape == (20,)
        assert np.all(se > 0)

    def test_derivative_input_validation(self):
        """Test derivative input validation."""
        with pytest.raises(ValueError, match="x_new cannot be empty"):
            self.spline.derivative([])

        with pytest.raises(ValueError, match="deriv_order must be positive"):
            self.spline.derivative(self.x_new, deriv_order=0)

    def test_derivative_approximation_quality(self):
        """Test that derivatives approximate analytical derivatives reasonably well."""
        # For sin(2*pi*x), first derivative should be approximately 2*pi*cos(2*pi*x)
        x_test = np.array([0.25, 0.5, 0.75])  # Points where we know the derivative
        dy_dx = self.spline.derivative(x_test, deriv_order=1)

        # At these points, analytical derivative of sin(2*pi*x) is:
        analytical = 2 * np.pi * np.cos(2 * np.pi * x_test)

        # Allow reasonable tolerance due to noise and smoothing
        assert_allclose(dy_dx, analytical, atol=3.0)


class TestPSplineEdgeCases:
    """Test edge cases and error conditions."""

    def test_linear_data(self):
        """Test fitting to perfectly linear data."""
        x = np.linspace(0, 1, 20)
        y = 2 * x + 1  # Perfect line

        spline = PSpline(x, y, lambda_=1e-6).fit()  # Very small smoothing
        y_pred = spline.predict(x)

        # Should fit linear data very well
        assert_allclose(y_pred, y, rtol=1e-3)

    def test_constant_data(self):
        """Test fitting to constant data."""
        x = np.linspace(0, 1, 20)
        y = np.ones(20) * 5

        spline = PSpline(x, y).fit()
        y_pred = spline.predict(x)

        # Should fit constant data well
        assert_allclose(y_pred, y, rtol=1e-2)

    def test_small_dataset(self):
        """Test with minimal dataset size."""
        x = np.array([0, 1])
        y = np.array([0, 1])

        # Should work with nseg and degree appropriate for small dataset
        spline = PSpline(x, y, nseg=2, degree=1).fit()
        assert spline.coef is not None

    def test_high_noise(self):
        """Test with high noise data."""
        np.random.seed(42)
        x = np.linspace(0, 1, 50)
        y = np.sin(2 * np.pi * x) + 2 * np.random.randn(50)  # Very noisy

        spline = PSpline(x, y, lambda_=100).fit()  # High smoothing
        assert spline.sigma2 > 1  # Should detect high noise
        assert spline.coef is not None


class TestPSplineTypes:
    """Test type-related functionality."""

    def test_array_like_inputs(self):
        """Test that various array-like inputs work."""
        # List inputs
        spline1 = PSpline([0, 1, 2, 3], [0, 1, 4, 9])
        assert spline1.x.dtype == float

        # Tuple inputs
        spline2 = PSpline((0, 1, 2, 3), (0, 1, 4, 9))
        assert spline2.y.dtype == float

        # Mixed numeric types
        spline3 = PSpline([0, 1, 2, 3], [0.0, 1.0, 4.0, 9.0])
        spline3.fit()
        assert spline3.coef is not None


if __name__ == "__main__":
    pytest.main([__file__])

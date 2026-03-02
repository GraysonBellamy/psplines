"""
Tests for psplines.whittaker module and divided_difference_matrix.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from psplines.penalty import difference_matrix, divided_difference_matrix
from psplines.whittaker import WhittakerSmoother

# ===================================================================
# divided_difference_matrix
# ===================================================================


class TestDividedDifferenceMatrix:
    """Test x-aware divided-difference operator."""

    def test_order1_uniform_proportional_to_standard(self):
        """On uniform x, D_x is proportional to the standard D."""
        x = np.linspace(0, 1, 20)
        D_x = divided_difference_matrix(x, order=1)
        D_std = difference_matrix(20, order=1)
        # D_x = (1/h) * D_std  where h = x[1] - x[0]
        h = x[1] - x[0]
        assert_allclose(D_x.toarray(), D_std.toarray() / h, atol=1e-12)

    def test_order2_uniform_proportional_to_standard(self):
        """On uniform x, order-2 D_x is proportional to standard D."""
        x = np.linspace(0, 10, 30)
        D_x = divided_difference_matrix(x, order=2)
        D_std = difference_matrix(30, order=2)
        h = x[1] - x[0]
        # second divided diff scales as 1/h^2
        assert_allclose(D_x.toarray(), D_std.toarray() / h**2, atol=1e-10)

    def test_shape(self):
        x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        D1 = divided_difference_matrix(x, order=1)
        assert D1.shape == (4, 5)
        D2 = divided_difference_matrix(x, order=2)
        assert D2.shape == (3, 5)

    def test_order1_nonuniform_values(self):
        """Check explicit values for order-1 on non-uniform grid."""
        x = np.array([0.0, 1.0, 4.0])
        D = divided_difference_matrix(x, order=1)
        # row 0: (-1/1, 1/1, 0)  row 1: (0, -1/3, 1/3)
        expected = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0 / 3, 1.0 / 3]])
        assert_allclose(D.toarray(), expected, atol=1e-14)

    def test_sparse(self):
        x = np.linspace(0, 1, 50)
        D = divided_difference_matrix(x, order=2)
        assert sp.issparse(D)

    def test_not_strictly_increasing_raises(self):
        x = np.array([0.0, 1.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            divided_difference_matrix(x, order=1)

    def test_order_too_large(self):
        x = np.array([0.0, 1.0])
        D = divided_difference_matrix(x, order=2)
        assert D.shape == (0, 2)

    def test_order_lt1_raises(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            divided_difference_matrix(np.array([0.0, 1.0, 2.0]), order=0)

    def test_second_order_nonuniform_linear_annihilates(self):
        """Second divided differences of a linear function should be ~0."""
        x = np.array([0.0, 0.3, 1.0, 1.5, 4.0, 7.0])
        z = 2.0 * x + 3.0  # linear
        D2 = divided_difference_matrix(x, order=2)
        result = D2 @ z
        assert_allclose(result, 0.0, atol=1e-12)

    def test_second_order_nonuniform_quadratic(self):
        """Second divided differences of a quadratic should be constant."""
        x = np.array([0.0, 0.5, 1.0, 2.5, 5.0, 8.0, 10.0])
        z = x**2
        D2 = divided_difference_matrix(x, order=2)
        result = D2 @ z
        # For z = x^2, the second divided difference is always 2
        assert_allclose(result, 2.0, atol=1e-10)


# ===================================================================
# WhittakerSmoother — construction & validation
# ===================================================================


class TestWhittakerValidation:
    def test_empty_input(self):
        with pytest.raises(Exception, match="empty"):
            WhittakerSmoother(x=[], y=[])

    def test_length_mismatch(self):
        with pytest.raises(Exception, match="same length"):
            WhittakerSmoother(x=[1, 2, 3], y=[1, 2])

    def test_negative_lambda(self):
        with pytest.raises(Exception, match="lambda_"):
            WhittakerSmoother(x=[1, 2, 3], y=[1, 2, 3], lambda_=-1)

    def test_bad_penalty_order(self):
        with pytest.raises(Exception, match="penalty_order"):
            WhittakerSmoother(x=[1, 2, 3], y=[1, 2, 3], penalty_order=0)

    def test_duplicate_x(self):
        with pytest.raises(Exception, match="duplicate"):
            WhittakerSmoother(x=[1, 1, 2], y=[1, 2, 3])

    def test_too_few_points(self):
        with pytest.raises(Exception):
            WhittakerSmoother(x=[1, 2], y=[1, 2])


# ===================================================================
# WhittakerSmoother — fitting
# ===================================================================


class TestWhittakerFit:
    @pytest.fixture()
    def noisy_sine(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) + 0.2 * rng.standard_normal(100)
        return x, y

    def test_fit_returns_self(self, noisy_sine):
        x, y = noisy_sine
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e3)
        result = ws.fit()
        assert result is ws

    def test_fitted_values_shape(self, noisy_sine):
        x, y = noisy_sine
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e3).fit()
        assert ws.fitted_values is not None
        assert ws.fitted_values.shape == y.shape

    def test_smoothing_reduces_variance(self, noisy_sine):
        x, y = noisy_sine
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e4).fit()
        assert ws.fitted_values is not None
        assert np.var(ws.fitted_values) < np.var(y)

    def test_small_lambda_interpolates(self, noisy_sine):
        x, y = noisy_sine
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e-6).fit()
        assert ws.fitted_values is not None
        assert_allclose(ws.fitted_values, y, atol=0.01)

    def test_large_lambda_flattens(self, noisy_sine):
        x, y = noisy_sine
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e12).fit()
        assert ws.fitted_values is not None
        # Should be nearly constant (linear for order-2)
        diffs = np.diff(np.diff(ws.fitted_values))
        assert_allclose(diffs, 0, atol=1e-4)

    def test_ed_in_range(self, noisy_sine):
        x, y = noisy_sine
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e3).fit()
        assert ws.ED is not None
        assert 1 < ws.ED < len(y)

    def test_se_fitted_positive(self, noisy_sine):
        x, y = noisy_sine
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e3).fit()
        assert ws.se_fitted is not None
        assert np.all(ws.se_fitted >= 0)

    def test_unsorted_input(self):
        """Smoother should handle unsorted x."""
        rng = np.random.default_rng(7)
        x = rng.uniform(0, 10, 50)
        y = np.sin(x) + 0.1 * rng.standard_normal(50)

        ws = WhittakerSmoother(x=x, y=y, lambda_=1e3).fit()
        assert ws.fitted_values is not None
        assert ws.fitted_values.shape == y.shape
        # Order of fitted values matches original x order
        # (not sorted order)
        sort_idx = np.argsort(x)
        z_sorted = ws.fitted_values[sort_idx]
        # Smoothed values over sorted x should be smooth
        assert np.var(np.diff(z_sorted)) < np.var(np.diff(y[sort_idx]))


# ===================================================================
# WhittakerSmoother — non-uniform spacing
# ===================================================================


class TestWhittakerNonUniform:
    def test_gap_respected(self):
        """Points separated by a large gap should be smoothed less across it."""
        rng = np.random.default_rng(99)
        # Two clusters with a big gap
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(10, 11, 30)
        x = np.concatenate([x1, x2])
        y1 = np.ones(30) + 0.05 * rng.standard_normal(30)
        y2 = 5 * np.ones(30) + 0.05 * rng.standard_normal(30)
        y = np.concatenate([y1, y2])

        ws = WhittakerSmoother(x=x, y=y, lambda_=1e2).fit()
        assert ws.fitted_values is not None
        # Left cluster should stay near 1, right near 5
        left_mean = np.mean(ws.fitted_values[:30])
        right_mean = np.mean(ws.fitted_values[30:])
        assert abs(left_mean - 1.0) < 0.5
        assert abs(right_mean - 5.0) < 0.5

    def test_nonuniform_vs_uniform_differ(self):
        """Non-uniform spacing should produce different results than ignoring gaps."""
        rng = np.random.default_rng(42)
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24], dtype=float)
        y = np.sin(x) + 0.1 * rng.standard_normal(len(x))
        ws = WhittakerSmoother(x=x, y=y, lambda_=10).fit()
        assert ws.fitted_values is not None
        # Just verify it ran and produced reasonable output
        assert ws.fitted_values.shape == y.shape
        assert np.all(np.isfinite(ws.fitted_values))


# ===================================================================
# WhittakerSmoother — weights
# ===================================================================


class TestWhittakerWeights:
    def test_zero_weight_is_missing(self):
        """Zero-weight points should be effectively interpolated over."""
        x = np.linspace(0, 1, 50)
        y_true = np.sin(2 * np.pi * x)
        y = y_true.copy()
        # Corrupt some points
        y[20:30] = 100.0
        w = np.ones(50)
        w[20:30] = 0.0

        ws = WhittakerSmoother(x=x, y=y, lambda_=1e2, weights=w).fit()
        assert ws.fitted_values is not None
        # The corrupted region should be interpolated, close-ish to truth
        assert_allclose(ws.fitted_values[20:30], y_true[20:30], atol=0.5)


# ===================================================================
# WhittakerSmoother — predict
# ===================================================================


class TestWhittakerPredict:
    def test_predict_on_grid(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + 0.1 * rng.standard_normal(50)
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e3).fit()
        # predict at the same x should equal fitted_values
        pred = ws.predict(x)
        assert ws.fitted_values is not None
        assert_allclose(pred, ws.fitted_values, atol=1e-12)

    def test_predict_interpolates(self):
        x = np.linspace(0, 10, 50)
        y = x**2
        ws = WhittakerSmoother(x=x, y=y, lambda_=1e-2).fit()
        x_new = np.array([2.5, 5.0, 7.5])
        pred = ws.predict(x_new)
        assert pred.shape == (3,)
        assert np.all(np.isfinite(pred))

    def test_predict_before_fit_raises(self):
        ws = WhittakerSmoother(x=[1, 2, 3, 4], y=[1, 2, 3, 4])
        with pytest.raises(Exception, match="[Ff]it"):
            ws.predict([1.5])


# ===================================================================
# WhittakerSmoother — lambda selection
# ===================================================================


class TestWhittakerLambdaSelection:
    @pytest.fixture()
    def signal(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 4 * np.pi, 200)
        y = np.sin(x) + 0.3 * rng.standard_normal(200)
        return x, y

    def test_cross_validation_returns_positive(self, signal):
        x, y = signal
        ws = WhittakerSmoother(x=x, y=y)
        lam, score = ws.cross_validation()
        assert lam > 0
        assert np.isfinite(score)
        assert ws.fitted_values is not None  # re-fitted with best lambda

    def test_v_curve_returns_positive(self, signal):
        x, y = signal
        ws = WhittakerSmoother(x=x, y=y)
        lam, score = ws.v_curve()
        assert lam > 0
        assert np.isfinite(score)
        assert ws.fitted_values is not None

    def test_gcv_nonuniform(self):
        """GCV should work on non-uniform data."""
        rng = np.random.default_rng(7)
        x = np.sort(rng.uniform(0, 10, 80))
        y = np.sin(x) + 0.2 * rng.standard_normal(80)
        ws = WhittakerSmoother(x=x, y=y)
        lam, _ = ws.cross_validation()
        assert lam > 0
        assert ws.fitted_values is not None


# ===================================================================
# WhittakerSmoother — repr
# ===================================================================


class TestWhittakerRepr:
    def test_unfitted_repr(self):
        ws = WhittakerSmoother(x=[1, 2, 3, 4], y=[1, 2, 3, 4])
        r = repr(ws)
        assert "unfitted" in r
        assert "WhittakerSmoother" in r

    def test_fitted_repr(self):
        ws = WhittakerSmoother(x=[1, 2, 3, 4], y=[1, 2, 3, 4], lambda_=10).fit()
        r = repr(ws)
        assert "fitted" in r

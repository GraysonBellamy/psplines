"""
Tests for adaptive and variable penalty functionality (§8.8).

Covers:
  - variable_penalty_matrix correctness
  - adaptive_penalty_matrix correctness
  - Exponential variable penalty fitting (penalty_gamma)
  - Adaptive (locally varying) penalty fitting
  - variable_penalty_cv grid search
  - Edge cases and validation
"""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from psplines.core import PSpline
from psplines.exceptions import ValidationError
from psplines.optimize import variable_penalty_cv
from psplines.penalty import (
    adaptive_penalty_matrix,
    difference_matrix,
    variable_penalty_matrix,
)

# ---------------------------------------------------------------------------
# Tests for variable_penalty_matrix
# ---------------------------------------------------------------------------


class TestVariablePenaltyMatrix:
    """Unit tests for penalty with exponentially varying weights."""

    def test_gamma_zero_equals_standard(self):
        """γ = 0 should give the standard D'D penalty."""
        n, order = 10, 2
        P_var = variable_penalty_matrix(n, order, gamma=0.0)
        D = difference_matrix(n, order)
        P_std = (D.T @ D).tocsr()
        assert_allclose(P_var.toarray(), P_std.toarray(), atol=1e-14)

    def test_positive_gamma_heavier_right(self):
        """γ > 0 increases penalty weights toward the right boundary."""
        n, order = 20, 2
        P_pos = variable_penalty_matrix(n, order, gamma=10.0)
        # Extract effective diagonal weights: compare penalty on left vs right diffs
        # The penalty should be stronger on the right
        alpha_left = np.zeros(n)
        alpha_left[2] = 1.0
        alpha_right = np.zeros(n)
        alpha_right[-3] = 1.0
        pen_left = float(alpha_left @ P_pos @ alpha_left)
        pen_right = float(alpha_right @ P_pos @ alpha_right)
        assert pen_right > pen_left

    def test_negative_gamma_heavier_left(self):
        """γ < 0 increases penalty weights toward the left boundary."""
        n, order = 20, 2
        P_neg = variable_penalty_matrix(n, order, gamma=-10.0)
        alpha_left = np.zeros(n)
        alpha_left[2] = 1.0
        alpha_right = np.zeros(n)
        alpha_right[-3] = 1.0
        pen_left = float(alpha_left @ P_neg @ alpha_left)
        pen_right = float(alpha_right @ P_neg @ alpha_right)
        assert pen_left > pen_right

    def test_shape_and_symmetry(self):
        """Result should be square, symmetric, PSD."""
        n = 15
        P = variable_penalty_matrix(n, 2, gamma=5.0)
        assert P.shape == (n, n)
        diff = P - P.T
        assert sp.linalg.norm(diff) < 1e-12
        eigvals = np.linalg.eigvalsh(P.toarray())
        assert np.all(eigvals >= -1e-12)

    def test_edge_order_ge_n(self):
        """Order >= n returns zero matrix."""
        P = variable_penalty_matrix(3, 5, gamma=1.0)
        assert P.shape == (3, 3)
        assert_allclose(P.toarray(), 0.0)


# ---------------------------------------------------------------------------
# Tests for adaptive_penalty_matrix
# ---------------------------------------------------------------------------


class TestAdaptivePenaltyMatrix:
    """Unit tests for per-segment weighted penalty."""

    def test_uniform_weights_equals_standard(self):
        """All-ones weights should give the standard D'D penalty."""
        n, order = 12, 2
        w = np.ones(n - order)
        P_adp = adaptive_penalty_matrix(n, order, weights=w)
        D = difference_matrix(n, order)
        P_std = (D.T @ D).tocsr()
        assert_allclose(P_adp.toarray(), P_std.toarray(), atol=1e-14)

    def test_none_weights_equals_standard(self):
        """None weights should give the standard D'D penalty."""
        n, order = 12, 2
        P_adp = adaptive_penalty_matrix(n, order, weights=None)
        D = difference_matrix(n, order)
        P_std = (D.T @ D).tocsr()
        assert_allclose(P_adp.toarray(), P_std.toarray(), atol=1e-14)

    def test_zero_weight_removes_penalty(self):
        """Zero weight on a segment should remove its penalty contribution."""
        n, order = 8, 1
        w = np.ones(n - order)
        w[3] = 0.0  # zero out one diff
        P = adaptive_penalty_matrix(n, order, weights=w)
        # The full penalty with all ones
        P_full = adaptive_penalty_matrix(n, order, weights=np.ones(n - order))
        # P should have smaller norm
        assert sp.linalg.norm(P) < sp.linalg.norm(P_full)

    def test_wrong_length_raises(self):
        """Wrong weight length raises ValueError."""
        with pytest.raises(ValueError, match="weights length"):
            adaptive_penalty_matrix(10, 2, weights=np.ones(5))

    def test_negative_weights_raise(self):
        """Negative weights raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            adaptive_penalty_matrix(10, 2, weights=-np.ones(8))

    def test_shape_and_symmetry(self):
        """Result should be square, symmetric, PSD."""
        rng = np.random.default_rng(42)
        n = 15
        w = rng.uniform(0.1, 10, size=n - 2)
        P = adaptive_penalty_matrix(n, 2, weights=w)
        assert P.shape == (n, n)
        diff = P - P.T
        assert sp.linalg.norm(diff) < 1e-12
        eigvals = np.linalg.eigvalsh(P.toarray())
        assert np.all(eigvals >= -1e-12)


# ---------------------------------------------------------------------------
# Tests for variable penalty PSpline fits
# ---------------------------------------------------------------------------


class TestVariablePenaltyFit:
    """Integration tests for PSpline with penalty_gamma."""

    def setup_method(self):
        np.random.seed(42)

    def test_gamma_zero_matches_standard(self):
        """penalty_gamma=0 should give same result as standard fit."""
        x = np.linspace(0, 1, 50)
        y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(50)

        ps_std = PSpline(x, y, lambda_=10).fit()
        ps_var = PSpline(x, y, lambda_=10, penalty_gamma=0.0).fit()

        assert_allclose(ps_std.fitted_values, ps_var.fitted_values, atol=1e-10)

    def test_variable_penalty_fits(self):
        """Variable penalty should produce a valid fit."""
        x = np.linspace(0, 1, 80)
        y = np.sin(4 * np.pi * x) * np.exp(-2 * x) + 0.1 * np.random.randn(80)

        ps = PSpline(x, y, lambda_=1, penalty_gamma=5.0)
        ps.fit()

        assert ps.fitted_values is not None
        assert ps.ED is not None
        assert ps.se_fitted is not None

    def test_variable_penalty_predict(self):
        """Predict from variable-penalty model."""
        x = np.linspace(0, 1, 50)
        y = x**2 + 0.05 * np.random.randn(50)

        ps = PSpline(x, y, lambda_=1, penalty_gamma=3.0).fit()
        x_new = np.linspace(0, 1, 20)
        y_hat = ps.predict(x_new)
        assert y_hat.shape == (20,)


# ---------------------------------------------------------------------------
# Tests for adaptive PSpline fits
# ---------------------------------------------------------------------------


class TestAdaptivePenaltyFit:
    """Integration tests for PSpline with adaptive=True."""

    def setup_method(self):
        np.random.seed(42)

    def test_adaptive_basic_fit(self):
        """Adaptive fit on simple data should converge."""
        x = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(100)

        ps = PSpline(x, y, lambda_=10, adaptive=True)
        ps.fit()

        assert ps.fitted_values is not None
        assert ps.coef is not None
        assert ps._adaptive_weights is not None
        assert ps.ED is not None

    def test_adaptive_weights_vary(self):
        """Adaptive weights should not be uniform for heterogeneous data."""
        x = np.linspace(0, 1, 200)
        # Signal with varying frequency: smooth left, wiggly right
        y = np.where(x < 0.5, x, x + 0.3 * np.sin(20 * np.pi * x))
        y += 0.05 * np.random.randn(200)

        ps = PSpline(x, y, lambda_=1, adaptive=True, adaptive_nseg=8)
        ps.fit()

        w = ps._adaptive_weights
        assert w is not None
        # Weights should vary (not all identical)
        assert np.std(w) > 0.01

    def test_adaptive_predict(self):
        """Predict from adaptive-penalty model."""
        x = np.linspace(0, 1, 80)
        y = x**2 + 0.1 * np.random.randn(80)

        ps = PSpline(x, y, lambda_=5, adaptive=True).fit()
        x_new = np.linspace(0, 1, 30)
        y_hat = ps.predict(x_new)
        assert y_hat.shape == (30,)

    def test_adaptive_with_se(self):
        """Adaptive fit should compute valid SEs."""
        x = np.linspace(0, 1, 60)
        y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(60)

        ps = PSpline(x, y, lambda_=5, adaptive=True).fit()
        assert ps.se_coef is not None
        assert np.all(ps.se_coef > 0)
        assert ps.se_fitted is not None

    def test_adaptive_invalid_nseg(self):
        """Invalid adaptive_nseg raises ValidationError."""
        x = np.linspace(0, 1, 20)
        y = np.random.randn(20)
        with pytest.raises(ValidationError, match="adaptive_nseg"):
            PSpline(x, y, adaptive=True, adaptive_nseg=0)


# ---------------------------------------------------------------------------
# Tests for variable_penalty_cv optimizer
# ---------------------------------------------------------------------------


class TestVariablePenaltyCV:
    """Tests for the 2D grid search optimizer."""

    def setup_method(self):
        np.random.seed(42)

    def test_basic_grid_search(self):
        """Grid search returns valid results."""
        x = np.linspace(0, 1, 60)
        y = np.sin(4 * np.pi * x) * np.exp(-2 * x) + 0.1 * np.random.randn(60)

        ps = PSpline(x, y, lambda_=1).fit()
        best_lam, best_gamma, best_score, scores = variable_penalty_cv(
            ps,
            gamma_range=(-5.0, 5.0),
            lambda_bounds=(1e-2, 1e4),
            num_gamma=11,
            num_lambda=11,
            criterion="gcv",
        )

        assert best_lam > 0
        assert np.isfinite(best_gamma)
        assert np.isfinite(best_score)
        assert scores.shape == (11, 11)

    def test_aic_criterion(self):
        """AIC criterion should also work."""
        x = np.linspace(0, 1, 50)
        y = x**2 + 0.1 * np.random.randn(50)

        ps = PSpline(x, y, lambda_=1).fit()
        best_lam, best_gamma, best_score, scores = variable_penalty_cv(
            ps,
            gamma_range=(-3.0, 3.0),
            lambda_bounds=(1e-1, 1e3),
            num_gamma=7,
            num_lambda=7,
            criterion="aic",
        )

        assert best_lam > 0
        assert np.isfinite(best_score)

    def test_invalid_criterion(self):
        """Invalid criterion raises error."""
        x = np.linspace(0, 1, 20)
        y = np.random.randn(20)
        ps = PSpline(x, y).fit()
        with pytest.raises(Exception, match="criterion"):
            variable_penalty_cv(ps, criterion="invalid")

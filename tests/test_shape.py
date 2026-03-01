"""
Tests for shape constraint functionality (§8.7).

Covers:
  - asymmetric_penalty_matrix correctness
  - Monotone increasing / decreasing fits
  - Convex / concave fits
  - Selective domain constraints via mask
  - Combined constraints (increasing + convex)
  - GLM + shape constraints (Poisson monotone)
  - Flat-slope subdomain penalty (slope_zero)
  - Uncertainty computation with shape penalties
"""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from psplines.core import PSpline
from psplines.exceptions import ValidationError
from psplines.penalty import (
    VALID_SHAPE_TYPES,
    asymmetric_penalty_matrix,
)

# ---------------------------------------------------------------------------
# Tests for asymmetric_penalty_matrix
# ---------------------------------------------------------------------------


class TestAsymmetricPenaltyMatrix:
    """Unit tests for the asymmetric penalty matrix builder."""

    def test_increasing_violation(self):
        """Non-increasing alpha should produce non-zero penalty."""
        alpha = np.array([1.0, 2.0, 1.5, 3.0, 2.5])  # violations at idx 1,3
        P = asymmetric_penalty_matrix(alpha, "increasing")
        assert sp.issparse(P)
        assert P.shape == (5, 5)
        # Penalty should be PSD
        eigvals = np.linalg.eigvalsh(P.toarray())
        assert np.all(eigvals >= -1e-12)
        # Penalty norm should be > 0 (violations exist)
        assert sp.linalg.norm(P) > 0

    def test_increasing_no_violation(self):
        """Strictly increasing alpha ⇒ zero penalty."""
        alpha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        P = asymmetric_penalty_matrix(alpha, "increasing")
        assert_allclose(P.toarray(), 0.0, atol=1e-14)

    def test_decreasing_violation(self):
        """Non-decreasing alpha should produce non-zero penalty for 'decreasing'."""
        alpha = np.array([5.0, 4.0, 4.5, 3.0, 3.5])  # violations at idx 1,3
        P = asymmetric_penalty_matrix(alpha, "decreasing")
        assert sp.linalg.norm(P) > 0

    def test_decreasing_no_violation(self):
        """Strictly decreasing alpha ⇒ zero penalty."""
        alpha = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        P = asymmetric_penalty_matrix(alpha, "decreasing")
        assert_allclose(P.toarray(), 0.0, atol=1e-14)

    def test_convex_violation(self):
        """Non-convex alpha should produce non-zero penalty."""
        # Concave parabola: violations everywhere
        j = np.arange(5, dtype=float)
        alpha = -((j - 2) ** 2) + 4
        P = asymmetric_penalty_matrix(alpha, "convex")
        assert sp.linalg.norm(P) > 0

    def test_convex_no_violation(self):
        """Convex alpha ⇒ zero penalty."""
        j = np.arange(5, dtype=float)
        alpha = (j - 2) ** 2  # convex parabola
        P = asymmetric_penalty_matrix(alpha, "convex")
        assert_allclose(P.toarray(), 0.0, atol=1e-14)

    def test_concave_no_violation(self):
        """Concave alpha ⇒ zero penalty."""
        j = np.arange(5, dtype=float)
        alpha = -((j - 2) ** 2) + 4
        P = asymmetric_penalty_matrix(alpha, "concave")
        assert_allclose(P.toarray(), 0.0, atol=1e-14)

    def test_nonneg_violation(self):
        """Negative alpha ⇒ non-zero penalty for 'nonneg'."""
        alpha = np.array([1.0, -0.5, 2.0, -1.0, 0.5])
        P = asymmetric_penalty_matrix(alpha, "nonneg")
        assert sp.linalg.norm(P) > 0

    def test_nonneg_no_violation(self):
        """All-positive alpha ⇒ zero penalty for 'nonneg'."""
        alpha = np.array([1.0, 2.0, 0.5, 3.0, 0.1])
        P = asymmetric_penalty_matrix(alpha, "nonneg")
        assert_allclose(P.toarray(), 0.0, atol=1e-14)

    def test_invalid_type(self):
        """Unknown constraint type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown constraint_type"):
            asymmetric_penalty_matrix(np.ones(5), "invalid")

    def test_mask_selective(self):
        """Mask zeros out penalty in selected region."""
        alpha = np.array(
            [3.0, 2.0, 1.0, 0.0, -1.0]
        )  # decreasing → all violations for 'increasing'
        mask = np.array([True, False, False, False])  # only first diff active
        P_masked = asymmetric_penalty_matrix(alpha, "increasing", mask=mask)
        P_full = asymmetric_penalty_matrix(alpha, "increasing")
        # masked should have smaller norm
        assert sp.linalg.norm(P_masked) < sp.linalg.norm(P_full)
        # masked should still be nonzero (first diff violates)
        assert sp.linalg.norm(P_masked) > 0

    def test_mask_wrong_length(self):
        """Wrong mask length raises ValueError."""
        with pytest.raises(ValueError, match="mask length"):
            asymmetric_penalty_matrix(
                np.ones(5), "increasing", mask=np.ones(2, dtype=bool)
            )

    def test_symmetry(self):
        """Result should be symmetric."""
        rng = np.random.default_rng(42)
        alpha = rng.standard_normal(10)
        for ctype in VALID_SHAPE_TYPES:
            P = asymmetric_penalty_matrix(alpha, ctype)
            diff = P - P.T
            assert sp.linalg.norm(diff) < 1e-12, f"Not symmetric for {ctype}"


# ---------------------------------------------------------------------------
# Tests for shape-constrained PSpline fitting
# ---------------------------------------------------------------------------


class TestShapeConstrainedFit:
    """Integration tests for PSpline with shape constraints."""

    def setup_method(self):
        np.random.seed(42)

    def test_monotone_increasing(self):
        """Fit to noisy monotone data should produce monotone output."""
        x = np.linspace(0, 1, 100)
        y_true = 3 * x + 1
        y = y_true + 0.2 * np.random.randn(100)

        ps = PSpline(x, y, lambda_=10, shape=[{"type": "increasing"}])
        ps.fit()

        # Fitted values should be (approximately) non-decreasing
        diffs = np.diff(ps.fitted_values)
        assert np.all(diffs >= -1e-6), f"Min diff: {diffs.min()}"

    def test_monotone_decreasing(self):
        """Fit to noisy decreasing data should produce decreasing output."""
        x = np.linspace(0, 1, 100)
        y_true = -2 * x + 5
        y = y_true + 0.2 * np.random.randn(100)

        ps = PSpline(x, y, lambda_=10, shape=[{"type": "decreasing"}])
        ps.fit()

        diffs = np.diff(ps.fitted_values)
        assert np.all(diffs <= 1e-6), f"Max diff: {diffs.max()}"

    def test_convex_fit(self):
        """Convex constraint on convex-ish data."""
        x = np.linspace(-1, 1, 80)
        y_true = x**2
        y = y_true + 0.1 * np.random.randn(80)

        ps = PSpline(x, y, lambda_=1, shape=[{"type": "convex"}])
        ps.fit()

        # Check second differences of fitted are >= 0
        fitted = ps.fitted_values
        second_diffs = fitted[2:] - 2 * fitted[1:-1] + fitted[:-2]
        assert np.all(second_diffs >= -1e-4)

    def test_combined_increasing_convex(self):
        """Combined increasing + convex constraints."""
        x = np.linspace(0, 2, 80)
        y_true = np.exp(x)
        y = y_true + 0.3 * np.random.randn(80)

        ps = PSpline(
            x,
            y,
            lambda_=1,
            shape=[{"type": "increasing"}, {"type": "convex"}],
        )
        ps.fit()

        # Should be non-decreasing
        diffs = np.diff(ps.fitted_values)
        assert np.all(diffs >= -1e-5)

    def test_selective_domain_increasing(self):
        """Monotone constraint only on right half of domain."""
        x = np.linspace(0, 2, 100)
        # Dip in first half, increasing in second half
        y_true = np.where(x < 1, np.sin(2 * np.pi * x), x)
        y = y_true + 0.1 * np.random.randn(100)

        ps = PSpline(
            x,
            y,
            lambda_=1,
            shape=[{"type": "increasing", "domain": (1.0, None)}],
        )
        ps.fit()

        # In the constrained region (x >= 1), should be non-decreasing
        mask = x >= 1.0
        fitted_right = ps.fitted_values[mask]
        diffs = np.diff(fitted_right)
        assert np.all(diffs >= -1e-4)

    def test_shape_with_uncertainty(self):
        """Shape-constrained fit should still produce valid SEs."""
        x = np.linspace(0, 1, 50)
        y = 2 * x + 0.1 * np.random.randn(50)

        ps = PSpline(x, y, lambda_=10, shape=[{"type": "increasing"}])
        ps.fit()

        assert ps.ED is not None
        assert ps.sigma2 is not None
        assert ps.se_coef is not None
        assert ps.se_fitted is not None
        assert np.all(ps.se_coef > 0)
        assert np.all(ps.se_fitted >= 0)

    def test_shape_predict(self):
        """Predictions from shape-constrained model should work."""
        x = np.linspace(0, 1, 50)
        y = 2 * x + 0.1 * np.random.randn(50)

        ps = PSpline(x, y, lambda_=10, shape=[{"type": "increasing"}])
        ps.fit()

        x_new = np.linspace(0, 1, 20)
        y_hat = ps.predict(x_new)
        assert y_hat.shape == (20,)
        # Predictions should also be non-decreasing
        assert np.all(np.diff(y_hat) >= -1e-5)

    def test_poisson_monotone(self):
        """Shape constraint with Poisson GLM."""
        np.random.seed(123)
        x = np.linspace(0, 3, 60)
        mu_true = np.exp(0.5 * x)
        y = np.random.poisson(mu_true)

        ps = PSpline(
            x,
            y,
            lambda_=10,
            family="poisson",
            shape=[{"type": "increasing"}],
        )
        ps.fit()

        # Log-fitted should be non-decreasing (on link scale)
        eta = np.log(np.maximum(ps.fitted_values, 1e-10))
        diffs = np.diff(eta)
        assert np.all(diffs >= -0.01)  # some numerical tolerance

    def test_invalid_shape_type(self):
        """Invalid shape type in spec raises ValidationError."""
        x = np.linspace(0, 1, 10)
        y = np.random.randn(10)
        with pytest.raises(ValidationError, match="shape.*type"):
            PSpline(x, y, shape=[{"type": "invalid_type"}])

    def test_invalid_shape_not_list(self):
        """Non-list shape raises ValidationError."""
        x = np.linspace(0, 1, 10)
        y = np.random.randn(10)
        with pytest.raises(ValidationError, match="shape must be a list"):
            PSpline(x, y, shape={"type": "increasing"})


# ---------------------------------------------------------------------------
# Tests for flat-slope subdomain penalty
# ---------------------------------------------------------------------------


class TestSlopeZeroConstraint:
    """Tests for constraints={'slope_zero': {'domain': (lo, hi)}}."""

    def test_slope_zero_at_right(self):
        """Force slope to zero beyond x=0.8."""
        np.random.seed(42)
        x = np.linspace(0, 1, 80)
        y = 3 * x + 0.1 * np.random.randn(80)

        ps = PSpline(
            x,
            y,
            lambda_=1,
            constraints={"slope_zero": {"domain": (0.8, 1.0)}},
        )
        ps.fit()

        # Derivative at x=0.9 should be close to zero
        deriv = ps.derivative(np.array([0.9]))
        assert abs(deriv[0]) < 0.5  # much less than the true slope of 3

    def test_slope_zero_preserves_fit(self):
        """Without constraint, fit should differ from constrained version."""
        np.random.seed(42)
        x = np.linspace(0, 1, 80)
        y = 3 * x + 0.1 * np.random.randn(80)

        ps_free = PSpline(x, y, lambda_=1).fit()
        ps_constrained = PSpline(
            x,
            y,
            lambda_=1,
            constraints={"slope_zero": {"domain": (0.8, 1.0)}},
        ).fit()

        # They should not be identical
        assert not np.allclose(ps_free.fitted_values, ps_constrained.fitted_values)

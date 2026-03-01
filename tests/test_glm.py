"""
Tests for GLM P-spline functionality (Tier 2).

Covers:
  - Poisson P-splines (§2.12.1)
  - Binomial P-splines (§2.12.2)
  - GLM standard errors and ED (§2.12.3)
  - Density estimation (§3.3)
  - Backward compatibility with Gaussian family
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from psplines.core import PSpline
from psplines.density import density_estimate
from psplines.exceptions import ConvergenceError, ValidationError
from psplines.glm import BinomialFamily, GaussianFamily, PoissonFamily, get_family
from psplines.optimize import aic, cross_validation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def coal_mine_data():
    """British coal mine disasters (simplified annual counts)."""
    np.random.seed(42)
    years = np.arange(1851, 1963, dtype=float)
    # Simulated counts loosely inspired by the real data:
    # higher rate early, lower later
    rate = np.exp(0.8 - 0.015 * (years - 1851))
    counts = np.random.poisson(rate)
    return years, counts.astype(float)


@pytest.fixture()
def kyphosis_data():
    """Simplified kyphosis-like binary response data."""
    np.random.seed(123)
    age = np.sort(np.random.uniform(1, 200, 80))
    # True probability peaks around age 100
    eta_true = -3 + 0.06 * age - 0.0003 * age**2
    pi_true = 1.0 / (1.0 + np.exp(-eta_true))
    y = np.random.binomial(1, pi_true).astype(float)
    return age, y


@pytest.fixture()
def bimodal_data():
    """Bimodal data for density estimation (Old Faithful-like)."""
    np.random.seed(42)
    n1 = np.random.normal(2.0, 0.3, 150)
    n2 = np.random.normal(4.5, 0.4, 100)
    return np.concatenate([n1, n2])


# ---------------------------------------------------------------------------
# GLM Family Unit Tests
# ---------------------------------------------------------------------------


class TestFamilies:
    """Test GLM family implementations."""

    def test_get_family_gaussian(self):
        fam = get_family("gaussian")
        assert isinstance(fam, GaussianFamily)
        assert fam.is_gaussian

    def test_get_family_poisson(self):
        fam = get_family("poisson")
        assert isinstance(fam, PoissonFamily)
        assert not fam.is_gaussian

    def test_get_family_binomial(self):
        fam = get_family("binomial")
        assert isinstance(fam, BinomialFamily)
        assert not fam.is_gaussian

    def test_get_family_unknown(self):
        with pytest.raises(ValueError, match="Unknown family"):
            get_family("gamma")

    def test_poisson_initialize(self):
        fam = PoissonFamily()
        y = np.array([0, 1, 5, 10], dtype=float)
        eta, mu, w = fam.initialize(y)
        assert_allclose(eta, np.log(y + 1))
        assert np.all(mu > 0)
        assert np.all(w > 0)

    def test_poisson_inverse_link(self):
        fam = PoissonFamily()
        eta = np.array([-1, 0, 1, 2], dtype=float)
        mu = fam.inverse_link(eta)
        assert_allclose(mu, np.exp(eta))

    def test_poisson_deviance(self):
        fam = PoissonFamily()
        y = np.array([3.0, 5.0, 0.0])
        mu = np.array([3.0, 5.0, 0.1])  # Near-perfect fit for first two
        dev = fam.deviance(y, mu)
        assert dev >= 0
        # Perfect fit should have zero deviance for matched elements
        dev_perfect = fam.deviance(y[:2], mu[:2])
        assert_allclose(dev_perfect, 0.0, atol=1e-10)

    def test_binomial_initialize(self):
        fam = BinomialFamily(trials=np.array([10, 10, 10], dtype=float))
        y = np.array([3, 5, 7], dtype=float)
        eta, mu, w = fam.initialize(y)
        assert np.all(np.isfinite(eta))
        assert np.all(mu > 0)
        assert np.all(w > 0)

    def test_binomial_inverse_link_bernoulli(self):
        fam = BinomialFamily()
        eta = np.array([-5, 0, 5], dtype=float)
        mu = fam.inverse_link(eta)
        # For Bernoulli, mu = pi = sigmoid(eta)
        expected = 1.0 / (1.0 + np.exp(-eta))
        assert_allclose(mu, expected)
        assert np.all(mu > 0)
        assert np.all(mu < 1)

    def test_gaussian_working_response_is_y(self):
        fam = GaussianFamily()
        y = np.array([1.0, 2.0, 3.0])
        eta = y.copy()
        mu = y.copy()
        z = fam.working_response(y, eta, mu)
        assert_allclose(z, y)

    def test_gaussian_phi(self):
        fam = GaussianFamily()
        # dev=10, n=20, ed=5 => phi = 10/(20-5) = 0.667
        phi = fam.phi(10.0, 20, 5.0)
        assert_allclose(phi, 10.0 / 15.0)


# ---------------------------------------------------------------------------
# Poisson P-Spline Tests
# ---------------------------------------------------------------------------


class TestPoissonPSpline:
    """Test Poisson P-spline fitting (§2.12.1)."""

    def test_basic_poisson_fit(self, coal_mine_data):
        """Poisson P-spline converges and produces positive fitted values."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=20, lambda_=100, family="poisson")
        ps.fit()

        assert ps.coef is not None
        assert ps.fitted_values is not None
        assert np.all(ps.fitted_values > 0)  # Poisson: always positive
        assert ps.n_iter_ is not None
        assert ps.n_iter_ < ps.max_iter  # Should converge

    def test_poisson_deviance_stored(self, coal_mine_data):
        """Deviance is computed and stored after fit."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=10, family="poisson").fit()

        assert ps.deviance_ is not None
        assert ps.deviance_ >= 0
        assert ps.phi_ == 1.0  # Poisson has phi=1

    def test_poisson_ed(self, coal_mine_data):
        """Effective dimension is reasonable."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=10, family="poisson").fit()

        assert ps.ED is not None
        assert 2 < ps.ED < len(years)

    def test_poisson_negative_y_rejected(self):
        """Negative counts should be rejected."""
        x = np.arange(10, dtype=float)
        y = np.array([1, 2, -1, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        with pytest.raises(ValidationError, match="non-negative"):
            PSpline(x, y, family="poisson")

    def test_poisson_predict_response_scale(self, coal_mine_data):
        """Predictions on response scale are positive."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=50, family="poisson").fit()

        x_new = np.linspace(years[0], years[-1], 50)
        mu_pred = ps.predict(x_new, type="response")
        assert np.all(mu_pred > 0)

    def test_poisson_predict_link_scale(self, coal_mine_data):
        """Predictions on link scale are log(mu)."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=50, family="poisson").fit()

        x_new = np.linspace(years[0], years[-1], 50)
        eta_pred = ps.predict(x_new, type="link")
        mu_pred = ps.predict(x_new, type="response")
        assert_allclose(np.exp(eta_pred), mu_pred, rtol=1e-10)

    def test_poisson_se_bands(self, coal_mine_data):
        """SE bands on response scale are positive and ordered."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=50, family="poisson").fit()

        x_new = np.linspace(years[5], years[-5], 30)
        mu_hat, lower, upper = ps.predict(x_new, return_se=True, type="response")

        assert np.all(mu_hat > 0)
        assert np.all(lower > 0)
        assert np.all(upper > 0)
        assert np.all(lower <= mu_hat)
        assert np.all(upper >= mu_hat)

    def test_poisson_se_link_scale(self, coal_mine_data):
        """SE on link scale is returned as (eta, se_eta)."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=50, family="poisson").fit()

        x_new = np.linspace(years[5], years[-5], 30)
        eta_hat, se_eta = ps.predict(x_new, return_se=True, type="link")

        assert np.all(np.isfinite(eta_hat))
        assert np.all(se_eta > 0)

    def test_poisson_offset(self):
        """Poisson with exposure offset: mu = exp(B*alpha) * exposure."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        exposure = np.random.uniform(100, 1000, 50)
        true_rate = np.exp(-0.3 + 0.05 * x)
        y = np.random.poisson(true_rate * exposure).astype(float)

        ps = PSpline(
            x,
            y,
            nseg=10,
            lambda_=10,
            family="poisson",
            offset=np.log(exposure),
        ).fit()

        assert ps.coef is not None
        assert ps.n_iter_ < ps.max_iter
        assert np.all(ps.fitted_values > 0)

    def test_poisson_conservation_sum(self):
        """Penalty order d=1: sum of fitted equals sum of observed."""
        np.random.seed(42)
        x = np.arange(50, dtype=float)
        y = np.random.poisson(5, 50).astype(float)

        ps = PSpline(x, y, nseg=20, lambda_=1, penalty_order=1, family="poisson").fit()
        assert_allclose(np.sum(ps.fitted_values), np.sum(y), rtol=0.05)

    def test_poisson_conservation_mean(self):
        """Penalty order d=2: mean preserved."""
        np.random.seed(42)
        x = np.arange(50, dtype=float)
        y = np.random.poisson(5, 50).astype(float)

        ps = PSpline(x, y, nseg=20, lambda_=1, penalty_order=2, family="poisson").fit()
        raw_mean = np.sum(x * y) / np.sum(y)
        fit_mean = np.sum(x * ps.fitted_values) / np.sum(ps.fitted_values)
        assert_allclose(fit_mean, raw_mean, rtol=0.1)

    def test_poisson_working_weights(self, coal_mine_data):
        """At convergence, W diagonal should equal mu."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=10, family="poisson").fit()

        # W stored is diag(mu) (possibly times user weights)
        w_diag = ps._W.diagonal()
        # For pure Poisson (no user weights), W_diag ≈ fitted_values
        assert_allclose(w_diag, ps.fitted_values, rtol=0.01)


# ---------------------------------------------------------------------------
# Binomial P-Spline Tests
# ---------------------------------------------------------------------------


class TestBinomialPSpline:
    """Test Binomial P-spline fitting (§2.12.2)."""

    def test_bernoulli_fit(self, kyphosis_data):
        """Bernoulli P-spline converges with probabilities in [0, 1]."""
        age, y = kyphosis_data
        ps = PSpline(age, y, nseg=15, lambda_=10, family="binomial").fit()

        assert ps.coef is not None
        assert ps.n_iter_ < ps.max_iter
        # Fitted values are mu = pi for Bernoulli
        assert np.all(ps.fitted_values >= 0)
        assert np.all(ps.fitted_values <= 1)

    def test_binomial_grouped(self):
        """Grouped binomial with trials > 1."""
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        trials = np.full(30, 20.0)
        pi_true = 1.0 / (1.0 + np.exp(-(x - 5)))
        y = np.random.binomial(20, pi_true).astype(float)

        ps = PSpline(x, y, nseg=10, lambda_=1, family="binomial", trials=trials).fit()

        assert ps.coef is not None
        # Fitted values are mu = t * pi, so should be in [0, trials]
        assert np.all(ps.fitted_values >= 0)
        assert np.all(ps.fitted_values <= trials + 0.01)

    def test_binomial_predict_response_bounded(self, kyphosis_data):
        """Predictions on response scale are bounded in [0, 1] for Bernoulli."""
        age, y = kyphosis_data
        ps = PSpline(age, y, nseg=15, lambda_=10, family="binomial").fit()

        x_new = np.linspace(age.min(), age.max(), 50)
        pi_pred = ps.predict(x_new, type="response")
        assert np.all(pi_pred >= 0)
        assert np.all(pi_pred <= 1)

    def test_binomial_se_bands_bounded(self, kyphosis_data):
        """SE bands on response scale are bounded."""
        age, y = kyphosis_data
        ps = PSpline(age, y, nseg=15, lambda_=10, family="binomial").fit()

        x_new = np.linspace(age.min() + 5, age.max() - 5, 30)
        pi_hat, lower, upper = ps.predict(x_new, return_se=True, type="response")

        assert np.all(lower >= 0)
        assert np.all(upper <= 1)
        assert np.all(lower <= pi_hat)
        assert np.all(upper >= pi_hat)

    def test_binomial_validation_y_exceeds_trials(self):
        """y > trials should be rejected."""
        x = np.arange(5, dtype=float)
        y = np.array([3, 5, 7, 2, 1], dtype=float)
        trials = np.full(5, 5.0)
        with pytest.raises(ValidationError, match="0 <= y <= trials"):
            PSpline(x, y, family="binomial", trials=trials)

    def test_bernoulli_validation_y_not_01(self):
        """Bernoulli y not in {0, 1} should be rejected."""
        x = np.arange(5, dtype=float)
        y = np.array([0, 1, 2, 0, 1], dtype=float)
        with pytest.raises(ValidationError, match="y in \\{0, 1\\}"):
            PSpline(x, y, family="binomial")

    def test_binomial_deviance(self, kyphosis_data):
        """Deviance is computed for binomial family."""
        age, y = kyphosis_data
        ps = PSpline(age, y, nseg=15, lambda_=10, family="binomial").fit()

        assert ps.deviance_ is not None
        assert ps.deviance_ >= 0
        assert ps.phi_ == 1.0

    def test_binomial_ed(self, kyphosis_data):
        """ED is reasonable for binomial."""
        age, y = kyphosis_data
        ps = PSpline(age, y, nseg=15, lambda_=100, family="binomial").fit()
        assert 1 < ps.ED < len(age)


# ---------------------------------------------------------------------------
# GLM SE and ED Tests
# ---------------------------------------------------------------------------


class TestGLMUncertainty:
    """Test GLM standard errors and effective dimension (§2.12.3)."""

    def test_glm_ed_with_converged_weights(self, coal_mine_data):
        """ED uses converged IRLS weights."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=10, family="poisson").fit()

        # Manually compute ED with stored W
        from psplines.penalty import difference_matrix
        from psplines.utils_math import effective_df

        nb = ps.B.shape[1]
        D = difference_matrix(nb, ps.penalty_order)
        ed_manual = effective_df(ps.B, D, ps.lambda_, W=ps._W)
        assert_allclose(ps.ED, ed_manual, rtol=1e-10)

    def test_poisson_se_coef(self, coal_mine_data):
        """Coefficient SEs are positive and finite."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=10, family="poisson").fit()

        assert ps.se_coef is not None
        assert np.all(ps.se_coef > 0)
        assert np.all(np.isfinite(ps.se_coef))

    def test_predict_type_validation(self, coal_mine_data):
        """Invalid type parameter is rejected."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=10, family="poisson").fit()
        with pytest.raises(ValidationError, match="type must be"):
            ps.predict(years[:5], type="invalid")


# ---------------------------------------------------------------------------
# Optimizer Tests with GLM
# ---------------------------------------------------------------------------


class TestGLMOptimizers:
    """Test lambda selection with GLM families."""

    def test_aic_poisson(self, coal_mine_data):
        """AIC finds reasonable lambda for Poisson P-spline."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=1, family="poisson").fit()

        lam_opt, score = aic(ps)
        assert lam_opt > 0
        assert np.isfinite(score)

    def test_gcv_poisson(self, coal_mine_data):
        """GCV works with Poisson family."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=1, family="poisson").fit()

        lam_opt, score = cross_validation(ps)
        assert lam_opt > 0
        assert np.isfinite(score)

    def test_aic_binomial(self, kyphosis_data):
        """AIC works with binomial family."""
        age, y = kyphosis_data
        ps = PSpline(age, y, nseg=10, lambda_=1, family="binomial").fit()

        lam_opt, score = aic(ps)
        assert lam_opt > 0
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Density Estimation Tests
# ---------------------------------------------------------------------------


class TestDensityEstimation:
    """Test smooth density estimation (§3.3)."""

    def test_density_basic(self, bimodal_data):
        """Density estimation produces valid output."""
        result = density_estimate(bimodal_data, bins=50, nseg=15)

        assert result.grid.shape[0] == 50
        assert result.density.shape[0] == 50
        assert np.all(result.density >= 0)
        assert result.lambda_ > 0

    def test_density_normalization(self, bimodal_data):
        """Density integrates to approximately 1."""
        result = density_estimate(bimodal_data, bins=100, nseg=20)

        integral = np.sum(result.density) * result.bin_width
        assert_allclose(integral, 1.0, rtol=0.05)

    def test_density_bimodal_detection(self, bimodal_data):
        """Density captures bimodal structure."""
        result = density_estimate(bimodal_data, bins=80, nseg=20)

        # Find local maxima
        d = result.density
        peaks = []
        for i in range(1, len(d) - 1):
            if d[i] > d[i - 1] and d[i] > d[i + 1]:
                peaks.append(result.grid[i])

        # Should detect two peaks near 2.0 and 4.5
        assert len(peaks) >= 2

    def test_density_fixed_lambda(self, bimodal_data):
        """Density estimation with fixed lambda (no AIC search)."""
        result = density_estimate(bimodal_data, bins=50, nseg=15, lambda_=1.0)
        assert result.lambda_ == 1.0
        assert np.all(result.density >= 0)

    def test_density_custom_domain(self):
        """Custom domain boundaries are respected."""
        np.random.seed(42)
        x = np.random.exponential(2, 200)
        result = density_estimate(x, bins=50, xl=0.0, nseg=15)

        # Grid should start at 0
        assert result.grid[0] > 0  # midpoint of first bin


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------


class TestGaussianBackwardCompat:
    """Ensure Gaussian family preserves existing behaviour."""

    def setup_method(self):
        np.random.seed(42)
        self.x = np.linspace(0, 1, 50)
        self.y = np.sin(2 * np.pi * self.x) + 0.1 * np.random.randn(50)

    def test_default_is_gaussian(self):
        """Default family is Gaussian."""
        ps = PSpline(self.x, self.y)
        assert ps._family_obj.is_gaussian

    def test_gaussian_explicit_matches_default(self):
        """Explicit family='gaussian' matches default behavior."""
        ps_default = PSpline(self.x, self.y, nseg=15).fit()
        ps_gaussian = PSpline(self.x, self.y, nseg=15, family="gaussian").fit()

        assert_allclose(ps_gaussian.coef, ps_default.coef, rtol=1e-12)
        assert_allclose(ps_gaussian.fitted_values, ps_default.fitted_values, rtol=1e-12)
        assert_allclose(ps_gaussian.ED, ps_default.ED, rtol=1e-12)
        assert_allclose(ps_gaussian.sigma2, ps_default.sigma2, rtol=1e-12)

    def test_gaussian_predict_consistency(self):
        """Predictions remain consistent for Gaussian family."""
        ps = PSpline(self.x, self.y, nseg=15).fit()
        y_pred = ps.predict(self.x)
        assert_allclose(y_pred, ps.fitted_values, rtol=1e-10)

    def test_gaussian_se_consistency(self):
        """SEs remain consistent for Gaussian family."""
        ps = PSpline(self.x, self.y, nseg=15).fit()
        x_new = np.linspace(0.1, 0.9, 20)
        y_pred, se = ps.predict(x_new, return_se=True)

        assert y_pred.shape == (20,)
        assert se.shape == (20,)
        assert np.all(se > 0)

    def test_weights_still_work(self):
        """Weighted Gaussian still works."""
        w = np.ones(50)
        w[20:30] = 0.0
        ps = PSpline(self.x, self.y, nseg=15, weights=w).fit()

        assert ps.coef is not None
        assert ps.sigma2 > 0

    def test_repr_includes_family(self):
        """repr shows family name."""
        ps = PSpline(self.x, self.y)
        assert "gaussian" in repr(ps)

        ps2 = PSpline(self.x, np.abs(self.y) + 0.1, family="poisson")
        assert "poisson" in repr(ps2)

    def test_cross_validation_gaussian(self):
        """cross_validation still works for Gaussian."""
        ps = PSpline(self.x, self.y, nseg=15)
        lam, score = cross_validation(ps)
        assert lam > 0
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Convergence and Edge Cases
# ---------------------------------------------------------------------------


class TestConvergenceAndEdgeCases:
    """Test IRLS convergence behavior."""

    def test_irls_converges_quickly(self, coal_mine_data):
        """IRLS typically converges in < 10 iterations."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=10, family="poisson").fit()
        assert ps.n_iter_ < 10

    def test_irls_max_iter_respected(self):
        """ConvergenceError raised when max_iter is too small."""
        np.random.seed(42)
        x = np.arange(20, dtype=float)
        y = np.random.poisson(5, 20).astype(float)

        ps = PSpline(x, y, nseg=5, lambda_=0.001, family="poisson", max_iter=1)
        with pytest.raises(ConvergenceError, match="did not converge"):
            ps.fit()

    def test_poisson_nearly_zero(self):
        """Poisson with near-zero counts converges to small fitted values."""
        np.random.seed(42)
        x = np.arange(20, dtype=float)
        y = np.random.poisson(0.1, 20).astype(float)  # Very sparse counts

        ps = PSpline(x, y, nseg=5, lambda_=100, family="poisson").fit()
        assert ps.coef is not None
        assert np.all(ps.fitted_values > 0)
        assert np.all(ps.fitted_values < 5)  # Should be small

    def test_weights_with_poisson(self, coal_mine_data):
        """User prior weights combine with IRLS weights (§2.12.3)."""
        years, counts = coal_mine_data
        w = np.ones(len(years))
        w[:20] = 0.5  # Down-weight early observations

        ps_no_w = PSpline(years, counts, nseg=15, lambda_=10, family="poisson").fit()
        ps_w = PSpline(
            years, counts, nseg=15, lambda_=10, family="poisson", weights=w
        ).fit()

        # Coefficients should differ
        assert not np.allclose(ps_w.coef, ps_no_w.coef)

    def test_predict_link_vs_response_consistent(self, coal_mine_data):
        """type='link' and type='response' are consistent via inverse link."""
        years, counts = coal_mine_data
        ps = PSpline(years, counts, nseg=15, lambda_=50, family="poisson").fit()

        x_new = np.linspace(years[0], years[-1], 30)
        eta = ps.predict(x_new, type="link")
        mu = ps.predict(x_new, type="response")
        assert_allclose(np.exp(eta), mu, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

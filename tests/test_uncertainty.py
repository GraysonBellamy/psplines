"""
Tests for the uncertainty helpers: credible_band, pointwise_se,
PSpline.coef_covariance (including deriv_constraint), and the removal of the
Bayesian fit in 0.3.0.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import psplines
from psplines import PSpline, credible_band, pointwise_se
from psplines.basis import b_spline_basis, b_spline_derivative_basis
from psplines.exceptions import FittingError, ValidationError

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _full_matrix_band(Bq, alpha_draws, ci_prob):
    """psplines 0.2.3 ``_bayes_predict``: forms the full draws x points matrix."""
    draws = alpha_draws @ Bq.T
    return (
        draws.mean(axis=0),
        np.percentile(draws, (1 - ci_prob) / 2 * 100, axis=0),
        np.percentile(draws, (1 + ci_prob) / 2 * 100, axis=0),
    )


def _workaround_band(Bq, alpha_draws, ci_prob, chunk):
    """Blocked band used in production before 0.3.0 (monkey-patched predict)."""
    n = Bq.shape[0]
    mean, lower, upper = np.empty(n), np.empty(n), np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        draws = alpha_draws @ Bq[s:e].T
        mean[s:e] = draws.mean(axis=0)
        lower[s:e] = np.percentile(draws, (1 - ci_prob) / 2 * 100, axis=0)
        upper[s:e] = np.percentile(draws, (1 + ci_prob) / 2 * 100, axis=0)
    return mean, lower, upper


def _old_compute_se(Bq, phi, Ainv):
    """psplines 0.2.3 ``PSpline._compute_se``."""
    Bq_dense = Bq.toarray() if sp.issparse(Bq) else np.asarray(Bq)
    V = (phi * Ainv) @ Bq_dense.T
    var = np.einsum("ij,ji->i", Bq_dense, V)
    return np.sqrt(np.maximum(var, 0.0))


def _assert_bands_equal(got, expected):
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_array_equal(g, e)


def _assert_bands_close(got, expected):
    """Equal up to BLAS rounding (draws held in a different memory layout)."""
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_allclose(g, e, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basis_and_draws():
    rng = np.random.default_rng(1)
    xq = np.sort(rng.uniform(0, 1, 1234))
    Bq_sp, _ = b_spline_basis(xq, 0.0, 1.0, nseg=15, degree=3)
    Bq = Bq_sp.toarray()
    alpha_draws = rng.standard_normal((300, Bq.shape[1])).cumsum(axis=1)
    return Bq_sp, Bq, alpha_draws


@pytest.fixture
def fitted_spline():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 1, 150)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(x.size)
    return PSpline(x, y, nseg=12, lambda_=1.0).fit()


# ---------------------------------------------------------------------------
# credible_band
# ---------------------------------------------------------------------------


class TestCredibleBand:
    @pytest.mark.parametrize("chunk_size", [1, 7, 500, 1234, 2000, 10_000])
    def test_matches_full_matrix(self, basis_and_draws, chunk_size):
        """Each point uses only its own column: any block size gives the
        full-matrix numbers (exact on common BLAS; checked to 1e-12 here)."""
        _, Bq, alpha_draws = basis_and_draws
        got = credible_band(Bq, alpha_draws, ci_prob=0.9, chunk_size=chunk_size)
        expected = _full_matrix_band(Bq, alpha_draws, 0.9)
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_allclose(g, e, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("chunk_size", [1, 7, 2000])
    def test_matches_blockwise_workaround_exactly(self, basis_and_draws, chunk_size):
        _, Bq, alpha_draws = basis_and_draws
        got = credible_band(Bq, alpha_draws, ci_prob=0.95, chunk_size=chunk_size)
        _assert_bands_equal(got, _workaround_band(Bq, alpha_draws, 0.95, chunk_size))

    def test_default_chunk_size_is_2000(self, basis_and_draws):
        _, Bq, alpha_draws = basis_and_draws
        Bq_big = np.vstack([Bq, Bq])  # 2468 rows -> two blocks at the default
        _assert_bands_equal(
            credible_band(Bq_big, alpha_draws),
            _workaround_band(Bq_big, alpha_draws, 0.95, 2000),
        )

    def test_sparse_basis_equals_dense(self, basis_and_draws):
        Bq_sp, Bq, alpha_draws = basis_and_draws
        for chunk_size in (13, 2000):
            _assert_bands_equal(
                credible_band(Bq_sp, alpha_draws, chunk_size=chunk_size),
                credible_band(Bq, alpha_draws, chunk_size=chunk_size),
            )
        _assert_bands_equal(
            credible_band(Bq_sp.tocoo(), alpha_draws),
            credible_band(Bq, alpha_draws),
        )

    def test_band_contains_mean(self, basis_and_draws):
        _, Bq, alpha_draws = basis_and_draws
        mean, lower, upper = credible_band(Bq, alpha_draws, chunk_size=100)
        assert mean.shape == lower.shape == upper.shape == (Bq.shape[0],)
        assert np.all(lower <= mean) and np.all(mean <= upper)

    def test_block_memory_is_bounded(self, basis_and_draws, monkeypatch):
        """Only n_samples x chunk_size products are formed at a time."""
        _, Bq, alpha_draws = basis_and_draws
        seen = []
        real_percentile = np.percentile

        def spy(a, *args, **kwargs):
            seen.append(np.shape(a))
            return real_percentile(a, *args, **kwargs)

        monkeypatch.setattr(np, "percentile", spy)
        credible_band(Bq, alpha_draws, chunk_size=100)
        assert seen
        assert all(s[0] == alpha_draws.shape[0] and s[1] <= 100 for s in seen)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"ci_prob": 0.0}, "ci_prob"),
            ({"ci_prob": 1.0}, "ci_prob"),
            ({"chunk_size": 0}, "chunk_size"),
            ({"chunk_size": -5}, "chunk_size"),
            ({"chunk_size": 2.5}, "chunk_size"),
        ],
    )
    def test_invalid_parameters(self, basis_and_draws, kwargs, match):
        _, Bq, alpha_draws = basis_and_draws
        with pytest.raises(ValidationError, match=match):
            credible_band(Bq, alpha_draws, **kwargs)

    def test_single_query_point(self, basis_and_draws):
        _, Bq, alpha_draws = basis_and_draws
        expected = credible_band(Bq, alpha_draws, chunk_size=1)
        for row in (0, 617, Bq.shape[0] - 1):
            for chunk_size in (1, 2000):
                got = credible_band(
                    Bq[row : row + 1], alpha_draws, chunk_size=chunk_size
                )
                for g, e in zip(got, expected, strict=True):
                    assert g.shape == (1,)
                    assert g[0] == e[row]

    def test_nan_rows_stay_local(self, basis_and_draws):
        """A NaN basis row (a point outside a non-extrapolating BSpline's
        domain) gives NaN at that point only, whatever the block size."""
        _, Bq, alpha_draws = basis_and_draws
        Bq_nan = Bq.copy()
        bad = [0, 5, 6, Bq.shape[0] - 1]
        Bq_nan[bad] = np.nan
        good = np.setdiff1d(np.arange(Bq.shape[0]), bad)
        for chunk_size in (1, 6, 7, 2000):
            reference = credible_band(Bq, alpha_draws, chunk_size=chunk_size)
            got = credible_band(Bq_nan, alpha_draws, chunk_size=chunk_size)
            for g, e in zip(got, reference, strict=True):
                assert np.all(np.isnan(g[bad]))
                np.testing.assert_array_equal(g[good], e[good])

    def test_invalid_shapes(self, basis_and_draws):
        _, Bq, alpha_draws = basis_and_draws
        with pytest.raises(ValidationError, match="columns"):
            credible_band(Bq, alpha_draws[:, :-1])
        with pytest.raises(ValidationError, match="2-D"):
            credible_band(Bq, alpha_draws[0])
        with pytest.raises(ValidationError, match="2-D"):
            credible_band(Bq[0], alpha_draws)
        with pytest.raises(ValidationError, match="at least one draw"):
            credible_band(Bq, alpha_draws[:0])


# ---------------------------------------------------------------------------
# pointwise_se
# ---------------------------------------------------------------------------


class TestPointwiseSE:
    def test_matches_explicit_diagonal(self, basis_and_draws):
        Bq_sp, Bq, _ = basis_and_draws
        rng = np.random.default_rng(4)
        L = rng.standard_normal((Bq.shape[1], Bq.shape[1]))
        cov = L @ L.T
        expected = np.sqrt(np.diag(Bq @ cov @ Bq.T))
        np.testing.assert_allclose(pointwise_se(Bq, cov), expected, rtol=1e-10)
        np.testing.assert_array_equal(pointwise_se(Bq_sp, cov), pointwise_se(Bq, cov))

    def test_clips_negative_variance(self):
        Bq = np.array([[1.0, 0.0], [0.0, 1.0]])
        cov = np.diag([4.0, -1e-18])
        np.testing.assert_array_equal(pointwise_se(Bq, cov), [2.0, 0.0])

    def test_shape_mismatch(self, basis_and_draws):
        _, Bq, _ = basis_and_draws
        nb = Bq.shape[1]
        with pytest.raises(ValidationError, match="cov"):
            pointwise_se(Bq, np.eye(nb + 1))
        with pytest.raises(ValidationError, match="2-D"):
            pointwise_se(Bq[0], np.eye(nb))

    def test_compute_se_bitwise_unchanged(self, fitted_spline):
        ps = fitted_spline
        xq = np.linspace(0, 1, 333)
        Bq, _ = b_spline_basis(xq, ps._xl, ps._xr, ps.nseg, ps.degree)
        dBq, _ = b_spline_derivative_basis(
            xq, ps._xl, ps._xr, ps.nseg, ps.degree, 1, ps.knots
        )
        for basis in (Bq, Bq.toarray(), dBq, ps.B):
            np.testing.assert_array_equal(
                ps._compute_se(basis), _old_compute_se(basis, ps.phi_, ps._Ainv)
            )
        np.testing.assert_array_equal(
            ps.predict(xq, return_se=True)[1], _old_compute_se(Bq, ps.phi_, ps._Ainv)
        )
        np.testing.assert_array_equal(
            ps.derivative(xq, return_se=True)[1],
            _old_compute_se(dBq, ps.phi_, ps._Ainv),
        )
        np.testing.assert_array_equal(
            ps.se_fitted, _old_compute_se(ps.B, ps.phi_, ps._Ainv)
        )


# ---------------------------------------------------------------------------
# PSpline.coef_covariance
# ---------------------------------------------------------------------------


def _coef_cov_fits():
    rng = np.random.default_rng(6)
    x = np.linspace(0, 1, 120)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(x.size)
    counts = rng.poisson(np.exp(1 + np.sin(2 * np.pi * x)))
    return {
        "gaussian": PSpline(x, y, nseg=12, lambda_=1.0),
        "weighted": PSpline(x, y, nseg=12, lambda_=1.0, weights=1 + x),
        "shape": PSpline(
            x, x + 0.1 * y, nseg=12, shape=[psplines.ShapeConstraint("increasing")]
        ),
        "poisson": PSpline(x, counts, nseg=12, lambda_=1.0, family="poisson"),
        "constrained": PSpline(
            x,
            y,
            nseg=12,
            lambda_=1.0,
            deriv_constraint=psplines.DerivConstraint(order=1, initial=0, final=0),
        ),
        "slope_zero": PSpline(
            x,
            y,
            nseg=12,
            lambda_=1.0,
            slope_zero=psplines.SlopeZeroConstraint(domain=(0.4, 0.6)),
        ),
        "adaptive": PSpline(x, y, nseg=12, lambda_=1.0, adaptive=True),
    }


_COV_KINDS = [
    "gaussian",
    "weighted",
    "shape",
    "poisson",
    "constrained",
    "slope_zero",
    "adaptive",
]


class TestCoefCovariance:
    def test_raises_before_fit(self):
        ps = _coef_cov_fits()["gaussian"]
        with pytest.raises(FittingError):
            _ = ps.coef_covariance

    def test_uses_total_penalty(self):
        """Shape, slope and adaptive penalties enter A."""
        fits = {k: v.fit() for k, v in _coef_cov_fits().items()}
        for kind in ("gaussian", "shape", "slope_zero", "adaptive"):
            ps = fits[kind]
            A = (ps._BtB + ps._P_total).toarray()
            np.testing.assert_array_equal(
                ps.coef_covariance, ps.phi_ * np.linalg.inv(A)
            )


class TestDerivConstraintUncertainty:
    """deriv_constraint imposes C alpha = 0 exactly; the uncertainty uses
    M = A^-1 - A^-1 C'(C A^-1 C')^-1 C A^-1 in place of A^-1."""

    @staticmethod
    def _fits():
        rng = np.random.default_rng(8)
        x = np.linspace(0, 1, 200)
        y = np.cos(np.pi * x) + 0.1 * rng.standard_normal(x.size)
        dc = psplines.DerivConstraint(order=1, initial=0, final=0)
        free = PSpline(x, y, nseg=20, lambda_=1.0).fit()
        pinned = PSpline(x, y, nseg=20, lambda_=1.0, deriv_constraint=dc).fit()
        return free, pinned

    def test_covariance_is_conditioned_on_the_constraint(self):
        free, ps = self._fits()
        A = (ps._BtB + ps._P_total).toarray()
        Ai = np.linalg.inv(A)
        C = ps._C.toarray()
        M = Ai - Ai @ C.T @ np.linalg.solve(C @ Ai @ C.T, C @ Ai)
        np.testing.assert_allclose(ps.coef_covariance, ps.phi_ * M, rtol=1e-9, atol=0)
        # M is the operator of the constrained solve: coef = M B'y
        np.testing.assert_allclose(ps._Ainv @ ps._Bty, ps.coef, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(C @ ps.coef, 0.0, atol=1e-10)
        assert ps.ED < free.ED

    def test_pinned_derivative_has_zero_se(self):
        _, ps = self._fits()
        d, se = ps.derivative(np.array([0.0, 0.5, 1.0]), return_se=True)
        np.testing.assert_allclose(d[[0, 2]], 0.0, atol=1e-10)
        assert se[1] > 0.1
        assert np.all(se[[0, 2]] < 1e-6 * se[1])
        # values are not pinned, so their SEs stay positive
        _, se_val = ps.predict(np.array([0.0, 1.0]), return_se=True)
        assert np.all(se_val > 0.01)

    def test_unconstrained_fit_unchanged(self):
        free, _ = self._fits()
        A = (free._BtB + free._P_total).toarray()
        np.testing.assert_array_equal(free._Ainv, np.linalg.inv(A))

    def test_unsorted_x(self):
        """The constraint sits at min(x) and max(x), not at x[0] and x[-1]."""
        rng = np.random.default_rng(9)
        x = np.linspace(0, 1, 150)
        y = np.sin(3 * x) + 0.05 * rng.standard_normal(x.size)
        dc = psplines.DerivConstraint(order=1, initial=0, final=0)
        perm = rng.permutation(x.size)
        a = PSpline(x, y, nseg=15, lambda_=1.0, deriv_constraint=dc).fit()
        b = PSpline(x[perm], y[perm], nseg=15, lambda_=1.0, deriv_constraint=dc).fit()
        np.testing.assert_allclose(b.coef, a.coef, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(b.derivative(np.array([0.0, 1.0])), 0.0, atol=1e-10)

    @pytest.mark.parametrize("kind", _COV_KINDS)
    def test_reproduces_analytic_se_bitwise(self, kind):
        ps = _coef_cov_fits()[kind].fit()
        cov = ps.coef_covariance
        nb = ps.coef.size
        assert cov.shape == (nb, nb)
        np.testing.assert_array_equal(cov, ps.phi_ * ps._Ainv)

        xq = np.linspace(0, 1, 211)
        Bq, _ = b_spline_basis(xq, ps._xl, ps._xr, ps.nseg, ps.degree)
        dBq, _ = b_spline_derivative_basis(
            xq, ps._xl, ps._xr, ps.nseg, ps.degree, 1, ps.knots
        )
        for basis in (Bq, Bq.toarray(), dBq, ps.B):
            se = pointwise_se(basis, cov)
            np.testing.assert_array_equal(se, ps._compute_se(basis))
            np.testing.assert_array_equal(se, _old_compute_se(basis, ps.phi_, ps._Ainv))
        np.testing.assert_array_equal(
            pointwise_se(Bq, cov), ps.predict(xq, return_se=True, scale="link")[1]
        )
        np.testing.assert_array_equal(
            pointwise_se(dBq, cov), ps.derivative(xq, return_se=True)[1]
        )
        np.testing.assert_array_equal(pointwise_se(ps.B, cov), ps.se_fitted)

    def test_returns_independent_copy(self):
        ps = _coef_cov_fits()["gaussian"].fit()
        se_before = ps.se_fitted.copy()
        cov = ps.coef_covariance
        cov[:] = 0.0
        assert ps.coef_covariance is not cov
        np.testing.assert_array_equal(ps._compute_se(ps.B), se_before)


def test_public_exports():
    assert psplines.credible_band is credible_band
    assert psplines.pointwise_se is pointwise_se
    assert "credible_band" in psplines.__all__
    assert "pointwise_se" in psplines.__all__


def test_bayesian_fit_removed():
    """0.3.0 removed the Bayesian fit; only the analytic and bootstrap
    uncertainty methods remain."""
    for name in (
        "bayes_fit",
        "posterior_draws",
        "plot_lam_trace",
        "plot_alpha_trace",
        "plot_posterior",
    ):
        assert not hasattr(PSpline, name)
    x = np.linspace(0, 1, 50)
    ps = PSpline(x, np.sin(3 * x), nseg=8).fit()
    with pytest.raises(ValidationError, match="'analytic' or 'bootstrap'"):
        ps.predict(x, se_method="bayes")
    with pytest.raises(ValidationError, match="'analytic' or 'bootstrap'"):
        ps.derivative(x, se_method="bayes")

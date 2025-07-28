"""
psplines.core
=============

Univariate P-spline smoother (Gaussian) with:
  - Sparse back-end (SciPy sparse matrices + spsolve)
  - Analytic point-wise standard errors (delta method)
  - Optional parametric residual bootstrap SEs (parallelizable)
  - Derivative boundary constraints

Based on Eilers & Marx (2021): Chapters 2, 3, and 8.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from numpy.typing import ArrayLike, NDArray
import pymc as pm
import arviz as az
from scipy.interpolate import BSpline
import pytensor

from .basis import b_spline_basis, b_spline_derivative_basis
from .penalty import difference_matrix
from .utils_math import effective_df

__all__ = ["PSpline"]


def _as1d(a: ArrayLike, dtype=float) -> NDArray:
    """
    Convert input to 1D contiguous numpy array.
    """
    arr = np.asarray(a, dtype=dtype).reshape(-1)
    return arr.copy(order="C") if not arr.flags["C_CONTIGUOUS"] else arr


@dataclass(slots=True)
class PSpline:
    """
    Univariate penalised B-spline smoother.
    """

    x: ArrayLike
    y: ArrayLike
    nseg: int = 20
    degree: int = 3
    lambda_: float = 10.0
    penalty_order: int = 2
    constraints: Optional[dict[str, Any]] = None

    # runtime
    B: Optional[sp.spmatrix] = None
    knots: Optional[NDArray] = None
    coef: Optional[NDArray] = None
    fitted_values: Optional[NDArray] = None

    # sparse cross-products
    _BtB: Optional[sp.spmatrix] = None
    _DtD: Optional[sp.spmatrix] = None
    _Bty: Optional[NDArray] = None

    # constraints
    _C: Optional[sp.spmatrix] = None
    _A_aug_static: Optional[sp.spmatrix] = None
    _b_aug: Optional[NDArray] = None

    # uncertainty
    ED: Optional[float] = None
    sigma2: Optional[float] = None
    se_coef: Optional[NDArray] = None
    se_fitted: Optional[NDArray] = None

    # Bayesian output
    trace: Any = None
    lambda_post: Optional[NDArray] = None
    _spline: Optional[BSpline] = None

    _xl: Optional[float] = None
    _xr: Optional[float] = None

    def __post_init__(self):
        self.x = _as1d(self.x)
        self.y = _as1d(self.y)
        if self.x.size != self.y.size:
            raise ValueError("x and y must be same length")
        self.constraints = self.constraints or {}

    def fit(self, *, xl: Optional[float] = None, xr: Optional[float] = None) -> PSpline:
        """
        Fit the P-spline model.
        """
        # domain
        self._xl = float(self.x.min() if xl is None else xl)
        self._xr = float(self.x.max() if xr is None else xr)
        # basis and penalty
        self.B, self.knots = b_spline_basis(
            self.x, self._xl, self._xr, self.nseg, self.degree
        )
        nb = self.B.shape[1]
        D = difference_matrix(nb, self.penalty_order)
        # sparse cross-products
        self._BtB = (self.B.T @ self.B).tocsr()
        self._DtD = (D.T @ D).tocsr()
        self._Bty = self.B.T @ self.y
        # constraints
        self._build_constraints(nb)
        # solve coefficients
        P = self.lambda_ * self._DtD
        self.coef = self._solve_coef(P)
        # fitted values
        self.fitted_values = self.B @ self.coef
        # analytic uncertainty
        self._update_uncertainty()
        return self

    def predict(
        self,
        x_new: ArrayLike,
        *,
        derivative_order: Optional[int] = None,
        return_se: bool = False,
        se_method: str = "analytic",
        B_boot: int = 5000,
        seed: Optional[int] = None,
        n_jobs: int = 1,
        hdi_prob: float = 0.95,
    ) -> NDArray | Tuple[NDArray, ...]:
        """
        Predict smooth (or derivative) with optional uncertainty.

        se_method:
          - 'analytic': delta-method SE (default)       -> (fhat, se)
          - 'bootstrap': parametric bootstrap SEs       -> (fhat, se)
          - 'bayes': posterior mean + HDI credible band -> (mean, lower, upper)
        """
        xq = _as1d(x_new)

        # Bayesian credible band
        if se_method == "bayes":
            if self.trace is None:
                raise RuntimeError("Call bayes_fit() first to sample the posterior.")
            if self._spline is None:
                raise RuntimeError("No BSpline stored. Run bayes_fit() first.")

            # Evaluate stored BSpline (or its k-th derivative)
            if derivative_order is None:
                Bq = self._spline(xq)
            else:
                Bq = self._spline(xq, nu=derivative_order)
            # to array for matmul
            Bq = Bq.toarray() if sp.issparse(Bq) else np.asarray(Bq)

            # posterior alpha draws: shape (n_samples, n_basis)
            alpha_draws = (
                self.trace.posterior["alpha"].stack(sample=("chain", "draw")).values
            )

            # If it came transposed (basis × samples), swap axes
            if (
                alpha_draws.shape[0] == Bq.shape[1]
                and alpha_draws.shape[1] != Bq.shape[1]
            ):
                alpha_draws = alpha_draws.T

            # Check dimensions now match: (n_samples, nb)
            S, p = alpha_draws.shape
            if p != Bq.shape[1]:
                raise ValueError(
                    f"Mismatch: posterior alpha has length {p}, but basis has {Bq.shape[1]} cols"
                )

            # draws of f^(k)(x): (n_samples, n_points)
            deriv_draws = alpha_draws @ Bq.T

            # summarize
            mean = deriv_draws.mean(axis=0)
            lower = np.percentile(deriv_draws, (1 - hdi_prob) / 2 * 100, axis=0)
            upper = np.percentile(deriv_draws, (1 + hdi_prob) / 2 * 100, axis=0)
            return mean, lower, upper

        # bootstrap SEs
        if se_method == "bootstrap" and return_se:
            return self._bootstrap_predict(xq, derivative_order, B_boot, seed, n_jobs)

        # analytic or plain
        if self.coef is None:
            raise RuntimeError("Call fit() first")
        if derivative_order is None:
            Bq, _ = b_spline_basis(xq, self._xl, self._xr, self.nseg, self.degree)
        else:
            Bq, _ = b_spline_derivative_basis(
                xq,
                self._xl,
                self._xr,
                self.nseg,
                self.degree,
                derivative_order,
                self.knots,
            )
        fhat = Bq @ self.coef
        if not return_se:
            return fhat
        if self.se_fitted is None:
            raise RuntimeError("Analytic SEs unavailable")
        cov_diag = self.se_coef**2
        B2 = Bq.multiply(Bq) if sp.issparse(Bq) else Bq**2
        se = np.sqrt(B2 @ cov_diag)
        return fhat, se

    def bayes_fit(
        self,
        a: float = 2.0,
        b: float = 0.1,
        c: float = 2.0,
        d: float = 1.0,
        draws: int = 2000,
        tune: int = 2000,
        chains: int = 4,
        cores: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
    ) -> Any:
        # Prepare basis and penalty
        self._xl = float(self.x.min()) if self._xl is None else self._xl
        self._xr = float(self.x.max()) if self._xr is None else self._xr
        B_sp, self.knots = b_spline_basis(
            self.x, self._xl, self._xr, self.nseg, self.degree
        )
        B = B_sp.toarray() if sp.issparse(B_sp) else B_sp
        nb = B.shape[1]
        D = difference_matrix(nb, self.penalty_order).toarray()
        y = self.y
        I_nb = np.eye(nb)

        # store BSpline object for predict
        coeffs = np.eye(nb)
        self._spline = BSpline(self.knots, coeffs, self.degree, extrapolate=False)

        with pm.Model() as model:
            lam = pm.Gamma("lam", alpha=a, beta=b, shape=D.shape[0])
            Q = pm.math.dot(D.T * lam, D)
            Q_j = Q + I_nb * 1e-6
            alpha = pm.MvNormal(
                "alpha",
                mu=pm.math.zeros(Q_j.shape[0]),
                tau=Q_j,
                shape=Q_j.shape[0],
            )
            sigma = pm.InverseGamma("sigma", alpha=c, beta=d)
            mu = pm.Deterministic("mu", pm.math.dot(B, alpha))
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
            print(pytensor.link.c.cmodule.default_blas_ldflags())
            print(pytensor.config.blas__ldflags)
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
            )

        # Store results
        self.trace = trace
        self.coef = (
            trace.posterior["alpha"]
            .stack(sample=("chain", "draw"))
            .mean(dim="sample")
            .values
        )
        self.fitted_values = (
            trace.posterior["mu"]
            .stack(sample=("chain", "draw"))
            .mean(dim="sample")
            .values
        )
        self.lambda_post = (
            trace.posterior["lam"]
            .stack(sample=("chain", "draw"))
            .mean(dim="sample")
            .values
        )
        return trace

    def plot_lam_trace(self, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot trace and marginal posterior for each lambda_j.
        """
        az.plot_trace(self.trace, var_names=["lam"], figsize=figsize)

    def plot_alpha_trace(self, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot trace and marginal posterior for alpha coefficients.
        """
        az.plot_trace(self.trace, var_names=["alpha"], figsize=figsize)

    def plot_posterior(self, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot posterior
        """
        az.plot_posterior(
            self.trace,
            var_names=["lam", "sigma"],
            figsize=figsize,
            hdi_prob=0.95,
            point_estimate="mean",
        )

    def _bootstrap_predict(
        self,
        xq: NDArray,
        deriv: Optional[int],
        B: int,
        seed: Optional[int],
        n_jobs: int,
    ) -> Tuple[NDArray, NDArray]:
        """
        Parametric residual bootstrap SEs (parallelized).
        """
        # baseline prediction and precompute evaluation basis once
        if deriv is None:
            baseline = self.predict(xq)
            Bq, _ = b_spline_basis(xq, self._xl, self._xr, self.nseg, self.degree)
        else:
            baseline = self.predict(xq, derivative_order=deriv)
            Bq, _ = b_spline_derivative_basis(
                xq, self._xl, self._xr, self.nseg, self.degree, deriv, self.knots
            )
        # prepare system
        lam = self.lambda_
        A = (self._BtB + lam * self._DtD).tocsr()
        has_C = self._C is not None
        if has_C:
            nc = self._C.shape[0]
            zero = sp.csr_matrix((nc, nc))
            top = sp.hstack([A, self._C.T], format="csr")
            bot = sp.hstack([self._C, zero], format="csr")
            A_aug = sp.vstack([top, bot], format="csr")
        # generate all residual replicates at once
        n = self.y.size
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((B, n)) * np.sqrt(self.sigma2)
        # cache B transpose
        BT = self.B.T

        def one(i):
            # single bootstrap replicate
            y_star = self.fitted_values + R[i]
            bty = BT @ y_star
            if not has_C:
                coef_s = spsolve(A, bty)
            else:
                rhs = np.concatenate([bty, np.zeros(nc)])
                sol = spsolve(A_aug, rhs)
                coef_s = sol[: A.shape[0]]
            return Bq @ coef_s

        sims = Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(B))
        sims = np.vstack(sims)
        se_boot = sims.std(axis=0, ddof=1)
        return baseline, se_boot

    def _build_constraints(self, nb: int) -> None:
        """
        Build boundary derivative constraints.
        """
        dcf = self.constraints.get("deriv")
        if not dcf:
            self._C = None
            return
        rows = []
        order = dcf.get("order", 1)
        if dcf.get("initial") == 0:
            B0, _ = b_spline_derivative_basis(
                self.x[0], self._xl, self._xr, self.nseg, self.degree, order, self.knots
            )
            rows.append(B0)
        if dcf.get("final") == 0:
            B1, _ = b_spline_derivative_basis(
                self.x[-1],
                self._xl,
                self._xr,
                self.nseg,
                self.degree,
                order,
                self.knots,
            )
            rows.append(B1)
        C = sp.vstack(rows).tocsr()
        nc = C.shape[0]
        zero = sp.csr_matrix((nc, nc))
        self._C = C
        self._A_aug_static = sp.vstack(
            [sp.hstack([self._BtB, C.T]), sp.hstack([C, zero])]
        ).tocsr()
        self._b_aug = np.concatenate([self._Bty, np.zeros(nc)])

    def _solve_coef(self, P: sp.spmatrix) -> NDArray:
        """
        Solve penalized system for coefficients.
        """
        A = (self._BtB + P).tocsr()
        if self._C is None:
            return spsolve(A, self._Bty)
        nc = self._C.shape[0]
        zero = sp.csr_matrix((nc, nc))
        top = sp.hstack([A, self._C.T], format="csr")
        bot = sp.hstack([self._C, zero], format="csr")
        A_aug = sp.vstack([top, bot], format="csr")
        rhs = np.concatenate([self._Bty, np.zeros(nc)])
        sol = spsolve(A_aug, rhs)
        return sol[: A.shape[0]]

    def _update_uncertainty(self) -> None:
        """
        Compute ED, sigma2, and analytic SEs.
        """
        nb = self.B.shape[1]
        self.ED = effective_df(
            self.B, difference_matrix(nb, self.penalty_order), self.lambda_
        )
        resid = self.y - self.fitted_values
        self.sigma2 = float(resid @ resid) / (len(self.y) - self.ED)
        A = (self._BtB + self.lambda_ * self._DtD).tocsr()
        diag = np.empty(nb)
        for i in range(nb):
            e = np.zeros(nb)
            e[i] = 1
            u = spsolve(A, e)
            diag[i] = self.sigma2 * u[i]
        self.se_coef = np.sqrt(diag)
        B2 = self.B.multiply(self.B) if sp.issparse(self.B) else self.B**2
        self.se_fitted = np.sqrt(B2 @ diag)

    def __repr__(self) -> str:
        st = "fitted" if self.coef is not None else "unfitted"
        return f"<PSpline {st}; n={self.x.size};seg={self.nseg};deg={self.degree};d={self.penalty_order}>"

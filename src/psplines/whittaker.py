"""
psplines.whittaker – Whittaker smoother with x-aware penalties
==============================================================

Implements the Whittaker smoother as described by Eilers (2003) — the
special case of P-splines where B = I (identity basis).  The smoother
operates directly on the data vector, solving:

.. math::

    (W + \\lambda\\, D_x^\\top D_x)\\, z = W\\, y

For non-uniformly spaced data the standard difference operator D is
replaced by the divided-difference operator *D_x* which weights each
finite difference by the reciprocal of the gap in *x*.  This ensures
that the roughness penalty is expressed in the natural units of the
independent variable rather than index position.

When *x* is uniformly spaced the standard (unweighted) difference
matrix is used automatically, which is numerically identical to the
classical Whittaker smoother.

References
----------
Eilers, P.H.C. (2003). "A perfect smoother", *Anal. Chem.* 75,
3631–3636.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import spsolve

from .exceptions import FittingError, OptimizationError, ValidationError
from .penalty import difference_matrix, divided_difference_matrix

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["WhittakerSmoother"]


def _is_uniform(x: np.ndarray, rtol: float = 1e-8) -> bool:
    """Return True if *x* is uniformly spaced within *rtol*."""
    h = np.diff(x)
    return bool(np.allclose(h, h[0], rtol=rtol, atol=0.0))


@dataclass(slots=True)
class WhittakerSmoother:
    """Whittaker smoother with x-aware penalty for non-uniform data.

    Solves ``(W + λ D'D) z = W y`` where *D* is either the standard
    difference operator (uniform spacing) or the divided-difference
    operator (non-uniform spacing).

    Parameters
    ----------
    x : array-like, shape (n,)
        Sample positions — must be finite.  Need not be sorted or
        uniformly spaced; the smoother sorts internally.
    y : array-like, shape (n,)
        Observed values at each *x*.
    lambda_ : float
        Smoothing parameter (> 0).
    penalty_order : int
        Difference order for the roughness penalty (1 or 2, default 2).
    weights : array-like, shape (n,), optional
        Non-negative observation weights.  Zero means "missing".
    """

    x: ArrayLike
    y: ArrayLike
    lambda_: float = 1e2
    penalty_order: int = 2
    weights: ArrayLike | None = None

    # --- results (populated by fit) ---
    fitted_values: NDArray | None = field(default=None, repr=False)
    ED: float | None = field(default=None, repr=False)
    se_fitted: NDArray | None = field(default=None, repr=False)

    # --- internal state ---
    _x_sorted: NDArray | None = field(default=None, repr=False)
    _sort_idx: NDArray | None = field(default=None, repr=False)
    _unsort_idx: NDArray | None = field(default=None, repr=False)
    _DtD: sp.spmatrix | None = field(default=None, repr=False)
    _D: sp.spmatrix | None = field(default=None, repr=False)
    _W: sp.spmatrix | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ init
    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()

        if self.x.size == 0 or self.y.size == 0:
            raise ValidationError("Input arrays cannot be empty")
        if self.x.size != self.y.size:
            raise ValidationError(
                f"x and y must have the same length "
                f"(got {self.x.size} and {self.y.size})"
            )
        if self.x.size < 3:
            raise ValidationError(f"Need at least 3 data points (got {self.x.size})")
        if not np.all(np.isfinite(self.x)):
            raise ValidationError("x contains non-finite values")
        if not np.all(np.isfinite(self.y)):
            raise ValidationError("y contains non-finite values")
        if self.lambda_ <= 0:
            raise ValidationError(f"lambda_ must be positive (got {self.lambda_})")
        if self.penalty_order < 1:
            raise ValidationError(
                f"penalty_order must be >= 1 (got {self.penalty_order})"
            )

        # Sort by x
        self._sort_idx = np.argsort(self.x)
        self._x_sorted = self.x[self._sort_idx]
        self._unsort_idx = np.argsort(self._sort_idx)

        if len(np.unique(self._x_sorted)) < self._x_sorted.size:
            raise ValidationError("x contains duplicate values")

        # Weights
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float).ravel()
            if w.size != self.x.size:
                raise ValidationError(
                    f"weights length ({w.size}) must match x length ({self.x.size})"
                )
            if np.any(w < 0):
                raise ValidationError("weights must be non-negative")
            self.weights = w

    # --------------------------------------------------------------- fitting
    def _build_penalty(self, x: np.ndarray) -> tuple[sp.spmatrix, sp.spmatrix]:
        """Return (D, D'D) for the given sorted x-vector."""
        n = x.size
        if _is_uniform(x):
            D = difference_matrix(n, self.penalty_order)
        else:
            D = divided_difference_matrix(x, self.penalty_order)
        DtD = (D.T @ D).tocsr()
        return D, DtD

    def fit(self) -> WhittakerSmoother:
        """Fit the smoother via a single sparse solve.

        Solves ``(W + λ D'D) z = W y`` and stores results in
        :attr:`fitted_values`, :attr:`ED`, and :attr:`se_fitted`.

        Returns
        -------
        WhittakerSmoother
            *self*, for method-chaining.
        """
        assert self._x_sorted is not None and self._sort_idx is not None
        x = self._x_sorted
        n = x.size
        y_sorted = np.asarray(self.y)[self._sort_idx]

        # Difference operator
        self._D, self._DtD = self._build_penalty(x)

        # Weight matrix
        if self.weights is not None:
            w = np.asarray(self.weights)[self._sort_idx]
        else:
            w = np.ones(n)
        self._W = sp.diags(w)

        # Solve  (W + λ D'D) z = W y
        A = (self._W + self.lambda_ * self._DtD).tocsr()  # type: ignore[operator]
        rhs = self._W @ y_sorted
        z = spsolve(A, rhs)

        # Effective degrees of freedom: ED = trace(W (W + λ D'D)^{-1})
        A_dense = A.toarray()  # type: ignore[attr-defined]
        W_dense = self._W.toarray()  # type: ignore[attr-defined]
        invA = np.linalg.inv(A_dense)
        self.ED = float(np.trace(W_dense @ invA))

        # Pointwise SEs: se_i = sqrt(phi * [A^{-1}]_ii)
        # phi = RSS / (n - ED) (Gaussian)
        resid = y_sorted - z
        dof = max(n - self.ED, 1.0)
        rss = float(resid @ (self._W @ resid))
        phi = rss / dof
        self.se_fitted = np.sqrt(np.maximum(phi * np.diag(invA), 0.0))

        # Unsort back to original order
        assert self._unsort_idx is not None
        self.fitted_values = z[self._unsort_idx]
        self.se_fitted = self.se_fitted[self._unsort_idx]

        return self

    # ------------------------------------------------------------- predict
    def predict(self, x_new: ArrayLike) -> NDArray:
        """Interpolate fitted values at new x-locations.

        Uses linear interpolation on the fitted (sorted) grid.

        Parameters
        ----------
        x_new : array-like
            Query positions.

        Returns
        -------
        NDArray
            Interpolated smoothed values.
        """
        if self.fitted_values is None:
            raise FittingError("Model not fitted. Call fit() first.")
        assert self._x_sorted is not None and self._sort_idx is not None
        xq = np.asarray(x_new, dtype=float).ravel()
        if xq.size == 0:
            raise ValidationError("x_new cannot be empty")
        # fitted_values is in original order; get sorted version for interp
        z_sorted = self.fitted_values[self._sort_idx]
        return np.asarray(np.interp(xq, self._x_sorted, z_sorted))

    # -------------------------------------------------------- lambda selection
    def _solve_for_lambda(
        self,
        lam: float,
        x: np.ndarray,
        y: np.ndarray,
        W: sp.spmatrix,
        DtD: sp.spmatrix,
        D: sp.spmatrix,
    ) -> tuple[np.ndarray, float, float, float]:
        """Solve for a single lambda, return (z, rss, penalty, edf)."""
        A = (W + lam * DtD).tocsr()  # type: ignore[operator]
        z = spsolve(A, W @ y)
        resid = y - z
        rss = float(resid @ (W @ resid))
        pen = float(np.sum((D @ z) ** 2))
        # ED via exact trace for moderate n
        A_dense = A.toarray()  # type: ignore[attr-defined]
        W_dense = W.toarray()  # type: ignore[attr-defined]
        edf = float(np.trace(W_dense @ np.linalg.inv(A_dense)))
        return z, rss, pen, edf

    def cross_validation(
        self,
        lambda_bounds: tuple[float, float] = (1e-6, 1e6),
    ) -> tuple[float, float]:
        """Select lambda via generalized cross-validation (GCV).

        Minimises ``GCV(λ) = (RSS / n) / (1 - ED / n)²`` over a
        bounded search in log₁₀(λ).

        Parameters
        ----------
        lambda_bounds : (float, float)
            Lower and upper bounds for the λ search.

        Returns
        -------
        best_lambda : float
        best_score : float
        """
        assert self._x_sorted is not None and self._sort_idx is not None
        x = self._x_sorted
        n = x.size
        y_sorted = np.asarray(self.y)[self._sort_idx]

        D, DtD = self._build_penalty(x)
        if self.weights is not None:
            w = np.asarray(self.weights)[self._sort_idx]
        else:
            w = np.ones(n)
        W = sp.diags(w)

        def obj(loglam: float) -> float:
            lam = 10.0**loglam
            _, rss, _, edf = self._solve_for_lambda(lam, x, y_sorted, W, DtD, D)
            denom = (1.0 - edf / n) ** 2
            return (rss / n) / denom if denom > 0 else np.inf

        res = minimize_scalar(
            obj,
            bounds=(np.log10(lambda_bounds[0]), np.log10(lambda_bounds[1])),
            method="bounded",
        )
        if not res.success:
            raise OptimizationError(f"GCV optimisation failed: {res.message}")
        best_lam = 10.0**res.x
        self.lambda_ = best_lam
        self.fit()
        return best_lam, float(res.fun)

    def v_curve(
        self,
        lambda_bounds: tuple[float, float] = (1e-6, 1e6),
        num_lambda: int = 81,
    ) -> tuple[float, float]:
        """Select lambda via the V-curve minimum-distance criterion.

        Sweeps a log-uniform grid of λ values, computes
        ``(log RSS, log penalty)`` at each, and picks the λ where
        consecutive-point distance is minimised.

        Parameters
        ----------
        lambda_bounds : (float, float)
            Lower and upper bounds for the λ grid.
        num_lambda : int
            Number of grid points.

        Returns
        -------
        best_lambda : float
        best_score : float
            Minimum distance value.
        """
        assert self._x_sorted is not None and self._sort_idx is not None
        x = self._x_sorted
        n = x.size
        y_sorted = np.asarray(self.y)[self._sort_idx]

        D, DtD = self._build_penalty(x)
        if self.weights is not None:
            w = np.asarray(self.weights)[self._sort_idx]
        else:
            w = np.ones(n)
        W = sp.diags(w)

        grid = np.logspace(
            np.log10(lambda_bounds[0]), np.log10(lambda_bounds[1]), num_lambda
        )
        log_rss = np.full(num_lambda, np.nan)
        log_pen = np.full(num_lambda, np.nan)

        for i, lam in enumerate(grid):
            _, rss, pen, _ = self._solve_for_lambda(lam, x, y_sorted, W, DtD, D)
            log_rss[i] = np.log(max(rss, 1e-300))
            log_pen[i] = np.log(max(pen, 1e-300))

        valid = np.isfinite(log_rss) & np.isfinite(log_pen)
        if valid.sum() < 2:
            raise OptimizationError("Not enough valid V-curve points")

        dr = np.diff(log_rss[valid])
        dp = np.diff(log_pen[valid])
        dist = np.hypot(dr, dp)
        mid = np.sqrt(grid[valid][:-1] * grid[valid][1:])
        idx = int(np.argmin(dist))
        best_lam = float(mid[idx])
        self.lambda_ = best_lam
        self.fit()
        return best_lam, float(dist[idx])

    # --------------------------------------------------------------- repr
    def __repr__(self) -> str:
        st = "fitted" if self.fitted_values is not None else "unfitted"
        n = np.asarray(self.x).size
        return (
            f"<WhittakerSmoother {st}; n={n}; "
            f"d={self.penalty_order}; λ={self.lambda_:.3g}>"
        )

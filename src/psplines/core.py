"""
psplines.core
=============

Univariate P-spline smoother with:
  - Gaussian, Poisson, and Binomial families (GLM via IRLS)
  - Sparse back-end (SciPy sparse matrices + spsolve)
  - Analytic point-wise standard errors (delta method)
  - Optional parametric residual bootstrap SEs (parallelizable)
  - Derivative boundary constraints

Based on Eilers & Marx (2021): Chapters 2, 3, and 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve

from .basis import b_spline_basis, b_spline_derivative_basis
from .exceptions import ConvergenceError, FittingError, ValidationError
from .glm import Family, get_family
from .penalty import (
    VALID_SHAPE_TYPES,
    adaptive_penalty_matrix,
    asymmetric_penalty_matrix,
    difference_matrix,
    variable_penalty_matrix,
)

__all__ = ["PSpline"]

_BAYES_INSTALL_MSG = (
    "Bayesian methods require PyMC and ArviZ. "
    "Install them with: pip install psplines[bayes]"
)


def _import_pymc() -> Any:
    """Lazily import pymc, raising a clear error if not installed."""
    try:
        import pymc as pm
    except ImportError:
        raise ImportError(_BAYES_INSTALL_MSG) from None
    return pm


def _import_arviz() -> Any:
    """Lazily import arviz, raising a clear error if not installed."""
    try:
        import arviz as az
    except ImportError:
        raise ImportError(_BAYES_INSTALL_MSG) from None
    return az


def _as1d(a: ArrayLike, dtype: type = float) -> NDArray:
    """
    Convert input to 1D contiguous numpy array.
    """
    arr: NDArray = np.asarray(a, dtype=dtype).reshape(-1)
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
    weights: ArrayLike | None = None
    constraints: dict[str, Any] | None = None

    # Shape constraint parameters (§8.7)
    shape: list[dict[str, Any]] | None = None
    shape_kappa: float = 1e8
    max_shape_iter: int = 50

    # Adaptive / variable penalty parameters (§8.8)
    adaptive: bool = False
    adaptive_nseg: int = 10
    adaptive_lambda: float = 100.0
    adaptive_max_iter: int = 20
    penalty_gamma: float | None = None  # exponential variable penalty

    # GLM parameters
    family: str | Family = "gaussian"
    trials: ArrayLike | None = None
    offset: ArrayLike | None = None
    max_iter: int = 25
    tol: float = 1e-8

    # runtime
    B: sp.spmatrix | None = None
    knots: NDArray | None = None
    coef: NDArray | None = None
    fitted_values: NDArray | None = None

    # sparse cross-products
    _BtB: sp.spmatrix | None = None
    _DtD: sp.spmatrix | None = None
    _Bty: NDArray | None = None
    _W: sp.spmatrix | None = None

    # constraints
    _C: sp.spmatrix | None = None
    # total penalty (may include shape / adaptive contributions)
    _P_total: sp.spmatrix | None = None
    # adaptive penalty weights (per-difference)
    _adaptive_weights: NDArray | None = None
    # uncertainty
    ED: float | None = None
    sigma2: float | None = None
    se_coef: NDArray | None = None
    se_fitted: NDArray | None = None
    _Ainv: NDArray | None = None

    # GLM state
    _family_obj: Family | None = field(default=None, repr=False)
    _eta: NDArray | None = None
    deviance_: float | None = None
    phi_: float | None = None
    n_iter_: int | None = None

    # Bayesian output
    trace: Any = None
    lambda_post: NDArray | None = None
    _spline: BSpline | None = None

    _xl: float | None = None
    _xr: float | None = None

    def __post_init__(self) -> None:
        # Convert and validate input arrays
        try:
            self.x = _as1d(self.x)
            self.y = _as1d(self.y)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid input arrays: {e}") from e

        # Validate array dimensions
        if self.x.size == 0 or self.y.size == 0:
            raise ValidationError("Input arrays cannot be empty")
        if self.x.size != self.y.size:
            raise ValidationError(
                f"x and y must have the same length (got {self.x.size} and {self.y.size})"
            )
        if self.x.size < 2:
            raise ValidationError(f"Need at least 2 data points (got {self.x.size})")

        # Check for invalid values
        if not np.all(np.isfinite(self.x)):
            raise ValidationError("x contains non-finite values (NaN or inf)")
        if not np.all(np.isfinite(self.y)):
            raise ValidationError("y contains non-finite values (NaN or inf)")
        if len(np.unique(self.x)) < 2:
            raise ValidationError("x must contain at least 2 unique values")

        # Validate parameters
        if self.nseg <= 0:
            raise ValidationError(f"nseg must be positive (got {self.nseg})")
        if self.degree < 0:
            raise ValidationError(f"degree must be non-negative (got {self.degree})")
        if self.lambda_ <= 0:
            raise ValidationError(f"lambda_ must be positive (got {self.lambda_})")
        if self.penalty_order < 1:
            raise ValidationError(
                f"penalty_order must be >= 1 (got {self.penalty_order})"
            )
        if self.nseg <= self.degree:
            raise ValidationError(
                f"nseg ({self.nseg}) must be greater than degree ({self.degree})"
            )

        # Validate and convert weights
        if self.weights is not None:
            try:
                self.weights = _as1d(self.weights)
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Invalid weights array: {e}") from e
            if self.weights.size != self.x.size:
                raise ValidationError(
                    f"weights must have the same length as x (got {self.weights.size} and {self.x.size})"
                )
            if not np.all(np.isfinite(self.weights)):
                raise ValidationError("weights contains non-finite values (NaN or inf)")
            if np.any(self.weights < 0):
                raise ValidationError("weights must be non-negative")

        # Resolve family
        trials_arr = None
        if self.trials is not None:
            try:
                trials_arr = _as1d(self.trials)
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Invalid trials array: {e}") from e
            if trials_arr.size != self.x.size:
                raise ValidationError(
                    f"trials must have the same length as x (got {trials_arr.size} and {self.x.size})"
                )
            self.trials = trials_arr

        offset_arr = None
        if self.offset is not None:
            try:
                offset_arr = _as1d(self.offset)
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Invalid offset array: {e}") from e
            if offset_arr.size != self.x.size:
                raise ValidationError(
                    f"offset must have the same length as x (got {offset_arr.size} and {self.x.size})"
                )
            self.offset = offset_arr

        self._family_obj = get_family(self.family, trials=trials_arr, offset=offset_arr)

        # Family-specific validation
        if not self._family_obj.is_gaussian:
            family_name = (
                self.family
                if isinstance(self.family, str)
                else type(self.family).__name__
            )
            if (
                isinstance(family_name, str)
                and family_name.lower() == "poisson"
                and np.any(self.y < 0)
            ):
                raise ValidationError("Poisson family requires non-negative y values")
            if isinstance(family_name, str) and family_name.lower() == "binomial":
                if trials_arr is not None:
                    if np.any(self.y < 0) or np.any(self.y > trials_arr):
                        raise ValidationError(
                            "Binomial family requires 0 <= y <= trials"
                        )
                else:
                    if np.any(self.y < 0) or np.any(self.y > 1):
                        raise ValidationError("Bernoulli family requires y in {0, 1}")

        # IRLS parameters
        if self.max_iter < 1:
            raise ValidationError(f"max_iter must be >= 1 (got {self.max_iter})")
        if self.tol <= 0:
            raise ValidationError(f"tol must be positive (got {self.tol})")

        # Shape constraint validation (§8.7)
        if self.shape is not None:
            if not isinstance(self.shape, list):
                raise ValidationError("shape must be a list of dicts")
            for i, spec in enumerate(self.shape):
                if not isinstance(spec, dict):
                    raise ValidationError(f"shape[{i}] must be a dict")
                stype = spec.get("type")
                if stype not in VALID_SHAPE_TYPES:
                    raise ValidationError(
                        f"shape[{i}]['type'] must be one of {sorted(VALID_SHAPE_TYPES)}, "
                        f"got {stype!r}"
                    )
                domain = spec.get("domain")
                if domain is not None and not (
                    isinstance(domain, (list, tuple)) and len(domain) == 2
                ):
                    raise ValidationError(
                        f"shape[{i}]['domain'] must be a (lo, hi) pair or None"
                    )
            if self.shape_kappa <= 0:
                raise ValidationError(
                    f"shape_kappa must be positive (got {self.shape_kappa})"
                )
            if self.max_shape_iter < 1:
                raise ValidationError(
                    f"max_shape_iter must be >= 1 (got {self.max_shape_iter})"
                )

        # Adaptive / variable penalty validation (§8.8)
        if self.adaptive:
            if self.adaptive_nseg < 1:
                raise ValidationError(
                    f"adaptive_nseg must be >= 1 (got {self.adaptive_nseg})"
                )
            if self.adaptive_lambda <= 0:
                raise ValidationError(
                    f"adaptive_lambda must be positive (got {self.adaptive_lambda})"
                )
            if self.adaptive_max_iter < 1:
                raise ValidationError(
                    f"adaptive_max_iter must be >= 1 (got {self.adaptive_max_iter})"
                )

        self.constraints = self.constraints or {}

    def fit(self, *, xl: float | None = None, xr: float | None = None) -> PSpline:
        """
        Fit the P-spline model.

        For Gaussian family, solves the penalized normal equations directly.
        For Poisson/Binomial families, uses IRLS (eq. 2.21–2.22).

        Parameters
        ----------
        xl : float, optional
            Left boundary of the domain. Defaults to min(x).
        xr : float, optional
            Right boundary of the domain. Defaults to max(x).

        Returns
        -------
        PSpline
            The fitted spline object.
        """
        # Validate and set domain
        x_array = np.asarray(self.x)
        x_min = float(np.min(x_array))
        x_max = float(np.max(x_array))

        if xl is not None:
            if not np.isfinite(xl):
                raise ValidationError("xl must be finite")
            if xl > x_min:
                raise ValidationError(f"xl ({xl}) must be <= min(x) ({x_min})")

        if xr is not None:
            if not np.isfinite(xr):
                raise ValidationError("xr must be finite")
            if xr < x_max:
                raise ValidationError(f"xr ({xr}) must be >= max(x) ({x_max})")

        self._xl = xl if xl is not None else x_min
        self._xr = xr if xr is not None else x_max

        if self._xl >= self._xr:
            raise ValidationError(f"xl ({self._xl}) must be < xr ({self._xr})")

        # basis and penalty
        self.B, self.knots = b_spline_basis(
            self.x, self._xl, self._xr, self.nseg, self.degree
        )
        nb = self.B.shape[1]

        # Build base penalty (may be modified by variable/adaptive modes)
        if self.penalty_gamma is not None:
            self._DtD = variable_penalty_matrix(
                nb, self.penalty_order, self.penalty_gamma
            )
        else:
            D = difference_matrix(nb, self.penalty_order)
            self._DtD = (D.T @ D).tocsr()

        # constraints
        self._build_constraints(nb)

        fam = self._family_obj
        assert fam is not None

        # Dispatch to the appropriate fitting strategy
        if self.adaptive:
            self._fit_adaptive(fam)
        elif self.shape:
            self._fit_shape_constrained(fam)
        elif fam.is_gaussian:
            self._fit_gaussian()
        else:
            self._fit_irls(fam)

        return self

    def _fit_gaussian(self) -> None:
        """Direct solve for Gaussian family (single iteration)."""
        assert self.B is not None
        assert self._DtD is not None
        self._setup_crossproducts()
        P_slope = self._build_slope_penalty()
        P: sp.spmatrix = self._DtD * self.lambda_ + P_slope  # type: ignore[operator]
        self._P_total = P
        self.coef = self._solve_coef(P)
        self.fitted_values = np.asarray(self.B @ self.coef).ravel()
        self._eta = self.fitted_values.copy()
        self.n_iter_ = 1
        self._update_uncertainty()

    def _fit_irls(self, fam: Family) -> None:
        """
        IRLS loop for GLM families (§2.12, eq. 2.21–2.22).

        Iterates until convergence or max_iter is reached.
        """
        y = np.asarray(self.y, dtype=float)
        assert self.B is not None
        assert self._DtD is not None
        B = self.B
        P_slope = self._build_slope_penalty()
        P: sp.spmatrix = self._DtD * self.lambda_ + P_slope  # type: ignore[operator]

        # Initialize
        eta, mu, w_diag = fam.initialize(y)

        coef_old = np.zeros(B.shape[1])

        for iteration in range(1, self.max_iter + 1):
            # Working response and weights
            z = fam.working_response(y, eta, mu)
            w_diag = fam.working_weights(mu, y=y)

            # Combine IRLS weights with user prior weights (§2.12.3 p.792)
            if self.weights is not None:
                w_diag = w_diag * self.weights

            W = sp.diags(w_diag)
            BtW = B.T @ W  # type: ignore[attr-defined]
            BtWB = (BtW @ B).tocsr()
            BtWz = BtW @ z

            # Store for downstream use
            self._W = W
            self._BtB = BtWB
            self._Bty = BtWz

            # Solve penalized system
            coef_new = self._solve_coef(P)

            # Update linear predictor and mean
            eta = np.asarray(B @ coef_new).ravel()
            mu = fam.inverse_link(eta)

            # Check convergence
            norm_diff = np.linalg.norm(coef_new - coef_old)
            norm_old = np.linalg.norm(coef_old) + 1e-10
            if norm_diff / norm_old < self.tol:
                coef_old = coef_new
                break

            coef_old = coef_new
        else:
            raise ConvergenceError(
                f"IRLS did not converge after {self.max_iter} iterations "
                f"(relative change: {norm_diff / norm_old:.2e})"
            )

        self.coef = coef_old
        self._eta = eta
        self.fitted_values = mu
        self._P_total = P
        self.n_iter_ = iteration
        self._update_uncertainty()

    # ------------------------------------------------------------------
    # Shape-constrained fitting (§8.7, eq. 8.14–8.15)
    # ------------------------------------------------------------------

    def _shape_mask_for_spec(
        self, spec: dict[str, Any], nb: int, diff_order: int
    ) -> np.ndarray | None:
        """
        Convert a ``domain=(lo, hi)`` specification into a boolean mask
        over the ``n - diff_order`` rows of the difference matrix.
        """
        domain = spec.get("domain")
        if domain is None:
            return None

        lo, hi = domain
        # Map coefficient indices to approximate x-positions via knot midpoints
        assert self._xl is not None and self._xr is not None
        dx = (self._xr - self._xl) / self.nseg
        # Coefficient j corresponds roughly to x = xl + (j - degree/2) * dx
        coef_x = self._xl + (np.arange(nb) - self.degree / 2) * dx
        # For diff_order d, difference j involves coefficients j..j+d
        # Use the midpoint of those coefficients as the position
        m = nb - diff_order
        diff_x = np.array([coef_x[j : j + diff_order + 1].mean() for j in range(m)])
        lo_val = lo if lo is not None else self._xl
        hi_val = hi if hi is not None else self._xr
        return (diff_x >= lo_val) & (diff_x <= hi_val)

    def _build_shape_penalty(self, alpha: np.ndarray) -> sp.spmatrix:
        """
        Build the combined shape penalty ``κ Σ D_k' V_k D_k`` for the
        current coefficient vector α.  Returns a sparse (nb, nb) matrix.
        """
        nb = alpha.shape[0]
        P_shape = sp.csr_matrix((nb, nb))
        for spec in self.shape:  # type: ignore[union-attr]
            from .penalty import _SHAPE_TYPES

            stype = spec["type"]
            diff_order, _ = _SHAPE_TYPES[stype]
            mask = self._shape_mask_for_spec(spec, nb, diff_order)
            P_shape = P_shape + asymmetric_penalty_matrix(
                alpha,
                stype,
                mask=mask,
            )
        return self.shape_kappa * P_shape

    def _fit_shape_constrained(self, fam: Family) -> None:
        """
        Iterative fitting with asymmetric shape penalties (§8.7).

        For Gaussian: direct solution per iteration.
        For GLM: runs full IRLS at each shape iteration.
        """
        assert self.B is not None
        assert self._DtD is not None
        B = self.B
        nb = B.shape[1]
        y = np.asarray(self.y, dtype=float)

        # Also build slope-zero penalty if requested
        P_slope = self._build_slope_penalty()

        # Initial solve without shape penalty
        if fam.is_gaussian:
            self._setup_crossproducts()
            P_base: sp.spmatrix = self._DtD * self.lambda_ + P_slope  # type: ignore[operator]
            alpha = self._solve_coef(P_base)
        else:
            eta, mu, _ = fam.initialize(y)
            alpha = np.zeros(nb)
            # One IRLS pass to get starting coefficients
            alpha, mu, eta = self._irls_inner(
                fam,
                y,
                self._DtD * self.lambda_ + P_slope,  # type: ignore[operator]
                alpha,
            )

        for shape_iter in range(1, self.max_shape_iter + 1):
            # Build asymmetric penalty from current α
            P_shape = self._build_shape_penalty(alpha)
            P_total: sp.spmatrix = self._DtD * self.lambda_ + P_shape + P_slope  # type: ignore[operator]

            if fam.is_gaussian:
                alpha_new = self._solve_coef(P_total)
            else:
                alpha_new, mu, eta = self._irls_inner(fam, y, P_total, alpha)

            # Check V stability (binary convergence)
            P_shape_new = self._build_shape_penalty(alpha_new)
            # Compare sparsity patterns as a convergence proxy
            V_new = P_shape_new - P_shape  # type: ignore[operator]
            diff_norm = (
                sp.linalg.norm(V_new)  # type: ignore[call-overload]
                if sp.issparse(V_new)
                else np.linalg.norm(np.asarray(V_new.toarray()))  # type: ignore[union-attr]
            )

            alpha = alpha_new
            if diff_norm < 1e-10:
                break

        self.coef = alpha
        self._P_total = (
            self._DtD * self.lambda_ + self._build_shape_penalty(alpha) + P_slope  # type: ignore[operator]
        )
        if fam.is_gaussian:
            self.fitted_values = np.asarray(B @ alpha).ravel()
            self._eta = self.fitted_values.copy()
        else:
            self._eta = np.asarray(B @ alpha).ravel()
            self.fitted_values = fam.inverse_link(self._eta)
        self.n_iter_ = shape_iter
        self._update_uncertainty()

    def _setup_crossproducts(self) -> None:
        """Pre-compute B'WB and B'Wy for Gaussian fitting."""
        assert self.B is not None
        if self.weights is not None:
            self._W = sp.diags(self.weights)  # type: ignore[arg-type]
            BtW = self.B.T @ self._W  # type: ignore[attr-defined]
            self._BtB = (BtW @ self.B).tocsr()
            self._Bty = BtW @ self.y
        else:
            self._W = None
            self._BtB = (self.B.T @ self.B).tocsr()  # type: ignore[attr-defined]
            self._Bty = self.B.T @ self.y  # type: ignore[attr-defined]

    def _irls_inner(
        self,
        fam: Family,
        y: np.ndarray,
        P: sp.spmatrix,
        alpha_init: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a full IRLS loop with penalty matrix P.  Returns (coef, mu, eta).
        """
        assert self.B is not None
        B = self.B
        eta, mu, _ = fam.initialize(y)
        coef_old = alpha_init.copy()

        for iteration in range(1, self.max_iter + 1):
            z = fam.working_response(y, eta, mu)
            w_diag = fam.working_weights(mu, y=y)
            if self.weights is not None:
                w_diag = w_diag * self.weights

            W = sp.diags(w_diag)
            BtW = B.T @ W  # type: ignore[attr-defined]
            self._W = W
            self._BtB = (BtW @ B).tocsr()
            self._Bty = BtW @ z

            coef_new = self._solve_coef(P)
            eta = np.asarray(B @ coef_new).ravel()
            mu = fam.inverse_link(eta)

            norm_diff = np.linalg.norm(coef_new - coef_old)
            norm_old = np.linalg.norm(coef_old) + 1e-10
            if norm_diff / norm_old < self.tol:
                return coef_new, mu, eta
            coef_old = coef_new

        return coef_old, mu, eta

    # ------------------------------------------------------------------
    # Flat-slope subdomain penalty (§8.7, eq. 8.12–8.13)
    # ------------------------------------------------------------------

    def _build_slope_penalty(self) -> sp.spmatrix:
        """
        Build a quadratic penalty that forces the slope to zero on specified
        subdomains.  Returns a sparse (nb, nb) matrix (already κ-scaled).

        Activated via ``constraints={"slope_zero": {"domain": (lo, hi)}}``.
        """
        assert self.B is not None
        nb = self.B.shape[1]
        spec = self.constraints.get("slope_zero") if self.constraints else None
        if spec is None:
            return sp.csr_matrix((nb, nb))

        domain = spec.get("domain")
        if domain is None or len(domain) != 2:
            raise ValidationError(
                "slope_zero constraint must have a 'domain' (lo, hi) pair"
            )
        lo, hi = domain
        kappa = spec.get("kappa", 1e8)
        n_grid = spec.get("n_grid", 50)

        # Evaluate derivative basis on a fine grid in [lo, hi]
        x_grid = np.linspace(lo, hi, n_grid)
        assert self._xl is not None and self._xr is not None
        B_deriv, _ = b_spline_derivative_basis(
            x_grid,
            self._xl,
            self._xr,
            self.nseg,
            self.degree,
            deriv_order=1,
            knots=self.knots,
        )
        # Penalty: κ * B̃'B̃  (forces B̃ @ Δα ≈ 0 on the grid)
        return (kappa * (B_deriv.T @ B_deriv)).tocsr()  # type: ignore[attr-defined, no-any-return]

    # ------------------------------------------------------------------
    # Adaptive penalty fitting (§8.8)
    # ------------------------------------------------------------------

    def _fit_adaptive(self, fam: Family) -> None:
        """
        Fit with adaptive (locally varying) penalty weights.

        Alternates between:
          1. Solve P-spline with current penalty weights V
          2. Estimate new weights from local roughness of α

        The penalty weights are modelled as a smooth function of the
        coefficient index: ``log v_j = B̆ β``, where ``B̆`` is a secondary
        B-spline basis over the index space, smoothed with its own λ.
        """
        assert self.B is not None
        B = self.B
        nb = B.shape[1]
        y = np.asarray(self.y, dtype=float)
        D = difference_matrix(nb, self.penalty_order)
        m = D.shape[0]  # number of differences

        # Secondary basis for modelling log-weights over index space
        idx = np.arange(m, dtype=float)
        from .basis import b_spline_basis as _bsb

        B_w, _ = _bsb(idx, 0, m - 1, self.adaptive_nseg, degree=3)
        nb_w = B_w.shape[1]
        D_w = difference_matrix(nb_w, 2)
        DtD_w = (D_w.T @ D_w).tocsr()

        # Start with uniform weights
        weights = np.ones(m)

        # Initial fit
        self._setup_crossproducts()
        P = adaptive_penalty_matrix(nb, self.penalty_order, weights)
        P_total: sp.spmatrix = P * self.lambda_  # type: ignore[operator]
        alpha = (
            self._solve_coef(P_total)
            if fam.is_gaussian
            else self._irls_inner(fam, y, P_total, np.zeros(nb))[0]
        )

        for adapt_iter in range(1, self.adaptive_max_iter + 1):
            # Local roughness: squared differences
            diffs = np.asarray(D @ alpha).ravel()
            roughness = diffs**2
            # Avoid log(0)
            log_r = np.log(np.maximum(roughness, 1e-20))

            # Smooth log-roughness with secondary P-spline
            BtB_w = (B_w.T @ B_w).tocsr()
            Bty_w = B_w.T @ log_r
            A_w = (BtB_w + self.adaptive_lambda * DtD_w).tocsr()
            beta_w = spsolve(A_w, Bty_w)
            log_v_smooth = np.asarray(B_w @ beta_w).ravel()

            # New weights: inverse of smoothed roughness
            # High roughness → low penalty weight (allow wiggles)
            # Low roughness → high penalty weight (enforce smoothness)
            weights_new = np.exp(-log_v_smooth)
            # Normalise so mean weight = 1 (global λ controls overall level)
            weights_new = weights_new / weights_new.mean()

            # Refit with new weights
            P = adaptive_penalty_matrix(nb, self.penalty_order, weights_new)
            P_total = P * self.lambda_  # type: ignore[operator]
            if fam.is_gaussian:
                alpha_new = self._solve_coef(P_total)
            else:
                alpha_new, mu, eta = self._irls_inner(fam, y, P_total, alpha)

            # Check convergence
            w_change = np.linalg.norm(weights_new - weights) / (
                np.linalg.norm(weights) + 1e-10
            )
            a_change = np.linalg.norm(alpha_new - alpha) / (
                np.linalg.norm(alpha) + 1e-10
            )
            weights = weights_new
            alpha = alpha_new
            if w_change < self.tol and a_change < self.tol:
                break

        self.coef = alpha
        self._adaptive_weights = weights
        self._P_total = (
            adaptive_penalty_matrix(nb, self.penalty_order, weights) * self.lambda_
        )  # type: ignore[operator]
        if fam.is_gaussian:
            self.fitted_values = np.asarray(B @ alpha).ravel()
            self._eta = self.fitted_values.copy()
        else:
            self._eta = np.asarray(B @ alpha).ravel()
            self.fitted_values = fam.inverse_link(self._eta)
        self.n_iter_ = adapt_iter
        self._update_uncertainty()

    def predict(
        self,
        x_new: ArrayLike,
        *,
        derivative_order: int | None = None,
        return_se: bool = False,
        se_method: str = "analytic",
        type: str = "response",
        B_boot: int = 5000,
        seed: int | None = None,
        n_jobs: int = 1,
        ci_prob: float = 0.95,
    ) -> NDArray | tuple[NDArray, ...]:
        """
        Predict smooth (or derivative) with optional uncertainty.

        Parameters
        ----------
        x_new : array-like
            Points at which to evaluate the spline.
        derivative_order : int, optional
            Order of derivative to compute. None for function values.
        return_se : bool, default False
            Whether to return standard errors.
        se_method : str, default "analytic"
            Method for computing uncertainty:
            - 'analytic': delta-method SE -> (fhat, se)
            - 'bootstrap': parametric bootstrap SEs -> (fhat, se)
            - 'bayes': posterior mean + credible interval -> (mean, lower, upper)
        type : str, default "response"
            Scale for predictions:
            - 'response': predictions on response scale (mu for Poisson, pi for Binomial)
            - 'link': predictions on linear predictor scale (eta)
            For Gaussian, both are identical.
        B_boot : int, default 5000
            Number of bootstrap replicates (for bootstrap method).
        seed : int, optional
            Random seed (for bootstrap method).
        n_jobs : int, default 1
            Number of parallel jobs (for bootstrap method).
        ci_prob : float, default 0.95
            Probability for credible interval (for Bayesian method).

        Returns
        -------
        NDArray or tuple
            Predictions, optionally with uncertainty estimates.
            For GLM with return_se=True and type="response":
              returns (mu_hat, lower, upper) with CI on response scale.
            For type="link" with return_se=True:
              returns (eta_hat, se_eta).
        """
        if self.coef is None:
            raise FittingError("Model not fitted. Call fit() first.")

        # Validate input
        try:
            xq = _as1d(x_new)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid x_new array: {e}") from e

        if xq.size == 0:
            raise ValidationError("x_new cannot be empty")
        if not np.all(np.isfinite(xq)):
            raise ValidationError("x_new contains non-finite values (NaN or inf)")

        # Validate parameters
        if derivative_order is not None and derivative_order <= 0:
            raise ValidationError(
                f"derivative_order must be positive (got {derivative_order})"
            )
        if se_method not in ("analytic", "bootstrap", "bayes"):
            raise ValidationError(
                f"se_method must be 'analytic', 'bootstrap', or 'bayes' (got '{se_method}')"
            )
        if type not in ("response", "link"):
            raise ValidationError(f"type must be 'response' or 'link' (got '{type}')")
        if B_boot <= 0:
            raise ValidationError(f"B_boot must be positive (got {B_boot})")
        if n_jobs == 0 or n_jobs < -1:
            raise ValidationError(f"n_jobs must be positive or -1 (got {n_jobs})")
        if not (0 < ci_prob < 1):
            raise ValidationError(f"ci_prob must be between 0 and 1 (got {ci_prob})")

        fam = self._family_obj
        assert fam is not None

        # Bayesian credible band
        if se_method == "bayes":
            if self.trace is None:
                raise FittingError("Call bayes_fit() first to sample the posterior.")
            if self._spline is None:
                raise FittingError("No BSpline stored. Run bayes_fit() first.")

            # Evaluate stored BSpline (or its k-th derivative)
            if derivative_order is None:
                Bq = self._spline(xq)
            else:
                Bq = self._spline(xq, nu=derivative_order)
            # to array for matmul
            Bq = Bq.toarray() if sp.issparse(Bq) else np.asarray(Bq)  # type: ignore[attr-defined]

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
                raise FittingError(
                    f"Mismatch: posterior alpha has length {p}, but basis has {Bq.shape[1]} cols"
                )

            # draws of f^(k)(x): (n_samples, n_points)
            deriv_draws = alpha_draws @ Bq.T

            # summarize
            mean = deriv_draws.mean(axis=0)
            lower = np.percentile(deriv_draws, (1 - ci_prob) / 2 * 100, axis=0)
            upper = np.percentile(deriv_draws, (1 + ci_prob) / 2 * 100, axis=0)
            return mean, lower, upper

        # Bootstrap SEs
        if se_method == "bootstrap" and return_se:
            return self._bootstrap_predict(xq, derivative_order, B_boot, seed, n_jobs)

        # Analytic or plain prediction
        if derivative_order is None:
            if self._xl is None or self._xr is None:
                raise FittingError("Domain bounds not set. Call fit() first.")
            Bq, _ = b_spline_basis(xq, self._xl, self._xr, self.nseg, self.degree)  # type: ignore[assignment]
        else:
            if self._xl is None or self._xr is None:
                raise FittingError("Domain bounds not set. Call fit() first.")
            Bq, _ = b_spline_derivative_basis(  # type: ignore[assignment]
                xq,
                self._xl,
                self._xr,
                self.nseg,
                self.degree,
                derivative_order,
                self.knots,
            )

        # Linear predictor
        eta_hat: NDArray = np.asarray(Bq @ self.coef).ravel()

        if not return_se:
            if type == "link" or fam.is_gaussian:
                return eta_hat
            return np.asarray(fam.inverse_link(eta_hat))

        if se_method != "analytic":
            raise ValidationError(f"Unknown se_method: {se_method}")

        if self._Ainv is None:
            raise FittingError("Analytic SEs unavailable. Call fit() first.")

        se_eta = self._compute_se(Bq)

        if type == "link" or fam.is_gaussian:
            return eta_hat, se_eta

        # Response scale: transform CI endpoints via inverse link (§2.12.3)
        z_val = 1.96  # ~95% CI half-width
        eta_lower = eta_hat - z_val * se_eta
        eta_upper = eta_hat + z_val * se_eta
        mu_hat = np.asarray(fam.inverse_link(eta_hat))
        mu_lower = np.asarray(fam.inverse_link(eta_lower))
        mu_upper = np.asarray(fam.inverse_link(eta_upper))
        return mu_hat, mu_lower, mu_upper

    def derivative(
        self,
        x_new: ArrayLike,
        *,
        deriv_order: int = 1,
        return_se: bool = False,
        se_method: str = "analytic",
        **kwargs: Any,
    ) -> NDArray | tuple[NDArray, ...]:
        """
        Compute k-th derivative of the fitted spline.

        Parameters
        ----------
        x_new : array-like
            Points at which to evaluate the derivative.
        deriv_order : int, default 1
            Order of derivative to compute.
        return_se : bool, default False
            Whether to return standard errors.
        se_method : str, default "analytic"
            Method for computing standard errors ("analytic" or "bootstrap").
        **kwargs
            Additional arguments passed to bootstrap method if applicable.

        Returns
        -------
        NDArray or tuple
            Derivative values, optionally with standard errors.
        """
        if self.coef is None:
            raise FittingError("Model not fitted. Call fit() first.")

        # Validate input
        try:
            xq = _as1d(x_new)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid x_new array: {e}") from e

        if xq.size == 0:
            raise ValidationError("x_new cannot be empty")
        if not np.all(np.isfinite(xq)):
            raise ValidationError("x_new contains non-finite values (NaN or inf)")
        if deriv_order <= 0:
            raise ValidationError(f"deriv_order must be positive (got {deriv_order})")
        if se_method not in ("analytic", "bootstrap", "bayes"):
            raise ValidationError(
                f"se_method must be 'analytic', 'bootstrap', or 'bayes' (got '{se_method}')"
            )

        # Use predict method for bootstrap and Bayesian methods
        if se_method in ("bootstrap", "bayes"):
            return self.predict(
                x_new,
                derivative_order=deriv_order,
                return_se=return_se,
                se_method=se_method,
                type="link",  # derivatives are always on link scale
                **kwargs,
            )

        # Direct computation for analytic method (more efficient)
        if self._xl is None or self._xr is None:
            raise FittingError("Domain bounds not set. Call fit() first.")
        Bq, _ = b_spline_derivative_basis(
            xq,
            self._xl,
            self._xr,
            self.nseg,
            self.degree,
            deriv_order,
            self.knots,
        )
        fhat: NDArray = np.asarray(Bq @ self.coef).ravel()

        if not return_se:
            return fhat

        if self._Ainv is None:
            raise FittingError("Analytic SEs unavailable. Call fit() first.")

        se = self._compute_se(Bq)
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
        random_seed: int | None = None,
        adaptive: bool = False,
    ) -> Any:
        """
        Fit the P-spline model using Bayesian inference via PyMC.

        Two modes are available:

        **Standard (default, adaptive=False)** — implements the Bayesian
        P-spline of Eilers & Marx §3.5 / Lang & Brezger (2003) with a
        single penalty parameter:

        .. math::

            \\lambda \\sim \\text{Gamma}(a, b), \\quad
            \\alpha \\mid \\lambda \\sim N(0, (\\lambda D'D + \\epsilon I)^{-1}), \\quad
            \\sigma \\sim \\text{InverseGamma}(c, d)

        **Adaptive (adaptive=True)** — uses one penalty parameter per row of
        the difference matrix, allowing spatially varying smoothness
        (similar to §8.8):

        .. math::

            \\lambda_j \\sim \\text{Gamma}(a, b), \\quad
            \\alpha \\mid \\lambda \\sim N(0, (D' \\Lambda D + \\epsilon I)^{-1})

        Requires the ``bayes`` optional dependencies: ``pip install psplines[bayes]``

        Parameters
        ----------
        a : float, default 2.0
            Shape parameter for the Gamma prior on penalty lambda(s).
        b : float, default 0.1
            Rate parameter for the Gamma prior on penalty lambda(s).
        c : float, default 2.0
            Shape parameter for the InverseGamma prior on sigma.
        d : float, default 1.0
            Scale parameter for the InverseGamma prior on sigma.
        draws : int, default 2000
            Number of MCMC posterior draws per chain.
        tune : int, default 2000
            Number of tuning (warmup) steps per chain.
        chains : int, default 4
            Number of MCMC chains.
        cores : int, default 4
            Number of CPU cores for parallel sampling.
        target_accept : float, default 0.9
            Target acceptance rate for the NUTS sampler.
        random_seed : int, optional
            Random seed for reproducibility.
        adaptive : bool, default False
            If False, use a single scalar penalty lambda (§3.5).
            If True, use per-difference lambdas for spatially adaptive
            smoothing (§8.8).

        Returns
        -------
        arviz.InferenceData
            The posterior trace object.
        """
        pm = _import_pymc()
        # Prepare basis and penalty
        x_array = np.asarray(self.x)
        self._xl = float(np.min(x_array)) if self._xl is None else self._xl
        self._xr = float(np.max(x_array)) if self._xr is None else self._xr
        B_sp, self.knots = b_spline_basis(
            self.x, self._xl, self._xr, self.nseg, self.degree
        )
        B = B_sp.toarray() if sp.issparse(B_sp) else B_sp
        nb = B.shape[1]
        D = difference_matrix(nb, self.penalty_order).toarray()
        DtD = D.T @ D
        y = self.y
        I_nb = np.eye(nb)

        # store BSpline object for predict
        coeffs = np.eye(nb)
        self._spline = BSpline(self.knots, coeffs, self.degree, extrapolate=False)

        with pm.Model():
            if adaptive:
                # Per-difference lambdas (spatially adaptive, §8.8)
                lam = pm.Gamma("lam", alpha=a, beta=b, shape=D.shape[0])
                Q = pm.math.dot(D.T * lam, D)
            else:
                # Single scalar lambda (standard Bayesian P-spline, §3.5)
                lam = pm.Gamma("lam", alpha=a, beta=b)
                Q = lam * DtD

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

    def plot_lam_trace(self, figsize: tuple[int, int] = (8, 6)) -> None:
        """
        Plot trace and marginal posterior for each lambda_j.
        """
        az = _import_arviz()
        az.plot_trace(self.trace, var_names=["lam"], figsize=figsize)

    def plot_alpha_trace(self, figsize: tuple[int, int] = (8, 6)) -> None:
        """
        Plot trace and marginal posterior for alpha coefficients.
        """
        az = _import_arviz()
        az.plot_trace(self.trace, var_names=["alpha"], figsize=figsize)

    def plot_posterior(self, figsize: tuple[int, int] = (8, 6)) -> None:
        """
        Plot posterior
        """
        az = _import_arviz()
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
        deriv: int | None,
        B: int,
        seed: int | None,
        n_jobs: int,
    ) -> tuple[NDArray, NDArray]:
        """
        Parametric residual bootstrap SEs (parallelized).
        """
        assert self._xl is not None and self._xr is not None
        assert self._BtB is not None and self._DtD is not None
        assert self.B is not None
        # baseline prediction and precompute evaluation basis once
        if deriv is None:
            baseline = self.predict(xq, type="link")
            Bq, _ = b_spline_basis(xq, self._xl, self._xr, self.nseg, self.degree)
        else:
            baseline = self.predict(xq, derivative_order=deriv, type="link")
            Bq, _ = b_spline_derivative_basis(
                xq, self._xl, self._xr, self.nseg, self.degree, deriv, self.knots
            )
        # prepare system
        lam = self.lambda_
        A = (self._BtB + self._DtD * lam).tocsr()  # type: ignore[operator]
        has_C = self._C is not None
        if has_C:
            assert self._C is not None
            nc = self._C.shape[0]
            zero = sp.csr_matrix((nc, nc))
            top = sp.hstack([A, self._C.T], format="csr")  # type: ignore[call-overload, attr-defined]
            bot = sp.hstack([self._C, zero], format="csr")  # type: ignore[call-overload]
            A_aug = sp.vstack([top, bot], format="csr")
        n = np.asarray(self.y).size
        rng = np.random.default_rng(seed)

        # For Gaussian: residual bootstrap on fitted values
        # Use sigma2 for variance (backward compat)
        _sigma2 = self.sigma2 if self.sigma2 is not None else 1.0
        if self._W is not None and self.weights is not None:
            w = np.asarray(self.weights)
            scale = np.where(w > 0, np.sqrt(_sigma2 / w), 0.0)
            R = rng.standard_normal((B, n)) * scale
        else:
            R = rng.standard_normal((B, n)) * np.sqrt(_sigma2)

        # For Gaussian bootstrap, fitted_values are on response scale
        # Use _eta for link-scale base
        eta_base: NDArray = np.asarray(
            self._eta if self._eta is not None else self.fitted_values
        )

        # cache weighted B transpose for RHS
        BT = self.B.T @ self._W if self._W is not None else self.B.T  # type: ignore[attr-defined]

        def one(i: int) -> NDArray:
            y_star = eta_base + R[i]
            bty = BT @ y_star
            if not has_C:
                coef_s = spsolve(A, bty)
            else:
                rhs = np.concatenate([bty, np.zeros(nc)])  # type: ignore[possibly-undefined]
                sol = spsolve(A_aug, rhs)  # type: ignore[possibly-undefined]
                coef_s = sol[: A.shape[0]]
            return np.asarray(Bq @ coef_s).ravel()

        sims = Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(B))
        sims = np.vstack(sims)
        se_boot = sims.std(axis=0, ddof=1)
        return np.asarray(baseline), se_boot  # type: ignore[arg-type]

    def _build_constraints(self, nb: int) -> None:
        """
        Build boundary derivative constraints.
        """
        if self.constraints is None:
            self._C = None
            return
        dcf = self.constraints.get("deriv")
        if not dcf:
            self._C = None
            return
        assert self._xl is not None and self._xr is not None
        x_arr = np.asarray(self.x)
        rows = []
        order = dcf.get("order", 1)
        if dcf.get("initial") == 0:
            B0, _ = b_spline_derivative_basis(
                x_arr[0], self._xl, self._xr, self.nseg, self.degree, order, self.knots
            )
            rows.append(B0)
        if dcf.get("final") == 0:
            B1, _ = b_spline_derivative_basis(
                x_arr[-1],
                self._xl,
                self._xr,
                self.nseg,
                self.degree,
                order,
                self.knots,
            )
            rows.append(B1)
        if not rows:
            self._C = None
            return
        self._C = sp.vstack(rows).tocsr()

    def _solve_coef(self, P: sp.spmatrix) -> NDArray:
        """
        Solve penalized system for coefficients.
        """
        assert self._BtB is not None and self._Bty is not None
        A = (self._BtB + P).tocsr()  # type: ignore[operator]
        if self._C is None:
            return spsolve(A, self._Bty)  # type: ignore[no-any-return]
        assert self._C is not None
        nc = self._C.shape[0]
        zero = sp.csr_matrix((nc, nc))
        top = sp.hstack([A, self._C.T], format="csr")  # type: ignore[call-overload, attr-defined]
        bot = sp.hstack([self._C, zero], format="csr")  # type: ignore[call-overload]
        A_aug = sp.vstack([top, bot], format="csr")
        rhs = np.concatenate([self._Bty, np.zeros(nc)])
        sol = spsolve(A_aug, rhs)
        return sol[: A.shape[0]]

    def _compute_se(self, Bq: sp.spmatrix | NDArray) -> NDArray:
        """
        Compute pointwise standard errors on the link scale.

        se_i = sqrt(phi * Bq[i,:] @ A^{-1} @ Bq[i,:]')

        For Gaussian: phi = sigma2.
        For Poisson/Binomial: phi = 1 (or estimated for overdispersion).

        Based on Eilers & Marx (2021), eq. (2.16) and (2.27).
        """
        if self._Ainv is None or self.phi_ is None:
            raise FittingError("Uncertainty not computed. Call fit() first.")
        Bq_dense = Bq.toarray() if sp.issparse(Bq) else np.asarray(Bq)  # type: ignore[union-attr]
        V = (self.phi_ * self._Ainv) @ Bq_dense.T
        var = np.einsum("ij,ji->i", Bq_dense, V)
        return np.sqrt(np.maximum(var, 0.0))  # type: ignore[no-any-return]

    def _update_uncertainty(self) -> None:
        """
        Compute ED, sigma2/phi, A^{-1}, and analytic SEs.

        For Gaussian: phi = sigma2 = RSS / (n - ED).
        For GLM: phi from family (1 for Poisson/Binomial), deviance stored.

        When shape or adaptive penalties are active, uses the total penalty
        ``_P_total`` instead of the basic ``λ D'D``.
        """
        if self.B is None or self.fitted_values is None:
            raise FittingError("Model must be fitted before computing uncertainty")

        fam = self._family_obj
        assert fam is not None

        # Use total penalty if available (shape / adaptive); else standard
        assert self._DtD is not None and self._BtB is not None
        P: sp.spmatrix = (
            self._P_total if self._P_total is not None else self._DtD * self.lambda_  # type: ignore[operator]
        )

        # ED = trace(A⁻¹ B'WB) where A = B'WB + P
        BtB = self._BtB
        A = (BtB + P).tocsr()  # type: ignore[operator]
        A_dense = A.toarray()
        BtB_dense: NDArray = BtB.toarray() if sp.issparse(BtB) else np.asarray(BtB)  # type: ignore[union-attr, attr-defined]
        invA = np.linalg.inv(A_dense)
        self.ED = float(np.trace(BtB_dense @ invA))

        if fam.is_gaussian:
            # Gaussian: deviance = RSS, phi = sigma2
            resid = np.asarray(self.y) - np.asarray(self.fitted_values)
            n_obs = np.asarray(self.y).size
            dof = n_obs - self.ED
            if dof <= 0:
                rss = 1.0
            elif self._W is not None:
                rss = float(resid @ (self._W @ resid))
            else:
                rss = float(resid @ resid)
            self.deviance_ = rss
            self.sigma2 = rss / dof if dof > 0 else 1.0
            self.phi_ = self.sigma2
        else:
            # GLM: deviance from family, phi from family
            self.deviance_ = fam.deviance(
                np.asarray(self.y), np.asarray(self.fitted_values)
            )
            self.phi_ = fam.phi(self.deviance_, np.asarray(self.y).size, self.ED)
            self.sigma2 = self.phi_  # alias for backward compat

        # Store A^{-1} for SE computation
        self._Ainv = invA

        # Coefficient SEs: sqrt(diag(phi * A^{-1}))
        self.se_coef = np.sqrt(np.abs(self.phi_ * np.diag(self._Ainv)))

        # Fitted value SEs on link scale
        assert self.B is not None
        se_link = self._compute_se(self.B)
        self.se_fitted = se_link

    def __repr__(self) -> str:
        st = "fitted" if self.coef is not None else "unfitted"
        n_obs = np.asarray(self.x).size
        fam_name = (
            self.family if isinstance(self.family, str) else type(self.family).__name__
        )
        return f"<PSpline {st}; n={n_obs};seg={self.nseg};deg={self.degree};d={self.penalty_order};family={fam_name}>"

"""psplines.optimize – Sparse smoothing‑parameter selection
====================================================

Utilities to choose Lambda for P‑splines via:
  • GCV (§3.1)
  • AIC (§3.2) — works for both Gaussian and GLM families
  • L‑curve and V‑curve (§3.3)

All computations use sparse back‑end (no dense BtB or DtD).

Usage:
```python
from psplines.optimize import cross_validation, aic, l_curve, v_curve
best_lam, score = cross_validation(ps)
```"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .core import PSpline
    from .glm import Family

import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve

from .exceptions import OptimizationError
from .penalty import difference_matrix, variable_penalty_matrix
from .utils_math import effective_df

__all__ = [
    "cross_validation",
    "aic",
    "l_curve",
    "v_curve",
    "variable_penalty_cv",
    "plot_diagnostics",
]


# ----------------------------------------------------------------------------
def _solve_coef_sparse(
    BtB: sp.spmatrix,
    DtD: sp.spmatrix,
    Bty: np.ndarray,
    lam: float,
    C: sp.spmatrix | None,
) -> np.ndarray:
    """
    Solve (BtB + lam*DtD) α = Bty with optional equality constraints C α = 0.
    Operates fully in sparse matrices via spsolve.
    """
    # penalized matrix
    A = (BtB + DtD * lam).tocsr()  # type: ignore[operator]
    if C is None:
        return spsolve(A, Bty)
    # build augmented system [[A, C^T]; [C, 0]]
    nc = C.shape[0]
    zero = csr_matrix((nc, nc))
    top = hstack([A, C.T], format="csr")  # type: ignore[call-overload, attr-defined]
    bot = hstack([C, zero], format="csr")  # type: ignore[call-overload]
    A_aug = vstack([top, bot], format="csr")
    rhs = np.concatenate([Bty, np.zeros(nc)])
    sol = spsolve(A_aug, rhs)
    return sol[: BtB.shape[0]]


def _irls_solve(
    B: sp.spmatrix,
    DtD: sp.spmatrix,
    y: np.ndarray,
    lam: float,
    fam: Family,
    C: sp.spmatrix | None,
    user_weights: np.ndarray | None = None,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run IRLS for a given lambda. Returns (coef, mu, eta).

    Used by both fit() and optimizer internals.
    """
    nb = B.shape[1]
    eta, mu, w_diag = fam.initialize(y)
    coef_old = np.zeros(nb)

    for _ in range(max_iter):
        z = fam.working_response(y, eta, mu)
        w_diag = fam.working_weights(mu, y=y)

        if user_weights is not None:
            w_diag = w_diag * user_weights

        W = diags(w_diag)
        BtW = B.T @ W  # type: ignore[attr-defined]
        BtWB = (BtW @ B).tocsr()
        BtWz = BtW @ z

        coef_new = _solve_coef_sparse(BtWB, DtD, BtWz, lam, C)
        eta = np.asarray(B @ coef_new).ravel()
        mu = fam.inverse_link(eta)

        norm_diff = np.linalg.norm(coef_new - coef_old)
        norm_old = np.linalg.norm(coef_old) + 1e-10
        if norm_diff / norm_old < tol:
            return coef_new, mu, eta

        coef_old = coef_new

    # Return best effort even if not converged (optimizer will evaluate score)
    return coef_old, mu, eta


# ----------------------------------------------------------------------------
def _optimise_lambda(
    ps: PSpline,
    score_fn: Callable[
        [float, np.ndarray, float, np.ndarray, sp.spmatrix | None], float
    ],
    bounds: tuple[float, float],
) -> tuple[float, float]:
    """
    Generic 1‑D bounded search over log10(lambda).
    score_fn(lam, coef, rss_or_dev, Dcoef, C) must return criterion to minimize.
    """
    if ps.B is None:
        ps.fit()
    # cache sparse matrices
    B = ps.B
    assert B is not None
    nb = B.shape[1]
    D = difference_matrix(nb, ps.penalty_order)
    DtD = (D.T @ D).tocsr()
    C = ps._C

    fam = ps._family_obj
    is_glm = fam is not None and not fam.is_gaussian

    # For Gaussian, precompute cross-products
    W = ps._W
    user_weights: np.ndarray | None = (
        np.asarray(ps.weights) if ps.weights is not None else None
    )
    _y = np.asarray(ps.y)
    if not is_glm:
        if W is not None:
            BtW = B.T @ W  # type: ignore[attr-defined]
            BtB = (BtW @ B).tocsr()
            Bty = BtW @ _y
        else:
            BtB = (B.T @ B).tocsr()  # type: ignore[attr-defined]
            Bty = B.T @ _y  # type: ignore[attr-defined]

    def obj(loglam: float) -> float:
        lam = 10**loglam

        if is_glm:
            assert fam is not None
            # Run IRLS for this lambda
            coef, mu, eta = _irls_solve(
                B,
                DtD,
                _y,
                lam,
                fam,
                C,
                user_weights=user_weights,
                max_iter=ps.max_iter,
                tol=ps.tol,
            )
            dev = fam.deviance(_y, mu)
            Dcoef = D @ coef
            return score_fn(lam, coef, dev, Dcoef, C)
        else:
            coef = _solve_coef_sparse(BtB, DtD, Bty, lam, C)  # type: ignore[possibly-undefined]
            fit = B @ coef
            resid = _y - fit
            if W is not None:
                rss = float(resid @ (W @ resid))
            else:
                rss = float(np.sum(resid**2))
            Dcoef = D @ coef
            return score_fn(lam, coef, rss, Dcoef, C)

    res = minimize_scalar(
        obj,
        bounds=(np.log10(bounds[0]), np.log10(bounds[1])),
        method="bounded",
    )
    if not res.success:
        raise OptimizationError(f"Lambda optimisation failed: {res.message}")
    lam_star = 10**res.x
    # update model
    ps.lambda_ = lam_star
    ps.fit()
    return lam_star, res.fun


# ----------------------------------------------------------------------------
def cross_validation(
    pspline: PSpline,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
) -> tuple[float, float]:
    """
    Find Lambda that minimizes GCV = (rss/n) / (1 - edf/n)^2.

    For GLM families, uses deviance in place of RSS.
    """

    def gcv(
        lam: float,
        coef: np.ndarray,
        rss_or_dev: float,
        Dcoef: np.ndarray,
        C: sp.spmatrix | None,
    ) -> float:
        n = np.asarray(pspline.y).size
        assert pspline.B is not None
        edf = effective_df(
            pspline.B,
            difference_matrix(pspline.B.shape[1], pspline.penalty_order),
            lam,
            W=pspline._W,
        )
        return (rss_or_dev / n) / (1 - edf / n) ** 2

    return _optimise_lambda(pspline, gcv, lambda_bounds)


# ----------------------------------------------------------------------------
def aic(
    pspline: PSpline,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
) -> tuple[float, float]:
    """
    Find Lambda that minimizes AIC = n*log(dev/n) + 2*edf.

    For Gaussian: dev = RSS. For GLM: deviance from family.
    Works for Poisson density estimation (§3.3) and binomial smoothing (§3.2).
    """

    def crit(
        lam: float,
        coef: np.ndarray,
        rss_or_dev: float,
        Dcoef: np.ndarray,
        C: sp.spmatrix | None,
    ) -> float:
        n = np.asarray(pspline.y).size
        assert pspline.B is not None
        edf = effective_df(
            pspline.B,
            difference_matrix(pspline.B.shape[1], pspline.penalty_order),
            lam,
            W=pspline._W,
        )
        return float(n * np.log(rss_or_dev / n) + 2 * edf)

    return _optimise_lambda(pspline, crit, lambda_bounds)


# ----------------------------------------------------------------------------
def _sweep_lambda(
    ps: PSpline,
    lambda_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (log_rss_or_dev, log_penalty) for each scalar Lambda in lambda_grid.
    Operates in sparse land. For GLM families, runs IRLS per lambda.
    """
    if ps.B is None:
        ps.fit()
    B = ps.B
    assert B is not None
    nb = B.shape[1]
    D = difference_matrix(nb, ps.penalty_order)
    DtD = (D.T @ D).tocsr()
    C = ps._C

    fam = ps._family_obj
    is_glm = fam is not None and not fam.is_gaussian
    user_weights: np.ndarray | None = (
        np.asarray(ps.weights) if ps.weights is not None else None
    )

    # For Gaussian, precompute cross-products
    W = ps._W
    _y = np.asarray(ps.y)
    if not is_glm:
        if W is not None:
            BtW = B.T @ W  # type: ignore[attr-defined]
            BtB = (BtW @ B).tocsr()
            Bty = BtW @ _y
        else:
            BtB = (B.T @ B).tocsr()  # type: ignore[attr-defined]
            Bty = B.T @ _y  # type: ignore[attr-defined]

    log_rss = np.full(lambda_grid.size, -np.inf)
    log_pen = np.full(lambda_grid.size, -np.inf)
    for i, lam in enumerate(lambda_grid):
        try:
            if is_glm:
                assert fam is not None
                coef, mu, eta = _irls_solve(
                    B,
                    DtD,
                    _y,
                    lam,
                    fam,
                    C,
                    user_weights=user_weights,
                    max_iter=ps.max_iter,
                    tol=ps.tol,
                )
                dev = fam.deviance(_y, mu)
                pen = float(np.sum((D @ coef) ** 2))
                log_rss[i] = np.log(max(dev, 1e-300))
                log_pen[i] = np.log(max(pen, 1e-300))
            else:
                coef = _solve_coef_sparse(BtB, DtD, Bty, lam, C)  # type: ignore[possibly-undefined]
                fit = B @ coef
                resid = _y - fit
                if W is not None:
                    rss = float(resid @ (W @ resid))
                else:
                    rss = float(np.sum(resid**2))
                pen = float(np.sum((D @ coef) ** 2))
                log_rss[i] = np.log(rss)
                log_pen[i] = np.log(pen)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            warnings.warn(
                f"Lambda sweep failed for lambda={lam:.2e}; skipping",
                stacklevel=2,
            )
            continue
    return log_rss, log_pen


# ----------------------------------------------------------------------------
def l_curve(
    pspline: PSpline,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
    num_lambda: int = 81,
    refine: bool = True,
    refine_factor: float = 10,
    refine_points: int = 81,
    smooth_kappa: bool = True,
) -> tuple[float, float]:
    """
    Pick lambda via maximum curvature of the L-curve.
    Implements two-stage search (coarse + refine), vectorized curvature,
    optional smoothing, and edge-case warnings.

    Parameters
    ----------
    pspline : PSpline
        Fitted PSpline instance (coefficient solver ready).
    lambda_bounds : tuple
        (min, max) bounds for initial lambda grid (log-uniform).
    num_lambda : int
        Number of points in the initial lambda grid.
    refine : bool
        Whether to perform a second, finer search around the coarse optimum.
    refine_factor : float
        Factor to widen/narrow bounds for refinement around coarse lambda.
    refine_points : int
        Number of points in the refined grid.
    smooth_kappa : bool
        Whether to apply a 3-point moving average to curvature values.
    """
    # Coarse grid search
    log_min, log_max = np.log10(lambda_bounds[0]), np.log10(lambda_bounds[1])
    grid = np.logspace(log_min, log_max, num_lambda)
    lr, lp = _sweep_lambda(pspline, grid)
    valid = np.isfinite(lr) & np.isfinite(lp)
    x, y, lamv = lp[valid], lr[valid], grid[valid]

    # Vectorized curvature calculation
    # central differences for dx, dy, ddx, ddy
    dx = (x[2:] - x[:-2]) * 0.5
    dy = (y[2:] - y[:-2]) * 0.5
    ddx = x[2:] - 2 * x[1:-1] + x[:-2]
    ddy = y[2:] - 2 * y[1:-1] + y[:-2]
    kappa = np.full_like(x, np.nan)
    denom = (dx * dx + dy * dy) ** 1.5
    kappa[1:-1] = np.abs(dx * ddy - dy * ddx) / denom

    # Optional smoothing of curvature
    kernel = np.ones(3) / 3
    if smooth_kappa:
        kappa = np.convolve(kappa, kernel, mode="same")

    # Identify coarse optimum
    idx = int(np.nanargmax(kappa))
    # Edge-case warning if optimum near boundary
    if idx < 2 or idx > len(x) - 3:
        warnings.warn(
            "L-curve optimum at boundary of grid; consider expanding lambda_bounds",
            UserWarning,
        )
    lam_corner = lamv[idx]
    kappa_corner = kappa[idx]

    # Optional refinement around coarse optimum
    if refine:
        lower = lam_corner / refine_factor
        upper = lam_corner * refine_factor
        log_l, log_u = np.log10(lower), np.log10(upper)
        grid2 = np.logspace(log_l, log_u, refine_points)
        lr2, lp2 = _sweep_lambda(pspline, grid2)
        valid2 = np.isfinite(lr2) & np.isfinite(lp2)
        x2, y2, lamv2 = lp2[valid2], lr2[valid2], grid2[valid2]

        dx2 = (x2[2:] - x2[:-2]) * 0.5
        dy2 = (y2[2:] - y2[:-2]) * 0.5
        ddx2 = x2[2:] - 2 * x2[1:-1] + x2[:-2]
        ddy2 = y2[2:] - 2 * y2[1:-1] + y2[:-2]
        kappa2 = np.full_like(x2, np.nan)
        denom2 = (dx2 * dx2 + dy2 * dy2) ** 1.5
        kappa2[1:-1] = np.abs(dx2 * ddy2 - dy2 * ddx2) / denom2
        if smooth_kappa:
            kappa2 = np.convolve(kappa2, kernel, mode="same")

        idx2 = int(np.nanargmax(kappa2))
        if idx2 < 2 or idx2 > len(x2) - 3:
            warnings.warn(
                "Refined L-curve optimum at boundary; expand refine_factor or refine_points",
                UserWarning,
            )
        lam_corner = lamv2[idx2]
        kappa_corner = kappa2[idx2]

    return _finish(pspline, lam_corner, kappa_corner)


# ----------------------------------------------------------------------------
def v_curve(
    pspline: PSpline,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
    num_lambda: int = 81,
) -> tuple[float, float]:
    """
    Pick Lambda via minimum distance on V‑curve.
    """
    grid = np.logspace(
        np.log10(lambda_bounds[0]), np.log10(lambda_bounds[1]), num_lambda
    )
    lr, lp = _sweep_lambda(pspline, grid)
    valid = np.isfinite(lr) & np.isfinite(lp)
    if valid.sum() < 2:
        raise OptimizationError("Not enough V‑curve points")
    dr = np.diff(lr[valid])
    dp = np.diff(lp[valid])
    dist = np.hypot(dr, dp)
    mid = np.sqrt(grid[valid][:-1] * grid[valid][1:])
    idx = int(np.argmin(dist))
    return _finish(pspline, mid[idx], dist[idx])


# ----------------------------------------------------------------------------
def variable_penalty_cv(
    pspline: PSpline,
    gamma_range: tuple[float, float] = (-20.0, 20.0),
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
    num_gamma: int = 41,
    num_lambda: int = 41,
    criterion: str = "gcv",
) -> tuple[float, float, float, np.ndarray]:
    """
    2-D grid search for variable-penalty parameters (λ, γ).

    The penalty matrix uses exponentially varying weights
    ``v_j = exp(γ j / m)`` (§8.8).  For each (λ, γ) pair the model is
    solved and scored with GCV or AIC.

    Parameters
    ----------
    pspline : PSpline
        A *fitted* PSpline instance (used for basis, knots, data).
    gamma_range : (float, float)
        Bounds for the γ grid (linear scale).
    lambda_bounds : (float, float)
        Bounds for the λ grid (log₁₀ scale).
    num_gamma : int
        Number of γ grid points.
    num_lambda : int
        Number of λ grid points.
    criterion : {{'gcv', 'aic'}}
        Selection criterion.

    Returns
    -------
    best_lambda : float
    best_gamma : float
    best_score : float
    scores : ndarray, shape (num_gamma, num_lambda)
        Full score surface.
    """
    if criterion not in ("gcv", "aic"):
        raise OptimizationError(f"criterion must be 'gcv' or 'aic', got {criterion!r}")

    if pspline.B is None:
        pspline.fit()

    B = pspline.B
    assert B is not None
    nb = B.shape[1]
    y = np.asarray(pspline.y, dtype=float)
    n = y.size
    C = pspline._C

    fam = pspline._family_obj
    is_glm = fam is not None and not fam.is_gaussian
    user_weights: np.ndarray | None = (
        np.asarray(pspline.weights) if pspline.weights is not None else None
    )

    # Pre-compute cross-products for Gaussian
    W = pspline._W
    if not is_glm:
        if W is not None:
            BtW = B.T @ W  # type: ignore[attr-defined]
            BtB = (BtW @ B).tocsr()
            Bty = BtW @ y
        else:
            BtB = (B.T @ B).tocsr()  # type: ignore[attr-defined]
            Bty = B.T @ y  # type: ignore[attr-defined]

    gamma_grid = np.linspace(gamma_range[0], gamma_range[1], num_gamma)
    lambda_grid = np.logspace(
        np.log10(lambda_bounds[0]),
        np.log10(lambda_bounds[1]),
        num_lambda,
    )
    scores = np.full((num_gamma, num_lambda), np.inf)

    for i, gamma in enumerate(gamma_grid):
        DtD_g = variable_penalty_matrix(nb, pspline.penalty_order, gamma)
        for j, lam in enumerate(lambda_grid):
            try:
                if is_glm:
                    assert fam is not None
                    coef, mu, eta = _irls_solve(
                        B,
                        DtD_g,
                        y,
                        lam,
                        fam,
                        C,
                        user_weights=user_weights,
                        max_iter=pspline.max_iter,
                        tol=pspline.tol,
                    )
                    dev = fam.deviance(y, mu)
                    # ED with this penalty
                    Bt = B.T  # type: ignore[attr-defined]
                    if W is not None:
                        assert pspline._W is not None
                        A = (Bt @ (pspline._W @ B) + lam * DtD_g).tocsr()  # type: ignore[operator]
                        BtB_glm: NDArray = (Bt @ (pspline._W @ B)).toarray()  # type: ignore[operator, attr-defined]
                    else:
                        A = (Bt @ B + lam * DtD_g).tocsr()
                        BtB_glm = (Bt @ B).toarray()  # type: ignore[attr-defined]
                    A_dense = A.toarray()
                    edf = float(np.trace(BtB_glm @ np.linalg.inv(A_dense)))
                    rss_or_dev = dev
                else:
                    P = (lam * DtD_g).tocsr()
                    A_sys = (BtB + P).tocsr()  # type: ignore[possibly-undefined, operator]
                    if C is None:
                        coef = spsolve(A_sys, Bty)  # type: ignore[possibly-undefined]
                    else:
                        coef = _solve_coef_sparse(BtB, DtD_g, Bty, lam, C)  # type: ignore[possibly-undefined]
                    fit_vals = B @ coef
                    resid = y - fit_vals
                    if W is not None:
                        rss_or_dev = float(resid @ (W @ resid))
                    else:
                        rss_or_dev = float(np.sum(resid**2))

                    A_dense = A_sys.toarray()
                    BtB_d: NDArray = (
                        BtB.toarray()  # type: ignore[possibly-undefined, attr-defined]
                        if sp.issparse(BtB)  # type: ignore[possibly-undefined]
                        else np.asarray(BtB)  # type: ignore[possibly-undefined]
                    )
                    edf = float(np.trace(BtB_d @ np.linalg.inv(A_dense)))

                if criterion == "gcv":
                    scores[i, j] = (rss_or_dev / n) / (1 - edf / n) ** 2
                else:  # aic
                    scores[i, j] = n * np.log(max(rss_or_dev / n, 1e-300)) + 2 * edf
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                continue

    # Find best
    idx = np.unravel_index(np.argmin(scores), scores.shape)
    best_gamma = float(gamma_grid[idx[0]])
    best_lambda = float(lambda_grid[idx[1]])
    best_score = float(scores[idx])

    # Update pspline
    pspline.lambda_ = best_lambda
    pspline.penalty_gamma = best_gamma
    pspline.fit()

    return best_lambda, best_gamma, best_score, scores


# ----------------------------------------------------------------------------
def _finish(ps: PSpline, lam: float, score: float) -> tuple[float, float]:
    """Update model with chosen Lambda and return (Lambda, score)."""
    ps.lambda_ = float(lam)
    ps.fit()
    return float(lam), float(score)


# ----------------------------------------------------------------------------
# optional diagnostic plotting
# -----------------------------------------------------------------------------
def plot_diagnostics(
    pspline: PSpline,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
    num_lambda: int = 81,
    which: tuple[str, ...] | None = None,
    show: bool = True,
) -> None:
    """
    Quick visual comparison of Lambda‑selection criteria.

    Parameters
    ----------
    pspline : PSpline
        Fitted P‑spline object (scalar Lambda).
    lambda_bounds : (float, float)
        Search grid bounds for Lambda.
    num_lambda : int
        Number of grid points.
    which : tuple of {{'gcv','aic','lcurve','vcurve'}} or None
        Sub‑plots to draw; default all.
    show : bool, default True
        Whether to call plt.show().
    """
    import matplotlib.pyplot as plt

    if which is None:
        which = ("gcv", "aic", "lcurve", "vcurve")
    which = tuple(w.lower() for w in which)

    # grid & raw curves
    grid = np.logspace(
        np.log10(lambda_bounds[0]), np.log10(lambda_bounds[1]), num_lambda
    )
    lr, lp = _sweep_lambda(pspline, grid)

    # effective df
    _B = pspline.B if pspline.B is not None else pspline.fit().B
    assert _B is not None
    D = difference_matrix(_B.shape[1], pspline.penalty_order)
    edf = np.array([effective_df(_B, D, lam, W=pspline._W) for lam in grid])

    # RSS/deviance, AIC, GCV
    n = np.asarray(pspline.y).size
    rss = np.exp(lr)
    sigma2 = rss / n
    aicv = n * np.log(sigma2) + 2 * edf
    gcvv = (rss / n) / (1 - edf / n) ** 2

    # set up plots
    fig, axes = plt.subplots(2, 2)
    ax = {
        "gcv": axes[0, 0],
        "aic": axes[0, 1],
        "lcurve": axes[1, 0],
        "vcurve": axes[1, 1],
    }

    if "gcv" in which:
        ax["gcv"].plot(np.log10(grid), gcvv, "o-")
        ax["gcv"].set_title("GCV score")
        ax["gcv"].set_xlabel("log10 Lambda")
    else:
        axes[0, 0].axis("off")

    if "aic" in which:
        ax["aic"].plot(np.log10(grid), aicv, "o-")
        ax["aic"].set_title("AIC score")
        ax["aic"].set_xlabel("log10 Lambda")
    else:
        axes[0, 1].axis("off")

    if "lcurve" in which:
        valid = np.isfinite(lr) & np.isfinite(lp)
        ax["lcurve"].plot(lp[valid], lr[valid], "o-")
        ax["lcurve"].set_title("L‑curve")
    else:
        axes[1, 0].axis("off")

    if "vcurve" in which:
        valid = np.isfinite(lr) & np.isfinite(lp)
        dist = np.hypot(np.diff(lr[valid]), np.diff(lp[valid]))
        mid = np.sqrt(grid[valid][:-1] * grid[valid][1:])
        ax["vcurve"].plot(np.log10(mid), dist, "o-")
        ax["vcurve"].set_title("V‑curve")
        ax["vcurve"].set_xlabel("log10 Lambda")
    else:
        axes[1, 1].axis("off")

    fig.tight_layout()
    if show:
        plt.show()

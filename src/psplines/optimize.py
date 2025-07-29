"""psplines.optimize – Sparse smoothing‑parameter selection 2025‑04‑18
====================================================

Utilities to choose Lambda for Gaussian P‑splines via:
  • GCV (§3.1)
  • AIC (§3.2)
  • L‑curve and V‑curve (§3.3)

All computations use sparse back‑end (no dense BtB or DtD).

Usage:
```python
from psplines.optimize import cross_validation, aic, l_curve, v_curve
best lam, score = cross_validation(ps)
```"""

from __future__ import annotations

from typing import Callable, Tuple
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse import vstack, hstack, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar

from .core import PSpline
from .penalty import difference_matrix
from .utils_math import effective_df

__all__ = ["cross_validation", "aic", "l_curve", "v_curve"]


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
    A = (BtB + lam * DtD).tocsr()
    if C is None:
        return spsolve(A, Bty)
    # build augmented system [[A, C^T]; [C, 0]]
    nc = C.shape[0]
    zero = csr_matrix((nc, nc))
    top = hstack([A, C.T], format="csr")
    bot = hstack([C, zero], format="csr")
    A_aug = vstack([top, bot], format="csr")
    rhs = np.concatenate([Bty, np.zeros(nc)])
    sol = spsolve(A_aug, rhs)
    return sol[: BtB.shape[0]]


# ----------------------------------------------------------------------------
def _optimise_lambda(
    ps: PSpline,
    score_fn: Callable[
        [float, np.ndarray, float, np.ndarray, sp.spmatrix | None], float
    ],
    bounds: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Generic 1‑D bounded search over log10(lambda).
    score_fn(lam, coef, rss, Dcoef, C) must return criterion to minimize.
    """
    if ps.B is None:
        ps.fit()
    # cache sparse matrices
    B = ps.B  # sparse
    nb = B.shape[1]
    D = difference_matrix(nb, ps.penalty_order)
    BtB = (B.T @ B).tocsr()
    DtD = (D.T @ D).tocsr()
    Bty = B.T @ ps.y
    C = ps._C

    def obj(loglam: float) -> float:
        lam = 10**loglam
        coef = _solve_coef_sparse(BtB, DtD, Bty, lam, C)
        fit = B @ coef
        rss = float(np.sum((ps.y - fit) ** 2))
        Dcoef = D @ coef
        return score_fn(lam, coef, rss, Dcoef, C)

    res = minimize_scalar(
        obj,
        bounds=(np.log10(bounds[0]), np.log10(bounds[1])),
        method="bounded",
    )
    if not res.success:
        raise RuntimeError(f"Lambda optimisation failed: {res.message}")
    lam_star = 10**res.x
    # update model
    ps.lambda_ = lam_star
    ps.fit()
    return lam_star, res.fun


# ----------------------------------------------------------------------------
def cross_validation(
    pspline: PSpline,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
) -> Tuple[float, float]:
    """
    Find Lambda that minimizes GCV = (rss/n) / (1 - edf/n)^2.
    """

    def gcv(lam, coef, rss, Dcoef, C):
        n = pspline.y.size
        edf = effective_df(
            pspline.B, difference_matrix(pspline.B.shape[1], pspline.penalty_order), lam
        )
        return (rss / n) / (1 - edf / n) ** 2

    return _optimise_lambda(pspline, gcv, lambda_bounds)


# ----------------------------------------------------------------------------
def aic(
    pspline: PSpline,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
) -> Tuple[float, float]:
    """
    Find Lambda that minimizes AIC = n*log(rss/n) + 2*edf.
    """

    def crit(lam, coef, rss, Dcoef, C):
        n = pspline.y.size
        edf = effective_df(
            pspline.B, difference_matrix(pspline.B.shape[1], pspline.penalty_order), lam
        )
        return n * np.log(rss / n) + 2 * edf

    return _optimise_lambda(pspline, crit, lambda_bounds)


# ----------------------------------------------------------------------------
def _sweep_lambda(
    ps: PSpline,
    lambda_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (log_rss, log_penalty) for each scalar Lambda in lambda_grid.
    Operates in sparse land.
    """
    if ps.B is None:
        ps.fit()
    B = ps.B
    nb = B.shape[1]
    D = difference_matrix(nb, ps.penalty_order)
    BtB = (B.T @ B).tocsr()
    DtD = (D.T @ D).tocsr()
    Bty = B.T @ ps.y
    C = ps._C
    log_rss = np.full(lambda_grid.size, -np.inf)
    log_pen = np.full(lambda_grid.size, -np.inf)
    for i, lam in enumerate(lambda_grid):
        try:
            coef = _solve_coef_sparse(BtB, DtD, Bty, lam, C)
            fit = B @ coef
            rss = float(np.sum((ps.y - fit) ** 2))
            pen = float(np.sum((D @ coef) ** 2))
            log_rss[i] = np.log(rss)
            log_pen[i] = np.log(pen)
        except Exception:
            continue
    return log_rss, log_pen


# ----------------------------------------------------------------------------
def l_curve(
    pspline,
    lambda_bounds=(1e-6, 1e6),
    num_lambda=81,
    refine=True,
    refine_factor=10,
    refine_points=81,
    smooth_kappa=True,
):
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
    if smooth_kappa:
        kernel = np.ones(3) / 3
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
) -> Tuple[float, float]:
    """
    Pick Lambda via minimum distance on V‑curve.
    """
    grid = np.logspace(
        np.log10(lambda_bounds[0]), np.log10(lambda_bounds[1]), num_lambda
    )
    lr, lp = _sweep_lambda(pspline, grid)
    valid = np.isfinite(lr) & np.isfinite(lp)
    if valid.sum() < 2:
        raise RuntimeError("Not enough V‑curve points")
    dr = np.diff(lr[valid])
    dp = np.diff(lp[valid])
    dist = np.hypot(dr, dp)
    mid = np.sqrt(grid[valid][:-1] * grid[valid][1:])
    idx = int(np.argmin(dist))
    return _finish(pspline, mid[idx], dist[idx])


# ----------------------------------------------------------------------------
def _finish(ps: PSpline, lam: float, score: float) -> Tuple[float, float]:
    """Update model with chosen Lambda and return (Lambda, score)."""
    ps.lambda_ = float(lam)
    ps.fit()
    return float(lam), float(score)


# ----------------------------------------------------------------------------
# optional diagnostic plotting
# -----------------------------------------------------------------------------
def plot_diagnostics(
    pspline: PSpline,
    lambda_bounds: Tuple[float, float] = (1e-6, 1e6),
    num_lambda: int = 81,
    which: Tuple[str, ...] | None = None,
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
    B = pspline.B if pspline.B is not None else pspline.fit().B
    D = difference_matrix(B.shape[1], pspline.penalty_order)
    edf = np.array([effective_df(B, D, lam) for lam in grid])

    # RSS, AIC, GCV
    n = pspline.y.size
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

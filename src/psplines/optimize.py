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
    pspline: PSpline,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
    num_lambda: int = 81,
) -> Tuple[float, float]:
    """
    Pick Lambda via maximum curvature of (log_pen, log_rss) curve.
    """
    grid = np.logspace(
        np.log10(lambda_bounds[0]), np.log10(lambda_bounds[1]), num_lambda
    )
    lr, lp = _sweep_lambda(pspline, grid)
    valid = np.isfinite(lr) & np.isfinite(lp)
    if valid.sum() < 5:
        raise RuntimeError("Not enough L‑curve points")
    x, y = lp[valid], lr[valid]
    lamv = grid[valid]
    kappa = np.full_like(x, np.nan)
    for i in range(2, len(x) - 2):
        dx = (x[i + 1] - x[i - 1]) / 2
        dy = (y[i + 1] - y[i - 1]) / 2
        ddx = x[i + 1] - 2 * x[i] + x[i - 1]
        ddy = y[i + 1] - 2 * y[i] + y[i - 1]
        kappa[i] = abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
    idx = int(np.nanargmax(kappa))
    return _finish(pspline, lamv[idx], kappa[idx])


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

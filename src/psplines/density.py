"""
psplines.density – Smooth density estimation via Poisson P-splines
==================================================================

Fits a Poisson P-spline to histogram bin counts and normalizes to
produce a continuous density estimate. Optimal λ selected via AIC.

Conservation of moments (§2.12.1, p.732–738):
  - penalty_order=1: total count preserved
  - penalty_order=2: total count + mean preserved
  - penalty_order=3: total count + mean + variance preserved

Based on Eilers & Marx (2021), §2.12.1 and §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

from .core import PSpline
from .optimize import aic as aic_optimize

__all__ = ["DensityResult", "density_estimate"]


@dataclass(slots=True)
class DensityResult:
    """Result of density estimation.

    Attributes
    ----------
    grid : NDArray
        Evaluation points (bin midpoints).
    density : NDArray
        Normalized smooth density values (integrates to ~1).
    mu : NDArray
        Raw fitted counts (before normalization).
    lambda_ : float
        Smoothing parameter used.
    pspline : PSpline
        The underlying fitted PSpline object.
    bin_width : float
        Width of histogram bins.
    """

    grid: NDArray
    density: NDArray
    mu: NDArray
    lambda_: float
    pspline: PSpline
    bin_width: float


def density_estimate(
    x: ArrayLike,
    bins: int = 100,
    xl: float | None = None,
    xr: float | None = None,
    nseg: int = 20,
    degree: int = 3,
    penalty_order: int = 3,
    lambda_: float | None = None,
    lambda_bounds: tuple[float, float] = (1e-6, 1e6),
) -> DensityResult:
    """
    Estimate a smooth density from raw data via Poisson P-spline on histogram counts.

    Parameters
    ----------
    x : array-like
        Raw data values.
    bins : int, default 100
        Number of histogram bins. Narrow bins give excellent results (§3.3).
    xl : float, optional
        Left boundary for histogram and B-spline domain.
        Defaults to min(x). Set carefully for bounded data (e.g., 0 for times).
    xr : float, optional
        Right boundary. Defaults to max(x).
    nseg : int, default 20
        Number of B-spline segments.
    degree : int, default 3
        B-spline degree.
    penalty_order : int, default 3
        Penalty order. d=3 preserves mean and variance of the raw histogram.
    lambda_ : float, optional
        Fixed smoothing parameter. If None, selected via AIC.
    lambda_bounds : tuple, default (1e-6, 1e6)
        Bounds for AIC search when lambda_ is None.

    Returns
    -------
    DensityResult
        Density estimation result with grid, density, and fitted PSpline.
    """
    x_arr = np.asarray(x, dtype=float).ravel()

    # Domain
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    hist_xl = xl if xl is not None else x_min
    hist_xr = xr if xr is not None else x_max

    # Build histogram
    bin_edges = np.linspace(hist_xl, hist_xr, bins + 1)
    counts, _ = np.histogram(x_arr, bins=bin_edges)
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Fit Poisson P-spline on counts
    counts_float = counts.astype(float)
    ps = PSpline(
        bin_mids,
        counts_float,
        nseg=nseg,
        degree=degree,
        lambda_=lambda_ if lambda_ is not None else 1.0,
        penalty_order=penalty_order,
        family="poisson",
    )
    ps.fit(xl=hist_xl, xr=hist_xr)

    # Select optimal lambda via AIC if not fixed
    if lambda_ is None:
        lam_opt, _ = aic_optimize(ps, lambda_bounds=lambda_bounds)
    else:
        lam_opt = lambda_

    # Fitted values (mu) are on response scale (counts)
    mu_raw = ps.fitted_values
    if mu_raw is None:
        raise RuntimeError("PSpline fitting failed: fitted_values is None")
    mu: np.ndarray = np.asarray(mu_raw)

    # Normalize to density: density = mu / (total_count * bin_width)
    total = np.sum(mu) * bin_width
    density = mu / total if total > 0 else mu

    return DensityResult(
        grid=bin_mids,
        density=density,
        mu=mu,
        lambda_=lam_opt,
        pspline=ps,
        bin_width=bin_width,
    )

"""
psplines.basis – Efficient B-spline basis generation
=================================================

Provides fast, sparse-friendly construction of:
  • B-spline regression basis
  • k-th derivative basis

Leverages SciPy's BSpline for vectorized evaluation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline
from numpy.typing import ArrayLike, NDArray

__all__ = ["b_spline_basis", "b_spline_derivative_basis"]


def _make_knots(xl: float, xr: float, nseg: int, degree: int) -> NDArray:
    """
    Construct knot vector with degree-fold boundary expansion.
    """
    dx = (xr - xl) / nseg
    # interior knots (nseg+1 points)
    interior = np.linspace(xl, xr, nseg + 1)
    # boundary expansions
    left = xl - dx * np.arange(degree, 0, -1)
    right = xr + dx * np.arange(1, degree + 1)
    return np.concatenate([left, interior, right])


def b_spline_basis(
    x: ArrayLike, xl: float, xr: float, nseg: int, degree: int = 3
) -> tuple[sp.csr_matrix, NDArray]:
    """
    Compute B-spline regression basis on [xl, xr].

    Parameters
    ----------
    x : array-like, shape (n,)
    xl, xr : floats
        Domain endpoints (xl < xr).
    nseg : int
        Number of equal-length segments.
    degree : int
        Degree of the spline.

    Returns
    -------
    B : csr_matrix, shape (n, nseg+degree)
        Sparse design matrix, each row has at most degree+1 nonzeros.
    knots : ndarray
        Full knot vector (length nseg + degree + 1 + degree).
    """
    x = np.asarray(x, float)
    knots = _make_knots(xl, xr, nseg, degree)
    # number of basis functions
    nb = nseg + degree
    # clip to domain
    x_clipped = np.clip(x, xl, xr)
    # build BSpline object: vectorized multi-coeff basis
    # coefficients as identity to extract each basis
    coeffs = np.eye(nb)
    spline = BSpline(knots, coeffs, degree, extrapolate=False)
    # evaluate basis at all x in one call
    B_full = spline(x_clipped)
    # convert to CSR for sparsity
    B = sp.csr_matrix(B_full)
    return B, knots


def b_spline_derivative_basis(
    x: ArrayLike,
    xl: float,
    xr: float,
    nseg: int,
    degree: int = 3,
    deriv_order: int = 1,
    knots: NDArray | None = None,
) -> tuple[sp.csr_matrix, NDArray]:
    """
    Compute k-th derivative of B-spline basis on [xl, xr].

    Parameters
    ----------
    x : array-like
    xl, xr : floats
    nseg : int
    degree : int
    deriv_order : int
        Order of derivative (0 returns same as b_spline_basis).
    knots : ndarray or None
        Precomputed knot vector.

    Returns
    -------
    B_deriv : csr_matrix, shape (n, nseg+degree)
        Sparse derivative basis.
    knots : ndarray
    """
    if deriv_order < 0:
        raise ValueError("deriv_order must be non-negative")
    if knots is None:
        knots = _make_knots(xl, xr, nseg, degree)
    # nothing to do for zero-order
    if deriv_order == 0:
        return b_spline_basis(x, xl, xr, nseg, degree)
    # if derivative exceeds degree, result is zero
    if deriv_order > degree:
        n = np.atleast_1d(x).shape[0]
        return sp.csr_matrix((n, nseg + degree)), knots

    x = np.asarray(x, float)
    x_clipped = np.clip(x, xl, xr)
    nb = nseg + degree
    # build multi-coefficient BSpline
    coeffs = np.eye(nb)
    spline = BSpline(knots, coeffs, degree, extrapolate=False)
    # evaluate derivative in one call
    B_deriv_full = spline(x_clipped, nu=deriv_order)
    B_deriv = sp.csr_matrix(B_deriv_full)
    return B_deriv, knots

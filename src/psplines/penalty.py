"""
psplines.penalty – Sparse finite-difference penalty matrix
========================================================

Generate a d-th order finite-difference operator in sparse form:

The matrix D (shape (n-d)×n) satisfies:
  (D @ α)[i] = ∑_{k=0}^d (-1)^{d-k} C(d,k) α[i+k]

References: Eilers & Marx (2021), Section 2.3 & Appendix C.
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.special import comb

__all__ = ["difference_matrix"]


def difference_matrix(n: int, order: int = 2) -> csr_matrix:
    """
    Create a sparse difference matrix of shape (n-order)×n.

    Parameters
    ----------
    n : int
        Length of the coefficient vector α.
    order : int
        Order d of the finite difference (must be >= 0).

    Returns
    -------
    D : csr_matrix
        Sparse (n-order)×n matrix implementing d-th order differences.
        If order=0, returns identity; if order>=n, returns an empty matrix.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        # zero-order = identity operator
        return diags([np.ones(n)], [0], shape=(n, n), format="csr")
    if order >= n:
        # no valid differences
        return csr_matrix((0, n))

    # number of rows
    m = n - order
    # offsets for diagonals: 0,1,...,order
    offsets = np.arange(order + 1)
    # build each diagonal of length m
    data = [((-1) ** (order - k)) * comb(order, k) * np.ones(m) for k in offsets]
    D = diags(data, offsets, shape=(m, n), format="csr")
    return D

"""
psplines.utils_math – Numerical utilities
========================================

Helper routines for P-spline smoothing:

- `effective_df`: exact or stochastic trace of smoother matrix via
  Hutchinson estimator, with LU factorization for many solves.

Based on Eilers & Marx (2021), eq. (2.15).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized, spsolve

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["effective_df", "solve_penalized"]


def solve_penalized(
    BtB: sp.spmatrix,
    Bty: np.ndarray,
    P: sp.spmatrix,
    C: sp.spmatrix | None = None,
) -> np.ndarray:
    """
    Solve the penalized system (B'WB + P) α = B'Wy with optional
    equality constraints C α = 0.

    When C is provided, solves the augmented saddle-point system:
        [[A, C'], [C, 0]] [α; λ] = [Bty; 0]

    Parameters
    ----------
    BtB : sparse (nb x nb)
        Precomputed B'WB (or B'B if unweighted).
    Bty : 1-D array, length nb
        Precomputed B'Wy (or B'y if unweighted).
    P : sparse (nb x nb)
        Total penalty matrix (already λ-scaled).
    C : sparse (nc x nb), optional
        Constraint matrix.

    Returns
    -------
    alpha : 1-D array, length nb
    """
    A = (BtB + P).tocsr()  # type: ignore[operator]
    if C is None:
        return spsolve(A, Bty)
    nc = C.shape[0]
    zero = sp.csr_matrix((nc, nc))
    top = sp.hstack([A, C.T], format="csr")  # type: ignore[call-overload, attr-defined]
    bot = sp.hstack([C, zero], format="csr")  # type: ignore[call-overload]
    A_aug = sp.vstack([top, bot], format="csr")
    rhs = np.concatenate([Bty, np.zeros(nc)])
    sol = spsolve(A_aug, rhs)
    return sol[: A.shape[0]]


def effective_df(
    B: sp.spmatrix,
    D: sp.spmatrix,
    lambda_: float | NDArray,
    BtB: sp.spmatrix | None = None,
    DtD: sp.spmatrix | None = None,
    *,
    W: sp.spmatrix | None = None,
    num_vectors: int = 20,
    exact_thresh: int = 400,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute effective degrees of freedom: trace(H) where
    H = B (B'W B + λ D'D)^{-1} B'W.

    If nb <= exact_thresh and dense, does an exact trace via inversion.
    Otherwise uses Hutchinson estimator with random ±1 vectors and
    LU factorization for fast solves.

    Parameters
    ----------
    B : (n x nb) sparse
        B-spline basis matrix.
    D : ((nb-order) x nb) sparse
        Difference penalty matrix.
    lambda_ : float or array_like
        Scalar or per-difference penalties for adaptive smoothing.
    BtB : (nb x nb) sparse, optional
        Precomputed B.T @ W @ B (if W provided) or B.T @ B.
    DtD : (nb x nb) sparse, optional
        Precomputed D.T @ D.
    W : (n x n) sparse, optional
        Diagonal weight matrix; default identity.
    num_vectors : int
        Number of random vectors for trace estimation.
    exact_thresh : int
        If nb <= exact_thresh and A dense, use exact trace.
    rng : np.random.Generator, optional
        Random number generator; if None, a new one is created.

    Returns
    -------
    edf : float
        Estimated effective degrees of freedom (>=0).
    """
    # prepare weight and cross-products
    if BtB is None:
        BtB = B.T @ B if W is None else B.T @ (W @ B)  # type: ignore[operator, attr-defined]
    if DtD is None:
        DtD = D.T @ D  # type: ignore[operator, attr-defined]
    # assemble system matrix A = BtB + λ DtD
    if np.isscalar(lambda_):
        A = BtB + lambda_ * DtD  # type: ignore[operator, arg-type]
    else:
        # adaptive penalties
        L = sp.diags(lambda_)  # type: ignore[arg-type]
        A = BtB + D.T @ (L @ D)  # type: ignore[operator, attr-defined]
    nb = A.shape[0]
    # exact trace if size small enough
    if nb <= exact_thresh:
        A_dense = A.toarray() if sp.issparse(A) else A  # type: ignore[union-attr, attr-defined]
        invA = np.linalg.inv(A_dense)
        G = (BtB.toarray() if sp.issparse(BtB) else BtB) @ invA  # type: ignore[union-attr, attr-defined]
        return float(np.trace(G))
    # stochastic Hutchinson estimator
    n = B.shape[0]
    rng = rng or np.random.default_rng()
    # factorize A once for multiple solves
    solve = factorized(A)
    # generate random ±1 matrix Z (n x num_vectors)
    Z = rng.choice([-1.0, 1.0], size=(n, num_vectors))
    # compute B^T Z  => shape (nb x num_vectors)
    BTZ = B.T @ Z  # type: ignore[operator, attr-defined]
    # solve A X = BTZ => X shape (nb x num_vectors)
    X = np.column_stack([solve(BTZ[:, i]) for i in range(num_vectors)])
    # compute B X => (n x num_vectors)
    BX = B @ X  # type: ignore[operator]
    # trace estimate = mean over random vectors of z_i^T (B X)_i
    trace_est = np.mean(np.einsum("ij,ij->j", Z, BX))
    return float(max(0.0, trace_est))

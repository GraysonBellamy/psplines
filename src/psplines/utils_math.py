"""
psplines.utils_math – Numerical utilities
========================================

Helper routines for P-spline smoothing:

- `effective_df`: exact or stochastic trace of smoother matrix via
  Hutchinson estimator, with LU factorization for many solves.
- `pointwise_se`: point-wise standard errors ``sqrt(diag(Bq C Bq'))``
  without forming the n x n matrix.
- `credible_band`: mean and equal-tailed band of ``Bq @ alpha`` over
  coefficient draws, evaluated in blocks of query points.

Based on Eilers & Marx (2021), eq. (2.15).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized, spsolve

from .exceptions import ValidationError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["credible_band", "effective_df", "pointwise_se", "solve_penalized"]


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


def _as_dense_2d(M: sp.spmatrix | sp.sparray | ArrayLike, name: str) -> NDArray:
    """Return ``M`` as a dense 2-D array (sparse input is densified)."""
    dense = M.toarray() if sp.issparse(M) else np.asarray(M)  # type: ignore[union-attr]
    if dense.ndim != 2:
        raise ValidationError(f"{name} must be 2-D (got shape {dense.shape})")
    return dense  # type: ignore[no-any-return]


def pointwise_se(
    Bq: sp.spmatrix | sp.sparray | ArrayLike,
    cov: sp.spmatrix | sp.sparray | ArrayLike,
) -> NDArray:
    """
    Point-wise standard errors of ``Bq @ alpha`` for coefficients with
    covariance ``cov``.

    Computes ``sqrt(max(diag(Bq @ cov @ Bq.T), 0))`` without forming the
    ``n_query x n_query`` matrix: ``V = cov @ Bq.T`` is ``n_basis x n_query``
    and the diagonal is the row-wise inner product of ``Bq`` with ``V.T``.
    This is the delta-method SE of Eilers & Marx (2021), eq. (2.16).

    Parameters
    ----------
    Bq : (n_query, n_basis) array or sparse matrix
        Basis (or derivative basis) evaluated at the query points.
    cov : (n_basis, n_basis) array or sparse matrix
        Covariance of the coefficients, e.g. ``phi * inv(B'WB + P)``.

    Returns
    -------
    se : (n_query,) ndarray
        Point-wise standard errors. Tiny negative variances from rounding
        are clipped to zero.
    """
    Bq_dense = _as_dense_2d(Bq, "Bq")
    cov_dense = _as_dense_2d(cov, "cov")
    nb = Bq_dense.shape[1]
    if cov_dense.shape != (nb, nb):
        raise ValidationError(
            f"cov must have shape ({nb}, {nb}) to match Bq (got {cov_dense.shape})"
        )
    V = cov_dense @ Bq_dense.T
    var = np.einsum("ij,ji->i", Bq_dense, V)
    return np.sqrt(np.maximum(var, 0.0))  # type: ignore[no-any-return]


def credible_band(
    Bq: sp.spmatrix | sp.sparray | ArrayLike,
    alpha_draws: ArrayLike,
    ci_prob: float = 0.95,
    chunk_size: int = 2000,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Posterior mean and equal-tailed credible band of ``Bq @ alpha``.

    For each query point ``i`` the posterior draws of the curve are
    ``alpha_draws @ Bq[i]``; this returns their mean and the
    ``(1 - ci_prob) / 2`` and ``(1 + ci_prob) / 2`` percentiles
    (``numpy.percentile``, linear interpolation).

    The draws x query-points matrix is never formed in full: query points
    are processed in blocks of ``chunk_size``, so peak memory is about
    ``n_samples x chunk_size`` doubles. Each query point uses only its own
    column, so the result depends on ``chunk_size`` only through rounding:
    the BLAS product may sum in a different order for a different block
    width (differences of order 1e-13 relative). The same ``chunk_size``
    gives the same numbers. A row of
    ``Bq`` containing NaN (e.g. a point outside the domain of a
    non-extrapolating ``BSpline``) gives NaN at that point only.

    Parameters
    ----------
    Bq : (n_query, n_basis) array or sparse matrix
        Basis (or derivative basis) evaluated at the query points.
    alpha_draws : (n_samples, n_basis) array
        Posterior draws of the spline coefficients, one draw per row,
        e.g. draws from ``N(spline.coef, spline.coef_covariance)``, or draws
        stored from an earlier Bayesian fit.
    ci_prob : float, default 0.95
        Probability mass of the credible band, in (0, 1).
    chunk_size : int, default 2000
        Number of query points evaluated per block.

    Returns
    -------
    mean, lower, upper : (n_query,) ndarrays
        Posterior mean and lower/upper credible limits at each query point.
    """
    if not (0 < ci_prob < 1):
        raise ValidationError(f"ci_prob must be between 0 and 1 (got {ci_prob})")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise ValidationError(f"chunk_size must be an integer (got {chunk_size!r})")
    if chunk_size <= 0:
        raise ValidationError(f"chunk_size must be positive (got {chunk_size})")

    sparse_basis = sp.issparse(Bq)
    Bq_arr: Any = Bq.tocsr() if sparse_basis else np.asarray(Bq)  # type: ignore[union-attr]
    if len(Bq_arr.shape) != 2:
        raise ValidationError(f"Bq must be 2-D (got shape {Bq_arr.shape})")
    draws_arr = np.asarray(alpha_draws)
    if draws_arr.ndim != 2:
        raise ValidationError(
            f"alpha_draws must be 2-D (n_samples, n_basis) (got shape {draws_arr.shape})"
        )
    n_query, nb = Bq_arr.shape
    if draws_arr.shape[1] != nb:
        raise ValidationError(
            f"alpha_draws has {draws_arr.shape[1]} columns but Bq has {nb}"
        )
    if draws_arr.shape[0] == 0:
        raise ValidationError("alpha_draws must contain at least one draw")

    q_lower = (1 - ci_prob) / 2 * 100
    q_upper = (1 + ci_prob) / 2 * 100
    mean = np.empty(n_query)
    lower = np.empty(n_query)
    upper = np.empty(n_query)
    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        block = Bq_arr[start:stop]
        if sparse_basis:
            block = block.toarray()
        draws = draws_arr @ block.T
        mean[start:stop] = draws.mean(axis=0)
        lower[start:stop] = np.percentile(draws, q_lower, axis=0)
        upper[start:stop] = np.percentile(draws, q_upper, axis=0)
    return mean, lower, upper

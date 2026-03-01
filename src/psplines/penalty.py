"""
psplines.penalty – Sparse finite-difference penalty matrices
============================================================

Generate d-th order finite-difference operators in sparse form, plus
specialised penalty matrices for shape constraints and adaptive smoothing.

The basic matrix D (shape (n-d)×n) satisfies:
  (D @ α)[i] = ∑_{k=0}^d (-1)^{d-k} C(d,k) α[i+k]

Additional penalty builders:
  • asymmetric_penalty_matrix  – shape constraints (§8.7)
  • variable_penalty_matrix    – exponential penalty weights (§8.8)
  • adaptive_penalty_matrix    – per-segment penalty weights (§8.8)

References: Eilers & Marx (2021), Sections 2.3, 8.7, 8.8 & Appendix C.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.special import comb

__all__ = [
    "difference_matrix",
    "asymmetric_penalty_matrix",
    "variable_penalty_matrix",
    "adaptive_penalty_matrix",
]


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


# ---------------------------------------------------------------------------
# Shape-constraint penalty (§8.7, eq. 8.14)
# ---------------------------------------------------------------------------

# Recognised constraint types and the (diff_order, sign) they map to.
# sign = +1 ⇒ penalise positive differences, sign = -1 ⇒ penalise negative.
_SHAPE_TYPES: dict[str, tuple[int, int]] = {
    "increasing": (1, -1),  # penalise Δα_j < 0
    "decreasing": (1, +1),  # penalise Δα_j > 0
    "convex": (2, -1),  # penalise Δ²α_j < 0
    "concave": (2, +1),  # penalise Δ²α_j > 0
    "nonneg": (0, -1),  # penalise α_j < 0
}

VALID_SHAPE_TYPES = frozenset(_SHAPE_TYPES)


def asymmetric_penalty_matrix(
    alpha: np.ndarray,
    constraint_type: str,
    *,
    mask: np.ndarray | None = None,
) -> csr_matrix:
    """
    Build the asymmetric penalty D'VD for a shape constraint.

    Given the current coefficient vector *alpha*, this constructs:
        P_shape = D_d' V D_d
    where ``D_d`` is the d-th order difference matrix and
    ``V = diag(v)`` with v_j = 1 when the constraint is *violated*
    and 0 otherwise.  Multiplied by a large κ and added to the
    normal equations, this drives the solution toward the constraint.

    Parameters
    ----------
    alpha : 1-D array, length n
        Current coefficient vector.
    constraint_type : str
        One of ``"increasing"``, ``"decreasing"``, ``"convex"``,
        ``"concave"``, or ``"nonneg"``.
    mask : bool array, optional
        Length ``n - d`` (where *d* is the diff-order implied by
        *constraint_type*).  When supplied, the penalty is only
        active where ``mask[j]`` is True (selective constraint, §8.7).

    Returns
    -------
    P_shape : csr_matrix, shape (n, n)
        Symmetric positive semi-definite penalty matrix.
        Must still be scaled by κ before adding to the system.
    """
    if constraint_type not in _SHAPE_TYPES:
        raise ValueError(
            f"Unknown constraint_type {constraint_type!r}; "
            f"choose from {sorted(_SHAPE_TYPES)}"
        )
    diff_order, sign = _SHAPE_TYPES[constraint_type]
    n = alpha.shape[0]
    D = difference_matrix(n, diff_order)
    diffs = D @ alpha  # shape (n - diff_order,)

    # v_j = 1 when the constraint is violated
    v = (diffs < 0).astype(float) if sign == -1 else (diffs > 0).astype(float)

    # selective constraint via mask
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != v.shape[0]:
            raise ValueError(
                f"mask length ({mask.shape[0]}) must match "
                f"n - diff_order ({v.shape[0]})"
            )
        v = v * mask

    V = diags(v)
    return csr_matrix((D.T @ V @ D).tocsr())


# ---------------------------------------------------------------------------
# Variable (exponential) penalty weights (§8.8)
# ---------------------------------------------------------------------------


def variable_penalty_matrix(
    n: int,
    order: int = 2,
    gamma: float = 0.0,
) -> csr_matrix:
    """
    Penalty matrix with exponentially varying weights.

    Builds ``D' V D`` where ``V = diag(exp(γ j / m))``, ``m = n - order``,
    and ``j = 0, …, m-1``.  When ``γ = 0`` this reduces to the standard
    penalty ``D'D``.

    Parameters
    ----------
    n : int
        Length of the coefficient vector α.
    order : int
        Order of the finite-difference penalty (default 2).
    gamma : float
        Exponential rate.  Positive → heavier penalty toward the right
        boundary; negative → heavier toward the left.

    Returns
    -------
    P : csr_matrix, shape (n, n)
        Weighted penalty matrix (does **not** include λ scaling).
    """
    D = difference_matrix(n, order)
    m = D.shape[0]
    if m == 0:
        return csr_matrix((n, n))
    j = np.arange(m, dtype=float)
    weights = np.exp(gamma * j / m)
    V = diags(weights)
    return csr_matrix((D.T @ V @ D).tocsr())


# ---------------------------------------------------------------------------
# Adaptive (per-segment) penalty weights (§8.8)
# ---------------------------------------------------------------------------


def adaptive_penalty_matrix(
    n: int,
    order: int = 2,
    weights: np.ndarray | None = None,
) -> csr_matrix:
    """
    Penalty matrix with arbitrary per-difference weights.

    Builds ``D' V D`` where ``V = diag(weights)``.  This is the building
    block for adaptive smoothing: the caller estimates *weights* (one per
    row of D) and passes them here.

    Parameters
    ----------
    n : int
        Length of the coefficient vector α.
    order : int
        Order of the finite-difference penalty (default 2).
    weights : 1-D array of length ``n - order``, optional
        Non-negative per-difference penalty weights.
        If *None*, all weights are 1 (standard penalty).

    Returns
    -------
    P : csr_matrix, shape (n, n)
        Weighted penalty matrix (does **not** include a global λ factor).
    """
    D = difference_matrix(n, order)
    m = D.shape[0]
    if m == 0:
        return csr_matrix((n, n))
    if weights is None:
        return csr_matrix((D.T @ D).tocsr())
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != m:
        raise ValueError(
            f"weights length ({weights.shape[0]}) must equal n - order ({m})"
        )
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    V = diags(weights)
    return csr_matrix((D.T @ V @ D).tocsr())

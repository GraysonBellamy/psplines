"""
Custom exception classes for PSplines package.
"""

from __future__ import annotations


class PSplineError(Exception):
    """Base exception for PSpline operations."""


class FittingError(PSplineError):
    """Raised when model fitting fails."""


class PredictionError(PSplineError):
    """Raised when prediction fails."""


class ValidationError(PSplineError):
    """Raised when input validation fails."""


class OptimizationError(PSplineError):
    """Raised when parameter optimization fails."""


class ConvergenceError(PSplineError):
    """Raised when IRLS iteration fails to converge."""

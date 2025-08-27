"""
Custom exception classes for PSplines package.
"""


class PSplineError(Exception):
    """Base exception for PSpline operations."""
    pass


class FittingError(PSplineError):
    """Raised when model fitting fails."""
    pass


class PredictionError(PSplineError):
    """Raised when prediction fails."""
    pass


class ValidationError(PSplineError):
    """Raised when input validation fails."""
    pass


class OptimizationError(PSplineError):
    """Raised when parameter optimization fails."""
    pass

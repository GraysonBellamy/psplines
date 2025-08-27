__version__ = "0.1.3"

from .basis import b_spline_basis
from .core import PSpline
from .exceptions import (
    FittingError,
    OptimizationError,
    PredictionError,
    PSplineError,
    ValidationError,
)
from .optimize import cross_validation
from .penalty import difference_matrix
from .utils import plot_fit

__all__ = [
    "PSpline",
    "b_spline_basis",
    "difference_matrix",
    "cross_validation",
    "plot_fit",
    # Exceptions
    "PSplineError",
    "FittingError",
    "PredictionError",
    "ValidationError",
    "OptimizationError",
]

__version__ = "0.1.3"

from .basis import b_spline_basis
from .core import PSpline
from .density import DensityResult, density_estimate
from .exceptions import (
    ConvergenceError,
    FittingError,
    OptimizationError,
    PredictionError,
    PSplineError,
    ValidationError,
)
from .glm import BinomialFamily, Family, GaussianFamily, PoissonFamily, get_family
from .optimize import cross_validation, variable_penalty_cv
from .penalty import (
    adaptive_penalty_matrix,
    asymmetric_penalty_matrix,
    difference_matrix,
    variable_penalty_matrix,
)
from .utils import plot_fit

__all__ = [
    "PSpline",
    "b_spline_basis",
    "difference_matrix",
    "asymmetric_penalty_matrix",
    "variable_penalty_matrix",
    "adaptive_penalty_matrix",
    "cross_validation",
    "variable_penalty_cv",
    "plot_fit",
    # GLM families
    "Family",
    "GaussianFamily",
    "PoissonFamily",
    "BinomialFamily",
    "get_family",
    # Density estimation
    "density_estimate",
    "DensityResult",
    # Exceptions
    "PSplineError",
    "FittingError",
    "PredictionError",
    "ValidationError",
    "OptimizationError",
    "ConvergenceError",
]

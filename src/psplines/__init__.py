__version__ = "0.2.0"

from .basis import b_spline_basis
from .core import DerivConstraint, PSpline, ShapeConstraint, SlopeZeroConstraint
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
from .optimize import (
    aic,
    cross_validation,
    l_curve,
    plot_diagnostics,
    v_curve,
    variable_penalty_cv,
)
from .penalty import (
    adaptive_penalty_matrix,
    asymmetric_penalty_matrix,
    difference_matrix,
    variable_penalty_matrix,
)
from .utils import plot_derivatives, plot_fit

__all__ = [
    # Core
    "PSpline",
    "ShapeConstraint",
    "DerivConstraint",
    "SlopeZeroConstraint",
    # Basis
    "b_spline_basis",
    # Penalty
    "difference_matrix",
    "asymmetric_penalty_matrix",
    "variable_penalty_matrix",
    "adaptive_penalty_matrix",
    # Optimization
    "cross_validation",
    "aic",
    "l_curve",
    "v_curve",
    "variable_penalty_cv",
    "plot_diagnostics",
    # Plotting
    "plot_fit",
    "plot_derivatives",
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

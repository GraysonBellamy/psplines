__version__ = "0.1.3"

from .basis import b_spline_basis
from .penalty import difference_matrix
from .core import PSpline
from .optimize import cross_validation
from .utils import plot_fit

__all__ = [
    "PSpline",
    "b_spline_basis", 
    "difference_matrix",
    "cross_validation",
    "plot_fit",
]

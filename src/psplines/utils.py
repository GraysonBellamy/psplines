from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import PSpline

import matplotlib.pyplot as plt
import numpy as np

from .exceptions import PSplineError

__all__ = ["plot_fit", "plot_derivatives"]


def plot_fit(
    pspline: PSpline, title: str = "P-spline Fit", subsample: int = 1000
) -> None:
    """
    Plot data and P-spline fit, subsampling for large datasets (inspired by Figure 2.9, Page 29).

    Parameters
    ----------
    pspline : PSpline
        Fitted PSpline object.
    title : str
        Plot title.
    subsample : int
        Number of points to plot (default: 1000).
    """
    n = len(pspline.x)  # type: ignore[arg-type]
    if n > subsample:
        # Deterministic stride-based subsampling
        idx = np.linspace(0, n - 1, subsample, dtype=int)
        x_plot = np.asarray(pspline.x)[idx]
        y_plot = np.asarray(pspline.y)[idx]
        fit_plot = np.asarray(pspline.fitted_values)[idx]
    else:
        x_plot = np.asarray(pspline.x)
        y_plot = np.asarray(pspline.y)
        fit_plot = np.asarray(pspline.fitted_values)

    plt.scatter(x_plot, y_plot, c="grey", s=1, label="Data")
    plt.plot(x_plot, fit_plot, c="blue", lw=2, label="Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_derivatives(
    pspline: PSpline,
    deriv_orders: list[int] | None = None,
    x_new: np.ndarray | None = None,
    title: str = "P-spline Derivatives",
    subsample: int = 1000,
) -> None:
    """
    Plot derivatives of the P-spline smoothed curve (Section 2.5, Page 20).

    Parameters
    ----------
    pspline : PSpline
        Fitted PSpline object.
    deriv_orders : list of int, optional
        List of derivative orders to plot (default: [1]).
    x_new : ndarray, optional
        Array of x values to evaluate derivatives (default: original x).
    title : str
        Plot title.
    subsample : int
        Number of points to plot (default: 1000).
    """
    if deriv_orders is None:
        deriv_orders = [1]

    if pspline.B is None or pspline.coef is None:
        raise ValueError("Model not fitted. Call fit() first.")

    x_eval: np.ndarray = np.asarray(pspline.x) if x_new is None else np.array(x_new)
    n = len(x_eval)

    # Deterministic stride-based subsampling
    if n > subsample:
        idx = np.linspace(0, n - 1, subsample, dtype=int)
        x_plot = x_eval[idx]
    else:
        x_plot = x_eval

    plt.figure()
    colors = ["red", "green", "purple", "orange"]

    for i, order in enumerate(deriv_orders):
        if order < 0:
            warnings.warn(f"Skipping invalid derivative order {order}", stacklevel=2)
            continue
        try:
            deriv = pspline.derivative(x_new=x_plot, deriv_order=order)
            deriv_arr = np.asarray(deriv)
            label = f"Order {order} Derivative"
            plt.plot(x_plot, deriv_arr, c=colors[i % len(colors)], lw=2, label=label)
            plt.scatter(
                [x_plot[0], x_plot[-1]],
                [deriv_arr[0], deriv_arr[-1]],
                c=colors[i % len(colors)],
                marker="o",
                s=100,
                label=f"{label} (Boundaries)",
            )
            plt.text(
                float(x_plot[0]),
                float(deriv_arr[0]),
                f"{deriv_arr[0]:.2e}",
                fontsize=8,
                verticalalignment="bottom",
            )
            plt.text(
                float(x_plot[-1]),
                float(deriv_arr[-1]),
                f"{deriv_arr[-1]:.2e}",
                fontsize=8,
                verticalalignment="bottom",
            )
        except (PSplineError, ValueError) as e:
            warnings.warn(
                f"Could not compute derivative of order {order}: {e}",
                stacklevel=2,
            )

    plt.xlabel("x")
    plt.ylabel("Derivative")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

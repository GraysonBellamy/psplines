import matplotlib.pyplot as plt
import numpy as np


def plot_fit(pspline, title="P-spline Fit", subsample=1000):
    """
    Plot data and P-spline fit, subsampling for large datasets (inspired by Figure 2.9, Page 29).

    Parameters:
    - pspline: PSpline object
    - title: Plot title
    - subsample: Number of points to plot (default: 1000)
    """
    n = len(pspline.x)
    if n > subsample:
        idx = np.random.choice(n, subsample, replace=False)
        idx = np.sort(idx)
        x_plot = pspline.x[idx]
        y_plot = pspline.y[idx]
        fit_plot = pspline.fitted_values[idx]
    else:
        idx = np.arange(n)
        x_plot = pspline.x
        y_plot = pspline.y
        fit_plot = pspline.fitted_values

    plt.scatter(x_plot, y_plot, c="grey", s=1, label="Data")
    plt.plot(x_plot, fit_plot, c="blue", lw=2, label="Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_derivatives(
    pspline, deriv_orders=[1], x_new=None, title="P-spline Derivatives", subsample=1000
):
    """
    Plot derivatives of the P-spline smoothed curve (Section 2.5, Page 20).

    Parameters:
    - pspline: PSpline object
    - deriv_orders: List of derivative orders to plot (default: [1])
    - x_new: Array of x values to evaluate derivatives (default: original x)
    - title: Plot title
    - subsample: Number of points to plot (default: 1000)

    Returns:
    - None (displays plot)
    """
    if pspline.B is None or pspline.coef is None:
        raise ValueError("Model not fitted. Call fit() first.")

    x_eval = pspline.x if x_new is None else np.array(x_new)
    n = len(x_eval)

    # Ensure boundaries are included
    if n > subsample:
        boundary_indices = [0, n - 1]
        subsample_inner = max(0, subsample - len(boundary_indices))
        inner_indices = np.random.choice(n - 2, subsample_inner, replace=False) + 1
        idx = np.concatenate([boundary_indices, inner_indices])
        idx = np.sort(idx)
        x_plot = x_eval[idx]
    else:
        idx = np.arange(n)
        x_plot = x_eval

    plt.figure()
    colors = ["red", "green", "purple", "orange"]

    for i, order in enumerate(deriv_orders):
        if order < 0:
            print(f"Warning: Skipping invalid derivative order {order}")
            continue
        try:
            deriv = pspline.derivative(x_new=x_plot, deriv_order=order)
            label = f"Order {order} Derivative"
            plt.plot(x_plot, deriv, c=colors[i % len(colors)], lw=2, label=label)
            plt.scatter(
                [x_plot[0], x_plot[-1]],
                [deriv[0], deriv[-1]],
                c=colors[i % len(colors)],
                marker="o",
                s=100,
                label=f"{label} (Boundaries)",
            )
            plt.text(
                x_plot[0],
                deriv[0],
                f"{deriv[0]:.2e}",
                fontsize=8,
                verticalalignment="bottom",
            )
            plt.text(
                x_plot[-1],
                deriv[-1],
                f"{deriv[-1]:.2e}",
                fontsize=8,
                verticalalignment="bottom",
            )
        except Exception as e:
            print(f"Warning: Could not compute derivative of order {order}: {e}")

    plt.xlabel("x")
    plt.ylabel("Derivative")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

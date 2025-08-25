# Utilities API

The utils module provides utility functions for plotting and visualization of P-spline results.

## Functions

::: psplines.utils.plot_fit
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Basic Plotting

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline
from psplines.utils import plot_fit

# Create and fit spline
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x) + 0.1 * np.random.randn(50)
spline = PSpline(x, y, nseg=15, lambda_=1.0)
spline.fit()

# Create basic plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_fit(spline, ax=ax)
plt.show()
```

### Customized Plotting

```python
# Plot with uncertainty and custom styling
x_new = np.linspace(0, 2*np.pi, 200)
fig, ax = plt.subplots(figsize=(12, 8))

plot_fit(spline, x_new=x_new, show_se=True, ax=ax,
         data_kws={'alpha': 0.6, 'color': 'darkblue'},
         fit_kws={'color': 'red', 'linewidth': 2},
         se_kws={'alpha': 0.2, 'color': 'red'})

ax.set_title('P-Spline Fit with Uncertainty')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

### Multiple Subplots

```python
# Create comprehensive diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Main fit plot
plot_fit(spline, ax=axes[0, 0], show_se=True)
axes[0, 0].set_title('Data and Fit')

# Residuals
residuals = spline.y - spline.fitted_values
axes[0, 1].scatter(spline.fitted_values, residuals, alpha=0.6)
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_title('Residuals vs Fitted')
axes[0, 1].set_xlabel('Fitted Values')
axes[0, 1].set_ylabel('Residuals')

# First derivative
dy_dx = spline.derivative(x_new, deriv_order=1)
axes[1, 0].plot(x_new, dy_dx, 'g-', linewidth=2)
axes[1, 0].set_title('First Derivative')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('dy/dx')

# Second derivative
d2y_dx2 = spline.derivative(x_new, deriv_order=2)
axes[1, 1].plot(x_new, d2y_dx2, 'm-', linewidth=2)
axes[1, 1].set_title('Second Derivative')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('d²y/dx²')

plt.tight_layout()
plt.show()
```

## Mathematical Visualization Utilities

While not part of the core API, here are some useful functions for understanding P-splines:

### Basis Function Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines.basis import b_spline_basis

def plot_basis_functions(xl=0, xr=1, nseg=5, degree=3):
    """Plot individual B-spline basis functions."""
    x = np.linspace(xl, xr, 200)
    B, knots = b_spline_basis(x, xl, xr, nseg, degree)
    
    plt.figure(figsize=(12, 6))
    for i in range(B.shape[1]):
        plt.plot(x, B[:, i].toarray().flatten(), 
                 alpha=0.7, label=f'B_{i}')
    
    # Mark knots
    interior_knots = knots[degree:-degree]
    plt.vlines(interior_knots, 0, 1, colors='red', 
               linestyles='dashed', alpha=0.5)
    
    plt.title(f'B-spline Basis Functions (degree={degree}, nseg={nseg})')
    plt.xlabel('x')
    plt.ylabel('Basis Function Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Example usage
plot_basis_functions(nseg=5, degree=3)
```

### Penalty Matrix Visualization

```python
import matplotlib.pyplot as plt
from psplines.penalty import difference_matrix

def plot_penalty_matrix(n=10, order=2):
    """Visualize the penalty matrix structure."""
    D = difference_matrix(n, order)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Penalty matrix D
    axes[0].spy(D, markersize=8)
    axes[0].set_title(f'Difference Matrix D_{order} ({D.shape[0]}×{D.shape[1]})')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    
    # Penalty matrix D^T D
    DtD = D.T @ D
    im = axes[1].imshow(DtD.toarray(), cmap='Blues')
    axes[1].set_title(f'Penalty Matrix D_{order}^T D_{order} ({DtD.shape[0]}×{DtD.shape[1]})')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_penalty_matrix(n=10, order=2)
```

### Smoothing Parameter Effect Visualization

```python
def plot_lambda_effect(spline, lambdas=None):
    """Show effect of different smoothing parameters."""
    if lambdas is None:
        lambdas = np.logspace(-2, 2, 5)
    
    x_plot = np.linspace(spline.x.min(), spline.x.max(), 200)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))
    
    for i, lam in enumerate(lambdas):
        # Create temporary spline with this lambda
        temp_spline = PSpline(spline.x, spline.y, 
                             nseg=spline.nseg, lambda_=lam)
        temp_spline.fit()
        y_fit = temp_spline.predict(x_plot)
        
        plt.plot(x_plot, y_fit, color=colors[i], 
                 linewidth=2, label=f'λ = {lam:.3f}')
    
    plt.scatter(spline.x, spline.y, alpha=0.6, 
                color='black', s=30, label='Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Effect of Smoothing Parameter λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage (assuming you have a fitted spline)
# plot_lambda_effect(spline)
```

## Plotting Best Practices

### Figure Styling

```python
# Use consistent styling across plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
```

### Color Schemes

- **Data points**: Use muted colors with transparency
- **Fit line**: Use bright, contrasting colors
- **Uncertainty bands**: Use same color as fit with low alpha
- **Derivatives**: Use distinct colors (green, magenta, etc.)

### Layout Recommendations

- **Single plot**: 10×6 inches for papers, 12×8 for presentations
- **Subplot grids**: 15×10 inches for 2×2 grids
- **Always** include axis labels and titles
- Use `tight_layout()` to avoid overlapping elements
- Save as PNG (300 DPI) for publications, PDF for vector graphics
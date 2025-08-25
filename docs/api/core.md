# Core API

The core module contains the main `PSpline` class that implements penalized B-spline smoothing.

## PSpline Class

::: psplines.core.PSpline
    options:
      members:
        - __init__
        - fit
        - predict
        - derivative
        - bayes_fit
        - plot_lam_trace
        - plot_alpha_trace
        - plot_posterior
      show_source: true
      show_bases: false
      heading_level: 3

## Usage Examples

### Basic Usage

```python
import numpy as np
from psplines import PSpline

# Generate data
x = np.linspace(0, 1, 50)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(50)

# Create and fit spline
spline = PSpline(x, y, nseg=20, lambda_=1.0)
spline.fit()

# Make predictions
x_new = np.linspace(0, 1, 100)
y_pred = spline.predict(x_new)
```

### With Uncertainty Quantification

```python
# Analytical standard errors
y_pred, se = spline.predict(x_new, return_se=True)

# Bootstrap confidence intervals
y_pred, se_boot = spline.predict(x_new, return_se=True, 
                                se_method="bootstrap", B_boot=1000)
```

### Derivative Computation

```python
# First derivative
dy_dx = spline.derivative(x_new, deriv_order=1)

# Second derivative with uncertainty
d2y_dx2, se = spline.derivative(x_new, deriv_order=2, return_se=True)
```

### Bayesian Inference

```python
# Fit Bayesian model
trace = spline.bayes_fit(draws=2000, tune=1000)

# Get posterior credible intervals
mean, lower, upper = spline.predict(x_new, se_method="bayes")
```

## Model Parameters

### Basis Function Parameters

- **`nseg`**: Number of B-spline segments
    - Controls the flexibility of the basis
    - More segments = more flexible fit
    - Typical values: 10-50

- **`degree`**: Degree of B-spline basis functions
    - Controls local smoothness
    - Higher degree = smoother derivatives
    - Typical values: 1-4 (3 is most common)

### Penalty Parameters

- **`lambda_`**: Smoothing parameter
    - Controls the trade-off between fit and smoothness  
    - Higher values = smoother fits
    - Can be selected automatically

- **`penalty_order`**: Order of the difference penalty
    - 1: Penalizes first differences (rough penalty on slopes)
    - 2: Penalizes second differences (rough penalty on curvature)
    - 3: Penalizes third differences (rough penalty on jerk)

### Constraint Parameters

- **`constraints`**: Dictionary specifying boundary constraints
    - `"deriv"`: Derivative constraints at boundaries
    - Example: `{"deriv": {"order": 1, "initial": 0, "final": 0}}`

## Model Attributes

After fitting, the following attributes are available:

- **`coef`**: B-spline coefficients
- **`fitted_values`**: Fitted values at input points
- **`knots`**: Knot vector used for B-splines
- **`ED`**: Effective degrees of freedom
- **`sigma2`**: Residual variance estimate
- **`se_coef`**: Standard errors of coefficients
- **`se_fitted`**: Standard errors of fitted values

## Error Handling

The `PSpline` class includes comprehensive input validation:

- **Array validation**: Checks for proper dimensions, finite values
- **Parameter validation**: Ensures valid ranges and types
- **State validation**: Verifies model is fitted before prediction
- **Constraint validation**: Validates constraint specifications

All errors provide descriptive messages with suggested solutions.
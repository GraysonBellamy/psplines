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
from psplines import PSpline, ShapeConstraint, SlopeZeroConstraint

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
                                se_method="bootstrap", n_boot=1000)
```

### Derivative Computation

```python
# First derivative
dy_dx = spline.derivative(x_new, deriv_order=1)

# Second derivative with uncertainty
d2y_dx2, se = spline.derivative(x_new, deriv_order=2, return_se=True)
```

### With Observation Weights

```python
# Heteroscedastic weights (trust right side of data more)
weights = 1.0 + 5.0 * x
spline = PSpline(x, y, weights=weights)
spline.fit()

# Missing data via zero weights
weights = np.ones_like(x)
weights[20:30] = 0.0  # mark observations 20-29 as missing
spline = PSpline(x, y, weights=weights)
spline.fit()  # penalty interpolates smoothly through the gap
```

### Poisson P-Spline (Count Data)

```python
# Smooth count data with log link
years = np.arange(1851, 1963, dtype=float)
counts = ...  # annual event counts

spline = PSpline(years, counts, nseg=20, lambda_=100, family="poisson")
spline.fit()

# Predictions are positive (response scale)
x_new = np.linspace(1851, 1962, 200)
mu_pred = spline.predict(x_new)  # scale="response" is default

# Confidence intervals on response scale (asymmetric, always positive)
mu_hat, lower, upper = spline.predict(x_new, return_se=True)

# Linear predictor (log scale)
eta = spline.predict(x_new, scale="link")
```

### Shape-Constrained Smoothing

```python
# Monotone increasing fit
x = np.linspace(0, 5, 60)
y = np.log(x + 1) + 0.2 * np.random.randn(60)

spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 shape=[ShapeConstraint(type="increasing")])
spline.fit()

# All first differences of the fitted coefficients will be ≥ 0
y_pred = spline.predict(np.linspace(0, 5, 200))
```

```python
# Concave fit
spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 shape=[ShapeConstraint(type="concave")])
spline.fit()
```

```python
# Multiple constraints: increasing AND concave
spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 shape=[ShapeConstraint(type="increasing"), ShapeConstraint(type="concave")])
spline.fit()
```

```python
# Selective domain constraint (monotone only for x ≤ 3)
spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 shape=[ShapeConstraint(type="increasing",
                         domain=(float(x.min()), 3.0))])
spline.fit()
```

### Variable Penalty (Exponential Weights)

```python
# Heavier smoothing toward the right boundary (γ > 0)
spline = PSpline(x, y, nseg=20, lambda_=1.0, penalty_gamma=5.0)
spline.fit()
```

### Adaptive Penalty (Spatially Varying Smoothness)

```python
# Nonparametric adaptive smoothing — the penalty weights are
# estimated from the data via a secondary B-spline basis.
spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 adaptive=True, adaptive_nseg=10,
                 adaptive_lambda=100.0)
spline.fit()

# The estimated per-difference weights are stored in:
print(spline._adaptive_weights)
```

### Poisson with Exposure Offsets

```python
# Rate modeling: mu = exp(B*alpha) * exposure
exposure = ...  # population at risk
spline = PSpline(x, counts, family="poisson", offset=np.log(exposure))
spline.fit()
```

### Binomial P-Spline (Binary/Proportion Data)

```python
# Bernoulli response (y in {0, 1})
spline = PSpline(age, presence, nseg=15, lambda_=10, family="binomial")
spline.fit()

# Fitted probabilities are in [0, 1]
pi_hat = spline.predict(age_new)

# Grouped binomial (y successes out of t trials)
spline = PSpline(x, successes, family="binomial", trials=trials_vec)
spline.fit()
```

### Density Estimation

```python
from psplines import density_estimate

# Smooth density from raw data (AIC-optimal lambda)
result = density_estimate(raw_data, bins=100, penalty_order=3)

# result.density integrates to ~1
# result.grid contains evaluation points
# result.pspline is the underlying fitted PSpline
```

### Bayesian Inference

```python
# Standard Bayesian P-spline (single λ, §3.5 of Eilers & Marx 2021)
trace = spline.bayes_fit(draws=2000, tune=1000)

# Adaptive Bayesian P-spline (per-difference λ_j, §8.8)
trace = spline.bayes_fit(draws=2000, tune=1000, adaptive=True)

# Get posterior credible intervals (works with either mode)
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

### Observation Weights

- **`weights`**: Optional array of non-negative observation weights
    - Same length as `x` and `y`
    - Replaces the normal equations with weighted form: $(B'WB + \lambda D'D)\alpha = B'Wy$
    - Higher weights give more influence to those observations
    - Zero weights effectively mark observations as missing — the penalty interpolates through gaps
    - `None` (default) treats all observations equally (equivalent to `weights=1`)

### Constraint Parameters

- **`constraints`**: Dictionary specifying boundary constraints
    - `"deriv"`: Derivative constraints at boundaries
    - Example: `{"deriv": {"order": 1, "initial": 0, "final": 0}}`

- **`slope_zero`**: Enforce flat slope in a subdomain
    - Example: `slope_zero=SlopeZeroConstraint(domain=(2.0, 4.0))`

### Shape Constraint Parameters

- **`shape`**: List of `ShapeConstraint` specifications (§8.7)
    - Each entry is a `ShapeConstraint` with `type` (required) and optional `domain`
    - Supported types: `"increasing"`, `"decreasing"`, `"convex"`, `"concave"`, `"nonneg"`
    - Optional `domain=(lo, hi)` restricts the constraint to that $x$-range
    - Multiple constraints can be combined (e.g. increasing + concave)
    - Example: `[ShapeConstraint(type="increasing"), ShapeConstraint(type="concave", domain=(0, 5))]`

- **`shape_kappa`**: Large penalty weight for shape violations (default $10^8$)
    - Higher values enforce the constraint more strictly
    - Too large may cause numerical issues

- **`max_shape_iter`**: Maximum iterations for the asymmetric-penalty loop (default 50)

### Adaptive / Variable Penalty Parameters

- **`adaptive`**: Enable nonparametric adaptive penalty (§8.8, default `False`)
    - Estimates spatially varying weights via a secondary B-spline basis
    - Alternates between weight estimation and coefficient estimation

- **`adaptive_nseg`**: Number of segments for the secondary weight basis (default 10)

- **`adaptive_lambda`**: Smoothing parameter for the weight basis (default 100.0)

- **`adaptive_max_iter`**: Maximum outer iterations for adaptive loop (default 20)

- **`penalty_gamma`**: Exponential variable penalty rate (§8.8, default `None`)
    - When set, uses weights $v_j = \exp(\gamma j / m)$
    - Positive $\gamma$ → heavier penalty toward the right boundary
    - Negative $\gamma$ → heavier penalty toward the left boundary
    - `None` (default) uses the standard uniform penalty

## Model Attributes

After fitting, the following attributes are available:

- **`coef`**: B-spline coefficients
- **`fitted_values`**: Fitted values at input points
- **`knots`**: Knot vector used for B-splines
- **`ED`**: Effective degrees of freedom
- **`phi_`**: Residual variance estimate
- **`se_coef`**: Standard errors of coefficients
- **`se_fitted`**: Standard errors of fitted values

## Error Handling

The `PSpline` class includes comprehensive input validation:

- **Array validation**: Checks for proper dimensions, finite values
- **Parameter validation**: Ensures valid ranges and types
- **State validation**: Verifies model is fitted before prediction
- **Constraint validation**: Validates constraint specifications

All errors provide descriptive messages with suggested solutions.
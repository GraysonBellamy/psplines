# Core Concepts

This guide explains the fundamental concepts behind P-splines, helping you understand how they work and when to use them.

## What are P-Splines?

**P-splines** (Penalized B-splines) are a modern approach to smoothing that combines the flexibility of B-spline basis functions with the automatic smoothing capabilities of penalized regression.

### The Big Picture

Traditional smoothing faces a dilemma:
- **Interpolation**: Fits data exactly but follows noise
- **Heavy smoothing**: Reduces noise but may miss important features

P-splines solve this by:
1. Using a flexible B-spline basis
2. Adding a penalty for "roughness"
3. Automatically balancing fit vs. smoothness

## Mathematical Foundation

### The P-Spline Model

Given data points $(x_i, y_i)$, P-splines fit a smooth function $f(x)$ by solving:

$$\min_\alpha (y - B\alpha)'W(y - B\alpha) + \lambda \|D_p \alpha\|^2$$

Where:
- $B$ is the B-spline basis matrix
- $\alpha$ are the B-spline coefficients
- $D_p$ is the $p$-th order difference matrix
- $\lambda$ is the smoothing parameter
- $W = \text{diag}(w)$ is an optional diagonal weight matrix (identity when no weights are specified)

### Key Components

#### 1. B-Spline Basis Functions

B-splines are piecewise polynomials that provide:

**Local Support**: Each basis function is non-zero only over a small interval
```python
# Example: basis functions have limited influence
basis = BSplineBasis(degree=3, n_segments=10, domain=(0, 1))
# Each function affects only ~4 segments
```

**Computational Efficiency**: Lead to sparse matrices for fast computation

**Smoothness**: Degree $d$ B-splines are $C^{d-1}$ smooth

#### 2. Difference Penalties

Instead of penalizing derivatives directly, P-splines penalize differences of coefficients:

**First-order differences** ($p=1$): 
$$\Delta^1 \alpha_j = \alpha_{j+1} - \alpha_j$$
Penalizes large changes between adjacent coefficients.

**Second-order differences** ($p=2$): 
$$\Delta^2 \alpha_j = \alpha_{j+2} - 2\alpha_{j+1} + \alpha_j$$
Penalizes changes in the slope (curvature).

**Higher orders**: Control higher-order smoothness properties.

#### 3. The Smoothing Parameter λ

Controls the bias-variance trade-off:

- **λ = 0**: Interpolation (high variance, low bias)  
- **λ → ∞**: Maximum smoothness (low variance, high bias)
- **Optimal λ**: Minimizes expected prediction error

#### 4. Observation Weights

Observation weights $w_i$ control how much each data point influences the fit. The weighted objective function is:

$$Q = (y - B\alpha)'W(y - B\alpha) + \lambda \|D\alpha\|^2$$

leading to the weighted normal equations $(B'WB + \lambda D'D)\alpha = B'Wy$.

**Key uses:**

- **Heteroscedastic data**: Give higher weight to more precise observations
- **Missing data**: Set $w_i = 0$ to exclude observations — the penalty automatically interpolates through gaps with a polynomial of degree $2d - 1$ in the coefficient indices
- **Importance weighting**: Emphasize certain regions of the data

```python
# Missing data: mark a gap with zero weights
weights = np.ones_like(x)
weights[20:30] = 0.0  # these observations are ignored
spline = PSpline(x, y, weights=weights)
spline.fit()  # smooth interpolation through the gap
```

## Practical Understanding

### How P-Splines Adapt

P-splines automatically adapt to local data characteristics:

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline

# Create data with varying complexity
x = np.linspace(0, 10, 100)
y_smooth = np.sin(x)                    # Smooth region
y_complex = 5 * np.sin(10*x) * (x > 7)  # Complex region  
y = y_smooth + y_complex + 0.1 * np.random.randn(100)

spline = PSpline(x, y, nseg=30)
# P-spline will automatically be smoother in smooth regions
# and more flexible in complex regions
```

### Effective Degrees of Freedom

A key concept is **effective degrees of freedom** (EdF):

```python
spline.fit()
print(f"Effective DoF: {spline.ED}")
print(f"Maximum DoF (interpolation): {len(y)}")
print(f"Minimum DoF (constant): 1")
```

EdF represents the "complexity" of the fitted model:
- **Low EdF**: Simple, smooth fits
- **High EdF**: Complex, flexible fits
- **Optimal EdF**: Best predictive performance

### Residual Analysis

Understanding residuals helps assess fit quality:

```python
# After fitting
y_pred = spline.predict(x)
residuals = y - y_pred

# Good fit characteristics:
# - Residuals randomly scattered around zero
# - No systematic patterns
# - Approximately constant variance

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
```

## Parameter Selection Deep Dive

### The Smoothing Parameter λ

The most critical parameter requiring automatic selection:

#### Cross-Validation Approach
Estimates out-of-sample prediction error:

```python
from psplines.optimize import cross_validation

# k-fold cross-validation
optimal_lambda, cv_score = cross_validation(
    spline, 
    cv_method='kfold', 
    k_folds=5
)
```

**Pros**: Robust, directly optimizes prediction
**Cons**: Computationally expensive

#### AIC Approach  
Balances fit quality and model complexity:

```python
from psplines.optimize import aic_selection

optimal_lambda, aic_score = aic_selection(spline)
```

**Pros**: Fast, good approximation
**Cons**: Assumes Gaussian errors

#### L-Curve Method
Finds the "corner" in the trade-off curve:

```python
from psplines.optimize import l_curve

optimal_lambda, curvature_info = l_curve(spline)
```

**Pros**: Intuitive geometric interpretation
**Cons**: Not always reliable

### Number of Segments (nseg)

Controls the flexibility of the basis:

#### Rules of Thumb
- **Start with**: `nseg = n_data_points / 4`
- **Minimum**: 5-10 (depends on data complexity)
- **Maximum**: Usually no more than `n_data_points / 2`

#### Adaptive Selection
```python
# Try different nseg values
nseg_values = [10, 20, 30, 40]
best_score = float('inf')
best_nseg = None

for nseg in nseg_values:
    spline_test = PSpline(x, y, nseg=nseg)
    _, cv_score = cross_validation(spline_test)
    
    if cv_score < best_score:
        best_score = cv_score
        best_nseg = nseg

print(f"Optimal nseg: {best_nseg}")
```

### Penalty Order

Controls the type of smoothness:

#### Order 1: Penalizes Slope Changes
```python
spline = PSpline(x, y, penalty_order=1)
# Results in piecewise-linear-like smoothness
# Good for: Step functions, trend analysis
```

#### Order 2: Penalizes Curvature Changes (Default)  
```python
spline = PSpline(x, y, penalty_order=2)
# Results in smooth curves
# Good for: Most applications, natural-looking curves
```

#### Order 3+: Penalizes Higher-Order Changes
```python  
spline = PSpline(x, y, penalty_order=3)
# Results in very smooth curves
# Good for: Very smooth phenomena, artistic curves
```

## Uncertainty Quantification

### Standard Errors

P-splines provide uncertainty estimates through:

#### Analytical Standard Errors
Based on the covariance matrix of coefficients:

```python
y_pred, se = spline.predict(x_new, return_se=True, se_method='analytic')

# Create confidence intervals
lower_ci = y_pred - 1.96 * se
upper_ci = y_pred + 1.96 * se
```

**Assumptions**: Gaussian errors, correct model specification

#### Bootstrap Standard Errors
Empirical estimation through resampling:

```python
y_pred, se = spline.predict(x_new, return_se=True,
                           se_method='bootstrap', n_boot=500)
```

**Advantages**: Fewer assumptions, empirical distribution

#### Bayesian Inference
Full posterior distribution (requires PyMC):

```python
# Standard Bayesian P-spline (single λ, §3.5)
trace = spline.bayes_fit(draws=1000, tune=1000)

# Adaptive Bayesian P-spline (per-difference λ_j for spatially varying smoothness)
trace = spline.bayes_fit(draws=1000, tune=1000, adaptive=True)

# Posterior credible intervals (works with either mode)
mean, lower, upper = spline.predict(x_new, return_se=True, se_method='bayes')
```

**Advantages**: Full uncertainty quantification, principled approach. The standard mode uses a single global penalty matching the book's formulation; the adaptive mode allows different smoothness in different regions of the curve.

### Confidence vs. Prediction Intervals

**Confidence Intervals**: Uncertainty in the mean function
```python
# Standard error of the fitted curve
y_pred, se_fit = spline.predict(x_new, return_se=True)
ci_lower = y_pred - 1.96 * se_fit
ci_upper = y_pred + 1.96 * se_fit
```

**Prediction Intervals**: Uncertainty for new observations
```python
# Include both fit uncertainty and noise
se_pred = np.sqrt(se_fit**2 + spline.phi_)  # Add noise variance
pi_lower = y_pred - 1.96 * se_pred  
pi_upper = y_pred + 1.96 * se_pred
```

## GLM P-Splines

Beyond Gaussian responses, P-splines can smooth count and binary data via **Generalized Linear Model** (GLM) families. The fitting uses Iteratively Reweighted Least Squares (IRLS).

### How GLM P-Splines Work

Instead of minimizing squared residuals directly, GLM P-splines iterate:

1. Compute a **working response** $z = \eta + W^{-1}(y - \mu)$
2. Compute **working weights** $W$ from the current mean $\mu$
3. Solve the weighted penalized system $(B'WB + \lambda D'D)\alpha = B'Wz$
4. Update the linear predictor $\eta = B\alpha$ and the mean $\mu = h(\eta)$
5. Repeat until coefficients converge

The link function $h$ maps the linear predictor to the mean response:

- **Poisson** (count data): log link, $\mu = \exp(\eta)$, weights $W = \text{diag}(\mu)$
- **Binomial** (binary/proportion data): logit link, $\mu = t \cdot \text{sigmoid}(\eta)$, weights $W = \text{diag}(\mu(1-\pi))$

### Poisson Example

```python
import numpy as np
from psplines import PSpline

# Smooth event counts over time
years = np.arange(1900, 2000, dtype=float)
counts = np.random.poisson(np.exp(0.5 * np.sin(years / 10)), len(years))

spline = PSpline(years, counts, nseg=20, lambda_=100, family="poisson")
spline.fit()

# Predictions are always positive (response scale)
mu_hat = spline.predict(years)
```

### Binomial Example

```python
# Probability of success as a function of dose
dose = np.linspace(0, 10, 50)
trials = np.full(50, 20)
successes = np.random.binomial(trials, 1 / (1 + np.exp(-(dose - 5))))

spline = PSpline(dose, successes, family="binomial", trials=trials, nseg=15)
spline.fit()

# Fitted probabilities are bounded in [0, 1]
pi_hat = spline.predict(dose)
```

### Uncertainty for GLM Models

Standard errors for GLM P-splines are computed on the **link scale** and transformed to the response scale via the inverse link. This produces asymmetric confidence intervals that respect natural bounds (positive for Poisson, [0, 1] for Binomial):

```python
mu_hat, lower, upper = spline.predict(x_new, return_se=True)
# lower and upper are on the response scale
```

### Density Estimation

A special application of Poisson P-splines: bin raw data into a histogram, fit a Poisson P-spline to the counts, and normalize to a proper density:

```python
from psplines import density_estimate

result = density_estimate(raw_data, bins=100, penalty_order=3)
# result.density integrates to ~1
# penalty_order=3 preserves mean and variance
```

## Advanced Concepts

### Shape Constraints (§8.7)

When domain knowledge requires the fitted curve to be monotone, convex, concave,
or non-negative, you can add **shape constraints** via an asymmetric penalty.

The idea is simple: for each iteration, identify which differences of the
coefficients **violate** the desired shape, and apply a huge penalty ($\kappa$)
only on those violations.  Iterate until all violations vanish:

```python
from psplines import PSpline, ShapeConstraint

# Monotone increasing fit
spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 shape=[ShapeConstraint(type="increasing")])
spline.fit()

# Convex fit
spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 shape=[ShapeConstraint(type="convex")])
spline.fit()

# Multiple constraints and selective domain
spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 shape=[ShapeConstraint(type="increasing"),
                        ShapeConstraint(type="concave", domain=(0.0, 5.0))])
spline.fit()
```

Available shape types: `"increasing"`, `"decreasing"`, `"convex"`, `"concave"`,
`"nonneg"`.  Shape constraints work with Gaussian, Poisson, and Binomial families.

### Adaptive and Variable Penalties (§8.8)

The standard P-spline applies a **uniform** penalty everywhere.  Two extensions
allow **spatially varying** smoothness:

**Variable penalty** — exponential weights $v_j = \exp(\gamma j/m)$ shift penalty
strength smoothly from one boundary to the other:

```python
# Heavier smoothing toward the right (γ > 0)
spline = PSpline(x, y, nseg=20, lambda_=1.0, penalty_gamma=5.0)
spline.fit()
```

**Adaptive penalty** — per-difference weights are estimated from the data via a
secondary B-spline basis.  Regions where the function changes rapidly get lighter
penalty; smoother regions get heavier penalty:

```python
spline = PSpline(x, y, nseg=20, lambda_=1.0,
                 adaptive=True, adaptive_nseg=10)
spline.fit()
```

Use `variable_penalty_cv()` from the optimize module for automatic
$(\lambda, \gamma)$ selection.

### Whittaker Smoother

The **Whittaker smoother** is the special case of P-splines where $B = I$ (the
identity matrix).  Instead of fitting B-spline coefficients, it operates
directly on the data vector by solving:

$$(W + \lambda D^\top D) z = W y$$

This is ideal for fast smoothing of gridded data (signals, spectra, time series)
where you don't need derivatives, GLM families, or prediction at arbitrary new
points.

**Non-uniform spacing** is handled via *divided differences*: when $x$-spacing
varies, the standard difference operator is replaced by $D_x$ which weights each
difference by $1/(x_{i+1} - x_i)$.  This ensures the roughness penalty is in
the natural units of $x$.

```python
from psplines import WhittakerSmoother

# Basic usage
ws = WhittakerSmoother(x, y, lambda_=1e4, penalty_order=2)
ws.fit()
smoothed = ws.fitted_values

# Automatic λ selection via GCV
ws = WhittakerSmoother(x, y)
lam, score = ws.cross_validation()

# Non-uniform data — divided differences are used automatically
x_irregular = np.sort(np.random.uniform(0, 10, 100))
ws = WhittakerSmoother(x_irregular, y, lambda_=100).fit()
```

See the [Whittaker Smoother API](../api/whittaker.md) for full details.

### Sparse Matrix Structure

P-splines exploit sparsity for efficiency:

```python
# B-spline basis matrix is sparse
print(f"Basis matrix shape: {spline.basis_matrix.shape}")
print(f"Basis matrix density: {spline.basis_matrix.nnz / spline.basis_matrix.size:.3f}")

# Penalty matrix is also sparse and banded
P = spline.penalty_matrix
print(f"Penalty matrix bandwidth: ~{2*spline.penalty_order + 1}")
```

### Computational Complexity

Understanding performance characteristics:

- **Matrix assembly**: $O(n \cdot m)$ where $n$ = data points, $m$ = basis functions
- **System solution**: $O(m^3)$ dense, $O(m^{1.5})$ sparse (typical)  
- **Prediction**: $O(k \cdot m)$ where $k$ = prediction points

### Knot Placement

While P-splines typically use uniform knots, understanding knot placement helps:

```python
# Examine knot structure
basis = spline.basis
print(f"Interior knots: {basis.knots[basis.degree:-basis.degree]}")
print(f"Boundary knots: {basis.knots[:basis.degree]} and {basis.knots[-basis.degree:]}")
```

**Uniform knots**: Work well for most applications
**Adaptive knots**: May improve efficiency for irregular data (advanced topic)

## When to Use P-Splines

### Ideal Scenarios

✅ **Noisy measurements** requiring smoothing
✅ **Derivative estimation** from data
✅ **Trend extraction** from time series
✅ **Interpolation** with uncertainty quantification
✅ **Large datasets** (sparse matrix efficiency)
✅ **Automatic smoothing** without manual parameter tuning

### Limitations

❌ **Highly oscillatory functions** (consider wavelets)
❌ **Very sparse data** (< 10 points per feature)  
❌ **Categorical predictors** (use GAMs or other methods)
❌ **Multidimensional smoothing** (use tensor products or alternatives)

### Alternatives Comparison

| Method | Flexibility | Speed | Automation | Uncertainty |
|--------|-------------|-------|------------|-------------|
| **P-splines** | High | Fast | Excellent | Yes |
| **Kernel smoothing** | Medium | Medium | Good | Limited |
| **Gaussian processes** | High | Slow | Good | Excellent |
| **Savitzky-Golay** | Low | Very fast | Poor | No |

## Summary

P-splines provide an optimal balance of:

- **Flexibility**: Through B-spline basis functions
- **Smoothness**: Through difference penalties  
- **Automation**: Through data-driven parameter selection
- **Efficiency**: Through sparse matrix computations
- **Uncertainty**: Through multiple quantification methods

The key insight is that smoothness can be achieved by penalizing differences of basis coefficients rather than derivatives of the function itself, leading to computationally efficient and statistically principled smoothing.

## Next Steps

- **[Quick Start](quick-start.md)**: Get started immediately
- **[Basic Usage Tutorial](../tutorials/basic-usage.md)**: Hands-on examples
- **[Mathematical Background](../theory/mathematical-background.md)**: Detailed theory
- **[Examples Gallery](../examples/gallery.md)**: Real-world applications
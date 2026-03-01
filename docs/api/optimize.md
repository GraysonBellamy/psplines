# Optimization API

The optimize module provides functions for automatic selection of the smoothing parameter λ using various criteria.

## Functions

::: psplines.optimize.cross_validation
    options:
      show_source: true
      heading_level: 3

::: psplines.optimize.aic
    options:
      show_source: true
      heading_level: 3

::: psplines.optimize.l_curve
    options:
      show_source: true
      heading_level: 3

::: psplines.optimize.v_curve
    options:
      show_source: true
      heading_level: 3

::: psplines.optimize.variable_penalty_cv
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Cross-Validation (Recommended)

```python
import numpy as np
from psplines import PSpline
from psplines.optimize import cross_validation

# Create spline
x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)
spline = PSpline(x, y, nseg=20)

# Find optimal lambda using cross-validation
best_lambda, cv_score = cross_validation(spline)
print(f"Optimal λ: {best_lambda:.6f}, CV score: {cv_score:.6f}")

# Use optimal lambda
spline.lambda_ = best_lambda
spline.fit()
```

### AIC-Based Selection

```python
from psplines.optimize import aic

# Find optimal lambda using AIC
best_lambda, aic_score = aic(spline)
print(f"Optimal λ: {best_lambda:.6f}, AIC: {aic_score:.6f}")
```

### GLM Models (Poisson, Binomial)

All optimizers work with GLM P-splines. For non-Gaussian families, scoring uses
deviance instead of RSS, and each candidate λ requires a full IRLS convergence:

```python
# Poisson P-spline with AIC-optimal lambda
counts = np.random.poisson(np.exp(np.sin(x)), 100)
spline_pois = PSpline(x, counts, nseg=20, family="poisson")
best_lambda, aic_score = aic(spline_pois)
spline_pois.lambda_ = best_lambda
spline_pois.fit()

# Binomial P-spline with cross-validation
trials = np.full(100, 20)
successes = np.random.binomial(trials, 1 / (1 + np.exp(-2 * np.sin(x))))
spline_bin = PSpline(x, successes, nseg=20, family="binomial", trials=trials)
best_lambda, cv_score = cross_validation(spline_bin)
spline_bin.lambda_ = best_lambda
spline_bin.fit()
```

### L-Curve Method

```python
from psplines.optimize import l_curve

# Find optimal lambda using L-curve
best_lambda, curvature = l_curve(spline)
print(f"Optimal λ: {best_lambda:.6f}, Curvature: {curvature:.6f}")
```

### Variable Penalty Parameter Selection

When using exponentially varying penalty weights (§8.8), `variable_penalty_cv`
performs a 2-D grid search over $(\lambda, \gamma)$:

```python
from psplines.optimize import variable_penalty_cv

# Fit an initial spline (required for basis info)
spline = PSpline(x, y, nseg=20)
spline.fit()

# 2-D grid search for best (λ, γ) using GCV
best_lambda, best_gamma, best_score, scores = variable_penalty_cv(
    spline,
    gamma_range=(-10, 10),
    lambda_bounds=(1e-4, 1e4),
    num_gamma=41,
    num_lambda=41,
    criterion="gcv",
)
print(f"Optimal λ={best_lambda:.4f}, γ={best_gamma:.2f}")

# Apply the optimal parameters
spline.lambda_ = best_lambda
spline.penalty_gamma = best_gamma
spline.fit()
```

### Comparing Methods

```python
import matplotlib.pyplot as plt

# Compare different methods
methods = {
    'CV': cross_validation,
    'AIC': aic,
    'L-curve': l_curve,
    'V-curve': v_curve
}

results = {}
for name, method in methods.items():
    try:
        lambda_opt, score = method(spline)
        results[name] = lambda_opt
        print(f"{name}: λ = {lambda_opt:.6f}")
    except Exception as e:
        print(f"{name} failed: {e}")

# Plot comparison
if results:
    methods_list = list(results.keys())
    lambdas = list(results.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(methods_list, lambdas)
    plt.yscale('log')
    plt.ylabel('Optimal λ')
    plt.title('Comparison of Parameter Selection Methods')
    plt.show()
```

## Mathematical Background

### Cross-Validation

Generalized Cross-Validation (GCV) minimizes:
$$\text{GCV}(\lambda) = \frac{n \|y - S_\lambda y\|^2}{[n - \text{tr}(S_\lambda)]^2}$$

where $S_\lambda = B(B^TB + \lambda D^TD)^{-1}B^T$ is the smoothing matrix.

**Advantages**:
- Well-established statistical foundation
- Good performance across various problems
- Automatic selection without user input

**Disadvantages**:
- Can be computationally expensive
- May oversmooth in some cases

### Akaike Information Criterion (AIC)

AIC balances fit quality and model complexity:
$$\text{AIC}(\lambda) = n \log(\hat{\sigma}^2) + 2 \cdot \text{ED}(\lambda)$$

where:
- $\hat{\sigma}^2 = \|y - S_\lambda y\|^2 / n$ is the residual variance
- $\text{ED}(\lambda) = \text{tr}(S_\lambda)$ is the effective degrees of freedom

For GLM families (Poisson, Binomial), AIC uses deviance instead of RSS:
$$\text{AIC}(\lambda) = \text{Dev}(\lambda) + 2 \cdot \text{ED}(\lambda)$$

**Advantages**:
- Information-theoretic foundation
- Fast computation
- Good for model comparison
- Works with both Gaussian and GLM families

**Disadvantages**:
- May not work well for all noise levels
- Less robust than cross-validation

### L-Curve Method

The L-curve plots $\log(\|D\alpha_\lambda\|^2)$ vs $\log(\|y - B\alpha_\lambda\|^2)$ and finds the point of maximum curvature.

Curvature is computed as:
$$\kappa(\lambda) = \frac{2(\rho' \eta'' - \rho'' \eta')}{(\rho'^2 + \eta'^2)^{3/2}}$$

where $\rho(\lambda) = \log(\|y - B\alpha_\lambda\|^2)$ and $\eta(\lambda) = \log(\|D\alpha_\lambda\|^2)$.

**Advantages**:
- Intuitive geometric interpretation
- No statistical assumptions
- Good for ill-posed problems

**Disadvantages**:
- Can be sensitive to noise
- May not have clear corner

### V-Curve Method

Similar to L-curve but uses different scaling and looks for valley shape.

### Variable Penalty Selection (§8.8)

For the exponential variable penalty $P(\gamma) = D^T\text{diag}(e^{\gamma j/m})D$,
`variable_penalty_cv` evaluates GCV (or AIC) over a 2-D grid of
$(\lambda, \gamma)$ values and returns the combination that minimises the criterion:

$$(\hat\lambda, \hat\gamma) = \arg\min_{\lambda, \gamma} \text{GCV}(\lambda, \gamma)$$

This adds a single extra hyperparameter $\gamma$ that controls the spatial
distribution of the penalty weight while $\lambda$ controls overall smoothness.

## Implementation Details

### Optimization Strategy

All methods use:
1. **Logarithmic search**: Test λ values on logarithmic grid
2. **Golden section search**: Refine around the optimal region
3. **Sparse linear algebra**: Efficient computation of smoothing matrices

### Default Parameter Ranges

- **λ range**: $10^{-6}$ to $10^6$ (12 orders of magnitude)
- **Grid points**: 50 logarithmically spaced values
- **Refinement**: Golden section search with tolerance $10^{-6}$

### Performance Considerations

- **Cross-validation**: $O(n^3)$ for dense problems, $O(n^2)$ for sparse
- **AIC**: $O(n^2)$ computation per λ value
- **L-curve**: $O(n^2)$ plus curvature computation
- **V-curve**: Similar to L-curve

### Numerical Stability

- Uses sparse Cholesky decomposition when possible
- Handles ill-conditioned matrices gracefully
- Monitors condition numbers and issues warnings

## Method Selection Guidelines

### Recommended Approach

1. **Start with cross-validation**: Most robust for general use
2. **Try AIC**: If CV is too slow or gives unreasonable results
3. **Use L-curve**: For regularization-heavy applications
4. **Compare methods**: Check consistency across approaches

### Problem-Specific Recommendations

- **Small datasets** (n < 100): Cross-validation or AIC
- **Large datasets** (n > 1000): AIC or L-curve for speed
- **High noise**: Cross-validation (more robust)
- **Low noise**: Any method should work well
- **Sparse data**: L-curve or V-curve
- **Time series**: Cross-validation with temporal structure

### Troubleshooting

If optimization fails:
1. Check input data for issues (NaN, infinite values)
2. Try different nseg values
3. Reduce the λ search range
4. Use a different optimization method
5. Manually specify λ based on domain knowledge
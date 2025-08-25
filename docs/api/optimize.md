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

### L-Curve Method

```python
from psplines.optimize import l_curve

# Find optimal lambda using L-curve
best_lambda, curvature = l_curve(spline)
print(f"Optimal λ: {best_lambda:.6f}, Curvature: {curvature:.6f}")
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

**Advantages**:
- Information-theoretic foundation
- Fast computation
- Good for model comparison

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
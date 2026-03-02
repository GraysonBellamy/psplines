# Whittaker Smoother API

The Whittaker smoother is the special case of P-splines where the basis matrix $B = I$ (identity). Instead of estimating B-spline coefficients, the smoother operates directly on the data vector $y$, solving:

$$(\mathbf{W} + \lambda\, \mathbf{D}^\top \mathbf{D})\, \mathbf{z} = \mathbf{W}\, \mathbf{y}$$

For non-uniformly spaced data, the standard difference operator $D$ is replaced by the **divided-difference operator** $D_x$ which weights each finite difference by the reciprocal of the gap in $x$. This ensures the roughness penalty is expressed in the natural units of the independent variable rather than index position.

## WhittakerSmoother Class

::: psplines.whittaker.WhittakerSmoother
    options:
      members:
        - __init__
        - fit
        - predict
        - cross_validation
        - v_curve
      show_source: true
      show_bases: false
      heading_level: 3

## Usage Examples

### Basic Smoothing

```python
import numpy as np
from psplines import WhittakerSmoother

# Noisy signal
x = np.linspace(0, 10, 200)
y = np.sin(x) + 0.3 * np.random.default_rng(42).standard_normal(200)

# Fit with a fixed lambda
ws = WhittakerSmoother(x, y, lambda_=1e4, penalty_order=2)
ws.fit()

# Smoothed values (same length as input, in original order)
z = ws.fitted_values
```

### Automatic Lambda Selection

```python
# GCV-based selection (recommended)
ws = WhittakerSmoother(x, y)
best_lam, gcv_score = ws.cross_validation()
print(f"Optimal λ = {best_lam:.2f}")

# V-curve selection (alternative)
ws2 = WhittakerSmoother(x, y)
best_lam, v_score = ws2.v_curve()
```

### Non-Uniform Spacing

The key advantage over the standard Whittaker smoother: gaps in $x$ are handled correctly via divided differences.

```python
# Irregularly sampled signal
rng = np.random.default_rng(7)
x = np.sort(rng.uniform(0, 10, 150))
y = np.sin(x) + 0.2 * rng.standard_normal(150)

ws = WhittakerSmoother(x, y, lambda_=100)
ws.fit()

# The smoother automatically uses divided differences
# when x-spacing is non-uniform
```

### Missing Data via Weights

Zero weights effectively mark observations as missing; the smoother interpolates through them:

```python
x = np.linspace(0, 10, 200)
y_true = np.sin(x)
y = y_true.copy()
y[80:120] = 99.0  # corrupt a region

weights = np.ones(200)
weights[80:120] = 0.0  # mark as missing

ws = WhittakerSmoother(x, y, lambda_=1e3, weights=weights)
ws.fit()
# ws.fitted_values smoothly interpolates through the gap
```

### Interpolation at New Points

```python
ws = WhittakerSmoother(x, y, lambda_=1e3).fit()

# Predict at new locations via linear interpolation
x_new = np.linspace(0, 10, 500)
z_new = ws.predict(x_new)
```

### Unsorted Input

The smoother accepts unsorted $x$ — it sorts internally and returns results in the original order:

```python
rng = np.random.default_rng(42)
x = rng.uniform(0, 10, 100)  # not sorted
y = np.sin(x) + 0.1 * rng.standard_normal(100)

ws = WhittakerSmoother(x, y, lambda_=1e3).fit()
# ws.fitted_values[i] corresponds to x[i], not to sorted x
```

## When to Use WhittakerSmoother vs PSpline

| Criterion | `WhittakerSmoother` | `PSpline` |
|-----------|---------------------|-----------|
| **Basis** | Identity ($B = I$) | B-spline basis |
| **Prediction at new points** | Linear interpolation | Exact B-spline evaluation |
| **Derivatives** | Not supported | Analytic via derivative basis |
| **GLM families** | Gaussian only | Gaussian, Poisson, Binomial |
| **Shape constraints** | Not supported | Full support |
| **Speed** | Single sparse solve | Single sparse solve (+ basis construction) |
| **Non-uniform spacing** | Handled via divided differences | Handled via B-spline basis |
| **Ideal for** | Fast signal smoothing, spectra, time series | General smoothing, derivatives, GLMs |

!!! tip "Rule of thumb"
    Use `WhittakerSmoother` when you want fast, simple smoothing on a fixed grid — especially for uniformly or near-uniformly sampled signals where you don't need derivatives or fancy GLM features. Use `PSpline` for everything else.

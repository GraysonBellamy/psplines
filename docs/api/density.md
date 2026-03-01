# Density Estimation API

The density module provides smooth density estimation via Poisson P-splines on histogram counts.

## density_estimate

::: psplines.density.density_estimate
    options:
      show_source: true
      heading_level: 3

## DensityResult

::: psplines.density.DensityResult
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Basic Density Estimation

```python
import numpy as np
from psplines import density_estimate

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 500)

# Estimate density (AIC-optimal lambda)
result = density_estimate(data, bins=100)

# Plot
import matplotlib.pyplot as plt
plt.plot(result.grid, result.density)
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Smooth Density Estimate")
plt.show()
```

### Bimodal Distribution

```python
# Mixture of two Gaussians
data = np.concatenate([
    np.random.normal(-2, 0.5, 300),
    np.random.normal(2, 0.8, 200),
])

result = density_estimate(data, bins=100, penalty_order=3)

# result.density integrates to ~1
print(f"Integral: {np.trapz(result.density, result.grid):.4f}")
```

### Fixed Smoothing Parameter

```python
# Use a specific lambda instead of AIC selection
result = density_estimate(data, bins=80, lambda_=10.0)
```

### Custom Domain

```python
# Restrict estimation to a specific range
result = density_estimate(data, bins=100, domain=(-5, 5))
```

## Method

The density estimation procedure (Eilers & Marx 2021, §3.3):

1. Bin the raw data into a histogram with `bins` equally spaced bins
2. Fit a Poisson P-spline to the bin counts (log link)
3. Select $\lambda$ via AIC (or use a fixed value)
4. Normalize the fitted counts to integrate to 1

Using `penalty_order=3` (default) preserves the mean and variance of the data
(conservation of moments: a penalty of order $d$ preserves moments up to order $d-1$).

### DensityResult Attributes

| Attribute | Description |
|-----------|-------------|
| `grid` | Bin midpoints (evaluation points) |
| `density` | Normalized density values (integrates to ~1) |
| `mu` | Fitted Poisson counts (before normalization) |
| `lambda_` | Selected (or fixed) smoothing parameter |
| `pspline` | The underlying fitted `PSpline` object |
| `bin_width` | Width of each histogram bin |

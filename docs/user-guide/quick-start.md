# Quick Start

Get up and running with PSplines in minutes! This guide provides the essentials to start smoothing your data.

## Installation

Install PSplines using pip:

```bash
pip install psplines
```

For all features including Bayesian inference:
```bash
pip install psplines[full]
```

## 5-Minute Example

Here's a complete example that demonstrates the core PSplines workflow:

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline
from psplines.optimize import cross_validation

# 1. Generate some noisy data
np.random.seed(42)
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x) + 0.1 * np.random.randn(50)

# 2. Create a P-spline
spline = PSpline(x, y, nseg=20)

# 3. Find optimal smoothing parameter
optimal_lambda, _ = cross_validation(spline)
spline.lambda_ = optimal_lambda

# 4. Fit the spline
spline.fit()

# 5. Make predictions
x_new = np.linspace(0, 2*np.pi, 200)
y_pred = spline.predict(x_new)

# 6. Get uncertainty estimates
y_pred_with_se, se = spline.predict(x_new, return_se=True)

# 7. Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label='Noisy data')
plt.plot(x_new, np.sin(x_new), 'g--', label='True function')
plt.plot(x_new, y_pred, 'r-', linewidth=2, label='P-spline fit')
plt.fill_between(x_new, y_pred_with_se - 1.96*se, y_pred_with_se + 1.96*se, 
                 alpha=0.3, color='red', label='95% Confidence')
plt.legend()
plt.title('P-Spline Smoothing')
plt.show()

# 8. Check model properties
print(f"Effective degrees of freedom: {spline.ED:.2f}")
print(f"Optimal lambda: {optimal_lambda:.6f}")
print(f"Residual variance: {spline.sigma2:.6f}")
```

That's it! You've successfully fitted a P-spline, optimized parameters, and quantified uncertainty.

## Key Concepts in 60 Seconds

### P-Splines = B-splines + Penalties

P-splines combine two powerful ideas:

1. **B-spline basis**: Flexible piecewise polynomials
2. **Difference penalties**: Smooth by penalizing roughness

The result: automatic smoothing that adapts to your data.

### Three Essential Parameters

- **`nseg`**: Number of segments (more = more flexible)
- **`lambda_`**: Smoothing parameter (higher = smoother)
- **`penalty_order`**: Type of smoothness (2 = penalize curvature)

### Automatic Parameter Selection

Don't guess parameters - let the data decide:

```python
from psplines.optimize import cross_validation, aic_selection

# Cross-validation (recommended)
optimal_lambda, cv_score = cross_validation(spline)

# AIC (faster alternative)
optimal_lambda, aic_score = aic_selection(spline)
```

## Common Use Cases

### Smoothing Noisy Measurements
```python
# Your experimental data
time_points = np.array([0, 1, 2, 3, 4, 5])
measurements = np.array([1.0, 2.1, 2.9, 4.2, 4.8, 6.1])

# Quick smooth
spline = PSpline(time_points, measurements, nseg=10)
optimal_lambda, _ = cross_validation(spline)
spline.lambda_ = optimal_lambda
spline.fit()

# Get smooth curve
smooth_time = np.linspace(0, 5, 100)
smooth_measurements = spline.predict(smooth_time)
```

### Derivative Estimation
```python
# After fitting your spline...
first_derivative = spline.derivative(smooth_time, deriv_order=1)
second_derivative = spline.derivative(smooth_time, deriv_order=2)

plt.subplot(3, 1, 1)
plt.plot(smooth_time, spline.predict(smooth_time), label='Function')

plt.subplot(3, 1, 2)
plt.plot(smooth_time, first_derivative, label="f'(x)")

plt.subplot(3, 1, 3)
plt.plot(smooth_time, second_derivative, label="f''(x)")
```

### Time Series Analysis
```python
# Smooth time series data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
values = your_time_series_data  # Your data here

# Convert dates to numeric for fitting
x_numeric = np.arange(len(dates))

# Fit P-spline
spline = PSpline(x_numeric, values, nseg=52)  # Weekly segments
optimal_lambda, _ = cross_validation(spline)
spline.lambda_ = optimal_lambda
spline.fit()

# Extract trend
trend = spline.predict(x_numeric)
```

## Parameter Selection Guide

### Quick Rules of Thumb

**Number of segments (`nseg`)**:
- Start with `nseg = n_data_points / 4`
- Minimum: 5-10 segments
- Maximum: Usually no more than 50

**Smoothing parameter (`lambda_`)**:
- Use automatic selection (cross-validation)
- If manual: try values between 0.1 and 100

### When to Use What

| Scenario | Recommendation |
|----------|----------------|
| **Exploratory analysis** | `nseg=20`, cross-validation |
| **Very noisy data** | Higher `lambda_` or fewer segments |
| **Smooth underlying function** | More segments, moderate `lambda_` |
| **Need fast results** | AIC selection, fewer segments |
| **Critical application** | Cross-validation, bootstrap uncertainty |

## Error Handling and Validation

### Input Validation
PSplines automatically validates inputs:

```python
try:
    spline = PSpline(x, y, nseg=20)
    spline.fit()
except ValueError as e:
    print(f"Input error: {e}")
```

### Model Diagnostics
Always check your model:

```python
# After fitting
print(f"R² ≈ {1 - np.var(y - spline.predict(x)) / np.var(y):.3f}")
print(f"Effective DoF: {spline.ED:.1f}")

# Plot residuals
residuals = y - spline.predict(x)
plt.scatter(spline.predict(x), residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
```

## Next Steps

Now that you have the basics:

1. **[Getting Started Guide](getting-started.md)**: Deeper understanding of concepts
2. **[Basic Usage Tutorial](../tutorials/basic-usage.md)**: Step-by-step walkthroughs  
3. **[Parameter Selection Tutorial](../tutorials/parameter-selection.md)**: Master optimization methods
4. **[Examples Gallery](../examples/gallery.md)**: Real-world applications
5. **[API Reference](../api/core.md)**: Complete function documentation

## Troubleshooting

### Common Issues

**"Spline too wiggly"**: Increase `lambda_` or reduce `nseg`
```python
spline.lambda_ *= 10  # More smoothing
```

**"Spline too smooth"**: Decrease `lambda_` or increase `nseg`
```python
spline.lambda_ /= 10  # Less smoothing
```

**"Fitting fails"**: Check for:
- Duplicate x values
- Missing/infinite values
- Very small datasets (need at least 5-10 points)

**"Slow performance"**: 
- Reduce `nseg` for large datasets
- Use AIC instead of cross-validation
- Consider subsampling very large datasets

### Getting Help

- **Documentation**: Browse the full documentation
- **Examples**: Check the examples gallery  
- **Issues**: Report bugs on [GitHub Issues](https://github.com/graysonbellamy/psplines/issues)
- **Questions**: Use [GitHub Discussions](https://github.com/graysonbellamy/psplines/discussions)

---

**Ready to dive deeper?** Check out the comprehensive tutorials and examples to master P-spline smoothing for your specific applications!
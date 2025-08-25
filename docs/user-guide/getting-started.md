# Getting Started

Welcome to PSplines! This guide will help you get up and running with penalized B-spline smoothing in Python.

## What are P-Splines?

P-splines (Penalized B-splines) are a powerful and flexible method for smoothing noisy data. They combine:

- **B-spline basis functions**: Provide local flexibility and computational efficiency
- **Difference penalties**: Control smoothness by penalizing rough behavior
- **Automatic parameter selection**: Data-driven choice of smoothing level

P-splines are particularly useful when you have:

- Noisy measurements that need smoothing
- Need for derivative estimation
- Desire for uncertainty quantification
- Large datasets requiring efficient computation

## Key Concepts

### B-Spline Basis

B-splines are piecewise polynomials defined over a set of knots. They provide:
- **Local support**: Changes in one region don't affect distant regions
- **Computational efficiency**: Sparse matrices and fast algorithms
- **Flexibility**: Can approximate complex curves with appropriate parameters

### Difference Penalties

Instead of penalizing derivatives directly, P-splines penalize differences of adjacent B-spline coefficients:
- **First-order differences**: Penalize large changes in coefficients (roughness of slopes)
- **Second-order differences**: Penalize large changes in slopes (roughness of curvature)
- **Higher-order differences**: Penalize higher-order roughness measures

### The P-Spline Objective

P-splines minimize the penalized least squares objective:

$$\min_\alpha \|y - B\alpha\|^2 + \lambda \|D_p \alpha\|^2$$

where:
- $y$ is the data vector
- $B$ is the B-spline basis matrix
- $\alpha$ are the B-spline coefficients
- $D_p$ is the $p$-th order difference matrix
- $\lambda$ is the smoothing parameter

## Your First P-Spline

Let's start with a simple example:

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline

# Generate some noisy data
np.random.seed(42)
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x) + 0.1 * np.random.randn(50)

# Create a P-spline
spline = PSpline(x, y, nseg=15, lambda_=1.0)

# Fit the spline
spline.fit()

# Make predictions
x_new = np.linspace(0, 2*np.pi, 200)
y_smooth = spline.predict(x_new)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label='Noisy data')
plt.plot(x_new, y_smooth, 'r-', linewidth=2, label='P-spline fit')
plt.plot(x_new, np.sin(x_new), 'g--', alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Your First P-Spline')
plt.show()
```

## Understanding the Parameters

### Number of Segments (`nseg`)

Controls the flexibility of the basis:
- **Fewer segments**: More constrained, smoother fits
- **More segments**: More flexible, can capture fine details
- **Typical range**: 10-50 for most applications

```python
# Compare different numbers of segments
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, nseg in enumerate([5, 15, 30]):
    spline = PSpline(x, y, nseg=nseg, lambda_=1.0)
    spline.fit()
    y_fit = spline.predict(x_new)
    
    axes[i].scatter(x, y, alpha=0.6)
    axes[i].plot(x_new, y_fit, 'r-', linewidth=2)
    axes[i].set_title(f'nseg = {nseg}')
    axes[i].set_xlabel('x')
    if i == 0:
        axes[i].set_ylabel('y')

plt.tight_layout()
plt.show()
```

### Smoothing Parameter (`lambda_`)

Controls the trade-off between fit and smoothness:
- **Small λ**: Less smoothing, closer to data
- **Large λ**: More smoothing, smoother curves
- **Optimal λ**: Can be selected automatically (see tutorials)

```python
# Compare different smoothing parameters
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, lam in enumerate([0.01, 1.0, 100.0]):
    spline = PSpline(x, y, nseg=15, lambda_=lam)
    spline.fit()
    y_fit = spline.predict(x_new)
    
    axes[i].scatter(x, y, alpha=0.6)
    axes[i].plot(x_new, y_fit, 'r-', linewidth=2)
    axes[i].set_title(f'λ = {lam}')
    axes[i].set_xlabel('x')
    if i == 0:
        axes[i].set_ylabel('y')

plt.tight_layout()
plt.show()
```

### Penalty Order (`penalty_order`)

Controls what type of smoothness is enforced:
- **Order 1**: Penalizes large first differences (rough slopes)
- **Order 2**: Penalizes large second differences (rough curvature) - **most common**
- **Order 3**: Penalizes large third differences (rough jerk)

## Common Workflows

### 1. Basic Smoothing

```python
# Load your data
# x, y = load_your_data()

# Create and fit spline
spline = PSpline(x, y, nseg=20)
spline.fit()

# Get smooth fit
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = spline.predict(x_smooth)
```

### 2. Automatic Parameter Selection

```python
from psplines.optimize import cross_validation

# Create spline
spline = PSpline(x, y, nseg=20)

# Find optimal smoothing parameter
optimal_lambda, cv_score = cross_validation(spline)
spline.lambda_ = optimal_lambda
spline.fit()
```

### 3. Uncertainty Quantification

```python
# Fit spline
spline = PSpline(x, y, nseg=20, lambda_=1.0)
spline.fit()

# Get predictions with uncertainty
y_pred, se = spline.predict(x_smooth, return_se=True)

# Plot with confidence bands
plt.fill_between(x_smooth, y_pred - 1.96*se, y_pred + 1.96*se, 
                 alpha=0.3, label='95% CI')
plt.plot(x_smooth, y_pred, 'r-', label='Fit')
plt.scatter(x, y, alpha=0.6, label='Data')
plt.legend()
```

### 4. Derivative Estimation

```python
# Fit spline
spline = PSpline(x, y, nseg=20, lambda_=1.0)
spline.fit()

# Compute derivatives
dy_dx = spline.derivative(x_smooth, deriv_order=1)
d2y_dx2 = spline.derivative(x_smooth, deriv_order=2)

# Plot derivatives
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(x, y, alpha=0.6)
axes[0].plot(x_smooth, spline.predict(x_smooth), 'r-')
axes[0].set_title('Function')

axes[1].plot(x_smooth, dy_dx, 'g-')
axes[1].set_title('First Derivative')

axes[2].plot(x_smooth, d2y_dx2, 'm-')
axes[2].set_title('Second Derivative')

plt.tight_layout()
```

## Next Steps

Now that you understand the basics, explore:

1. **[Installation Guide](installation.md)**: Detailed installation instructions
2. **[Quick Start](quick-start.md)**: More examples and use cases
3. **[Core Concepts](core-concepts.md)**: Deeper mathematical understanding
4. **[Tutorials](../tutorials/basic-usage.md)**: Step-by-step walkthroughs
5. **[Examples](../examples/gallery.md)**: Real-world applications

## Common Questions

### Q: How do I choose the number of segments?

**A**: Start with `nseg = n/4` where `n` is your data size, then adjust based on your needs. More segments = more flexibility but potentially more overfitting.

### Q: How do I choose the smoothing parameter?

**A**: Use automatic selection methods like `cross_validation()` or `aic()`. These provide data-driven choices for the optimal smoothing level.

### Q: When should I use different penalty orders?

**A**: 
- Order 2 (default) works well for most applications
- Order 1 for piecewise linear trends
- Order 3 for very smooth curves

### Q: How do I handle large datasets?

**A**: P-splines are naturally efficient with sparse matrices. For very large datasets, consider:
- Using fewer segments initially
- Batching the data if memory is limited
- Using the bootstrap uncertainty only when needed

### Q: Can I constrain the fit at boundaries?

**A**: Yes! Use the `constraints` parameter to specify derivative constraints at boundaries. See the [Advanced Features](../tutorials/advanced-features.md) tutorial.
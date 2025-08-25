# Basic Usage Tutorial

This tutorial provides a comprehensive introduction to using PSplines for data smoothing and analysis.

## Introduction

P-splines (Penalized B-splines) are a flexible method for smoothing noisy data. This tutorial will guide you through the fundamental concepts and practical usage.

## Setting Up Your Environment

First, ensure you have PSplines installed with all dependencies:

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline
from psplines.optimize import cross_validation, aic_selection, l_curve

# Set random seed for reproducible results
np.random.seed(42)
```

## Creating Sample Data

Let's generate some noisy data to work with throughout this tutorial:

```python
# Generate synthetic data with noise
n_points = 100
x = np.linspace(0, 4*np.pi, n_points)
true_function = np.sin(x) * np.exp(-x/8)
noise_level = 0.1
y = true_function + noise_level * np.random.randn(n_points)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, s=30, label='Noisy observations')
plt.plot(x, true_function, 'g--', linewidth=2, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Sample Data: Noisy Sine Wave with Exponential Decay')
plt.grid(True, alpha=0.3)
plt.show()
```

## Basic P-Spline Fitting

### Creating and Fitting a P-Spline

The simplest way to use PSplines is to create a `PSpline` object and fit it:

```python
# Create a P-spline with 20 segments
spline = PSpline(x, y, nseg=20, lambda_=1.0)

# Fit the spline
spline.fit()

# Create evaluation points for smooth curve
x_eval = np.linspace(x.min(), x.max(), 200)
y_pred = spline.predict(x_eval)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, s=30, label='Data')
plt.plot(x_eval, y_pred, 'r-', linewidth=2, label='P-spline fit')
plt.plot(x, true_function, 'g--', linewidth=2, alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Basic P-Spline Fit')
plt.grid(True, alpha=0.3)
plt.show()
```

### Understanding Key Parameters

#### Number of Segments (`nseg`)

The number of segments controls the flexibility of the spline basis:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

nseg_values = [5, 15, 30, 50]

for i, nseg in enumerate(nseg_values):
    spline = PSpline(x, y, nseg=nseg, lambda_=1.0)
    spline.fit()
    y_pred = spline.predict(x_eval)
    
    axes[i].scatter(x, y, alpha=0.6, s=20)
    axes[i].plot(x_eval, y_pred, 'r-', linewidth=2)
    axes[i].plot(x, true_function, 'g--', alpha=0.7)
    axes[i].set_title(f'nseg = {nseg}, DoF = {spline.ED:.1f}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Effect of Number of Segments', y=1.02, fontsize=14)
plt.show()
```

#### Smoothing Parameter (`lambda_`)

The smoothing parameter controls the trade-off between fit and smoothness:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

lambda_values = [0.001, 0.1, 10.0, 1000.0]

for i, lam in enumerate(lambda_values):
    spline = PSpline(x, y, nseg=20, lambda_=lam)
    spline.fit()
    y_pred = spline.predict(x_eval)
    
    axes[i].scatter(x, y, alpha=0.6, s=20)
    axes[i].plot(x_eval, y_pred, 'r-', linewidth=2)
    axes[i].plot(x, true_function, 'g--', alpha=0.7)
    axes[i].set_title(f'λ = {lam}, DoF = {spline.ED:.1f}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Effect of Smoothing Parameter', y=1.02, fontsize=14)
plt.show()
```

## Automatic Parameter Selection

Instead of manually choosing parameters, you can use automatic selection methods:

### Cross-Validation

```python
# Create spline without specifying lambda
spline = PSpline(x, y, nseg=20)

# Use cross-validation to find optimal lambda
optimal_lambda, cv_score = cross_validation(spline, lambda_min=1e-5, lambda_max=1e3)
print(f"Optimal λ: {optimal_lambda:.6f}")
print(f"CV score: {cv_score:.6f}")

# Fit with optimal parameter
spline.lambda_ = optimal_lambda
spline.fit()

y_pred = spline.predict(x_eval)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, s=30, label='Data')
plt.plot(x_eval, y_pred, 'r-', linewidth=2, label=f'CV optimal (λ={optimal_lambda:.4f})')
plt.plot(x, true_function, 'g--', linewidth=2, alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Cross-Validation Optimized P-Spline (DoF = {spline.ED:.1f})')
plt.grid(True, alpha=0.3)
plt.show()
```

### AIC Selection

```python
# AIC-based selection
spline_aic = PSpline(x, y, nseg=20)
optimal_lambda_aic, aic_score = aic_selection(spline_aic, lambda_min=1e-5, lambda_max=1e3)
print(f"AIC optimal λ: {optimal_lambda_aic:.6f}")
print(f"AIC score: {aic_score:.6f}")

spline_aic.lambda_ = optimal_lambda_aic
spline_aic.fit()
```

### Comparing Selection Methods

```python
# Compare different selection methods
methods = ['Manual', 'Cross-Validation', 'AIC']
lambdas = [1.0, optimal_lambda, optimal_lambda_aic]
colors = ['blue', 'red', 'orange']

plt.figure(figsize=(12, 8))

for method, lam, color in zip(methods, lambdas, colors):
    spline_compare = PSpline(x, y, nseg=20, lambda_=lam)
    spline_compare.fit()
    y_pred_compare = spline_compare.predict(x_eval)
    
    plt.plot(x_eval, y_pred_compare, color=color, linewidth=2, 
             label=f'{method} (λ={lam:.4f}, DoF={spline_compare.ED:.1f})')

plt.scatter(x, y, alpha=0.6, s=30, color='gray', label='Data')
plt.plot(x, true_function, 'g--', linewidth=2, alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Comparison of Parameter Selection Methods')
plt.grid(True, alpha=0.3)
plt.show()
```

## Working with Derivatives

P-splines can efficiently compute derivatives of the fitted function:

```python
# Fit spline with optimal parameters
spline = PSpline(x, y, nseg=20, lambda_=optimal_lambda)
spline.fit()

# Compute function and derivatives
y_pred = spline.predict(x_eval)
dy_dx = spline.derivative(x_eval, deriv_order=1)
d2y_dx2 = spline.derivative(x_eval, deriv_order=2)

# True derivatives for comparison
true_dy_dx = np.cos(x_eval) * np.exp(-x_eval/8) - (1/8) * np.sin(x_eval) * np.exp(-x_eval/8)
true_d2y_dx2 = (-np.sin(x_eval) - (1/4)*np.cos(x_eval) + (1/64)*np.sin(x_eval)) * np.exp(-x_eval/8)

# Plot function and derivatives
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Function
axes[0].scatter(x, y, alpha=0.6, s=20, label='Data')
axes[0].plot(x_eval, y_pred, 'r-', linewidth=2, label='P-spline')
axes[0].plot(x_eval, true_function, 'g--', alpha=0.7, label='True')
axes[0].set_title('Function')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# First derivative
axes[1].plot(x_eval, dy_dx, 'r-', linewidth=2, label="P-spline f'")
axes[1].plot(x_eval, true_dy_dx, 'g--', alpha=0.7, label="True f'")
axes[1].set_title('First Derivative')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Second derivative
axes[2].plot(x_eval, d2y_dx2, 'r-', linewidth=2, label='P-spline f"')
axes[2].plot(x_eval, true_d2y_dx2, 'g--', alpha=0.7, label='True f"')
axes[2].set_title('Second Derivative')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Accessing Model Information

After fitting, you can access various model properties:

```python
print("=== Model Information ===")
print(f"Number of data points: {spline.n}")
print(f"Number of basis functions: {spline.nb}")
print(f"Effective degrees of freedom: {spline.ED:.2f}")
print(f"Residual variance (σ²): {spline.sigma2:.6f}")
print(f"Residual standard deviation (σ): {np.sqrt(spline.sigma2):.6f}")
print(f"Smoothing parameter (λ): {spline.lambda_}")
print(f"B-spline degree: {spline.degree}")
print(f"Penalty order: {spline.penalty_order}")

# Model diagnostics
residuals = y - spline.predict(x)
print(f"\n=== Diagnostics ===")
print(f"Mean squared error: {np.mean(residuals**2):.6f}")
print(f"Mean absolute error: {np.mean(np.abs(residuals)):.6f}")
print(f"R-squared (approx): {1 - np.var(residuals)/np.var(y):.4f}")
```

## Handling Different Data Scenarios

### Non-uniform Data Spacing

P-splines work well with non-uniformly spaced data:

```python
# Create non-uniform data
n_points = 80
x_nonuniform = np.sort(np.random.uniform(0, 4*np.pi, n_points))
y_nonuniform = (np.sin(x_nonuniform) * np.exp(-x_nonuniform/8) + 
                0.1 * np.random.randn(n_points))

# Fit P-spline
spline_nonuniform = PSpline(x_nonuniform, y_nonuniform, nseg=25)
optimal_lambda_nu, _ = cross_validation(spline_nonuniform)
spline_nonuniform.lambda_ = optimal_lambda_nu
spline_nonuniform.fit()

# Evaluate and plot
x_eval_nu = np.linspace(x_nonuniform.min(), x_nonuniform.max(), 200)
y_pred_nu = spline_nonuniform.predict(x_eval_nu)

plt.figure(figsize=(10, 6))
plt.scatter(x_nonuniform, y_nonuniform, alpha=0.6, s=30, label='Non-uniform data')
plt.plot(x_eval_nu, y_pred_nu, 'r-', linewidth=2, label='P-spline fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('P-Spline with Non-Uniform Data Spacing')
plt.grid(True, alpha=0.3)
plt.show()
```

### Sparse Data

P-splines can handle sparse data by using fewer segments:

```python
# Create sparse data
x_sparse = x[::5]  # Take every 5th point
y_sparse = y[::5]

print(f"Sparse data: {len(x_sparse)} points (originally {len(x)} points)")

# Fit with fewer segments for sparse data
spline_sparse = PSpline(x_sparse, y_sparse, nseg=8)
optimal_lambda_sparse, _ = cross_validation(spline_sparse)
spline_sparse.lambda_ = optimal_lambda_sparse
spline_sparse.fit()

y_pred_sparse = spline_sparse.predict(x_eval)

plt.figure(figsize=(10, 6))
plt.scatter(x_sparse, y_sparse, alpha=0.8, s=50, label=f'Sparse data (n={len(x_sparse)})')
plt.plot(x_eval, y_pred_sparse, 'r-', linewidth=2, label='P-spline fit')
plt.plot(x, true_function, 'g--', alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('P-Spline with Sparse Data')
plt.grid(True, alpha=0.3)
plt.show()
```

## Common Pitfalls and Solutions

### Over-smoothing

If your fit looks too smooth and misses important features:

```python
# Demonstrate over-smoothing
spline_oversmooth = PSpline(x, y, nseg=20, lambda_=1000.0)
spline_oversmooth.fit()
y_oversmooth = spline_oversmooth.predict(x_eval)

# Solution: reduce lambda or use automatic selection
spline_corrected = PSpline(x, y, nseg=20, lambda_=0.1)
spline_corrected.fit()
y_corrected = spline_corrected.predict(x_eval)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.6, s=30)
plt.plot(x_eval, y_oversmooth, 'r-', linewidth=2, label=f'Over-smooth (λ=1000, DoF={spline_oversmooth.ED:.1f})')
plt.plot(x, true_function, 'g--', alpha=0.7, label='True')
plt.title('Problem: Over-smoothing')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(x, y, alpha=0.6, s=30)
plt.plot(x_eval, y_corrected, 'r-', linewidth=2, label=f'Corrected (λ=0.1, DoF={spline_corrected.ED:.1f})')
plt.plot(x, true_function, 'g--', alpha=0.7, label='True')
plt.title('Solution: Reduced λ')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Under-smoothing (Overfitting)

If your fit follows the noise too closely:

```python
# Demonstrate under-smoothing
spline_undersmooth = PSpline(x, y, nseg=40, lambda_=0.001)
spline_undersmooth.fit()
y_undersmooth = spline_undersmooth.predict(x_eval)

# Solution: increase lambda or reduce nseg
spline_corrected2 = PSpline(x, y, nseg=20, lambda_=10.0)
spline_corrected2.fit()
y_corrected2 = spline_corrected2.predict(x_eval)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.6, s=30)
plt.plot(x_eval, y_undersmooth, 'r-', linewidth=2, label=f'Under-smooth (λ=0.001, DoF={spline_undersmooth.ED:.1f})')
plt.plot(x, true_function, 'g--', alpha=0.7, label='True')
plt.title('Problem: Under-smoothing')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(x, y, alpha=0.6, s=30)
plt.plot(x_eval, y_corrected2, 'r-', linewidth=2, label=f'Corrected (λ=10, DoF={spline_corrected2.ED:.1f})')
plt.plot(x, true_function, 'g--', alpha=0.7, label='True')
plt.title('Solution: Increased λ')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Summary

In this tutorial, you learned:

1. **Basic P-spline fitting** with the `PSpline` class
2. **Key parameters**: `nseg` (flexibility) and `lambda_` (smoothing)
3. **Automatic parameter selection** using cross-validation and AIC
4. **Derivative computation** for fitted functions
5. **Model diagnostics** and information extraction
6. **Handling different data scenarios**: non-uniform, sparse data
7. **Common pitfalls** and their solutions

## Next Steps

- **[Parameter Selection](parameter-selection.md)**: Deep dive into optimization methods
- **[Uncertainty Methods](uncertainty-methods.md)**: Learn about confidence intervals and bootstrap
- **[Advanced Features](advanced-features.md)**: Constraints, Bayesian inference, and more
- **[Examples Gallery](../examples/gallery.md)**: Real-world applications
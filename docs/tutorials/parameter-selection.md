# Parameter Selection Tutorial

This tutorial covers the various methods available in PSplines for automatic parameter selection, helping you choose optimal smoothing parameters for your data.

## Introduction

One of the most critical aspects of P-spline fitting is selecting the appropriate smoothing parameter λ (lambda). This parameter controls the bias-variance trade-off:

- **Small λ**: Less smoothing, lower bias, higher variance
- **Large λ**: More smoothing, higher bias, lower variance

This tutorial demonstrates all available methods for automatic λ selection.

## Setup and Sample Data

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline
from psplines.optimize import cross_validation, aic_selection, l_curve

# Generate sample data
np.random.seed(42)
n = 100
x = np.linspace(0, 2*np.pi, n)
true_function = np.sin(2*x) * np.exp(-x/3)
noise_std = 0.15
y = true_function + noise_std * np.random.randn(n)

# Evaluation points
x_eval = np.linspace(0, 2*np.pi, 200)
true_eval = np.sin(2*x_eval) * np.exp(-x_eval/3)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, s=30, label='Noisy data')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Sample Data for Parameter Selection')
plt.grid(True, alpha=0.3)
plt.show()
```

## Cross-Validation (Recommended)

Cross-validation is generally the most reliable method for parameter selection.

### Basic Cross-Validation

```python
# Create spline object
spline = PSpline(x, y, nseg=25)

# Perform cross-validation
optimal_lambda, cv_score = cross_validation(
    spline, 
    lambda_min=1e-6, 
    lambda_max=1e2,
    n_lambda=50,
    cv_method='kfold',
    k_folds=5
)

print(f"Optimal λ (CV): {optimal_lambda:.6f}")
print(f"CV score: {cv_score:.6f}")

# Fit with optimal parameter
spline.lambda_ = optimal_lambda
spline.fit()

# Evaluate
y_pred_cv = spline.predict(x_eval)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, s=30, label='Data')
plt.plot(x_eval, y_pred_cv, 'r-', linewidth=2, 
         label=f'CV optimal (λ={optimal_lambda:.4f}, DoF={spline.ED:.1f})')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Cross-Validation Parameter Selection')
plt.grid(True, alpha=0.3)
plt.show()
```

### Understanding the CV Curve

```python
# Generate lambda values for CV curve
lambda_values = np.logspace(-6, 2, 50)
cv_scores = []
dof_values = []

for lam in lambda_values:
    spline_temp = PSpline(x, y, nseg=25, lambda_=lam)
    spline_temp.fit()
    
    # Compute CV score manually for demonstration
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_errors = []
    
    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        spline_cv = PSpline(x_train, y_train, nseg=25, lambda_=lam)
        spline_cv.fit()
        y_pred_test = spline_cv.predict(x_test)
        cv_errors.append(np.mean((y_test - y_pred_test)**2))
    
    cv_scores.append(np.mean(cv_errors))
    dof_values.append(spline_temp.ED)

# Find minimum
min_idx = np.argmin(cv_scores)
optimal_lambda_manual = lambda_values[min_idx]

# Plot CV curve
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# CV score vs lambda
ax1.semilogx(lambda_values, cv_scores, 'b-', linewidth=2, label='CV score')
ax1.axvline(optimal_lambda_manual, color='r', linestyle='--', 
            label=f'Minimum (λ={optimal_lambda_manual:.4f})')
ax1.set_xlabel('λ (smoothing parameter)')
ax1.set_ylabel('Cross-Validation Score')
ax1.set_title('Cross-Validation Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# DoF vs lambda
ax2.semilogx(lambda_values, dof_values, 'g-', linewidth=2, label='Degrees of Freedom')
ax2.axvline(optimal_lambda_manual, color='r', linestyle='--')
ax2.set_xlabel('λ (smoothing parameter)')
ax2.set_ylabel('Effective Degrees of Freedom')
ax2.set_title('Model Complexity vs Smoothing Parameter')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Manual CV optimal λ: {optimal_lambda_manual:.6f}")
print(f"Built-in CV optimal λ: {optimal_lambda:.6f}")
```

## AIC Selection

The Akaike Information Criterion balances model fit and complexity.

```python
# AIC-based selection
spline_aic = PSpline(x, y, nseg=25)
optimal_lambda_aic, aic_score = aic_selection(
    spline_aic,
    lambda_min=1e-6,
    lambda_max=1e2,
    n_lambda=50
)

print(f"Optimal λ (AIC): {optimal_lambda_aic:.6f}")
print(f"AIC score: {aic_score:.6f}")

# Fit and evaluate
spline_aic.lambda_ = optimal_lambda_aic
spline_aic.fit()
y_pred_aic = spline_aic.predict(x_eval)

# Plot AIC curve
lambda_values_aic = np.logspace(-6, 2, 50)
aic_scores = []

for lam in lambda_values_aic:
    spline_temp = PSpline(x, y, nseg=25, lambda_=lam)
    spline_temp.fit()
    
    # Compute AIC
    n = len(y)
    residuals = y - spline_temp.predict(x)
    mse = np.mean(residuals**2)
    aic = n * np.log(mse) + 2 * spline_temp.ED
    aic_scores.append(aic)

min_aic_idx = np.argmin(aic_scores)
optimal_lambda_aic_manual = lambda_values_aic[min_aic_idx]

plt.figure(figsize=(10, 6))
plt.semilogx(lambda_values_aic, aic_scores, 'b-', linewidth=2, label='AIC score')
plt.axvline(optimal_lambda_aic_manual, color='r', linestyle='--',
            label=f'Minimum (λ={optimal_lambda_aic_manual:.4f})')
plt.xlabel('λ (smoothing parameter)')
plt.ylabel('AIC Score')
plt.title('AIC Selection Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## L-Curve Method

The L-curve method finds the corner in the trade-off between fit and smoothness.

```python
# L-curve selection
spline_lcurve = PSpline(x, y, nseg=25)
optimal_lambda_lcurve, curvature_info = l_curve(
    spline_lcurve,
    lambda_min=1e-6,
    lambda_max=1e2,
    n_lambda=50
)

print(f"Optimal λ (L-curve): {optimal_lambda_lcurve:.6f}")

# Fit and evaluate
spline_lcurve.lambda_ = optimal_lambda_lcurve
spline_lcurve.fit()
y_pred_lcurve = spline_lcurve.predict(x_eval)

# Create L-curve plot
lambda_values_lc = np.logspace(-6, 2, 50)
residual_norms = []
penalty_norms = []

for lam in lambda_values_lc:
    spline_temp = PSpline(x, y, nseg=25, lambda_=lam)
    spline_temp.fit()
    
    residuals = y - spline_temp.predict(x)
    residual_norm = np.linalg.norm(residuals)
    
    # Compute penalty norm
    penalty_norm = np.linalg.norm(spline_temp.penalty_matrix @ spline_temp.alpha)
    
    residual_norms.append(residual_norm)
    penalty_norms.append(penalty_norm)

# Plot L-curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(residual_norms, penalty_norms, 'b-', linewidth=2, marker='o', markersize=3)
optimal_idx = np.argmin(np.abs(lambda_values_lc - optimal_lambda_lcurve))
plt.loglog(residual_norms[optimal_idx], penalty_norms[optimal_idx], 
           'ro', markersize=10, label=f'L-curve corner (λ={optimal_lambda_lcurve:.4f})')
plt.xlabel('||Residuals||')
plt.ylabel('||Penalty||')
plt.title('L-Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot curvature
curvatures = []
for i, lam in enumerate(lambda_values_lc):
    if i > 0 and i < len(lambda_values_lc) - 1:
        # Simple curvature approximation
        x1, y1 = np.log(residual_norms[i-1]), np.log(penalty_norms[i-1])
        x2, y2 = np.log(residual_norms[i]), np.log(penalty_norms[i])
        x3, y3 = np.log(residual_norms[i+1]), np.log(penalty_norms[i+1])
        
        # Curvature formula for parametric curve
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2
        
        curvature = abs(dx1*dy2 - dy1*dx2) / (dx1**2 + dy1**2)**1.5 if (dx1**2 + dy1**2) > 0 else 0
        curvatures.append(curvature)
    else:
        curvatures.append(0)

plt.subplot(1, 2, 2)
plt.semilogx(lambda_values_lc, curvatures, 'g-', linewidth=2, label='Curvature')
plt.axvline(optimal_lambda_lcurve, color='r', linestyle='--', label=f'Max curvature')
plt.xlabel('λ (smoothing parameter)')
plt.ylabel('Curvature')
plt.title('L-Curve Curvature')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Generalized Cross-Validation (GCV)

GCV is an efficient approximation to leave-one-out cross-validation.

```python
def gcv_score(spline):
    """Compute GCV score for fitted spline."""
    n = spline.n
    residuals = spline.y - spline.predict(spline.x)
    rss = np.sum(residuals**2)
    
    # GCV formula
    gcv = (n * rss) / (n - spline.ED)**2
    return gcv

# GCV selection
lambda_values_gcv = np.logspace(-6, 2, 50)
gcv_scores = []

for lam in lambda_values_gcv:
    spline_temp = PSpline(x, y, nseg=25, lambda_=lam)
    spline_temp.fit()
    gcv_scores.append(gcv_score(spline_temp))

# Find optimal
min_gcv_idx = np.argmin(gcv_scores)
optimal_lambda_gcv = lambda_values_gcv[min_gcv_idx]

print(f"Optimal λ (GCV): {optimal_lambda_gcv:.6f}")

# Fit with optimal parameter
spline_gcv = PSpline(x, y, nseg=25, lambda_=optimal_lambda_gcv)
spline_gcv.fit()
y_pred_gcv = spline_gcv.predict(x_eval)

# Plot GCV curve
plt.figure(figsize=(10, 6))
plt.semilogx(lambda_values_gcv, gcv_scores, 'b-', linewidth=2, label='GCV score')
plt.axvline(optimal_lambda_gcv, color='r', linestyle='--',
            label=f'Minimum (λ={optimal_lambda_gcv:.4f})')
plt.xlabel('λ (smoothing parameter)')
plt.ylabel('GCV Score')
plt.title('Generalized Cross-Validation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Comparing All Methods

Let's compare all parameter selection methods:

```python
# Collect all methods and their optimal lambdas
methods = {
    'Cross-Validation': optimal_lambda,
    'AIC': optimal_lambda_aic,
    'L-Curve': optimal_lambda_lcurve,
    'GCV': optimal_lambda_gcv
}

# Fit splines with each method
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

colors = ['red', 'blue', 'green', 'orange']
predictions = {}

for i, (method, lam) in enumerate(methods.items()):
    spline_method = PSpline(x, y, nseg=25, lambda_=lam)
    spline_method.fit()
    y_pred_method = spline_method.predict(x_eval)
    predictions[method] = y_pred_method
    
    # Calculate metrics
    residuals = y - spline_method.predict(x)
    mse = np.mean(residuals**2)
    
    axes[i].scatter(x, y, alpha=0.6, s=20, color='gray')
    axes[i].plot(x_eval, y_pred_method, color=colors[i], linewidth=2, 
                label=f'{method}')
    axes[i].plot(x_eval, true_eval, 'g--', alpha=0.7, linewidth=1.5, label='True')
    axes[i].set_title(f'{method}\nλ={lam:.4f}, DoF={spline_method.ED:.1f}, MSE={mse:.4f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary comparison
print("\n=== Parameter Selection Comparison ===")
print(f"{'Method':<20} {'Lambda':<12} {'DoF':<8} {'MSE':<12}")
print("-" * 55)

for method, lam in methods.items():
    spline_temp = PSpline(x, y, nseg=25, lambda_=lam)
    spline_temp.fit()
    residuals = y - spline_temp.predict(x)
    mse = np.mean(residuals**2)
    print(f"{method:<20} {lam:<12.6f} {spline_temp.ED:<8.2f} {mse:<12.6f}")
```

## Method-Specific Parameters and Options

### Advanced Cross-Validation Options

```python
# Different CV methods
cv_methods = ['kfold', 'loo']  # k-fold and leave-one-out

for cv_method in cv_methods:
    if cv_method == 'kfold':
        opt_lambda, score = cross_validation(
            PSpline(x, y, nseg=25),
            cv_method='kfold',
            k_folds=10,  # More folds for better estimate
            lambda_min=1e-6,
            lambda_max=1e2,
            n_lambda=30
        )
    else:  # Leave-one-out
        opt_lambda, score = cross_validation(
            PSpline(x, y, nseg=25),
            cv_method='loo',
            lambda_min=1e-6,
            lambda_max=1e2,
            n_lambda=30
        )
    
    print(f"{cv_method.upper()} CV: λ = {opt_lambda:.6f}, score = {score:.6f}")
```

### Grid Search vs. Optimization

```python
# Grid search (exhaustive)
lambda_grid = np.logspace(-4, 1, 100)  # Fine grid
spline_grid = PSpline(x, y, nseg=25)

best_lambda_grid = None
best_score = float('inf')

for lam in lambda_grid:
    spline_grid.lambda_ = lam
    spline_grid.fit()
    
    # Use GCV as criterion
    score = gcv_score(spline_grid)
    
    if score < best_score:
        best_score = score
        best_lambda_grid = lam

print(f"Grid search optimal λ: {best_lambda_grid:.6f}")

# Compare with optimization-based approach
from scipy.optimize import minimize_scalar

def gcv_objective(log_lambda):
    """Objective function for optimization."""
    lam = 10**log_lambda
    spline_temp = PSpline(x, y, nseg=25, lambda_=lam)
    spline_temp.fit()
    return gcv_score(spline_temp)

# Optimize
result = minimize_scalar(gcv_objective, bounds=(-6, 2), method='bounded')
optimal_lambda_opt = 10**result.x

print(f"Optimization-based optimal λ: {optimal_lambda_opt:.6f}")
```

## Choosing the Right Method

### Decision Guidelines

```python
def recommend_method(n_points, noise_level='unknown'):
    """
    Recommend parameter selection method based on data characteristics.
    """
    recommendations = []
    
    if n_points < 50:
        recommendations.append("• Use AIC or L-curve (CV may be unreliable with small samples)")
    elif n_points < 200:
        recommendations.append("• Cross-validation (5-fold) or AIC are good choices")
    else:
        recommendations.append("• Cross-validation (10-fold) or GCV for efficiency")
    
    if noise_level == 'high':
        recommendations.append("• Consider L-curve method - more robust to high noise")
    elif noise_level == 'low':
        recommendations.append("• AIC works well with low-noise data")
    
    return recommendations

# Example usage
n = len(x)
print(f"Data size: {n} points")
print("Recommendations:")
for rec in recommend_method(n, 'medium'):
    print(rec)
```

### Performance Comparison

```python
import time

# Time different methods
methods_to_time = [
    ('Cross-Validation', lambda s: cross_validation(s, n_lambda=20)),
    ('AIC', lambda s: aic_selection(s, n_lambda=20)),
    ('L-Curve', lambda s: l_curve(s, n_lambda=20))
]

print("=== Performance Comparison ===")
print(f"{'Method':<20} {'Time (s)':<10} {'Lambda':<12}")
print("-" * 45)

for method_name, method_func in methods_to_time:
    spline_test = PSpline(x, y, nseg=25)
    
    start_time = time.time()
    opt_lambda, _ = method_func(spline_test)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"{method_name:<20} {elapsed:<10.4f} {opt_lambda:<12.6f}")
```

## Advanced Tips

### Multiple Criteria Consensus

```python
# Use multiple methods and take consensus
all_lambdas = [
    optimal_lambda,      # CV
    optimal_lambda_aic,  # AIC
    optimal_lambda_gcv,  # GCV
    optimal_lambda_lcurve # L-curve
]

# Geometric mean as consensus (works well for log-scale parameters)
consensus_lambda = np.exp(np.mean(np.log(all_lambdas)))
print(f"Consensus λ (geometric mean): {consensus_lambda:.6f}")

# Fit with consensus parameter
spline_consensus = PSpline(x, y, nseg=25, lambda_=consensus_lambda)
spline_consensus.fit()
y_pred_consensus = spline_consensus.predict(x_eval)

# Compare consensus with individual methods
plt.figure(figsize=(12, 8))
plt.scatter(x, y, alpha=0.6, s=30, color='gray', label='Data')

methods_plot = [
    ('CV', optimal_lambda, 'red'),
    ('AIC', optimal_lambda_aic, 'blue'),
    ('GCV', optimal_lambda_gcv, 'green'),
    ('Consensus', consensus_lambda, 'purple')
]

for method, lam, color in methods_plot:
    spline_temp = PSpline(x, y, nseg=25, lambda_=lam)
    spline_temp.fit()
    y_pred_temp = spline_temp.predict(x_eval)
    plt.plot(x_eval, y_pred_temp, color=color, linewidth=2, alpha=0.8,
             label=f'{method} (λ={lam:.4f})')

plt.plot(x_eval, true_eval, 'g--', linewidth=2, alpha=0.7, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Consensus Parameter Selection')
plt.grid(True, alpha=0.3)
plt.show()
```

## Summary

In this tutorial, you learned:

1. **Cross-validation**: Most reliable, especially k-fold CV
2. **AIC selection**: Good balance of accuracy and efficiency
3. **L-curve method**: Robust to noise, geometric interpretation
4. **GCV**: Efficient approximation to leave-one-out CV
5. **Method selection**: Guidelines based on data characteristics
6. **Advanced techniques**: Grid search, optimization, consensus methods

### Recommendations:
- **General use**: 5-10 fold cross-validation
- **Small datasets**: AIC or L-curve
- **Large datasets**: GCV or AIC for efficiency
- **High noise**: L-curve method
- **Critical applications**: Use consensus of multiple methods

## Next Steps

- **[Uncertainty Methods](uncertainty-methods.md)**: Learn about confidence intervals
- **[Advanced Features](advanced-features.md)**: Constraints and specialized techniques
- **[Basic Usage](basic-usage.md)**: Review fundamental concepts
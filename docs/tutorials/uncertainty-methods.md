# Uncertainty Quantification Tutorial

This tutorial covers the different methods available in PSplines for quantifying uncertainty in your fitted curves, including analytical standard errors, bootstrap methods, and Bayesian approaches.

## Introduction

Uncertainty quantification is crucial for:
- Understanding the reliability of your smooth fits
- Making statistical inferences from your results
- Communicating confidence in predictions
- Identifying regions where more data might be needed

PSplines offers three main approaches to uncertainty quantification:
1. **Analytical standard errors** (fast, approximate)
2. **Bootstrap methods** (slower, empirical)
3. **Bayesian inference** (comprehensive, requires PyMC)

## Setup and Sample Data

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline
from psplines.optimize import cross_validation
import warnings
warnings.filterwarnings('ignore')

# Generate sample data with known uncertainty
np.random.seed(42)
n = 80
x = np.sort(np.random.uniform(0, 3*np.pi, n))
true_function = np.sin(x) * np.exp(-x/5) + 0.2 * np.cos(3*x)
noise_std = 0.12
y = true_function + noise_std * np.random.randn(n)

# Evaluation points
x_eval = np.linspace(0, 3*np.pi, 200)
true_eval = np.sin(x_eval) * np.exp(-x_eval/5) + 0.2 * np.cos(3*x_eval)

# Plot data
plt.figure(figsize=(12, 6))
plt.scatter(x, y, alpha=0.7, s=40, label='Noisy observations')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Sample Data for Uncertainty Analysis')
plt.grid(True, alpha=0.3)
plt.show()

# Fit optimal P-spline
spline = PSpline(x, y, nseg=30)
optimal_lambda, _ = cross_validation(spline)
spline.lambda_ = optimal_lambda
spline.fit()

print(f"Optimal λ: {optimal_lambda:.6f}")
print(f"Effective DoF: {spline.ED:.2f}")
print(f"Estimated σ²: {spline.sigma2:.6f}")
```

## Method 1: Analytical Standard Errors

The fastest method uses analytical formulas based on the covariance matrix.

### Basic Usage

```python
# Get predictions with analytical standard errors
y_pred_analytical, se_analytical = spline.predict(x_eval, return_se=True, se_method='analytic')

# Create confidence bands
confidence_level = 0.95
z_score = 1.96  # For 95% confidence
lower_band = y_pred_analytical - z_score * se_analytical
upper_band = y_pred_analytical + z_score * se_analytical

# Plot results
plt.figure(figsize=(12, 8))
plt.scatter(x, y, alpha=0.7, s=40, color='gray', label='Data')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
plt.plot(x_eval, y_pred_analytical, 'r-', linewidth=2, label='P-spline fit')
plt.fill_between(x_eval, lower_band, upper_band, alpha=0.3, color='red', 
                 label=f'{int(confidence_level*100)}% Confidence Band')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Analytical Standard Errors')
plt.grid(True, alpha=0.3)
plt.show()

# Compute coverage statistics
# Points where true function lies within confidence band
in_band = (true_eval >= lower_band) & (true_eval <= upper_band)
coverage = np.mean(in_band)
print(f"Empirical coverage: {coverage:.3f} (expected: {confidence_level:.3f})")
```

### Understanding the Analytical Method

The analytical method computes standard errors using:

```python
# Show the mathematical foundation
print("=== Analytical Method Details ===")
print(f"Model: y = Bα + ε, where ε ~ N(0, σ²I)")
print(f"Standard errors based on: SE(f(x)) = σ √[b(x)ᵀ(BᵀB + λPᵀP)⁻¹BᵀB(BᵀB + λPᵀP)⁻¹b(x)]")
print(f"Where:")
print(f"  - B is the basis matrix")
print(f"  - P is the penalty matrix")
print(f"  - b(x) is the basis vector at point x")
print(f"  - σ² = {spline.sigma2:.6f} (estimated from residuals)")

# Visualize pointwise standard errors
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_eval, se_analytical, 'b-', linewidth=2, label='Standard Error')
plt.axhline(y=noise_std, color='g', linestyle='--', label=f'True noise std ({noise_std})')
plt.xlabel('x')
plt.ylabel('Standard Error')
plt.title('Pointwise Standard Errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Show relative uncertainty
relative_se = se_analytical / np.abs(y_pred_analytical)
plt.plot(x_eval, relative_se, 'purple', linewidth=2, label='Relative SE')
plt.xlabel('x')
plt.ylabel('Relative Standard Error')
plt.title('Relative Uncertainty')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Mean standard error: {np.mean(se_analytical):.4f}")
print(f"Min/Max standard error: {np.min(se_analytical):.4f} / {np.max(se_analytical):.4f}")
```

## Method 2: Bootstrap Methods

Bootstrap methods provide empirical estimates by resampling from the fitted model.

### Parametric Bootstrap

```python
# Parametric bootstrap (assumes Gaussian errors)
y_pred_bootstrap, se_bootstrap = spline.predict(
    x_eval, 
    return_se=True, 
    se_method='bootstrap',
    bootstrap_method='parametric',
    B_boot=500,  # Number of bootstrap samples
    n_jobs=2     # Parallel processing
)

# Create confidence bands
lower_band_boot = y_pred_bootstrap - 1.96 * se_bootstrap
upper_band_boot = y_pred_bootstrap + 1.96 * se_bootstrap

# Plot comparison
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.scatter(x, y, alpha=0.7, s=30, color='gray', label='Data')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
plt.plot(x_eval, y_pred_analytical, 'r-', linewidth=2, label='Fit')
plt.fill_between(x_eval, lower_band, upper_band, alpha=0.3, color='red', 
                 label='Analytical 95% CI')
plt.title('Analytical Standard Errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.scatter(x, y, alpha=0.7, s=30, color='gray', label='Data')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
plt.plot(x_eval, y_pred_bootstrap, 'b-', linewidth=2, label='Fit')
plt.fill_between(x_eval, lower_band_boot, upper_band_boot, alpha=0.3, color='blue', 
                 label='Bootstrap 95% CI')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bootstrap Standard Errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare coverage
in_band_boot = (true_eval >= lower_band_boot) & (true_eval <= upper_band_boot)
coverage_boot = np.mean(in_band_boot)

print("=== Method Comparison ===")
print(f"Analytical coverage: {coverage:.3f}")
print(f"Bootstrap coverage:  {coverage_boot:.3f}")
print(f"Expected coverage:   0.950")
```

### Residual Bootstrap

```python
# Residual bootstrap (resamples residuals)
y_pred_resid_boot, se_resid_boot = spline.predict(
    x_eval, 
    return_se=True, 
    se_method='bootstrap',
    bootstrap_method='residual',
    B_boot=500,
    n_jobs=2
)

# Compare all three methods
methods_data = [
    ('Analytical', se_analytical, 'red'),
    ('Parametric Bootstrap', se_bootstrap, 'blue'),
    ('Residual Bootstrap', se_resid_boot, 'green')
]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for method_name, se_values, color in methods_data:
    plt.plot(x_eval, se_values, color=color, linewidth=2, label=method_name)

plt.axhline(y=noise_std, color='black', linestyle='--', alpha=0.7, 
            label=f'True noise std ({noise_std})')
plt.xlabel('x')
plt.ylabel('Standard Error')
plt.title('Standard Error Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Show differences from analytical
plt.plot(x_eval, se_bootstrap - se_analytical, 'blue', linewidth=2, 
         label='Parametric - Analytical')
plt.plot(x_eval, se_resid_boot - se_analytical, 'green', linewidth=2, 
         label='Residual - Analytical')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('x')
plt.ylabel('SE Difference')
plt.title('Differences from Analytical Method')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compute correlation between methods
corr_param = np.corrcoef(se_analytical, se_bootstrap)[0, 1]
corr_resid = np.corrcoef(se_analytical, se_resid_boot)[0, 1]

print(f"Correlation with analytical method:")
print(f"  Parametric bootstrap: {corr_param:.4f}")
print(f"  Residual bootstrap:   {corr_resid:.4f}")
```

### Bootstrap Distribution Analysis

```python
# Analyze bootstrap distributions at specific points
eval_indices = [25, 50, 100, 150]  # Different x positions
eval_points = x_eval[eval_indices]

# Collect bootstrap samples for these points
n_boot = 200
bootstrap_samples = {i: [] for i in eval_indices}

print("Generating bootstrap samples...")
for b in range(n_boot):
    if b % 50 == 0:
        print(f"Bootstrap sample {b}/{n_boot}")
    
    # Generate bootstrap sample
    y_boot = spline.predict(x) + np.sqrt(spline.sigma2) * np.random.randn(len(x))
    
    # Fit new spline
    spline_boot = PSpline(x, y_boot, nseg=30, lambda_=optimal_lambda)
    spline_boot.fit()
    
    # Predict at evaluation points
    y_pred_boot = spline_boot.predict(eval_points)
    
    for i, idx in enumerate(eval_indices):
        bootstrap_samples[idx].append(y_pred_boot[i])

# Plot bootstrap distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, idx in enumerate(eval_indices):
    samples = np.array(bootstrap_samples[idx])
    
    # Plot histogram
    axes[i].hist(samples, bins=30, alpha=0.7, density=True, color='skyblue', 
                 edgecolor='black')
    
    # Add normal approximation
    mean_sample = np.mean(samples)
    std_sample = np.std(samples)
    
    x_norm = np.linspace(samples.min(), samples.max(), 100)
    y_norm = (1/np.sqrt(2*np.pi*std_sample**2)) * np.exp(-(x_norm - mean_sample)**2/(2*std_sample**2))
    axes[i].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal approx.')
    
    # Add true value and analytical prediction
    true_val = true_eval[idx]
    pred_val = y_pred_analytical[idx]
    
    axes[i].axvline(true_val, color='green', linestyle='--', linewidth=2, 
                   label=f'True ({true_val:.3f})')
    axes[i].axvline(pred_val, color='red', linestyle='-', linewidth=2, 
                   label=f'Predicted ({pred_val:.3f})')
    
    axes[i].set_title(f'Bootstrap Distribution at x={eval_points[i]:.2f}')
    axes[i].set_xlabel('Predicted Value')
    axes[i].set_ylabel('Density')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    print(f"Point x={eval_points[i]:.2f}: Bootstrap SE = {std_sample:.4f}, "
          f"Analytical SE = {se_analytical[idx]:.4f}")

plt.tight_layout()
plt.show()
```

## Method 3: Bayesian Inference

When PyMC is available, you can perform full Bayesian inference.

```python
# Check if Bayesian methods are available
try:
    import pymc as pm
    import arviz as az
    bayesian_available = True
    print("PyMC available - Bayesian methods enabled")
except ImportError:
    bayesian_available = False
    print("PyMC not available - skipping Bayesian methods")
    print("Install with: pip install pymc arviz")

if bayesian_available:
    # Bayesian P-spline fitting
    print("Performing Bayesian inference...")
    
    # Set up Bayesian model (this might take a moment)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = spline.bayes_fit(
            draws=1000,
            tune=1000, 
            chains=2,
            target_accept=0.95,
            return_inferencedata=True
        )
    
    # Get Bayesian predictions
    y_pred_bayes, se_bayes = spline.predict(
        x_eval, 
        return_se=True, 
        se_method='bayes',
        bayes_samples=trace
    )
    
    # Create credible intervals
    # Sample from posterior predictive
    n_samples = 500
    y_samples = []
    
    # Extract posterior samples of coefficients
    alpha_samples = trace.posterior['alpha'].values.reshape(-1, spline.nb)
    
    for i in range(min(n_samples, len(alpha_samples))):
        spline_temp = PSpline(x, y, nseg=30, lambda_=optimal_lambda)
        spline_temp.alpha = alpha_samples[i]
        y_sample = spline_temp.predict(x_eval)
        y_samples.append(y_sample)
    
    y_samples = np.array(y_samples)
    
    # Compute credible intervals
    lower_credible = np.percentile(y_samples, 2.5, axis=0)
    upper_credible = np.percentile(y_samples, 97.5, axis=0)
    
    # Plot Bayesian results
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, alpha=0.7, s=30, color='gray', label='Data')
    plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
    plt.plot(x_eval, y_pred_bayes, 'purple', linewidth=2, label='Bayesian fit')
    plt.fill_between(x_eval, lower_credible, upper_credible, alpha=0.3, 
                     color='purple', label='95% Credible Interval')
    plt.title('Bayesian P-Spline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare all uncertainty methods
    plt.subplot(2, 1, 2)
    plt.plot(x_eval, se_analytical, 'red', linewidth=2, label='Analytical')
    plt.plot(x_eval, se_bootstrap, 'blue', linewidth=2, label='Bootstrap')
    plt.plot(x_eval, se_bayes, 'purple', linewidth=2, label='Bayesian')
    plt.axhline(y=noise_std, color='black', linestyle='--', alpha=0.7, 
                label=f'True noise std ({noise_std})')
    plt.xlabel('x')
    plt.ylabel('Standard Error')
    plt.title('Uncertainty Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Bayesian model diagnostics
    print("=== Bayesian Diagnostics ===")
    print(az.summary(trace, var_names=['tau', 'sigma']))
    
    # Coverage analysis
    in_credible = (true_eval >= lower_credible) & (true_eval <= upper_credible)
    coverage_bayes = np.mean(in_credible)
    
    print(f"\n=== Coverage Comparison ===")
    print(f"Analytical (frequentist CI): {coverage:.3f}")
    print(f"Bootstrap (frequentist CI):  {coverage_boot:.3f}")
    print(f"Bayesian (credible interval): {coverage_bayes:.3f}")
    print(f"Expected coverage:           0.950")
    
else:
    print("Skipping Bayesian analysis - PyMC not available")
```

## Practical Guidance: When to Use Each Method

### Performance Comparison

```python
import time

# Time each method
methods_to_time = [
    ('Analytical', lambda: spline.predict(x_eval, return_se=True, se_method='analytic')),
    ('Bootstrap (100)', lambda: spline.predict(x_eval, return_se=True, se_method='bootstrap', 
                                             B_boot=100, n_jobs=1)),
    ('Bootstrap (500)', lambda: spline.predict(x_eval, return_se=True, se_method='bootstrap', 
                                             B_boot=500, n_jobs=1))
]

print("=== Performance Comparison ===")
print(f"{'Method':<20} {'Time (s)':<10} {'Relative Speed':<15}")
print("-" * 50)

times = {}
for method_name, method_func in methods_to_time:
    start_time = time.time()
    _, _ = method_func()
    end_time = time.time()
    elapsed = end_time - start_time
    times[method_name] = elapsed

# Calculate relative speeds
analytical_time = times['Analytical']
for method_name, elapsed in times.items():
    relative_speed = analytical_time / elapsed
    print(f"{method_name:<20} {elapsed:<10.4f} {relative_speed:<15.1f}x")

if bayesian_available:
    print(f"{'Bayesian':<20} {'~10-60s':<10} {'~0.01x':<15}")
```

### Decision Framework

```python
def recommend_uncertainty_method(n_points, computation_time='medium', 
                               distribution_assumptions='normal'):
    """
    Recommend uncertainty quantification method based on context.
    """
    recommendations = []
    
    print(f"=== Uncertainty Method Recommendations ===")
    print(f"Data points: {n_points}")
    print(f"Computation time preference: {computation_time}")
    print(f"Distribution assumptions: {distribution_assumptions}")
    print()
    
    if computation_time == 'fast':
        recommendations.append("✓ Use ANALYTICAL method")
        recommendations.append("  - Fastest option (seconds)")
        recommendations.append("  - Good approximation for well-behaved data")
        
    elif computation_time == 'medium':
        recommendations.append("✓ Use BOOTSTRAP method")
        recommendations.append("  - Good balance of accuracy and speed")
        recommendations.append("  - More robust than analytical")
        if n_points > 100:
            recommendations.append("  - Use parametric bootstrap for efficiency")
        else:
            recommendations.append("  - Use residual bootstrap for small samples")
            
    else:  # 'slow'
        if bayesian_available:
            recommendations.append("✓ Use BAYESIAN method")
            recommendations.append("  - Most comprehensive uncertainty quantification")
            recommendations.append("  - Provides full posterior distribution")
            recommendations.append("  - Best for critical applications")
        else:
            recommendations.append("✓ Use BOOTSTRAP method (Bayesian not available)")
            recommendations.append("  - Use high number of bootstrap samples (1000+)")
    
    # Additional considerations
    print("Primary recommendations:")
    for rec in recommendations:
        print(rec)
    
    print()
    print("Additional considerations:")
    
    if distribution_assumptions == 'non-normal':
        print("• Non-normal errors: Prefer bootstrap or Bayesian methods")
    
    if n_points < 50:
        print("• Small sample: Be cautious with bootstrap methods")
        print("• Small sample: Consider Bayesian approach with informative priors")
    
    if n_points > 1000:
        print("• Large sample: Analytical method often sufficient")
        print("• Large sample: Use parallel bootstrap if more accuracy needed")

# Example usage
recommend_uncertainty_method(len(x), 'medium', 'normal')
```

## Advanced Uncertainty Topics

### Simultaneous Confidence Bands

For simultaneous confidence over the entire curve:

```python
# Compute simultaneous confidence bands using bootstrap
# These are wider than pointwise bands
def simultaneous_confidence_bands(spline, x_eval, confidence_level=0.95, n_boot=500):
    """
    Compute simultaneous confidence bands that control family-wise error rate.
    """
    n_points = len(x_eval)
    bootstrap_curves = []
    
    for b in range(n_boot):
        # Generate bootstrap sample
        y_boot = spline.predict(spline.x) + np.sqrt(spline.sigma2) * np.random.randn(spline.n)
        
        # Fit bootstrap spline
        spline_boot = PSpline(spline.x, y_boot, nseg=spline.nseg, lambda_=spline.lambda_)
        spline_boot.fit()
        
        # Predict
        y_pred_boot = spline_boot.predict(x_eval)
        bootstrap_curves.append(y_pred_boot)
    
    bootstrap_curves = np.array(bootstrap_curves)
    
    # Compute simultaneous bands using max deviation approach
    y_pred_mean = np.mean(bootstrap_curves, axis=0)
    deviations = np.abs(bootstrap_curves - y_pred_mean[None, :])
    max_deviations = np.max(deviations, axis=1)
    
    # Find quantile that gives desired coverage
    alpha = 1 - confidence_level
    critical_value = np.percentile(max_deviations, 100 * (1 - alpha))
    
    # Create simultaneous bands
    lower_sim = y_pred_mean - critical_value
    upper_sim = y_pred_mean + critical_value
    
    return y_pred_mean, lower_sim, upper_sim, critical_value

# Compute simultaneous bands
y_pred_sim, lower_sim, upper_sim, crit_val = simultaneous_confidence_bands(
    spline, x_eval, confidence_level=0.95, n_boot=200
)

# Compare pointwise vs simultaneous
plt.figure(figsize=(14, 8))
plt.scatter(x, y, alpha=0.7, s=30, color='gray', label='Data')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
plt.plot(x_eval, y_pred_analytical, 'r-', linewidth=2, label='P-spline fit')

# Pointwise bands
plt.fill_between(x_eval, lower_band, upper_band, alpha=0.3, color='blue', 
                 label='95% Pointwise CI')

# Simultaneous bands
plt.fill_between(x_eval, lower_sim, upper_sim, alpha=0.2, color='red', 
                 label='95% Simultaneous CI')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Pointwise vs Simultaneous Confidence Bands')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Critical value for simultaneous bands: {crit_val:.4f}")
print(f"Average pointwise half-width: {np.mean(1.96 * se_analytical):.4f}")
print(f"Simultaneous half-width: {crit_val:.4f}")
print(f"Ratio (simultaneous/pointwise): {crit_val / np.mean(1.96 * se_analytical):.2f}")
```

### Prediction Intervals vs Confidence Intervals

```python
# Prediction intervals include both model uncertainty and noise
def prediction_intervals(spline, x_eval, confidence_level=0.95, method='analytical'):
    """
    Compute prediction intervals that account for both model and noise uncertainty.
    """
    # Get model uncertainty
    y_pred, se_model = spline.predict(x_eval, return_se=True, se_method=method)
    
    # Add noise uncertainty
    se_total = np.sqrt(se_model**2 + spline.sigma2)
    
    # Create prediction intervals
    alpha = 1 - confidence_level
    z_score = 1.96  # For 95% intervals
    
    lower_pi = y_pred - z_score * se_total
    upper_pi = y_pred + z_score * se_total
    
    return y_pred, lower_pi, upper_pi, se_model, se_total

# Compute prediction intervals
y_pred_pi, lower_pi, upper_pi, se_model_pi, se_total_pi = prediction_intervals(
    spline, x_eval, confidence_level=0.95
)

# Plot comparison
plt.figure(figsize=(14, 8))
plt.scatter(x, y, alpha=0.7, s=30, color='gray', label='Data')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
plt.plot(x_eval, y_pred_pi, 'r-', linewidth=2, label='P-spline fit')

# Confidence intervals (model uncertainty only)
plt.fill_between(x_eval, y_pred_pi - 1.96*se_model_pi, y_pred_pi + 1.96*se_model_pi, 
                 alpha=0.4, color='blue', label='95% Confidence Interval')

# Prediction intervals (model + noise uncertainty)
plt.fill_between(x_eval, lower_pi, upper_pi, alpha=0.2, color='red', 
                 label='95% Prediction Interval')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Confidence Intervals vs Prediction Intervals')
plt.grid(True, alpha=0.3)
plt.show()

print("=== Interval Comparison ===")
print(f"Average confidence interval width: {np.mean(2 * 1.96 * se_model_pi):.4f}")
print(f"Average prediction interval width: {np.mean(2 * 1.96 * se_total_pi):.4f}")
print(f"Model uncertainty contribution: {np.mean(se_model_pi**2):.6f}")
print(f"Noise uncertainty contribution: {spline.sigma2:.6f}")
print(f"Total uncertainty: {np.mean(se_total_pi**2):.6f}")
```

## Summary

This tutorial covered three main approaches to uncertainty quantification:

### Method Summary

| Method | Speed | Accuracy | Assumptions | Use Case |
|--------|-------|----------|-------------|----------|
| **Analytical** | ⚡⚡⚡ | Good | Normal errors | Quick assessment |
| **Bootstrap** | ⚡⚡ | Very Good | Minimal | General use |
| **Bayesian** | ⚡ | Excellent | Full model | Critical applications |

### Key Takeaways

1. **Analytical methods** are fastest and work well for well-behaved data
2. **Bootstrap methods** provide robust estimates with minimal assumptions
3. **Bayesian methods** offer the most comprehensive uncertainty quantification
4. **Choose based on**: computational budget, required accuracy, and application criticality
5. **Simultaneous confidence bands** are wider than pointwise bands
6. **Prediction intervals** are wider than confidence intervals (include noise)

### Recommendations

- **Exploratory analysis**: Use analytical standard errors
- **Production applications**: Use bootstrap methods
- **Critical decisions**: Consider Bayesian approaches
- **Multiple comparisons**: Use simultaneous confidence bands
- **Forecasting**: Use prediction intervals

## Next Steps

- **[Advanced Features](advanced-features.md)**: Constraints and specialized techniques  
- **[Parameter Selection](parameter-selection.md)**: Optimizing smoothing parameters
- **[Basic Usage](basic-usage.md)**: Review fundamental concepts
# Examples Gallery

This gallery showcases practical applications of PSplines across different domains and use cases.

## Overview

The examples demonstrate:
- **Basic Usage**: Core functionality and common patterns
- **Parameter Selection**: Automated optimization techniques  
- **Uncertainty Quantification**: Confidence intervals and bootstrap methods
- **Real-World Applications**: Time series, scientific data, and practical scenarios

All example code is available in the `/examples/` directory of the repository.

---

## 🚀 Quick Start Examples

### Example 1: Basic Smoothing
**File**: `01_basic_usage.py`

A comprehensive introduction to PSplines covering data generation, fitting, prediction, and visualization.

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline

# Generate noisy data
np.random.seed(42)
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x) + 0.1 * np.random.randn(50)

# Create and fit P-spline
spline = PSpline(x, y, nseg=15, lambda_=1.0)
spline.fit()

# Make predictions
x_new = np.linspace(0, 2*np.pi, 200)
y_pred = spline.predict(x_new)

# Plot results
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x_new, y_pred, 'r-', linewidth=2, label='P-spline')
plt.legend()
```

**Key Features Demonstrated**:
- Basic P-spline creation and fitting
- Parameter specification (`nseg`, `lambda_`)
- Prediction on new data points
- Visualization techniques

**Run this example**:
```bash
python examples/01_basic_usage.py
```

---

### Example 2: Parameter Optimization
**File**: `02_parameter_selection.py`

Demonstrates various methods for automatic parameter selection including cross-validation, AIC, and L-curve methods.

```python
from psplines.optimize import cross_validation, aic_selection, l_curve

# Automatic lambda selection using cross-validation
spline = PSpline(x, y, nseg=20)
optimal_lambda, cv_score = cross_validation(spline)
spline.lambda_ = optimal_lambda
spline.fit()

print(f"Optimal λ: {optimal_lambda:.6f}")
print(f"Effective DoF: {spline.ED:.2f}")
```

**Key Features Demonstrated**:
- Cross-validation optimization
- AIC-based parameter selection
- L-curve method
- Comparison of different selection criteria
- Performance vs accuracy trade-offs

**Run this example**:
```bash
python examples/02_parameter_selection.py
```

---

### Example 3: Uncertainty Quantification
**File**: `03_uncertainty_methods.py`

Comprehensive demonstration of uncertainty quantification methods including analytical standard errors, bootstrap methods, and Bayesian inference.

```python
# Get predictions with uncertainty estimates
y_pred, se = spline.predict(x_new, return_se=True, se_method='analytic')

# Create confidence bands
confidence_level = 0.95
z_score = 1.96
lower_band = y_pred - z_score * se
upper_band = y_pred + z_score * se

# Visualize with confidence intervals
plt.fill_between(x_new, lower_band, upper_band, alpha=0.3, label='95% CI')
```

**Key Features Demonstrated**:
- Analytical standard errors
- Bootstrap uncertainty quantification
- Bayesian inference (when PyMC available)
- Confidence vs prediction intervals
- Method comparison and validation

**Run this example**:
```bash
python examples/03_uncertainty_methods.py
```

---

### Example 4: Real-World Application
**File**: `04_real_world_application.py`

A complete workflow for analyzing real-world time series data with trend extraction, seasonal components, and forecasting.

**Key Features Demonstrated**:
- Data preprocessing and exploration
- Multi-component analysis (trend + seasonal)
- Model diagnostics and validation
- Practical forecasting workflow
- Performance optimization for larger datasets

**Run this example**:
```bash
python examples/04_real_world_application.py
```

---

## 📊 Domain-Specific Applications

### Scientific Data Analysis

#### Signal Processing
```python
# Smooth noisy measurements while preserving signal characteristics
def smooth_signal(t, signal, noise_level='auto'):
    """
    Smooth noisy signal data using P-splines.
    
    Parameters:
    - t: time points
    - signal: noisy signal values  
    - noise_level: 'auto' for automatic detection or numeric value
    """
    spline = PSpline(t, signal, nseg=min(len(t)//4, 50))
    
    if noise_level == 'auto':
        optimal_lambda, _ = cross_validation(spline)
        spline.lambda_ = optimal_lambda
    else:
        # Use noise level to guide smoothing
        spline.lambda_ = 1.0 / noise_level**2
    
    spline.fit()
    return spline

# Example usage
t = np.linspace(0, 10, 1000)
true_signal = np.sin(t) + 0.5*np.sin(3*t)
noisy_signal = true_signal + 0.1*np.random.randn(1000)

spline = smooth_signal(t, noisy_signal)
smooth_signal_values = spline.predict(t)
```

#### Experimental Data Fitting
```python
# Fit experimental data with derivative constraints
def fit_experimental_data(x_data, y_data, derivative_at_zero=None):
    """
    Fit experimental data with optional derivative constraints.
    """
    spline = PSpline(x_data, y_data, nseg=25)
    
    # Optimize smoothing parameter
    optimal_lambda, _ = cross_validation(spline, cv_method='kfold', k_folds=10)
    spline.lambda_ = optimal_lambda
    spline.fit()
    
    # Check derivative constraint if provided
    if derivative_at_zero is not None:
        computed_derivative = spline.derivative(np.array([0]), deriv_order=1)[0]
        print(f"Expected derivative at x=0: {derivative_at_zero}")
        print(f"Computed derivative at x=0: {computed_derivative:.4f}")
        print(f"Difference: {abs(computed_derivative - derivative_at_zero):.4f}")
    
    return spline

# Example with enzyme kinetics data
concentrations = np.array([0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
reaction_rates = np.array([0, 2.1, 3.8, 6.2, 9.1, 10.8, 11.9])
reaction_rates += 0.2 * np.random.randn(len(reaction_rates))  # Add noise

spline = fit_experimental_data(concentrations, reaction_rates, derivative_at_zero=0)
```

### Financial Time Series

#### Stock Price Analysis
```python
def analyze_stock_prices(dates, prices, return_components=False):
    """
    Analyze stock price data using P-splines for trend extraction.
    
    Returns trend component and optionally volatility estimates.
    """
    # Convert dates to numeric values for fitting
    x_numeric = np.arange(len(dates))
    
    # Log prices for multiplicative models
    log_prices = np.log(prices)
    
    # Fit trend with appropriate smoothing
    trend_spline = PSpline(x_numeric, log_prices, nseg=min(len(dates)//10, 100))
    optimal_lambda, _ = cross_validation(trend_spline)
    trend_spline.lambda_ = optimal_lambda
    trend_spline.fit()
    
    # Extract trend
    log_trend = trend_spline.predict(x_numeric)
    trend = np.exp(log_trend)
    
    # Compute residuals (detrended returns)
    residuals = log_prices - log_trend
    
    if return_components:
        # Estimate local volatility using absolute residuals
        abs_residuals = np.abs(residuals)
        vol_spline = PSpline(x_numeric, abs_residuals, nseg=min(len(dates)//5, 50))
        vol_lambda, _ = cross_validation(vol_spline)
        vol_spline.lambda_ = vol_lambda
        vol_spline.fit()
        
        volatility = vol_spline.predict(x_numeric)
        
        return {
            'trend': trend,
            'residuals': residuals,
            'volatility': volatility,
            'trend_spline': trend_spline,
            'vol_spline': vol_spline
        }
    
    return trend, residuals

# Example usage
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
# Simulate stock prices (in practice, load real data)
log_returns = 0.0008 + 0.02 * np.random.randn(len(dates))
log_prices = np.cumsum(log_returns)
prices = 100 * np.exp(log_prices)

results = analyze_stock_prices(dates, prices, return_components=True)
```

### Biomedical Applications

#### Growth Curve Analysis
```python
def analyze_growth_curves(time_points, measurements, subject_ids=None):
    """
    Analyze biological growth curves with P-splines.
    
    Handles individual and population-level analysis.
    """
    results = {}
    
    if subject_ids is None:
        # Single growth curve
        spline = PSpline(time_points, measurements, nseg=20)
        optimal_lambda, _ = cross_validation(spline)
        spline.lambda_ = optimal_lambda
        spline.fit()
        
        # Compute growth rate (first derivative)
        growth_rate = spline.derivative(time_points, deriv_order=1)
        
        results['spline'] = spline
        results['growth_rate'] = growth_rate
        
    else:
        # Multiple subjects
        unique_subjects = np.unique(subject_ids)
        individual_results = {}
        
        for subject in unique_subjects:
            mask = subject_ids == subject
            t_subj = time_points[mask]
            y_subj = measurements[mask]
            
            if len(t_subj) > 5:  # Minimum points for fitting
                spline = PSpline(t_subj, y_subj, nseg=min(15, len(t_subj)//3))
                optimal_lambda, _ = cross_validation(spline, n_lambda=20)
                spline.lambda_ = optimal_lambda
                spline.fit()
                
                individual_results[subject] = {
                    'spline': spline,
                    'time': t_subj,
                    'measurements': y_subj
                }
        
        results['individual'] = individual_results
        
        # Population average
        if len(individual_results) > 1:
            # Create common time grid
            t_common = np.linspace(time_points.min(), time_points.max(), 100)
            
            # Evaluate each individual curve on common grid
            individual_curves = []
            for subj_data in individual_results.values():
                try:
                    curve = subj_data['spline'].predict(t_common)
                    individual_curves.append(curve)
                except:
                    continue
            
            if individual_curves:
                individual_curves = np.array(individual_curves)
                
                # Fit spline to mean curve
                mean_curve = np.mean(individual_curves, axis=0)
                population_spline = PSpline(t_common, mean_curve, nseg=25)
                pop_lambda, _ = cross_validation(population_spline)
                population_spline.lambda_ = pop_lambda
                population_spline.fit()
                
                results['population'] = {
                    'spline': population_spline,
                    'mean_curve': mean_curve,
                    'individual_curves': individual_curves,
                    'time_grid': t_common
                }
    
    return results

# Example: Bacterial growth data
time_hours = np.array([0, 2, 4, 6, 8, 12, 16, 20, 24])
optical_density = np.array([0.1, 0.15, 0.25, 0.45, 0.8, 1.2, 1.4, 1.45, 1.5])
optical_density += 0.05 * np.random.randn(len(optical_density))  # Measurement noise

growth_analysis = analyze_growth_curves(time_hours, optical_density)
```

## 🛠 Advanced Techniques

### Memory-Efficient Processing
```python
def process_large_dataset(x, y, max_points=5000):
    """
    Memory-efficient P-spline fitting for large datasets.
    """
    n = len(x)
    
    if n <= max_points:
        # Standard approach
        spline = PSpline(x, y, nseg=min(50, n//10))
        optimal_lambda, _ = cross_validation(spline)
        spline.lambda_ = optimal_lambda
        spline.fit()
        return spline
    
    else:
        # Subsample for parameter estimation
        subsample_idx = np.random.choice(n, max_points, replace=False)
        x_sub = x[subsample_idx]
        y_sub = y[subsample_idx]
        
        # Find optimal parameters on subsample
        spline_sub = PSpline(x_sub, y_sub, nseg=40)
        optimal_lambda, _ = cross_validation(spline_sub)
        
        # Apply to full dataset with found parameters
        spline_full = PSpline(x, y, nseg=min(60, n//50), lambda_=optimal_lambda)
        spline_full.fit()
        
        return spline_full
```

### Parallel Processing
```python
def parallel_bootstrap_uncertainty(spline, x_eval, n_boot=1000, n_jobs=-1):
    """
    Compute bootstrap uncertainty estimates in parallel.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        print("joblib not available, using sequential processing")
        n_jobs = 1
    
    def bootstrap_sample(i):
        # Generate bootstrap sample
        n = spline.n
        bootstrap_y = (spline.predict(spline.x) + 
                      np.sqrt(spline.sigma2) * np.random.randn(n))
        
        # Fit bootstrap spline
        boot_spline = PSpline(spline.x, bootstrap_y, 
                             nseg=spline.nseg, lambda_=spline.lambda_)
        boot_spline.fit()
        
        # Predict at evaluation points
        return boot_spline.predict(x_eval)
    
    if n_jobs == 1:
        # Sequential
        bootstrap_predictions = [bootstrap_sample(i) for i in range(n_boot)]
    else:
        # Parallel
        bootstrap_predictions = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_sample)(i) for i in range(n_boot)
        )
    
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Compute statistics
    mean_pred = np.mean(bootstrap_predictions, axis=0)
    std_pred = np.std(bootstrap_predictions, axis=0)
    
    # Confidence intervals
    lower_ci = np.percentile(bootstrap_predictions, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_predictions, 97.5, axis=0)
    
    return {
        'mean': mean_pred,
        'std': std_pred,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'samples': bootstrap_predictions
    }
```

## 🔍 Model Diagnostics

### Residual Analysis
```python
def comprehensive_diagnostics(spline, x_data, y_data):
    """
    Perform comprehensive diagnostic analysis of P-spline fit.
    """
    # Basic predictions and residuals
    y_pred = spline.predict(x_data)
    residuals = y_data - y_pred
    
    # Standardized residuals
    residual_std = np.std(residuals)
    standardized_residuals = residuals / residual_std
    
    # Diagnostic statistics
    diagnostics = {
        'mse': np.mean(residuals**2),
        'mae': np.mean(np.abs(residuals)),
        'r_squared': 1 - np.var(residuals) / np.var(y_data),
        'effective_dof': spline.ED,
        'aic': spline.n * np.log(np.mean(residuals**2)) + 2 * spline.ED,
        'residual_autocorr': np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    }
    
    # Normality test (Shapiro-Wilk)
    try:
        from scipy.stats import shapiro
        _, p_value_normality = shapiro(standardized_residuals)
        diagnostics['normality_p'] = p_value_normality
    except:
        pass
    
    # Outlier detection (using IQR method)
    Q1, Q3 = np.percentile(standardized_residuals, [25, 75])
    IQR = Q3 - Q1
    outlier_threshold = 1.5 * IQR
    outliers = np.abs(standardized_residuals) > (Q3 + outlier_threshold)
    diagnostics['n_outliers'] = np.sum(outliers)
    diagnostics['outlier_fraction'] = np.sum(outliers) / len(residuals)
    
    return diagnostics, residuals, standardized_residuals

# Example usage in model validation
def validate_model(spline, x_data, y_data, plot=True):
    """
    Validate P-spline model with diagnostic plots and statistics.
    """
    diagnostics, residuals, std_residuals = comprehensive_diagnostics(
        spline, x_data, y_data
    )
    
    print("=== Model Diagnostics ===")
    print(f"R² = {diagnostics['r_squared']:.4f}")
    print(f"MSE = {diagnostics['mse']:.6f}")
    print(f"MAE = {diagnostics['mae']:.6f}")
    print(f"Effective DoF = {diagnostics['effective_dof']:.2f}")
    print(f"AIC = {diagnostics['aic']:.2f}")
    print(f"Outliers: {diagnostics['n_outliers']} ({diagnostics['outlier_fraction']:.1%})")
    print(f"Residual autocorrelation = {diagnostics['residual_autocorr']:.4f}")
    
    if 'normality_p' in diagnostics:
        print(f"Normality test p-value = {diagnostics['normality_p']:.4f}")
        if diagnostics['normality_p'] < 0.05:
            print("⚠️  Warning: Residuals may not be normally distributed")
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs fitted
        y_pred = spline.predict(x_data)
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # QQ plot for normality
        axes[0, 1].hist(std_residuals, bins=20, density=True, alpha=0.7)
        x_norm = np.linspace(std_residuals.min(), std_residuals.max(), 100)
        y_norm = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_norm**2)
        axes[0, 1].plot(x_norm, y_norm, 'r-', linewidth=2, label='Standard Normal')
        axes[0, 1].set_xlabel('Standardized Residuals')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals vs order (for autocorrelation)
        axes[1, 0].plot(residuals, 'o-', alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Observation Order')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Order')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scale-location plot
        sqrt_abs_residuals = np.sqrt(np.abs(std_residuals))
        axes[1, 1].scatter(y_pred, sqrt_abs_residuals, alpha=0.6)
        axes[1, 1].set_xlabel('Fitted Values')
        axes[1, 1].set_ylabel('√|Standardized Residuals|')
        axes[1, 1].set_title('Scale-Location Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return diagnostics
```

## 📚 Running the Examples

### Prerequisites
Make sure you have PSplines installed with all dependencies:
```bash
pip install psplines[full]
```

Or for development:
```bash
git clone https://github.com/graysonbellamy/psplines.git
cd psplines
uv sync --dev
```

### Running Individual Examples
```bash
# Basic usage
python examples/01_basic_usage.py

# Parameter selection comparison
python examples/02_parameter_selection.py

# Uncertainty quantification methods
python examples/03_uncertainty_methods.py

# Real-world application workflow
python examples/04_real_world_application.py
```

### Interactive Exploration
For interactive exploration, use Jupyter notebooks:
```bash
jupyter notebook examples/
```

## 🎯 Best Practices

Based on these examples, here are key best practices:

### 1. Parameter Selection
- **Default**: Use cross-validation for robust parameter selection
- **Fast prototyping**: Start with `lambda_=1.0` and `nseg=20`
- **Large datasets**: Use AIC or reduce `n_lambda` in CV for speed
- **Critical applications**: Compare multiple methods (CV, AIC, L-curve)

### 2. Model Validation
- **Always** check residual plots for patterns
- **Use** diagnostic statistics (R², AIC, effective DoF)
- **Test** on held-out data when possible
- **Consider** bootstrap validation for small datasets

### 3. Computational Efficiency
- **Memory**: Use fewer segments for very large datasets
- **Speed**: Enable parallel processing (`n_jobs=-1`) for bootstrap
- **Storage**: Consider subsampling for datasets > 10,000 points

### 4. Domain-Specific Considerations
- **Time series**: Check for autocorrelation in residuals
- **Scientific data**: Validate derivatives match physical expectations
- **Financial data**: Use log-prices for multiplicative models
- **Biomedical**: Consider growth rate constraints and biological limits

## 📖 Additional Resources

- **[API Reference](../api/core.md)**: Complete function documentation
- **[Tutorials](../tutorials/basic-usage.md)**: Step-by-step guides
- **[Theory](../theory/mathematical-background.md)**: Mathematical foundations
- **[GitHub Issues](https://github.com/graysonbellamy/psplines/issues)**: Report bugs or request features

---

*All examples are provided under the same license as the PSplines package and include both synthetic and real-world applications to help you get started quickly.*
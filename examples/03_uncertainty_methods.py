#!/usr/bin/env python3
"""
Example 3: Uncertainty Quantification Methods
=============================================

This example demonstrates different methods for quantifying uncertainty in P-spline fits:
1. Analytical standard errors (delta method)
2. Parametric bootstrap 
3. Bayesian credible intervals
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from psplines import PSpline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def generate_test_data(n_points=60, seed=42):
    """Generate test data with known uncertainty structure."""
    np.random.seed(seed)
    x = np.linspace(0, 8, n_points)
    
    # True function: damped oscillation
    y_true = 2 * np.exp(-x/4) * np.sin(2*x) + 0.5
    
    # Heteroscedastic noise (increasing with x)
    noise_std = 0.1 + 0.05 * x/8
    y_noisy = y_true + noise_std * np.random.randn(n_points)
    
    return x, y_noisy, y_true, noise_std

def compare_uncertainty_methods(spline, x_pred, n_bootstrap=1000):
    """Compare different uncertainty quantification methods."""
    print("Comparing uncertainty methods...")
    results = {}
    
    # 1. Analytical standard errors (delta method)
    print("  - Computing analytical standard errors...")
    try:
        y_pred_analytical, se_analytical = spline.predict(x_pred, return_se=True, se_method="analytic")
        results['analytical'] = (y_pred_analytical, se_analytical, None)
        print(f"    ✓ Analytical method completed")
    except Exception as e:
        print(f"    ✗ Analytical method failed: {e}")
        results['analytical'] = None
    
    # 2. Parametric bootstrap
    print(f"  - Computing bootstrap standard errors ({n_bootstrap} replicates)...")
    try:
        y_pred_bootstrap, se_bootstrap = spline.predict(
            x_pred, return_se=True, se_method="bootstrap", 
            B_boot=n_bootstrap, n_jobs=1, seed=42
        )
        results['bootstrap'] = (y_pred_bootstrap, se_bootstrap, None)
        print(f"    ✓ Bootstrap method completed")
    except Exception as e:
        print(f"    ✗ Bootstrap method failed: {e}")
        results['bootstrap'] = None
    
    # 3. Bayesian credible intervals
    print("  - Computing Bayesian credible intervals...")
    try:
        # First fit Bayesian model (this may take a while)
        print("    Sampling posterior (this may take 30-60 seconds)...")
        trace = spline.bayes_fit(draws=1000, tune=500, chains=2, cores=1)
        
        mean_bayes, lower_bayes, upper_bayes = spline.predict(
            x_pred, se_method="bayes", hdi_prob=0.95
        )
        
        # Convert to standard error approximation
        se_bayes = (upper_bayes - lower_bayes) / (2 * 1.96)  # Approximate
        results['bayesian'] = (mean_bayes, se_bayes, (lower_bayes, upper_bayes))
        print(f"    ✓ Bayesian method completed")
        
    except Exception as e:
        print(f"    ✗ Bayesian method failed: {e}")
        results['bayesian'] = None
    
    return results

def main():
    print("PSplines Uncertainty Quantification Example")
    print("=" * 50)
    
    # Generate test data
    print("1. Generating test data with heteroscedastic noise...")
    x, y_noisy, y_true, noise_std = generate_test_data()
    print(f"   Generated {len(x)} data points with varying noise levels")
    
    # Create and fit spline
    print("\n2. Fitting P-spline model...")
    spline = PSpline(x, y_noisy, nseg=12, lambda_=0.5)
    spline.fit()
    print(f"   Fitted with {spline.nseg} segments")
    print(f"   Effective degrees of freedom: {spline.ED:.2f}")
    
    # Prediction points
    x_pred = np.linspace(x.min(), x.max(), 100)
    
    # Compare uncertainty methods
    print("\n3. Comparing uncertainty quantification methods...")
    results = compare_uncertainty_methods(spline, x_pred, n_bootstrap=500)
    
    # Create comprehensive comparison plot
    print("\n4. Creating comparison plots...")
    
    n_methods = sum(1 for r in results.values() if r is not None)
    if n_methods == 0:
        print("   No methods succeeded - creating basic plot...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.scatter(x, y_noisy, alpha=0.6, label='Data')
        y_basic = spline.predict(x_pred)
        ax.plot(x_pred, y_basic, 'r-', label='P-spline fit')
        ax.set_title('Basic P-spline Fit (Uncertainty Methods Failed)')
        ax.legend()
        plt.show()
        return
    
    fig = plt.figure(figsize=(18, 12))
    
    # Main comparison plot
    ax_main = plt.subplot(2, 3, (1, 2))
    
    # Plot data and true function
    plt.scatter(x, y_noisy, alpha=0.6, s=40, color='gray', label='Noisy data', zorder=5)
    plt.plot(x_pred, np.interp(x_pred, x, y_true), 'g-', linewidth=2, 
            label='True function', alpha=0.8, zorder=3)
    
    # Plot different uncertainty methods
    colors = {'analytical': 'blue', 'bootstrap': 'red', 'bayesian': 'purple'}
    alphas = {'analytical': 0.3, 'bootstrap': 0.2, 'bayesian': 0.25}
    
    for method, result in results.items():
        if result is None:
            continue
            
        y_pred, se, bounds = result
        color = colors[method]
        alpha = alphas[method]
        
        # Plot mean prediction
        plt.plot(x_pred, y_pred, color=color, linewidth=2, 
                label=f'{method.capitalize()} mean', zorder=4)
        
        # Plot uncertainty bands
        if bounds is not None:  # Bayesian case with credible intervals
            lower, upper = bounds
            plt.fill_between(x_pred, lower, upper, alpha=alpha, color=color,
                           label=f'{method.capitalize()} 95% CI', zorder=1)
        else:  # Analytical or bootstrap with standard errors
            plt.fill_between(x_pred, y_pred - 1.96*se, y_pred + 1.96*se, 
                           alpha=alpha, color=color,
                           label=f'{method.capitalize()} 95% CI', zorder=1)
    
    plt.title('Uncertainty Quantification Methods Comparison', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Standard error comparison plot
    if len([r for r in results.values() if r is not None]) >= 2:
        ax_se = plt.subplot(2, 3, 3)
        
        for method, result in results.items():
            if result is None:
                continue
            _, se, bounds = result
            
            if bounds is not None:  # Convert Bayesian bounds to SE
                lower, upper = bounds
                se = (upper - lower) / (2 * 1.96)
                
            plt.plot(x_pred, se, color=colors[method], linewidth=2, 
                    alpha=0.8, label=f'{method.capitalize()}')
        
        plt.title('Standard Error Comparison')
        plt.xlabel('x')
        plt.ylabel('Standard Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Coverage analysis (if we have multiple methods)
    if len([r for r in results.values() if r is not None]) >= 2:
        ax_coverage = plt.subplot(2, 3, 4)
        
        # Approximate coverage by checking how often true function is within CI
        y_true_pred = np.interp(x_pred, x, y_true)
        coverage_results = {}
        
        for method, result in results.items():
            if result is None:
                continue
            y_pred, se, bounds = result
            
            if bounds is not None:
                lower, upper = bounds
            else:
                lower = y_pred - 1.96 * se
                upper = y_pred + 1.96 * se
            
            # Calculate coverage (proportion of true values within CI)
            coverage = np.mean((y_true_pred >= lower) & (y_true_pred <= upper))
            coverage_results[method] = coverage
        
        methods = list(coverage_results.keys())
        coverages = list(coverage_results.values())
        bars = plt.bar(methods, coverages, color=[colors[m] for m in methods], alpha=0.7)
        plt.axhline(y=0.95, color='red', linestyle='--', label='Nominal 95%')
        plt.title('Coverage Analysis')
        plt.ylabel('Empirical Coverage')
        plt.ylim(0, 1)
        
        # Add percentage labels on bars
        for bar, cov in zip(bars, coverages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{cov:.1%}', ha='center', va='bottom')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Residuals analysis
    ax_resid = plt.subplot(2, 3, 5)
    
    # Use analytical method for residuals if available, otherwise first available
    method_for_residuals = 'analytical' if results.get('analytical') else list(results.keys())[0]
    if results[method_for_residuals]:
        y_fitted = spline.fitted_values
        residuals = y_noisy - y_fitted
        
        plt.scatter(y_fitted, residuals, alpha=0.6, s=40)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.title('Residuals vs Fitted')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
    
    # QQ plot for residuals normality check
    ax_qq = plt.subplot(2, 3, 6)
    
    if results.get('analytical'):
        residuals = y_noisy - spline.fitted_values
        standardized_residuals = residuals / np.sqrt(spline.sigma2)
        
        # Simple QQ plot
        sorted_residuals = np.sort(standardized_residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = np.linspace(-2.5, 2.5, n)
        
        plt.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)
        plt.plot([-3, 3], [-3, 3], 'r--', alpha=0.8, label='Perfect normal')
        plt.title('Q-Q Plot (Normality Check)')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/uncertainty_methods_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n5. Summary Statistics:")
    print(f"   Model fit:")
    print(f"     - Effective DoF: {spline.ED:.2f}")
    print(f"     - Residual variance: {spline.sigma2:.6f}")
    print(f"     - RMSE: {np.sqrt(np.mean((y_noisy - spline.fitted_values)**2)):.4f}")
    
    print(f"\n   Uncertainty method comparison:")
    for method, result in results.items():
        if result is None:
            print(f"     - {method.capitalize()}: Failed")
        else:
            y_pred, se, bounds = result
            avg_se = np.mean(se)
            print(f"     - {method.capitalize()}: Average SE = {avg_se:.4f}")
            
            # Coverage if we can compute it
            if len(x_pred) == len(x):
                y_true_pred = y_true
            else:
                y_true_pred = np.interp(x_pred, x, y_true)
                
            if bounds is not None:
                lower, upper = bounds
            else:
                lower = y_pred - 1.96 * se
                upper = y_pred + 1.96 * se
                
            coverage = np.mean((y_true_pred >= lower) & (y_true_pred <= upper))
            print(f"       Coverage: {coverage:.1%}")
    
    print("\n✓ Uncertainty quantification example completed!")
    print("  Plot saved as 'examples/uncertainty_methods_output.png'")

if __name__ == "__main__":
    main()
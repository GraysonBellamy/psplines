#!/usr/bin/env python3
"""
Example 2: Automatic Parameter Selection
========================================

This example demonstrates different methods for automatically selecting the optimal
smoothing parameter (lambda) for P-splines.
"""

import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline
from psplines.optimize import cross_validation, aic, l_curve, v_curve

def generate_test_data(n_points=80, seed=123):
    """Generate test data with moderate noise."""
    np.random.seed(seed)
    x = np.linspace(0, 10, n_points)
    y_true = np.sin(x) * np.exp(-x/8) + 0.5 * np.cos(2*x) * np.exp(-x/10)
    y_noisy = y_true + 0.15 * np.random.randn(n_points)
    return x, y_noisy, y_true

def compare_smoothing_parameters(spline, lambda_range):
    """Compare different smoothing parameters."""
    results = []
    
    for lam in lambda_range:
        # Create temporary spline with this lambda
        temp_spline = PSpline(spline.x, spline.y, nseg=spline.nseg, 
                             lambda_=lam, degree=spline.degree)
        temp_spline.fit()
        
        # Calculate metrics
        y_fitted = temp_spline.fitted_values
        mse = np.mean((spline.y - y_fitted)**2)
        ed = temp_spline.ED
        
        results.append({
            'lambda': lam,
            'mse': mse,
            'ed': ed,
            'fitted_values': y_fitted
        })
    
    return results

def main():
    print("PSplines Parameter Selection Example")
    print("=" * 45)
    
    # Generate test data
    print("1. Generating test data...")
    x, y_noisy, y_true = generate_test_data()
    spline = PSpline(x, y_noisy, nseg=15)
    print(f"   Generated {len(x)} data points")
    
    # Test different automatic selection methods
    print("\n2. Testing automatic parameter selection methods...")
    
    methods_results = {}
    
    try:
        print("   - Cross-validation...")
        lambda_cv, cv_score = cross_validation(spline)
        methods_results['Cross-validation'] = lambda_cv
        print(f"     Optimal λ = {lambda_cv:.6f}, Score = {cv_score:.6f}")
    except Exception as e:
        print(f"     Cross-validation failed: {e}")
        methods_results['Cross-validation'] = None
    
    try:
        print("   - AIC...")
        lambda_aic, aic_score = aic(spline)
        methods_results['AIC'] = lambda_aic
        print(f"     Optimal λ = {lambda_aic:.6f}, Score = {aic_score:.6f}")
    except Exception as e:
        print(f"     AIC failed: {e}")
        methods_results['AIC'] = None
    
    try:
        print("   - L-curve...")
        lambda_l, l_score = l_curve(spline)
        methods_results['L-curve'] = lambda_l
        print(f"     Optimal λ = {lambda_l:.6f}, Curvature = {l_score:.6f}")
    except Exception as e:
        print(f"     L-curve failed: {e}")
        methods_results['L-curve'] = None
        
    try:
        print("   - V-curve...")
        lambda_v, v_score = v_curve(spline)
        methods_results['V-curve'] = lambda_v
        print(f"     Optimal λ = {lambda_v:.6f}, Score = {v_score:.6f}")
    except Exception as e:
        print(f"     V-curve failed: {e}")
        methods_results['V-curve'] = None
    
    # Use cross-validation result (most reliable) or fallback to reasonable default
    best_lambda = methods_results.get('Cross-validation') or 1.0
    print(f"\n3. Using λ = {best_lambda:.6f} for final fit...")
    
    # Fit final model
    spline.lambda_ = best_lambda
    spline.fit()
    
    # Compare with different lambda values
    print("\n4. Comparing different smoothing levels...")
    lambda_range = np.logspace(-3, 2, 20)
    comparison_results = compare_smoothing_parameters(spline, lambda_range)
    
    # Make predictions
    x_pred = np.linspace(x.min(), x.max(), 200)
    y_pred = spline.predict(x_pred)
    
    # Create plots
    print("\n5. Creating comparison plots...")
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Different smoothing levels
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(x, y_noisy, alpha=0.6, s=30, color='gray', label='Noisy data')
    
    # Show a few different smoothing levels
    demo_lambdas = [0.001, 0.1, 1.0, 100.0]
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, lam in enumerate(demo_lambdas):
        demo_spline = PSpline(x, y_noisy, lambda_=lam, nseg=15)
        demo_spline.fit()
        y_demo = demo_spline.predict(x_pred)
        plt.plot(x_pred, y_demo, color=colors[i], alpha=0.8, 
                label=f'λ = {lam}', linewidth=2)
    
    plt.title('Effect of Different λ Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: MSE vs Lambda
    ax2 = plt.subplot(2, 3, 2)
    lambdas = [r['lambda'] for r in comparison_results]
    mses = [r['mse'] for r in comparison_results]
    plt.semilogx(lambdas, mses, 'bo-', alpha=0.7)
    plt.axvline(best_lambda, color='red', linestyle='--', alpha=0.8, label=f'Selected λ = {best_lambda:.6f}')
    plt.title('MSE vs Smoothing Parameter')
    plt.xlabel('λ (log scale)')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Effective DoF vs Lambda
    ax3 = plt.subplot(2, 3, 3)
    eds = [r['ed'] for r in comparison_results]
    plt.semilogx(lambdas, eds, 'go-', alpha=0.7)
    plt.axvline(best_lambda, color='red', linestyle='--', alpha=0.8, label=f'Selected λ')
    plt.title('Effective DoF vs λ')
    plt.xlabel('λ (log scale)')
    plt.ylabel('Effective Degrees of Freedom')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Final fit with uncertainty
    ax4 = plt.subplot(2, 3, 4)
    y_pred_se, se = spline.predict(x_pred, return_se=True)
    plt.scatter(x, y_noisy, alpha=0.6, s=30, color='gray', label='Data')
    plt.plot(x_pred, y_pred, 'r-', linewidth=2, label=f'P-spline (λ={best_lambda:.6f})')
    plt.fill_between(x_pred, y_pred - 1.96*se, y_pred + 1.96*se, 
                    alpha=0.3, color='red', label='95% CI')
    if len(y_true) == len(x):
        y_true_interp = np.interp(x_pred, x, y_true)
        plt.plot(x_pred, y_true_interp, 'g--', alpha=0.7, label='True function')
    plt.title('Optimal Fit with Uncertainty')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Method comparison
    ax5 = plt.subplot(2, 3, 5)
    valid_methods = {k: v for k, v in methods_results.items() if v is not None}
    if valid_methods:
        methods = list(valid_methods.keys())
        lambdas_method = list(valid_methods.values())
        colors_method = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        bars = plt.bar(methods, lambdas_method, color=colors_method, alpha=0.7)
        plt.yscale('log')
        plt.title('Optimal λ by Method')
        plt.ylabel('Optimal λ (log scale)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, lamb in zip(bars, lambdas_method):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{lamb:.6f}', ha='center', va='bottom', fontsize=8)
    else:
        plt.text(0.5, 0.5, 'No methods succeeded', ha='center', va='center', transform=ax5.transAxes)
        plt.title('Method Comparison - Failed')
    
    # Plot 6: Bias-variance tradeoff illustration
    ax6 = plt.subplot(2, 3, 6)
    plt.semilogx(lambdas, mses, 'o-', alpha=0.7, label='Total MSE')
    
    # Approximate bias^2 and variance components
    # This is a simplified illustration
    min_mse = min(mses)
    bias_approx = [(mse - min_mse) * 0.7 + 0.001 for mse in mses]  # Simplified bias approximation
    var_approx = [min_mse - b + 0.02/lam for lam, b in zip(lambdas, bias_approx)]  # Simplified variance
    
    plt.semilogx(lambdas, bias_approx, 's-', alpha=0.6, label='Bias² (approx)')
    plt.semilogx(lambdas, var_approx, '^-', alpha=0.6, label='Variance (approx)')
    plt.axvline(best_lambda, color='red', linestyle='--', alpha=0.8, label='Selected λ')
    plt.title('Bias-Variance Tradeoff')
    plt.xlabel('λ (log scale)')
    plt.ylabel('Error Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/parameter_selection_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n6. Summary:")
    print(f"   Selected smoothing parameter: λ = {best_lambda:.6f}")
    print(f"   Effective degrees of freedom: {spline.ED:.2f}")
    print(f"   Final MSE: {np.mean((y_noisy - spline.fitted_values)**2):.6f}")
    print(f"   R²: {1 - np.sum((y_noisy - spline.fitted_values)**2) / np.sum((y_noisy - np.mean(y_noisy))**2):.4f}")
    
    print("\n✓ Parameter selection example completed!")
    print("  Plot saved as 'examples/parameter_selection_output.png'")

if __name__ == "__main__":
    main()
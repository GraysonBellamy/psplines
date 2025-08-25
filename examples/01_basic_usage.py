#!/usr/bin/env python3
"""
Example 1: Basic PSpline Usage
=============================

This example demonstrates the basic usage of PSplines for smoothing noisy data.
"""

import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline

def generate_test_data(n_points=100, noise_level=0.1, seed=42):
    """Generate noisy sine wave data for testing."""
    np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, n_points)
    y_true = np.sin(x)
    y_noisy = y_true + noise_level * np.random.randn(n_points)
    return x, y_noisy, y_true

def main():
    print("PSplines Basic Usage Example")
    print("=" * 40)
    
    # Generate test data
    print("1. Generating noisy sine wave data...")
    x, y_noisy, y_true = generate_test_data()
    print(f"   Generated {len(x)} data points")
    
    # Create and fit P-spline
    print("\n2. Creating and fitting P-spline...")
    spline = PSpline(x, y_noisy, nseg=20, lambda_=1.0, degree=3)
    spline.fit()
    print(f"   Fitted with {spline.nseg} segments")
    print(f"   Effective degrees of freedom: {spline.ED:.2f}")
    print(f"   Residual variance: {spline.sigma2:.6f}")
    
    # Make predictions
    print("\n3. Making predictions...")
    x_pred = np.linspace(0, 2*np.pi, 200)
    y_pred = spline.predict(x_pred)
    print(f"   Made predictions at {len(x_pred)} points")
    
    # Predictions with uncertainty
    print("\n4. Computing uncertainty estimates...")
    y_pred_se, se = spline.predict(x_pred, return_se=True)
    print(f"   Computed analytical standard errors")
    
    # Compute derivatives
    print("\n5. Computing derivatives...")
    dy_dx = spline.derivative(x_pred, deriv_order=1)
    d2y_dx2 = spline.derivative(x_pred, deriv_order=2)
    print(f"   Computed 1st and 2nd derivatives")
    
    # Calculate fit quality metrics
    y_fitted = spline.fitted_values
    mse = np.mean((y_noisy - y_fitted)**2)
    r_squared = 1 - np.sum((y_noisy - y_fitted)**2) / np.sum((y_noisy - np.mean(y_noisy))**2)
    
    print(f"\n6. Fit Quality:")
    print(f"   Mean Squared Error: {mse:.6f}")
    print(f"   R-squared: {r_squared:.4f}")
    
    # Plotting
    print("\n7. Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PSplines Basic Usage Example', fontsize=16)
    
    # Main fit plot
    axes[0, 0].scatter(x, y_noisy, alpha=0.5, s=20, label='Noisy data')
    axes[0, 0].plot(x_pred, y_true[::2] if len(y_true) == len(x_pred) else np.sin(x_pred), 
                   'g-', label='True function', linewidth=1)
    axes[0, 0].plot(x_pred, y_pred, 'r-', label='P-spline fit', linewidth=2)
    axes[0, 0].fill_between(x_pred, y_pred - 1.96*se, y_pred + 1.96*se, 
                           alpha=0.3, color='red', label='95% CI')
    axes[0, 0].set_title('Data and Fit with Uncertainty')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_noisy - y_fitted
    axes[0, 1].scatter(y_fitted, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 1].set_title('Residuals vs Fitted Values')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # First derivative
    axes[1, 0].plot(x_pred, dy_dx, 'b-', linewidth=2, label='1st derivative')
    axes[1, 0].plot(x_pred, np.cos(x_pred), 'g--', alpha=0.7, label='True derivative')
    axes[1, 0].set_title('First Derivative')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel("dy/dx")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Second derivative
    axes[1, 1].plot(x_pred, d2y_dx2, 'm-', linewidth=2, label='2nd derivative')
    axes[1, 1].plot(x_pred, -np.sin(x_pred), 'g--', alpha=0.7, label='True 2nd derivative')
    axes[1, 1].set_title('Second Derivative')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel("d²y/dx²")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/basic_usage_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Example completed successfully!")
    print("  Plot saved as 'examples/basic_usage_output.png'")

if __name__ == "__main__":
    main()
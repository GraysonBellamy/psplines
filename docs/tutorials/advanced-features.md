# Advanced Features Tutorial

This tutorial covers advanced PSplines features including constraints, different penalty orders, specialized basis configurations, and advanced computational techniques.

## Introduction

Beyond basic smoothing, PSplines offers sophisticated features for specialized applications:

- **Boundary constraints**: Control derivatives at endpoints
- **Interior constraints**: Force specific values or derivatives
- **Variable penalty orders**: Different smoothness assumptions
- **Sparse data handling**: Techniques for irregular or sparse datasets
- **Large dataset optimization**: Memory-efficient approaches
- **Custom basis configurations**: Fine-tuning knot placement and degrees

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline, ShapeConstraint, SlopeZeroConstraint, BSplineBasis, build_penalty_matrix
from psplines.optimize import cross_validation, l_curve
import scipy.sparse as sp
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)
```

## Boundary Constraints

Control the behavior of your spline at the boundaries.

### Derivative Constraints

```python
# Generate sample data with known boundary behavior
n = 60
x = np.linspace(0, 2*np.pi, n)
# Function with known derivatives at boundaries
true_function = x**2 * np.sin(x)
# f'(0) = 0, f'(2π) = (2π)^2 * cos(2π) + 2(2π) * sin(2π) = (2π)^2
true_deriv_0 = 0.0
true_deriv_end = (2*np.pi)**2

y = true_function + 0.1 * np.random.randn(n)

# Evaluation points
x_eval = np.linspace(0, 2*np.pi, 200)
true_eval = x_eval**2 * np.sin(x_eval)

# Unconstrained fit
spline_unconstrained = PSpline(x, y, nseg=25)
optimal_lambda, _ = cross_validation(spline_unconstrained)
spline_unconstrained.lambda_ = optimal_lambda
spline_unconstrained.fit()

# Constrained fit with derivative constraints
# Note: This is a conceptual example - full constraint implementation 
# would require extending the PSpline class
def fit_with_constraints(x_data, y_data, nseg, lambda_val, constraints=None):
    """
    Conceptual framework for constrained P-spline fitting.
    In practice, this would modify the linear system to include constraints.
    """
    # Create basic spline
    spline = PSpline(x_data, y_data, nseg=nseg, lambda_=lambda_val)
    spline.fit()
    
    # For demonstration, we'll show how constraints would affect the solution
    # In a full implementation, this would modify the normal equations:
    # (B^T B + λ P^T P + C^T C) α = B^T y + C^T d
    # where C is the constraint matrix and d are the constraint values
    
    if constraints is not None:
        print(f"Would apply constraints: {constraints}")
        # This would implement the constraint logic
    
    return spline

# Plot unconstrained vs conceptual constrained approach
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.scatter(x, y, alpha=0.6, s=30, label='Data')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
y_pred_uncon = spline_unconstrained.predict(x_eval)
plt.plot(x_eval, y_pred_uncon, 'r-', linewidth=2, label='Unconstrained P-spline')
plt.title('Unconstrained Fit')
plt.legend()
plt.grid(True, alpha=0.3)

# Show derivatives at boundaries
dy_dx_uncon = spline_unconstrained.derivative(x_eval, deriv_order=1)
actual_deriv_0 = spline_unconstrained.derivative(np.array([0]), deriv_order=1)[0]
actual_deriv_end = spline_unconstrained.derivative(np.array([2*np.pi]), deriv_order=1)[0]

plt.subplot(2, 2, 2)
plt.plot(x_eval, dy_dx_uncon, 'r-', linewidth=2, label="Unconstrained f'")
true_deriv_eval = 2*x_eval*np.sin(x_eval) + x_eval**2*np.cos(x_eval)
plt.plot(x_eval, true_deriv_eval, 'g--', linewidth=2, label="True f'")
plt.axhline(y=true_deriv_0, color='blue', linestyle=':', label=f"f'(0) = {true_deriv_0}")
plt.axhline(y=true_deriv_end, color='orange', linestyle=':', label=f"f'(2π) = {true_deriv_end:.1f}")
plt.title('First Derivatives')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"=== Boundary Derivative Analysis ===")
print(f"True f'(0): {true_deriv_0:.4f}")
print(f"Unconstrained f'(0): {actual_deriv_0:.4f}")
print(f"True f'(2π): {true_deriv_end:.4f}")
print(f"Unconstrained f'(2π): {actual_deriv_end:.4f}")

# Demonstrate constraint framework concept
plt.subplot(2, 1, 2)
plt.scatter(x, y, alpha=0.6, s=30, label='Data')
plt.plot(x_eval, true_eval, 'g--', linewidth=2, label='True function')
plt.plot(x_eval, y_pred_uncon, 'r-', linewidth=2, label='Unconstrained')

# Manually adjust for demonstration (this would be automatic with constraints)
# This is just for visualization - not a real constraint implementation
spline_demo = PSpline(x, y, nseg=25, lambda_=optimal_lambda * 2)  # More smoothing
spline_demo.fit()
y_pred_demo = spline_demo.predict(x_eval)
plt.plot(x_eval, y_pred_demo, 'b-', linewidth=2, alpha=0.7, 
         label='Higher smoothing (demo)')

plt.title('Comparison of Smoothing Approaches')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Value Constraints

```python
# Monotone increasing fit using shape constraints (§8.7)
x_mono = np.linspace(0, 5, 60)
y_true_mono = np.log(x_mono + 1) + 0.5 * x_mono
y_mono = y_true_mono + 0.3 * np.random.randn(60)

# Unconstrained fit for comparison
spline_free = PSpline(x_mono, y_mono, nseg=20)
opt_lambda, _ = cross_validation(spline_free)
spline_free.lambda_ = opt_lambda
spline_free.fit()

# Shape-constrained fit: enforce monotone increasing
spline_mono = PSpline(x_mono, y_mono, nseg=20, lambda_=opt_lambda,
                      shape=[ShapeConstraint(type="increasing")])
spline_mono.fit()

x_eval = np.linspace(0, 5, 200)
dy_free = spline_free.derivative(x_eval, deriv_order=1)
dy_mono = spline_mono.derivative(x_eval, deriv_order=1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_mono, y_mono, alpha=0.5, s=20, label='Data')
plt.plot(x_eval, spline_free.predict(x_eval), 'r-', label='Unconstrained')
plt.plot(x_eval, spline_mono.predict(x_eval), 'b-', label='Monotone constrained')
plt.legend(); plt.title('Fits')

plt.subplot(1, 2, 2)
plt.plot(x_eval, dy_free, 'r-', label="Unconstrained f'")
plt.plot(x_eval, dy_mono, 'b-', label="Monotone f'")
plt.axhline(0, color='k', ls='--', alpha=0.5)
plt.legend(); plt.title('First Derivatives')

plt.tight_layout()
plt.show()

print(f"Unconstrained: min derivative = {dy_free.min():.4f}")
print(f"Constrained:   min derivative = {dy_mono.min():.4f}")
```

## Shape Constraints (§8.7)

Shape constraints use an asymmetric penalty to enforce monotonicity, convexity,
concavity, or non-negativity of the fitted curve—with optional selective domains.

### All Constraint Types

```python
from psplines.penalty import VALID_SHAPE_TYPES

print("Available shape types:", sorted(VALID_SHAPE_TYPES))
# ['concave', 'convex', 'decreasing', 'increasing', 'nonneg']
```

### Combined Constraints

You can stack multiple constraints for simultaneous enforcement:

```python
# Increasing AND concave (classic saturating curve)
x = np.linspace(0, 10, 80)
y = np.sqrt(x) + 0.15 * np.random.randn(80)

spline = PSpline(x, y, nseg=25, lambda_=1.0,
                 shape=[ShapeConstraint(type="increasing"),
                        ShapeConstraint(type="concave")])
spline.fit()

x_eval = np.linspace(0, 10, 200)
plt.scatter(x, y, alpha=0.5, s=20, label='Data')
plt.plot(x_eval, spline.predict(x_eval), 'r-', linewidth=2,
         label='Increasing + Concave')
plt.legend()
plt.title('Combined Shape Constraints')
plt.show()
```

### Selective Domain Constraints

Apply a constraint only in a sub-range of $x$:

```python
# Function that increases up to x=5, then levels off
x = np.linspace(0, 10, 100)
y_true = np.where(x < 5, x, 5.0) + 0.2 * np.random.randn(100)

# Only enforce monotonicity for x ∈ [0, 5]
spline = PSpline(x, y_true, nseg=30, lambda_=1.0,
                 shape=[ShapeConstraint(type="increasing",
                         domain=(0.0, 5.0))])
spline.fit()

x_eval = np.linspace(0, 10, 200)
plt.scatter(x, y_true, alpha=0.5, s=20)
plt.plot(x_eval, spline.predict(x_eval), 'r-', linewidth=2)
plt.axvline(5, color='gray', ls='--', alpha=0.5, label='Constraint boundary')
plt.legend()
plt.title('Selective Domain: Increasing for x ≤ 5 only')
plt.show()
```

### Shape Constraints with GLM Families

Shape constraints work with Poisson and Binomial models too:

```python
# Monotone Poisson P-spline
x = np.linspace(0, 5, 80)
counts = np.random.poisson(np.exp(0.3 * x), 80)

spline = PSpline(x, counts, nseg=20, lambda_=10.0, family="poisson",
                 shape=[ShapeConstraint(type="increasing")])
spline.fit()

mu_hat = spline.predict(np.linspace(0, 5, 200))
```

## Adaptive and Variable Penalties (§8.8)

Standard P-splines apply the same penalty uniformly. When the underlying
function has regions of different complexity, adaptive or variable penalties
can improve the fit.

### Exponential Variable Penalty

The `penalty_gamma` parameter applies exponentially varying weights
$v_j = \exp(\gamma j/m)$:

```python
# Simulate a function that is smooth on the left and wiggly on the right
x = np.linspace(0, 10, 150)
y_true = np.where(x < 5, 0.5*x, 0.5*x + np.sin(5*x))
y = y_true + 0.3 * np.random.randn(150)

# Standard fit
spline_std = PSpline(x, y, nseg=30, lambda_=1.0)
spline_std.fit()

# Variable penalty: lighter penalty on the right (γ < 0)
spline_var = PSpline(x, y, nseg=30, lambda_=1.0, penalty_gamma=-5.0)
spline_var.fit()

x_eval = np.linspace(0, 10, 300)
plt.figure(figsize=(12, 5))
plt.scatter(x, y, alpha=0.3, s=10, color='gray')
plt.plot(x_eval, y_true[0:1].repeat(300) if False else
         np.where(x_eval < 5, 0.5*x_eval, 0.5*x_eval + np.sin(5*x_eval)),
         'g--', label='True')
plt.plot(x_eval, spline_std.predict(x_eval), 'r-', label='Standard penalty')
plt.plot(x_eval, spline_var.predict(x_eval), 'b-', label='Variable penalty (γ=-5)')
plt.legend()
plt.title('Variable Penalty: Lighter Smoothing on the Right')
plt.show()
```

### Automatic (λ, γ) Selection

Use `variable_penalty_cv` to find the optimal pair via a 2-D grid search:

```python
from psplines.optimize import variable_penalty_cv

spline = PSpline(x, y, nseg=30)
spline.fit()

best_lam, best_gam, best_score, scores = variable_penalty_cv(
    spline,
    gamma_range=(-10, 10),
    lambda_bounds=(1e-3, 1e3),
    num_gamma=31,
    num_lambda=31,
    criterion="gcv",
)
print(f"Optimal λ = {best_lam:.4f}, γ = {best_gam:.2f}, GCV = {best_score:.6f}")

# Visualise the GCV surface
import matplotlib.pyplot as plt
plt.imshow(scores, aspect='auto', origin='lower',
           extent=[np.log10(1e-3), np.log10(1e3), -10, 10])
plt.colorbar(label='GCV')
plt.xlabel('log₁₀(λ)')
plt.ylabel('γ')
plt.title('Variable Penalty GCV Surface')
plt.show()
```

### Nonparametric Adaptive Penalty

The `adaptive=True` mode estimates per-difference weights from the data.
A secondary B-spline basis models the log-residuals, producing spatially
varying penalty weights $w_j$:

```python
# Adaptive smoothing
spline_adapt = PSpline(x, y, nseg=30, lambda_=1.0,
                       adaptive=True, adaptive_nseg=10,
                       adaptive_lambda=100.0)
spline_adapt.fit()

x_eval = np.linspace(0, 10, 300)
plt.scatter(x, y, alpha=0.3, s=10, color='gray')
plt.plot(x_eval, spline_adapt.predict(x_eval), 'b-', linewidth=2,
         label='Adaptive penalty')
plt.legend()
plt.title('Nonparametric Adaptive Penalty')
plt.show()
```

## Different Penalty Orders

Explore how different penalty orders affect the smoothness characteristics.

```python
# Generate data with different smoothness characteristics
x_pen = np.linspace(0, 4*np.pi, 80)
y_smooth = np.sin(x_pen) * np.exp(-x_pen/8)  # Smooth function
y_wiggly = y_smooth + 0.1 * np.sin(15*x_pen)  # Added high-frequency component
y_data = y_wiggly + 0.05 * np.random.randn(80)

x_eval_pen = np.linspace(0, 4*np.pi, 200)
y_smooth_eval = np.sin(x_eval_pen) * np.exp(-x_eval_pen/8)

# Compare different penalty orders
penalty_orders = [1, 2, 3, 4]
colors = ['red', 'blue', 'green', 'purple']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, penalty_order in enumerate(penalty_orders):
    # Create spline with specific penalty order
    spline_pen = PSpline(x_pen, y_data, nseg=25, penalty_order=penalty_order)
    
    # Find optimal lambda
    opt_lambda, _ = cross_validation(spline_pen, n_lambda=30)
    spline_pen.lambda_ = opt_lambda
    spline_pen.fit()
    
    # Evaluate
    y_pred_pen = spline_pen.predict(x_eval_pen)
    
    # Plot
    axes[i].scatter(x_pen, y_data, alpha=0.6, s=20, color='gray', label='Noisy data')
    axes[i].plot(x_eval_pen, y_smooth_eval, 'g--', linewidth=2, alpha=0.7, 
                 label='Smooth truth')
    axes[i].plot(x_eval_pen, y_pred_pen, color=colors[i], linewidth=2, 
                 label=f'P{penalty_order} penalty')
    axes[i].set_title(f'Penalty Order {penalty_order} (DoF = {spline_pen.ED:.1f})')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    # Analyze smoothness
    derivatives = []
    for deriv_order in range(1, min(penalty_order + 2, 4)):
        deriv = spline_pen.derivative(x_eval_pen, deriv_order=deriv_order)
        roughness = np.sqrt(np.mean(np.diff(deriv)**2))
        derivatives.append(roughness)
    
    print(f"Penalty Order {penalty_order}: λ={opt_lambda:.4f}, DoF={spline_pen.ED:.1f}")

plt.tight_layout()
plt.show()

# Detailed comparison of penalty matrices
print("\n=== Penalty Matrix Properties ===")
for penalty_order in penalty_orders:
    # Build penalty matrix
    nb = 30  # Example number of basis functions
    P = build_penalty_matrix(nb, penalty_order)
    
    # Analyze properties
    rank = np.linalg.matrix_rank(P.toarray())
    null_space_dim = nb - rank
    
    print(f"Order {penalty_order}: rank={rank}, null space dim={null_space_dim}")
    
    # Show what the penalty penalizes
    if penalty_order == 1:
        print("  Penalizes: Large first differences (rough slopes)")
    elif penalty_order == 2:
        print("  Penalizes: Large second differences (rough curvature)")
    elif penalty_order == 3:
        print("  Penalizes: Large third differences (rough jerk)")
    else:
        print(f"  Penalizes: Large {penalty_order}-th differences")
```

## Custom Basis Configuration

Fine-tune the B-spline basis for specialized applications.

```python
# Demonstrate custom basis configurations
def create_custom_basis_demo():
    """
    Show how different basis configurations affect the fit.
    """
    # Generate data with varying complexity
    x_basis = np.linspace(0, 10, 100)
    # Complex function with different behaviors in different regions
    y_true_basis = (np.sin(x_basis) * (x_basis < 3) + 
                   0.2 * x_basis**2 * ((x_basis >= 3) & (x_basis < 7)) + 
                   np.sin(2*x_basis) * (x_basis >= 7))
    y_basis = y_true_basis + 0.1 * np.random.randn(100)
    
    x_eval_basis = np.linspace(0, 10, 200)
    y_true_eval = (np.sin(x_eval_basis) * (x_eval_basis < 3) + 
                  0.2 * x_eval_basis**2 * ((x_eval_basis >= 3) & (x_eval_basis < 7)) + 
                  np.sin(2*x_eval_basis) * (x_eval_basis >= 7))
    
    # Different basis configurations
    configurations = [
        ("Uniform knots, degree 3", {"nseg": 25, "degree": 3}),
        ("Dense knots, degree 3", {"nseg": 40, "degree": 3}),
        ("Uniform knots, degree 1", {"nseg": 25, "degree": 1}),
        ("Uniform knots, degree 5", {"nseg": 25, "degree": 5})
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, (config_name, config_params) in enumerate(configurations):
        # Create spline with custom configuration
        spline_config = PSpline(x_basis, y_basis, **config_params)
        
        # Optimize and fit
        opt_lambda, _ = cross_validation(spline_config, n_lambda=20)
        spline_config.lambda_ = opt_lambda
        spline_config.fit()
        
        # Evaluate
        y_pred_config = spline_config.predict(x_eval_basis)
        
        # Plot
        axes[i].scatter(x_basis, y_basis, alpha=0.5, s=15, color='gray')
        axes[i].plot(x_eval_basis, y_true_eval, 'g--', linewidth=2, label='True')
        axes[i].plot(x_eval_basis, y_pred_config, 'r-', linewidth=2, label='P-spline')
        axes[i].set_title(f'{config_name}\nDoF = {spline_config.ED:.1f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Calculate fit quality
        mse = np.mean((y_basis - spline_config.predict(x_basis))**2)
        print(f"{config_name}: MSE = {mse:.4f}, DoF = {spline_config.ED:.1f}")
    
    plt.tight_layout()
    plt.show()

create_custom_basis_demo()
```

### Non-uniform Knot Placement

```python
# Demonstrate adaptive knot placement
def adaptive_knot_demo():
    """
    Show how non-uniform knot placement can improve fits for irregular data.
    """
    # Generate data with varying complexity
    x_adapt = np.concatenate([
        np.linspace(0, 2, 20),      # Sparse region
        np.linspace(2, 4, 60),      # Dense region (complex behavior)
        np.linspace(4, 6, 20)       # Sparse region
    ])
    
    # Function with different complexity in different regions
    y_true_adapt = np.where(
        (x_adapt >= 2) & (x_adapt <= 4),
        np.sin(10 * x_adapt) * 0.5,  # High frequency in middle
        0.1 * x_adapt                 # Linear elsewhere
    )
    y_adapt = y_true_adapt + 0.1 * np.random.randn(len(x_adapt))
    
    x_eval_adapt = np.linspace(0, 6, 200)
    y_true_eval_adapt = np.where(
        (x_eval_adapt >= 2) & (x_eval_adapt <= 4),
        np.sin(10 * x_eval_adapt) * 0.5,
        0.1 * x_eval_adapt
    )
    
    # Standard uniform knot placement
    spline_uniform = PSpline(x_adapt, y_adapt, nseg=25)
    opt_lambda_unif, _ = cross_validation(spline_uniform)
    spline_uniform.lambda_ = opt_lambda_unif
    spline_uniform.fit()
    
    # Conceptual adaptive placement (in practice, this would require 
    # custom knot vector generation)
    # For demonstration, we use more segments (which approximates denser knots)
    spline_dense = PSpline(x_adapt, y_adapt, nseg=40)
    opt_lambda_dense, _ = cross_validation(spline_dense)
    spline_dense.lambda_ = opt_lambda_dense * 2  # More smoothing to compensate
    spline_dense.fit()
    
    # Compare results
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x_adapt, y_adapt, alpha=0.6, s=20, c=x_adapt, cmap='viridis', 
                label='Data (colored by x)')
    plt.colorbar(label='x position')
    plt.plot(x_eval_adapt, y_true_eval_adapt, 'g--', linewidth=2, label='True function')
    y_pred_uniform = spline_uniform.predict(x_eval_adapt)
    plt.plot(x_eval_adapt, y_pred_uniform, 'r-', linewidth=2, 
             label=f'Uniform knots (DoF={spline_uniform.ED:.1f})')
    plt.title('Uniform Knot Spacing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(x_adapt, y_adapt, alpha=0.6, s=20, c=x_adapt, cmap='viridis')
    plt.plot(x_eval_adapt, y_true_eval_adapt, 'g--', linewidth=2, label='True function')
    y_pred_dense = spline_dense.predict(x_eval_adapt)
    plt.plot(x_eval_adapt, y_pred_dense, 'b-', linewidth=2, 
             label=f'More segments (DoF={spline_dense.ED:.1f})')
    plt.title('Denser Basis (Adaptive Approximation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate regional fit quality
    regions = [
        (0, 2, "Sparse region 1"),
        (2, 4, "Complex region"),
        (4, 6, "Sparse region 2")
    ]
    
    print("=== Regional Fit Quality ===")
    for start, end, region_name in regions:
        mask = (x_adapt >= start) & (x_adapt <= end)
        if np.any(mask):
            x_region = x_adapt[mask]
            y_region = y_adapt[mask]
            
            pred_uniform = spline_uniform.predict(x_region)
            pred_dense = spline_dense.predict(x_region)
            
            mse_uniform = np.mean((y_region - pred_uniform)**2)
            mse_dense = np.mean((y_region - pred_dense)**2)
            
            print(f"{region_name}: Uniform MSE = {mse_uniform:.4f}, "
                  f"Dense MSE = {mse_dense:.4f}")

adaptive_knot_demo()
```

## Large Dataset Optimization

Techniques for handling large datasets efficiently.

```python
# Demonstrate large dataset techniques
def large_dataset_demo():
    """
    Show optimization techniques for large datasets.
    """
    # Generate large dataset
    n_large = 2000
    print(f"Generating large dataset with {n_large} points...")
    
    x_large = np.sort(np.random.uniform(0, 10, n_large))
    y_true_large = np.sin(x_large) + 0.1 * x_large + 0.5 * np.sin(5*x_large)
    y_large = y_true_large + 0.15 * np.random.randn(n_large)
    
    # Standard approach (for comparison)
    print("Standard approach...")
    import time
    
    start_time = time.time()
    spline_standard = PSpline(x_large, y_large, nseg=50)
    opt_lambda, _ = cross_validation(spline_standard, n_lambda=20)  # Fewer lambda values
    spline_standard.lambda_ = opt_lambda
    spline_standard.fit()
    standard_time = time.time() - start_time
    
    # Memory-efficient approach with fewer segments
    print("Memory-efficient approach...")
    start_time = time.time()
    spline_efficient = PSpline(x_large, y_large, nseg=30)  # Fewer segments
    opt_lambda_eff, _ = cross_validation(spline_efficient, n_lambda=15)
    spline_efficient.lambda_ = opt_lambda_eff
    spline_efficient.fit()
    efficient_time = time.time() - start_time
    
    # Subsampling approach for very large datasets
    print("Subsampling approach...")
    subsample_size = 500
    subsample_indices = np.random.choice(n_large, subsample_size, replace=False)
    x_sub = x_large[subsample_indices]
    y_sub = y_large[subsample_indices]
    
    start_time = time.time()
    spline_sub = PSpline(x_sub, y_sub, nseg=25)
    opt_lambda_sub, _ = cross_validation(spline_sub)
    spline_sub.lambda_ = opt_lambda_sub
    spline_sub.fit()
    subsample_time = time.time() - start_time
    
    # Evaluate all approaches
    x_eval_large = np.linspace(0, 10, 300)
    y_true_eval_large = (np.sin(x_eval_large) + 0.1 * x_eval_large + 
                        0.5 * np.sin(5*x_eval_large))
    
    y_pred_standard = spline_standard.predict(x_eval_large)
    y_pred_efficient = spline_efficient.predict(x_eval_large)
    y_pred_sub = spline_sub.predict(x_eval_large)
    
    # Plot comparison (subsample for visualization)
    vis_indices = np.random.choice(n_large, 200, replace=False)
    x_vis = x_large[vis_indices]
    y_vis = y_large[vis_indices]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(x_vis, y_vis, alpha=0.4, s=10, color='gray')
    plt.plot(x_eval_large, y_true_eval_large, 'g--', linewidth=2, label='True')
    plt.plot(x_eval_large, y_pred_standard, 'r-', linewidth=2, 
             label=f'Standard (50 seg, DoF={spline_standard.ED:.1f})')
    plt.title(f'Standard Approach ({standard_time:.2f}s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(x_vis, y_vis, alpha=0.4, s=10, color='gray')
    plt.plot(x_eval_large, y_true_eval_large, 'g--', linewidth=2, label='True')
    plt.plot(x_eval_large, y_pred_efficient, 'b-', linewidth=2, 
             label=f'Efficient (30 seg, DoF={spline_efficient.ED:.1f})')
    plt.title(f'Memory Efficient ({efficient_time:.2f}s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.scatter(x_sub, y_sub, alpha=0.6, s=15, color='orange', label='Subsample')
    plt.plot(x_eval_large, y_true_eval_large, 'g--', linewidth=2, label='True')
    plt.plot(x_eval_large, y_pred_sub, 'purple', linewidth=2, 
             label=f'Subsampled (DoF={spline_sub.ED:.1f})')
    plt.title(f'Subsampling ({subsample_time:.2f}s, n={subsample_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error comparison
    plt.subplot(2, 2, 4)
    errors_standard = np.abs(y_pred_standard - y_true_eval_large)
    errors_efficient = np.abs(y_pred_efficient - y_true_eval_large)
    errors_sub = np.abs(y_pred_sub - y_true_eval_large)
    
    plt.plot(x_eval_large, errors_standard, 'r-', alpha=0.7, label='Standard')
    plt.plot(x_eval_large, errors_efficient, 'b-', alpha=0.7, label='Efficient')
    plt.plot(x_eval_large, errors_sub, 'purple', alpha=0.7, label='Subsampled')
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title('Absolute Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    print("\n=== Large Dataset Performance Summary ===")
    print(f"Dataset size: {n_large} points")
    print(f"Standard (50 seg):   {standard_time:.2f}s, MSE = {np.mean(errors_standard**2):.4f}")
    print(f"Efficient (30 seg):  {efficient_time:.2f}s, MSE = {np.mean(errors_efficient**2):.4f}")
    print(f"Subsampled ({subsample_size} pts): {subsample_time:.2f}s, MSE = {np.mean(errors_sub**2):.4f}")
    print(f"Speed improvement (efficient): {standard_time/efficient_time:.1f}x")
    print(f"Speed improvement (subsample): {standard_time/subsample_time:.1f}x")

large_dataset_demo()
```

## Specialized Smoothing Applications

### Periodic Data

```python
# Handle periodic data
def periodic_spline_demo():
    """
    Demonstrate handling of periodic data.
    """
    # Generate periodic data
    n_period = 60
    x_period = np.linspace(0, 2*np.pi, n_period)
    y_true_period = np.sin(2*x_period) + 0.5*np.cos(5*x_period)
    y_period = y_true_period + 0.2 * np.random.randn(n_period)
    
    # Standard spline (non-periodic)
    spline_nonperiodic = PSpline(x_period, y_period, nseg=20)
    opt_lambda_np, _ = cross_validation(spline_nonperiodic)
    spline_nonperiodic.lambda_ = opt_lambda_np
    spline_nonperiodic.fit()
    
    # For true periodic splines, we'd need to modify the basis construction
    # Here we demonstrate the concept by ensuring boundary continuity
    x_extended = np.concatenate([x_period, x_period + 2*np.pi])
    y_extended = np.concatenate([y_period, y_period])  # Replicate data
    
    spline_extended = PSpline(x_extended, y_extended, nseg=25)
    opt_lambda_ext, _ = cross_validation(spline_extended, n_lambda=15)
    spline_extended.lambda_ = opt_lambda_ext
    spline_extended.fit()
    
    # Evaluate
    x_eval_period = np.linspace(0, 4*np.pi, 300)  # Two periods
    y_true_eval_period = np.sin(2*x_eval_period) + 0.5*np.cos(5*x_eval_period)
    
    y_pred_np = spline_nonperiodic.predict(x_eval_period[x_eval_period <= 2*np.pi])
    y_pred_ext = spline_extended.predict(x_eval_period)
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.scatter(x_period, y_period, alpha=0.7, s=40, color='red', label='Original data')
    x_plot_short = x_eval_period[x_eval_period <= 2*np.pi]
    plt.plot(x_plot_short, y_true_eval_period[:len(x_plot_short)], 'g--', 
             linewidth=2, label='True function')
    plt.plot(x_plot_short, y_pred_np, 'r-', linewidth=2, label='Standard P-spline')
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    plt.axvline(x=2*np.pi, color='black', linestyle=':', alpha=0.5)
    plt.title('Standard (Non-Periodic) P-Spline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.scatter(x_period, y_period, alpha=0.7, s=40, color='red', label='Original data')
    plt.scatter(x_period + 2*np.pi, y_period, alpha=0.7, s=40, color='blue', 
                label='Replicated data')
    plt.plot(x_eval_period, y_true_eval_period, 'g--', linewidth=2, label='True function')
    plt.plot(x_eval_period, y_pred_ext, 'b-', linewidth=2, label='Extended data P-spline')
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    plt.axvline(x=2*np.pi, color='black', linestyle=':', alpha=0.5)
    plt.axvline(x=4*np.pi, color='black', linestyle=':', alpha=0.5)
    plt.title('Pseudo-Periodic P-Spline (Extended Data)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Check boundary continuity
    boundary_left = spline_nonperiodic.predict(np.array([0]))[0]
    boundary_right = spline_nonperiodic.predict(np.array([2*np.pi]))[0]
    
    boundary_left_ext = spline_extended.predict(np.array([0]))[0]
    boundary_right_ext = spline_extended.predict(np.array([2*np.pi]))[0]
    
    print("=== Boundary Analysis ===")
    print(f"Standard spline:")
    print(f"  f(0) = {boundary_left:.4f}")
    print(f"  f(2π) = {boundary_right:.4f}")
    print(f"  Difference = {abs(boundary_right - boundary_left):.4f}")
    print(f"Extended spline:")
    print(f"  f(0) = {boundary_left_ext:.4f}")
    print(f"  f(2π) = {boundary_right_ext:.4f}")
    print(f"  Difference = {abs(boundary_right_ext - boundary_left_ext):.4f}")

periodic_spline_demo()
```

### Multi-Scale Data

```python
# Handle data with multiple scales
def multiscale_demo():
    """
    Demonstrate techniques for multi-scale data.
    """
    # Generate multi-scale data
    x_multi = np.linspace(0, 10, 120)
    
    # Multiple components at different scales
    trend = 0.1 * x_multi**2  # Slow trend
    seasonal = np.sin(2*np.pi*x_multi/2)  # Seasonal component
    high_freq = 0.2 * np.sin(20*x_multi)  # High frequency noise
    
    y_true_multi = trend + seasonal + high_freq
    y_multi = y_true_multi + 0.1 * np.random.randn(120)
    
    # Different smoothing approaches
    lambdas = [0.01, 1.0, 100.0]  # Under-smooth, balanced, over-smooth
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original data and components
    axes[0, 0].plot(x_multi, trend, 'g-', linewidth=2, label='Trend')
    axes[0, 0].plot(x_multi, seasonal, 'b-', linewidth=2, label='Seasonal')
    axes[0, 0].plot(x_multi, high_freq, 'r-', alpha=0.7, label='High freq')
    axes[0, 0].plot(x_multi, y_true_multi, 'k-', linewidth=2, label='Total')
    axes[0, 0].scatter(x_multi, y_multi, alpha=0.4, s=10, color='gray')
    axes[0, 0].set_title('Data Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Different smoothing levels
    for i, lam in enumerate(lambdas):
        ax = axes[0, 1] if i == 0 else axes[1, i-1]
        
        spline_scale = PSpline(x_multi, y_multi, nseg=30, lambda_=lam)
        spline_scale.fit()
        
        x_eval_multi = np.linspace(0, 10, 200)
        y_pred_multi = spline_scale.predict(x_eval_multi)
        
        trend_eval = 0.1 * x_eval_multi**2
        seasonal_eval = np.sin(2*np.pi*x_eval_multi/2)
        high_freq_eval = 0.2 * np.sin(20*x_eval_multi)
        y_true_eval = trend_eval + seasonal_eval + high_freq_eval
        
        ax.scatter(x_multi, y_multi, alpha=0.4, s=10, color='gray')
        ax.plot(x_eval_multi, y_true_eval, 'g--', linewidth=2, alpha=0.7, label='True')
        ax.plot(x_eval_multi, y_pred_multi, 'r-', linewidth=2, 
                label=f'P-spline (λ={lam})')
        
        if lam == 0.01:
            title_suffix = "Under-smoothed"
        elif lam == 1.0:
            title_suffix = "Balanced"
        else:
            title_suffix = "Over-smoothed"
            
        ax.set_title(f'{title_suffix} (λ={lam}, DoF={spline_scale.ED:.1f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Frequency analysis
    print("=== Multi-Scale Analysis ===")
    print("Recommendation: Use cross-validation or multiple smoothing levels")
    print("For trend extraction: Use high λ (over-smooth)")
    print("For feature detection: Use low λ (under-smooth)")
    print("For general use: Use CV-optimal λ")

multiscale_demo()
```

## Observation Weights

Observation weights let you control how much each data point influences the fit.
The normal equations become $(B'WB + \lambda D'D)\alpha = B'Wy$ where $W = \text{diag}(w)$.

### Heteroscedastic Data

When observations have unequal precision, use weights proportional to the
inverse variance (or any measure of reliability):

```python
import numpy as np
from psplines import PSpline

np.random.seed(42)
x = np.linspace(0, 1, 100)
# Noise increases with x
noise_sd = 0.05 + 0.4 * x
y = np.sin(2 * np.pi * x) + noise_sd * np.random.randn(100)

# Weight inversely proportional to variance
weights = 1.0 / noise_sd**2

spline_unweighted = PSpline(x, y, nseg=20, lambda_=1.0).fit()
spline_weighted = PSpline(x, y, nseg=20, lambda_=1.0, weights=weights).fit()

# The weighted fit trusts the low-noise (left) region more
print(f"Unweighted ED: {spline_unweighted.ED:.1f}")
print(f"Weighted ED:   {spline_weighted.ED:.1f}")
```

### Missing Data via Zero Weights

Setting $w_i = 0$ removes an observation from the fit entirely. The penalty
interpolates smoothly through the gap — no special handling required:

```python
x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)

# Mark a contiguous block as missing
weights = np.ones(100)
weights[40:60] = 0.0
y[40:60] = 999.0  # dummy values — ignored because w=0

spline = PSpline(x, y, nseg=20, weights=weights).fit()

# Fitted values in the gap are smooth interpolations, not 999
print(f"Fit at gap midpoint: {spline.fitted_values[50]:.3f}")
print(f"True value:          {np.sin(2 * np.pi * x[50]):.3f}")
```

With a second-order penalty, the interpolation through the gap is cubic in the
coefficient indices (degree $2d - 1$ where $d$ is the penalty order).

### Weights with Automatic Lambda Selection

All optimizers (`cross_validation`, `aic`, `l_curve`, `v_curve`) respect
observation weights automatically:

```python
from psplines.optimize import cross_validation

weights = 1.0 + 5.0 * x  # emphasize right side
spline = PSpline(x, y, nseg=20, weights=weights)
optimal_lambda, score = cross_validation(spline)
print(f"Optimal lambda: {optimal_lambda:.4f}")
```

## GLM P-Splines

P-splines extend beyond Gaussian responses to count and binary data via GLM families.

### Poisson Smoothing (Count Data)

```python
import numpy as np
from psplines import PSpline

# Simulate event counts
np.random.seed(42)
x = np.linspace(0, 10, 80)
true_rate = np.exp(1 + 0.5 * np.sin(x))
counts = np.random.poisson(true_rate)

# Fit Poisson P-spline (log link)
spline = PSpline(x, counts, nseg=20, lambda_=100, family="poisson")
spline.fit()

x_eval = np.linspace(0, 10, 200)
mu_hat = spline.predict(x_eval)  # positive fitted rates

# Confidence intervals (asymmetric, always positive)
mu_hat, lower, upper = spline.predict(x_eval, return_se=True)

import matplotlib.pyplot as plt
plt.scatter(x, counts, alpha=0.5, label="Counts")
plt.plot(x_eval, mu_hat, "r-", label="Poisson P-spline")
plt.fill_between(x_eval, lower, upper, alpha=0.3, color="red")
plt.plot(x_eval, np.exp(1 + 0.5 * np.sin(x_eval)), "g--", label="True rate")
plt.legend()
plt.show()
```

### Poisson with Exposure Offsets

For rate modeling where observations have different exposures (e.g., population at risk):

```python
# mu = exp(B*alpha) * exposure  =>  log(mu) = B*alpha + log(exposure)
exposure = np.random.uniform(100, 1000, len(x))
rate_counts = np.random.poisson(true_rate * exposure / 500)

spline = PSpline(x, rate_counts, family="poisson", offset=np.log(exposure),
                 nseg=20, lambda_=100)
spline.fit()
```

### Binomial Smoothing (Binary/Proportion Data)

```python
# Dose-response curve
dose = np.linspace(0, 10, 60)
true_prob = 1 / (1 + np.exp(-(dose - 5)))

# Bernoulli response (y in {0, 1})
y_binary = np.random.binomial(1, true_prob)
spline = PSpline(dose, y_binary, nseg=15, lambda_=10, family="binomial")
spline.fit()

# Fitted probabilities in [0, 1]
dose_eval = np.linspace(0, 10, 200)
pi_hat = spline.predict(dose_eval)

plt.scatter(dose, y_binary, alpha=0.3, label="Binary response")
plt.plot(dose_eval, pi_hat, "r-", label="Binomial P-spline")
plt.plot(dose_eval, 1 / (1 + np.exp(-(dose_eval - 5))), "g--", label="True prob")
plt.legend()
plt.show()
```

### Grouped Binomial

When you observe $y$ successes out of $t$ trials:

```python
trials = np.full(60, 25)
successes = np.random.binomial(trials, true_prob)

spline = PSpline(dose, successes, family="binomial", trials=trials, nseg=15)
spline.fit()

# predict returns estimated probabilities (mu/trials on response scale)
pi_hat = spline.predict(dose_eval)
```

### Density Estimation

Smooth density estimation from raw data using a Poisson P-spline on histogram counts:

```python
from psplines import density_estimate

# Generate sample from a mixture distribution
data = np.concatenate([
    np.random.normal(-2, 0.5, 300),
    np.random.normal(2, 0.8, 200),
])

result = density_estimate(data, bins=100, penalty_order=3)

plt.plot(result.grid, result.density, "r-", linewidth=2)
plt.xlabel("x")
plt.ylabel("Density")
plt.title(f"Density Estimate (λ = {result.lambda_:.2f})")
plt.show()

# The density integrates to ~1
print(f"Integral: {np.trapz(result.density, result.grid):.4f}")
```

### Automatic Lambda Selection for GLM

All optimizers (CV, AIC, L-curve, V-curve) work with GLM families:

```python
from psplines.optimize import aic, cross_validation

spline = PSpline(x, counts, nseg=20, family="poisson")
best_lambda, score = aic(spline)
spline.lambda_ = best_lambda
spline.fit()
```

## Summary

This tutorial covered advanced PSplines features:

### Key Advanced Features

1. **Observation Weights**: Heteroscedastic data, missing data via zero weights
2. **GLM Families**: Poisson (counts), Binomial (binary/proportions) via IRLS
3. **Density Estimation**: Smooth densities from raw data via Poisson P-splines
4. **Shape Constraints (§8.7)**: Monotonicity, convexity, concavity, non-negativity via asymmetric penalty — with selective domain support and GLM compatibility
5. **Variable Penalties (§8.8)**: Exponential weights ($\gamma$) with automatic $(\lambda, \gamma)$ selection via `variable_penalty_cv`
6. **Adaptive Penalties (§8.8)**: Nonparametric spatially varying smoothness via secondary B-spline basis
7. **Penalty Orders**: Different smoothness assumptions (1st, 2nd, 3rd order)
8. **Custom Basis**: Non-uniform knots, different degrees
9. **Large Datasets**: Memory-efficient techniques, subsampling
10. **Specialized Applications**: Periodic data, multi-scale analysis

### Practical Guidelines

- **Penalty Order**: Use 2nd order (default) for most applications
- **Shape Constraints**: Use when domain knowledge dictates shape (e.g. dose–response curves must be monotone)
- **Variable Penalty**: Use when one boundary needs more/less regularisation
- **Adaptive Penalty**: Use when the function has regions of varying complexity
- **Large Data**: Reduce segments or subsample if memory/speed is critical

### Advanced Techniques Summary

| Technique | When to Use | Computational Cost | Complexity |
|-----------|-------------|-------------------|------------|
| Observation weights | Heteroscedastic or missing data | Similar | Low |
| Shape constraints | Known monotonicity/convexity | Moderate (iterative) | Low |
| Variable penalty | Boundary-asymmetric smoothness | Similar | Low |
| Adaptive penalty | Spatially varying complexity | Higher (alternating) | Medium |
| Higher penalty order | Very smooth data | Similar | Low |
| Custom knots | Irregular complexity | Similar | Medium |
| Subsampling | Very large datasets | Much lower | Low |
| Extended basis | Periodic data | Similar | Medium |

## Next Steps

- **[Parameter Selection](parameter-selection.md)**: Optimize smoothing parameters
- **[Uncertainty Methods](uncertainty-methods.md)**: Quantify prediction uncertainty  
- **[Basic Usage](basic-usage.md)**: Review fundamental concepts
- **[Examples Gallery](../examples/gallery.md)**: Real-world applications
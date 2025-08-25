# PSplines - Penalized B-Spline Smoothing for Python

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**PSplines** is a high-performance Python library for univariate penalized B-spline (P-spline) smoothing, implementing the methods described in Eilers & Marx (2021). It provides efficient sparse-matrix implementations with analytical uncertainty quantification, parametric bootstrap, and Bayesian inference capabilities.

## Key Features

- **Fast Sparse Implementation**: Uses SciPy sparse matrices and optimized solvers
- **Multiple Uncertainty Methods**: Analytical (delta method), bootstrap, and Bayesian approaches  
- **Flexible Configuration**: Customizable basis functions, penalty orders, and constraints
- **Derivative Computation**: Efficient computation of spline derivatives with uncertainty
- **Automatic Parameter Selection**: Cross-validation, AIC, L-curve, and V-curve methods
- **Boundary Constraints**: Support for derivative boundary conditions
- **Comprehensive Validation**: Extensive input validation and error handling

## Installation

### Using pip
```bash
pip install psplines
```

### Using uv (recommended for development)
```bash
git clone https://github.com/graysonbellamy/psplines.git
cd psplines
uv sync
```

### From source
```bash
git clone https://github.com/graysonbellamy/psplines.git
cd psplines
pip install -e .
```

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from psplines import PSpline

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)

# Create and fit P-spline
spline = PSpline(x, y, nseg=20, lambda_=1.0)
spline.fit()

# Make predictions with uncertainty
x_new = np.linspace(0, 2*np.pi, 200)
y_pred, se = spline.predict(x_new, return_se=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Data')
plt.plot(x_new, y_pred, 'r-', label='P-spline fit')
plt.fill_between(x_new, y_pred - 1.96*se, y_pred + 1.96*se, 
                 alpha=0.3, label='95% CI')
plt.legend()
plt.show()
```

## Core API

### PSpline Class

The main class for penalized B-spline fitting:

```python
spline = PSpline(
    x,                    # Input points (array-like)
    y,                    # Response values (array-like)  
    nseg=20,              # Number of B-spline segments
    degree=3,             # B-spline degree
    lambda_=10.0,         # Smoothing parameter
    penalty_order=2,      # Order of difference penalty
    constraints=None      # Boundary constraints (dict)
)
```

### Key Methods

#### Fitting
```python
# Basic fitting
spline.fit()

# With custom domain
spline.fit(xl=0, xr=10)
```

#### Prediction
```python
# Basic prediction  
y_pred = spline.predict(x_new)

# With analytical standard errors
y_pred, se = spline.predict(x_new, return_se=True)

# With bootstrap standard errors
y_pred, se = spline.predict(x_new, return_se=True, se_method="bootstrap", B_boot=1000)
```

#### Derivatives
```python
# First derivative
dy_dx = spline.derivative(x_new, deriv_order=1)

# Second derivative with uncertainty
d2y_dx2, se = spline.derivative(x_new, deriv_order=2, return_se=True)
```

#### Bayesian Inference
```python
# Fit Bayesian model
trace = spline.bayes_fit(draws=2000, tune=1000)

# Get posterior credible intervals
mean, lower, upper = spline.predict(x_new, se_method="bayes", hdi_prob=0.95)
```

## Parameter Selection

PSplines provides several methods for automatic smoothing parameter selection:

```python
from psplines.optimize import cross_validation, aic, l_curve

# Cross-validation (recommended)
best_lambda, cv_score = cross_validation(spline)

# Akaike Information Criterion
best_lambda, aic_score = aic(spline)

# L-curve method
best_lambda, curvature = l_curve(spline)

# Use optimal parameter
spline.lambda_ = best_lambda
spline.fit()
```

## Advanced Usage

### Boundary Constraints

Enforce derivative constraints at boundaries:

```python
# Zero first derivative at boundaries (natural spline)
constraints = {
    "deriv": {
        "order": 1,
        "initial": 0,
        "final": 0
    }
}

spline = PSpline(x, y, constraints=constraints)
spline.fit()
```

### Different Penalty Orders

- `penalty_order=1`: Penalizes differences (rough penalty on slopes)
- `penalty_order=2`: Penalizes second differences (rough penalty on curvature) 
- `penalty_order=3`: Penalizes third differences (rough penalty on rate of curvature change)

### Custom Smoothing

```python
# Very smooth fit
smooth_spline = PSpline(x, y, lambda_=1000)

# More flexible fit  
flexible_spline = PSpline(x, y, lambda_=0.1)

# High-degree spline
high_deg_spline = PSpline(x, y, degree=5, nseg=30)
```

## Performance Tips

1. **Sparse Operations**: The library automatically uses sparse matrices for efficiency
2. **Vectorized Predictions**: Predict on multiple points simultaneously
3. **Optimal nseg**: Generally 10-50 segments work well; too many can cause overfitting
4. **Bootstrap Parallelization**: Use `n_jobs=-1` for parallel bootstrap computation

```python
# Efficient batch prediction
y_pred = spline.predict(large_x_array)

# Parallel bootstrap
y_pred, se = spline.predict(x_new, return_se=True, se_method="bootstrap", 
                           B_boot=5000, n_jobs=-1)
```

## Mathematical Background

PSplines combine B-spline basis functions with discrete difference penalties:

- **Basis**: B-splines of degree `d` with `nseg` equally-spaced segments  
- **Penalty**: Discrete differences of order `p` (typically 1, 2, or 3)
- **Objective**: Minimize `||y - Bα||² + λ||D_p α||²` where `B` is the basis matrix and `D_p` is the difference matrix

The library implements efficient algorithms for:
- Sparse matrix operations via SciPy
- Analytical standard errors via the delta method
- Effective degrees of freedom computation
- Cross-validation and information criteria

## Requirements

- Python >= 3.10
- NumPy >= 1.21
- SciPy >= 1.7
- PyMC >= 5.0 (for Bayesian methods)
- Matplotlib >= 3.4 (for plotting utilities)
- PyTensor >= 2.0 (for Bayesian methods)
- ArviZ >= 0.12 (for Bayesian diagnostics)
- Joblib >= 1.0 (for parallel processing)

## Examples

Complete examples are available in the `examples/` directory:

- **Basic Usage**: Core functionality demonstration
- **Parameter Selection**: Automatic lambda optimization  
- **Uncertainty Methods**: Comparison of different uncertainty approaches
- **Real-World Application**: Time series analysis workflow

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/graysonbellamy/psplines.git
cd psplines
uv sync --dev
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=psplines --cov-report=html

# Run specific test module  
uv run pytest tests/test_core.py -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PSplines in your research, please cite:

```bibtex
@software{bellamy2024psplines,
  author = {Bellamy, Grayson},
  title = {PSplines: Penalized B-Spline Smoothing for Python},
  year = {2024},
  url = {https://github.com/graysonbellamy/psplines}
}
```

## References

- Eilers, P. H. C., & Marx, B. D. (2021). *Practical Smoothing: The Joys of P-splines*. Cambridge University Press.
- de Boor, C. (2001). *A Practical Guide to Splines*. Springer-Verlag.

## Changelog

### Version 0.1.3
- Fixed dead code bug in derivative method
- Added comprehensive input validation
- Optimized diagonal computation for uncertainty
- Enhanced error messages and documentation
- Added extensive test suite

### Version 0.1.2
- Initial release with core P-spline functionality
- Basic fitting, prediction, and derivative computation
- Bayesian inference capabilities

---

**Questions or Issues?** Please open an issue on [GitHub](https://github.com/graysonbellamy/psplines/issues).
# PSplines: Penalized B-Spline Smoothing for Python

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/psplines.svg)](https://badge.fury.io/py/psplines)

**PSplines** is a high-performance Python library for univariate penalized B-spline (P-spline) smoothing, implementing the methods described in Eilers & Marx (2021). It provides efficient sparse-matrix implementations with analytical uncertainty quantification, parametric bootstrap, and Bayesian inference capabilities.

## Key Features

- **Fast Sparse Implementation**: Uses SciPy sparse matrices and optimized solvers
- **Multiple Uncertainty Methods**: Analytical (delta method), bootstrap, and Bayesian approaches  
- **Flexible Configuration**: Customizable basis functions, penalty orders, and constraints
- **Derivative Computation**: Efficient computation of spline derivatives with uncertainty
- **Automatic Parameter Selection**: Cross-validation, AIC, L-curve, and V-curve methods
- **Boundary Constraints**: Support for derivative boundary conditions
- **Comprehensive Validation**: Extensive input validation and error handling

## Quick Example

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

## Installation

### Using pip
```bash
pip install psplines
```

### From source
```bash
git clone https://github.com/graysonbellamy/psplines.git
cd psplines
pip install -e .
```

## Documentation Structure

- **[User Guide](user-guide/getting-started.md)**: Start here if you're new to P-splines
- **[Tutorials](tutorials/basic-usage.md)**: Step-by-step examples and walkthroughs
- **[Examples](examples/gallery.md)**: Complete application examples
- **[API Reference](api/core.md)**: Detailed documentation of all classes and functions
- **[Mathematical Background](theory/mathematical-background.md)**: Theory and algorithms

## What are P-Splines?

P-splines (Penalized B-splines) are a powerful smoothing technique that combines:

- **B-spline basis functions**: Flexible, local basis functions
- **Difference penalties**: Regularization to control smoothness
- **Automatic parameter selection**: Data-driven smoothing parameter choice

The method solves the penalized least squares problem:

$$\min_\alpha \|y - B\alpha\|^2 + \lambda \|D_p \alpha\|^2$$

where:
- $B$ is the B-spline basis matrix
- $\alpha$ are the B-spline coefficients
- $D_p$ is the $p$-th order difference matrix
- $\lambda$ is the smoothing parameter

## Use Cases

PSplines are ideal for:

- **Signal processing**: Noise reduction and trend extraction
- **Time series analysis**: Smooth trend estimation and forecasting
- **Scientific computing**: Data smoothing and derivative estimation  
- **Statistics**: Nonparametric regression and curve fitting
- **Engineering**: Control system design and signal analysis

## Performance

PSplines leverages sparse matrix operations for efficiency:

- **Memory efficient**: Uses sparse matrices throughout
- **Fast computation**: Optimized linear algebra operations
- **Scalable**: Handles large datasets effectively
- **Parallel processing**: Bootstrap uncertainty with multiprocessing

## Getting Help

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/graysonbellamy/psplines/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/graysonbellamy/psplines/discussions)
- **Documentation**: Browse this documentation for guides and examples

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

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/graysonbellamy/psplines/blob/main/LICENSE) file for details.
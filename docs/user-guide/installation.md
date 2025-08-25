# Installation

This guide covers the various ways to install PSplines and its dependencies.

## Requirements

PSplines requires Python 3.10 or later and depends on several scientific computing packages:

### Core Dependencies

- **NumPy** (≥ 1.21): Array operations and linear algebra
- **SciPy** (≥ 1.7): Sparse matrices, optimization, and spline functions
- **Matplotlib** (≥ 3.4): Basic plotting functionality

### Optional Dependencies

For full functionality, you may want:

- **PyMC** (≥ 5.0): Bayesian inference capabilities
- **PyTensor** (≥ 2.0): Backend for PyMC
- **ArviZ** (≥ 0.12): Bayesian diagnostics and visualization
- **Joblib** (≥ 1.0): Parallel processing for bootstrap methods

## Installation Methods

### Method 1: Using pip (Recommended)

The easiest way to install PSplines is using pip:

```bash
pip install psplines
```

This installs PSplines and all required dependencies.

#### Install with Optional Dependencies

To install with all optional dependencies for full functionality:

```bash
pip install psplines[full]
```

Or install specific optional dependencies:

```bash
pip install psplines[bayesian]  # For Bayesian features
pip install psplines[parallel]  # For parallel processing
```

### Method 2: Using conda/mamba

If you're using conda or mamba:

```bash
conda install -c conda-forge psplines
```

Or with mamba:

```bash
mamba install -c conda-forge psplines
```

### Method 3: Development Installation

For development or to get the latest features:

```bash
git clone https://github.com/graysonbellamy/psplines.git
cd psplines
pip install -e .
```

For development with all tools:

```bash
git clone https://github.com/graysonbellamy/psplines.git
cd psplines
pip install -e .[dev]
```

### Method 4: Using uv (Fast and Modern)

If you have [uv](https://github.com/astral-sh/uv) installed:

```bash
uv add psplines
```

For development:

```bash
git clone https://github.com/graysonbellamy/psplines.git
cd psplines
uv sync --dev
```

## Verifying Your Installation

After installation, verify that PSplines works correctly:

```python
import psplines
print(f"PSplines version: {psplines.__version__}")

# Run a simple test
import numpy as np
from psplines import PSpline

# Generate test data
np.random.seed(42)
x = np.linspace(0, 1, 20)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(20)

# Create and fit spline
spline = PSpline(x, y, nseg=10)
spline.fit()

print("✓ PSplines installed successfully!")
print(f"  - Effective DoF: {spline.ED:.2f}")
print(f"  - Residual variance: {spline.sigma2:.4f}")
```

## Testing Optional Features

### Bayesian Functionality

```python
try:
    import pymc
    import arviz
    print("✓ Bayesian features available")
    
    # Test Bayesian fitting (this may take a moment)
    trace = spline.bayes_fit(draws=100, tune=100, chains=1)
    print("✓ Bayesian inference working")
except ImportError as e:
    print(f"✗ Bayesian features not available: {e}")
```

### Parallel Processing

```python
try:
    from joblib import Parallel, delayed
    print("✓ Parallel processing available")
    
    # Test bootstrap with parallel processing
    x_new = np.linspace(0, 1, 10)
    y_pred, se = spline.predict(x_new, return_se=True, 
                               se_method="bootstrap", B_boot=100, n_jobs=2)
    print("✓ Parallel bootstrap working")
except ImportError as e:
    print(f"✗ Parallel processing not available: {e}")
```

## Environment-Specific Instructions

### Jupyter Notebooks

PSplines works great in Jupyter notebooks. For the best experience:

```bash
pip install psplines matplotlib ipywidgets
jupyter notebook
```

### Google Colab

In Google Colab, simply run:

```python
!pip install psplines
import psplines
```

### Docker

Create a Dockerfile for containerized usage:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install psplines matplotlib jupyter

# Set working directory
WORKDIR /workspace

# Start Jupyter by default
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

### Virtual Environments

#### Using venv

```bash
python -m venv psplines_env
source psplines_env/bin/activate  # On Windows: psplines_env\Scripts\activate
pip install psplines
```

#### Using conda

```bash
conda create -n psplines python=3.11
conda activate psplines
conda install -c conda-forge psplines
```

## Troubleshooting

### Common Issues

#### Issue: "No module named 'psplines'"

**Solution**: Ensure you've activated the correct environment and PSplines is installed:

```bash
pip list | grep psplines
python -c "import psplines; print('OK')"
```

#### Issue: Import errors with SciPy/NumPy

**Solution**: Update to compatible versions:

```bash
pip install --upgrade numpy scipy
```

#### Issue: Bayesian features not working

**Solution**: Install PyMC and its dependencies:

```bash
pip install pymc arviz pytensor
```

#### Issue: Slow performance

**Solution**: Ensure you have optimized BLAS libraries:

```bash
# Check current BLAS configuration
python -c "import numpy; numpy.show_config()"

# Install optimized BLAS (choose one)
conda install mkl  # Intel MKL
conda install openblas  # OpenBLAS
```

#### Issue: Memory errors with large datasets

**Solutions**:
1. Use fewer segments: `nseg=min(n//10, 50)`
2. Use single precision: Convert arrays to `float32`
3. Process data in chunks if memory is very limited

### Platform-Specific Notes

#### Windows

- Install Microsoft C++ Build Tools if compilation fails
- Use Anaconda for easier dependency management

#### macOS

- On Apple Silicon (M1/M2), use conda-forge for best compatibility:
  ```bash
  conda install -c conda-forge psplines
  ```

#### Linux

- Install development headers if building from source:
  ```bash
  sudo apt-get install python3-dev libopenblas-dev  # Ubuntu/Debian
  sudo yum install python3-devel openblas-devel     # CentOS/RHEL
  ```

### Performance Optimization

#### For maximum performance:

1. **Use optimized BLAS**: Install MKL or OpenBLAS
2. **Compile from source**: May give slight performance improvements
3. **Use appropriate data types**: `float64` for precision, `float32` for speed
4. **Enable parallel processing**: Set `n_jobs=-1` for bootstrap methods

#### Check your NumPy configuration:

```python
import numpy as np
np.show_config()  # Shows BLAS/LAPACK configuration
```

## Getting Help

If you encounter issues not covered here:

1. **Check the FAQ**: See [Common Questions](getting-started.md#common-questions)
2. **Search GitHub Issues**: [Issues Page](https://github.com/graysonbellamy/psplines/issues)
3. **Ask for help**: Open a new issue with:
   - Your operating system and Python version
   - Complete error messages
   - Minimal example that reproduces the problem

## Next Steps

After successful installation:

1. **[Getting Started](getting-started.md)**: Learn the basics
2. **[Quick Start](quick-start.md)**: Jump into examples
3. **[Tutorials](../tutorials/basic-usage.md)**: Comprehensive guides
4. **[Examples](../examples/gallery.md)**: Real-world applications
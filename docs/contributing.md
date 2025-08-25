# Contributing to PSplines

Thank you for your interest in contributing to PSplines! This guide will help you get started with contributing to the project.

## Ways to Contribute

There are many ways to contribute to PSplines:

- **Report bugs** or suggest features via [GitHub Issues](https://github.com/graysonbellamy/psplines/issues)
- **Improve documentation** by fixing typos, adding examples, or clarifying explanations
- **Add new features** such as additional optimization methods or smoothing techniques
- **Fix bugs** identified in the issue tracker
- **Add examples** demonstrating real-world applications
- **Improve performance** through code optimization or algorithmic improvements

## Development Setup

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/psplines.git
cd psplines
```

### 2. Development Environment

We recommend using [uv](https://github.com/astral-sh/uv) for development:

```bash
# Install uv if you haven't already
pip install uv

# Create development environment
uv sync --dev

# Activate the environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

Alternatively, with pip:
```bash
pip install -e .[dev]
```

### 3. Verify Installation

Run the tests to make sure everything is working:
```bash
pytest tests/
```

## Development Workflow

### 1. Create a Branch

Create a new branch for your contribution:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

### 2. Make Changes

Make your changes following the project's coding standards:

- **Code Style**: We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- **Type Hints**: Use type hints for all new code
- **Docstrings**: Follow NumPy docstring conventions
- **Tests**: Add tests for new functionality

### 3. Run Quality Checks

Before submitting, run the quality checks:

```bash
# Lint code
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy src/psplines

# Run tests
pytest tests/ -v

# Check test coverage
pytest tests/ --cov=psplines --cov-report=html
```

### 4. Commit Changes

Make clear, descriptive commit messages:
```bash
git add .
git commit -m "Add cross-validation method for parameter selection

- Implement k-fold and leave-one-out CV
- Add comprehensive tests
- Update documentation with examples"
```

### 5. Push and Create Pull Request

```bash
git push origin your-branch-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to related issues (if any)
- Screenshots for UI changes (if applicable)

## Coding Standards

### Code Style

We use `ruff` for both linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

Configuration is in `pyproject.toml`.

### Type Hints

Use type hints for all functions and methods:

```python
from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike

def predict(
    self, 
    x: ArrayLike, 
    return_se: bool = False,
    se_method: str = 'analytic'
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Predict values at new points.
    
    Parameters
    ----------
    x : ArrayLike
        Points at which to evaluate the spline
    return_se : bool, default False
        Whether to return standard errors
    se_method : str, default 'analytic'
        Method for computing standard errors
        
    Returns
    -------
    predictions : np.ndarray or tuple
        Predicted values, optionally with standard errors
    """
```

### Documentation

#### Docstrings

Use NumPy-style docstrings:

```python
def cross_validation(
    spline: PSpline,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e3,
    n_lambda: int = 50
) -> Tuple[float, float]:
    """Find optimal smoothing parameter using cross-validation.
    
    Parameters
    ----------
    spline : PSpline
        P-spline object to optimize
    lambda_min : float, default 1e-6
        Minimum lambda value to test
    lambda_max : float, default 1e3
        Maximum lambda value to test
    n_lambda : int, default 50
        Number of lambda values to test
        
    Returns
    -------
    optimal_lambda : float
        Optimal smoothing parameter
    cv_score : float
        Cross-validation score at optimal lambda
        
    Examples
    --------
    >>> from psplines import PSpline
    >>> from psplines.optimize import cross_validation
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 50)
    >>> y = np.sin(2*np.pi*x) + 0.1*np.random.randn(50)
    >>> spline = PSpline(x, y, nseg=20)
    >>> optimal_lambda, score = cross_validation(spline)
    >>> spline.lambda_ = optimal_lambda
    >>> spline.fit()
    """
```

#### Comments

Use clear, concise comments for complex algorithms:

```python
# Solve the penalized least squares system:
# (B^T B + lambda * P^T P) alpha = B^T y
A = self.basis_matrix.T @ self.basis_matrix + self.lambda_ * penalty_matrix
b = self.basis_matrix.T @ self.y
self.alpha = spsolve(A, b)
```

### Testing

#### Test Structure

Place tests in `tests/` directory with files matching `test_*.py`:

```
tests/
├── test_core.py          # Tests for core PSpline class
├── test_basis.py         # Tests for basis functions
├── test_penalty.py       # Tests for penalty matrices
├── test_optimize.py      # Tests for optimization functions
└── conftest.py           # Shared test fixtures
```

#### Writing Tests

Use pytest conventions:

```python
import pytest
import numpy as np
from psplines import PSpline

class TestPSpline:
    """Test the PSpline class."""
    
    def test_basic_fitting(self):
        """Test basic spline fitting functionality."""
        # Generate test data
        x = np.linspace(0, 1, 20)
        y = np.sin(2*np.pi*x) + 0.1*np.random.randn(20)
        
        # Create and fit spline
        spline = PSpline(x, y, nseg=10, lambda_=1.0)
        spline.fit()
        
        # Check basic properties
        assert spline.alpha is not None
        assert len(spline.alpha) == spline.nb
        assert spline.ED > 0
        assert spline.sigma2 > 0
    
    def test_prediction(self):
        """Test prediction functionality."""
        x = np.linspace(0, 1, 20)
        y = np.sin(2*np.pi*x)
        
        spline = PSpline(x, y, nseg=10, lambda_=1.0)
        spline.fit()
        
        # Test prediction
        x_new = np.linspace(0, 1, 10)
        y_pred = spline.predict(x_new)
        
        assert len(y_pred) == len(x_new)
        assert np.all(np.isfinite(y_pred))
    
    def test_input_validation(self):
        """Test input validation."""
        x = np.linspace(0, 1, 20)
        y = np.sin(2*np.pi*x)
        
        # Test invalid nseg
        with pytest.raises(ValueError, match="nseg must be positive"):
            PSpline(x, y, nseg=0)
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="x and y must have the same length"):
            PSpline(x, y[:-1])
```

#### Test Coverage

Aim for high test coverage:

```bash
# Run tests with coverage
pytest tests/ --cov=psplines --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation

### Building Documentation

The documentation uses MkDocs with Material theme:

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

### Adding Examples

When adding new examples:

1. Create the example script in `examples/`
2. Add it to the [Examples Gallery](examples/gallery.md)
3. Include clear docstrings and comments
4. Test the example thoroughly

### Updating API Documentation

API documentation is auto-generated from docstrings. To update:

1. Ensure your docstrings follow NumPy conventions
2. The documentation will automatically include new public functions
3. For new modules, add them to the appropriate `docs/api/*.md` file

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Pre-release Checklist

Before creating a release:

1. **Update version** in `src/psplines/__init__.py`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite**: `pytest tests/ -v`
4. **Check code quality**: `ruff check . && mypy src/psplines`
5. **Build documentation**: `mkdocs build`
6. **Test examples**: Run all example scripts
7. **Update dependencies** if needed

## Getting Help

### Where to Ask Questions

- **General usage questions**: [GitHub Discussions](https://github.com/graysonbellamy/psplines/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/graysonbellamy/psplines/issues)
- **Feature requests**: [GitHub Issues](https://github.com/graysonbellamy/psplines/issues)

### Development Questions

If you're working on a contribution and need help:

1. Check existing issues and discussions
2. Look at the codebase for similar implementations
3. Create a draft pull request for early feedback
4. Reach out via GitHub Issues with the `question` label

## Code of Conduct

Please be respectful and constructive in all interactions. We want to maintain a welcoming environment for contributors of all backgrounds and experience levels.

### Guidelines

- **Be respectful** of different opinions and approaches
- **Be constructive** in feedback and criticism
- **Be patient** with newcomers and questions
- **Be collaborative** in finding solutions

## Recognition

Contributors will be recognized in:

- **README.md** contributor list
- **CHANGELOG.md** for significant contributions
- **GitHub contributor graphs**

Thank you for contributing to PSplines! Your efforts help make this a better tool for the scientific and data science communities.

## Quick Reference

### Common Commands

```bash
# Development setup
git clone https://github.com/YOUR_USERNAME/psplines.git
cd psplines
uv sync --dev

# Quality checks
ruff check . && ruff format .
mypy src/psplines
pytest tests/ -v --cov=psplines

# Documentation
mkdocs serve

# Commit workflow
git checkout -b feature/my-feature
# make changes
git add . && git commit -m "Clear commit message"
git push origin feature/my-feature
# create pull request
```

### File Structure

```
psplines/
├── src/psplines/           # Source code
│   ├── __init__.py
│   ├── core.py            # Main PSpline class
│   ├── basis.py           # B-spline basis functions
│   ├── penalty.py         # Penalty matrices
│   ├── optimize.py        # Parameter optimization
│   └── utils.py           # Utility functions
├── tests/                  # Test suite
├── docs/                   # Documentation source
├── examples/               # Example scripts
├── pyproject.toml          # Project configuration
└── README.md
```
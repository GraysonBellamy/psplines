"""
Shared test fixtures and configuration for psplines tests.
"""
import pytest
import numpy as np
from psplines.core import PSpline


@pytest.fixture
def simple_data():
    """Simple test data for basic functionality."""
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 4, 9, 16, 25])  # y = x^2
    return x, y


@pytest.fixture
def noisy_sine_data():
    """Noisy sine wave data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 2*np.pi, 50)
    y = np.sin(x) + 0.1 * np.random.randn(50)
    return x, y


@pytest.fixture
def fitted_spline(noisy_sine_data):
    """Pre-fitted spline for testing prediction methods."""
    x, y = noisy_sine_data
    spline = PSpline(x, y, nseg=15, lambda_=1.0)
    return spline.fit()


@pytest.fixture
def large_dataset():
    """Larger dataset for performance testing."""
    np.random.seed(123)
    x = np.linspace(0, 10, 200)
    y = np.sin(x) * np.exp(-x/5) + 0.05 * np.random.randn(200)
    return x, y
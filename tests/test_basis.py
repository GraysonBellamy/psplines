"""
Tests for psplines.basis module.
"""
import pytest
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal

from psplines.basis import b_spline_basis, b_spline_derivative_basis


class TestBSplineBasis:
    """Test B-spline basis construction."""

    def test_basic_construction(self):
        """Test basic B-spline basis construction."""
        x = np.linspace(0, 1, 20)
        B, knots = b_spline_basis(x, 0, 1, nseg=5, degree=3)
        
        assert sp.issparse(B)
        assert B.shape[0] == len(x)
        assert B.shape[1] == 5 + 3  # nseg + degree
        assert len(knots) == 5 + 1 + 2 * 3  # nseg + 1 + 2*degree
        
    def test_partition_of_unity(self):
        """Test that B-spline basis forms partition of unity."""
        x = np.linspace(0, 1, 50)
        B, _ = b_spline_basis(x, 0, 1, nseg=10, degree=3)
        
        # Sum of basis functions should be 1 (within numerical tolerance)
        row_sums = np.array(B.sum(axis=1)).flatten()
        assert_allclose(row_sums, 1.0, rtol=1e-12)
        
    def test_different_degrees(self):
        """Test basis construction with different degrees."""
        x = np.linspace(0, 1, 20)
        
        for degree in [1, 2, 3, 4]:
            B, knots = b_spline_basis(x, 0, 1, nseg=5, degree=degree)
            assert B.shape[1] == 5 + degree
            
    def test_boundary_values(self):
        """Test behavior at domain boundaries."""
        x = np.array([0.0, 1.0])
        B, _ = b_spline_basis(x, 0, 1, nseg=5, degree=3)
        
        # Should have non-zero values at boundaries
        assert B[0, :].sum() > 0
        assert B[1, :].sum() > 0
        
    def test_outside_domain(self):
        """Test behavior outside the specified domain."""
        x = np.array([-0.1, 1.1])
        B, _ = b_spline_basis(x, 0, 1, nseg=5, degree=3)
        
        # Should handle extrapolation gracefully
        assert B.shape[0] == 2
        assert not np.any(np.isnan(B.data))


class TestBSplineDerivativeBasis:
    """Test B-spline derivative basis construction."""

    def test_first_derivative_basis(self):
        """Test first derivative basis construction."""
        x = np.linspace(0, 1, 20)
        knots = np.linspace(-0.3, 1.3, 15)  # Mock knots
        
        B_deriv, _ = b_spline_derivative_basis(x, 0, 1, nseg=5, degree=3, 
                                             derivative_order=1, knots=knots)
        
        assert sp.issparse(B_deriv)
        assert B_deriv.shape[0] == len(x)
        
    def test_second_derivative_basis(self):
        """Test second derivative basis construction."""
        x = np.linspace(0, 1, 20)
        knots = np.linspace(-0.3, 1.3, 15)
        
        B_deriv, _ = b_spline_derivative_basis(x, 0, 1, nseg=5, degree=3,
                                             derivative_order=2, knots=knots)
        
        assert sp.issparse(B_deriv)
        assert B_deriv.shape[0] == len(x)
        
    def test_derivative_consistency(self):
        """Test that derivative basis is consistent with numerical differentiation."""
        # This is a more complex test that would require numerical derivatives
        # For now, just test that the function runs without error
        x = np.linspace(0.1, 0.9, 10)
        knots = np.linspace(-0.3, 1.3, 15)
        
        B_deriv, _ = b_spline_derivative_basis(x, 0, 1, nseg=5, degree=3,
                                             derivative_order=1, knots=knots)
        
        # Should not contain NaN or inf values
        assert not np.any(np.isnan(B_deriv.data))
        assert not np.any(np.isinf(B_deriv.data))


class TestBasisEdgeCases:
    """Test edge cases for basis functions."""

    def test_single_point(self):
        """Test basis evaluation at single point."""
        x = np.array([0.5])
        B, knots = b_spline_basis(x, 0, 1, nseg=5, degree=3)
        
        assert B.shape[0] == 1
        assert np.abs(B.sum() - 1.0) < 1e-12  # Partition of unity
        
    def test_very_small_domain(self):
        """Test basis on very small domain."""
        x = np.linspace(0, 1e-6, 10)
        B, _ = b_spline_basis(x, 0, 1e-6, nseg=3, degree=2)
        
        assert B.shape[0] == 10
        assert not np.any(np.isnan(B.data))
        
    def test_large_degree(self):
        """Test basis with large degree."""
        x = np.linspace(0, 1, 20)
        B, _ = b_spline_basis(x, 0, 1, nseg=10, degree=6)
        
        assert B.shape[1] == 10 + 6
        row_sums = np.array(B.sum(axis=1)).flatten()
        assert_allclose(row_sums, 1.0, rtol=1e-10)


class TestKnotConstruction:
    """Test knot vector construction."""

    def test_knot_ordering(self):
        """Test that knots are properly ordered."""
        from psplines.basis import _make_knots
        
        knots = _make_knots(0, 1, nseg=5, degree=3)
        
        # Knots should be non-decreasing
        assert np.all(np.diff(knots) >= 0)
        
        # Should have proper multiplicities at boundaries
        assert np.sum(knots == knots[0]) == 3  # degree multiplicity
        assert np.sum(knots == knots[-1]) == 3  # degree multiplicity
        
    def test_knot_count(self):
        """Test correct number of knots."""
        from psplines.basis import _make_knots
        
        nseg, degree = 5, 3
        knots = _make_knots(0, 1, nseg, degree)
        
        expected_count = (nseg + 1) + 2 * degree
        assert len(knots) == expected_count


if __name__ == "__main__":
    pytest.main([__file__])
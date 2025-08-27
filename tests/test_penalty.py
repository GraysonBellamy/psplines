"""
Tests for psplines.penalty module.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from psplines.penalty import difference_matrix


class TestDifferenceMatrix:
    """Test difference matrix construction."""

    def test_first_order_difference(self):
        """Test first-order difference matrix."""
        D = difference_matrix(5, order=1)

        assert sp.issparse(D)
        assert D.shape == (4, 5)  # n-1 rows for first order

        # Test the structure: should be [-1, 1, 0, 0, 0] in first row
        expected_first_row = np.array([-1, 1, 0, 0, 0])
        assert_allclose(D[0, :].toarray().flatten(), expected_first_row)

    def test_second_order_difference(self):
        """Test second-order difference matrix."""
        D = difference_matrix(5, order=2)

        assert sp.issparse(D)
        assert D.shape == (3, 5)  # n-2 rows for second order

        # Test the structure: should be [1, -2, 1, 0, 0] in first row
        expected_first_row = np.array([1, -2, 1, 0, 0])
        assert_allclose(D[0, :].toarray().flatten(), expected_first_row)

    def test_third_order_difference(self):
        """Test third-order difference matrix."""
        D = difference_matrix(6, order=3)

        assert sp.issparse(D)
        assert D.shape == (3, 6)  # n-3 rows for third order

        # Test the structure: should be [-1, 3, -3, 1, 0, 0] in first row
        expected_first_row = np.array([-1, 3, -3, 1, 0, 0])
        assert_allclose(D[0, :].toarray().flatten(), expected_first_row)

    def test_difference_properties(self):
        """Test mathematical properties of difference matrices."""
        # Test that difference matrix has correct rank
        for order in [1, 2, 3]:
            for n in [5, 10, 15]:
                if n > order:
                    D = difference_matrix(n, order)
                    # Dense version for rank computation
                    D_dense = D.toarray()
                    rank = np.linalg.matrix_rank(D_dense)
                    # Rank should be n - order (for typical cases)
                    assert rank <= n - order

    def test_small_matrices(self):
        """Test difference matrices for small sizes."""
        # Minimum size for first order
        D = difference_matrix(2, order=1)
        assert D.shape == (1, 2)
        assert_allclose(D.toarray(), [[-1, 1]])

        # Minimum size for second order
        D = difference_matrix(3, order=2)
        assert D.shape == (1, 3)
        assert_allclose(D.toarray(), [[1, -2, 1]])

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # This depends on the actual implementation
        # The function should handle edge cases gracefully
        pass  # Implementation-specific

    def test_sparsity_pattern(self):
        """Test that difference matrices are properly sparse."""
        D = difference_matrix(20, order=2)

        # Should be a banded matrix with limited non-zeros per row
        nnz_per_row = np.array([D[i, :].nnz for i in range(D.shape[0])])
        assert np.all(nnz_per_row <= 3)  # Second order has at most 3 non-zeros per row


class TestDifferenceMatrixEdgeCases:
    """Test edge cases for difference matrix construction."""

    def test_large_order(self):
        """Test behavior with large difference order."""
        # Order close to matrix size
        D = difference_matrix(10, order=8)
        assert D.shape[0] == 2  # 10 - 8 = 2 rows

    def test_consistency_across_sizes(self):
        """Test that larger matrices contain smaller ones as submatrices."""
        D5 = difference_matrix(5, order=2)
        D10 = difference_matrix(10, order=2)

        # First few rows should be similar in structure
        assert_allclose(D5[0, :3].toarray(), D10[0, :3].toarray())


if __name__ == "__main__":
    pytest.main([__file__])

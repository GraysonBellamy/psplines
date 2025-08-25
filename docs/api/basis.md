# Basis Functions API

The basis module provides functions for constructing B-spline basis matrices and their derivatives.

## Functions

::: psplines.basis.b_spline_basis
    options:
      show_source: true
      heading_level: 3

::: psplines.basis.b_spline_derivative_basis
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Basic B-spline Basis

```python
import numpy as np
from psplines.basis import b_spline_basis

# Create evaluation points
x = np.linspace(0, 1, 50)

# Generate B-spline basis matrix
B, knots = b_spline_basis(x, xl=0, xr=1, nseg=10, degree=3)

print(f"Basis shape: {B.shape}")  # (50, 13) for degree=3, nseg=10
print(f"Number of knots: {len(knots)}")  # 17 knots total
```

### Derivative Basis

```python
from psplines.basis import b_spline_derivative_basis

# Generate first derivative basis
B_deriv, knots = b_spline_derivative_basis(
    x, xl=0, xr=1, nseg=10, degree=3, 
    derivative_order=1, knots=knots
)

print(f"Derivative basis shape: {B_deriv.shape}")
```

## Mathematical Background

### B-spline Basis Construction

B-splines of degree $d$ are defined recursively:

$$B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i,d}(x) = \frac{x - t_i}{t_{i+d} - t_i} B_{i,d-1}(x) + \frac{t_{i+d+1} - x}{t_{i+d+1} - t_{i+1}} B_{i+1,d-1}(x)$$

where $t_i$ are the knots.

### Knot Vector Construction

For $n$ segments and degree $d$:

- Interior knots: $n+1$ equally spaced points
- Boundary knots: $d$ repeated knots at each end
- Total knots: $n + 1 + 2d$
- Number of basis functions: $n + d$

### Properties

- **Local support**: Each basis function is non-zero over at most $d+1$ knot spans
- **Partition of unity**: $\sum_i B_{i,d}(x) = 1$ for all $x$
- **Non-negativity**: $B_{i,d}(x) \geq 0$ for all $i, x$
- **Smoothness**: $B_{i,d}(x)$ is $C^{d-1}$ continuous

### Derivative Computation

The $k$-th derivative of a B-spline is computed using the recursive formula:

$$\frac{d^k}{dx^k} B_{i,d}(x) = \frac{d!}{(d-k)!} \sum_{j=0}^k (-1)^{k-j} \binom{k}{j} \frac{B_{i+j,d-k}(x)}{(t_{i+d+1-j} - t_{i+j})^k}$$

## Implementation Details

### Sparse Matrix Format

- All basis matrices are returned as `scipy.sparse.csr_matrix`
- Efficient storage for large problems
- Fast matrix-vector operations

### Numerical Considerations

- Uses SciPy's `BSpline` class for robust evaluation
- Handles edge cases at boundaries
- Maintains numerical stability for high-degree splines

### Performance Tips

1. **Reuse knots**: Pass the same knot vector to derivative functions
2. **Batch evaluation**: Evaluate multiple points simultaneously
3. **Appropriate degree**: Degree 3 (cubic) is usually sufficient
4. **Segment selection**: More segments increase flexibility but computational cost
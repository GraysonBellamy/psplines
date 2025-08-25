# Penalty Matrices API

The penalty module provides functions for constructing difference penalty matrices used in P-spline smoothing.

## Functions

::: psplines.penalty.difference_matrix
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### First-Order Differences

```python
import numpy as np
from psplines.penalty import difference_matrix

# First-order difference matrix
D1 = difference_matrix(n=5, order=1)
print(D1.toarray())
# [[-1  1  0  0  0]
#  [ 0 -1  1  0  0]
#  [ 0  0 -1  1  0]
#  [ 0  0  0 -1  1]]
```

### Second-Order Differences

```python
# Second-order difference matrix  
D2 = difference_matrix(n=5, order=2)
print(D2.toarray())
# [[ 1 -2  1  0  0]
#  [ 0  1 -2  1  0]
#  [ 0  0  1 -2  1]]
```

### Higher-Order Differences

```python
# Third-order difference matrix
D3 = difference_matrix(n=6, order=3)
print(D3.toarray())
# [[-1  3 -3  1  0  0]
#  [ 0 -1  3 -3  1  0]
#  [ 0  0 -1  3 -3  1]]
```

## Mathematical Background

### Difference Operators

The $p$-th order difference operator $\Delta^p$ is defined recursively:

- $\Delta^0 \alpha_i = \alpha_i$ (identity)
- $\Delta^1 \alpha_i = \alpha_{i+1} - \alpha_i$ (first difference)
- $\Delta^p \alpha_i = \Delta^{p-1} \alpha_{i+1} - \Delta^{p-1} \alpha_i$

### Matrix Representation

The difference matrix $D_p$ has entries such that $(D_p \alpha)_i = \Delta^p \alpha_i$.

For first-order differences ($p=1$):
$$D_1 = \begin{pmatrix}
-1 & 1 & 0 & \cdots & 0 \\
0 & -1 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & -1 & 1
\end{pmatrix}$$

For second-order differences ($p=2$):
$$D_2 = \begin{pmatrix}
1 & -2 & 1 & 0 & \cdots & 0 \\
0 & 1 & -2 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & 1 & -2 & 1
\end{pmatrix}$$

### Properties

- **Dimension**: For $n$ coefficients, $D_p$ has size $(n-p) \times n$
- **Rank**: The rank of $D_p$ is $n-p$ (for typical cases)
- **Sparsity**: Each row has at most $p+1$ non-zero entries
- **Banded structure**: All non-zeros lie within $p+1$ diagonals

### Penalty Interpretation

The penalty term $\lambda \|D_p \alpha\|^2$ penalizes:

- **$p=1$**: Large first differences (rough slopes)
- **$p=2$**: Large second differences (rough curvature)  
- **$p=3$**: Large third differences (rough rate of curvature change)

## Implementation Details

### Sparse Storage

- Returns `scipy.sparse.csr_matrix` for efficient storage
- Memory usage: $O((n-p) \times (p+1))$ instead of $O((n-p) \times n)$
- Fast matrix-vector multiplication

### Numerical Properties

- **Condition number**: Increases with penalty order
- **Null space**: $D_p$ has a null space of dimension $p$
- **Regularization**: The penalty $\lambda \|D_p \alpha\|^2$ regularizes the fit

### Construction Algorithm

The implementation uses efficient sparse matrix construction:

1. Compute binomial coefficients for the difference operator
2. Build row indices, column indices, and data arrays
3. Construct sparse matrix in COO format
4. Convert to CSR format for efficient operations

## Usage in P-Splines

In the P-spline objective function:
$$\min_\alpha \|y - B\alpha\|^2 + \lambda \|D_p \alpha\|^2$$

The penalty matrix appears as $P = \lambda D_p^T D_p$, leading to the linear system:
$$(B^T B + \lambda D_p^T D_p) \alpha = B^T y$$

### Penalty Order Selection

- **Order 1**: Good for piecewise linear trends
- **Order 2**: Most common choice, penalizes curvature
- **Order 3**: For very smooth curves, penalizes jerk
- **Higher orders**: Rarely needed, may cause numerical issues
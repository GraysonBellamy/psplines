# Penalty Matrices API

The penalty module provides functions for constructing difference penalty matrices used in P-spline smoothing.

## Functions

::: psplines.penalty.difference_matrix
    options:
      show_source: true
      heading_level: 3

::: psplines.penalty.asymmetric_penalty_matrix
    options:
      show_source: true
      heading_level: 3

::: psplines.penalty.variable_penalty_matrix
    options:
      show_source: true
      heading_level: 3

::: psplines.penalty.adaptive_penalty_matrix
    options:
      show_source: true
      heading_level: 3

::: psplines.penalty.divided_difference_matrix
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

### Asymmetric Penalty for Shape Constraints

```python
import numpy as np
from psplines.penalty import asymmetric_penalty_matrix

# Coefficients that violate monotonicity (decreasing segment)
alpha = np.array([1.0, 2.0, 3.0, 2.5, 4.0, 5.0])

# Build penalty that targets decreasing violations
P = asymmetric_penalty_matrix(alpha, "increasing")
print(P.toarray())  # non-zero only at violation (index 3)

# With a selective domain mask (only enforce on first 3 diffs)
mask = np.array([True, True, True, False, False])
P_masked = asymmetric_penalty_matrix(alpha, "increasing", mask=mask)
```

### Variable (Exponential) Penalty Weights

```python
from psplines.penalty import variable_penalty_matrix

# Standard penalty (γ = 0)
P0 = variable_penalty_matrix(n=20, order=2, gamma=0.0)

# Heavier penalty toward the right boundary
P_right = variable_penalty_matrix(n=20, order=2, gamma=5.0)

# Heavier penalty toward the left boundary
P_left = variable_penalty_matrix(n=20, order=2, gamma=-5.0)
```

### Adaptive Per-Difference Weights

```python
from psplines.penalty import adaptive_penalty_matrix

# Per-difference weights (e.g. from a secondary smoothing pass)
weights = np.ones(18)        # n=20, order=2 → 18 differences
weights[5:10] = 0.1          # less penalty in the middle

P = adaptive_penalty_matrix(n=20, order=2, weights=weights)
```

### Divided Differences for Non-Uniform Spacing

When data are non-uniformly spaced, the standard difference matrix treats all gaps
as equal.  The divided-difference matrix weights each difference by the reciprocal
of the gap in $x$, giving a roughness penalty in the natural units of the
independent variable:

```python
import numpy as np
from psplines.penalty import divided_difference_matrix

# Non-uniform sample positions
x = np.array([0.0, 1.0, 4.0, 5.0, 10.0])

# First-order divided differences
D1 = divided_difference_matrix(x, order=1)
print(D1.toarray())
# [[-1.    1.    0.    0.    0.  ]
#  [ 0.   -0.33  0.33  0.    0.  ]
#  [ 0.    0.   -1.    1.    0.  ]
#  [ 0.    0.    0.   -0.2   0.2 ]]

# Second-order divided differences
D2 = divided_difference_matrix(x, order=2)
print(D2.shape)  # (3, 5)

# Key property: second divided differences annihilate linear functions
z_linear = 2.0 * x + 3.0
print(D2 @ z_linear)  # ≈ [0, 0, 0]

# For quadratics, the result is constant
z_quad = x ** 2
print(D2 @ z_quad)    # ≈ [2, 2, 2]
```

On uniformly spaced $x$, `divided_difference_matrix` is proportional to the standard `difference_matrix`:

```python
x_uniform = np.linspace(0, 1, 20)
h = x_uniform[1] - x_uniform[0]

D_std = difference_matrix(20, order=2)
D_div = divided_difference_matrix(x_uniform, order=2)

# D_div ≈ D_std / h²
np.allclose(D_div.toarray(), D_std.toarray() / h**2)  # True
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

### Shape Constraints via Asymmetric Penalties (§8.7)

Shape constraints (monotonicity, convexity, etc.) are enforced via the asymmetric
penalty of Eilers & Marx (2021, eq. 8.14–8.15).  Define a diagonal matrix
$V = \text{diag}(v)$ where $v_j = 1$ when the $j$-th difference **violates** the
constraint and $v_j = 0$ otherwise.  The shape penalty is:

$$P_{\text{shape}} = \kappa \, D_d^T V \, D_d$$

where $d$ is the implied difference order (1 for monotonicity, 2 for convexity,
0 for non-negativity) and $\kappa$ is a large constant (default $10^8$).
This penalty is re-evaluated at each iteration, updating $V$ based on the
current solution, until the coefficients converge.

Supported constraint types:

| Type | Difference order | Penalises |
|------|-----------------|-----------|
| `increasing` | 1 | $\Delta\alpha_j < 0$ |
| `decreasing` | 1 | $\Delta\alpha_j > 0$ |
| `convex` | 2 | $\Delta^2\alpha_j < 0$ |
| `concave` | 2 | $\Delta^2\alpha_j > 0$ |
| `nonneg` | 0 | $\alpha_j < 0$ |

A **selective domain mask** restricts the penalty to a sub-range of the
coefficient indices so that the constraint applies only in a chosen region
of the $x$-domain.

### Variable and Adaptive Penalty Weights (§8.8)

#### Exponential variable weights

Replace the standard penalty $D^T D$ with $D^T V D$ where
$V = \text{diag}\!\bigl(\exp(\gamma j/m)\bigr)$, $j = 0, \ldots, m-1$.
This yields heavier/lighter regularisation near one boundary:

$$P(\gamma) = D_p^T \, \text{diag}\!\bigl(e^{\gamma j/m}\bigr) \, D_p$$

#### Adaptive per-difference weights

A secondary B-spline basis of $K_w$ segments is fitted to the log-residuals to
obtain per-difference weights $w_j$.  The penalty becomes $D^T \text{diag}(w) D$,
allowing spatially varying smoothness:

$$P_{\text{adapt}} = D_p^T \, \text{diag}(w_1, \ldots, w_{m}) \, D_p$$

### Penalty Order Selection

- **Order 1**: Good for piecewise linear trends
- **Order 2**: Most common choice, penalizes curvature
- **Order 3**: For very smooth curves, penalizes jerk
- **Higher orders**: Rarely needed, may cause numerical issues

### Divided Differences for Non-Uniform Data

The standard difference matrix $D_p$ assumes equal spacing between coefficients.
When applied to the Whittaker smoother (where $B = I$ and coefficients **are** the
data values), non-uniform spacing in $x$ requires correcting the differences.

The divided-difference operator $D_x$ replaces each finite difference with

$$(D_x z)_i = \frac{z_{i+1} - z_i}{x_{i+1} - x_i}$$

and second-order divided differences are built recursively via midpoints.  The
penalty $\lambda \|D_x z\|^2$ then measures roughness in the units of $x$ rather
than index position.  This is used automatically by
[`WhittakerSmoother`](whittaker.md) when non-uniform spacing is detected.

See also: [`divided_difference_matrix`](#psplines.penalty.divided_difference_matrix).
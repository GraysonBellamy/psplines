# Mathematical Background

This section provides the mathematical foundation for P-splines, covering the theory, algorithms, and computational aspects.

## Overview

P-splines (Penalized B-splines) combine the flexibility of B-spline basis functions with difference penalties to create a powerful smoothing framework. The method was introduced by Eilers and Marx (1996) and has become a cornerstone of modern statistical smoothing.

## The P-Spline Model

### Basic Formulation

Given data points $(x_i, y_i)$ for $i = 1, \ldots, n$, P-splines fit a smooth function $f(x)$ by solving:

$$\min_\alpha \|y - B\alpha\|^2 + \lambda \|D_p \alpha\|^2$$

where:
- $y = (y_1, \ldots, y_n)^T$ is the response vector
- $B$ is the $n \times m$ B-spline basis matrix
- $\alpha = (\alpha_1, \ldots, \alpha_m)^T$ are the B-spline coefficients  
- $D_p$ is the $(m-p) \times m$ $p$-th order difference matrix
- $\lambda > 0$ is the smoothing parameter

The smooth function is then given by:
$$f(x) = \sum_{j=1}^m \alpha_j B_j(x)$$

### Matrix Form Solution

The solution to the P-spline optimization problem is:
$$\hat{\alpha} = (B^T B + \lambda D_p^T D_p)^{-1} B^T y$$

And the fitted values are:
$$\hat{y} = B\hat{\alpha} = S_\lambda y$$

where $S_\lambda = B(B^T B + \lambda D_p^T D_p)^{-1} B^T$ is the smoothing matrix.

## B-Spline Basis Functions

### Definition

B-splines of degree $d$ are defined recursively using the Cox-de Boor formula:

**Degree 0 (indicator functions):**
$$B_{i,0}(x) = \begin{cases} 
1 & \text{if } t_i \leq x < t_{i+1} \\
0 & \text{otherwise}
\end{cases}$$

**Higher degrees ($d \geq 1$):**
$$B_{i,d}(x) = \frac{x - t_i}{t_{i+d} - t_i} B_{i,d-1}(x) + \frac{t_{i+d+1} - x}{t_{i+d+1} - t_{i+1}} B_{i+1,d-1}(x)$$

where $\{t_i\}$ is the knot sequence.

### Knot Vector Construction

For equally-spaced knots on interval $[a,b]$ with $K$ segments:

1. **Interior knots**: $\{a, a + \frac{b-a}{K}, a + 2\frac{b-a}{K}, \ldots, b\}$ ($K+1$ knots)
2. **Extended knots**: Add $d$ knots at each boundary:
   - Left: $\{a-d\frac{b-a}{K}, \ldots, a-\frac{b-a}{K}\}$
   - Right: $\{b+\frac{b-a}{K}, \ldots, b+d\frac{b-a}{K}\}$
3. **Total knots**: $K + 1 + 2d$
4. **Basis functions**: $K + d$

### Properties

**Local Support**: Each $B_{i,d}(x)$ is non-zero only on $[t_i, t_{i+d+1})$.

**Partition of Unity**: $\sum_{i} B_{i,d}(x) = 1$ for all $x$ in the interior.

**Non-negativity**: $B_{i,d}(x) \geq 0$ for all $i,x$.

**Smoothness**: $B_{i,d} \in C^{d-1}$, i.e., $(d-1)$ times continuously differentiable.

### Derivatives

The $k$-th derivative of a B-spline satisfies:
$$\frac{d^k}{dx^k} B_{i,d}(x) = \frac{d!}{(d-k)!} \sum_{j=0}^k (-1)^{k-j} \binom{k}{j} \frac{B_{i+j,d-k}(x)}{(t_{i+d+1-j} - t_{i+j})^k}$$

This allows efficient computation of derivative basis matrices.

## Difference Penalties

### Difference Operators

The $p$-th order forward difference operator is defined recursively:
- $\Delta^0 \alpha_i = \alpha_i$
- $\Delta^1 \alpha_i = \alpha_{i+1} - \alpha_i$
- $\Delta^p \alpha_i = \Delta^{p-1} \alpha_{i+1} - \Delta^{p-1} \alpha_i$

### Difference Matrices

The difference matrix $D_p$ has entries such that $(D_p \alpha)_i = \Delta^p \alpha_i$.

**First-order differences ($p=1$):**
$$D_1 = \begin{pmatrix}
-1 & 1 & 0 & \cdots & 0 \\
0 & -1 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & -1 & 1
\end{pmatrix}$$

**Second-order differences ($p=2$):**
$$D_2 = \begin{pmatrix}
1 & -2 & 1 & 0 & \cdots & 0 \\
0 & 1 & -2 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & 1 & -2 & 1
\end{pmatrix}$$

**General form**: For $p$-th order differences, row $i$ of $D_p$ contains the binomial coefficients:
$$D_p[i, i+j] = (-1)^{p-j} \binom{p}{j}, \quad j = 0, 1, \ldots, p$$

### Penalty Interpretation

The penalty $\|D_p \alpha\|^2$ controls different aspects of smoothness:

- **$p=1$**: $\sum_{i=1}^{m-1} (\alpha_{i+1} - \alpha_i)^2$ penalizes large differences
- **$p=2$**: $\sum_{i=1}^{m-2} (\alpha_{i+2} - 2\alpha_{i+1} + \alpha_i)^2$ penalizes large second differences
- **$p=3$**: Penalizes large third differences (changes in curvature)

## Statistical Properties

### Degrees of Freedom

The effective degrees of freedom is:
$$\text{df}(\lambda) = \text{tr}(S_\lambda) = \text{tr}[B(B^T B + \lambda D_p^T D_p)^{-1} B^T]$$

This measures the complexity of the fitted model.

### Bias-Variance Decomposition

The mean squared error can be decomposed as:
$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

where:
- **Bias** increases with $\lambda$ (more smoothing)
- **Variance** decreases with $\lambda$ (less flexibility)
- **Noise** is irreducible error

### Uncertainty Quantification

#### Analytical Standard Errors

Under the assumption $y \sim N(B\alpha^*, \sigma^2 I)$, the covariance matrix of $\hat{\alpha}$ is:
$$\text{Cov}(\hat{\alpha}) = \sigma^2 (B^T B + \lambda D_p^T D_p)^{-1} B^T B (B^T B + \lambda D_p^T D_p)^{-1}$$

For predictions $f(x_0) = b(x_0)^T \hat{\alpha}$ where $b(x_0)$ is the basis vector at $x_0$:
$$\text{Var}(f(x_0)) = \sigma^2 b(x_0)^T (B^T B + \lambda D_p^T D_p)^{-1} b(x_0)$$

#### Bootstrap Methods

Parametric bootstrap generates replicates:
$$y^{(b)} = \hat{y} + \epsilon^{(b)}, \quad \epsilon^{(b)} \sim N(0, \hat{\sigma}^2 I)$$

Each replicate gives $\hat{\alpha}^{(b)}$ and $\hat{f}^{(b)}(x_0)$, allowing empirical variance estimation.

## Computational Aspects

### Sparse Matrix Exploitation

- **Basis matrix $B$**: Typically has $(d+1)$ non-zeros per row
- **Penalty matrix $D_p^T D_p$**: Banded with bandwidth $(2p+1)$
- **System matrix**: $B^T B + \lambda D_p^T D_p$ is sparse and symmetric

### Numerical Solution

The linear system $(B^T B + \lambda D_p^T D_p) \alpha = B^T y$ is solved using:
1. **Cholesky decomposition** for small-medium problems
2. **Sparse direct solvers** for large sparse problems  
3. **Iterative methods** for very large problems

### Complexity Analysis

- **Matrix construction**: $O(nm)$ for basis, $O(m^2)$ for penalties
- **System solution**: $O(m^3)$ dense, $O(m^{3/2})$ sparse (typically)
- **Total complexity**: $O(nm + m^2)$ to $O(nm + m^3)$ depending on sparsity

## Parameter Selection Theory

### Generalized Cross-Validation (GCV)

GCV minimizes:
$$\text{GCV}(\lambda) = \frac{n \|y - S_\lambda y\|^2}{(n - \text{tr}(S_\lambda))^2}$$

This approximates leave-one-out cross-validation efficiently.

### Akaike Information Criterion (AIC)

AIC balances fit and complexity:
$$\text{AIC}(\lambda) = n \log\left(\frac{\|y - S_\lambda y\|^2}{n}\right) + 2 \cdot \text{tr}(S_\lambda)$$

### L-Curve Method

Plots $\log(\|D_p \hat{\alpha}\|^2)$ vs $\log(\|y - B\hat{\alpha}\|^2)$ and selects the point of maximum curvature:
$$\kappa(\lambda) = \frac{2(\rho' \eta'' - \rho'' \eta')}{(\rho'^2 + \eta'^2)^{3/2}}$$

where $\rho(\lambda) = \log(\|y - B\hat{\alpha}\|^2)$ and $\eta(\lambda) = \log(\|D_p \hat{\alpha}\|^2)$.

## Bayesian P-Splines

The `bayes_fit()` method offers two modes controlled by the `adaptive` parameter.

### Standard Model (adaptive=False, default)

This implements the Bayesian P-spline of §3.5 (Eilers & Marx 2021; Lang & Brezger 2003) with a **single** penalty parameter:

$$y \mid \alpha, \sigma \sim N(B\alpha, \sigma^2 I)$$
$$\alpha \mid \lambda \sim N\!\bigl(0,\; (\lambda D_p^T D_p + \epsilon I)^{-1}\bigr)$$
$$\lambda \sim \text{Gamma}(a, b)$$
$$\sigma \sim \text{InverseGamma}(c, d)$$

where $\epsilon = 10^{-6}$ is a small jitter for numerical stability (since $D_p^T D_p$ is rank-deficient with a null space of dimension $p$).

### Posterior Distribution

The posterior for $\alpha$ is:
$$\alpha \mid y, \lambda, \sigma^2 \sim N(\mu_\alpha, \Sigma_\alpha)$$

where:
$$\Sigma_\alpha = (B^T B / \sigma^2 + \lambda D_p^T D_p + \epsilon I)^{-1}$$
$$\mu_\alpha = \Sigma_\alpha B^T y / \sigma^2$$

### Adaptive Model (adaptive=True)

This uses **per-difference** penalty parameters for spatially varying smoothness (related to §8.8):

$$\lambda_j \sim \text{Gamma}(a, b), \quad j = 1, \ldots, m-p$$
$$\alpha \mid \lambda \sim N\!\bigl(0,\; (D_p^T \Lambda D_p + \epsilon I)^{-1}\bigr)$$

where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_{m-p})$. Each $\lambda_j$ controls the penalty on the $j$-th difference, allowing different regions of the curve to have different amounts of smoothness.

### Markov Chain Monte Carlo

Both modes use PyMC's NUTS sampler (Hamiltonian Monte Carlo) to draw from the joint posterior. The posterior mean of $\alpha$ gives the point estimate, and posterior quantiles provide credible intervals.

### Prior Sensitivity

The book (§3.5) notes that Lang & Brezger's inverse-Gamma priors for variances can favor over-smoothing depending on hyperparameter choices (Jullion & Lambert, 2007). The default priors $\text{Gamma}(2, 0.1)$ for $\lambda$ and $\text{InverseGamma}(2, 1)$ for $\sigma$ are moderately informative. Users should consider adjusting these for their specific application.

## Extensions and Variants

### Multidimensional P-Splines

For functions $f(x_1, x_2)$, use tensor product bases:
$$f(x_1, x_2) = \sum_{i,j} \alpha_{ij} B_i(x_1) B_j(x_2)$$

with penalties on both dimensions.

### Non-Gaussian Responses

For exponential family responses:
$$\min_\alpha -\ell(\mu) + \lambda \|D_p \alpha\|^2$$

where $\mu = g^{-1}(B\alpha)$ and $g$ is the link function.

### Shape Constraints via Asymmetric Penalties (§8.7)

Shape constraints (monotonicity, convexity, concavity, non-negativity) are enforced
through the iterative asymmetric penalty framework of Eilers & Marx (2021,
equations 8.14–8.15).

#### Formulation

Define a diagonal indicator matrix $V = \text{diag}(v_1, \ldots, v_{m-d})$ where

$$v_j = \begin{cases}
1 & \text{if the } j\text{-th difference violates the constraint}, \\
0 & \text{otherwise.}
\end{cases}$$

The shape-specific penalty is:

$$P_{\text{shape}} = \kappa \, D_d^T V \, D_d$$

where $d$ is the difference order implied by the constraint type and $\kappa$ is a
large constant (default $10^8$).  This is added to the standard penalty:

$$(B^T W B + \lambda D_p^T D_p + \kappa \, D_d^T V \, D_d) \, \alpha = B^T W y$$

#### Iterative Algorithm

1. Solve the unconstrained system to obtain $\hat\alpha^{(0)}$.
2. Compute $V^{(k)}$ from $\hat\alpha^{(k)}$ (mark violations).
3. Solve $(B^TWB + \lambda D_p^TD_p + \kappa D_d^TV^{(k)}D_d)\alpha = B^TWy$ to get $\hat\alpha^{(k+1)}$.
4. Repeat steps 2–3 until $\|\hat\alpha^{(k+1)} - \hat\alpha^{(k)}\| < \varepsilon$ or a maximum number of iterations.

#### Constraint Types

| Type | Diff order $d$ | Violation condition |
|------|:-:|---|
| Increasing | 1 | $\Delta\alpha_j < 0$ |
| Decreasing | 1 | $\Delta\alpha_j > 0$ |
| Convex | 2 | $\Delta^2\alpha_j < 0$ |
| Concave | 2 | $\Delta^2\alpha_j > 0$ |
| Non-negative | 0 | $\alpha_j < 0$ |

Multiple constraints can be stacked by summing their respective $\kappa D_d^TVD_d$
contributions.  A **selective domain mask** restricts the diagonal entries of $V$ so
that the constraint is enforced only in a chosen sub-range of the coefficient index
(corresponding to a sub-range of $x$).

### Variable and Adaptive Penalties (§8.8)

#### Exponential Variable Weights

Replace the standard $D^TD$ with a weighted version:

$$P(\gamma) = D_p^T \, \text{diag}\!\bigl(e^{\gamma \cdot 0/m},\; e^{\gamma \cdot 1/m},\; \ldots,\; e^{\gamma(m-1)/m}\bigr) \, D_p$$

where $m = n_{\text{coef}} - p$.  The parameter $\gamma$ controls the spatial
distribution of penalty weight:

- $\gamma > 0$: heavier regularisation toward the right boundary
- $\gamma < 0$: heavier regularisation toward the left boundary
- $\gamma = 0$: recovers the standard penalty $D^TD$

The optimal $(\lambda, \gamma)$ pair can be selected by a 2-D grid search over
GCV or AIC — implemented as `variable_penalty_cv()`.

#### Nonparametric Adaptive Penalty

A secondary B-spline basis $B_w$ of $K_w$ segments is used to model the
log squared differences of the current coefficients:

$$\log\bigl((\Delta^p \hat\alpha_j)^2 + \epsilon\bigr) \approx B_w \beta$$

Fitting $\beta$ with a separate smoothing parameter $\lambda_w$ gives smooth
per-difference weights:

$$w_j = \exp(B_w \hat\beta)_j$$

The primary penalty becomes $D_p^T \text{diag}(w) D_p$ and the procedure
alternates between estimating $\alpha$ (given $w$) and estimating $w$ (given
$\alpha$) until convergence.  This allows the fitted curve to be flexible in
regions of rapid change and smooth elsewhere — fully data-driven without
specifying the form of the weight function.

## References

1. Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.

2. Eilers, P. H. C., & Marx, B. D. (2021). *Practical Smoothing: The Joys of P-splines*. Cambridge University Press.

3. Ruppert, D., Wand, M. P., & Carroll, R. J. (2003). *Semiparametric Regression*. Cambridge University Press.

4. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*. Chapman and Hall/CRC.
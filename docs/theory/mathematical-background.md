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

### Model Specification

In the Bayesian framework:
$$y \mid \alpha, \sigma^2 \sim N(B\alpha, \sigma^2 I)$$
$$\alpha \mid \tau \sim N(0, (\lambda D_p^T D_p)^{-1})$$
$$\lambda \sim \text{Gamma}(a, b)$$
$$\sigma^2 \sim \text{InverseGamma}(c, d)$$

### Posterior Distribution

The posterior for $\alpha$ is:
$$\alpha \mid y, \lambda, \sigma^2 \sim N(\mu_\alpha, \Sigma_\alpha)$$

where:
$$\Sigma_\alpha = \sigma^2 (B^T B + \lambda D_p^T D_p)^{-1}$$
$$\mu_\alpha = \Sigma_\alpha B^T y / \sigma^2$$

### Markov Chain Monte Carlo

MCMC sampling alternates between:
1. **Update $\alpha$**: Multivariate normal conditional
2. **Update $\lambda$**: Gamma conditional (conjugate prior)
3. **Update $\sigma^2$**: Inverse-Gamma conditional (conjugate prior)

This provides full posterior distributions for uncertainty quantification.

## Extensions and Variants

### Multidimensional P-Splines

For functions $f(x_1, x_2)$, use tensor product bases:
$$f(x_1, x_2) = \sum_{i,j} \alpha_{ij} B_i(x_1) B_j(x_2)$$

with penalties on both dimensions.

### Non-Gaussian Responses

For exponential family responses:
$$\min_\alpha -\ell(\mu) + \lambda \|D_p \alpha\|^2$$

where $\mu = g^{-1}(B\alpha)$ and $g$ is the link function.

### Varying Penalties

Allow spatially varying penalties:
$$\lambda \|W D_p \alpha\|^2$$

where $W$ is a diagonal weight matrix.

## References

1. Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.

2. Eilers, P. H. C., & Marx, B. D. (2021). *Practical Smoothing: The Joys of P-splines*. Cambridge University Press.

3. Ruppert, D., Wand, M. P., & Carroll, R. J. (2003). *Semiparametric Regression*. Cambridge University Press.

4. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*. Chapman and Hall/CRC.
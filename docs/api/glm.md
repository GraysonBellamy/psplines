# GLM Families API

The GLM module provides distribution family abstractions for P-spline smoothing beyond Gaussian responses, using Iteratively Reweighted Least Squares (IRLS).

## Family Protocol

::: psplines.glm.Family
    options:
      show_source: true
      heading_level: 3

## GaussianFamily

::: psplines.glm.GaussianFamily
    options:
      show_source: true
      heading_level: 3

## PoissonFamily

::: psplines.glm.PoissonFamily
    options:
      show_source: true
      heading_level: 3

## BinomialFamily

::: psplines.glm.BinomialFamily
    options:
      show_source: true
      heading_level: 3

## get_family

::: psplines.glm.get_family
    options:
      show_source: true
      heading_level: 3

## Overview

The GLM framework extends P-splines to non-Gaussian responses via IRLS. Instead of solving the normal equations directly, the fitting loop iterates:

1. Compute working response $z = \eta + W^{-1}(y - \mu)$
2. Compute working weights $W$ from the current $\mu$
3. Solve $(B'WB + \lambda D'D)\alpha = B'Wz$
4. Update $\eta = B\alpha$, $\mu = h(\eta)$
5. Repeat until convergence

Each family defines the link function $h$, the working weights, and the deviance.

### Supported Families

| Family | Link | $\mu = h(\eta)$ | Working weights $W$ | Scale $\phi$ |
|--------|------|------------------|---------------------|---------------|
| Gaussian | Identity | $\eta$ | $1$ | Estimated |
| Poisson | Log | $\exp(\eta)$ | $\text{diag}(\mu)$ | 1 |
| Binomial | Logit | $t \cdot \text{sigmoid}(\eta)$ | $\text{diag}(\mu(1-\pi))$ | 1 |

### Usage with PSpline

Pass a family string (or instance) to the `PSpline` constructor:

```python
from psplines import PSpline

# Poisson (count data)
spline = PSpline(x, counts, family="poisson")

# Poisson with exposure offset (rate modeling)
spline = PSpline(x, counts, family="poisson", offset=np.log(exposure))

# Binomial — Bernoulli (binary response)
spline = PSpline(x, binary_y, family="binomial")

# Binomial — grouped (y successes out of t trials)
spline = PSpline(x, successes, family="binomial", trials=trials_vec)
```

### Custom Families

Any object satisfying the `Family` protocol can be passed directly:

```python
from psplines.glm import Family

class MyFamily:
    """Custom family implementing the Family protocol."""

    @property
    def is_gaussian(self) -> bool:
        return False

    def initialize(self, y, **kwargs):
        ...

    def working_response(self, y, eta, mu):
        ...

    def working_weights(self, mu, **kwargs):
        ...

    def deviance(self, y, mu):
        ...

    def inverse_link(self, eta):
        ...

    def phi(self, deviance, n, ed):
        ...

spline = PSpline(x, y, family=MyFamily())
```

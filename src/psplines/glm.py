"""
psplines.glm – GLM family/link abstractions for P-spline smoothing
===================================================================

Provides distribution families for use with the IRLS fitting loop:
  - GaussianFamily: identity link (standard P-spline)
  - PoissonFamily: log link (count smoothing, density estimation)
  - BinomialFamily: logit link (binary/grouped response)

Based on Eilers & Marx (2021), §2.12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "Family",
    "GaussianFamily",
    "PoissonFamily",
    "BinomialFamily",
    "get_family",
]

# Small constant to avoid log(0) and division by zero
_EPS = 1e-10


@runtime_checkable
class Family(Protocol):
    """Protocol for GLM family implementations."""

    def initialize(
        self, y: NDArray, **kwargs: object
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Return (eta, mu, W_diag) starting values."""
        ...

    def working_response(self, y: NDArray, eta: NDArray, mu: NDArray) -> NDArray:
        """Compute working dependent variable z = eta + W^{-1}(y - mu)."""
        ...

    def working_weights(self, mu: NDArray, **kwargs: object) -> NDArray:
        """Compute diagonal of IRLS weight matrix W."""
        ...

    def deviance(self, y: NDArray, mu: NDArray) -> float:
        """Compute model deviance."""
        ...

    def inverse_link(self, eta: NDArray) -> NDArray:
        """Map linear predictor to mean: mu = h(eta)."""
        ...

    def phi(self, deviance: float, n: int, ed: float) -> float:
        """Compute scale parameter phi."""
        ...

    @property
    def is_gaussian(self) -> bool:
        """Whether this is the Gaussian (identity link) family."""
        ...


class GaussianFamily:
    """
    Gaussian family with identity link.

    This wraps the existing Gaussian P-spline behaviour so the IRLS loop
    converges in a single iteration.
    """

    @property
    def is_gaussian(self) -> bool:
        return True

    def initialize(
        self, y: NDArray, **kwargs: object
    ) -> tuple[NDArray, NDArray, NDArray]:
        eta = y.copy()
        mu = y.copy()
        w = np.ones_like(y)
        return eta, mu, w

    def working_response(self, y: NDArray, eta: NDArray, mu: NDArray) -> NDArray:
        return y

    def working_weights(self, mu: NDArray, **kwargs: object) -> NDArray:
        return np.ones_like(mu)

    def deviance(self, y: NDArray, mu: NDArray) -> float:
        return float(np.sum((y - mu) ** 2))

    def inverse_link(self, eta: NDArray) -> NDArray:
        return eta.copy()  # type: ignore[no-any-return]

    def phi(self, deviance: float, n: int, ed: float) -> float:
        dof = n - ed
        if dof <= 0:
            return 1.0
        return deviance / dof


class PoissonFamily:
    """
    Poisson family with log link (§2.12.1, eq. 2.18–2.22).

    Parameters
    ----------
    offset : NDArray or None
        Log-scale exposure offset. When provided, mu = exp(eta) * exp(offset),
        i.e. eta_total = B @ alpha + offset.
    """

    def __init__(self, offset: NDArray | None = None) -> None:
        self.offset = offset

    @property
    def is_gaussian(self) -> bool:
        return False

    def initialize(
        self, y: NDArray, **kwargs: object
    ) -> tuple[NDArray, NDArray, NDArray]:
        # eta_0 = log(y + 1)  (eq. 2.22 setup, p.720)
        eta = np.log(y + 1.0)
        if self.offset is not None:
            eta -= self.offset  # eta stores B*alpha part only
        mu = self.inverse_link(eta)
        w = np.maximum(mu, _EPS)
        return eta, mu, w

    def working_response(self, y: NDArray, eta: NDArray, mu: NDArray) -> NDArray:
        # z = eta_full + W^{-1}(y - mu)  (eq. 2.22)
        eta_full = eta if self.offset is None else eta + self.offset
        w = np.maximum(mu, _EPS)
        z = eta_full + (y - mu) / w
        if self.offset is not None:
            z -= self.offset  # working response for B*alpha part
        return z  # type: ignore[no-any-return]

    def working_weights(self, mu: NDArray, **kwargs: object) -> NDArray:
        # W = diag(mu)
        return np.maximum(mu, _EPS)  # type: ignore[no-any-return]

    def deviance(self, y: NDArray, mu: NDArray) -> float:
        # dev = 2 * sum(y*log(y/mu) - (y - mu))  (eq. 2.19)
        mu_safe = np.maximum(mu, _EPS)
        # Handle y=0 case: 0*log(0/mu) = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(y > 0, y * np.log(y / mu_safe), 0.0)
        return float(2.0 * np.sum(ratio - (y - mu_safe)))

    def inverse_link(self, eta: NDArray) -> NDArray:
        # mu = exp(eta_full)
        eta_full = eta if self.offset is None else eta + self.offset
        return np.exp(eta_full)  # type: ignore[no-any-return]

    def phi(self, deviance: float, n: int, ed: float) -> float:
        # phi = 1 for Poisson (or estimated for overdispersion)
        return 1.0


class BinomialFamily:
    """
    Binomial family with logit link (§2.12.2, eq. 2.23–2.25).

    Parameters
    ----------
    trials : NDArray or None
        Number of trials per observation. Defaults to 1 (Bernoulli).
    """

    def __init__(self, trials: NDArray | None = None) -> None:
        self.trials = trials

    @property
    def is_gaussian(self) -> bool:
        return False

    def _get_trials(self, y: NDArray) -> NDArray:
        if self.trials is not None:
            return self.trials
        return np.ones_like(y)

    def initialize(
        self, y: NDArray, **kwargs: object
    ) -> tuple[NDArray, NDArray, NDArray]:
        # pi_0 = (y + 1) / (t + 2), eta_0 = logit(pi_0)  (p.762)
        t = self._get_trials(y)
        pi0 = (y + 1.0) / (t + 2.0)
        pi0 = np.clip(pi0, _EPS, 1.0 - _EPS)
        eta = np.log(pi0 / (1.0 - pi0))
        mu = t * pi0
        pi_val = pi0
        w = mu * (1.0 - pi_val)
        return eta, mu, np.maximum(w, _EPS)

    def working_response(self, y: NDArray, eta: NDArray, mu: NDArray) -> NDArray:
        # z = eta + W^{-1}(y - mu)
        t = self._get_trials(y)
        pi_val = np.clip(mu / np.maximum(t, _EPS), _EPS, 1.0 - _EPS)
        w = np.maximum(mu * (1.0 - pi_val), _EPS)
        return eta + (y - mu) / w  # type: ignore[no-any-return]

    def working_weights(self, mu: NDArray, **kwargs: object) -> NDArray:
        # W = diag(mu * (1 - pi))  where pi = mu/t
        y_placeholder = kwargs.get("y")
        t = self._get_trials(
            mu if y_placeholder is None else y_placeholder  # type: ignore[arg-type]
        )
        pi_val = np.clip(mu / np.maximum(t, _EPS), _EPS, 1.0 - _EPS)
        return np.maximum(mu * (1.0 - pi_val), _EPS)  # type: ignore[no-any-return]

    def deviance(self, y: NDArray, mu: NDArray) -> float:
        # dev = 2 * sum(y*log(y/mu) + (t-y)*log((t-y)/(t-mu)))  (eq. 2.24)
        t = self._get_trials(y)
        mu_safe = np.clip(mu, _EPS, t - _EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = np.where(y > 0, y * np.log(y / mu_safe), 0.0)
            term2 = np.where(t - y > 0, (t - y) * np.log((t - y) / (t - mu_safe)), 0.0)
        return float(2.0 * np.sum(term1 + term2))

    def inverse_link(self, eta: NDArray) -> NDArray:
        # mu = t * sigmoid(eta)
        pi_val = 1.0 / (1.0 + np.exp(-eta))
        t = self.trials if self.trials is not None else np.ones_like(eta)
        return t * pi_val  # type: ignore[no-any-return]

    def phi(self, deviance: float, n: int, ed: float) -> float:
        # phi = 1 for binomial
        return 1.0


def get_family(
    family: str | Family,
    trials: NDArray | None = None,
    offset: NDArray | None = None,
) -> Family:
    """
    Resolve a family string or instance into a Family object.

    Parameters
    ----------
    family : str or Family
        One of "gaussian", "poisson", "binomial", or a Family instance.
    trials : NDArray, optional
        Binomial trials vector.
    offset : NDArray, optional
        Poisson offset (log-exposure).

    Returns
    -------
    Family
        Resolved family instance.
    """
    if isinstance(family, str):
        name = family.lower()
        if name == "gaussian":
            return GaussianFamily()
        if name == "poisson":
            return PoissonFamily(offset=offset)
        if name == "binomial":
            return BinomialFamily(trials=trials)
        raise ValueError(
            f"Unknown family '{family}'. Choose from: 'gaussian', 'poisson', 'binomial'"
        )
    return family

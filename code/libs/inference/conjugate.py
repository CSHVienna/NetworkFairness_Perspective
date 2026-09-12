"""Conjugate prior families for closed-form Bayesian parameter updating.

Each concrete class pairs a prior distribution with a likelihood family so that
the posterior is in the same family as the prior.  Calling ``update`` returns a
**new** immutable instance with updated hyper-parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from libs.distributions.base import Distribution
from libs.distributions.continuous import BetaDist, GammaDist, NormalDist


# ======================================================================
# Abstract base
# ======================================================================

class ConjugatePrior(ABC):
    """Interface for conjugate-prior update rules."""

    @abstractmethod
    def prior_distribution(self) -> Distribution:
        """Return the current prior as a Distribution object."""

    @abstractmethod
    def update(self, data: Any) -> "ConjugatePrior":
        """Return a *new* instance with hyper-parameters updated by *data*."""

    @abstractmethod
    def posterior_distribution(self) -> Distribution:
        """Alias kept for clarity; same as ``prior_distribution`` after update."""

    @property
    @abstractmethod
    def hyperparams(self) -> dict[str, Any]:
        """Current hyper-parameter values."""

    def __repr__(self) -> str:
        hp = ", ".join(f"{k}={v}" for k, v in self.hyperparams.items())
        return f"{self.__class__.__name__}({hp})"


# ======================================================================
# Beta–Binomial  (prior on success probability p)
# ======================================================================

class BetaBinomialConjugate(ConjugatePrior):
    """Beta(alpha, beta) prior for a Binomial/Bernoulli likelihood.

    Update rule::

        alpha' = alpha + successes
        beta'  = beta  + (trials - successes)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def prior_distribution(self) -> BetaDist:
        return BetaDist(self.alpha, self.beta)

    def update(self, successes: int, trials: int) -> "BetaBinomialConjugate":
        return BetaBinomialConjugate(
            alpha=self.alpha + successes,
            beta=self.beta + (trials - successes),
        )

    def posterior_distribution(self) -> BetaDist:
        return self.prior_distribution()

    @property
    def hyperparams(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}


# ======================================================================
# Normal–Normal  (prior on mean mu, known likelihood variance)
# ======================================================================

class NormalNormalConjugate(ConjugatePrior):
    """Normal(mu0, sigma0) prior for a Normal likelihood with known sigma.

    Update rule (single observation or batch)::

        precision_0     = 1 / sigma0^2
        precision_lik   = n / sigma_likelihood^2
        precision_post  = precision_0 + precision_lik
        mu_post         = (precision_0 * mu0 + precision_lik * x_bar) / precision_post
        sigma_post      = sqrt(1 / precision_post)
    """

    def __init__(
        self,
        mu0: float,
        sigma0: float,
        sigma_likelihood: float,
    ) -> None:
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma_likelihood = sigma_likelihood

    def prior_distribution(self) -> NormalDist:
        return NormalDist(self.mu0, self.sigma0)

    def update(self, data: np.ndarray | list[float]) -> "NormalNormalConjugate":
        data = np.asarray(data, dtype=float).ravel()
        n = len(data)
        x_bar = data.mean()

        prec_0 = 1.0 / (self.sigma0 ** 2)
        prec_lik = n / (self.sigma_likelihood ** 2)
        prec_post = prec_0 + prec_lik

        mu_post = (prec_0 * self.mu0 + prec_lik * x_bar) / prec_post
        sigma_post = np.sqrt(1.0 / prec_post)

        return NormalNormalConjugate(
            mu0=float(mu_post),
            sigma0=float(sigma_post),
            sigma_likelihood=self.sigma_likelihood,
        )

    def posterior_distribution(self) -> NormalDist:
        return self.prior_distribution()

    @property
    def hyperparams(self) -> dict[str, Any]:
        return {
            "mu0": self.mu0,
            "sigma0": self.sigma0,
            "sigma_likelihood": self.sigma_likelihood,
        }


# ======================================================================
# Gamma–Poisson  (prior on Poisson rate lambda)
# ======================================================================

class GammaPoissonConjugate(ConjugatePrior):
    """Gamma(alpha, beta) prior for a Poisson likelihood.

    Update rule::

        alpha' = alpha + sum(x_i)
        beta'  = beta  + n
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def prior_distribution(self) -> GammaDist:
        return GammaDist(self.alpha, self.beta)

    def update(self, data: np.ndarray | list[int]) -> "GammaPoissonConjugate":
        data = np.asarray(data, dtype=float).ravel()
        return GammaPoissonConjugate(
            alpha=self.alpha + float(data.sum()),
            beta=self.beta + len(data),
        )

    def posterior_distribution(self) -> GammaDist:
        return self.prior_distribution()

    @property
    def hyperparams(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}


# ======================================================================
# Dirichlet–Multinomial  (prior on category probabilities)
# ======================================================================

class DirichletMultinomialConjugate(ConjugatePrior):
    """Dirichlet(alphas) prior for a Multinomial likelihood.

    Update rule::

        alphas' = alphas + observed_counts

    ``prior_distribution`` returns a *BetaDist* when k=2 (for convenience),
    otherwise raises NotImplementedError because ``scipy.stats`` does not
    ship a full Dirichlet frozen distribution with pdf/sample on the simplex.
    Sampling is provided via ``sample_posterior``.
    """

    def __init__(self, alphas: list[float] | np.ndarray) -> None:
        self.alphas = np.asarray(alphas, dtype=float)
        if (self.alphas <= 0).any():
            raise ValueError("All concentration parameters must be positive")

    def prior_distribution(self) -> Distribution:
        if len(self.alphas) == 2:
            return BetaDist(self.alphas[0], self.alphas[1])
        raise NotImplementedError(
            "Full Dirichlet Distribution wrapper not implemented; "
            "use sample_posterior() for k > 2."
        )

    def update(
        self, counts: np.ndarray | list[int]
    ) -> "DirichletMultinomialConjugate":
        counts = np.asarray(counts, dtype=float)
        if counts.shape != self.alphas.shape:
            raise ValueError("counts shape must match alphas shape")
        return DirichletMultinomialConjugate(self.alphas + counts)

    def posterior_distribution(self) -> Distribution:
        return self.prior_distribution()

    def sample_posterior(
        self, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw *n* samples from Dirichlet(alphas) — shape (n, k)."""
        rng = rng or np.random.default_rng()
        return rng.dirichlet(self.alphas, size=n)

    @property
    def hyperparams(self) -> dict[str, Any]:
        return {"alphas": self.alphas.tolist()}


# ======================================================================
# Normal–LogNormal  (prior on log-scale mean of a LogNormal likelihood)
# ======================================================================

class NormalLogNormalConjugate(ConjugatePrior):
    """Normal(mu0, sigma0) prior on the log-mean of a LogNormal likelihood.

    Since log(X) ~ Normal(mu, sigma_likelihood) when X ~ LogNormal, the
    update is identical to Normal–Normal but applied to log-transformed data.
    """

    def __init__(
        self,
        mu0: float,
        sigma0: float,
        sigma_likelihood: float,
    ) -> None:
        self._inner = NormalNormalConjugate(mu0, sigma0, sigma_likelihood)

    def prior_distribution(self) -> NormalDist:
        return self._inner.prior_distribution()

    def update(
        self, data: np.ndarray | list[float]
    ) -> "NormalLogNormalConjugate":
        log_data = np.log(np.asarray(data, dtype=float))
        updated_inner = self._inner.update(log_data)
        new = NormalLogNormalConjugate.__new__(NormalLogNormalConjugate)
        new._inner = updated_inner
        return new

    def posterior_distribution(self) -> NormalDist:
        return self._inner.prior_distribution()

    @property
    def hyperparams(self) -> dict[str, Any]:
        return self._inner.hyperparams

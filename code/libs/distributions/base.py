"""Abstract base classes for probability distributions."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Distribution(ABC):
    """
    Abstract base for all probability distributions.

    Every concrete distribution wraps a scipy.stats frozen distribution
    and exposes a unified API for evaluation, sampling, and introspection.
    """

    @abstractmethod
    def pdf_or_pmf(self, x) -> np.ndarray:
        """Evaluate the density (continuous) or mass (discrete) at *x*."""

    def log_prob(self, x) -> np.ndarray:
        """Log of pdf_or_pmf.  Override for numerical stability."""
        return np.log(self.pdf_or_pmf(x))

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw *n* independent samples."""

    @property
    @abstractmethod
    def mean(self) -> float:
        ...

    @property
    @abstractmethod
    def variance(self) -> float:
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Parameter dict that can recreate this distribution."""

    @property
    @abstractmethod
    def support(self) -> tuple[float, float]:
        """(lower, upper) bounds of the support."""

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"


class ContinuousDistribution(Distribution):
    """Base for distributions defined on a continuous domain."""

    @abstractmethod
    def pdf(self, x) -> np.ndarray:
        ...

    @abstractmethod
    def cdf(self, x) -> np.ndarray:
        ...

    def pdf_or_pmf(self, x) -> np.ndarray:
        return self.pdf(x)

    def log_prob(self, x) -> np.ndarray:
        return self._frozen.logpdf(np.asarray(x, dtype=float))


class DiscreteDistribution(Distribution):
    """Base for distributions defined on a discrete (integer) domain."""

    @abstractmethod
    def pmf(self, x) -> np.ndarray:
        ...

    def pdf_or_pmf(self, x) -> np.ndarray:
        return self.pmf(x)

    def log_prob(self, x) -> np.ndarray:
        return self._frozen.logpmf(np.asarray(x, dtype=float))

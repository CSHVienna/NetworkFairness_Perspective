"""Concrete continuous probability distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import powerlaw as pl_pkg
from scipy import stats

from libs.distributions.base import ContinuousDistribution
from libs.distributions.registry import register


@register("beta")
class BetaDist(ContinuousDistribution):
    """Beta distribution on [0, 1]."""

    def __init__(self, alpha: float, beta: float) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta
        self._frozen = stats.beta(alpha, beta)

    def pdf(self, x) -> np.ndarray:
        return self._frozen.pdf(np.asarray(x, dtype=float))

    def cdf(self, x) -> np.ndarray:
        return self._frozen.cdf(np.asarray(x, dtype=float))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.beta(self.alpha, self.beta, size=n)

    @property
    def mean(self) -> float:
        return float(self._frozen.mean())

    @property
    def variance(self) -> float:
        return float(self._frozen.var())

    @property
    def params(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)


@register("normal")
class NormalDist(ContinuousDistribution):
    """Gaussian distribution on (-inf, +inf)."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self._frozen = stats.norm(loc=mu, scale=sigma)

    def pdf(self, x) -> np.ndarray:
        return self._frozen.pdf(np.asarray(x, dtype=float))

    def cdf(self, x) -> np.ndarray:
        return self._frozen.cdf(np.asarray(x, dtype=float))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=n)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma ** 2

    @property
    def params(self) -> dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma}

    @property
    def support(self) -> tuple[float, float]:
        return (float("-inf"), float("inf"))


@register("lognormal")
class LogNormalDist(ContinuousDistribution):
    """Log-normal distribution on (0, +inf).

    Parametrised by *mu* and *sigma* of the underlying normal variable
    (i.e.  if X ~ Normal(mu, sigma) then exp(X) ~ LogNormal(mu, sigma)).
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        # scipy parametrises lognorm as shape=sigma, scale=exp(mu)
        self._frozen = stats.lognorm(s=sigma, scale=np.exp(mu))

    def pdf(self, x) -> np.ndarray:
        return self._frozen.pdf(np.asarray(x, dtype=float))

    def cdf(self, x) -> np.ndarray:
        return self._frozen.cdf(np.asarray(x, dtype=float))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.lognormal(self.mu, self.sigma, size=n)

    @property
    def mean(self) -> float:
        return float(self._frozen.mean())

    @property
    def variance(self) -> float:
        return float(self._frozen.var())

    @property
    def params(self) -> dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma}

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, float("inf"))


# --- Original PowerLawDist (scipy-based, commented out) ---
# @register("pareto")
# @register("power_law")
# class PowerLawDist(ContinuousDistribution):
#     """Power-law (Pareto Type I), optionally truncated, on ``[xmin, xmax]``.
#
#     PDF ``f(x) ∝ x^{-(alpha+1)}`` for ``xmin <= x <= xmax``, renormalised
#     when *xmax* is finite.  Values outside the support return a small
#     ``EPSILON`` density so that likelihood ratios remain finite — this is
#     equivalent to mixing the truncated distribution with a tiny uniform
#     background and prevents degenerate posteriors when good/bad supports
#     don't fully overlap.
#
#     Parameters
#     ----------
#     alpha
#         Shape (power-law exponent); larger → thinner tail.
#     xmin
#         Lower bound of the support (strictly positive).
#     xmax
#         Upper truncation point.  ``None`` or ``inf`` → standard
#         (untruncated) power law on ``[xmin, +inf)``.
#     """
#
#     EPSILON: float = 1e-10
#
#     def __init__(
#         self,
#         alpha: float,
#         xmin: float = 1.0,
#         xmax: float | None = None,
#     ) -> None:
#         if alpha <= 0 or xmin <= 0:
#             raise ValueError("alpha and xmin must be positive")
#         self.alpha = float(alpha)
#         self.xmin = float(xmin)
#         self.xmax = float(xmax) if xmax is not None else float("inf")
#         if np.isfinite(self.xmax) and self.xmax <= self.xmin:
#             raise ValueError(
#                 f"xmax ({self.xmax}) must be greater than xmin ({self.xmin})"
#             )
#         self._frozen = stats.pareto(self.alpha, loc=0.0, scale=self.xmin)
#         self._truncated = np.isfinite(self.xmax)
#         self._norm = (
#             1.0 - (self.xmin / self.xmax) ** self.alpha
#             if self._truncated
#             else 1.0
#         )
#         self._log_norm = np.log(self._norm)
#
#     def _in_support(self, x: np.ndarray) -> np.ndarray:
#         return (x >= self.xmin) & (x <= self.xmax)
#
#     def pdf(self, x) -> np.ndarray:
#         x = np.asarray(x, dtype=float)
#         raw = self._frozen.pdf(x) / self._norm
#         return np.where(self._in_support(x), raw, self.EPSILON)
#
#     def cdf(self, x) -> np.ndarray:
#         x = np.asarray(x, dtype=float)
#         if not self._truncated:
#             return self._frozen.cdf(x)
#         clipped = np.clip(x, self.xmin, self.xmax)
#         return self._frozen.cdf(clipped) / self._norm
#
#     def log_prob(self, x) -> np.ndarray:
#         """Log-density with epsilon floor outside the support."""
#         x = np.asarray(x, dtype=float)
#         raw = self._frozen.logpdf(x) - self._log_norm
#         return np.where(self._in_support(x), raw, np.log(self.EPSILON))
#
#     def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
#         rng = rng or np.random.default_rng()
#         if self._truncated:
#             u = rng.uniform(0.0, 1.0, size=n)
#             return self.xmin * (1.0 - u * self._norm) ** (-1.0 / self.alpha)
#         return self._frozen.rvs(size=n, random_state=rng)
#
#     @property
#     def mean(self) -> float:
#         a, s = self.alpha, self.xmin
#         if not self._truncated:
#             return float("inf") if a <= 1 else float(a * s / (a - 1))
#         m = self.xmax
#         if abs(a - 1.0) < 1e-12:
#             return float(s * np.log(m / s) / self._norm)
#         return float(
#             a * s**a / (self._norm * (a - 1)) * (s ** (1 - a) - m ** (1 - a))
#         )
#
#     @property
#     def variance(self) -> float:
#         a, s = self.alpha, self.xmin
#         if not self._truncated:
#             if a <= 2:
#                 return float("inf")
#             return float(a * s**2 / ((a - 1) ** 2 * (a - 2)))
#         m = self.xmax
#         if abs(a - 2.0) < 1e-12:
#             ex2 = a * s**a / self._norm * np.log(m / s)
#         else:
#             ex2 = (
#                 a * s**a / (self._norm * (2.0 - a)) * (m ** (2 - a) - s ** (2 - a))
#             )
#         return float(ex2 - self.mean**2)
#
#     @property
#     def params(self) -> dict[str, Any]:
#         d: dict[str, Any] = {"alpha": self.alpha, "xmin": self.xmin}
#         if self._truncated:
#             d["xmax"] = self.xmax
#         return d
#
#     @property
#     def support(self) -> tuple[float, float]:
#         return (self.xmin, self.xmax)
# --- End original PowerLawDist ---


@register("pareto")
@register("power_law")
class PowerLawDist(ContinuousDistribution):
    """Power-law distribution backed by the ``powerlaw`` package.

    PDF  ``f(x) ∝ x^{-alpha}``  for  ``xmin <= x <= xmax``.

    Sampling is delegated to ``powerlaw.Power_Law.generate_random``.
    Analytical formulas are used for ``pdf``, ``cdf``, ``log_prob``,
    ``mean``, and ``variance``.

    Parameters
    ----------
    alpha
        Power-law exponent (passed as ``parameters=[alpha]`` to
        ``powerlaw.Power_Law``).
    xmin
        Lower bound of the support (strictly positive).
    xmax
        Upper truncation point.  ``None`` or ``inf`` → untruncated.
    """

    EPSILON: float = 1e-10

    def __init__(
        self,
        alpha: float,
        xmin: float = 1.0,
        xmax: float | None = None,
    ) -> None:
        if alpha <= 0 or xmin <= 0:
            raise ValueError("alpha and xmin must be positive")
        self.alpha = float(alpha)
        self.xmin = float(xmin)
        self.xmax = float(xmax) if xmax is not None else float("inf")
        if np.isfinite(self.xmax) and self.xmax <= self.xmin:
            raise ValueError(
                f"xmax ({self.xmax}) must be greater than xmin ({self.xmin})"
            )
        self._truncated = np.isfinite(self.xmax)

        kw: dict[str, Any] = {"xmin": self.xmin, "parameters": [self.alpha]}
        if self._truncated:
            kw["xmax"] = self.xmax
        self._pl = pl_pkg.Power_Law(**kw)

        # Z = ∫_{xmin}^{xmax} x^{-alpha} dx  (normalisation constant)
        a = self.alpha
        if self._truncated:
            if abs(a - 1.0) < 1e-12:
                self._Z = np.log(self.xmax / self.xmin)
            else:
                self._Z = (
                    (self.xmin ** (1 - a) - self.xmax ** (1 - a)) / (a - 1)
                )
        else:
            if a <= 1:
                raise ValueError("alpha must be > 1 for untruncated power law")
            self._Z = self.xmin ** (1 - a) / (a - 1)
        self._log_Z = np.log(self._Z)

    def _in_support(self, x: np.ndarray) -> np.ndarray:
        return (x >= self.xmin) & (x <= self.xmax)

    def pdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        raw = x ** (-self.alpha) / self._Z
        return np.where(self._in_support(x), raw, self.EPSILON)

    def cdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        a = self.alpha
        ub = self.xmax if self._truncated else np.inf
        clipped = np.clip(x, self.xmin, ub)
        if abs(a - 1.0) < 1e-12:
            vals = np.log(clipped / self.xmin) / self._Z
        else:
            vals = (
                (clipped ** (1 - a) - self.xmin ** (1 - a)) / ((1 - a) * self._Z)
            )
        return np.where(x < self.xmin, 0.0, np.where(x > ub, 1.0, vals))

    def log_prob(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        raw = -self.alpha * np.log(x) - self._log_Z
        return np.where(self._in_support(x), raw, np.log(self.EPSILON))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        return self._pl.generate_random(n)

    @property
    def mean(self) -> float:
        a = self.alpha
        if not self._truncated:
            if a <= 2:
                return float("inf")
            return float((a - 1) / (a - 2) * self.xmin)
        if abs(a - 2.0) < 1e-12:
            return float(np.log(self.xmax / self.xmin) / self._Z)
        return float(
            (self.xmax ** (2 - a) - self.xmin ** (2 - a)) / ((2 - a) * self._Z)
        )

    @property
    def variance(self) -> float:
        a = self.alpha
        if not self._truncated:
            if a <= 3:
                return float("inf")
            return float(
                (a - 1) * self.xmin ** 2 / ((a - 2) ** 2 * (a - 3))
            )
        if abs(a - 3.0) < 1e-12:
            ex2 = np.log(self.xmax / self.xmin) / self._Z
        else:
            ex2 = (
                (self.xmax ** (3 - a) - self.xmin ** (3 - a))
                / ((3 - a) * self._Z)
            )
        return float(ex2 - self.mean ** 2)

    @property
    def params(self) -> dict[str, Any]:
        d: dict[str, Any] = {"alpha": self.alpha, "xmin": self.xmin}
        if self._truncated:
            d["xmax"] = self.xmax
        return d

    @property
    def support(self) -> tuple[float, float]:
        return (self.xmin, self.xmax)


@register("gamma")
class GammaDist(ContinuousDistribution):
    """Gamma distribution on (0, +inf) with shape *alpha* and rate *beta*.

    The rate parametrisation (beta = 1/scale) is used so that the
    Gamma–Poisson conjugate pair has the natural update rule.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta
        self._frozen = stats.gamma(a=alpha, scale=1.0 / beta)

    def pdf(self, x) -> np.ndarray:
        return self._frozen.pdf(np.asarray(x, dtype=float))

    def cdf(self, x) -> np.ndarray:
        return self._frozen.cdf(np.asarray(x, dtype=float))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.gamma(self.alpha, 1.0 / self.beta, size=n)

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    @property
    def variance(self) -> float:
        return self.alpha / (self.beta ** 2)

    @property
    def params(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, float("inf"))

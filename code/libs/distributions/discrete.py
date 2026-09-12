"""Concrete discrete probability distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from libs.distributions.base import DiscreteDistribution
from libs.distributions.registry import register


@register("poisson")
class PoissonDist(DiscreteDistribution):
    """Poisson distribution on {0, 1, 2, ...}."""

    def __init__(self, lam: float) -> None:
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.lam = lam
        self._frozen = stats.poisson(mu=lam)

    def pmf(self, x) -> np.ndarray:
        return self._frozen.pmf(np.asarray(x, dtype=float))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.poisson(self.lam, size=n)

    @property
    def mean(self) -> float:
        return self.lam

    @property
    def variance(self) -> float:
        return self.lam

    @property
    def params(self) -> dict[str, Any]:
        return {"lam": self.lam}

    @property
    def support(self) -> tuple[float, float]:
        return (0, float("inf"))


@register("binomial")
class BinomialDist(DiscreteDistribution):
    """Binomial distribution on {0, 1, ..., n}."""

    def __init__(self, n: int, p: float) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.n = int(n)
        self.p = p
        self._frozen = stats.binom(n=self.n, p=p)

    def pmf(self, x) -> np.ndarray:
        return self._frozen.pmf(np.asarray(x, dtype=float))

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.binomial(self.n, self.p, size=n_samples)

    @property
    def mean(self) -> float:
        return self.n * self.p

    @property
    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)

    @property
    def params(self) -> dict[str, Any]:
        return {"n": self.n, "p": self.p}

    @property
    def support(self) -> tuple[float, float]:
        return (0, self.n)


@register("hypergeometric")
class HypergeometricDist(DiscreteDistribution):
    """Hypergeometric distribution.

    Parameters
    ----------
    N : total population size
    K : number of success states in the population
    n : number of draws
    """

    def __init__(self, N: int, K: int, n: int) -> None:
        if not (0 <= K <= N):
            raise ValueError("Need 0 <= K <= N")
        if not (0 <= n <= N):
            raise ValueError("Need 0 <= n <= N")
        self.N = int(N)
        self.K = int(K)
        self.n = int(n)
        self._frozen = stats.hypergeom(M=self.N, n=self.K, N=self.n)

    def pmf(self, x) -> np.ndarray:
        return self._frozen.pmf(np.asarray(x, dtype=float))

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.hypergeometric(self.K, self.N - self.K, self.n, size=n_samples)

    @property
    def mean(self) -> float:
        return float(self._frozen.mean())

    @property
    def variance(self) -> float:
        return float(self._frozen.var())

    @property
    def params(self) -> dict[str, Any]:
        return {"N": self.N, "K": self.K, "n": self.n}

    @property
    def support(self) -> tuple[float, float]:
        lo = max(0, self.n - (self.N - self.K))
        hi = min(self.n, self.K)
        return (lo, hi)


@register("multinomial")
class MultinomialDist(DiscreteDistribution):
    """Multinomial distribution over *k* categories.

    ``pmf`` and ``log_prob`` accept arrays of shape (..., k).
    ``sample`` returns shape (n_samples, k).
    """

    def __init__(self, n: int, probs: list[float] | np.ndarray) -> None:
        self.n = int(n)
        self.probs = np.asarray(probs, dtype=float)
        if not np.isclose(self.probs.sum(), 1.0):
            raise ValueError("probs must sum to 1")
        self._frozen = stats.multinomial(n=self.n, p=self.probs)

    def pmf(self, x) -> np.ndarray:
        return self._frozen.pmf(np.asarray(x, dtype=float))

    def log_prob(self, x) -> np.ndarray:
        return self._frozen.logpmf(np.asarray(x, dtype=float))

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.multinomial(self.n, self.probs, size=n_samples)

    @property
    def mean(self) -> float:
        return float(self.n * self.probs[0])

    @property
    def mean_vector(self) -> np.ndarray:
        return self.n * self.probs

    @property
    def variance(self) -> float:
        return float(self.n * self.probs[0] * (1 - self.probs[0]))

    @property
    def covariance_matrix(self) -> np.ndarray:
        return self._frozen.cov()

    @property
    def params(self) -> dict[str, Any]:
        return {"n": self.n, "probs": self.probs.tolist()}

    @property
    def support(self) -> tuple[float, float]:
        return (0, self.n)


_REF_KEY_DECIMALS = 10


def _ref_mass_key(x: float) -> float:
    """Round so convolution keys stay stable (float trust × score, sums)."""
    return round(float(x), _REF_KEY_DECIMALS)


def _pmf_single_reference(
    scores: np.ndarray,
    trust_weights: np.ndarray,
    score_probs: np.ndarray,
    trust_probs: np.ndarray,
) -> dict[float, float]:
    """P(W=w) for W = trust_weight * score, independent categorical draws."""
    s = np.asarray(scores, dtype=float).ravel()
    tw = np.asarray(trust_weights, dtype=float).ravel()
    sp = np.asarray(score_probs, dtype=float).ravel()
    tp = np.asarray(trust_probs, dtype=float).ravel()
    if len(sp) != len(s) or len(tp) != len(tw):
        raise ValueError(
            "score_probs must match scores length; trust_probs must match trust_weights length"
        )
    if not np.isclose(sp.sum(), 1.0) or not np.isclose(tp.sum(), 1.0):
        raise ValueError("score_probs and trust_probs must each sum to 1")
    out: dict[float, float] = {}
    for i, pt in enumerate(tp):
        for j, ps in enumerate(sp):
            w = _ref_mass_key(float(tw[i]) * float(s[j]))
            out[w] = out.get(w, 0.0) + float(pt * ps)
    return out


def _convolve_pmf_mass(d1: dict[float, float], d2: dict[float, float]) -> dict[float, float]:
    out: dict[float, float] = {}
    for k1, p1 in d1.items():
        for k2, p2 in d2.items():
            k = _ref_mass_key(float(k1) + float(k2))
            out[k] = out.get(k, 0.0) + p1 * p2
    return out


@register("reference_weighted_sum")
class ReferenceWeightedSumDist(DiscreteDistribution):
    """Network-style signal: *n_refs* i.i.d. references, each ``W = t × s``.

    - ``scores``: support of the reference rating (e.g. ``[1,2,3,4,5]`` or ``1..10``).
    - ``trust_weights``: multipliers for each trust category (e.g. ``[-1,0,1]`` or
      ``[0, 0.5, 1]``, or ``[-2,-1,0,1,2]``).
    - ``score_probs`` / ``trust_probs``: same length as ``scores`` / ``trust_weights``.

    If ``scores`` and ``trust_weights`` are omitted and ``len(score_probs)==5``,
    ``len(trust_probs)==3``, defaults are ``scores=[1,2,3,4,5]``,
    ``trust_weights=[-1,0,1]`` (backward compatible).

    Observed signal = sum of *n_refs* i.i.d. draws of ``W``.
    """

    def __init__(
        self,
        score_probs: list[float] | np.ndarray,
        trust_probs: list[float] | np.ndarray,
        n_refs: int = 3,
        scores: list[float] | np.ndarray | None = None,
        trust_weights: list[float] | np.ndarray | None = None,
    ) -> None:
        self.score_probs = np.asarray(score_probs, dtype=float).ravel()
        self.trust_probs = np.asarray(trust_probs, dtype=float).ravel()
        self.n_refs = int(n_refs)
        if self.n_refs < 1:
            raise ValueError("n_refs must be >= 1")

        if scores is None and trust_weights is None:
            if self.score_probs.size == 5 and self.trust_probs.size == 3:
                self.scores = np.arange(1, 6, dtype=float)
                self.trust_weights = np.array([-1.0, 0.0, 1.0], dtype=float)
            else:
                raise ValueError(
                    "Specify scores= and trust_weights= unless using legacy 5×3 "
                    "(implicit scores 1..5 and trust -1,0,+1)"
                )
        else:
            if scores is None or trust_weights is None:
                raise ValueError("scores and trust_weights must both be set (or both omitted for legacy 5×3)")
            self.scores = np.asarray(scores, dtype=float).ravel()
            self.trust_weights = np.asarray(trust_weights, dtype=float).ravel()

        if self.score_probs.size != self.scores.size:
            raise ValueError(
                f"score_probs length {self.score_probs.size} != len(scores) {self.scores.size}"
            )
        if self.trust_probs.size != self.trust_weights.size:
            raise ValueError(
                f"trust_probs length {self.trust_probs.size} != len(trust_weights) {self.trust_weights.size}"
            )

        self._single_pmf = _pmf_single_reference(
            self.scores, self.trust_weights, self.score_probs, self.trust_probs
        )
        joint: dict[float, float] = dict(self._single_pmf)
        for _ in range(self.n_refs - 1):
            joint = _convolve_pmf_mass(joint, self._single_pmf)

        self._aggregate_pmf = joint
        sk = np.array(sorted(joint.keys()), dtype=float)
        pv = np.array([joint[k] for k in sk], dtype=float)
        self._support_lo = float(np.min(sk))
        self._support_hi = float(np.max(sk))
        self._mean = float(np.sum(sk * pv))
        self._var = float(np.sum((sk - self._mean) ** 2 * pv))

        # Sampling: categorical over single-ref masses
        self._single_keys = np.array(sorted(self._single_pmf.keys()), dtype=float)
        self._single_probs = np.array(
            [self._single_pmf[k] for k in self._single_keys], dtype=float
        )
        self._single_probs = self._single_probs / self._single_probs.sum()

    def pmf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        out = np.zeros_like(flat, dtype=float)
        for i, v in enumerate(flat):
            out[i] = self._aggregate_pmf.get(_ref_mass_key(v), 0.0)
        return out.reshape(x.shape)

    def log_prob(self, x) -> np.ndarray:
        p = np.asarray(self.pmf(x), dtype=float)
        return np.log(np.maximum(p, 1e-300))

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        draws = rng.choice(self._single_keys, size=(n_samples, self.n_refs), p=self._single_probs)
        total = np.zeros(n_samples, dtype=float)
        for j in range(self.n_refs):
            total = np.array([_ref_mass_key(float(total[i]) + float(draws[i, j])) for i in range(n_samples)])
        return total

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._var

    @property
    def params(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "score_probs": self.score_probs.tolist(),
            "trust_probs": self.trust_probs.tolist(),
            "n_refs": self.n_refs,
            "scores": self.scores.tolist(),
            "trust_weights": self.trust_weights.tolist(),
        }
        return d

    @property
    def support(self) -> tuple[float, float]:
        return (self._support_lo, self._support_hi)

    def single_reference_pmf(self) -> dict[float, float]:
        """P(one letter product) for breakdown plots."""
        return dict(self._single_pmf)

    def aggregate_mass_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Sorted (values, probs) for the sum over n_refs."""
        sk = np.array(sorted(self._aggregate_pmf.keys()), dtype=float)
        return sk, np.array([self._aggregate_pmf[k] for k in sk], dtype=float)

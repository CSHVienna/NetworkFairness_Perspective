"""Signal – a named evidence channel with good/bad likelihoods."""

from __future__ import annotations

import numpy as np

from libs.distributions.base import Distribution


class Signal:
    """One observable signal in a Bayesian decision model.

    A signal wraps two distributions:
      - ``likelihood_good``:  P(signal | state = good)
      - ``likelihood_bad``:   P(signal | state = bad)

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"rec_letter"``).
    likelihood_good, likelihood_bad : Distribution
        Conditional distributions of the signal given each true state.
    signal_type : ``"network"`` | ``"non_network"``
        Whether this signal originates from the social network.
    """

    VALID_TYPES = ("network", "non_network")

    def __init__(
        self,
        name: str,
        likelihood_good: Distribution,
        likelihood_bad: Distribution,
        signal_type: str = "network",
        xmin: float | None = None,
        xmax: float | None = None,
    ) -> None:
        if signal_type not in self.VALID_TYPES:
            raise ValueError(
                f"signal_type must be one of {self.VALID_TYPES}, got '{signal_type}'"
            )
        self.name = name
        self.likelihood_good = likelihood_good
        self.likelihood_bad = likelihood_bad
        self.signal_type = signal_type
        self.xmin = xmin
        self.xmax = xmax

    # ------------------------------------------------------------------
    # Likelihood helpers
    # ------------------------------------------------------------------

    def likelihood_ratio(self, x) -> np.ndarray:
        """P(x | good) / P(x | bad)."""
        x = np.asarray(x, dtype=float)
        num = self.likelihood_good.pdf_or_pmf(x)
        den = self.likelihood_bad.pdf_or_pmf(x)
        return np.where(den > 0, num / den, np.inf)

    def log_likelihood_ratio(self, x) -> np.ndarray:
        """log P(x | good) - log P(x | bad).  Numerically stable."""
        x = np.asarray(x, dtype=float)
        return (
            self.likelihood_good.log_prob(x)
            - self.likelihood_bad.log_prob(x)
        )

    # ------------------------------------------------------------------
    # Discriminability
    # ------------------------------------------------------------------

    def d_prime(self) -> float:
        """Discriminability index (d').

        d' = |mu_good - mu_bad| / sqrt(0.5 * (var_good + var_bad))
        """
        mu_g = self.likelihood_good.mean
        mu_b = self.likelihood_bad.mean
        var_g = self.likelihood_good.variance
        var_b = self.likelihood_bad.variance
        pooled_std = np.sqrt(0.5 * (var_g + var_b))
        if pooled_std == 0:
            return float("inf")
        return float(abs(mu_g - mu_b) / pooled_std)

    def __repr__(self) -> str:
        return (
            f"Signal(name='{self.name}', type='{self.signal_type}', "
            f"d'={self.d_prime():.2f})"
        )

    def __str__(self) -> str:
        if self.signal_type == "network":
            return self.signal_type.lower()
        elif self.signal_type == "non_network":
            if self.name == "n_citations":
                return "citations"
            elif self.name == "n_publications":
                return "publications"
            else:
                return self.name
        return self.__repr__()

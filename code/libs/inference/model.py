"""Bayesian decision model – computes P(good | signals)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from libs.inference.signal import Signal


class BayesianDecisionModel:
    """Compute posteriors over a binary state given observed signals.

    Assumes conditional independence of signals given the true state::

        P(good | s1, s2, ...) ∝ P(good) · ∏_i P(s_i | good)

    Parameters
    ----------
    prior_good : float
        Prior probability that the candidate/item is *good*.
    signals : list[Signal]
        Evidence channels.  Order does not matter.
    """

    def __init__(self, prior_good: float, signals: list[Signal]) -> None:
        if not 0 < prior_good < 1:
            raise ValueError("prior_good must be in (0, 1)")
        self.prior_good = prior_good
        self._signals = {s.name: s for s in signals}

    # ------------------------------------------------------------------
    # Signal access
    # ------------------------------------------------------------------

    @property
    def signals(self) -> dict[str, Signal]:
        return dict(self._signals)

    @property
    def signal_names(self) -> list[str]:
        return list(self._signals)

    def get_signals_by_type(
        self, signal_type: Literal["network", "non_network"]
    ) -> list[Signal]:
        return [s for s in self._signals.values() if s.signal_type == signal_type]

    # ------------------------------------------------------------------
    # Core posterior computation
    # ------------------------------------------------------------------

    def log_posterior_odds(self, observations: dict[str, float]) -> float:
        """log P(good | obs) / P(bad | obs)  (log-odds form for stability)."""
        log_prior_odds = np.log(self.prior_good) - np.log(1 - self.prior_good)
        total_llr = 0.0
        for name, x in observations.items():
            sig = self._signals[name]
            total_llr += float(sig.log_likelihood_ratio(x))
        return log_prior_odds + total_llr

    def posterior(self, observations: dict[str, float]) -> float:
        """P(good | all observed signals).

        Parameters
        ----------
        observations : dict mapping signal name -> observed value
            Only signals present in the dict are used.
        """
        log_odds = self.log_posterior_odds(observations)
        return _sigmoid(log_odds)

    def posterior_by_signal(self, signal_name: str, x: float) -> float:
        """P(good | single signal), ignoring all other signals."""
        return self.posterior({signal_name: x})

    def posterior_by_type(
        self,
        observations: dict[str, float],
        signal_type: Literal["network", "non_network"],
    ) -> float:
        """P(good | signals of a given type only)."""
        subset = {
            name: observations[name]
            for name in observations
            if self._signals[name].signal_type == signal_type
        }
        return self.posterior(subset)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self, observations: dict[str, float], threshold: float = 0.5
    ) -> str:
        """Return ``'good'`` or ``'bad'`` based on the posterior."""
        return "good" if self.posterior(observations) >= threshold else "bad"

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sig_list = ", ".join(self._signals)
        return (
            f"BayesianDecisionModel(prior_good={self.prior_good}, "
            f"signals=[{sig_list}])"
        )


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    exp_x = np.exp(x)
    return float(exp_x / (1.0 + exp_x))

"""Game-theoretic payoff modelling for binary hire/reject decisions.

Provides:

- ``PayoffMatrix``: define utility/cost for each (decision, true_state) cell.
- ``expected_payoff``: given a posterior and a payoff matrix, return the
  expected payoff for each decision.
- ``optimal_decision``: return the decision that maximises expected payoff.
- ``optimal_threshold``: the posterior p* above which hiring is optimal.

The payoff-optimal threshold is:

    p* = (u_reject_bad - u_hire_bad)
       / (u_hire_good - u_hire_bad - u_reject_good + u_reject_bad)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PayoffMatrix:
    """Payoff matrix for binary decision (hire/reject) under binary true state (good/bad).

    Parameters
    ----------
    hire_good : float
        Payoff for hiring a truly good candidate.
    hire_bad : float
        Payoff (cost) for hiring a truly bad candidate.
    reject_good : float
        Payoff (cost) for rejecting a truly good candidate.
    reject_bad : float
        Payoff for correctly rejecting a bad candidate.
    """

    hire_good: float = 0.0
    hire_bad: float = 0.0
    reject_good: float = 0.0
    reject_bad: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PayoffMatrix:
        return cls(
            hire_good=float(d.get("hire_good", 0.0)),
            hire_bad=float(d.get("hire_bad", 0.0)),
            reject_good=float(d.get("reject_good", 0.0)),
            reject_bad=float(d.get("reject_bad", 0.0)),
        )

    def expected_payoff_hire(self, posterior_good: float | np.ndarray) -> float | np.ndarray:
        """Expected payoff of hiring given P(good)."""
        p = np.asarray(posterior_good, dtype=float)
        return p * self.hire_good + (1 - p) * self.hire_bad

    def expected_payoff_reject(self, posterior_good: float | np.ndarray) -> float | np.ndarray:
        """Expected payoff of rejecting given P(good)."""
        p = np.asarray(posterior_good, dtype=float)
        return p * self.reject_good + (1 - p) * self.reject_bad

    def expected_payoff(self, decision: str, posterior_good: float) -> float:
        """Expected payoff of *decision* given P(good)."""
        if decision == "hire":
            return float(self.expected_payoff_hire(posterior_good))
        elif decision == "reject":
            return float(self.expected_payoff_reject(posterior_good))
        raise ValueError(f"Unknown decision: {decision!r}; expected 'hire' or 'reject'")

    def optimal_decision(self, posterior_good: float | np.ndarray) -> str | np.ndarray:
        """Return the decision that maximises expected payoff.

        For scalar input returns ``'hire'`` or ``'reject'``.
        For array input returns an ndarray of strings.
        """
        p = np.asarray(posterior_good, dtype=float)
        hire_better = self.expected_payoff_hire(p) >= self.expected_payoff_reject(p)
        if p.ndim == 0:
            return "hire" if bool(hire_better) else "reject"
        return np.where(hire_better, "hire", "reject")

    def optimal_threshold(self) -> float:
        """Posterior p* above which hiring maximises expected payoff.

        p* = (u_reject_bad - u_hire_bad)
           / (u_hire_good - u_hire_bad - u_reject_good + u_reject_bad)

        Returns NaN if the denominator is zero (payoffs are degenerate).
        """
        denom = self.hire_good - self.hire_bad - self.reject_good + self.reject_bad
        if denom == 0:
            return float("nan")
        return (self.reject_bad - self.hire_bad) / denom

    def realized_payoff(
        self,
        decisions: np.ndarray,
        true_states: np.ndarray,
    ) -> float:
        """Total realised payoff over a population of decisions.

        Parameters
        ----------
        decisions : array of ``'hire'`` / ``'reject'``
        true_states : array of ``'good'`` / ``'bad'`` (or 1/0)
        """
        decisions = np.asarray(decisions)
        true_states = np.asarray(true_states)
        if true_states.dtype.kind in ("i", "f", "b"):
            is_good = true_states.astype(bool)
        else:
            is_good = np.array([s == "good" for s in true_states.ravel()])
        is_hire = np.array([d == "hire" for d in decisions.ravel()])

        total = 0.0
        total += np.sum(is_hire & is_good) * self.hire_good
        total += np.sum(is_hire & ~is_good) * self.hire_bad
        total += np.sum(~is_hire & is_good) * self.reject_good
        total += np.sum(~is_hire & ~is_good) * self.reject_bad
        return float(total)

    def to_dict(self) -> dict[str, float]:
        return {
            "hire_good": self.hire_good,
            "hire_bad": self.hire_bad,
            "reject_good": self.reject_good,
            "reject_bad": self.reject_bad,
        }

    def __repr__(self) -> str:
        return (
            f"PayoffMatrix(hire_good={self.hire_good}, hire_bad={self.hire_bad}, "
            f"reject_good={self.reject_good}, reject_bad={self.reject_bad}, "
            f"p*={self.optimal_threshold():.4f})"
        )

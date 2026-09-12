"""Bayesian inference engine: signals, models, and conjugate priors.

Quick start::

    from libs.inference import Signal, BayesianDecisionModel
"""

from libs.inference.conjugate import (
    BetaBinomialConjugate,
    ConjugatePrior,
    DirichletMultinomialConjugate,
    GammaPoissonConjugate,
    NormalLogNormalConjugate,
    NormalNormalConjugate,
)
from libs.inference.model import BayesianDecisionModel
from libs.inference.signal import Signal

__all__ = [
    "Signal",
    "BayesianDecisionModel",
    "ConjugatePrior",
    "BetaBinomialConjugate",
    "NormalNormalConjugate",
    "GammaPoissonConjugate",
    "DirichletMultinomialConjugate",
    "NormalLogNormalConjugate",
]

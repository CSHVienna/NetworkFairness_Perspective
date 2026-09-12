"""Probability distributions with a unified API.

Import concrete classes directly::

    from libs.distributions import BetaDist, NormalDist, PoissonDist

Or use the registry factory for config-driven construction::

    from libs.distributions import create_distribution
    dist = create_distribution("beta", {"alpha": 8, "beta": 2})
"""

from libs.distributions.base import (
    ContinuousDistribution,
    DiscreteDistribution,
    Distribution,
)
from libs.distributions.continuous import (
    BetaDist,
    GammaDist,
    LogNormalDist,
    NormalDist,
    PowerLawDist,
)
from libs.distributions.discrete import (
    BinomialDist,
    HypergeometricDist,
    MultinomialDist,
    PoissonDist,
)
from libs.distributions.registry import (
    create_distribution,
    name_for_distribution,
    registered_names,
)

__all__ = [
    "Distribution",
    "ContinuousDistribution",
    "DiscreteDistribution",
    "BetaDist",
    "NormalDist",
    "LogNormalDist",
    "GammaDist",
    "PowerLawDist",
    "PoissonDist",
    "BinomialDist",
    "HypergeometricDist",
    "MultinomialDist",
    "create_distribution",
    "name_for_distribution",
    "registered_names",
]

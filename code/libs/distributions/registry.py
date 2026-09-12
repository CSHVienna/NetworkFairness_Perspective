"""
Distribution factory – maps (name, params) to Distribution instances.

Every concrete distribution class uses the ``@register`` decorator so that
config files can reference distributions by short string names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libs.distributions.base import Distribution

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator that adds a distribution to the global registry."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def create_distribution(name: str, params: dict) -> "Distribution":
    """Instantiate a distribution by its registered name and parameters.

    >>> create_distribution("beta", {"alpha": 8, "beta": 2})
    BetaDist(alpha=8, beta=2)
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown distribution '{name}'. Registered: {available}"
        )
    return _REGISTRY[name](**params)


def registered_names() -> list[str]:
    """Return all registered distribution names."""
    return sorted(_REGISTRY)


def name_for_distribution(dist: "Distribution") -> str | None:
    """Return the registry name for the distribution's class, or None if not registered.

    >>> from libs.distributions import create_distribution
    >>> d = create_distribution("normal", {"mu": 0, "sigma": 1})
    >>> name_for_distribution(d)
    'normal'
    """
    cls = type(dist)
    for name, registered_cls in _REGISTRY.items():
        if registered_cls is cls:
            return name
    return None

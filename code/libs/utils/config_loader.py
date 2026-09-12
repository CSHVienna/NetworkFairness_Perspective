"""Load YAML configs: signal bundles (network / non_network), payoffs, and full scenarios."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

import libs.distributions.continuous as _continuous  # noqa: F401
import libs.distributions.discrete as _discrete  # noqa: F401
from libs.distributions.registry import create_distribution
from libs.inference.model import BayesianDecisionModel
from libs.inference.signal import Signal
from libs.payoffs.game import PayoffMatrix


@dataclass(frozen=True)
class SignalBundle:
    """A partial config: one side of the model (network or non_network signals)."""

    kind: Literal["network", "non_network"]
    name: str
    signals: list[Signal]
    simulation: dict[str, Any] | None
    source_path: Path | None = None


def load_signal_bundle(path: str | Path) -> SignalBundle:
    """Load a **bundle** YAML (``config/network/*.yaml`` or ``config/non_network/*.yaml``).

    Required structure::

        bundle:
          kind: network | non_network
          name: "short_id"
        signals:
          signal_name:
            type: network
            # Either legacy blocks:
            params_shared: { n_refs, scores, trust_weights }
            likelihood_good: {distribution: reference_weighted_sum, params: {...}}
            likelihood_bad:  {distribution: reference_weighted_sum, params: {...}}
            # Or named reference layout (like ProofOfConcept default.yaml):
            n_refs: 3
            reference_scores: {label: numeric_value, ...}
            referee_trust: {label: numeric_weight, ...}
            aggregation: sum   # optional; only sum is supported
            reference_score_good_raw / bad_raw: {same keys as reference_scores}
            referee_trust_good_raw / bad_raw: {same keys as referee_trust}
        simulation:  # optional
          seed: 42
    """
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    bundle = cfg.get("bundle") or {}
    kind = bundle.get("kind")
    if kind not in ("network", "non_network"):
        raise ValueError(
            f"{path}: bundle.kind must be 'network' or 'non_network', got {kind!r}"
        )
    name = bundle.get("name") or path.stem

    signals_cfg = cfg.get("signals") or {}
    if not signals_cfg:
        raise ValueError(f"{path}: no signals defined")

    signals = _signals_from_dict(signals_cfg)
    for s in signals:
        if s.signal_type != kind:
            raise ValueError(
                f"{path}: signal {s.name!r} has type {s.signal_type!r} but bundle.kind is {kind!r}"
            )

    return SignalBundle(
        kind=kind,
        name=name,
        signals=signals,
        simulation=cfg.get("simulation"),
        source_path=path,
    )


def load_prior(path: str | Path) -> float:
    """Load a prior config YAML.

    Expected structure::

        prior:
          good: 0.35
    """
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    prior_cfg = cfg.get("prior")
    if not prior_cfg:
        raise ValueError(f"{path}: missing 'prior' key")
    good = prior_cfg.get("good")
    if good is None:
        raise ValueError(f"{path}: missing 'prior.good' value")
    good = float(good)
    if not (0 < good < 1):
        raise ValueError(f"{path}: prior.good must be in (0, 1), got {good}")
    return good


def load_payoffs(path: str | Path) -> PayoffMatrix:
    """Load a payoff config YAML.

    Expected structure::

        payoffs:
          hire_good: 0
          hire_bad: -50
          reject_good: -100
          reject_bad: 0
    """
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    payoffs_cfg = cfg.get("payoffs")
    if not payoffs_cfg:
        raise ValueError(f"{path}: missing 'payoffs' key")
    return PayoffMatrix.from_dict(payoffs_cfg)


def combine_model(
    prior_good: float,
    *,
    network: str | Path | None = None,
    non_network: str | Path | None = None,
) -> BayesianDecisionModel:
    """Build a model by merging network and/or non_network bundle paths.

    At least one of *network* or *non_network* must be given.
    Signal names must be unique across bundles.

    Example::

        model = combine_model(
            0.35,
            network=CONFIG / "network/reference_trust_scores.yaml",
            non_network=CONFIG / "non_network/h_index_poisson.yaml",
        )
    """
    if network is None and non_network is None:
        raise ValueError("combine_model: pass at least one of network=, non_network=")

    merged: list[Signal] = []
    seen: set[str] = set()

    def add_bundle(path: str | Path | None, expected: Literal["network", "non_network"]) -> None:
        if path is None:
            return
        b = load_signal_bundle(path)
        if b.kind != expected:
            raise ValueError(f"{path}: expected {expected} bundle, got kind={b.kind!r}")
        for s in b.signals:
            if s.name in seen:
                raise ValueError(f"Duplicate signal name {s.name!r} when combining bundles")
            seen.add(s.name)
            merged.append(s)

    add_bundle(network, "network")
    add_bundle(non_network, "non_network")

    return BayesianDecisionModel(prior_good=prior_good, signals=merged)


def load_scenario(path: str | Path) -> BayesianDecisionModel:
    """Build a ``BayesianDecisionModel`` from YAML.

    **1. Composite scenario** (combine bundles by path)::

        scenario:
          name: "..."
        prior:
          good: 0.35
        components:
          network: network/reference_trust_scores.yaml
          non_network: non_network/h_index_poisson.yaml
        simulation: ...

        Paths are resolved relative to the scenario file's directory.

    **2. Legacy monolithic** (single file with all signals)::

        prior:
          good: 0.35
        signals:
          ...
    """
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "components" in cfg:
        prior_good = float(cfg["prior"]["good"])
        base = path.parent
        comp = cfg["components"] or {}
        net = comp.get("network")
        nn = comp.get("non_network")
        if not net and not nn:
            raise ValueError(f"{path}: components must include network and/or non_network path")
        return combine_model(
            prior_good,
            network=(base / net) if net else None,
            non_network=(base / nn) if nn else None,
        )

    if "prior" not in cfg or "signals" not in cfg:
        raise ValueError(
            f"{path}: expected either 'components' + 'prior', or legacy 'prior' + 'signals'"
        )

    prior_good = float(cfg["prior"]["good"])
    signals = _signals_from_dict(cfg["signals"])
    return BayesianDecisionModel(prior_good=prior_good, signals=signals)


def load_scenario_raw(path: str | Path) -> dict[str, Any]:
    """Return the parsed YAML dict for the scenario file (no merging of bundles)."""
    with open(Path(path).resolve()) as f:
        return yaml.safe_load(f)


def load_merged_scenario_raw(path: str | Path) -> dict[str, Any]:
    """Like ``load_scenario_raw`` but inlines bundle signal specs for composite scenarios.

    Useful for dashboards that expect a flat ``signals`` dict.
    """
    path = Path(path).resolve()
    raw = load_scenario_raw(path)
    if "components" not in raw:
        return raw
    base = path.parent
    comp = raw["components"] or {}
    signals: dict[str, Any] = {}
    for key in ("network", "non_network"):
        rel = comp.get(key)
        if not rel:
            continue
        bpath = base / rel
        with open(bpath) as f:
            partial = yaml.safe_load(f)
        sigs = partial.get("signals") or {}
        for sn, sc in sigs.items():
            signals[sn] = _expand_signal_config_for_display(copy.deepcopy(sc))
    out = {**raw, "signals": signals}
    out.pop("components", None)
    return out


def load_all_scenarios(
    config_dir: str | Path,
) -> dict[str, BayesianDecisionModel]:
    """Load every ``*.yaml`` / ``*.yml`` in *config_dir* (non-recursive).

    Point *config_dir* at ``config/combined/`` to load every composite scenario.
    Do not pass ``config/network/`` (bundles only, not full scenarios).
    """
    config_dir = Path(config_dir)
    models: dict[str, BayesianDecisionModel] = {}
    for p in sorted(config_dir.glob("*.y*ml")):
        raw = load_scenario_raw(p)
        name = raw.get("scenario", {}).get("name", p.stem)
        models[name] = load_scenario(p)
    return models


def _is_named_reference_signal(sig_cfg: dict[str, Any]) -> bool:
    return (
        isinstance(sig_cfg.get("reference_scores"), dict)
        and isinstance(sig_cfg.get("referee_trust"), dict)
        and isinstance(sig_cfg.get("reference_score_good_raw"), dict)
        and isinstance(sig_cfg.get("reference_score_bad_raw"), dict)
        and isinstance(sig_cfg.get("referee_trust_good_raw"), dict)
        and isinstance(sig_cfg.get("referee_trust_bad_raw"), dict)
    )


def _normalize_raw_to_probs(raw: dict[str, Any], keys: list[str], *, label: str) -> list[float]:
    arr = np.array([float(raw.get(k, 0.0)) for k in keys], dtype=float)
    s = float(arr.sum())
    if s <= 0:
        raise ValueError(f"{label}: raw weights must sum to a positive value, got sum={s}")
    return (arr / s).tolist()


def _named_reference_to_likelihood_signal(sig_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert reference_scores / *_raw blocks into reference_weighted_sum likelihoods."""
    ref_scores = sig_cfg["reference_scores"]
    ref_trust = sig_cfg["referee_trust"]
    if not ref_scores or not ref_trust:
        raise ValueError("reference_scores and referee_trust must be non-empty dicts")
    score_keys = list(ref_scores.keys())
    trust_keys = list(ref_trust.keys())
    scores = [float(ref_scores[k]) for k in score_keys]
    trust_weights = [float(ref_trust[k]) for k in trust_keys]

    agg = sig_cfg.get("aggregation", "sum")
    if agg != "sum":
        raise ValueError(f"aggregation must be 'sum' (only mode supported), got {agg!r}")

    for name, raw in (
        ("reference_score_good_raw", sig_cfg["reference_score_good_raw"]),
        ("reference_score_bad_raw", sig_cfg["reference_score_bad_raw"]),
    ):
        extra = set(raw) - set(score_keys)
        if extra:
            raise ValueError(f"{name}: unknown keys {extra!r}; expected subset of reference_scores keys")
    for name, raw in (
        ("referee_trust_good_raw", sig_cfg["referee_trust_good_raw"]),
        ("referee_trust_bad_raw", sig_cfg["referee_trust_bad_raw"]),
    ):
        extra = set(raw) - set(trust_keys)
        if extra:
            raise ValueError(f"{name}: unknown keys {extra!r}; expected subset of referee_trust keys")

    sg = _normalize_raw_to_probs(sig_cfg["reference_score_good_raw"], score_keys, label="reference_score_good_raw")
    sb = _normalize_raw_to_probs(sig_cfg["reference_score_bad_raw"], score_keys, label="reference_score_bad_raw")
    tg = _normalize_raw_to_probs(sig_cfg["referee_trust_good_raw"], trust_keys, label="referee_trust_good_raw")
    tb = _normalize_raw_to_probs(sig_cfg["referee_trust_bad_raw"], trust_keys, label="referee_trust_bad_raw")

    n_refs = int(sig_cfg.get("n_refs", 3))
    shared = {"n_refs": n_refs, "scores": scores, "trust_weights": trust_weights}
    return {
        "type": sig_cfg.get("type", "network"),
        "likelihood_good": {
            "distribution": "reference_weighted_sum",
            "params": {**shared, "score_probs": sg, "trust_probs": tg},
        },
        "likelihood_bad": {
            "distribution": "reference_weighted_sum",
            "params": {**shared, "score_probs": sb, "trust_probs": tb},
        },
    }


def _expand_params_shared_in_signal(sig_cfg: dict[str, Any]) -> dict[str, Any]:
    """Inline ``params_shared`` into both likelihood param dicts (for merged YAML / display)."""
    shared = sig_cfg.get("params_shared")
    if not shared:
        return sig_cfg
    out = copy.deepcopy(sig_cfg)
    out.pop("params_shared", None)
    for lk in ("likelihood_good", "likelihood_bad"):
        spec = copy.deepcopy(sig_cfg[lk])
        spec["params"] = {**copy.deepcopy(shared), **(spec.get("params") or {})}
        out[lk] = spec
    return out


def _expand_signal_config_for_display(sig_cfg: dict[str, Any]) -> dict[str, Any]:
    """Named reference -> explicit likelihood params; else params_shared merge."""
    if _is_named_reference_signal(sig_cfg):
        return _named_reference_to_likelihood_signal(sig_cfg)
    return _expand_params_shared_in_signal(sig_cfg)


def _signals_from_dict(signals_cfg: dict[str, Any]) -> list[Signal]:
    signals: list[Signal] = []
    for sig_name, sig_cfg in signals_cfg.items():
        sc = _expand_signal_config_for_display(sig_cfg)
        lik_good = _build_distribution(sc["likelihood_good"])
        lik_bad = _build_distribution(sc["likelihood_bad"])

        xmin = _resolve_bound(sc, "likelihood_good", "likelihood_bad", "xmin", min)
        xmax = _resolve_bound(sc, "likelihood_good", "likelihood_bad", "xmax", max)

        signals.append(
            Signal(
                name=sig_name,
                likelihood_good=lik_good,
                likelihood_bad=lik_bad,
                signal_type=sc.get("type", "network"),
                xmin=xmin,
                xmax=xmax,
            )
        )
    return signals


def _resolve_bound(
    sc: dict[str, Any],
    lik_good_key: str,
    lik_bad_key: str,
    bound_key: str,
    agg: Any,
) -> float | None:
    """Extract a signal-level xmin/xmax from the signal config or its likelihoods."""
    if bound_key in sc:
        return float(sc[bound_key])
    vals: list[float] = []
    for lik_key in (lik_good_key, lik_bad_key):
        lik = sc[lik_key]
        v = lik.get(bound_key) or (lik.get("params") or {}).get(bound_key)
        if v is not None:
            vals.append(float(v))
    return agg(vals) if vals else None


def _build_distribution(spec: dict) -> Any:
    params = dict(spec["params"])
    for key in ("xmin", "xmax"):
        if key in spec and key not in params:
            params[key] = spec[key]
    return create_distribution(spec["distribution"], params)

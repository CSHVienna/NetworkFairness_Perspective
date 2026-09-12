# Config layout

## Bundles (building blocks)

| Path | Purpose |
|------|---------|
| `network/*.yaml` | Network signals only. Must include `bundle.kind: network`. |
| `non_network/*.yaml` | Non-network bundles. Examples: `poisson`, `gaussian`, `lognormal`, `power_law`. |

### Reference-letter network signal (`reference_weighted_sum`)

**Preferred (named) layout** — same idea as ProofOfConcept `default.yaml`: numeric scales plus **raw** weights that are normalized to probabilities in code.

```yaml
rec_letter:
  type: network
  n_refs: 3
  aggregation: sum    # optional; only sum is implemented
  reference_scores:
    very_poor: 1
    poor: 2
    # ... labels -> numeric recommendation score
  referee_trust:
    untrustworthy: -1
    unknown: 0
    trustworthy: 1
  reference_score_good_raw:   # same keys as reference_scores
    very_poor: 0.03
    ...
  reference_score_bad_raw:
    ...
  referee_trust_good_raw:     # same keys as referee_trust
    untrustworthy: 0.06
    ...
  referee_trust_bad_raw:
    ...
```

**Legacy layout** (still supported): `params_shared` with `scores`, `trust_weights`, `n_refs`, plus `likelihood_good` / `likelihood_bad` each with `distribution: reference_weighted_sum` and `params.score_probs` / `trust_probs` (already normalized).

Omit `scores` / `trust_weights` in legacy mode only when using exactly 5 score probs and 3 trust probs (implicit 1–5 and −1/0/+1).

## Combined scenarios (`combined/*.yaml`)

Full models use **`components`** with paths **relative to the scenario file** (use `../network/`, `../non_network/`).

```yaml
prior:
  good: 0.35
components:
  network: ../network/reference_trust_scores.yaml
  non_network: ../non_network/h_index_poisson.yaml
```

## Python

```python
from pathlib import Path
from libs.utils.config_loader import load_scenario, combine_model, load_signal_bundle

CONFIG = Path(".../config")

# Full scenario from composite YAML
model = load_scenario(CONFIG / "combined" / "reference_trust_scores.yaml")

# Mix bundles without a scenario file
model = combine_model(
    0.35,
    network=CONFIG / "network/beta_rec_letter.yaml",
    non_network=CONFIG / "non_network/h_index_gaussian.yaml",
)

# Inspect one side
bundle = load_signal_bundle(CONFIG / "network/reference_scores_only.yaml")
```

Legacy monolithic YAML (single file with `prior` + `signals`, no `components`) is still supported by `load_scenario`.

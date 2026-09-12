# Network Fairness Perspective

Simulation code and analysis for a perspective paper on how **network signals**
(e.g. reference letters from trusted referees) and **non-network signals**
(e.g. citations, publications, h-index) combine in hiring decisions.

The core is a Bayesian decision model: a candidate is either *good* or *bad*, a
decision-maker observes one or more noisy signals, and hires when the posterior
probability of *good* exceeds a threshold. The threshold is either a fixed 0.5 or
the payoff-optimal `p*` derived from a hire/reject payoff matrix. Synthetic
candidates are simulated to measure accuracy, precision, and recall of each
signal alone and in combination, and to draw decision boundaries in the
network vs. non-network signal plane.

A small focus-group study (three groups) complements the simulations with
practitioners' stated minimum requirements and rankings of hiring criteria.

## Repository layout

```
code/
  libs/
    distributions/   Unified API over scipy/powerlaw distributions (Beta, Normal,
                     LogNormal, Gamma, PowerLaw, Poisson, Binomial, ...)
    inference/       Signal, BayesianDecisionModel, conjugate priors
    payoffs/         PayoffMatrix and the payoff-optimal threshold p*
    simulation/      DataSimulator, SimulationResults, decision-boundary helpers
    visualization/   Paper-style matplotlib/seaborn plots
    utils/           YAML config loading, path helpers, focus-group analysis
  scripts/
    run_simulation.py   Command-line entry point for one simulation run
  notebooks/
    1_bayesian_model.ipynb   Likelihoods and posteriors per signal
    2_simulations.ipynb      Simulations, metrics tables, decision boundaries
    3_focus_groups.ipynb     Focus-group statistics and plots
config/
  network/       Network signal bundles (reference scores x referee trust)
  non_network/   Non-network signal bundles (poisson, gaussian, lognormal, power_law)
  payoffs/       Hire/reject payoff matrices
  prior/         Prior P(good)
  README.md      Config format reference
results/
  plots/         Figures used in the paper
  simulations/   One folder per run: PDFs plus summary.json
requirements.txt
```

`config/focus/` and `data/` hold the focus-group metadata and raw responses.
They are excluded from version control and are needed only for notebook 3.

## Setup

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The library lives under `code/` and is imported as `libs.*`, so add that folder
to the Python path when running from the repository root:

```bash
export PYTHONPATH=$PYTHONPATH:code/
```

The notebooks do this themselves with a `sys.path` line in their first cell.

## Running a simulation

```bash
python code/scripts/run_simulation.py \
    --network      config/network/main_paper.yaml \
    --non-network  config/non_network/main_paper.yaml \
    --payoffs      config/payoffs/main_paper.yaml \
    --prior        config/prior/main_paper.yaml \
    --n-candidates 1000 \
    --seed 42 \
    --output-dir   results/simulations
```

The script creates a descriptively named subfolder inside `--output-dir`, for
example
`net_reference_trust_scores__nn_n_citations_lognormal__pstar_0.33__n_1000__seed_42/`,
containing:

- `prior.pdf`, `likelihood_<signal>.pdf`, `posterior_vs_<signal>.pdf`
- `posterior_histogram.pdf`, `posterior_heatmap.pdf`, `posterior_2d_decision_boundary.pdf`
- `accuracy_bars.pdf`, `metrics_comparison.pdf`
- `summary.json` with accuracy, precision, recall, thresholds, and realized payoff

Swap the `--non-network` bundle to compare signal families, for example
`config/non_network/poisson.yaml` (publication counts) or
`config/non_network/power_law.yaml` (h-index).

## Using the library directly

```python
from pathlib import Path
from libs.utils.config_loader import combine_model, load_payoffs, load_prior
from libs.simulation.simulator import DataSimulator

CONFIG = Path("config")

model = combine_model(
    load_prior(CONFIG / "prior/main_paper.yaml"),
    network=CONFIG / "network/main_paper.yaml",
    non_network=CONFIG / "non_network/main_paper.yaml",
)
payoffs = load_payoffs(CONFIG / "payoffs/main_paper.yaml")

sim = DataSimulator(model, n_candidates=1000, seed=42, payoff_matrix=payoffs)
results = sim.simulate(top_k=None)   # or a fraction, e.g. 0.2, for a top-k rule
print(results.accuracy_posterior, results.accuracy_payoff)
```

See [config/README.md](config/README.md) for the YAML formats, including the
reference-letter network signal (`n_refs` letters, each with a recommendation
score and a referee-trust level, aggregated by sum).

## Main-paper configuration

| Setting | Value |
|---|---|
| Prior P(good) | 0.35 |
| Network signal | 3 reference letters, score 1 to 5, trust in {-1, 0, +1} |
| Non-network signal | Citations, LogNormal(5.0, 0.5) if good, LogNormal(2.0, 1.0) if bad |
| Payoffs | hire_good 0, hire_bad -50, reject_good -100, reject_bad 0 |
| Optimal threshold p* | 0.33 |

## Reproducing the paper figures

1. Run notebook 1 to regenerate per-signal likelihood and posterior figures.
2. Run notebook 2 to regenerate the simulations, metrics tables (LaTeX), and
   decision-boundary figures in `results/plots/`.
3. Run notebook 3 for the focus-group figures. This requires the private
   `config/focus/focus.yaml` and the `data/*.xlsx` files.

## Citation

If you use this code, please cite the repository:

```bibtex
@software{espin2026networkfairness,
  author  = {Esp{\'i}n, Lisette},
  title   = {NetworkFairness\_Perspective: Bayesian simulations of network and non-network signals in hiring},
  year    = {2026},
  url     = {https://github.com/CSHVienna/NetworkFairness_Perspective},
  note    = {GitHub repository}
}
```

A reference to the accompanying paper will be added here once it is published.

## License

This work is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](LICENSE)
license. You may share and adapt it for non-commercial purposes with attribution,
provided derivatives are released under the same license. See
[creativecommons.org/licenses/by-nc-sa/4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
for a summary.

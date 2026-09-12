#!/usr/bin/env python3
"""Run a simulation: generate candidates, apply the Bayesian model, evaluate accuracy.

Usage
-----
    python code/scripts/run_simulation.py \
        --network   config/network/main_paper.yaml \
        --non-network config/non_network/main_paper.yaml \
        --payoffs   config/payoffs/main_paper.yaml \
        --prior     config/prior/main_paper.yaml \
        --n-candidates 1000 \
        --seed 42 \
        --output-dir results/simulations

The script creates a descriptively-named subfolder inside --output-dir
containing all plots and a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (from root project folder) export PYTHONPATH=$PYTHONPATH:code/
# ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(ROOT / "code"))

import numpy as np
from matplotlib import pyplot as plt

from libs.utils.config_loader import combine_model, load_payoffs, load_prior, load_signal_bundle
from libs.simulation.simulator import DataSimulator
from libs.payoffs.game import PayoffMatrix
from libs.visualization.plots import (
    COLOR_GOOD,
    COLOR_BAD,
    set_paper_style,
    save_fig,
    plot_prior,
    plot_signal_distributions,
    plot_posterior_vs_signal,
    plot_posterior_histogram,
    plot_accuracy_bars,
    plot_posterior_heatmap_network_nonnetwork,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate candidates and evaluate Bayesian hiring model."
    )
    p.add_argument("--network", required=True, help="Path to network signal bundle YAML")
    p.add_argument("--non-network", required=True, help="Path to non-network signal bundle YAML")
    p.add_argument("--payoffs", required=True, help="Path to payoffs YAML")
    p.add_argument("--prior", required=True, help="Path to prior config YAML (or a float for backwards compat)")
    p.add_argument("--n-candidates", type=int, default=1000, help="Number of candidates")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output-dir", default="results/simulations", help="Parent output directory")
    return p.parse_args()


def build_run_name(
    net_bundle_name: str,
    nn_bundle_name: str,
    payoff_matrix: PayoffMatrix,
    n_candidates: int,
    seed: int,
) -> str:
    pstar = payoff_matrix.optimal_threshold()
    parts = [
        f"net_{net_bundle_name}",
        f"nn_{nn_bundle_name}",
        f"pstar_{pstar:.2f}",
        f"n_{n_candidates}",
        f"seed_{seed}",
    ]
    return "__".join(parts)


def plot_signal_likelihoods(model, out_dir: Path) -> None:
    """Plot each signal's good/bad likelihood distributions."""
    for name, sig in model.signals.items():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        plot_signal_distributions(sig, ax=ax)
        save_fig(fig, out_dir / f"likelihood_{name}.pdf", show=False)


def plot_posteriors_per_signal(model, out_dir: Path) -> None:
    """Plot P(good | single signal) for each signal."""
    for name, sig in model.signals.items():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        plot_posterior_vs_signal(model, signal_name=name, ax=ax)
        save_fig(fig, out_dir / f"posterior_vs_{name}.pdf", show=False)


def plot_simulation_summary(results, model, payoff_matrix, out_dir: Path) -> None:
    """Posterior histogram, accuracy bars, and confusion summary."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_posterior_histogram(results.posteriors, results.true_states, ax=ax)
    if results.threshold_payoff is not None:
        ax.axvline(
            results.threshold_payoff,
            color="orange",
            linestyle="--",
            alpha=0.8,
            label=f"payoff threshold p*={results.threshold_payoff:.2f}",
        )
        ax.legend()
    save_fig(fig, out_dir / "posterior_histogram.pdf", show=False)

    accuracies = dict(results.per_signal_accuracy_posterior)
    accuracies["combined (p≥0.5)"] = results.accuracy_posterior
    if results.accuracy_payoff is not None:
        accuracies[f"combined (p≥{results.threshold_payoff:.2f})"] = results.accuracy_payoff
        for name, acc in results.per_signal_accuracy_payoff.items():
            accuracies[f"{name} (p≥{results.threshold_payoff:.2f})"] = acc
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_accuracy_bars(accuracies, ax=ax)
    save_fig(fig, out_dir / "accuracy_bars.pdf", show=False)


def plot_metrics_comparison(results, out_dir: Path) -> None:
    """Bar chart comparing posterior-based vs payoff-based precision/recall/accuracy."""
    labels = ["Accuracy", "Precision", "Recall"]
    post_vals = [results.accuracy_posterior, results.precision_posterior, results.recall_posterior]
    payoff_vals = [results.accuracy_payoff, results.precision_payoff, results.recall_payoff]

    if any(v is None for v in payoff_vals):
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width / 2, post_vals, width, label="Posterior (p≥0.5)", color=COLOR_GOOD, alpha=0.8)
    bars2 = ax.bar(
        x + width / 2,
        payoff_vals,
        width,
        label=f"Payoff (p≥{results.threshold_payoff:.2f})",
        color="C1",
        alpha=0.8,
    )

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Posterior vs Payoff decision rule")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    save_fig(fig, out_dir / "metrics_comparison.pdf", show=False)


def write_summary(results, payoff_matrix, run_name, out_dir: Path) -> None:
    summary = {
        "run_name": run_name,
        "n_candidates": results.n_candidates,
        "seed": results.seed,
        "threshold_posterior": results.threshold_posterior,
        "threshold_payoff": results.threshold_payoff,
        "accuracy_posterior": results.accuracy_posterior,
        "accuracy_payoff": results.accuracy_payoff,
        "precision_posterior": results.precision_posterior,
        "precision_payoff": results.precision_payoff,
        "recall_posterior": results.recall_posterior,
        "recall_payoff": results.recall_payoff,
        "realized_payoff": results.realized_payoff,
        "per_signal_accuracy_posterior": results.per_signal_accuracy_posterior,
        "per_signal_accuracy_payoff": results.per_signal_accuracy_payoff,
    }
    if payoff_matrix is not None:
        summary["payoff_matrix"] = payoff_matrix.to_dict()
    path = out_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary written to {path}")


def main() -> None:
    args = parse_args()

    net_path = Path(args.network)
    nn_path = Path(args.non_network)
    payoff_path = Path(args.payoffs)

    try:
        prior_good = float(args.prior)
    except ValueError:
        prior_good = load_prior(args.prior)

    net_bundle = load_signal_bundle(net_path)
    nn_bundle = load_signal_bundle(nn_path)
    payoff_matrix = load_payoffs(payoff_path)

    model = combine_model(
        prior_good,
        network=net_path,
        non_network=nn_path,
    )

    run_name = build_run_name(
        net_bundle.name, nn_bundle.name, payoff_matrix, args.n_candidates, args.seed
    )
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    set_paper_style()

    print(f"Model: {model}")
    print(f"Payoffs: {payoff_matrix}")
    print(f"Optimal threshold p* = {payoff_matrix.optimal_threshold():.4f}")
    print(f"Simulating {args.n_candidates} candidates (seed={args.seed})...")

    sim = DataSimulator(
        model=model,
        n_candidates=args.n_candidates,
        seed=args.seed,
        payoff_matrix=payoff_matrix,
    )
    results = sim.simulate(top_k=None)

    print(f"\n{'='*60}")
    print(f"  Posterior rule (p >= 0.5)")
    print(f"    Accuracy:  {results.accuracy_posterior:.4f}")
    print(f"    Precision: {results.precision_posterior:.4f}")
    print(f"    Recall:    {results.recall_posterior:.4f}")
    print(f"  Payoff rule (p >= {results.threshold_payoff:.4f})")
    print(f"    Accuracy:  {results.accuracy_payoff:.4f}")
    print(f"    Precision: {results.precision_payoff:.4f}")
    print(f"    Recall:    {results.recall_payoff:.4f}")
    print(f"    Realized payoff: {results.realized_payoff:.0f}")
    print(f"  Per-signal accuracy (p>=0.5):")
    for name, acc in results.per_signal_accuracy_posterior.items():
        print(f"    {name}: {acc:.4f}")
    if results.per_signal_accuracy_payoff:
        print(f"  Per-signal accuracy (p>={results.threshold_payoff:.4f}):")
        for name, acc in results.per_signal_accuracy_payoff.items():
            print(f"    {name}: {acc:.4f}")
    print(f"{'='*60}\n")

    print("Generating plots...")

    fig, ax = plt.subplots(figsize=(3.5, 3))
    plot_prior(model, ax=ax)
    save_fig(fig, out_dir / "prior.pdf", show=False)

    plot_signal_likelihoods(model, out_dir)
    plot_posteriors_per_signal(model, out_dir)
    plot_simulation_summary(results, model, payoff_matrix, out_dir)
    plot_metrics_comparison(results, out_dir)

    # -- plain heatmap with contour iso-lines --
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_posterior_heatmap_network_nonnetwork(
        model,
        ax=ax,
        gaussian_sigma=1.5,
        contour_levels={"start": 0.1, "stop": 1.0, "step": 0.1},
    )
    save_fig(fig, out_dir / "posterior_heatmap.pdf", show=False)

    # -- heatmap with decision boundaries + simulated observations --
    net_signals = model.get_signals_by_type("network")
    non_signals = model.get_signals_by_type("non_network")
    if len(net_signals) == 1 and len(non_signals) == 1:
        s_net, s_non = net_signals[0], non_signals[0]
        is_good = results.true_states == "good"
        obs_non = results.observations[s_non.name]
        obs_net = results.observations[s_net.name]

        boundaries = [
            {"level": 0.5, "linestyle": "solid", "label": "Posterior rule (p \u2265 0.5)", "fmt": "p=%.1f"},
        ]
        if payoff_matrix is not None:
            pstar = payoff_matrix.optimal_threshold()
            if 0 < pstar < 1:
                boundaries.append({
                    "level": pstar,
                    "linestyle": "dashed",
                    "label": f"Payoff-optimal (p \u2265 {pstar:.2f})",
                    "fmt": "p=p*=%.2f",
                })

        scatter = [
            {"x": obs_non[is_good], "y": obs_net[is_good], "color": COLOR_GOOD, "label": "true good", "sample_n": 300},
            {"x": obs_non[~is_good], "y": obs_net[~is_good], "color": COLOR_BAD, "label": "true bad", "sample_n": 300},
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_posterior_heatmap_network_nonnetwork(
            model,
            ax=ax,
            n_x=120,
            n_y=120,
            gaussian_sigma=1.5,
            decision_boundaries=boundaries,
            scatter_data=scatter,
            aesthetics={
                "title": "Posterior & decision boundaries",
                "cbar_label": "P(good | signals)",
            },
        )
        save_fig(fig, out_dir / "posterior_2d_decision_boundary.pdf", show=False)
    else:
        print("Skipping 2D boundary plot: need exactly 1 network and 1 non-network signal.")

    write_summary(results, payoff_matrix, run_name, out_dir)
    print(f"\nDone! All outputs in: {out_dir}")


if __name__ == "__main__":
    main()

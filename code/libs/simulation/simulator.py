"""Data simulation and accuracy evaluation.

- ``DataSimulator``: generate synthetic candidates with true states and
  observed signals drawn from the model's likelihood distributions.
- ``compute_accuracy``: classify each simulated candidate via the posterior
  and compare against the true state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from libs.inference.model import BayesianDecisionModel
from libs.payoffs.game import PayoffMatrix

DEFAULT_POSTERIOR_THRESHOLD = 0.5
DECISION_HIRE = "hire"
DECISION_REJECT = "reject"
LATENT_GOOD = "good"
LATENT_BAD = "bad"

_RESULT_COLS = [
    "accuracy_posterior", "accuracy_payoff", "accuracy_topk",
    "precision_posterior", "precision_payoff", "precision_topk",
    "recall_posterior", "recall_payoff", "recall_topk",
    "threshold_posterior", "threshold_payoff",
    "realized_payoff", "n_candidates", "seed", "top_k",
]


@dataclass
class SimulationResults:
    """Container for one simulation run."""

    true_states: np.ndarray
    observations: dict[str, np.ndarray]
    posteriors: np.ndarray
    decisions_posterior: np.ndarray
    decisions_payoff: np.ndarray | None
    accuracy_posterior: float
    accuracy_payoff: float | None
    precision_posterior: float
    precision_payoff: float | None
    recall_posterior: float
    recall_payoff: float | None
    threshold_posterior: float
    threshold_payoff: float | None
    realized_payoff: float | None
    decisions_topk: np.ndarray | None = None
    accuracy_topk: float | None = None
    precision_topk: float | None = None
    recall_topk: float | None = None
    threshold_topk: dict[str, float] | None = None
    top_k: float | None = None
    per_signal_accuracy_posterior: dict[str, float] = field(default_factory=dict)
    per_signal_accuracy_payoff: dict[str, float] = field(default_factory=dict)
    per_signal_accuracy_topk: dict[str, float] = field(default_factory=dict)
    n_candidates: int = 0
    seed: int | None = None
    payoff_matrix: PayoffMatrix | None = None


class DataSimulator:
    """Generate synthetic datasets from a BayesianDecisionModel.

    Parameters
    ----------
    model : BayesianDecisionModel
        The model whose prior and likelihoods define the data-generating process.
    n_candidates : int
        Number of candidates to simulate per run.
    seed : int | None
        Random seed for reproducibility.
    payoff_matrix : PayoffMatrix | None
        If provided, also evaluate the payoff-optimal decision rule.
    top_k : float | None
        If provided, evaluate the top-right-corner decision rule.  This is the
        percentage of the range to descend from the maximum for each signal
        (passed to ``top_of_range_threshold``).  Candidates at or above every
        signal threshold are hired.
    """

    def __init__(
        self,
        model: BayesianDecisionModel,
        n_candidates: int = 500,
        seed: int | None = None,
        payoff_matrix: PayoffMatrix | None = None,
    ):
        self.model = model
        self.n_candidates = n_candidates
        self.seed = seed
        self.payoff_matrix = payoff_matrix
        np.random.seed(seed)

    def simulate(self, top_k: float | None) -> SimulationResults:
        """Generate candidates, compute posteriors, classify, and evaluate."""
        rng = np.random.default_rng(self.seed)

        is_good = rng.random(self.n_candidates) < self.model.prior_good
        true_states = np.where(is_good, LATENT_GOOD, LATENT_BAD)

        observations: dict[str, np.ndarray] = {}
        for name, sig in self.model.signals.items():
            vals = np.empty(self.n_candidates, dtype=float)
            n_good = int(is_good.sum())
            n_bad = self.n_candidates - n_good
            if n_good > 0:
                vals[is_good] = sig.likelihood_good.sample(n_good, rng=rng)
            if n_bad > 0:
                vals[~is_good] = sig.likelihood_bad.sample(n_bad, rng=rng)
            observations[name] = vals

        posteriors = np.array([
            self.model.posterior({name: observations[name][i] for name in observations})
            for i in range(self.n_candidates)
        ])

        threshold_posterior = DEFAULT_POSTERIOR_THRESHOLD
        decisions_posterior = np.where(posteriors >= threshold_posterior, DECISION_HIRE, DECISION_REJECT)
        accuracy_posterior, precision_posterior, recall_posterior = _metrics(
            decisions_posterior, true_states
        )

        threshold_payoff = None
        decisions_payoff = None
        accuracy_payoff = None
        precision_payoff = None
        recall_payoff = None
        realized_payoff = None

        if self.payoff_matrix is not None:
            threshold_payoff = self.payoff_matrix.optimal_threshold()
            decisions_payoff = np.where(posteriors >= threshold_payoff, DECISION_HIRE, DECISION_REJECT)
            accuracy_payoff, precision_payoff, recall_payoff = _metrics(
                decisions_payoff, true_states
            )
            realized_payoff = self.payoff_matrix.realized_payoff(decisions_payoff, true_states)

        threshold_topk = None
        decisions_topk = None
        accuracy_topk = None
        precision_topk = None
        recall_topk = None

        if top_k is not None:
            threshold_topk = {
                name: top_of_range_threshold(observations[name], top_k)
                for name in observations
            }
            mask = np.ones(self.n_candidates, dtype=bool)
            for name, thresh in threshold_topk.items():
                mask &= observations[name] >= thresh
            decisions_topk = np.where(mask, DECISION_HIRE, DECISION_REJECT)
            accuracy_topk, precision_topk, recall_topk = _metrics(
                decisions_topk, true_states
            )

        per_signal_accuracy_posterior = _per_signal_accuracies(
            self.model, observations, self.n_candidates, true_states, threshold_posterior
        )
        per_signal_accuracy_payoff = (
            _per_signal_accuracies(
                self.model, observations, self.n_candidates, true_states, threshold_payoff
            )
            if threshold_payoff is not None
            else {}
        )
        per_signal_accuracy_topk = (
            _per_signal_accuracies_topk(
                observations, true_states, threshold_topk
            )
            if threshold_topk is not None
            else {}
        )

        return SimulationResults(
            n_candidates=self.n_candidates,
            true_states=true_states,
            observations=observations,
            posteriors=posteriors,
            top_k=top_k,
            payoff_matrix=self.payoff_matrix,

            decisions_posterior=decisions_posterior,
            decisions_payoff=decisions_payoff,
            decisions_topk=decisions_topk,

            accuracy_posterior=accuracy_posterior,
            accuracy_payoff=accuracy_payoff,
            accuracy_topk=accuracy_topk,

            precision_posterior=precision_posterior,
            precision_payoff=precision_payoff,
            precision_topk=precision_topk,

            recall_posterior=recall_posterior,
            recall_payoff=recall_payoff,
            recall_topk=recall_topk,

            threshold_posterior=threshold_posterior,
            threshold_payoff=threshold_payoff,
            threshold_topk=threshold_topk,

            
            per_signal_accuracy_posterior=per_signal_accuracy_posterior,
            per_signal_accuracy_payoff=per_signal_accuracy_payoff,
            per_signal_accuracy_topk=per_signal_accuracy_topk,
            
            realized_payoff=realized_payoff,
            seed=self.seed,
        )


def _metrics(
    decisions: np.ndarray, true_states: np.ndarray
) -> tuple[float, float, float]:
    """Compute accuracy, precision, and recall.

    "hire" on "good" = true positive; "reject" on "bad" = true negative.
    """
    is_good = true_states == LATENT_GOOD
    is_hire = decisions == DECISION_HIRE

    tp = np.sum(is_hire & is_good)
    tn = np.sum(~is_hire & ~is_good)
    fp = np.sum(is_hire & ~is_good)
    fn = np.sum(~is_hire & is_good)

    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return accuracy, precision, recall


def _per_signal_accuracies(
    model: BayesianDecisionModel,
    observations: dict[str, np.ndarray],
    n_candidates: int,
    true_states: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute per-signal accuracy using each signal in isolation."""
    result = {}
    for name in model.signals:
        single_posteriors = np.array([
            model.posterior({name: observations[name][i]})
            for i in range(n_candidates)
        ])
        decisions = np.where(single_posteriors >= threshold, DECISION_HIRE, DECISION_REJECT)
        acc, _, _ = _metrics(decisions, true_states)
        result[name] = acc
    return result


def _per_signal_accuracies_topk(
    observations: dict[str, np.ndarray],
    true_states: np.ndarray,
    threshold_topk: dict[str, float],
) -> dict[str, float]:
    """Compute per-signal accuracy using each signal's top-k threshold in isolation."""
    result = {}
    for name, thresh in threshold_topk.items():
        decisions = np.where(
            observations[name] >= thresh, DECISION_HIRE, DECISION_REJECT
        )
        acc, _, _ = _metrics(decisions, true_states)
        result[name] = acc
    return result


@dataclass
class RegionFilterResult:
    """Candidates filtered by a rectangular signal-value region."""

    mask_inside: np.ndarray
    indices_inside: np.ndarray
    indices_outside: np.ndarray
    n_inside: int
    n_outside: int
    network_threshold: float
    non_network_threshold: float


@dataclass
class DecisionBoundaryResult:
    """Candidates split by a decision boundary (hire vs reject)."""

    mask_hire: np.ndarray
    indices_hire: np.ndarray
    indices_reject: np.ndarray
    n_hire: int
    n_reject: int
    threshold: float


def candidates_by_decision(
    result: SimulationResults,
) -> dict[str, DecisionBoundaryResult]:
    """Split candidates by posterior and payoff decision boundaries.

    Returns a dict with keys ``'posterior'`` and, when the simulation
    included a payoff matrix, ``'payoff'``.
    """
    out: dict[str, DecisionBoundaryResult] = {}

    mask_post = result.decisions_posterior == DECISION_HIRE
    out["posterior"] = DecisionBoundaryResult(
        mask_hire=mask_post,
        indices_hire=np.where(mask_post)[0],
        indices_reject=np.where(~mask_post)[0],
        n_hire=int(mask_post.sum()),
        n_reject=int((~mask_post).sum()),
        threshold=result.threshold_posterior,
    )

    if result.decisions_payoff is not None and result.threshold_payoff is not None:
        mask_pay = result.decisions_payoff == DECISION_HIRE
        out["payoff"] = DecisionBoundaryResult(
            mask_hire=mask_pay,
            indices_hire=np.where(mask_pay)[0],
            indices_reject=np.where(~mask_pay)[0],
            n_hire=int(mask_pay.sum()),
            n_reject=int((~mask_pay).sum()),
            threshold=result.threshold_payoff,
        )

    return out


def candidates_in_top_percentiles(
    observations: dict[str, np.ndarray],
    network_signal: str,
    non_network_signal: str,
    network_percentile: float,
    non_network_percentile: float,
) -> RegionFilterResult:
    """Return candidates above both signal percentiles (the top-right corner).

    Parameters
    ----------
    observations : dict[str, np.ndarray]
        Signal observations per candidate (from ``SimulationResults.observations``).
    network_signal, non_network_signal : str
        Signal names (keys in *observations*).
    network_percentile, non_network_percentile : float
        Percentile thresholds (0–100).  Candidates **at or above** each
        threshold are considered "top" on that axis.
    """
    net_vals = observations[network_signal]
    nn_vals = observations[non_network_signal]

    net_threshold = float(np.percentile(net_vals, network_percentile))
    nn_threshold = float(np.percentile(nn_vals, non_network_percentile))

    mask = (net_vals >= net_threshold) & (nn_vals >= nn_threshold)

    return RegionFilterResult(
        mask_inside=mask,
        indices_inside=np.where(mask)[0],
        indices_outside=np.where(~mask)[0],
        n_inside=int(mask.sum()),
        n_outside=int((~mask).sum()),
        network_threshold=net_threshold,
        non_network_threshold=nn_threshold,
    )


def top_of_range_threshold(values: np.ndarray, k_percent: float) -> float:
    """Return the value that is ``k_percent`` of the way down from the maximum.

    Parameters
    ----------
    values : np.ndarray
        Array of signal values.
    k_percent : float
        Percentage of the range to descend from the maximum (0–100).
        ``k_percent=0`` returns the maximum; ``k_percent=100`` returns the minimum.
    """
    lo = float(np.min(values))
    hi = float(np.max(values))
    return hi - (k_percent / 100.0) * (hi - lo)


def candidates_in_region(
    observations: dict[str, np.ndarray],
    network_signal: str,
    non_network_signal: str,
    network_min: float,
    non_network_min: float,
) -> RegionFilterResult:
    """Return candidates at or above explicit signal-value thresholds.

    Parameters
    ----------
    observations : dict[str, np.ndarray]
        Signal observations per candidate (from ``SimulationResults.observations``).
    network_signal, non_network_signal : str
        Signal names (keys in *observations*).
    network_min, non_network_min : float
        Minimum signal values.  Candidates with **both** signals >= these
        values are "inside" the region.
    """
    net_vals = observations[network_signal]
    nn_vals = observations[non_network_signal]

    mask = (net_vals >= network_min) & (nn_vals >= non_network_min)

    return RegionFilterResult(
        mask_inside=mask,
        indices_inside=np.where(mask)[0],
        indices_outside=np.where(~mask)[0],
        n_inside=int(mask.sum()),
        n_outside=int((~mask).sum()),
        network_threshold=float(network_min),
        non_network_threshold=float(non_network_min),
    )


def summarize_decisions(
    results: dict[tuple[str, str], SimulationResults],
    network_min: float = 10,
    non_network_min: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Summarise hire/reject splits for each simulation result.

    Parameters
    ----------
    results : dict[(network_name, non_network_name), SimulationResults]
        Simulation runs keyed by signal-name pairs.
    network_min : float
        Minimum network signal value for the top-right region rule.
    non_network_min : dict[str, float] | None
        Per-non-network-signal minimum values for the top-right region rule.
        Defaults to ``{'n_citations': 440, 'n_publications': 30, 'h_index': 120}``.
    """
    if non_network_min is None:
        non_network_min = {"n_citations": 440, "n_publications": 30, "h_index": 120}

    rows: list[dict[str, Any]] = []
    for (n_name, nn_name), result in results.items():
        filt = candidates_in_region(
            result.observations,
            network_signal=n_name,
            non_network_signal=nn_name,
            network_min=network_min,
            non_network_min=non_network_min[nn_name],
        )
        splits = candidates_by_decision(result)

        rows.append({
            "n_name": n_name,
            "nn_name": nn_name,
            "inside_top_right": filt.n_inside,
            "outside_top_right": filt.n_outside,
            "inside_posterior": splits["posterior"].n_hire,
            "outside_posterior": splits["posterior"].n_reject,
            "inside_payoff": splits["payoff"].n_hire,
            "outside_payoff": splits["payoff"].n_reject,
        })

    return pd.DataFrame(rows)


def decisions_to_latex(df: pd.DataFrame) -> str:
    """Generate a LaTeX table from *summarize_decisions* output.

    Columns: Observed signals | Posterior-based (inside / outside) |
    Payoff-based (inside / outside) | Top-right rule (inside / outside).
    """
    rows: list[str] = []
    for _, r in df.iterrows():
        net_name = r["n_name"].replace("_", r"\_")
        nn_name = r["nn_name"].replace("_", r"\_")
        signals_col = f"{net_name}, {nn_name}"

        posterior = f"{int(r['inside_posterior'])} / {int(r['outside_posterior'])}"
        payoff = f"{int(r['inside_payoff'])} / {int(r['outside_payoff'])}"
        top_right = f"{int(r['inside_top_right'])} / {int(r['outside_top_right'])}"

        rows.append(f"{signals_col} & {posterior} & {payoff} & {top_right} \\\\")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        (r"Observed signals & Posterior-based & Payoff-based & Top-right rule \\"),
        (r"(net., non-net.) & good / bad & good / bad & good / bad \\"),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def summarize_results(
    results: dict[tuple[str, str], SimulationResults],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (n_name, nn_name), result in results.items():
        obj: dict[str, Any] = {"network_signal": n_name, "nn_name": nn_name}
        obj |= {k: v for k, v in result.__dict__.items() if k in _RESULT_COLS}
        obj["network_signal_accuracy_posterior"] = result.per_signal_accuracy_posterior.get(n_name)
        obj["non_network_signal_accuracy_posterior"] = result.per_signal_accuracy_posterior.get(nn_name)
        obj["network_signal_accuracy_payoff"] = result.per_signal_accuracy_payoff.get(n_name)
        obj["non_network_signal_accuracy_payoff"] = result.per_signal_accuracy_payoff.get(nn_name)
        obj["network_signal_accuracy_topk"] = result.per_signal_accuracy_topk.get(n_name)
        obj["non_network_signal_accuracy_topk"] = result.per_signal_accuracy_topk.get(nn_name)
        rows.append(obj)
    return pd.DataFrame(rows)


def results_to_latex(df: pd.DataFrame) -> str:
    """Generate a formatted LaTeX table from *summarize_results* output.

    Bolds the best value per column: highest for accuracy, precision,
    and recall; highest (least-negative) for realized payoff.
    In the 1D Accuracy ratio column, ratios > 1 are bolded (network
    signal outperforms non-network signal individually).
    Requires ``\\usepackage{booktabs}`` in the LaTeX preamble.
    """
    higher_is_better = {
        "acc_post": "accuracy_posterior",
        "acc_pay": "accuracy_payoff",
        "acc_topk": "accuracy_topk",
        "prec_post": "precision_posterior",
        "prec_pay": "precision_payoff",
        "prec_topk": "precision_topk",
        "rec_post": "recall_posterior",
        "rec_pay": "recall_payoff",
        "rec_topk": "recall_topk",
    }
    best = {k: df[col].max() for k, col in higher_is_better.items()}
    best["payoff"] = df["realized_payoff"].max()

    def _b(val: float, best_val: float) -> str:
        s = f"{val:.2f}"
        return rf"\textbf{{{s}}}" if np.isclose(val, best_val) else s

    def _payoff_fmt(val: float, best_val: float) -> str:
        abs_v = abs(val)
        if abs_v >= 10_000:
            num = (f"{int(abs_v // 1_000)}k" if abs_v % 1_000 == 0
                   else f"{abs_v / 1_000:.1f}k")
        else:
            num = f"{abs_v:,.0f}"
        body = f"$-${num}" if val < 0 else num
        return rf"\textbf{{{body}}}" if np.isclose(val, best_val) else body

    def _ratio(net_acc: float, nn_acc: float) -> str:
        ratio = net_acc / nn_acc if nn_acc > 0 else float("inf")
        s = f"{ratio:.2f}"
        return rf"\textbf{{{s}}}" if ratio > 1 else s

    rows: list[str] = []
    for _, r in df.iterrows():
        net_name = r["network_signal"].replace("_", r"\_")
        nn_name = r["nn_name"].replace("_", r"\_")
        signals_col = f"{net_name}, {nn_name}"

        acc = (f"{_b(r['accuracy_posterior'], best['acc_post'])} / "
               f"{_b(r['accuracy_payoff'], best['acc_pay'])} / "
               f"{_b(r['accuracy_topk'], best['acc_topk'])}")
        prec = (f"{_b(r['precision_posterior'], best['prec_post'])} / "
                f"{_b(r['precision_payoff'], best['prec_pay'])} / "
                f"{_b(r['precision_topk'], best['prec_topk'])}")
        rec = (f"{_b(r['recall_posterior'], best['rec_post'])} / "
               f"{_b(r['recall_payoff'], best['rec_pay'])} / "
               f"{_b(r['recall_topk'], best['rec_topk'])}")
        pay = _payoff_fmt(r["realized_payoff"], best["payoff"])

        ratio_post = _ratio(r["network_signal_accuracy_posterior"],
                            r["non_network_signal_accuracy_posterior"])
        ratio_pay = _ratio(r["network_signal_accuracy_payoff"],
                           r["non_network_signal_accuracy_payoff"])
        ratio_topk = _ratio(r["network_signal_accuracy_topk"],
                            r["non_network_signal_accuracy_topk"])
        ind_acc = f"{ratio_post} / {ratio_pay} / {ratio_topk}"

        rows.append(
            f"{signals_col} & {acc} & {prec} & {rec} & {ind_acc} & {pay} \\\\"
        )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        (r"{Observed signals} & {Accuracy$\uparrow$} & "
         r"{Precision$\uparrow$} & {Recall$\uparrow$} & "
         r"{1D Accuracy ratio} & "
         r"{Realized$\downarrow$} \\"),
        (r"(net., non-net.) & post.\ / pay.\ / topk & post.\ / pay.\ / topk "
         r"& post.\ / pay.\ / topk "
         r"& post.\ / pay.\ / topk & payoff \\"),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{1D Accuracy ratio shows the ratio of network to "
        r"non-network individual signal accuracy; values $>1$ (bold) "
        r"indicate the network signal is more accurate alone.}",
        r"\end{table}",
    ]
    return "\n".join(lines)

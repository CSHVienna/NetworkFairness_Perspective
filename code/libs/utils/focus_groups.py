"""
Focus group analysis: load config, plot minimum-requirement answers, ranking distribution.
Used by notebooks/focus_groups.ipynb.
"""
from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


# --- Config and data loading -------------------------------------------------

def load_focus_config(config_path: str | Path = "config/focus/focus.yaml") -> dict[str, Any]:
    """Load focus group metadata from YAML. Path relative to current working directory."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_focus_data(
    config_path: str | Path = "config/focus/focus.yaml",
    data_dir: str | Path = "data",
) -> dict[int, dict[str, Any]]:
    """
    Load all focus group DataFrames and metadata from config.
    Returns dict[group_id, {"df", "fn", "n", "where", "when", "n_participants", "url"}].
    Paths are relative to current working directory (e.g. notebook directory).
    """
    config = load_focus_config(config_path)
    data_root = Path(data_dir)
    data = {}
    for group in config["focus_groups"]:
        gid = group["id"]
        fn = data_root / group["filename"]
        df = pd.read_excel(fn, sheet_name="Voters", index_col="Voter", skiprows=[0, 1])
        data[gid] = {
            "fn": str(fn),
            "n": df.shape[0],
            "df": df,
            "where": group["where"],
            "when": group["when"],
            "n_participants": group["n_participants"],
            "url": group["url"],
        }
    return data


# --- Minimum threshold columns (shared) --------------------------------------

def _normalize_meritocracy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Meritocracy -> Skills in column names."""
    out = df.copy()
    for c in [c for c in out.columns if ": Merit" in c]:
        out = out.rename(columns={c: c.replace("Meritocracy", "Skills")})
    return out


def _get_minimum_columns_pairs(df: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """
    Detect minimum-requirement question columns and pair (role -> [Skills, Social capital]).
    Returns (minimum_columns, minimum_columns_pairs) filtered to PhD, Postdoc, Professor.
    """
    minimum_columns = [
        c for c in df.columns
        if c.startswith("What do you think are the minimum")
        or c.startswith("What do you think is the minimal")
    ]
    pairs: dict[str, list[str]] = {}
    for c in minimum_columns:
        keyword = (
            ": (min) " if ": (min) " in c
            else ": Ski" if ": Ski" in c
            else ": Soci" if ": Soci" in c
            else ": (minimum) "
        )
        parts = c.split(keyword)
        if "is the minimal" in c:
            if len(parts) > 1 and parts[1] == "lls":
                parts[1] = "Skills"
            if len(parts) > 1 and parts[1] == "al capital":
                parts[1] = "Social capital"
        key = parts[0].split(": ")[-1]
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(parts[1] if len(parts) > 1 else "")
    # Keep only PhD, Postdoc, Professor
    pairs = {k: v for k, v in pairs.items() if "PhD" in k or "Postdoc" in k or "Professor" in k}
    return minimum_columns, pairs


def _preprocess_minimum_values(
    df: pd.DataFrame,
    minimum_columns: list[str],
    minimum_columns_pairs: dict[str, list[str]],
) -> None:
    """Scale values that are <=10 (0-10 scale) to 0-100 in place."""
    for key, vals in minimum_columns_pairs.items():
        cols = [c for val in vals for c in minimum_columns if key in c and val in c]
        if len(cols) >= 2:
            df.loc[:, cols[0]] = df.loc[:, cols[0]].apply(lambda x: x * 10 if x <= 10 else x)
            df.loc[:, cols[1]] = df.loc[:, cols[1]].apply(lambda x: x * 10 if x <= 10 else x)


def _prepare_minimum_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Normalize columns, get pairs, preprocess. Returns (df, minimum_columns, minimum_columns_pairs)."""
    df = _normalize_meritocracy_columns(df)
    minimum_columns, minimum_columns_pairs = _get_minimum_columns_pairs(df)
    _preprocess_minimum_values(df, minimum_columns, minimum_columns_pairs)
    return df, minimum_columns, minimum_columns_pairs


def _mean_text_y(cols: list[str]) -> float:
    return 0.40 if any("Professor" in c for c in cols) else 0.95


# --- Plot: single group minimum thresholds -----------------------------------

def plot_minimum_threshold_answers(
    df: pd.DataFrame,
    aggregate: bool = False,
) -> None:
    """Plot minimum skills/social-capital requirements for one focus group."""
    df, minimum_columns, minimum_columns_pairs = _prepare_minimum_df(df)
    colors = cycle(sns.color_palette("tab10", n_colors=len(minimum_columns_pairs)))
    mu = r"$\mu$"

    if aggregate:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2), sharex=True, sharey=True)
        for key, vals in minimum_columns_pairs.items():
            cols = [c for val in vals for c in minimum_columns if key in c and val in c]
            if len(cols) < 2:
                continue
            mean_x = df[cols[0]].mean()
            mean_y = df[cols[1]].mean()
            std_x = df[cols[0]].std()
            std_y = df[cols[1]].std()
            color = next(colors)
            sns.scatterplot(x=[mean_x], y=[mean_y], s=100, ax=ax, color=color, label=key)
            ax.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, fmt="none", capsize=6, ecolor=color)
        ax.text(0.03, 0.24, f"n: {df.shape[0]}", color="gray", fontsize=8,
                verticalalignment="top", horizontalalignment="left", transform=ax.transAxes)
        ax.set_xlabel(vals[0])
        ax.set_ylabel(vals[1])
    else:
        fig, axes = plt.subplots(
            1, len(minimum_columns_pairs), figsize=(10, 2), sharex=True, sharey=True
        )
        axes = axes if hasattr(axes, "__len__") else [axes]
        for ax, (key, vals) in zip(axes, minimum_columns_pairs.items()):
            cols = [c for val in vals for c in minimum_columns if key in c and val in c]
            if len(cols) < 2:
                continue
            sns.scatterplot(data=df, x=cols[0], y=cols[1], ax=ax, color=next(colors))
            mean_x = df[cols[0]].mean()
            mean_y = df[cols[1]].mean()
            ax.axvline(mean_x, color="gray", linestyle="--")
            ax.axhline(mean_y, color="gray", linestyle="--")
            y = _mean_text_y(cols)
            ax.text(0.03, y, f"{mu} {vals[0]}: {mean_x:.2f}", color="gray", fontsize=8,
                    verticalalignment="top", horizontalalignment="left", transform=ax.transAxes)
            ax.text(0.03, y - 0.12, f"{mu} {vals[1]}: {mean_y:.2f}", color="gray", fontsize=8,
                    verticalalignment="top", horizontalalignment="left", transform=ax.transAxes)
            ax.text(0.03, y - 0.24, f"n: {df.shape[0]}", color="gray", fontsize=8,
                    verticalalignment="top", horizontalalignment="left", transform=ax.transAxes)
            ax.set_title(key)
            ax.set_xlabel(vals[0])
            ax.set_ylabel(vals[1])

    smooth = 5
    plt.xlim(0 - smooth, 100 + smooth)
    plt.ylim(0 - smooth, 100 + smooth)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    plt.close()


# --- Plot: all groups minimum thresholds -------------------------------------

def plot_minimum_threshold_answers_all(
    data: dict[int, dict[str, Any]],
    aggregate: bool = False,
    fn: str | Path | None = None,
    row_labels: bool | list[str] = False,
    row_label_kwargs: dict[str, Any] | None = None,
    row_label_offset: tuple[float, float] = (-48, 0),
) -> None:
    """
    Plot minimum requirements for all focus groups (grid: one row per group).

    ``row_labels``: if True, label each row "A)", "B)", ... at the top left of
    its first panel; if a list of strings, use those labels instead.
    ``row_label_kwargs`` are passed to ``ax.annotate`` (e.g. fontsize, fontweight).
    ``row_label_offset`` is the (x, y) offset in points from the top-left corner
    of the row's first panel; the default places the label left of the y-axis label.
    """
    nrows = len(data)
    if row_labels is True:
        labels = [f"{chr(ord('A') + i)})" for i in range(nrows)]
    elif row_labels:
        labels = list(row_labels)
    else:
        labels = None
    label_kw = dict(fontsize=12)
    label_kw.update(row_label_kwargs or {})

    def _ylabel(text: str) -> str:
        # Break the y-axis label onto two lines when row labels are shown
        if labels is None or not text:
            return text
        words = text.split()
        if len(words) < 2:
            return text
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    height = 2.0
    fig = None

    for fg_id, obj in data.items():
        row_id = fg_id - 1
        df, minimum_columns, minimum_columns_pairs = _prepare_minimum_df(obj["df"].copy())
        colors = cycle(sns.color_palette("tab10", n_colors=len(minimum_columns_pairs)))
        n_cols = 1 if aggregate else len(minimum_columns_pairs)

        if fig is None:
            fig, axes = plt.subplots(
                nrows, n_cols, figsize=(10, nrows * height), sharex=True, sharey=True
            )
            if nrows == 1 and n_cols == 1:
                axes = [[axes]]
            elif nrows == 1:
                axes = [axes]
            elif n_cols == 1:
                axes = [[ax] for ax in axes]

        mu = r"$\mu$"

        if aggregate:
            ax = axes[row_id][0]
            for key, vals in minimum_columns_pairs.items():
                cols = [c for val in vals for c in minimum_columns if key in c and val in c]
                if len(cols) < 2:
                    continue
                mean_x = df[cols[0]].mean()
                mean_y = df[cols[1]].mean()
                std_x = df[cols[0]].std()
                std_y = df[cols[1]].std()
                color = next(colors)
                sns.scatterplot(x=[mean_x], y=[mean_y], s=100, ax=ax, color=color, label=key)
                ax.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, fmt="none", capsize=6, ecolor=color)
            ax.text(0.03, 0.24, f"n: {df.shape[0]}", color="gray", fontsize=10,
                    verticalalignment="top", horizontalalignment="left", transform=ax.transAxes)
            ax.set_xlabel(vals[0] if row_id == nrows - 1 else "")
            ax.set_ylabel(_ylabel(vals[1]))
        else:
            for col_id, (key, vals) in enumerate(minimum_columns_pairs.items()):
                ax = axes[row_id][col_id]
                cols = [c for val in vals for c in minimum_columns if key in c and val in c]
                if len(cols) < 2:
                    continue
                sns.scatterplot(data=df, x=cols[0], y=cols[1], ax=ax, color=next(colors))
                mean_x = df[cols[0]].mean()
                mean_y = df[cols[1]].mean()
                ax.axvline(mean_x, color="gray", linestyle="--")
                ax.axhline(mean_y, color="gray", linestyle="--")
                y = _mean_text_y(cols)
                ax.text(0.03, y, f"{mu} {vals[0]}: {mean_x:.2f}", color="gray", fontsize=10,
                        verticalalignment="top", horizontalalignment="left", transform=ax.transAxes)
                ax.text(0.03, y - 0.12, f"{mu} {vals[1]}: {mean_y:.2f}", color="gray", fontsize=10,
                        verticalalignment="top", horizontalalignment="left", transform=ax.transAxes)
                ax.text(0.03, y - 0.24, f"n: {df.shape[0]}", color="gray", fontsize=10,
                        verticalalignment="top", horizontalalignment="left", transform=ax.transAxes)
                ax.set_title(key if row_id == 0 else "")
                ax.set_xlabel(vals[0] if row_id == nrows - 1 else "")
                ax.set_ylabel(_ylabel(vals[1]) if col_id == 0 else "")

        if labels is not None and row_id < len(labels):
            axes[row_id][0].annotate(
                labels[row_id], xy=(0, 1), xycoords="axes fraction",
                xytext=row_label_offset, textcoords="offset points",
                ha="right", va="top", annotation_clip=False, **label_kw,
            )

    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if fn is not None:
        plt.savefig(fn, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()


# --- Ranking task ------------------------------------------------------------

RANKING_QUESTION_PATTERN = (
    "You need to hire one of these candidates. In order of preference, "
    "how would you rank them? (from best to worst):"
)


# def get_ranking_stats(
#     data: dict[int, dict[str, Any]],
#     group_id: int,
#     pattern: str | None = None,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Compute rank counts and rank percentages for one focus group.
#     Returns (rank_counts, rank_percent).
#     """
#     pattern = pattern or RANKING_QUESTION_PATTERN
#     cols = [c for c in data[group_id]["df"].columns if c.startswith(pattern)]
#     tmp = data[group_id]["df"][cols].copy()
#     tmp = tmp.rename(columns={c: c.replace(pattern, "").strip() for c in tmp.columns})
#     tmp = tmp[["P 1", "P 2", "P 3", "P 4"]]
#     tmp = tmp.rename(columns={"P 1": "C1", "P 2": "C2", "P 3": "C3", "P 4": "C4"})
#     rank_counts = tmp.apply(pd.Series.value_counts).fillna(0)
#     rank_percent = rank_counts / rank_counts.sum()
#     return rank_counts, rank_percent


def get_ranking_stats(
    data: dict[int, dict[str, Any]],
    group_id: int,
    pattern: str | None = None,
    complete_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rank counts and rank percentages for one focus group.

    A response (one row / voter) is *complete* when it has a rank for every
    candidate column (C1..C4). If ``complete_only`` is True, incomplete
    responses are dropped before counting; otherwise every non-missing
    answer is counted (a voter who ranked only some candidates still
    contributes those ranks).

    Returns (rank_counts, rank_percent).
    """
    pattern = pattern or RANKING_QUESTION_PATTERN
    cols = [c for c in data[group_id]["df"].columns if c.startswith(pattern)]
    tmp = data[group_id]["df"][cols].copy()
    tmp = tmp.rename(columns={c: c.replace(pattern, "").strip() for c in tmp.columns})
    tmp = tmp[["P 1", "P 2", "P 3", "P 4"]]
    tmp = tmp.rename(columns={"P 1": "C1", "P 2": "C2", "P 3": "C3", "P 4": "C4"})
    if complete_only:
        tmp = tmp.dropna(axis=0, how="any")
    rank_counts = tmp.apply(pd.Series.value_counts).fillna(0)
    rank_percent = rank_counts / rank_counts.sum()
    return rank_counts, rank_percent


def plot_ranking_distribution(rank_percent: pd.DataFrame) -> None:
    """Plot bar chart of ranking distribution per candidate."""
    rank_percent.T.plot(kind="bar", cmap="Blues_r")
    plt.ylabel("Proportion of votes")
    plt.title("Ranking distribution per candidate")
    plt.xticks(rotation=0)
    plt.show()

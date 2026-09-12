"""Plotting utilities for model diagnostics and dashboards.

- ``set_paper_style``:           set global look and feel (seaborn context/style, figsize, grid).
- ``reset_style``:               reset to matplotlib defaults.
- ``plot_signal_distributions``: overlaid good/bad PDFs or PMFs; optional log–log via ``aesthetics['loglog']``.
- ``plot_posterior_histogram``:  histogram of P(good | all signals) from simulation.
- ``plot_accuracy_bars``:        per-signal and combined accuracy bar chart.
- ``plot_dashboard``:            4-panel figure combining the above.
- ``COLOR_GOOD`` / ``COLOR_BAD``: semantic colors for good/bad series (override module-level constants to retheme).
- ``plot_posterior_1d``: P(good | x) vs a single signal value.
- ``plot_posterior_vs_signal``: P(good|x) and P(bad|x) vs signal; optional log-x via ``aesthetics['logx']``.
- ``plot_posterior_heatmap_network_nonnetwork``: heatmap P(good | both); optional ``logx`` / ``logy`` in aesthetics.
- ``plot_reference_weighted_breakdown`` / ``plot_reference_signal_breakdown``: four-panel figure.
- ``plot_reference_breakdown_*`` / ``plot_reference_signal_breakdown_*``: each panel on a chosen ``Axes`` (trust, score, single_letter, aggregate).
- ``plot_prior``: bar chart of P(good) and P(bad).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

from libs.distributions.base import ContinuousDistribution, DiscreteDistribution
from libs.distributions import name_for_distribution
from libs.distributions.discrete import ReferenceWeightedSumDist

if TYPE_CHECKING:
    from libs.inference.model import BayesianDecisionModel

# ---------------------------------------------------------------------------
# Central style config (used by all plot_* when ax is None)
# ---------------------------------------------------------------------------

_DEFAULT_FIGSIZE = (6, 3.5)
_DEFAULT_CELLSIZE = (6, 3.5)
_DEFAULT_GRID = False
_DEFAULT_GRID_ALPHA = None

_STYLE_OVERRIDES: dict[str, Any] = {}

MM  = 1 / 25.4          # mm → inches
COL = 89  * MM          # single Nature column
DBL = 183 * MM          # double Nature column
PANEL_A_PDF_RASTER_DPI = 1200

# Semantic colors for good vs. bad states (muted seafoam green / burnt orange).
COLOR_GOOD = "#016837" #"#006837" #"#6AAF6A"
COLOR_BAD = "#c01a27" #"#A70526" #"#D4735A"
COLOR_MID = "#FFFBD4" #"#C8A840" 
COLORS = [COLOR_GOOD,
            '#83AD61',
            '#9FAC58',
            '#B6AA4F',
            COLOR_MID,
            '#CC9C47',
            '#CF8F4E',
            '#D28254',
            COLOR_BAD]

def _get_style(key: str, default: Any) -> Any:
    """Return style value from overrides or default."""
    return _STYLE_OVERRIDES.get(key, default)


def set_paper_style(
    *,
    style: str = "ticks",
    context: str = "paper",
    font_scale: float = 1.2,
    font_family: str | list[str] | None = "Helvetica",
    grid: bool = False,
    grid_alpha: float = None,
    figsize: tuple[float, float] = (6, 3.5),
    rc: dict[str, Any] | None = None,
) -> None:
    """Set global look and feel for all plots in this module.

    Uses seaborn's ``set_style`` and ``set_context``, and stores overrides
    (figsize, grid, grid_alpha) that ``plot_*`` functions use when creating
    figures.

    Parameters
    ----------
    style : str
        Seaborn style: ``"whitegrid"``, ``"darkgrid"``, ``"white"``, ``"dark"``, ``"ticks"``.
    context : str
        Seaborn context: ``"paper"`` (smaller), ``"notebook"``, ``"talk"``, ``"poster"``.
    font_scale : float
        Scale factor for font sizes in the chosen context.
    font_family : str or list of str, optional
        Font family for text. Use a generic name (e.g. ``"serif"``, ``"sans-serif"``,
        ``"monospace"``) or a specific font name (e.g. ``"Times New Roman"``). To prefer
        several fonts in order, pass a list (e.g. ``["Helvetica", "Arial"]``); the first
        available is used.
    grid : bool
        Whether to draw grid in plot_* axes.
    grid_alpha : float
        Grid line alpha (0–1).
    figsize : tuple
        Default figure size (width, height) for single-panel plots.
    rc : dict, optional
        Extra matplotlib rcParams to set (e.g. ``{"axes.titlesize": 12}``).
    """
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    if font_family is not None:
        plt.rcParams["font.family"] = font_family
    _STYLE_OVERRIDES["figsize"] = figsize
    _STYLE_OVERRIDES["grid"] = grid
    _STYLE_OVERRIDES["grid_alpha"] = grid_alpha
    if rc:
        plt.rcParams.update(rc)


def reset_style() -> None:
    """Reset plotting style and context to matplotlib defaults.

    Clears seaborn style/context and any overrides set by ``set_paper_style``.
    """
    sns.reset_orig()
    plt.rcdefaults()
    _STYLE_OVERRIDES.clear()


# ---------------------------------------------------------------------------
# Plot functions (read from _STYLE_OVERRIDES when creating figures)
# ---------------------------------------------------------------------------


def _apply_labels(ax: Any, opts: dict[str, Any]) -> None:
    """Apply title, axis labels, x-tick visibility, and legend from *opts*.

    **Text** (only applied when the matching ``show_*`` is True; default True):
    ``title``, ``xlabel``, ``ylabel``. Use ``None`` for empty text.

    **Flags**
        ``show_title``, ``show_xlabel``, ``show_ylabel``, ``show_legend`` — default True.
        ``show_xticks`` — default **True**. Set to ``False`` to remove x tick marks
        and tick labels (e.g. for inset or composite figures).
    """
    if opts.get("show_title", True):
        t = opts.get("title", "")
        ax.set_title("" if t is None else t)
    else:
        ax.set_title("")

    if opts.get("show_xlabel", True):
        xl = opts.get("xlabel", "")
        ax.set_xlabel("" if xl is None else xl)
    else:
        ax.set_xlabel("")

    if opts.get("show_ylabel", True):
        yl = opts.get("ylabel", "")
        ax.set_ylabel("" if yl is None else yl)
    else:
        ax.set_ylabel("")

    if not opts.get("show_xticks", True):
        ax.set_xticks([])

    if not opts.get("show_yticks", True):
        ax.set_yticks([])

    if not opts.get("show_ytickslabels", True):
        ax.set_yticklabels([])

    if not opts.get("show_xtickslabels", True):
        ax.set_xticklabels([])

    if opts.get("show_legend", True):
        legend_kw: dict[str, Any] = {}
        if "legend_title" in opts:
            legend_kw["title"] = opts["legend_title"]
        if "legend_loc" in opts:
            legend_kw["loc"] = opts["legend_loc"]
        if "legend_fontsize" in opts:
            legend_kw["fontsize"] = opts["legend_fontsize"]
        ax.legend(**legend_kw)
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


def save_fig(
    fig: plt.Figure,
    fn: str | Path,
    bbox_inches: str = "tight",
    dpi: int = 600,
    pad_inches: float | None = None,
    show: bool = True,
) -> None:
    fn = str(fn)
    has_embeds = fn.endswith(".pdf") and getattr(fig, "_pdf_embeds", None)

    save_kw: dict[str, Any] = {"dpi": dpi, "bbox_inches": bbox_inches}
    if pad_inches is not None:
        save_kw["pad_inches"] = pad_inches

    if has_embeds:
        # Capture tight bbox in inches BEFORE savefig (which may mutate fig.dpi).
        # This lets us analytically map figure-fraction coords into the
        # tight-cropped PDF's coordinate space for the embed overlay.
        fig.canvas.draw()
        screen_dpi = fig.dpi
        renderer = fig.canvas.get_renderer()
        tight = fig.get_tightbbox(renderer)
        fig._embed_mapping = {
            "tight_x0_in": tight.x0 / screen_dpi,
            "tight_y0_in": tight.y0 / screen_dpi,
            "fig_w_in": fig.get_size_inches()[0],
            "fig_h_in": fig.get_size_inches()[1],
            "pad_in": save_kw.get("pad_inches", 0.1),
        }

    fig.savefig(fn, **save_kw)

    if has_embeds:
        _composite_pdf_embeds(fig, fn)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved figure to {fn}")


def create_panel_figure(
    panel_layout: list | list[list],
    *,
    figsize: tuple[float, float] | None = None,
    width_ratios: list[float] | list[list[float] | None] | None = None,
    height_ratios: list[float] | None = None,
    panel_labels: list[str] | bool = True,
    wspace: float = 0.35,
    hspace: float = 0.4,
    inner_wspace: float = 0.1,
    inner_hspace: float = 0.3,
    inner_sharey: bool = False,
    inner_sharex: bool = False,
    label_fontsize: float = 14,
    label_fontweight: str = "bold",
    label_offset: tuple[float, float] = (0.0, 1.05),
) -> tuple[plt.Figure, list[np.ndarray]]:
    """Create a figure composed of labeled panels, each containing its own sub-grid of axes.

    Supports single-row and multi-row layouts.

    Parameters
    ----------
    panel_layout : list
        **Single row** — flat list of ``int`` or ``(nrows, ncols)`` tuples::

            [4, 1]          # panel a: 4 axes, panel b: 1 axis
            [(2, 2), 1]     # panel a: 2×2 sub-grid, panel b: 1 axis

        **Multi-row** — list of lists (one list per row)::

            [[5], [4, 1]]   # row 1: one panel (5 axes); row 2: two panels (4, 1)

    figsize : (width, height), optional
        Figure size in inches.  Auto-computed from the layout when ``None``.
    width_ratios : list of float *or* list of (list | None), optional
        **Single row**: ``[0.8, 0.2]`` — relative panel widths.
        **Multi-row**: one entry per row, each a list of floats or ``None`` for auto::

            [None, [0.8, 0.2]]   # auto for row 1, 80/20 for row 2

        Defaults to proportional to the number of columns in each panel.
    height_ratios : list of float, optional
        Relative heights per row (multi-row only).  Defaults to proportional to the
        tallest sub-grid in each row.
    panel_labels : list of str or bool
        ``True`` → auto-label ``a)``, ``b)``, … across all rows;
        ``False`` / ``None`` → no labels; or pass explicit strings.
    wspace : float
        Horizontal space between panels within a row.
    row_hspace : float
        Vertical space between rows (multi-row only).
    inner_wspace / inner_hspace : float
        Spacing between axes *inside* a panel.
    inner_sharey / inner_sharex : bool
        Whether axes within each panel share their y / x axis.
    label_fontsize, label_fontweight : float, str
        Font properties for panel labels.
    label_offset : (x, y)
        Label position in axes-fraction coords relative to the panel's first axis.

    Returns
    -------
    fig : Figure
    panels : list of np.ndarray
        Flat list across all rows.  ``panels[i]`` is a numpy array of ``Axes`` for
        panel *i*.  For a single-axis panel, ``panels[i][0]`` gives the axis.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # --- Detect single-row vs multi-row ---------------------------------
    is_multi = len(panel_layout) > 0 and all(isinstance(row, list) for row in panel_layout)
    if not is_multi:
        rows_raw: list[list] = [panel_layout]
    else:
        rows_raw = panel_layout

    # --- Normalize each row's entries to (nrows, ncols) tuples ----------
    rows: list[list[tuple[int, int]]] = []
    for row in rows_raw:
        rows.append([
            (1, p) if isinstance(p, int) else (p[0], p[1])
            for p in row
        ])

    n_rows = len(rows)

    # --- Width ratios per row -------------------------------------------
    if width_ratios is None:
        wr_per_row: list[list[float]] = [
            [float(nc) for _, nc in row] for row in rows
        ]
    elif is_multi:
        wr_per_row = [
            wr if wr is not None else [float(nc) for _, nc in rows[i]]
            for i, wr in enumerate(width_ratios)
        ]
    else:
        wr_per_row = [width_ratios]  # type: ignore[list-item]

    # --- Height ratios --------------------------------------------------
    if height_ratios is None:
        height_ratios = [float(max(nr for nr, _ in row)) for row in rows]

    # --- Default figsize ------------------------------------------------
    if figsize is None:
        max_cols = max(sum(nc for _, nc in row) for row in rows)
        total_rows = sum(max(nr for nr, _ in row) for row in rows)
        figsize = (
            3.0 * max_cols + 0.6,
            3.0 * total_rows + 0.4 * (n_rows - 1),
        )

    fig = plt.figure(figsize=figsize)

    # --- Outer grid: one slot per row -----------------------------------
    outer = GridSpec(
        n_rows, 1, figure=fig,
        height_ratios=height_ratios,
        hspace=hspace,
    )

    # --- Panel labels ---------------------------------------------------
    total_panels = sum(len(row) for row in rows)
    if panel_labels is True:
        labels: list[str | None] = [chr(ord("a") + i) for i in range(total_panels)]
    elif not panel_labels:
        labels = [None] * total_panels
    else:
        labels = list(panel_labels) + [None] * max(0, total_panels - len(panel_labels))  # type: ignore[arg-type]

    # --- Build panels ---------------------------------------------------
    panels: list[np.ndarray] = []
    label_idx = 0

    for row_idx, row in enumerate(rows):
        row_gs = GridSpecFromSubplotSpec(
            1, len(row),
            subplot_spec=outer[row_idx],
            wspace=wspace,
            width_ratios=wr_per_row[row_idx],
        )

        for panel_idx, (nr, nc) in enumerate(row):
            inner = GridSpecFromSubplotSpec(
                nr, nc, subplot_spec=row_gs[panel_idx],
                wspace=inner_wspace, hspace=inner_hspace,
            )
            axes: list[plt.Axes] = []
            ref_y: plt.Axes | None = None
            ref_x: plt.Axes | None = None
            for r in range(nr):
                for c in range(nc):
                    kw: dict[str, Any] = {}
                    if inner_sharey and ref_y is not None:
                        kw["sharey"] = ref_y
                    if inner_sharex and ref_x is not None:
                        kw["sharex"] = ref_x
                    ax = fig.add_subplot(inner[r, c], **kw)
                    if ref_y is None:
                        ref_y = ax
                    if ref_x is None:
                        ref_x = ax
                    if inner_sharey and c > 0:
                        ax.tick_params(labelleft=False)
                    if inner_sharex and r < nr - 1:
                        ax.tick_params(labelbottom=False)
                    axes.append(ax)

            arr = np.asarray(axes, dtype=object)
            if labels[label_idx] is not None:
                lx, ly = label_offset
                axes[0].text(
                    lx, ly, str(labels[label_idx]),
                    transform=axes[0].transAxes,
                    fontsize=label_fontsize,
                    fontweight=label_fontweight,
                    va="bottom", ha="left",
                )
            panels.append(arr)
            label_idx += 1

    return fig, panels


def create_figure(nrows: int, ncols: int, figsize: tuple[float, float] = _DEFAULT_FIGSIZE, **kwargs) -> tuple[plt.Figure, np.ndarray[plt.Axes, Any]]:
    """Create a figure with a grid of subplots."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


# ---------------------------------------------------------------------------
# Vector PDF embedding
# ---------------------------------------------------------------------------

def embed_pdf_in_panel(
    fig: plt.Figure,
    panel_axes: np.ndarray,
    pdf_path: str,
    *,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
) -> plt.Axes:
    """Replace a panel's sub-axes with a blank placeholder for a vector PDF.

    The source PDF is composited as **vector** graphics when the figure is
    saved with :func:`save_fig`.  No rasterisation occurs.

    Parameters
    ----------
    fig : Figure
        The parent figure (returned by ``create_panel_figure``).
    panel_axes : ndarray of Axes
        The axes array for the panel, e.g. ``panels[0]``.
    pdf_path : str
        Path to the PDF file to embed.
    scale : float
        Uniform scale factor applied to the embedded PDF (default 1.0).
        Values > 1 enlarge, < 1 shrink.  The PDF is scaled about its centre.
    dx, dy : float
        Horizontal / vertical offset in **figure-fraction** units.
        Positive *dx* shifts right, positive *dy* shifts up.

    Returns
    -------
    ax : Axes
        The blank placeholder axis (invisible, used for layout only).
    """
    bboxes = [ax.get_position() for ax in panel_axes.flat]
    x0 = min(b.x0 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)

    for ax in panel_axes.flat:
        texts = [c for c in ax.get_children()
                 if isinstance(c, plt.Text) and c.get_text()]
        for t in texts:
            t.set_figure(fig)
            fig.texts.append(t)
        ax.remove()

    ax_new = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    # Do NOT use axis("off") — that excludes the axes from the tight bbox,
    # causing bbox_inches='tight' to crop the placeholder region entirely.
    # Instead hide every visual element while keeping the axes "on".
    ax_new.set_xticks([])
    ax_new.set_yticks([])
    for spine in ax_new.spines.values():
        spine.set_visible(False)
    ax_new.patch.set_alpha(0)

    if not hasattr(fig, "_pdf_embeds"):
        fig._pdf_embeds = []
    fig._pdf_embeds.append({
        "pdf_path": pdf_path,
        "bbox": (x0, y0, x1, y1),
        "scale": scale,
        "dx": dx,
        "dy": dy,
    })

    return ax_new


def _composite_pdf_embeds(fig: plt.Figure, output_path: str) -> None:
    """Post-process a saved PDF to overlay vector PDF embeds.

    Called automatically by :func:`save_fig` when the figure carries embeds
    registered by :func:`embed_pdf_in_panel`.  Requires PyMuPDF.

    Uses an analytical mapping from figure-fraction coordinates into the
    (possibly tight-cropped) output PDF's point space, based on the tight
    bounding box captured before saving.
    """
    embeds = getattr(fig, "_pdf_embeds", None)
    if not embeds:
        return

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "Vector PDF embedding requires PyMuPDF: pip install pymupdf"
        ) from exc

    import tempfile, shutil

    doc = fitz.open(output_path)
    page = doc[0]
    page_w = page.rect.width
    page_h = page.rect.height

    m = getattr(fig, "_embed_mapping", None)
    if m is not None:
        tx0 = m["tight_x0_in"]
        ty0 = m["tight_y0_in"]
        fw = m["fig_w_in"]
        fh = m["fig_h_in"]
        pad = m["pad_in"]

        def _fig_frac_to_pdf_pt(fx: float, fy: float) -> tuple[float, float]:
            x_pt = (fx * fw - tx0 + pad) * 72
            y_pt = page_h - (fy * fh - ty0 + pad) * 72
            return x_pt, y_pt
    else:
        def _fig_frac_to_pdf_pt(fx: float, fy: float) -> tuple[float, float]:
            return fx * page_w, (1 - fy) * page_h

    for embed in embeds:
        src = fitz.open(embed["pdf_path"])
        x0, y0, x1, y1 = embed["bbox"]
        scale = embed.get("scale", 1.0)
        dx_frac = embed.get("dx", 0.0)
        dy_frac = embed.get("dy", 0.0)

        left, top = _fig_frac_to_pdf_pt(x0, y1)   # top-left
        right, bot = _fig_frac_to_pdf_pt(x1, y0)   # bottom-right
        panel_w = right - left

        src_page = src[0]
        src_aspect = src_page.rect.height / src_page.rect.width

        target_w = panel_w * scale
        target_h = target_w * src_aspect
        dx_pt = dx_frac * page_w
        dy_pt = dy_frac * page_h
        h_center = (left + right) / 2 + dx_pt
        v_center = (top + bot) / 2 - dy_pt

        rect = fitz.Rect(
            h_center - target_w / 2,
            v_center - target_h / 2,
            h_center + target_w / 2,
            v_center + target_h / 2,
        )
        page.show_pdf_page(rect, src, 0)
        src.close()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    doc.save(tmp_path)
    doc.close()
    shutil.move(tmp_path, output_path)


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    """Vectorized numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    zn = z[~pos]
    e = np.exp(zn)
    out[~pos] = e / (1.0 + e)
    return out


def _continuous_log_scale_from_aesthetics(
    d_good: ContinuousDistribution,
    d_bad: ContinuousDistribution,
    aesthetics: dict[str, Any],
    key: str,
) -> bool:
    """Use log scale only when *aesthetics[key]* is ``True`` and support allows it (positive ppf)."""
    if aesthetics.get(key) is not True:
        return False
    try:
        lo = min(
            float(d_good._frozen.ppf(1e-4)),
            float(d_bad._frozen.ppf(1e-4)),
        )
        return lo > 0.0
    except Exception:
        return False


def _continuous_should_logx(
    d_good: ContinuousDistribution,
    d_bad: ContinuousDistribution,
    aesthetics: dict[str, Any],
) -> bool:
    return _continuous_log_scale_from_aesthetics(d_good, d_bad, aesthetics, "logx")


def _continuous_should_logy(
    d_good: ContinuousDistribution,
    d_bad: ContinuousDistribution,
    aesthetics: dict[str, Any],
) -> bool:
    return _continuous_log_scale_from_aesthetics(d_good, d_bad, aesthetics, "logy")


def _continuous_should_loglog(
    d_good: ContinuousDistribution,
    d_bad: ContinuousDistribution,
    aesthetics: dict[str, Any],
) -> bool:
    return _continuous_log_scale_from_aesthetics(d_good, d_bad, aesthetics, "loglog")


def _loglog_pdf_x_grid(
    d_good: ContinuousDistribution,
    d_bad: ContinuousDistribution,
    *,
    n_points: int,
    observed_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Geometric grid in x spanning both distributions (via scipy ppf)."""
    q_lo, q_hi = 1e-4, 1.0 - 1e-4
    g1 = float(d_good._frozen.ppf(q_lo))
    g2 = float(d_good._frozen.ppf(q_hi))
    b1 = float(d_bad._frozen.ppf(q_lo))
    b2 = float(d_bad._frozen.ppf(q_hi))
    x_min = max(min(g1, b1), 1e-30)
    x_max = max(g2, b2)
    if observed_range is not None:
        x_min = min(x_min, max(observed_range[0], 1e-30))
        x_max = max(x_max, observed_range[1])
    if not np.isfinite(x_max) or x_max <= x_min * 1.001:
        x_max = x_min * 1e6
    return np.geomspace(x_min, x_max, int(n_points))


def plot_prior(
    prior_good_or_model: float | Any,
    ax=None,
    *,
    aesthetics: dict[str, Any] | None = None,
):
    """Bar chart of the binary prior: P(good) and P(bad).

    Parameters
    ----------
    prior_good_or_model
        A float in ``(0, 1)``, or any object with a ``prior_good`` attribute
        (e.g. ``BayesianDecisionModel``).
    aesthetics : dict, optional
        ``figsize``, ``title``, ``xlabel``, ``ylabel``,
        ``show_title``, ``show_xlabel``, ``show_ylabel``, ``show_xticks``, ``show_legend``
        (see ``_apply_labels``).
    """
    if hasattr(prior_good_or_model, "prior_good"):
        pg = float(prior_good_or_model.prior_good)
    else:
        pg = float(prior_good_or_model)
    if not 0 < pg < 1:
        raise ValueError("prior_good must lie in (0, 1)")

    ae = aesthetics or {}
    if ax is None:
        figsize = ae.get("figsize", (3.2, 2.8))
        _, ax = plt.subplots(figsize=figsize)

    labels = ["good", "bad"]
    ax.bar(
        [0, 1],
        [pg, 1.0 - pg],
        color=[COLOR_GOOD, COLOR_BAD],
        width=0.5,
        edgecolor="white",
        linewidth=0.8,
        label=labels,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    for i, p in enumerate([pg, 1.0 - pg]):
        ax.text(i, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=9)

    defaults = {
        "title": "Prior",
        "xlabel": "state",
        "ylabel": "probability",
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_legend": True,
        'legend_title': 'outcome',
    }
    _apply_labels(ax, {**defaults, **ae})
    show_grid = _get_style("grid", _DEFAULT_GRID)
    grid_alpha = _get_style("grid_alpha", _DEFAULT_GRID_ALPHA)
    if show_grid:
        ax.grid(show_grid, axis="y", alpha=grid_alpha)
    return ax


def _clip_grid_to_signal_bounds(x: np.ndarray, signal: Any) -> np.ndarray:
    """Filter grid *x* to [signal.xmin, signal.xmax] when those bounds are set."""
    sig_xmin = getattr(signal, "xmin", None)
    sig_xmax = getattr(signal, "xmax", None)
    if sig_xmin is not None or sig_xmax is not None:
        mask = np.ones(len(x), dtype=bool)
        if sig_xmin is not None:
            mask &= x >= float(sig_xmin)
        if sig_xmax is not None:
            mask &= x <= float(sig_xmax)
        return x[mask]
    return x


def _x_grid_for_signal(
    signal: Any,
    n_points: int = 300,
    observed_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """1D grid of x values covering both likelihood supports (continuous or discrete).

    When *observed_range* ``(lo, hi)`` is given the grid is widened to cover the
    observed data — but only on axes where the signal has no explicit bound
    (``signal.xmin`` / ``signal.xmax``), so model-defined supports are respected.
    """
    d_good = signal.likelihood_good
    d_bad = signal.likelihood_bad
    low_g, high_g = d_good.support
    low_b, high_b = d_bad.support
    low = min(low_g, low_b)
    high = max(high_g, high_b)
    has_explicit_xmin = getattr(signal, "xmin", None) is not None
    has_explicit_xmax = getattr(signal, "xmax", None) is not None
    if has_explicit_xmin:
        low = float(signal.xmin)
    if has_explicit_xmax:
        high = float(signal.xmax)
    if isinstance(d_good, ReferenceWeightedSumDist) and isinstance(
        d_bad, ReferenceWeightedSumDist
    ):
        kg, _ = d_good.aggregate_mass_points()
        kb, _ = d_bad.aggregate_mass_points()
        return np.sort(np.union1d(kg, kb))
    if isinstance(d_good, ContinuousDistribution):
        if not np.isfinite(low):
            low = min(d_good.mean, d_bad.mean) - 4 * np.sqrt(max(d_good.variance, d_bad.variance))
        if not np.isfinite(high):
            high = max(d_good.mean, d_bad.mean) + 4 * np.sqrt(max(d_good.variance, d_bad.variance))
        if not np.isfinite(low) or not np.isfinite(high):
            q_lo, q_hi = 1e-4, 1.0 - 1e-4
            lo_g = float(d_good._frozen.ppf(q_lo))
            lo_b = float(d_bad._frozen.ppf(q_lo))
            hi_g = float(d_good._frozen.ppf(q_hi))
            hi_b = float(d_bad._frozen.ppf(q_hi))
            if not np.isfinite(low):
                low = min(lo_g, lo_b)
            if not np.isfinite(high):
                high = max(hi_g, hi_b)
        if observed_range is not None:
            if not has_explicit_xmin:
                low = min(low, observed_range[0])
            if not has_explicit_xmax:
                high = max(high, observed_range[1])
        return np.linspace(low, high, n_points)
    if not np.isfinite(high):
        high = min(
            int(max(d_good.mean, d_bad.mean) + 4 * np.sqrt(max(d_good.variance, d_bad.variance))),
            500,
        )
    if observed_range is not None:
        if not has_explicit_xmin and np.isfinite(observed_range[0]):
            low = min(low, observed_range[0])
        if not has_explicit_xmax and np.isfinite(observed_range[1]):
            high = max(high, observed_range[1])
    # Do not clip at 0: discrete signals can be negative (e.g. reference_weighted_sum).
    low_i = int(np.floor(low)) if np.isfinite(low) else 0
    high_i = int(high) if np.isfinite(high) else 500
    return np.arange(low_i, high_i + 1, dtype=float)


def _x_grid_posterior_vs_signal(
    model: Any,
    signal_name: str,
    n_points: int,
    aesthetics: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    """Return (x_grid, use_log_x) for posterior-vs-signal plots."""
    sig = model.signals[signal_name]
    d_good = sig.likelihood_good
    d_bad = sig.likelihood_bad
    if isinstance(d_good, ContinuousDistribution) and _continuous_should_logx(
        d_good, d_bad, aesthetics
    ):
        x = _loglog_pdf_x_grid(d_good, d_bad, n_points=n_points)
        x = _clip_grid_to_signal_bounds(x, sig)
        return x, True
    return _x_grid_for_signal(sig, n_points=n_points), False


def _posterior_p_good_single_signal(model: Any, signal_name: str, x: np.ndarray) -> np.ndarray:
    """P(good | x) for one signal, vectorized over *x*."""
    sig = model.signals[signal_name]
    log_prior_odds = np.log(model.prior_good) - np.log(1.0 - model.prior_good)
    llr = sig.log_likelihood_ratio(np.asarray(x, dtype=float))
    return _sigmoid_np(log_prior_odds + llr)


def _good_bad_colormap() -> LinearSegmentedColormap:
    """Colormap: posterior 0 → bad color, 1 → good color."""
    # return LinearSegmentedColormap.from_list(
    #     "posterior_good_bad", [COLOR_BAD, COLOR_MID, COLOR_GOOD], N=44
    # )
    # return LinearSegmentedColormap.from_list(
    #     "posterior_good_bad", COLORS[::-1], N=256
    # )
    # return plt.get_cmap("BrBG")
    return plt.get_cmap("RdYlGn")

def _validate_reference_weighted_pair(
    dist_good: ReferenceWeightedSumDist,
    dist_bad: ReferenceWeightedSumDist,
) -> None:
    if dist_good.n_refs != dist_bad.n_refs:
        raise ValueError("dist_good and dist_bad must have the same n_refs")
    if dist_good.trust_weights.shape != dist_bad.trust_weights.shape or np.any(
        dist_good.trust_weights != dist_bad.trust_weights
    ):
        raise ValueError("dist_good and dist_bad must use the same trust_weights")
    if dist_good.scores.shape != dist_bad.scores.shape or np.any(dist_good.scores != dist_bad.scores):
        raise ValueError("dist_good and dist_bad must use the same scores")


def _maybe_grid_reference_breakdown_ax(ax: Any, *, apply_grid: bool | None) -> None:
    use = _get_style("grid", _DEFAULT_GRID) if apply_grid is None else apply_grid
    if use:
        ax.grid(use, alpha=_get_style("grid_alpha", _DEFAULT_GRID_ALPHA))


def _annotate_dprime(
    ax: Any,
    d_prime_value: float,
    aesthetics: dict[str, Any],
) -> None:
    """Place a d′ text annotation on *ax*.

    Reads ``dprime_pos`` from *aesthetics* (default ``(0.98, 0.97)``).
    """
    dp_text = f"d\u2032 = {d_prime_value:.2f}"
    dp_pos = aesthetics.get("dprime_pos", (0.98, 0.97))
    ax.text(
        dp_pos[0], dp_pos[1], dp_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#444444",
    )


def _dprime_from_dists(dist_good, dist_bad) -> float:
    """Compute d′ = |μ_g − μ_b| / √(0.5·(σ²_g + σ²_b))."""
    pooled_std = np.sqrt(0.5 * (dist_good.variance + dist_bad.variance))
    if pooled_std == 0:
        return float("inf")
    return float(abs(dist_good.mean - dist_bad.mean) / pooled_std)


def _dprime_from_discrete(values: np.ndarray, probs_good: np.ndarray, probs_bad: np.ndarray) -> float:
    """Compute d′ from discrete probability vectors over shared support values."""
    mu_g = float(np.dot(probs_good, values))
    mu_b = float(np.dot(probs_bad, values))
    var_g = float(np.dot(probs_good, (values - mu_g) ** 2))
    var_b = float(np.dot(probs_bad, (values - mu_b) ** 2))
    pooled_std = np.sqrt(0.5 * (var_g + var_b))
    if pooled_std == 0:
        return float("inf")
    return float(abs(mu_g - mu_b) / pooled_std)


def plot_reference_breakdown_trust(
    ax: Any,
    dist_good: ReferenceWeightedSumDist,
    dist_bad: ReferenceWeightedSumDist,
    *,
    aesthetics: dict[str, Any] | None = None,
) -> Any:
    """Panel 1: :math:`P(t \\mid \\mathrm{outcome})` over trust weights (grouped bars).

    aesthetics : dict, optional
        ``bar_width`` (default 0.36), ``apply_grid`` (default from global style),
        plus keys for ``_apply_labels``: ``title``, ``xlabel``, ``ylabel``,
        ``show_title``, ``show_xlabel``, ``show_ylabel``, ``show_xticks``, ``show_legend``.
    """
    ae = aesthetics or {}
    _validate_reference_weighted_pair(dist_good, dist_bad)
    order = np.argsort(dist_good.trust_weights)
    trust_vals = np.asarray(dist_good.trust_weights)[order]
    probs_g = np.asarray(dist_good.trust_probs)[order]
    probs_b = np.asarray(dist_bad.trust_probs)[order]
    n_t = len(trust_vals)
    w = float(ae.get("bar_width", 0.36))
    x_trust = np.arange(n_t)
    labels_trust = [str(t) for t in trust_vals]
    ax.bar(
        x_trust - w / 2,
        probs_g,
        width=w,
        color=COLOR_GOOD,
        label="good",
        alpha=0.9,
    )
    ax.bar(
        x_trust + w / 2,
        probs_b,
        width=w,
        color=COLOR_BAD,
        label="bad",
        alpha=0.9,
    )
    if ae.get("show_xticks", True):
        ax.set_xticks(x_trust)
        ax.set_xticklabels(labels_trust, rotation=0, ha="right")
    defaults = {
        "title": r"Trust weight: $P(t \mid \mathrm{outcome})$",
        "xlabel": "trust weight",
        "ylabel": "probability",
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_xticks": True,
        "show_legend": True,
    }
    _apply_labels(ax, {**defaults, **ae})
    _maybe_grid_reference_breakdown_ax(ax, apply_grid=ae.get("apply_grid"))

    if ae.get("show_dprime", True):
        _annotate_dprime(ax, _dprime_from_discrete(trust_vals, probs_g, probs_b), ae)

    return ax


def plot_reference_breakdown_score(
    ax: Any,
    dist_good: ReferenceWeightedSumDist,
    dist_bad: ReferenceWeightedSumDist,
    *,
    aesthetics: dict[str, Any] | None = None,
) -> Any:
    """Panel 2: :math:`P(\\mathrm{score} \\mid \\mathrm{outcome})` (grouped bars).

    aesthetics
        Same pattern as ``plot_reference_breakdown_trust`` (``bar_width``, ``apply_grid``, label keys).
    """
    ae = aesthetics or {}
    _validate_reference_weighted_pair(dist_good, dist_bad)
    order = np.argsort(dist_good.scores)
    score_vals = np.asarray(dist_good.scores)[order]
    probs_g = np.asarray(dist_good.score_probs)[order]
    probs_b = np.asarray(dist_bad.score_probs)[order]
    n_s = len(score_vals)
    w = float(ae.get("bar_width", 0.36))
    x_score = np.arange(n_s)

    ax.bar(
        x_score - w / 2,
        probs_g,
        width=w,
        color=COLOR_GOOD,
        label="good",
        alpha=0.9,
    )
    ax.bar(
        x_score + w / 2,
        probs_b,
        width=w,
        color=COLOR_BAD,
        label="bad",
        alpha=0.9,
    )
    if ae.get("show_xticks", True):
        ax.set_xticks(x_score)
        ax.set_xticklabels([str(int(s)) if s == int(s) else str(s) for s in score_vals])
    defaults = {
        "title": r"Reference score: $P(\mathrm{score} \mid \mathrm{outcome})$",
        "xlabel": "reference score",
        "ylabel": "probability",
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_xticks": True,
        "show_legend": True,
    }
    _apply_labels(ax, {**defaults, **ae})
    _maybe_grid_reference_breakdown_ax(ax, apply_grid=ae.get("apply_grid"))

    if ae.get("show_dprime", True):
        _annotate_dprime(ax, _dprime_from_discrete(score_vals, probs_g, probs_b), ae)

    return ax


def plot_reference_breakdown_single_letter(
    ax: Any,
    dist_good: ReferenceWeightedSumDist,
    dist_bad: ReferenceWeightedSumDist,
    *,
    aesthetics: dict[str, Any] | None = None,
) -> Any:
    """Panel 3: one-letter product :math:`\\mathrm{trust}\\times\\mathrm{score}` PMF (step curves).

    aesthetics
        ``apply_grid`` and ``_apply_labels`` keys (no ``bar_width``).
    """
    ae = aesthetics or {}
    _validate_reference_weighted_pair(dist_good, dist_bad)
    one_g = dist_good.single_reference_pmf()
    one_b = dist_bad.single_reference_pmf()
    keys = sorted(set(one_g) | set(one_b))
    xw = np.array(keys, dtype=float)
    yg = np.array([one_g.get(k, 0.0) for k in keys])
    yb = np.array([one_b.get(k, 0.0) for k in keys])
    mu_g = float(np.sum(xw * yg))
    mu_b = float(np.sum(xw * yb))
    order = np.argsort(xw)
    xw, yg, yb = xw[order], yg[order], yb[order]
    ax.fill_between(xw, yg, alpha=0.45, color=COLOR_GOOD, step="mid")
    ax.fill_between(xw, yb, alpha=0.45, color=COLOR_BAD, step="mid")
    ax.plot(
        xw,
        yg,
        color=COLOR_GOOD,
        drawstyle="steps-mid",
        linewidth=1.2,
        label=f"good (μ={mu_g:.2f})",
    )
    ax.plot(
        xw,
        yb,
        color=COLOR_BAD,
        drawstyle="steps-mid",
        linewidth=1.2,
        label=f"bad (μ={mu_b:.2f})",
    )
    defaults = {
        "title": r"Single letter: $P(\mathrm{trust}\times\mathrm{score} \mid \mathrm{outcome})$",
        "xlabel": r"score $\times$ trust",
        "ylabel": "probability",
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_xticks": True,
        "show_legend": True,
    }
    _apply_labels(ax, {**defaults, **ae})
    _maybe_grid_reference_breakdown_ax(ax, apply_grid=ae.get("apply_grid"))
    return ax


def plot_reference_breakdown_aggregate(
    ax: Any,
    dist_good: ReferenceWeightedSumDist,
    dist_bad: ReferenceWeightedSumDist,
    *,
    aesthetics: dict[str, Any] | None = None,
) -> Any:
    """Panel 4: sum over *n_refs* i.i.d. letters (aggregate PMF, step curves).

    aesthetics
        ``apply_grid`` and ``_apply_labels`` keys. Default ``title`` includes ``n_refs``.
        ``dprime_pos: (x, y)`` — axes-fraction coordinates for the d′
        annotation (default ``(0.98, 0.97)``).
    """
    ae = aesthetics or {}

    label_legend = {'good':"good (μ={dist_good.mean:.2f})", 'bad':"bad (μ={dist_bad.mean:.2f})"}
    if ae.pop('simple_legend', False):
        label_legend = {'good':"good", 'bad':"bad"}

    _validate_reference_weighted_pair(dist_good, dist_bad)
    n_refs = dist_good.n_refs
    kg, _pg = dist_good.aggregate_mass_points()
    kb, _pb = dist_bad.aggregate_mass_points()
    xs = np.sort(np.union1d(kg, kb))
    ag = dist_good.pmf(xs)
    ab = dist_bad.pmf(xs)
    ax.fill_between(xs, ag, alpha=0.45, color=COLOR_GOOD, step="mid")
    ax.fill_between(xs, ab, alpha=0.45, color=COLOR_BAD, step="mid")
    ax.plot(
        xs,
        ag,
        color=COLOR_GOOD,
        drawstyle="steps-mid",
        linewidth=1.2,
        label=label_legend['good'],
    )
    ax.plot(
        xs,
        ab,
        color=COLOR_BAD,
        drawstyle="steps-mid",
        linewidth=1.2,
        label=label_legend['bad'],
    )
    default_title = (
        rf"Aggregate ({n_refs} letters): "
        r"$P(\sum \mathrm{trust}\times\mathrm{score} \mid \mathrm{outcome})$"
    )
    defaults = {
        "title": default_title,
        "xlabel": "aggregate score",
        "ylabel": "probability",
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_xticks": True,
        "show_legend": True,
    }
    _apply_labels(ax, {**defaults, **ae})
    _maybe_grid_reference_breakdown_ax(ax, apply_grid=ae.get("apply_grid"))
    _annotate_dprime(ax, _dprime_from_dists(dist_good, dist_bad), ae)

    return ax


def plot_reference_weighted_breakdown(
    dist_good: ReferenceWeightedSumDist,
    dist_bad: ReferenceWeightedSumDist,
    *,
    figsize: tuple[float, float] | None = None,
    aesthetics: dict[str, Any] | None = None,
) -> plt.Figure:
    """Four-panel decomposition matching reference × trust network signal.

    Same as calling the four ``plot_reference_breakdown_*`` functions on a 1×4 grid.

    Parameters
    ----------
    dist_good, dist_bad
        ``ReferenceWeightedSumDist`` instances (same ``n_refs``), e.g. from a
        ``Signal``'s ``likelihood_good`` / ``likelihood_bad``.
    aesthetics : dict, optional
        Passed to each panel (``show_title``, ``show_legend``, ``bar_width``, …).
        Avoid ``title`` / ``xlabel`` / ``ylabel`` here unless you want the same text on
        all four axes; use single-panel functions for custom labels per plot.
    """
    _validate_reference_weighted_pair(dist_good, dist_bad)
    ae0 = aesthetics or {}
    figsize = figsize or ae0.get("figsize") or (14.0, 3.2)
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    ae = {**ae0, "apply_grid": False}
    plot_reference_breakdown_trust(axes[0], dist_good, dist_bad, aesthetics=ae)
    plot_reference_breakdown_score(axes[1], dist_good, dist_bad, aesthetics=ae)
    plot_reference_breakdown_single_letter(axes[2], dist_good, dist_bad, aesthetics=ae)
    plot_reference_breakdown_aggregate(axes[3], dist_good, dist_bad, aesthetics=ae)
    show_grid = _get_style("grid", _DEFAULT_GRID)
    grid_alpha = _get_style("grid_alpha", _DEFAULT_GRID_ALPHA)
    for ax in axes:
        if show_grid:
            ax.grid(show_grid, alpha=grid_alpha)
    return fig


def _signal_to_ref_weighted(signal: Any) -> tuple[ReferenceWeightedSumDist, ReferenceWeightedSumDist]:
    dg, db = signal.likelihood_good, signal.likelihood_bad
    if not isinstance(dg, ReferenceWeightedSumDist) or not isinstance(db, ReferenceWeightedSumDist):
        raise TypeError(
            "signal must use ReferenceWeightedSumDist for both likelihood_good and likelihood_bad"
        )
    return dg, db


def plot_reference_signal_breakdown(
    signal,
    *,
    figsize: tuple[float, float] | None = None,
    aesthetics: dict[str, Any] | None = None,
) -> plt.Figure:
    """Same as ``plot_reference_weighted_breakdown`` but takes a ``Signal``."""
    dg, db = _signal_to_ref_weighted(signal)
    return plot_reference_weighted_breakdown(dg, db, figsize=figsize, aesthetics=aesthetics)


def plot_reference_signal_breakdown_trust(
    signal, ax: Any, *, aesthetics: dict[str, Any] | None = None
) -> Any:
    """Trust-weight panel for a reference ``Signal``; see ``plot_reference_breakdown_trust``."""
    dg, db = _signal_to_ref_weighted(signal)
    return plot_reference_breakdown_trust(ax, dg, db, aesthetics=aesthetics)


def plot_reference_signal_breakdown_score(
    signal, ax: Any, *, aesthetics: dict[str, Any] | None = None
) -> Any:
    """Score panel; see ``plot_reference_breakdown_score``."""
    dg, db = _signal_to_ref_weighted(signal)
    return plot_reference_breakdown_score(ax, dg, db, aesthetics=aesthetics)


def plot_reference_signal_breakdown_single_letter(
    signal, ax: Any, *, aesthetics: dict[str, Any] | None = None
) -> Any:
    """Single-letter PMF panel; see ``plot_reference_breakdown_single_letter``."""
    dg, db = _signal_to_ref_weighted(signal)
    return plot_reference_breakdown_single_letter(ax, dg, db, aesthetics=aesthetics)


def plot_reference_signal_breakdown_aggregate(
    signal, ax: Any, *, aesthetics: dict[str, Any] | None = None
) -> Any:
    """Aggregate PMF panel; see ``plot_reference_breakdown_aggregate``."""
    dg, db = _signal_to_ref_weighted(signal)
    return plot_reference_breakdown_aggregate(ax, dg, db, aesthetics=aesthetics)


def plot_signal_distributions(signal, ax=None, *, aesthetics: dict[str, Any] | None = None):
    """Plot overlaid good/bad PDFs or PMFs for a single Signal.

    aesthetics : dict, optional
        figsize, title, xlabel, ylabel, show_*, n_points (grid size; default 300).
        ``loglog: True`` — log x and log y with a geometric *x* grid (continuous
        signals only; requires strictly positive support). Default is linear axes.
        ``model_details_pos: (x, y)`` — axes-fraction coordinates for the
        distribution-model annotation (default ``(0.08, 0.97)``).
        ``dprime_pos: (x, y)`` — axes-fraction coordinates for the d′
        annotation.  When omitted the label is placed just below the legend
        (or at ``(0.98, 0.97)`` if there is no legend).

    The signal's ``xmin``/``xmax`` attributes (set via the YAML config) override
    the automatic grid bounds when present.
    """
    ae = aesthetics or {}
    if ax is None:
        figsize = ae.get("figsize", _get_style("figsize", _DEFAULT_FIGSIZE))
        _, ax = plt.subplots(figsize=figsize)

    d_good = signal.likelihood_good
    d_bad = signal.likelihood_bad
    low_g, high_g = d_good.support
    low_b, high_b = d_bad.support
    low = min(low_g, low_b)
    high = max(high_g, high_b)
    n_points = int(ae.get("n_points", 300))

    if isinstance(d_good, ContinuousDistribution):
        use_logx = _continuous_should_loglog(d_good, d_bad, ae) or _continuous_should_logx(d_good, d_bad, ae)
        use_logy = _continuous_should_loglog(d_good, d_bad, ae) or ae.get("logy", False)

        if use_logx:
            x = _loglog_pdf_x_grid(d_good, d_bad, n_points=n_points)
        else:
            if not np.isfinite(low):
                low = min(d_good.mean, d_bad.mean) - 4 * np.sqrt(
                    max(d_good.variance, d_bad.variance)
                )
            if not np.isfinite(high):
                high = max(d_good.mean, d_bad.mean) + 4 * np.sqrt(
                    max(d_good.variance, d_bad.variance)
                )
            if not np.isfinite(low) or not np.isfinite(high):
                q_lo, q_hi = 1e-4, 1.0 - 1e-4
                if not np.isfinite(low):
                    low = min(float(d_good._frozen.ppf(q_lo)), float(d_bad._frozen.ppf(q_lo)))
                if not np.isfinite(high):
                    high = max(float(d_good._frozen.ppf(q_hi)), float(d_bad._frozen.ppf(q_hi)))
            if getattr(signal, "xmin", None) is not None:
                low = float(signal.xmin)
            if getattr(signal, "xmax", None) is not None:
                high = float(signal.xmax)
            x = np.linspace(low, high, n_points)

        x = _clip_grid_to_signal_bounds(x, signal)

        y_g = np.asarray(d_good.pdf_or_pmf(x), dtype=float)
        y_b = np.asarray(d_bad.pdf_or_pmf(x), dtype=float)
        if use_logy:
            y_g = np.maximum(y_g, 1e-300)
            y_b = np.maximum(y_b, 1e-300)

        ax.plot(x, y_g, color=COLOR_GOOD, label="P(x | good)", linewidth=2)
        ax.plot(x, y_b, color=COLOR_BAD, label="P(x | bad)", linewidth=2)
        if use_logx:
            ax.set_xscale("log")
        if use_logy:
            ax.set_yscale("log")
    elif isinstance(d_good, ReferenceWeightedSumDist) and isinstance(
        d_bad, ReferenceWeightedSumDist
    ):
        x = np.sort(np.union1d(*[d.aggregate_mass_points()[0] for d in (d_good, d_bad)]))
        y_g = d_good.pdf_or_pmf(x)
        y_b = d_bad.pdf_or_pmf(x)
        if len(x) > 1:
            w = float(np.clip(0.4 * np.min(np.diff(x)), 0.02, 0.5))
        else:
            w = 0.35
        ax.bar(x - w / 2, y_g, width=w, color=COLOR_GOOD, label="P(x | good)", alpha=0.9)
        ax.bar(x + w / 2, y_b, width=w, color=COLOR_BAD, label="P(x | bad)", alpha=0.9)
    else:
        # Discrete: integer grid over finite support
        if not np.isfinite(high):
            high = min(
                int(max(d_good.mean, d_bad.mean) + 4 * np.sqrt(max(d_good.variance, d_bad.variance))),
                500,
            )
        low = int(np.floor(low)) if np.isfinite(low) else 0
        high = int(high) if np.isfinite(high) else 500
        x = np.arange(low, high + 1, dtype=float)
        y_g = d_good.pdf_or_pmf(x)
        y_b = d_bad.pdf_or_pmf(x)
        w = 0.35
        ax.bar(x - w / 2, y_g, width=w, color=COLOR_GOOD, label="P(x | good)", alpha=0.9)
        ax.bar(x + w / 2, y_b, width=w, color=COLOR_BAD, label="P(x | bad)", alpha=0.9)

    default_ylabel = "density" if isinstance(d_good, ContinuousDistribution) else "probability"
    defaults = {
        "title": f"Signal: {signal.name}",
        "xlabel": signal.name,
        "ylabel": default_ylabel,
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_legend": True,
    }
    _apply_labels(ax, {**defaults, **ae})

    # -- annotation: distribution model + params (non-network signals only) --
    is_network = isinstance(d_good, ReferenceWeightedSumDist)
    if not is_network:
        dist_name = name_for_distribution(d_good)
        if dist_name is not None:
            def _fmt_params(d):

                def _rename(k):
                    if k in ['xmin','xmax']:
                        _k = k[1:]
                        return "$x_{" + _k + "}$"
                    if k in ['alpha', 'sigma', 'mu', 'beta', 'lambda']:
                        return f"$\\{k}$"
                    if k == 'lam':
                        return "$\\lambda$"
                    return k

                return ", ".join(
                    f"{_rename(k)}={v:.3g}" if isinstance(v, float) else f"{_rename(k)}={v}"
                    for k, v in d.params.items()
                )
            label = (f"{dist_name}\n"
                     f"  good: {_fmt_params(d_good)}\n"
                     f"  bad:  {_fmt_params(d_bad)}")
            mdp = ae.get("model_details_pos", (0.08, 0.97))
            ax.text(
                mdp[0], mdp[1], label,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, family="monospace", color="#555555",
            )

    # -- annotation: d-prime (always shown) ---------------------------------
    _annotate_dprime(ax, signal.d_prime(), ae)
    show_grid = _get_style("grid", _DEFAULT_GRID)
    grid_alpha = _get_style("grid_alpha", _DEFAULT_GRID_ALPHA)
    if show_grid:
        log_any = ax.get_xscale() == "log" or ax.get_yscale() == "log"
        ax.grid(show_grid, alpha=grid_alpha, which="both" if log_any else "major")
    return ax


def plot_posterior_1d(
    model: "BayesianDecisionModel",
    signal_name: str,
    ax=None,
    *,
    n_points: int = 300,
    aesthetics: dict[str, Any] | None = None,
):
    """Plot P(good | x) as a function of a single observed signal *x*.

    Parameters
    ----------
    model : BayesianDecisionModel
    signal_name : str
        Which signal varies on the x-axis; posterior uses only this signal.
    n_points : int
        Grid resolution (continuous signals only; discrete uses integer support).

    aesthetics : dict, optional
        figsize, title, xlabel, ylabel, show_title, show_xlabel, show_xticks, show_ylabel, show_legend.
    """
    if signal_name not in model.signals:
        raise KeyError(f"unknown signal {signal_name!r}; known: {model.signal_names}")
    sig = model.signals[signal_name]
    x = _x_grid_for_signal(sig, n_points=n_points)
    p_good = _posterior_p_good_single_signal(model, signal_name, x)

    if ax is None:
        ae = aesthetics or {}
        figsize = ae.get("figsize", _get_style("figsize", _DEFAULT_FIGSIZE))
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(x, p_good, color=COLOR_GOOD, linewidth=2, label="P(good | x)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_ylim(-0.02, 1.02)
    defaults = {
        "title": f"P(good | {signal_name})",
        "xlabel": signal_name,
        "ylabel": "P(good | x)",
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_legend": True,
    }
    _apply_labels(ax, {**defaults, **(aesthetics or {})})
    show_grid = _get_style("grid", _DEFAULT_GRID)
    grid_alpha = _get_style("grid_alpha", _DEFAULT_GRID_ALPHA)
    if show_grid:
        ax.grid(show_grid, alpha=grid_alpha)
    return ax


def plot_posterior_vs_signal(
    model: "BayesianDecisionModel",
    signal_name: str | None = None,
    ax=None,
    *,
    n_points: int = 300,
    aesthetics: dict[str, Any] | None = None,
):
    """P(good | x) and P(bad | x) = 1 − P(good | x) vs signal value.

    If *signal_name* is given, draws one axes. If ``None``, one subplot per
    signal in a grid (returns ``(fig, axes)``).

    Pass ``logx: True`` in *aesthetics* for a log *x*-axis and geometric grid
    (continuous signals with positive support only); default is linear.

    aesthetics : dict, optional
        ``logx``: set ``True`` to enable. For single panel:
        usual title/xlabel keys. For grid: ``cell_size``, ``subplot_figscale``;
        ``sharey`` (default False): if True, y-axis label is drawn only on the
        first column of the grid (avoids repeated "probability" labels).
    """
    names = [signal_name] if signal_name is not None else list(model.signal_names)
    if not names:
        raise ValueError("model has no signals to plot")
    for n in names:
        if n not in model.signals:
            raise KeyError(f"unknown signal {n!r}")

    ae = aesthetics or {}

    label_legend = {'good':"P(good | x)", 'bad':"P(bad | x)"}
    if ae.pop('simple_legend', False):
        label_legend = {'good':"good", 'bad':"bad"}

    def _draw_one(ax_i: Any, name: str, *, show_ylabel: bool = True, show_title: bool = True) -> None:
        x, log_x = _x_grid_posterior_vs_signal(model, name, n_points, ae)
        p_good = _posterior_p_good_single_signal(model, name, x)
        p_bad = 1.0 - p_good
        ax_i.plot(x, p_good, color=COLOR_GOOD, linewidth=2, label="P(good | x)")
        ax_i.plot(x, p_bad, color=COLOR_BAD, linewidth=2, label="P(bad | x)")
        ax_i.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax_i.set_ylim(-0.02, 1.02)
        if log_x:
            ax_i.set_xscale("log")
        ax_i.set_title(name if show_title else "")
        ax_i.set_xlabel(name)
        ylabel = ae.get("ylabel", "probability")
        ax_i.set_ylabel(ylabel if show_ylabel else "")
        ax_i.legend(loc=ae.get("legend_loc", "best"), fontsize=ae.get("legend_fontsize", 8))
        show_grid = _get_style("grid", _DEFAULT_GRID)
        grid_alpha = _get_style("grid_alpha", _DEFAULT_GRID_ALPHA)
        if show_grid:
            ax_i.grid(
                show_grid,
                alpha=grid_alpha,
                which="both" if log_x else "major",
            )

    figsize = ae.get("figsize", _get_style("figsize", _DEFAULT_FIGSIZE))
    sharey = ae.get("sharey", False)

    if len(names) == 1:
        name = names[0]
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, sharey=sharey)
        x, log_x = _x_grid_posterior_vs_signal(model, name, n_points, ae)
        p_good = _posterior_p_good_single_signal(model, name, x)
        p_bad = 1.0 - p_good
        ax.plot(x, p_good, color=COLOR_GOOD, linewidth=2, label=label_legend['good'])
        ax.plot(x, p_bad, color=COLOR_BAD, linewidth=2, label=label_legend['bad'])
        # ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_ylim(-0.02, 1.02)
        if log_x:
            ax.set_xscale("log")
        defaults = {
            "title": f"P(state | x) — {name}",
            "xlabel": name,
            "ylabel": "probability",
            "show_title": True,
            "show_xlabel": True,
            "show_ylabel": True,
            "show_legend": True,
        }
        _apply_labels(ax, {**defaults, **ae})
        show_grid = _get_style("grid", _DEFAULT_GRID)
        grid_alpha = _get_style("grid_alpha", _DEFAULT_GRID_ALPHA)
        if show_grid:
            ax.grid(show_grid, alpha=grid_alpha, which="both" if log_x else "major")
        return ax

    n = len(names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    scale = float(ae.get("subplot_figscale", 1.2))
    base_w, base_h = ae.get("cell_size", _get_style("cell_size", _DEFAULT_CELLSIZE))
    figsize = (base_w * ncols * scale / 2.5, base_h * nrows * scale / 2.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=sharey)
    flat = axes.ravel()
    for i, name in enumerate(names):
        first_col = i % ncols == 0
        show_yl = (not sharey) or first_col
        show_tl = ae.get('show_title', True)
        _draw_one(flat[i], name, show_ylabel=show_yl, show_title=show_tl)
    for j in range(len(names), len(flat)):
        flat[j].set_visible(False)
    fig.suptitle(ae.get("suptitle", "P(good | x) and P(bad | x) by signal"), y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_posterior_heatmap_network_nonnetwork(
    model: "BayesianDecisionModel",
    ax=None,
    *,
    network_signal_name: str | None = None,
    non_network_signal_name: str | None = None,
    n_x: int = 120,
    n_y: int = 120,
    aesthetics: dict[str, Any] | None = None,
    gaussian_sigma: float | None = None,
    contour_levels: dict[str, Any] | None = None,
    posterior_regions: list[dict[str, Any]] | None = None,
    decision_boundaries: list[dict[str, Any]] | None = None,
    scatter_data: list[dict[str, Any]] | None = None,
    region_boxes: list[dict[str, Any]] | None = None,
):
    """Heatmap of P(good | x_non_net, x_net): x = non-network signal, y = network signal.

    Colors run from *COLOR_BAD* (P ≈ 0) to *COLOR_GOOD* (P ≈ 1).

    Pass ``logx: True`` / ``logy: True`` in *aesthetics* to use log axes and a
    geometric grid on that dimension (continuous signals with positive support).

    If the model has exactly one ``network`` and one ``non_network`` signal, those
    are used automatically. Otherwise pass *network_signal_name* and
    *non_network_signal_name*.

    Parameters
    ----------
    gaussian_sigma : float, optional
        Sigma for 1-D Gaussian smoothing applied along each discrete-distribution
        axis.  Ignored when neither signal is discrete.
    contour_levels : dict, optional
        Contour-line specification.  Required keys: ``start``, ``stop``, ``step``
        (fed to ``np.arange``).  Optional styling keys: ``colors`` (default
        ``"k"``), ``linewidths`` (default 0.6), ``fontsize`` (default 8),
        ``fmt`` (default ``"%.2f"``), ``alpha`` (default 0.7).
    posterior_regions : list of dict, optional
        Each dict must have ``x`` and ``y`` (percentiles 0–1 of the plotted grid
        range) and ``label`` (str).  A marker and annotated box are drawn at each
        point showing *label* and the posterior value.  Optional per-region keys:
        ``fontsize`` (default 8), ``fmt`` (default ``".2f"``).
    decision_boundaries : list of dict, optional
        Per-level contour lines with individual styling.  Each dict must have
        ``level`` (float, the posterior threshold).  Optional keys: ``color``
        (default ``"black"``), ``linewidth`` (default 2.5), ``linestyle``
        (default ``"solid"``), ``label`` (legend text), ``fmt`` (clabel format,
        default ``"p=%.2f"``).
    scatter_data : list of dict, optional
        Overlay scatter points on the heatmap.  Each dict must have ``x`` and
        ``y`` (array-like).  Optional keys: ``color`` (default ``COLOR_GOOD``),
        ``edgecolors`` (default ``"white"``), ``s`` (marker size, default 20),
        ``alpha`` (default 0.5), ``linewidths`` (edge width, default 0.4),
        ``label`` (legend text), ``sample_n`` (int — subsample to at most this
        many points), ``seed`` (int, RNG seed for subsampling, default 0).
    region_boxes : list of dict, optional
        Rectangular regions drawn on the heatmap.  Each dict must have
        ``x_min`` and ``y_min`` (the lower-left corner in data coordinates).
        Optional keys: ``x_max``, ``y_max`` (default to axis upper limits),
        ``color`` (edge colour, default ``"white"``), ``linewidth`` (default 2),
        ``linestyle`` (default ``"--"``), ``facecolor`` (default ``"none"``),
        ``alpha`` (default 0.9), ``label`` (legend text).

    Aesthetics keys
    ---------------
    show_heatmap : bool, default True
        If False, the filled heatmap and colorbar are suppressed; scatter
        overlays, decision boundaries, and contour lines are still drawn.
    """
    net_list = model.get_signals_by_type("network")
    non_list = model.get_signals_by_type("non_network")
    if network_signal_name is None:
        if len(net_list) != 1:
            raise ValueError(
                f"expected exactly one network signal; got {len(net_list)}. "
                "Pass network_signal_name=..."
            )
        s_net = net_list[0]
        network_signal_name = s_net.name
    else:
        s_net = model.signals[network_signal_name]
        if s_net.signal_type != "network":
            raise ValueError(f"{network_signal_name!r} is not a network signal")

    if non_network_signal_name is None:
        if len(non_list) != 1:
            raise ValueError(
                f"expected exactly one non_network signal; got {len(non_list)}. "
                "Pass non_network_signal_name=..."
            )
        s_non = non_list[0]
        non_network_signal_name = s_non.name
    else:
        s_non = model.signals[non_network_signal_name]
        if s_non.signal_type != "non_network":
            raise ValueError(f"{non_network_signal_name!r} is not a non_network signal")

    ae_hm = aesthetics or {}
    d_non_g, d_non_b = s_non.likelihood_good, s_non.likelihood_bad
    d_net_g, d_net_b = s_net.likelihood_good, s_net.likelihood_bad

    x_obs_range: tuple[float, float] | None = None
    y_obs_range: tuple[float, float] | None = None
    if scatter_data is not None:
        all_xs = [np.asarray(sd["x"]).ravel() for sd in scatter_data if "x" in sd]
        all_ys = [np.asarray(sd["y"]).ravel() for sd in scatter_data if "y" in sd]
        if all_xs:
            cat_x = np.concatenate(all_xs)
            if len(cat_x) > 0:
                x_obs_range = (float(np.nanmin(cat_x)), float(np.nanmax(cat_x)))
        if all_ys:
            cat_y = np.concatenate(all_ys)
            if len(cat_y) > 0:
                y_obs_range = (float(np.nanmin(cat_y)), float(np.nanmax(cat_y)))

    if isinstance(d_non_g, ContinuousDistribution) and _continuous_should_logx(
        d_non_g, d_non_b, ae_hm
    ):
        x_non = _clip_grid_to_signal_bounds(
            _loglog_pdf_x_grid(d_non_g, d_non_b, n_points=n_x, observed_range=x_obs_range), s_non
        )
        log_x = True
    else:
        x_non = _x_grid_for_signal(s_non, n_points=n_x, observed_range=x_obs_range)
        log_x = False

    if isinstance(d_net_g, ContinuousDistribution) and _continuous_should_logy(
        d_net_g, d_net_b, ae_hm
    ):
        y_net = _clip_grid_to_signal_bounds(
            _loglog_pdf_x_grid(d_net_g, d_net_b, n_points=n_y, observed_range=y_obs_range), s_net
        )
        log_y = True
    else:
        y_net = _x_grid_for_signal(s_net, n_points=n_y, observed_range=y_obs_range)
        log_y = False

    X_non, Y_net = np.meshgrid(x_non, y_net)

    log_prior_odds = np.log(model.prior_good) - np.log(1.0 - model.prior_good)
    llr_non = s_non.log_likelihood_ratio(X_non)
    llr_net = s_net.log_likelihood_ratio(Y_net)
    p_good = _sigmoid_np(log_prior_odds + llr_non + llr_net)

    # ---- Gaussian smoothing along discrete axes ----
    if gaussian_sigma is not None:
        from scipy.ndimage import gaussian_filter1d

        x_is_discrete = isinstance(d_non_g, DiscreteDistribution)
        y_is_discrete = isinstance(d_net_g, DiscreteDistribution)
        if x_is_discrete:
            p_good = gaussian_filter1d(p_good, sigma=gaussian_sigma, axis=1)
        if y_is_discrete:
            p_good = gaussian_filter1d(p_good, sigma=gaussian_sigma, axis=0)

    if ax is None:
        ae = aesthetics or {}
        figsize = ae.get("figsize", (7, 5.5))
        _, ax = plt.subplots(figsize=figsize)

    if ae_hm.get("show_heatmap", True):
        cmap = _good_bad_colormap()
        n_levels = ae_hm.get("n_color_levels", 256)
        levels = np.linspace(0.0, 1.0, n_levels)
        im = ax.contourf(
            X_non,
            Y_net,
            p_good,
            levels=levels,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        if ae_hm.get("show_cbar", True):
            sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=1.0))
            cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(0.0, 1.20, 0.20))
            cbar_fontsize = ae_hm.get("cbar_fontsize", None)
            cbar_tick_fontsize = ae_hm.get("cbar_tick_fontsize", 9)
            if cbar_tick_fontsize is not None:
                cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
            if ae_hm.get("cbar_label") is not None:
                cbar.set_label(ae_hm["cbar_label"], fontsize=cbar_fontsize)
            elif ae_hm.get("cbar_use_names", False):
                cbar.set_label(f"P(good | {s_non.name}, {s_net.name})", fontsize=cbar_fontsize)
            else:
                cbar.set_label(f"P(good | {s_non}, {s_net})", fontsize=cbar_fontsize)

    # ---- Contour lines ----
    if contour_levels is not None:
        levels = np.arange(
            contour_levels["start"],
            contour_levels["stop"],
            contour_levels["step"],
        )
        cs = ax.contour(
            X_non,
            Y_net,
            p_good,
            levels=levels,
            colors=contour_levels.get("colors", "k"),
            linewidths=contour_levels.get("linewidths", 0.6),
            alpha=contour_levels.get("alpha", 0.7),
        )
        ax.clabel(
            cs,
            inline=True,
            fontsize=contour_levels.get("fontsize", 8),
            fmt=contour_levels.get("fmt", "%.2f"),
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # ---- Posterior region annotations ----
    if posterior_regions is not None:
        x_lo, x_hi = float(x_non.min()), float(x_non.max())
        y_lo, y_hi = float(y_net.min()), float(y_net.max())

        for region in posterior_regions:
            pct_x, pct_y = float(region["x"]), float(region["y"])
            label = region["label"]
            if log_x:
                xv = 10 ** (np.log10(x_lo) + pct_x * (np.log10(x_hi) - np.log10(x_lo)))
            else:
                xv = x_lo + pct_x * (x_hi - x_lo)
            if log_y:
                yv = 10 ** (np.log10(y_lo) + pct_y * (np.log10(y_hi) - np.log10(y_lo)))
            else:
                yv = y_lo + pct_y * (y_hi - y_lo)

            xv_eval = float(x_non[np.argmin(np.abs(x_non - xv))])
            yv_eval = float(y_net[np.argmin(np.abs(y_net - yv))])

            pv = float(
                _sigmoid_np(
                    log_prior_odds
                    + s_non.log_likelihood_ratio(np.array([xv_eval]))
                    + s_net.log_likelihood_ratio(np.array([yv_eval]))
                )[0]
            )

            fmt = region.get("fmt", ".0%")
            pv_text = f"{pv:{fmt}}"
            fontsize = region.get("fontsize", 9)

            ax.text(
                xv, yv,
                f"{label}\n{pv_text}",
                ha="center", va="center",
                fontsize=fontsize,
                fontweight="bold",
                color="white",
                linespacing=1.3,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="none",
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.95,
                ),
                zorder=6,
            )

    # ---- Decision boundary contour lines ----
    from matplotlib.lines import Line2D
    legend_handles: list[Any] = []
    if decision_boundaries is not None:
        for db in decision_boundaries:
            lvl = db["level"]
            if not (0 < lvl < 1):
                continue
            ls = db.get("linestyle", "solid")
            lw = db.get("linewidth", 2.5)
            lc = db.get("color", "black")
            fmt = db.get("fmt", "p=%.2f")
            cs_db = ax.contour(
                X_non, Y_net, p_good,
                levels=[lvl], colors=lc, linewidths=lw, linestyles=ls,
            )
            ax.clabel(cs_db, inline=True, fontsize=9, fmt=fmt)
            lbl = db.get("label")
            if lbl:
                legend_handles.append(
                    Line2D([], [], color=lc, linewidth=lw, linestyle=ls, label=lbl)
                )

    # ---- Scatter overlays ----
    if scatter_data is not None:
        for sd in scatter_data:
            xs, ys = np.asarray(sd["x"]), np.asarray(sd["y"])
            sample_n = sd.get("sample_n")
            if sample_n is not None and len(xs) > sample_n:
                rng = np.random.default_rng(sd.get("seed", 0))
                idx = rng.choice(len(xs), size=sample_n, replace=False)
                xs, ys = xs[idx], ys[idx]
            ax.scatter(
                xs, ys,
                c=sd.get("color", COLOR_GOOD),
                edgecolors=sd.get("edgecolors", "white"),
                s=sd.get("s", 20),
                alpha=sd.get("alpha", 0.5),
                linewidths=sd.get("linewidths", 0.4),
                zorder=5,
                marker=sd.get("marker", None),
                label=sd.get("label"),
            )

    # ---- Region boxes ----
    if region_boxes is not None:
        for rb in region_boxes:
            rx_min = float(rb["x_min"])
            ry_min = float(rb["y_min"])
            ec = rb.get("color", "white")
            lw = rb.get("linewidth", 2)
            ls = rb.get("linestyle", "--")
            alpha = rb.get("alpha", 0.9)

            ax.axvline(rx_min, color='lightgray', linewidth=lw, linestyle=ls, alpha=0.5, zorder=6)
            ax.axhline(ry_min, color='lightgray', linewidth=lw, linestyle=ls, alpha=0.5, zorder=6)

            ax.plot([rx_min, rx_min], [ry_min, y_net.max()],
                    color=ec, linewidth=lw, linestyle=ls, alpha=alpha, zorder=6)
            ax.plot([rx_min, x_non.max()], [ry_min, ry_min],
                    color=ec, linewidth=lw, linestyle=ls, alpha=alpha, zorder=6)

            lbl = rb.get("label")
            if lbl:
                from matplotlib.lines import Line2D as _Line2D
                legend_handles.append(
                    _Line2D([], [], color=ec, linewidth=lw, linestyle=ls, label=lbl)
                )

    has_overlays = bool(legend_handles) or (scatter_data is not None) or (region_boxes is not None)

    defaults = {
        "title": "P(good | non-network & network)",
        "xlabel": non_network_signal_name,
        "ylabel": network_signal_name,
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_legend": has_overlays,
    }
    merged = {**defaults, **ae_hm}
    _apply_labels(ax, merged)

    if has_overlays and merged.get("show_legend", has_overlays):
        existing = ax.get_legend_handles_labels()
        scatter_handles = existing[0] if existing else []
        all_handles = legend_handles + scatter_handles
        if all_handles:
            leg = ax.legend(
                handles=all_handles,
                loc=ae_hm.get("legend_loc", "lower right"),
                fontsize=ae_hm.get("legend_fontsize", 8),
                facecolor="white",
                framealpha=1.0,
            )
            leg.set_zorder(20)

    show_grid = _get_style("grid", _DEFAULT_GRID)
    if show_grid:
        ax.grid(
            show_grid,
            alpha=_get_style("grid_alpha", _DEFAULT_GRID_ALPHA),
            which="both" if (log_x or log_y) else "major",
        )
    return ax


def plot_posterior_histogram(
    posteriors,
    true_states,
    ax=None,
    *,
    aesthetics: dict[str, Any] | None = None,
):
    """Histogram of posterior probabilities coloured by true state.

    Parameters
    ----------
    posteriors : array-like
        P(good | signals) for each candidate, shape (n,).
    true_states : array-like
        True state per candidate: 1 / True / "good" for good, 0 / False / "bad" for bad.

    aesthetics : dict, optional
        Keys: figsize, title, xlabel, ylabel, show_title, show_xlabel, show_xticks, show_ylabel, show_legend.
        Defaults: "Posterior histogram by true state", "P(good | signals)", "density".
    """
    if ax is None:
        ae = aesthetics or {}
        figsize = ae.get("figsize", _get_style("figsize", _DEFAULT_FIGSIZE))
        _, ax = plt.subplots(figsize=figsize)

    posteriors = np.asarray(posteriors, dtype=float).ravel()
    true_states = np.asarray(true_states)

    # Normalize to boolean mask: True = good (1d, same length as posteriors)
    true_flat = np.asarray(true_states).ravel()
    if true_flat.dtype.kind in ("U", "O"):
        is_good = np.array([s == "good" for s in true_flat])
    elif true_flat.dtype == bool:
        is_good = true_flat.copy()
    else:
        is_good = (true_flat == 1) | (true_flat == 1.0)
    if len(is_good) != len(posteriors):
        raise ValueError("posteriors and true_states must have the same length")

    bins = np.linspace(0, 1, 31)
    ax.hist(
        posteriors[is_good],
        bins=bins,
        color=COLOR_GOOD,
        alpha=0.7,
        label="true good",
        density=True,
    )
    ax.hist(
        posteriors[~is_good],
        bins=bins,
        color=COLOR_BAD,
        alpha=0.7,
        label="true bad",
        density=True,
    )
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.8)
    defaults = {
        "title": "Posterior histogram by true state",
        "xlabel": "P(good | signals)",
        "ylabel": "density",
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_legend": True,
    }
    _apply_labels(ax, {**defaults, **(aesthetics or {})})
    show_grid = _get_style("grid", _DEFAULT_GRID)
    grid_alpha = _get_style("grid_alpha", _DEFAULT_GRID_ALPHA)
    ax.grid(show_grid, alpha=grid_alpha)
    return ax


def plot_accuracy_bars(
    accuracies: dict[str, float],
    ax=None,
    *,
    aesthetics: dict[str, Any] | None = None,
):
    """Bar chart of classification accuracy by signal and combined.

    aesthetics : dict, optional
        Keys: figsize, title, xlabel, ylabel, show_title, show_xlabel, show_xticks, show_ylabel.
        Defaults: "Classification accuracy", "", "Accuracy".
    """
    if ax is None:
        ae = aesthetics or {}
        figsize = ae.get("figsize", _get_style("figsize", _DEFAULT_FIGSIZE))
        _, ax = plt.subplots(figsize=figsize)

    labels = list(accuracies.keys())
    values = [accuracies[k] for k in labels]
    x = np.arange(len(labels))
    ax.bar(x, values, color="C0", alpha=0.85, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    defaults = {
        "title": "Classification accuracy",
        "xlabel": "",
        "ylabel": "Accuracy",
        "show_title": True,
        "show_xlabel": True,
        "show_ylabel": True,
        "show_legend": False,
    }
    _apply_labels(ax, {**defaults, **(aesthetics or {})})
    show_grid = _get_style("grid", _DEFAULT_GRID)
    grid_alpha = _get_style("grid_alpha", _DEFAULT_GRID_ALPHA)
    ax.grid(show_grid, axis="y", alpha=grid_alpha)
    return ax


def plot_dashboard(model, simulation_results, title: str | None = None):
    """Full 4-panel dashboard figure for a scenario."""
    raise NotImplementedError

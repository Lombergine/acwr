"""Plots for the quantities this package computes.

Two functions, both returning the axes they drew on so a caller can keep
editing, and neither calling ``show`` or ``savefig``. Matplotlib is an optional
dependency: ``pip install acwr[plot]``.

The design choices are deliberate and worth stating, because the usual plot of
an acute:chronic ratio makes two mistakes. It puts load and ratio on one pair
of axes with two different scales, which invites the reader to compare two
quantities whose relative heights mean nothing, and it shades a "sweet spot"
band behind the line, which asserts the very thresholds this package declines
to ship. :func:`plot_acwr` uses stacked panels with one scale each, and draws a
reference line at 1.0 because that one is arithmetic rather than a claim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .diagnostics import WindowSensitivity
from .ratio import Decay, Method, acwr

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_acwr", "plot_window_sensitivity"]

# Categorical slots and surfaces from a palette validated for colour-vision
# deficiency separation and for contrast against the chart surface.
_INK = "#0b0b0b"
_INK2 = "#52514e"
_GRID = "#e6e5e1"
_RULE = "#9a998f"
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
_BAR = "#86b6ef"
# A single hue, light to dark, which is what a magnitude encoding takes.
_SEQUENTIAL = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
)


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib, which is an optional dependency. "
            "Install it with: pip install 'acwr[plot]'"
        ) from exc
    return plt


def _recede(ax: Axes) -> None:
    """Push the chart furniture behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=_INK2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def plot_acwr(
    load: pd.Series,
    acute: int = 7,
    chronic: int = 28,
    *,
    methods: tuple[Method, ...] = ("rac",),
    decay: Decay = "span",
    axes: tuple[Axes, Axes] | None = None,
    figsize: tuple[float, float] = (11.0, 6.0),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Daily load above, the ratio below, one scale each.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`.
    acute, chronic
        Window lengths, passed to :func:`acwr.acwr`.
    methods
        Which methods to draw. One by default. Up to three may be given, and
        they are drawn in the order passed with a legend.
    decay
        Passed through for the exponentially weighted method.
    axes
        An existing pair of axes to draw on. A new figure is made when this is
        omitted.
    figsize
        Size of the new figure, ignored when ``axes`` is given.

    Returns
    -------
    (matplotlib.figure.Figure, (matplotlib.axes.Axes, matplotlib.axes.Axes))
        The figure and the two axes, load first.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If more than three methods are given, or none.

    Notes
    -----
    Infinite ratios are dropped rather than clipped to the top of the axis,
    because a line touching the top of a chart reads as a large finite value
    and an infinite ratio is a different statement.

    No threshold band is drawn. The horizontal line at 1.0 is there because a
    ratio of 1.0 means this week matched the recent average, which is a fact
    about division rather than a claim about injury.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> load = pd.Series(
    ...     rng.gamma(2.0, 30.0, 120),
    ...     index=pd.date_range("2026-01-01", periods=120, freq="D"),
    ... )
    >>> fig, (top, bottom) = plot_acwr(load)
    >>> bottom.get_ylabel()
    'acute : chronic'
    """
    plt = _require_matplotlib()
    if not methods:
        raise ValueError("give at least one method to draw")
    if len(methods) > 3:
        raise ValueError(
            f"at most three methods can be drawn legibly, got {len(methods)}"
        )

    if axes is None:
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[1.0, 1.4], sharex=True
        )
    else:
        top, bottom = axes
        fig = top.get_figure()

    for ax in (top, bottom):
        _recede(ax)

    # Vertical lines rather than bars. A multi-year log puts more days on the
    # axis than there are pixels, and touching bars of equal colour antialias
    # into darker seams at the shared edges, which reads as structure that is
    # not in the data.
    top.vlines(
        load.index, 0.0, load.to_numpy(), color=_BAR, linewidth=0.9, antialiased=True
    )
    top.set_ylim(bottom=0.0)
    top.set_ylabel("load per day", color=_INK2, fontsize=9.5)
    top.set_title(
        "Daily load", color=_INK, fontsize=11.5, fontweight="semibold", loc="left"
    )

    for method, colour in zip(methods, _SERIES, strict=False):
        ratio = acwr(load, acute=acute, chronic=chronic, method=method, decay=decay)
        assert isinstance(ratio, pd.Series)
        drawn = ratio.replace([np.inf, -np.inf], np.nan)
        bottom.plot(
            drawn.index,
            drawn.to_numpy(),
            color=colour,
            linewidth=1.6,
            solid_capstyle="round",
            label=method,
        )

    bottom.axhline(1.0, color=_RULE, linewidth=1.0, linestyle=(0, (4, 3)))
    bottom.set_ylabel("acute : chronic", color=_INK2, fontsize=9.5)
    bottom.set_title(
        f"Ratio, {acute} against {chronic} days",
        color=_INK,
        fontsize=11.5,
        fontweight="semibold",
        loc="left",
    )
    if len(methods) > 1:
        bottom.legend(
            frameon=False, fontsize=9.5, labelcolor=_INK2, loc="upper left", ncol=3
        )

    return fig, (top, bottom)


def plot_window_sensitivity(
    result: WindowSensitivity,
    *,
    value: str = "agreement",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> tuple[Figure, Axes]:
    """Draw the window sweep as a grid, acute across and chronic down.

    Parameters
    ----------
    result
        A :class:`~acwr.WindowSensitivity` from
        :func:`acwr.window_sensitivity`.
    value
        Which column to shade: ``"agreement"``, the Jaccard overlap with the
        reference pair, or ``"flagged_fraction"``.
    ax
        An existing axes to draw on.
    figsize
        Size of the new figure, ignored when ``ax`` is given.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    KeyError
        If ``value`` names a column the table does not have.

    Notes
    -----
    Magnitude gets one hue running light to dark, never a rainbow, so the
    ordering of the colours matches the ordering of the numbers. Every cell is
    labelled with its value as well as shaded, which is what lets the figure be
    read without relying on colour. Cells with no comparable day are left blank
    rather than shaded at zero.

    The reference pair is outlined, because the figure's point is how far the
    other pairs sit from it.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import acwr
    >>> rng = np.random.default_rng(0)
    >>> load = pd.Series(
    ...     rng.gamma(2.0, 30.0, 400),
    ...     index=pd.date_range("2025-01-01", periods=400, freq="D"),
    ... )
    >>> fig, ax = plot_window_sensitivity(acwr.window_sensitivity(load))
    >>> ax.get_xlabel()
    'acute window (days)'
    """
    plt = _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    table = result.table
    if value not in table.columns:
        raise KeyError(
            f"{value!r} is not a column of the sweep; "
            f"choose from {sorted(table.columns)}"
        )

    grid = table.pivot(index="chronic", columns="acute", values=value)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # `with_extremes` rather than `set_bad`, which matplotlib has marked for
    # deprecation. A package that emits warnings from its own code trains its
    # users to ignore warnings.
    cmap = LinearSegmentedColormap.from_list(
        "acwr_sequential", _SEQUENTIAL
    ).with_extremes(bad="#f5f4f1")
    data = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    mesh = ax.imshow(data, cmap=cmap, aspect="auto", origin="upper")

    ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
    ax.set_xlabel("acute window (days)", color=_INK2, fontsize=9.5)
    ax.set_ylabel("chronic window (days)", color=_INK2, fontsize=9.5)
    ax.tick_params(colors=_INK2, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    label = {
        "agreement": "Agreement with the reference pair (Jaccard)",
        "flagged_fraction": "Share of days flagged",
    }.get(value, value)
    ax.set_title(
        f"{label}\nthreshold {result.threshold:g}, {result.method}, "
        f"reference {result.reference[0]}/{result.reference[1]}",
        color=_INK,
        fontsize=11.5,
        fontweight="semibold",
        loc="left",
    )

    # The label colour is decided by where the cell sits on the colour ramp,
    # not by where it sits in the data. The ramp runs light to dark, so the
    # switch to white ink happens at a fixed fraction of its range; keying off
    # the data mean instead puts dark ink on a dark cell whenever the values
    # happen to be skewed.
    finite = data.compressed()
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    span = (high - low) or 1.0
    for row, chronic in enumerate(grid.index):
        for col, acute in enumerate(grid.columns):
            cell = grid.iat[row, col]
            if not np.isfinite(cell):
                continue
            position = (float(cell) - low) / span
            ax.text(
                col,
                row,
                f"{cell:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if position > 0.45 else _INK,
            )
            if (acute, chronic) == result.reference:
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=_SERIES[1],
                        linewidth=2.4,
                    )
                )

    fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    return fig, ax

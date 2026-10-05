"""Tests for the plotting helpers.

A chart cannot be asserted pixel by pixel, so these check the things that go
wrong in practice: the wrong number of axes, a missing label, a silently
dropped series, an infinite value drawn as a finite one, and the figure
failing to render at all. The last matters most, because a plotting function
that raises on real input is worse than none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import acwr  # noqa: E402
from acwr.plot import plot_acwr, plot_window_sensitivity  # noqa: E402


@pytest.fixture
def load() -> pd.Series:
    rng = np.random.default_rng(0)
    values = rng.gamma(2.0, 30.0, 300)
    values[rng.random(300) < 0.25] = 0.0
    return pd.Series(
        values, index=pd.date_range("2025-01-01", periods=300, freq="D"), name="load"
    )


def test_plot_acwr_returns_two_axes_with_one_scale_each(load):
    """Load and ratio never share a y-axis, which is the usual mistake."""
    fig, (top, bottom) = plot_acwr(load)
    assert top is not bottom
    assert top.get_ylabel() == "load per day"
    assert bottom.get_ylabel() == "acute : chronic"
    fig.canvas.draw()  # the real test: it renders


def test_plot_acwr_draws_one_line_per_method_and_labels_them(load):
    fig, (_, bottom) = plot_acwr(load, methods=("rac", "rau", "ewma"))
    assert len(bottom.get_lines()) == 3 + 1  # three methods plus the rule at 1.0
    assert {line.get_label() for line in bottom.get_lines()} >= {"rac", "rau", "ewma"}
    assert bottom.get_legend() is not None
    fig.canvas.draw()


def test_a_single_method_gets_no_legend_because_the_title_names_it(load):
    _, (_, bottom) = plot_acwr(load)
    assert bottom.get_legend() is None


def test_the_window_lengths_appear_in_the_title(load):
    _, (_, bottom) = plot_acwr(load, acute=5, chronic=35)
    # The titles are left-aligned, so the centre title is empty.
    assert "5 against 35 days" in bottom.get_title(loc="left")


def test_an_infinite_ratio_is_dropped_rather_than_drawn_at_the_top(load):
    """A line touching the top of a chart reads as a large finite value."""
    values = load.copy()
    values.iloc[:40] = 0.0
    _, (_, bottom) = plot_acwr(values, methods=("rau",))
    drawn = bottom.get_lines()[0].get_ydata()
    assert not np.isinf(np.asarray(drawn, dtype=float)).any()


def test_plot_acwr_can_draw_onto_axes_it_was_given(load):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1)
    returned_fig, returned = plot_acwr(load, axes=(axes[0], axes[1]))
    assert returned_fig is fig
    assert returned[0] is axes[0]


@pytest.mark.parametrize("methods", [(), ("rac", "rau", "ewma", "rac")])
def test_an_unusable_number_of_methods_is_rejected(load, methods):
    with pytest.raises(ValueError):
        plot_acwr(load, methods=methods)


def test_plot_window_sensitivity_renders_and_labels_both_axes(load):
    result = acwr.window_sensitivity(load)
    fig, ax = plot_window_sensitivity(result)
    assert ax.get_xlabel() == "acute window (days)"
    assert ax.get_ylabel() == "chronic window (days)"
    fig.canvas.draw()


def test_every_cell_carries_its_own_number_so_colour_is_not_the_only_encoding(load):
    result = acwr.window_sensitivity(load)
    _, ax = plot_window_sensitivity(result)
    drawn = {t.get_text() for t in ax.texts}
    finite = result.table["agreement"].dropna()
    assert len(drawn) >= len(set(finite.round(2)))


def test_a_cell_with_no_comparable_day_is_left_blank(load):
    short = load.iloc[:41]
    result = acwr.window_sensitivity(short)
    assert result.table["agreement"].isna().any()
    _, ax = plot_window_sensitivity(result)
    labelled = len(ax.texts)
    assert labelled == int(result.table["agreement"].notna().sum())


def test_the_other_column_can_be_shaded_instead(load):
    result = acwr.window_sensitivity(load)
    _, ax = plot_window_sensitivity(result, value="flagged_fraction")
    assert "flagged" in ax.get_title(loc="left").lower()


def test_an_unknown_column_is_rejected_by_name(load):
    result = acwr.window_sensitivity(load)
    with pytest.raises(KeyError, match="nonsense"):
        plot_window_sensitivity(result, value="nonsense")


def test_the_helpers_are_reachable_without_importing_matplotlib_eagerly():
    """`import acwr` must not need matplotlib, but `acwr.plot_acwr` must work.

    Matplotlib is an optional dependency, so it is imported on first touch
    through a module-level ``__getattr__`` rather than at import time.
    """
    assert callable(acwr.plot_acwr)
    assert "plot_window_sensitivity" in dir(acwr)
    with pytest.raises(AttributeError):
        _ = acwr.plot_nothing


def test_drawing_emits_no_warnings_of_its_own(load, recwarn):
    """A library that warns from its own code teaches users to ignore warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, _ = plot_acwr(load, methods=("rac", "rau"))
        fig.canvas.draw()
        fig2, _ = plot_window_sensitivity(acwr.window_sensitivity(load))
        fig2.canvas.draw()

    from_acwr = [
        w
        for w in caught
        if "acwr" in str(w.filename) and "test_" not in str(w.filename)
    ]
    assert from_acwr == [], [str(w.message) for w in from_acwr]

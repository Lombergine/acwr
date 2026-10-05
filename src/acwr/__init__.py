"""Acute:chronic workload ratios in Python, with the tools to doubt them.

The package computes the three published forms of the acute:chronic workload
ratio and ships the diagnostics that the metric's critics proposed, so a user
can check on their own data whether the ratio is carrying information or merely
rescaling its own numerator. It also computes the companion quantities from
Foster (1998), training monotony and training strain, and runs any of these
across a squad rather than one athlete.

A short example, from a table of activities to a ratio::

    import pandas as pd
    import acwr

    activities = pd.read_csv("training.csv")       # date, load
    load = acwr.daily_load(activities)             # continuous daily series
    ratio = acwr.acwr(load, method="rac")          # the standard form

And the check that the literature says you should run::

    print(acwr.rescaling_check(load))
    # ACWR vs acute load alone, n=1172: Spearman 0.512 [0.461, 0.559], Pearson 0.390

That line is from three years of one runner's log. A Spearman correlation near
1.0 would mean the ratio is its own numerator in different units; at 0.51 the
denominator is reordering days and therefore carrying something. For why the
number matters, see :mod:`acwr.diagnostics`, the README, and the worked example
in the documentation, every figure of which is reproducible from the page
itself.

Reported figures elsewhere in this package's documentation are either computed
from a seeded series shown alongside them or from the author's own training
log, and a test executes the worked example end to end so that a number in the
prose cannot drift away from the code.
"""

from __future__ import annotations

from .diagnostics import (
    NullComparison,
    RescalingResult,
    ThresholdReport,
    WindowSensitivity,
    compare_to_null,
    null_chronic_acwr,
    rescaling_check,
    threshold_from_history,
    window_sensitivity,
)
from .foster import monotony, srpe_load, strain, weekly_blocks
from .load import daily_load
from .panel import by_athlete
from .progression import block_change, week_to_week_change
from .quality import Finding, LoadQuality, check_load
from .ratio import METHODS, acute_load, acwr, chronic_load

__version__ = "0.1.1"

__all__ = [
    "__version__",
    # building the series
    "daily_load",
    # the ratio
    "acwr",
    "acute_load",
    "chronic_load",
    "METHODS",
    # checking the data before trusting any of it
    "check_load",
    "LoadQuality",
    "Finding",
    # Foster's companion metrics
    "srpe_load",
    "monotony",
    "strain",
    "weekly_blocks",
    # how much has load changed week on week
    "week_to_week_change",
    "block_change",
    # a squad rather than one athlete
    "by_athlete",
    # checking it
    "rescaling_check",
    "null_chronic_acwr",
    "compare_to_null",
    "window_sensitivity",
    "threshold_from_history",
    "RescalingResult",
    "NullComparison",
    "WindowSensitivity",
    "ThresholdReport",
    # drawing it, which needs matplotlib
    "plot_acwr",
    "plot_window_sensitivity",
]

# The plotting helpers are reachable as `acwr.plot_acwr` but are not imported
# eagerly, because matplotlib is an optional dependency and importing the
# package must not require it. PEP 562 module __getattr__ defers the import to
# the moment one of them is first touched.
_LAZY = {"plot_acwr": "acwr.plot", "plot_window_sensitivity": "acwr.plot"}


def __getattr__(name: str) -> object:
    """Import the optional plotting helpers on first use."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    """Include the lazily imported names in tab completion."""
    return sorted(__all__)

"""Week-to-week load change, the other number applied practice quotes.

Before the acute:chronic ratio became the standard, and alongside it since,
the simplest question asked of a training log was how much this week differs
from the last one. @gabbett2016 reports that when load rises by 15 percent or
more over the preceding week, injury risk rises sharply, and that figure is
quoted widely enough that a package computing the ratio and not this would be
answering half the question.

The quantity is a plain percentage change between consecutive blocks. The part
worth getting right is which blocks, because a calendar week and a trailing
seven days give different answers from the same log, and a training week that
runs Monday to Sunday and one that runs Sunday to Saturday move the long run
from one block into the other. Both forms are here and the choice is explicit.

This module reports the number and no threshold. The 15 percent figure comes
from the same body of work whose foundations :mod:`acwr.diagnostics` exists to
let a user test, and shipping it as a band would contradict everything else in
the package.

References
----------
Gabbett, T. J. (2016). The training-injury prevention paradox: should athletes
be training smarter and harder? *British Journal of Sports Medicine*, 50(5),
273-280.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ratio import _as_daily

__all__ = ["block_change", "week_to_week_change"]


def week_to_week_change(
    load: pd.Series,
    window: int = 7,
    *,
    as_percent: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """Change in trailing-window load against the window before it.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`.
    window
        Length of each block in days. Default 7, so the comparison is the last
        seven days against the seven before them.
    as_percent
        ``True`` returns a percentage, so 15.0 means a 15 percent rise.
        ``False`` returns a plain ratio, where the same rise is 0.15.
    min_periods
        Minimum observations required in each window. Defaults to a full
        window, so the first ``2 * window - 1`` days are ``NaN``.

    Returns
    -------
    pandas.Series
        Named ``"week_to_week"``.

    Notes
    -----
    A week following complete rest gives ``inf``, since any positive load is an
    infinite increase on zero. That is arithmetically correct and practically
    the single most important case in the whole metric, which is a fair summary
    of the problem with it: the number is least informative exactly where the
    risk is most discussed.

    Two consecutive weeks of complete rest give ``NaN``.

    This is a trailing comparison computed every day, not a weekly report. For
    the calendar-week form, use :func:`block_change`.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2026-01-01", periods=14, freq="D")
    >>> load = pd.Series([10.0] * 7 + [12.0] * 7, index=idx)
    >>> round(float(week_to_week_change(load).iloc[-1]), 1)
    20.0

    Seventy units became eighty-four, which is a rise of twenty percent.
    Coming back from a week off, the same function has nothing useful to say:

    >>> back = pd.Series([0.0] * 7 + [10.0] * 7, index=idx)
    >>> float(week_to_week_change(back).iloc[-1])
    inf
    """
    load = _as_daily(load)
    if window < 1:
        raise ValueError("window must be at least 1 day")

    mp = window if min_periods is None else min_periods
    current = load.rolling(window=window, min_periods=mp).sum()
    previous = current.shift(window)

    with np.errstate(divide="ignore", invalid="ignore"):
        change = (current - previous) / previous
    if as_percent:
        change = change * 100.0
    return change.rename("week_to_week")


def block_change(
    load: pd.Series,
    *,
    as_percent: bool = True,
    week_ends: str = "W-SUN",
) -> pd.DataFrame:
    """Calendar-week totals and the change between consecutive weeks.

    The form most training logs are read in: one row per week, each compared
    with the week above it.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`.
    as_percent
        As in :func:`week_to_week_change`.
    week_ends
        Pandas resampling rule naming the day a week ends on. Changing it moves
        the boundary, and a long run sitting either side of that boundary moves
        with it.

    Returns
    -------
    pandas.DataFrame
        Indexed by week ending date, with ``n_days``, ``n_observed``,
        ``total`` and ``change``.

    Notes
    -----
    A week in which any covered day is unobserved reports ``NaN`` for its
    total and its change, matching :func:`acwr.weekly_blocks`. The first and
    last rows may cover fewer than seven days, which ``n_days`` reports, so
    check it before reading the ends as weekly changes.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2026-01-05", periods=21, freq="D")  # a Monday
    >>> load = pd.Series([10.0] * 7 + [11.0] * 7 + [13.0] * 7, index=idx)
    >>> block_change(load)[["total", "change"]].round(2)
                total  change
    2026-01-11   70.0     NaN
    2026-01-18   77.0   10.00
    2026-01-25   91.0   18.18
    """
    load = _as_daily(load)
    grouped = load.resample(week_ends)
    out = pd.DataFrame(
        {
            "n_days": grouped.size(),
            "n_observed": grouped.count(),
            "total": grouped.sum(min_count=1),
        }
    )
    out.loc[out["n_observed"] < out["n_days"], "total"] = np.nan

    previous = out["total"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        change = (out["total"] - previous) / previous
    out["change"] = change * 100.0 if as_percent else change
    out.index.name = load.index.name
    return out

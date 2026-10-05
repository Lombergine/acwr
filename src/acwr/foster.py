"""Session RPE load, training monotony and training strain (Foster, 1998).

These three quantities are reported alongside the acute:chronic ratio in most
applied monitoring work, and they answer a different question. The ratio asks
whether this week is harder than the recent average. Monotony asks whether the
week was *the same every day*, on the argument that seven identical moderate
days are harder to recover from than a week with a hard day and a rest day.
Strain combines the two by scaling the weekly total by how evenly it was spread.

The definitions are stable across the literature:

    session load = session duration in minutes * session RPE
    monotony     = mean daily load over a week / standard deviation of it
    strain       = total load over the week * monotony

Three details are not stable across the literature, and each one moves the
answer, so this module makes all three explicit rather than picking silently.

The standard deviation can be the sample one or the population one. For a
seven day window the sample standard deviation is larger by a factor of
``sqrt(7/6)``, so monotony computed with ``ddof=1`` comes out about 7 percent
*lower* than the same week computed with ``ddof=0``. Published papers rarely
say which they used. This module defaults to ``ddof=1``, which is what pandas,
R and most spreadsheet implementations do, and exposes the parameter.

The window can be a calendar week or a rolling one. Foster's original work used
calendar weeks. A rolling window answers the question on every day instead of
once every seven days, which is what a monitoring tool needs, so that is the
default here; :func:`weekly_blocks` gives the calendar version.

Rest days have to be in the window as zeros. Monotony is a statement about
variability, and dropping the zeros removes exactly the variability it is
measuring: a week of five identical runs and two rest days has high variability
with the zeros and none at all without them. :func:`acwr.daily_load` produces
the continuous series this needs.

References
----------
Foster, C. (1998). Monitoring training in athletes with reference to
overtraining syndrome. *Medicine & Science in Sports & Exercise*, 30(7),
1164-1168.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ratio import _as_daily

__all__ = ["monotony", "srpe_load", "strain", "weekly_blocks"]


def srpe_load(
    duration_min: pd.Series | pd.DataFrame,
    rpe: pd.Series | None = None,
    *,
    duration_col: str = "duration_min",
    rpe_col: str = "rpe",
) -> pd.Series:
    """Session RPE load: duration in minutes multiplied by rated exertion.

    Parameters
    ----------
    duration_min
        Session durations in minutes, or a frame holding both columns.
    rpe
        Rated perceived exertion for each session. Required when
        ``duration_min`` is a Series, ignored when it is a frame.
    duration_col, rpe_col
        Column names to use when a frame is passed.

    Returns
    -------
    pandas.Series
        Load in arbitrary units, named ``"load"``.

    Notes
    -----
    The RPE scale is not checked against a fixed range. Foster's work uses the
    0 to 10 category ratio scale, but the 6 to 20 Borg scale is also reported,
    and a scale conversion applied silently would be worse than none. Negative
    durations or ratings are rejected, since neither is meaningful.

    Examples
    --------
    >>> import pandas as pd
    >>> srpe_load(pd.Series([60.0, 45.0]), pd.Series([6.0, 8.0])).tolist()
    [360.0, 360.0]

    The example is deliberate: an hour at RPE 6 and three quarters of an hour
    at RPE 8 produce the same number, which is the central assumption of the
    method and the main thing people dispute about it.
    """
    if isinstance(duration_min, pd.DataFrame):
        missing = {duration_col, rpe_col} - set(duration_min.columns)
        if missing:
            raise KeyError(f"frame is missing column(s): {sorted(missing)}")
        dur = duration_min[duration_col].astype(float)
        eff = duration_min[rpe_col].astype(float)
    else:
        if rpe is None:
            raise ValueError("rpe is required when duration_min is a Series")
        dur = duration_min.astype(float)
        eff = rpe.astype(float)
        if not dur.index.equals(eff.index):
            raise ValueError("duration and rpe must share an index")

    if (dur.dropna() < 0).any():
        raise ValueError("session duration cannot be negative")
    if (eff.dropna() < 0).any():
        raise ValueError("RPE cannot be negative")
    return (dur * eff).rename("load")


def monotony(
    load: pd.Series,
    window: int = 7,
    *,
    ddof: int = 1,
    min_periods: int | None = None,
) -> pd.Series:
    """Training monotony: mean daily load over a window divided by its spread.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`, as
        returned by :func:`acwr.daily_load`. Rest days must be present as
        zeros; see the module docstring for why.
    window
        Length of the trailing window in days. Default 7.
    ddof
        Delta degrees of freedom for the standard deviation. ``1`` is the
        sample standard deviation and the default; ``0`` is the population one.
        For a seven day window ``ddof=1`` gives a value about 7 percent lower
        than ``ddof=0``, so numbers from different tools are only comparable
        when this matches.
    min_periods
        Minimum observations required before a value is produced. Defaults to
        a full window.

    Returns
    -------
    pandas.Series
        Monotony, named ``"monotony"``.

    Notes
    -----
    A window in which every day carries the same load has zero spread, so
    monotony is ``inf`` when that load is positive and ``NaN`` when the window
    is entirely rest. Both are left as they are. The ``inf`` is not a defect in
    the data: a week of seven identical sessions really is the maximum of what
    this statistic measures, and replacing it with a large finite number would
    hide that.

    Applied work often quotes 2.0 as the level above which monotony is a
    concern. That threshold rests on the same body of evidence as the ACWR
    thresholds, so this function reports the number and does not band it.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2026-01-01", periods=7, freq="D")
    >>> steady = pd.Series([50.0] * 7, index=idx)
    >>> float(monotony(steady).iloc[-1])
    inf

    A week with a rest day and a hard day is far less monotonous than one that
    repeats the same session, even at an identical weekly total:

    >>> varied = pd.Series([0.0, 80.0, 40.0, 0.0, 90.0, 60.0, 80.0], index=idx)
    >>> round(float(monotony(varied).iloc[-1]), 3)
    1.321
    """
    load = _as_daily(load)
    if window < 2:
        raise ValueError("window must be at least 2 days; a spread needs two points")
    if ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, got {ddof!r}")
    # No further guard is needed: window is at least 2 and ddof at most 1, so
    # there is always at least one degree of freedom.

    mp = window if min_periods is None else min_periods
    roll = load.rolling(window=window, min_periods=mp)
    mean = roll.mean()
    sd = roll.std(ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = mean / sd
    return out.rename("monotony")


def strain(
    load: pd.Series,
    window: int = 7,
    *,
    ddof: int = 1,
    min_periods: int | None = None,
) -> pd.Series:
    """Training strain: total load over a window scaled by its monotony.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`.
    window
        Length of the trailing window in days. Default 7.
    ddof
        Passed to :func:`monotony`.
    min_periods
        Minimum observations required before a value is produced.

    Returns
    -------
    pandas.Series
        Strain, named ``"strain"``.

    Notes
    -----
    Strain carries the units of load multiplied by a dimensionless ratio, so
    its scale depends entirely on the load unit chosen. Strain computed from
    session RPE load is not comparable with strain computed from kilometres,
    and neither is comparable across athletes using different RPE scales. The
    quantity is useful within one athlete's own history and misleading between
    athletes.

    Because strain multiplies by monotony, it inherits the ``inf`` of a
    perfectly even window.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2026-01-01", periods=7, freq="D")
    >>> varied = pd.Series([0.0, 80.0, 40.0, 0.0, 90.0, 60.0, 80.0], index=idx)
    >>> round(float(strain(varied).iloc[-1]), 1)
    462.2
    """
    load = _as_daily(load)
    mp = window if min_periods is None else min_periods
    total = load.rolling(window=window, min_periods=mp).sum()
    mono = monotony(load, window=window, ddof=ddof, min_periods=min_periods)
    return (total * mono).rename("strain")


def weekly_blocks(
    load: pd.Series,
    *,
    ddof: int = 1,
    week_ends: str = "W-SUN",
) -> pd.DataFrame:
    """Monotony and strain per calendar week, which is Foster's original form.

    The rolling versions in :func:`monotony` and :func:`strain` answer on every
    day. This answers once per week, over blocks that do not overlap, which is
    how the quantities were originally reported and what to use when comparing
    against a published table.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`.
    ddof
        Delta degrees of freedom for the standard deviation, as in
        :func:`monotony`.
    week_ends
        Pandas resampling rule naming the day a week ends on. The default
        ``"W-SUN"`` ends weeks on Sunday. A training week that runs Monday to
        Sunday and one that runs Sunday to Saturday will not agree, and the
        disagreement is larger than it looks, because moving the boundary moves
        the long run between blocks.

    Returns
    -------
    pandas.DataFrame
        One row per week, indexed by the week ending date, with columns
        ``n_days``, ``n_observed``, ``total``, ``mean``, ``sd``, ``monotony``
        and ``strain``.

    Notes
    -----
    A week in which any covered day is unobserved reports ``NaN`` throughout,
    which matches what the rolling :func:`monotony` does on the same week.
    Only :func:`acwr.daily_load` with ``missing="na"`` produces unobserved
    days; under the default ``"zero"`` every day is a real observation and no
    week is ever suppressed on this ground.

    A partial week at either end of the series covers fewer than seven days
    and is reported as it stands, since every day it does cover was observed.
    ``n_days`` says how many days the row is built from, so check it before
    comparing the ends against full weeks.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2026-01-05", periods=14, freq="D")  # a Monday
    >>> load = pd.Series([10.0, 0.0, 20.0, 10.0, 0.0, 30.0, 10.0] * 2, index=idx)
    >>> weekly_blocks(load)[["n_days", "total", "monotony"]].round(3)
                n_days  total  monotony
    2026-01-11       7   80.0     1.069
    2026-01-18       7   80.0     1.069
    """
    load = _as_daily(load)
    grouped = load.resample(week_ends)
    # `size` counts the calendar days the block covers; `count` counts the
    # days actually observed. Resample aggregations skip NaN, so without this
    # a week with four recorded days out of seven would report a complete
    # looking monotony computed from four.
    n_days = grouped.size()
    n_observed = grouped.count()
    out = pd.DataFrame(
        {
            "n_days": n_days,
            "n_observed": n_observed,
            "total": grouped.sum(min_count=1),
            "mean": grouped.mean(),
            "sd": grouped.std(ddof=ddof),
        }
    )
    incomplete = out["n_observed"] < out["n_days"]
    out.loc[incomplete, ["total", "mean", "sd"]] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out["monotony"] = out["mean"] / out["sd"]
    out["strain"] = out["total"] * out["monotony"]
    out.index.name = load.index.name
    return out

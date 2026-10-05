"""Turn a table of training activities into a continuous daily load series.

Every acute:chronic workload ratio is computed over a daily series, so this
step comes before any of the methods in :mod:`acwr.ratio`. Two decisions get
made here and both matter more than they look.

The first is what to do with days that carry no activity. The default is to
treat them as zero load, because a rest day is a real observation of not
training rather than a gap in the record. Passing ``missing="na"`` instead
marks them as missing, which is the right choice when a stretch of the record
is absent for some other reason, such as a lost watch or a device that was
never worn.

The second is that the series must be continuous. A rolling 28 day window
computed over a frame that only contains days the athlete trained is not a 28
day window at all, and this is the most common way published ACWR numbers go
wrong.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

__all__ = ["daily_load"]


def daily_load(
    activities: pd.DataFrame,
    date_col: str = "date",
    load_col: str = "load",
    missing: Literal["zero", "na"] = "zero",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Aggregate activities to one load value per calendar day.

    Parameters
    ----------
    activities
        One row per activity. Several activities on the same day are summed.
    date_col
        Name of the date column. Parsed with :func:`pandas.to_datetime` and
        truncated to calendar days, so a timestamp is fine.
    load_col
        Name of the column holding the load value. Any numeric quantity works:
        session RPE, duration in minutes, distance, training stress score. The
        ratio is scale free, so the unit only has to be consistent. A value
        that is present but cannot be read as a number raises, because under
        ``missing="zero"`` it would otherwise become a rest day and take its
        load with it.
    missing
        ``"zero"`` fills days with no activity with 0.0, treating them as rest.
        ``"na"`` leaves them as ``NaN``, which propagates through the rolling
        windows and yields ``NaN`` ratios for any window that touches a gap.
    start, end
        Optional bounds for the returned series. Defaults to the first and last
        dates present in ``activities``. Both are truncated to calendar days,
        so a bound carrying a time of day behaves the same as the date alone.

    Returns
    -------
    pandas.Series
        Load indexed by a complete daily :class:`~pandas.DatetimeIndex`, named
        ``"load"``.

    Raises
    ------
    KeyError
        If ``date_col`` or ``load_col`` is not present.
    ValueError
        If ``activities`` is empty, if any load value is negative, or if
        ``missing`` is not one of the two accepted values.

    Examples
    --------
    >>> import pandas as pd
    >>> acts = pd.DataFrame(
    ...     {"date": ["2026-01-01", "2026-01-01", "2026-01-04"],
    ...      "load": [30.0, 20.0, 60.0]}
    ... )
    >>> daily_load(acts)
    2026-01-01    50.0
    2026-01-02     0.0
    2026-01-03     0.0
    2026-01-04    60.0
    Freq: D, Name: load, dtype: float64

    Two activities on 1 January are summed, and the two days with no training
    are filled with zero rather than skipped.
    """
    if missing not in ("zero", "na"):
        raise ValueError(f"missing must be 'zero' or 'na', got {missing!r}")
    for col in (date_col, load_col):
        if col not in activities.columns:
            raise KeyError(f"column {col!r} not found in activities")
    if len(activities) == 0:
        raise ValueError("activities is empty")

    dates = pd.to_datetime(activities[date_col]).dt.normalize()
    raw = activities[load_col]
    loads = pd.to_numeric(raw, errors="coerce").astype(float)

    # A value that was present but could not be read is not a rest day, and
    # under missing="zero" it would silently become one. Refuse it instead:
    # "1,200" with a thousands separator costs 1200 units of load and leaves
    # no trace. A value that was already missing in the input is a different
    # thing and is left to the `missing` policy.
    unreadable = loads.isna() & raw.notna()
    if unreadable.any():
        bad = raw[unreadable].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"{int(unreadable.sum())} value(s) in {load_col!r} could not be read "
            f"as numbers, for example {bad}. Clean or drop them rather than "
            "letting them become rest days."
        )

    if (loads < 0).any():
        raise ValueError("load values must be non-negative")

    per_day = (
        pd.DataFrame({"date": dates, "load": loads})
        .groupby("date", sort=True)["load"]
        .sum(min_count=1)
    )

    # Normalise the bounds. The activity dates are already at midnight, so a
    # bound carrying a time of day would build a date_range offset from them,
    # match nothing on reindex, and return a series of rest days.
    lo = pd.Timestamp(start).normalize() if start is not None else per_day.index.min()
    hi = pd.Timestamp(end).normalize() if end is not None else per_day.index.max()
    if lo > hi:
        raise ValueError(f"start {lo.date()} is after end {hi.date()}")

    full = pd.date_range(lo, hi, freq="D")
    series = per_day.reindex(full)
    if missing == "zero":
        series = series.fillna(0.0)

    series.index.name = None
    return series.rename("load")

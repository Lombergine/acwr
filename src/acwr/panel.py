"""Running the same computation across a squad rather than one athlete.

Everything else in this package takes one athlete's daily series. Monitoring
is usually done on a roster, and the step from one athlete to twenty is the
place where hand-rolled analysis goes wrong: a rolling window applied to a
frame sorted by date rather than grouped by athlete will quietly average one
runner's Tuesday into another's Wednesday, and the result looks plausible.

:func:`by_athlete` does the grouping and the daily-series construction once,
per athlete, and applies whichever function you pass to each one separately.

    >>> import pandas as pd
    >>> import acwr
    >>> activities = pd.DataFrame(
    ...     {
    ...         "athlete": ["ana", "ana", "ben", "ben"],
    ...         "date": ["2026-01-01", "2026-01-03", "2026-01-01", "2026-01-02"],
    ...         "load": [30.0, 45.0, 60.0, 20.0],
    ...     }
    ... )
    >>> out = acwr.by_athlete(activities, acwr.acwr, acute=1, chronic=2)
    >>> out
      athlete       date  acwr
    0     ana 2026-01-01   NaN
    1     ana 2026-01-02   0.0
    2     ana 2026-01-03   2.0
    3     ben 2026-01-01   NaN
    4     ben 2026-01-02   0.5

Each athlete's calendar runs from their own first activity to their own last
by default, because padding a runner who joined in March back to January would
invent three months of rest days they never had. Pass ``common_calendar=True``
when the athletes genuinely share a season and you want aligned rows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .load import daily_load

__all__ = ["by_athlete"]


def by_athlete(
    activities: pd.DataFrame,
    func: Callable[..., pd.Series | pd.DataFrame],
    *,
    athlete_col: str = "athlete",
    date_col: str = "date",
    load_col: str = "load",
    missing: str = "zero",
    common_calendar: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Apply a per-athlete computation to a frame holding several athletes.

    Parameters
    ----------
    activities
        Long frame with one row per activity, carrying an athlete identifier,
        a date and a load.
    func
        Any function taking a daily load Series and returning a Series or a
        DataFrame indexed the same way. :func:`acwr.acwr`,
        :func:`acwr.monotony` and :func:`acwr.strain` all qualify.
    athlete_col, date_col, load_col
        Column names in ``activities``.
    missing
        Passed to :func:`acwr.daily_load`. ``"zero"`` treats days without an
        activity as rest, ``"na"`` treats them as unrecorded.
    common_calendar
        If ``True``, every athlete is padded to the full span of the frame so
        all of them have the same dates. ``False`` by default, so each athlete
        spans only their own first to last activity.
    **kwargs
        Forwarded to ``func`` unchanged.

    Returns
    -------
    pandas.DataFrame
        Long frame with the athlete column, the date column, and whatever
        columns ``func`` produced.

    Raises
    ------
    KeyError
        If any named column is absent.
    ValueError
        If the frame is empty, or if an athlete identifier is missing.

    Notes
    -----
    Athletes are processed independently and in sorted order, so no athlete's
    training can leak into another's window. An athlete whose history is
    shorter than the chronic window yields all-``NaN`` rows rather than being
    dropped, which keeps the roster visible in the output.
    """
    for col in (athlete_col, date_col, load_col):
        if col not in activities.columns:
            raise KeyError(f"activities is missing the column {col!r}")
    if activities.empty:
        raise ValueError("activities frame is empty")
    if activities[athlete_col].isna().any():
        raise ValueError(f"{athlete_col!r} contains missing values")

    span: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if common_calendar:
        dates = pd.to_datetime(activities[date_col])
        span = (dates.min().normalize(), dates.max().normalize())

    frames = []
    for name, part in activities.groupby(athlete_col, sort=True):
        series = daily_load(
            part,
            date_col=date_col,
            load_col=load_col,
            missing=missing,  # type: ignore[arg-type]
            start=None if span is None else span[0],
            end=None if span is None else span[1],
        )
        result = func(series, **kwargs)
        out = result.to_frame() if isinstance(result, pd.Series) else result.copy()
        out = out.reset_index(names=date_col)
        out.insert(0, athlete_col, name)
        frames.append(out)

    return pd.concat(frames, ignore_index=True)

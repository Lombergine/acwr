"""The three published ways of computing an acute:chronic workload ratio.

All three divide a short term load average by a long term one. They differ in
how the two averages are formed, and the differences are not cosmetic: the same
training history can produce ratios that disagree about whether an athlete is
in a risky week.

Rolling average coupled (``"rac"``) is the original and by far the most widely
reported form. The chronic window contains the acute window, so today's load
appears in both the numerator and the denominator. Impellizzeri et al. (2020)
set out why that shared term creates a mathematical dependency between the two
halves of the ratio.

Rolling average uncoupled (``"rau"``) removes the overlap by taking the chronic
average over the days that precede the acute window. For the usual 7 and 28 day
settings the chronic average covers 21 days, not 28.

Exponentially weighted (``"ewma"``) replaces both flat averages with decaying
ones, so a session three days ago counts for more than the same session three
weeks ago. Williams et al. (2017) proposed it on the grounds that a flat window
treats every day inside it as equally recent, which no model of fitness or
fatigue actually claims.

None of these is endorsed here as a predictor of injury. See
:mod:`acwr.diagnostics` and the README.

References
----------
Williams, S., West, S., Cross, M. J., & Stokes, K. A. (2017). Better way to
determine the acute:chronic workload ratio? *British Journal of Sports
Medicine*, 51(3), 209-210.

Impellizzeri, F. M., Tenan, M. S., Kempton, T., Novak, A., & Coutts, A. J.
(2020). Acute:chronic workload ratio: conceptual issues and fundamental
pitfalls. *International Journal of Sports Physiology and Performance*, 15(6),
907-913.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["METHODS", "acute_load", "acwr", "chronic_load"]

METHODS = ("rac", "rau", "ewma")
DECAYS = ("span", "half_life")

Method = Literal["rac", "rau", "ewma"]
Decay = Literal["span", "half_life"]


def _alpha(n: int, decay: Decay) -> float:
    """Smoothing constant for an exponentially weighted mean over ``n`` days.

    Two parameterisations are in use and the same integer means different
    things under each, so the choice is explicit rather than assumed.

    ``"span"`` gives ``2 / (n + 1)``, which is what Williams et al. (2017)
    specify and is therefore the default. ``"half_life"`` gives
    ``1 - exp(-ln 2 / n)``, under which ``n`` is the number of days for a
    contribution to decay by half. The R package Athlytics uses the half-life
    form, so matching its output requires ``decay="half_life"``.

    For ``n = 7`` the two differ considerably: 0.25 against about 0.094.
    """
    if n < 1:
        raise ValueError("decay period must be at least 1 day")
    if decay == "span":
        return 2.0 / (n + 1.0)
    if decay == "half_life":
        return float(1.0 - np.exp(-np.log(2.0) / n))
    raise ValueError(f"decay must be one of {DECAYS}, got {decay!r}")


def _check_periods(acute: int, chronic: int) -> None:
    if acute < 1 or chronic < 1:
        raise ValueError("acute and chronic must both be at least 1 day")
    if chronic <= acute:
        raise ValueError(
            f"chronic ({chronic}) must be longer than acute ({acute}); "
            "a chronic window at or below the acute window has no long term "
            "component to compare against"
        )


def _as_daily(load: pd.Series) -> pd.Series:
    if not isinstance(load.index, pd.DatetimeIndex):
        raise TypeError(
            "load must be indexed by a DatetimeIndex; build it with acwr.daily_load()"
        )
    if len(load) == 0:
        raise ValueError("load series is empty")
    gaps = load.index.to_series().diff().dropna()
    if len(gaps) and not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError(
            "load index must be a complete daily series with no missing days; "
            "build it with acwr.daily_load(), which fills the gaps"
        )
    return load.astype(float)


def acute_load(
    load: pd.Series,
    acute: int = 7,
    method: Method = "rac",
    min_periods: int | None = None,
    decay: Decay = "span",
) -> pd.Series:
    """Short term load average.

    For ``"rac"`` and ``"rau"`` this is the rolling mean over the last ``acute``
    days inclusive of the current day. For ``"ewma"`` it is an exponentially
    weighted mean whose smoothing constant comes from :func:`_alpha`.
    """
    load = _as_daily(load)
    if acute < 1:
        raise ValueError("acute must be at least 1 day")
    mp = acute if min_periods is None else min_periods
    if method == "ewma":
        return (
            load.ewm(alpha=_alpha(acute, decay), adjust=False, min_periods=mp)
            .mean()
            .rename("acute")
        )
    return load.rolling(window=acute, min_periods=mp).mean().rename("acute")


def chronic_load(
    load: pd.Series,
    acute: int = 7,
    chronic: int = 28,
    method: Method = "rac",
    min_periods: int | None = None,
    decay: Decay = "span",
) -> pd.Series:
    """Long term load average.

    The window depends on the method. ``"rac"`` averages the last ``chronic``
    days inclusive of today, so it overlaps the acute window. ``"rau"`` averages
    the ``chronic - acute`` days that end the day before the acute window
    begins, so there is no overlap. ``"ewma"`` uses an exponentially weighted
    mean whose smoothing constant comes from :func:`_alpha`.
    """
    load = _as_daily(load)
    _check_periods(acute, chronic)

    if method == "ewma":
        # An exponentially weighted mean has no window to fill, so without a
        # warm-up it reports a ratio on day one, where both legs equal that
        # day's load and the ratio is exactly 1.0 by construction. Defaulting
        # to the same `chronic` warm-up as the other two methods keeps the
        # three comparable on the same days, which the diagnostics rely on.
        mp = chronic if min_periods is None else min_periods
        return (
            load.ewm(alpha=_alpha(chronic, decay), adjust=False, min_periods=mp)
            .mean()
            .rename("chronic")
        )

    if method == "rac":
        mp = chronic if min_periods is None else min_periods
        return load.rolling(window=chronic, min_periods=mp).mean().rename("chronic")

    if method == "rau":
        window = chronic - acute
        mp = window if min_periods is None else min_periods
        # Average the `window` days that end `acute` days ago, so the acute
        # window and the chronic window share no days at all.
        return (
            load.shift(acute)
            .rolling(window=window, min_periods=mp)
            .mean()
            .rename("chronic")
        )

    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def acwr(
    load: pd.Series,
    acute: int = 7,
    chronic: int = 28,
    method: Method = "rac",
    smooth: int | None = None,
    min_periods: int | None = None,
    return_components: bool = False,
    decay: Decay = "span",
    smooth_center: bool = False,
) -> pd.Series | pd.DataFrame:
    """Compute the acute:chronic workload ratio.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`, as
        returned by :func:`acwr.daily_load`.
    acute
        Length of the acute window in days. Default 7.
    chronic
        Length of the chronic window in days. Default 28. Must exceed ``acute``.
    method
        ``"rac"`` rolling average coupled, ``"rau"`` rolling average uncoupled,
        or ``"ewma"`` exponentially weighted. Default ``"rac"``, which is what
        the published literature almost always means by ACWR.
    smooth
        If given, apply a trailing rolling mean of this many days to the ratio
        itself. Off by default, because smoothing the ratio after the fact makes
        it harder to say what any single value represents.
    decay
        Only used when ``method="ewma"``. ``"span"`` sets the smoothing constant
        to ``2 / (N + 1)``, which is what Williams et al. (2017) specify, and is
        the default. ``"half_life"`` sets it to ``1 - exp(-ln 2 / N)``, which is
        what the R package Athlytics uses. The same ``N`` means different things
        under the two, so comparing output across packages requires matching
        this.
    smooth_center
        Whether the smoothing window is centred. ``False`` by default and that
        default is deliberate: a centred window averages days either side of the
        one being reported, so today's smoothed value would depend on training
        that has not happened yet. For monitoring in real time the window has to
        be trailing. Set it to ``True`` only for retrospective plots, where
        using the future is acceptable because all of it is already known.
    min_periods
        Minimum observations required before a value is produced. Defaults to
        a full window, so the first ``chronic - 1`` days are ``NaN`` rather
        than being computed from a partial history. The exponentially weighted
        method has no window to fill, so it takes the same ``chronic`` warm-up
        by default, which keeps the three methods comparable on the same days.
    return_components
        If ``True``, return a frame with ``load``, ``acute``, ``chronic`` and
        ``acwr`` columns instead of only the ratio. Useful when checking whether
        a high ratio came from a hard week or from an unusually easy month.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The ratio, named ``"acwr"``, or the four column frame.

    Notes
    -----
    Days on which chronic load is zero give ``inf`` where acute load is
    positive and ``NaN`` where both are zero. Dividing by a zero chronic load is
    not meaningful, so these are left as they are rather than being filled with
    a number that would look like a result.

    The coupled form has a ceiling that the uncoupled form does not. Written
    with sums instead of means, ``RAC = (chronic / acute) * (S_acute /
    S_chronic)``, and the acute window sits inside the chronic one, so
    ``S_acute <= S_chronic`` and the ratio cannot exceed ``chronic / acute``.
    At the usual 7 and 28 day settings that ceiling is exactly 4.0. A reading
    of 4.0 therefore does not mean four times the usual load, it means every
    unit of load in the last month was accumulated in the last week.

    Coupling also compresses the ratio toward 1.0 from either side. ``RAC``
    reads lower than ``RAU`` on any day the ratio is above 1 and higher on any
    day it is below 1. Since every published threshold sits above 1.0, in the
    region that gets monitored the coupled form can only understate.

    Examples
    --------
    >>> import pandas as pd
    >>> load = pd.Series(
    ...     [50.0] * 28 + [100.0] * 7,
    ...     index=pd.date_range("2026-01-01", periods=35, freq="D"),
    ... )
    >>> r = acwr(load, method="rac")
    >>> round(float(r.iloc[-1]), 4)
    1.6

    After four steady weeks at 50 and then one week at 100, the coupled ratio
    reads 1.6 rather than 2.0, because the hard week sits inside the chronic
    window too and has pulled the denominator up from 50 to 62.5. The uncoupled
    form does not do that:

    >>> round(float(acwr(load, method="rau").iloc[-1]), 4)
    2.0
    """
    load = _as_daily(load)
    _check_periods(acute, chronic)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    a = acute_load(
        load, acute=acute, method=method, min_periods=min_periods, decay=decay
    )
    c = chronic_load(
        load,
        acute=acute,
        chronic=chronic,
        method=method,
        min_periods=min_periods,
        decay=decay,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = a / c
    ratio = ratio.rename("acwr")

    if smooth is not None:
        if smooth < 1:
            raise ValueError("smooth must be at least 1 day")
        # A full window either way. With min_periods=1 a centred smoother
        # emits a value for any day with a single non-NaN neighbour, which
        # back-fills the chronic warm-up with what is really a one-day mean.
        ratio = (
            ratio.rolling(window=smooth, center=smooth_center, min_periods=smooth)
            .mean()
            .rename("acwr")
        )

    if return_components:
        return pd.DataFrame({"load": load, "acute": a, "chronic": c, "acwr": ratio})
    return ratio

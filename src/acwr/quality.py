"""Checking a training log before computing anything from it.

Every number this package produces inherits whatever is wrong with the series
it was given, and training logs are reliably wrong in a small number of ways.
A watch left running records a forty minute kilometre. A log kept in
kilometres for a year and then in metres produces a thousandfold step that no
rolling window can survive. A season break looks identical to a month the
athlete forgot to sync. None of these raises an error anywhere, and all of
them change the answer.

:func:`check_load` runs the checks and reports what it finds. It decides
nothing and alters nothing, because the right response to most of these
depends on knowledge the package does not have: whether a gap was rest or a
flat battery is a question for the athlete, not for a library.

The checks come from problems found in real logs rather than from imagination.
Three years of one runner's Strava export contained two physically impossible
day records, 44 runs with no heart rate, and six breaks of ten days or more,
all of which this module flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ratio import _as_daily

__all__ = ["Finding", "LoadQuality", "check_load"]

Severity = str  # "error" | "warning" | "note"


@dataclass(frozen=True)
class Finding:
    """One thing worth knowing about a load series before using it."""

    check: str
    severity: Severity
    message: str
    dates: tuple[pd.Timestamp, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - presentation only
        mark = {"error": "ERROR", "warning": "warn ", "note": "note "}[self.severity]
        return f"  [{mark}] {self.check}: {self.message}"


@dataclass(frozen=True)
class LoadQuality:
    """What :func:`check_load` found, and whether anything is disqualifying."""

    n_days: int
    n_training_days: int
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[Finding, ...]:
        """Findings that make a computed ratio meaningless rather than suspect."""
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Findings that change the answer without invalidating it."""
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def ok(self) -> bool:
        """``True`` when nothing disqualifying was found.

        A series can be ``ok`` and still carry warnings. The distinction is
        between a series that cannot support the computation at all and one
        that supports it with caveats the user should know about.
        """
        return len(self.errors) == 0

    def __str__(self) -> str:  # pragma: no cover - presentation only
        head = (
            f"Load quality: {self.n_days} days, {self.n_training_days} with training, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )
        if not self.findings:
            return head + "\n  nothing to report"
        return "\n".join([head, *(str(f) for f in self.findings)])


def _longest_zero_run(load: pd.Series) -> tuple[int, pd.Timestamp | None]:
    is_zero = (load == 0).to_numpy()
    best = run = 0
    best_end = end = -1
    for i, z in enumerate(is_zero):
        if z:
            run += 1
            end = i
            if run > best:
                best, best_end = run, end
        else:
            run = 0
    return best, (load.index[best_end] if best_end >= 0 else None)


def check_load(
    load: pd.Series,
    *,
    acute: int = 7,
    chronic: int = 28,
    break_days: int = 10,
    outlier_factor: float = 3.0,
    unit_shift_factor: float = 50.0,
) -> LoadQuality:
    """Report what is wrong, or merely surprising, about a daily load series.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`, as
        returned by :func:`acwr.daily_load`.
    acute, chronic
        The windows the series is about to be used with. Only used to judge
        whether the history is long enough to support them.
    break_days
        A run of this many consecutive zero-load days is reported as a break.
        Default 10.
    outlier_factor
        A day is reported as an outlier when its load exceeds this multiple of
        the 95th percentile of training days. Default 3.0.

        The comparison is against the upper tail rather than the median
        deliberately. Training load is strongly right-skewed: a long run or a
        race is several times a typical day and is not an error. Measured on
        gamma-distributed loads, the largest of 400 days routinely sits five
        times the median and under 2.3 times the 95th percentile, so a rule
        written against the median would flag every athlete's longest run and
        be ignored within a week. The watch-left-running record that motivated
        this check sat around four times the 95th percentile.
    unit_shift_factor
        A suspected change of unit is reported when the median training load of
        one half of the series differs from the other by more than this factor.
        Default 50.0, chosen to sit well above any plausible change in fitness
        and well below the thousandfold step of kilometres against metres.

    Returns
    -------
    LoadQuality
        Carrying every finding, with ``ok`` false when any is an error.

    Notes
    -----
    Severities mean specific things. An **error** makes the computation
    meaningless: no training at all, or a history shorter than the chronic
    window. A **warning** changes the answer without invalidating it: a
    suspected unit change, a long break, an extreme dynamic range. A **note**
    is something to be aware of when reading the output, such as a series that
    is mostly rest.

    Nothing here is corrected automatically. Whether a thirty day gap was a
    planned break or a month of unsynced training is not a question a library
    can answer, and guessing would propagate silently into every number that
    follows.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2026-01-01", periods=40, freq="D")
    >>> load = pd.Series([10.0] * 40, index=idx)
    >>> report = check_load(load)
    >>> report.ok
    True

    A series shorter than the chronic window cannot produce a ratio at all,
    and saying so is more useful than returning a column of ``NaN``:

    >>> short = pd.Series([10.0] * 20, index=pd.date_range("2026-01-01", periods=20))
    >>> report = check_load(short)
    >>> report.ok
    False
    >>> print(report.errors[0].message)
    20 days of history is shorter than the 28 day chronic window
    """
    load = _as_daily(load)
    findings: list[Finding] = []

    n_days = len(load)
    observed = load.dropna()
    training = observed[observed > 0]
    n_training = len(training)

    # --- things that make the computation meaningless --------------------
    if n_days < chronic:
        findings.append(
            Finding(
                "history",
                "error",
                f"{n_days} days of history is shorter than the "
                f"{chronic} day chronic window",
            )
        )
    if n_training == 0:
        findings.append(
            Finding("no training", "error", "every observed day is zero or missing")
        )

    if (observed < 0).any():
        bad = observed[observed < 0]
        findings.append(
            Finding(
                "negative load",
                "error",
                f"{len(bad)} day(s) carry a negative load",
                tuple(bad.index[:5]),
            )
        )

    # --- things that change the answer ------------------------------------
    missing = int(load.isna().sum())
    if missing:
        findings.append(
            Finding(
                "unobserved days",
                "warning",
                f"{missing} day(s) are NaN rather than zero, so every window "
                "touching them yields NaN",
                tuple(load.index[load.isna()][:5]),
            )
        )

    if n_training >= 10:
        upper = float(np.percentile(training.to_numpy(), 95))
        big = training[training > outlier_factor * upper]
        if len(big):
            findings.append(
                Finding(
                    "outlier days",
                    "warning",
                    f"{len(big)} day(s) exceed {outlier_factor:g} times the 95th "
                    f"percentile training day of {upper:.4g}, the largest being "
                    f"{float(big.max()):.4g}",
                    tuple(big.nlargest(5).index),
                )
            )

        half = len(load) // 2
        first = training[training.index < load.index[half]]
        second = training[training.index >= load.index[half]]
        if len(first) >= 5 and len(second) >= 5:
            a, b = float(first.median()), float(second.median())
            lo, hi = sorted((a, b))
            if lo > 0 and hi / lo > unit_shift_factor:
                findings.append(
                    Finding(
                        "suspected unit change",
                        "warning",
                        f"the median training day is {a:.4g} in the first half of "
                        f"the series and {b:.4g} in the second, a factor of "
                        f"{hi / lo:.0f}. A change of recording unit is far more "
                        "likely than a change of that size in training",
                    )
                )

        smallest = float(training.min())
        largest = float(training.max())
        if smallest > 0 and largest / smallest > 1e9:
            findings.append(
                Finding(
                    "extreme dynamic range",
                    "warning",
                    f"training loads span {largest / smallest:.1e}, which is wide "
                    "enough for the rolling sums to lose precision",
                )
            )

    longest, ends = _longest_zero_run(load.fillna(0.0))
    if longest >= break_days:
        findings.append(
            Finding(
                "break",
                "warning",
                f"the longest run of zero-load days is {longest} days, ending "
                f"{ends.date() if ends is not None else 'unknown'}. The ratio on "
                "the days after it is dominated by the break rather than by the "
                "training",
                (ends,) if ends is not None else (),
            )
        )

    # --- things to keep in mind when reading the output --------------------
    if n_days:
        rest_fraction = 1.0 - (n_training / n_days)
        if rest_fraction > 0.5:
            findings.append(
                Finding(
                    "mostly rest",
                    "note",
                    f"{100 * rest_fraction:.0f}% of days carry no load, so the "
                    "chronic window is mostly zeros and the ratio will be "
                    "volatile",
                )
            )

    if n_days >= chronic:
        usable = n_days - chronic + 1
        if usable < 30:
            findings.append(
                Finding(
                    "few usable days",
                    "note",
                    f"only {usable} day(s) will carry a ratio once the "
                    f"{chronic} day warm-up is taken out",
                )
            )

    if n_training and float(training.min()) == float(training.max()):
        findings.append(
            Finding(
                "constant load",
                "note",
                "every training day carries the same load, so the ratio is 1.0 "
                "wherever it is defined and monotony is infinite",
            )
        )

    return LoadQuality(
        n_days=n_days, n_training_days=n_training, findings=tuple(findings)
    )

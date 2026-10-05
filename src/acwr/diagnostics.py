"""Tools for checking whether an ACWR is telling you anything.

The ratio has a large literature arguing it does not predict injury, and the
sharpest version of that argument is a test rather than an opinion.
Impellizzeri et al. (2021) recomputed the ratio after replacing the chronic
load with contrived values, some fixed and some random, and found the
association with injury was about the same. If a denominator can be replaced
with noise without changing the result, the denominator was not doing work, and
what remains is the acute load on a different scale.

This module implements that check so anyone using the package can run it on
their own data rather than taking either side on faith. Two functions matter.

:func:`rescaling_check` needs no outcome variable at all. It reports how closely
the ratio tracks the acute load by itself. A correlation near one means the
ratio is carrying almost no information that the numerator did not already
carry.

:func:`compare_to_null` needs an outcome. It computes the association between
injury and the real ratio, then between injury and a ratio built on a fake
chronic load, and reports both with bootstrap intervals. The comparison is the
point. A real ratio that beats its own null model is evidence worth having; one
that does not is the Impellizzeri result reproduced on your own athletes.

References
----------
Impellizzeri, F. M., Woodcock, S., Coutts, A. J., Fanchini, M., McCall, A., &
Vigotsky, A. D. (2021). What role do chronic workloads play in the acute to
chronic workload ratio? Time to dismiss ACWR and its underlying theory. *Sports
Medicine*, 51(3), 581-592.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .ratio import Decay, Method, _as_daily, acute_load, acwr, chronic_load

__all__ = [
    "NullComparison",
    "RescalingResult",
    "ThresholdReport",
    "WindowSensitivity",
    "compare_to_null",
    "null_chronic_acwr",
    "rescaling_check",
    "threshold_from_history",
    "window_sensitivity",
]

NullKind = Literal["shuffle", "normal", "constant"]


@dataclass(frozen=True)
class RescalingResult:
    """How much of the ratio is explained by its own numerator."""

    n: int
    spearman: float
    pearson: float
    spearman_ci95: tuple[float, float]

    def __str__(self) -> str:  # pragma: no cover - presentation only
        lo, hi = self.spearman_ci95
        return (
            f"ACWR vs acute load alone, n={self.n}: "
            f"Spearman {self.spearman:.3f} [{lo:.3f}, {hi:.3f}], "
            f"Pearson {self.pearson:.3f}"
        )


@dataclass(frozen=True)
class NullComparison:
    """Association with an outcome for the real ratio and for a null ratio."""

    n: int
    n_events: int
    measure: str
    real: float
    real_ci95: tuple[float, float]
    null: float
    null_ci95: tuple[float, float]
    null_kind: str
    n_null_draws: int
    null_distribution: np.ndarray = field(repr=False)

    @property
    def real_beats_null(self) -> bool:
        """``True`` only if the real statistic exceeds the null interval.

        Deliberately one sided. A real ratio that lands *below* its null is not
        beating anything, and an earlier version of this property returned
        ``True`` in that case, which read as a pass when it was the opposite.
        Use :attr:`real_outside_null` for the two sided question.

        This is descriptive rather than a hypothesis test.
        """
        _, hi = self.null_ci95
        return bool(self.real > hi)

    @property
    def real_outside_null(self) -> bool:
        """``True`` if the real statistic falls outside the null interval either way.

        Being outside it below is itself informative: it means the real chronic
        denominator was actively costing you signal that a random one left
        alone. With the coupled method that happens when the outcome responds
        to acute load, because the shared window damps the spikes.
        """
        lo, hi = self.null_ci95
        return bool(self.real < lo or self.real > hi)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        rlo, rhi = self.real_ci95
        nlo, nhi = self.null_ci95
        if self.real_beats_null:
            verdict = "real ratio is ABOVE the null interval"
        elif self.real < nlo:
            verdict = (
                "real ratio is BELOW the null interval: the real denominator "
                "cost signal that a random one did not"
            )
        else:
            verdict = (
                "real ratio is INSIDE the null interval: the chronic "
                "denominator added nothing here"
            )
        return (
            f"{self.measure}, n={self.n} ({self.n_events} events)\n"
            f"  real ACWR:          {self.real:.3f} [{rlo:.3f}, {rhi:.3f}]\n"
            f"  {self.null_kind:<10} null:    {self.null:.3f} "
            f"[{nlo:.3f}, {nhi:.3f}]  ({self.n_null_draws} draws)\n"
            f"  {verdict}"
        )


def _spearman_ci(
    x: np.ndarray, y: np.ndarray, draws: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    out = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        out.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if len(out) < 10:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return (float(lo), float(hi))


def rescaling_check(
    load: pd.Series,
    acute: int = 7,
    chronic: int = 28,
    method: Method = "rac",
    draws: int = 2000,
    seed: int = 0,
) -> RescalingResult:
    """Correlate the ratio against the acute load on its own.

    The ratio is acute load divided by chronic load. When chronic load varies
    little relative to acute load, the division barely changes the ordering of
    days, and the ratio becomes a rescaled copy of its numerator. That is the
    core of the Impellizzeri critique and it can be checked without any injury
    data.

    Returns
    -------
    RescalingResult
        Spearman and Pearson correlations between the ratio and the acute load,
        with a bootstrap interval on the Spearman.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> load = pd.Series(
    ...     rng.gamma(2.0, 25.0, 400),
    ...     index=pd.date_range("2026-01-01", periods=400, freq="D"),
    ... )
    >>> res = rescaling_check(load)
    >>> res.spearman > 0.8
    True

    On noisy day to day training the ratio tracks the acute load closely, which
    is exactly the point the critique makes.
    """
    frame = acwr(
        load,
        acute=acute,
        chronic=chronic,
        method=method,
        return_components=True,
    )
    joined = frame[["acute", "acwr"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < 3:
        raise ValueError(
            f"only {len(joined)} usable days after dropping incomplete windows; "
            f"need at least 3. A {chronic} day chronic window needs a longer "
            "load series."
        )
    a = joined["acute"].to_numpy()
    r = joined["acwr"].to_numpy()
    return RescalingResult(
        n=len(joined),
        spearman=float(stats.spearmanr(a, r).statistic),
        pearson=float(stats.pearsonr(a, r).statistic),
        spearman_ci95=_spearman_ci(a, r, draws=draws, seed=seed),
    )


def null_chronic_acwr(
    load: pd.Series,
    acute: int = 7,
    chronic: int = 28,
    method: Method = "rac",
    kind: NullKind = "shuffle",
    seed: int = 0,
) -> pd.Series:
    """Build an ACWR whose denominator carries no real information.

    The numerator is the genuine acute load. The denominator is replaced,
    following Impellizzeri et al. (2021), by one of three contrivances.

    Parameters
    ----------
    kind
        ``"shuffle"`` randomly permutes the real chronic series, so the
        denominator keeps its exact distribution and loses its timing.
        ``"normal"`` draws from a normal distribution matched to the real
        chronic mean and standard deviation, truncated at a small positive
        floor. ``"constant"`` uses the mean of the real chronic series, which is
        the fixed-value version of the test.
    seed
        Seeds the draw so a given null series is reproducible.

    Returns
    -------
    pandas.Series
        A ratio series named ``"acwr_null"``, aligned to ``load``.
    """
    a = acute_load(load, acute=acute, method=method)
    c = chronic_load(load, acute=acute, chronic=chronic, method=method)

    valid = c.notna() & np.isfinite(c)
    real = c[valid].to_numpy()
    if len(real) < 2:
        raise ValueError("not enough chronic load values to build a null model")

    rng = np.random.default_rng(seed)
    if kind == "shuffle":
        fake = rng.permutation(real)
    elif kind == "normal":
        floor = max(real.mean() * 1e-3, 1e-9)
        fake = np.clip(
            rng.normal(real.mean(), real.std(ddof=1) or 1.0, len(real)), floor, None
        )
    elif kind == "constant":
        fake = np.full(len(real), real.mean())
    else:
        raise ValueError(
            f"kind must be 'shuffle', 'normal' or 'constant', got {kind!r}"
        )

    c_null = pd.Series(np.nan, index=c.index, dtype=float)
    c_null.loc[valid] = fake
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a / c_null).rename("acwr_null")


def _association(x: np.ndarray, y: np.ndarray, binary: bool) -> tuple[float, str]:
    """AUC for a binary outcome, Spearman otherwise.

    Returns ``nan`` rather than warning when either side is constant, since a
    correlation is undefined there and a bootstrap resample can easily produce
    a constant draw.
    """
    if binary:
        pos = x[y == 1]
        neg = x[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return (float("nan"), "AUC")
        u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
        return (float(u / (len(pos) * len(neg))), "AUC")
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return (float("nan"), "Spearman rho")
    return (float(stats.spearmanr(x, y).statistic), "Spearman rho")


def compare_to_null(
    load: pd.Series,
    outcome: pd.Series,
    acute: int = 7,
    chronic: int = 28,
    method: Method = "rac",
    kind: NullKind = "shuffle",
    n_null_draws: int = 200,
    boot_draws: int = 2000,
    seed: int = 0,
) -> NullComparison:
    """Run the Impellizzeri test on your own data.

    Computes the association between ``outcome`` and the real ratio, then repeats
    it ``n_null_draws`` times against ratios whose chronic denominator has been
    replaced by a contrivance, and reports both.

    Parameters
    ----------
    load
        Daily load series, as from :func:`acwr.daily_load`.
    outcome
        Aligned to ``load`` by index. A column of 0 and 1, or booleans, is
        treated as a binary event and scored by AUC. Anything else is treated as
        continuous and scored by Spearman.
    kind
        Which contrivance to use for the denominator. See
        :func:`null_chronic_acwr`.
    n_null_draws
        How many null series to generate. The spread across these draws is what
        the null interval describes.

    Returns
    -------
    NullComparison
        Both statistics with bootstrap intervals, and a
        :attr:`~NullComparison.real_beats_null` flag.

    Notes
    -----
    This is a descriptive comparison, not a significance test, and it says
    nothing about causation. A real ratio inside its own null interval means the
    chronic denominator added nothing on this dataset. It does not establish
    that the ratio is useless everywhere, and a result outside the interval does
    not establish that the ratio predicts injury.
    """
    real_ratio = acwr(load, acute=acute, chronic=chronic, method=method)
    frame = pd.DataFrame({"acwr": real_ratio, "outcome": outcome}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna()
    if len(frame) < 10:
        raise ValueError(
            f"only {len(frame)} aligned, complete observations; need at least 10"
        )

    y = frame["outcome"].to_numpy()
    uniq = np.unique(y)
    binary = bool(set(uniq.tolist()) <= {0, 1}) and len(uniq) == 2

    real_stat, measure = _association(frame["acwr"].to_numpy(), y, binary)

    keep = frame.index
    null_stats = []
    for d in range(n_null_draws):
        nr = null_chronic_acwr(
            load,
            acute=acute,
            chronic=chronic,
            method=method,
            kind=kind,
            seed=seed + d,
        )
        nr = nr.replace([np.inf, -np.inf], np.nan).reindex(keep)
        ok = nr.notna()
        if ok.sum() < 10:
            # A null draw usable on fewer than ten days describes nothing, so
            # it is skipped rather than counted. Reaching this needs a draw
            # that is unusable on more days than the real ratio was, which the
            # test suite has not managed to construct; it is kept as a guard
            # rather than removed, because the alternative is a distribution
            # built from a handful of points.
            continue
        s, _ = _association(nr[ok].to_numpy(), y[ok.to_numpy()], binary)
        if not np.isnan(s):
            null_stats.append(s)

    if len(null_stats) < 10:
        raise ValueError("could not build enough null draws to describe a distribution")
    null_arr = np.asarray(null_stats)

    # Interval on the real statistic: resample observations.
    rng = np.random.default_rng(seed)
    x_real = frame["acwr"].to_numpy()
    n = len(x_real)
    boots = []
    for _ in range(boot_draws):
        idx = rng.integers(0, n, n)
        if binary and len(np.unique(y[idx])) < 2:
            continue
        s, _ = _association(x_real[idx], y[idx], binary)
        if not np.isnan(s):
            boots.append(s)
    real_ci = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if len(boots) >= 10
        else (float("nan"), float("nan"))
    )

    return NullComparison(
        n=n,
        n_events=int(y.sum()) if binary else n,
        measure=measure,
        real=real_stat,
        real_ci95=real_ci,
        null=float(null_arr.mean()),
        null_ci95=(
            float(np.percentile(null_arr, 2.5)),
            float(np.percentile(null_arr, 97.5)),
        ),
        null_kind=kind,
        n_null_draws=len(null_arr),
        null_distribution=null_arr,
    )


@dataclass(frozen=True)
class WindowSensitivity:
    """How much the flagged days depend on the window lengths you chose."""

    threshold: float
    method: Method
    reference: tuple[int, int]
    table: pd.DataFrame

    @property
    def agreement_range(self) -> tuple[float, float]:
        """Lowest and highest agreement with the reference pair."""
        other = self.table[~self.table["is_reference"]]["agreement"].dropna()
        if other.empty:
            return (float("nan"), float("nan"))
        return (float(other.min()), float(other.max()))

    def __str__(self) -> str:  # pragma: no cover - presentation only
        lo, hi = self.agreement_range
        ref_rate = float(
            self.table.loc[self.table["is_reference"], "flagged_fraction"].iloc[0]
        )
        return (
            f"Window sensitivity at threshold {self.threshold:g} "
            f"({self.method}, {len(self.table)} window pairs)\n"
            f"  reference {self.reference[0]}/{self.reference[1]} flags "
            f"{100 * ref_rate:.1f}% of days\n"
            f"  flagged fraction across pairs: "
            f"{100 * self.table['flagged_fraction'].min():.1f}% to "
            f"{100 * self.table['flagged_fraction'].max():.1f}%\n"
            f"  agreement with the reference: {lo:.2f} to {hi:.2f} (Jaccard)"
        )


def window_sensitivity(
    load: pd.Series,
    acute_options: Sequence[int] = (5, 7, 10, 14),
    chronic_options: Sequence[int] = (21, 28, 35, 42),
    *,
    threshold: float = 1.5,
    method: Method = "rac",
    reference: tuple[int, int] = (7, 28),
    decay: Decay = "span",
) -> WindowSensitivity:
    """Ask how much the answer depends on the windows nobody justifies.

    Seven and twenty-eight days are conventional. They are not derived from
    anything, and papers that use them rarely say why. This sweeps a grid of
    window pairs, flags the days each one puts above ``threshold``, and reports
    how far those sets of days agree with the pair you would have used anyway.

    Agreement is the Jaccard index: the number of days both pairs flag divided
    by the number either flags. A value of 1.0 means the choice of windows made
    no difference to which days got flagged. A value of 0.5 means half the
    flagged days came from the window choice rather than from the training.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`.
    acute_options, chronic_options
        Window lengths to sweep. Pairs where the chronic window does not exceed
        the acute one are skipped.
    threshold
        The value above which a day counts as flagged. Default 1.5, which is
        the figure applied practice quotes most often.
    method
        Which ratio to compute, as in :func:`acwr.acwr`.
    reference
        The pair everything is compared against. Default ``(7, 28)``.
    decay
        Passed through when ``method="ewma"``.

    Returns
    -------
    WindowSensitivity
        Holding ``table``, one row per pair with the flagged count, the flagged
        fraction, and the Jaccard agreement with the reference.

    Notes
    -----
    Days are compared only where both the pair and the reference produce a
    finite ratio, so a long chronic window is not penalised for the extra
    warm-up days it needs. A pair sharing no comparable day with the
    reference, which happens when its chronic window is longer than the
    series, reports an agreement of ``NaN`` rather than 1.0. The reference
    pair must be inside the grid.

    This is descriptive. A low agreement does not prove the metric is
    worthless, and a high one does not validate it. What it does is put a
    number on a choice that usually goes unexamined.
    """
    load = _as_daily(load)
    if reference[0] >= reference[1]:
        raise ValueError("reference chronic window must exceed the acute one")
    if reference[0] not in acute_options or reference[1] not in chronic_options:
        raise ValueError(f"reference pair {reference} must be inside the swept grid")

    def flags(a: int, c: int) -> pd.Series:
        r = acwr(load, acute=a, chronic=c, method=method, decay=decay)
        assert isinstance(r, pd.Series)
        return r.replace([np.inf, -np.inf], np.nan).dropna() > threshold

    ref_flags = flags(*reference)
    rows = []
    for a in acute_options:
        for c in chronic_options:
            if c <= a:
                continue
            f = flags(a, c)
            shared = f.index.intersection(ref_flags.index)
            x, y = f.loc[shared], ref_flags.loc[shared]
            if len(shared) == 0:
                # No day is comparable, usually because the chronic window is
                # longer than the series. Reporting 1.0 here would read as
                # perfect agreement from a pair that produced nothing.
                jaccard = float("nan")
            else:
                union = int((x | y).sum())
                jaccard = 1.0 if union == 0 else float((x & y).sum()) / union
            rows.append(
                {
                    "acute": a,
                    "chronic": c,
                    "n_days": len(f),
                    "n_flagged": int(f.sum()),
                    "flagged_fraction": float(f.mean()) if len(f) else float("nan"),
                    "agreement": jaccard,
                    "is_reference": (a, c) == reference,
                }
            )

    table = pd.DataFrame(rows).sort_values(["acute", "chronic"]).reset_index(drop=True)
    return WindowSensitivity(
        threshold=threshold, method=method, reference=reference, table=table
    )


@dataclass(frozen=True)
class ThresholdReport:
    """What a universal threshold means for one athlete's own distribution."""

    method: Method
    n: int
    percentiles: dict[float, float]
    reference: float
    reference_flagged: int

    @property
    def reference_flagged_fraction(self) -> float:
        """Share of this athlete's usable days that the reference flags."""
        return self.reference_flagged / self.n if self.n else float("nan")

    @property
    def reference_percentile(self) -> float:
        """Where the reference threshold sits in this athlete's own history."""
        return 100.0 * (1.0 - self.reference_flagged_fraction)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        lines = [
            f"Thresholds on this athlete's own history "
            f"({self.method}, n={self.n} usable days)",
            f"  {self.reference:g} flags {self.reference_flagged} days "
            f"({100 * self.reference_flagged_fraction:.1f}%), which is this "
            f"athlete's {self.reference_percentile:.1f}th percentile",
        ]
        lines.append("  their own percentiles:")
        for pct, value in sorted(self.percentiles.items()):
            lines.append(f"    {pct:g}th  {value:.2f}")
        return "\n".join(lines)


def threshold_from_history(
    load: pd.Series,
    percentiles: Sequence[float] = (50, 75, 90, 95, 99),
    *,
    acute: int = 7,
    chronic: int = 28,
    method: Method = "rac",
    reference: float = 1.5,
    decay: Decay = "span",
) -> ThresholdReport:
    """Put a universal threshold next to the athlete it is being applied to.

    The figure quoted in applied practice is 1.5, the same number for a
    marathoner in a base block and a sprinter in competition. This reports what
    that number actually picks out of one athlete's own history, and what cut
    would be needed to pick out a given share of their hardest days instead.

    Parameters
    ----------
    load
        Daily load indexed by a complete :class:`~pandas.DatetimeIndex`.
    percentiles
        Which percentiles of this athlete's ratio distribution to report.
    acute, chronic, method, decay
        Passed to :func:`acwr.acwr`.
    reference
        The universal threshold to compare against. Default 1.5.

    Returns
    -------
    ThresholdReport

    Notes
    -----
    Infinite ratios, which a return from complete rest produces under the
    uncoupled and exponentially weighted methods, are excluded from the
    percentiles and counted as flagged. Leaving them in would make every
    percentile above the first infinite one meaningless.

    This is descriptive and is not a recommendation to use a personalised
    threshold. A threshold chosen from an athlete's own distribution flags a
    fixed share of their days by construction, whatever their training was
    like, which is a different failure from the one it fixes. The point of the
    function is to make the arbitrariness of the universal number visible
    rather than to replace it with a different arbitrary number.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> load = pd.Series(
    ...     rng.gamma(2.0, 30.0, 400),
    ...     index=pd.date_range("2025-01-01", periods=400, freq="D"),
    ... )
    >>> report = threshold_from_history(load)
    >>> report.reference_flagged_fraction < 0.05
    True
    """
    ratio = acwr(load, acute=acute, chronic=chronic, method=method, decay=decay)
    assert isinstance(ratio, pd.Series)
    usable = ratio.dropna()
    if len(usable) == 0:
        raise ValueError("no usable days; the series is shorter than the warm-up")

    finite = usable.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) == 0:
        raise ValueError("every usable day is infinite; percentiles are undefined")

    values = {float(p): float(np.percentile(finite.to_numpy(), p)) for p in percentiles}
    flagged = int((usable > reference).sum())
    return ThresholdReport(
        method=method,
        n=len(usable),
        percentiles=values,
        reference=float(reference),
        reference_flagged=flagged,
    )

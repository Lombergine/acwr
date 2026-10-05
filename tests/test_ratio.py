"""Tests for the three ratio methods.

Where a value can be worked out by hand it is pinned to the hand-computed
number rather than to whatever the implementation happened to produce first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acwr import acute_load, acwr, chronic_load


def days(values: list[float], start: str = "2026-01-01") -> pd.Series:
    return pd.Series(
        [float(v) for v in values],
        index=pd.date_range(start, periods=len(values), freq="D"),
        name="load",
    )


# --------------------------------------------------------------------------
# the number everyone cites
# --------------------------------------------------------------------------


def test_steady_training_gives_a_ratio_of_one():
    load = days([50.0] * 60)
    r = acwr(load, method="rac")
    assert r.dropna().apply(lambda v: v == pytest.approx(1.0)).all()


def test_coupled_ratio_matches_hand_computation():
    # 28 days at 50, then 7 days at 100.
    # acute   = mean(last 7)  = 100
    # chronic = mean(last 28) = (21*50 + 7*100) / 28 = 62.5
    load = days([50.0] * 28 + [100.0] * 7)
    assert float(acwr(load, method="rac").iloc[-1]) == pytest.approx(100 / 62.5)


def test_uncoupled_ratio_matches_hand_computation():
    # chronic = mean of the 21 days ending 7 days ago = 50 exactly
    load = days([50.0] * 28 + [100.0] * 7)
    assert float(acwr(load, method="rau").iloc[-1]) == pytest.approx(2.0)


def test_coupled_is_always_the_more_conservative_of_the_two_on_a_spike():
    """Sharing the spike with the denominator pulls the coupled ratio down."""
    load = days([50.0] * 28 + [100.0] * 7)
    rac = float(acwr(load, method="rac").iloc[-1])
    rau = float(acwr(load, method="rau").iloc[-1])
    assert rac < rau


# --------------------------------------------------------------------------
# window mechanics
# --------------------------------------------------------------------------


def test_acute_and_chronic_windows_do_not_overlap_when_uncoupled():
    """A spike confined to the acute window must leave uncoupled chronic flat."""
    load = days([10.0] * 30 + [999.0] * 7)
    c = chronic_load(load, acute=7, chronic=28, method="rau")
    assert float(c.iloc[-1]) == pytest.approx(10.0)


def test_coupled_chronic_does_see_the_spike():
    load = days([10.0] * 30 + [999.0] * 7)
    c = chronic_load(load, acute=7, chronic=28, method="rac")
    assert float(c.iloc[-1]) > 10.0


def test_full_windows_are_required_before_a_value_appears():
    load = days([50.0] * 40)
    r = acwr(load, acute=7, chronic=28, method="rac")
    assert r.iloc[:27].isna().all()
    assert not np.isnan(r.iloc[27])


def test_uncoupled_also_needs_the_full_span():
    load = days([50.0] * 40)
    r = acwr(load, acute=7, chronic=28, method="rau")
    # shift(7) over a 21 day window still needs 28 days of history
    assert r.iloc[:27].isna().all()
    assert not np.isnan(r.iloc[27])


def test_min_periods_allows_partial_windows():
    load = days([50.0] * 10)
    r = acwr(load, acute=7, chronic=28, method="rac", min_periods=3)
    assert not np.isnan(r.iloc[2])


# --------------------------------------------------------------------------
# ewma
# --------------------------------------------------------------------------


def test_ewma_lambda_follows_the_published_formula():
    """Williams et al. (2017) set lambda = 2 / (N + 1)."""
    load = days([0.0] * 10 + [100.0] + [0.0] * 10)
    a = acute_load(load, acute=7, method="ewma")
    lam = 2.0 / (7.0 + 1.0)
    # the day of the spike takes lambda * 100 plus the decayed prior, which is 0
    assert float(a.iloc[10]) == pytest.approx(lam * 100.0)
    # the next day decays by (1 - lambda)
    assert float(a.iloc[11]) == pytest.approx(lam * 100.0 * (1 - lam))


def test_ewma_warms_up_like_the_other_methods_so_the_three_are_comparable():
    """An exponentially weighted mean has no window, so it needs a warm-up.

    Without one it reports a ratio on day one, where both legs equal that
    day's load and the ratio is exactly 1.0 by construction rather than by
    anything the athlete did. Worse, the three methods would then be computed
    over different numbers of days, and the diagnostics compare them.
    """
    load = days([50.0] * 40)
    for method in ("rac", "rau", "ewma"):
        r = acwr(load, method=method)
        assert int(r.isna().sum()) == 27, f"{method} should warm up for 28 days"
    assert not np.isnan(acwr(load, method="ewma").iloc[27])


def test_ewma_honours_an_explicit_min_periods():
    load = days([50.0] * 40)
    assert int(acwr(load, method="ewma", min_periods=1).isna().sum()) == 0
    assert int(acwr(load, method="ewma", min_periods=10).isna().sum()) == 9


def test_ewma_on_steady_load_converges_to_one():
    load = days([50.0] * 400)
    r = acwr(load, method="ewma")
    assert float(r.iloc[-1]) == pytest.approx(1.0, abs=1e-9)


def test_ewma_weights_recent_days_more_heavily_than_rolling_does():
    """A spike seven days ago should have faded more under EWMA."""
    load = days([50.0] * 50 + [300.0] + [50.0] * 6)
    rac = float(acwr(load, method="rac").iloc[-1])
    ewma = float(acwr(load, method="ewma").iloc[-1])
    assert ewma < rac


# --------------------------------------------------------------------------
# edge cases that should not silently produce a number
# --------------------------------------------------------------------------


def test_zero_chronic_load_gives_inf_not_a_made_up_number():
    load = days([0.0] * 28 + [10.0] * 7)
    r = acwr(load, method="rau")
    assert np.isinf(float(r.iloc[-1]))


def test_no_training_at_all_gives_nan():
    load = days([0.0] * 40)
    r = acwr(load, method="rac")
    assert np.isnan(float(r.iloc[-1]))


def test_chronic_must_exceed_acute():
    load = days([50.0] * 40)
    with pytest.raises(ValueError, match="must be longer than"):
        acwr(load, acute=7, chronic=7)
    with pytest.raises(ValueError, match="must be longer than"):
        acwr(load, acute=28, chronic=7)


def test_unknown_method_is_rejected():
    load = days([50.0] * 40)
    with pytest.raises(ValueError, match="method must be one of"):
        acwr(load, method="rolling")


def test_non_datetime_index_is_rejected():
    load = pd.Series([50.0] * 40)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        acwr(load)


def test_series_with_a_missing_day_is_rejected():
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"])
    load = pd.Series([1.0, 2.0, 3.0], index=idx)
    with pytest.raises(ValueError, match="complete daily series"):
        acwr(load, acute=1, chronic=2)


def test_empty_series_is_rejected():
    load = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="empty"):
        acwr(load)


# --------------------------------------------------------------------------
# shape of the output
# --------------------------------------------------------------------------


def test_components_frame_reconstructs_the_ratio():
    load = days([50.0, 60.0] * 30)
    frame = acwr(load, return_components=True)
    assert list(frame.columns) == ["load", "acute", "chronic", "acwr"]
    ok = frame.dropna()
    recomputed = ok["acute"] / ok["chronic"]
    assert np.allclose(recomputed.to_numpy(), ok["acwr"].to_numpy())


def test_output_is_aligned_to_the_input_index():
    load = days([50.0] * 40)
    r = acwr(load)
    assert r.index.equals(load.index)
    assert r.name == "acwr"


def test_smoothing_reduces_variance_without_shifting_the_mean_much():
    rng = np.random.default_rng(0)
    load = days(rng.gamma(2.0, 25.0, 300).tolist())
    raw = acwr(load).dropna()
    smoothed = acwr(load, smooth=7).dropna()
    assert smoothed.std() < raw.std()
    assert float(smoothed.mean()) == pytest.approx(float(raw.mean()), rel=0.1)


def test_smooth_must_be_positive():
    load = days([50.0] * 40)
    with pytest.raises(ValueError, match="smooth must be"):
        acwr(load, smooth=0)


# --------------------------------------------------------------------------
# cross-check against the Athlytics R implementation
#
# Athlytics (rOpenSci) is the reference implementation in R. Its source was
# read at github.com/ropensci/Athlytics and the coupled rolling-average path
# reduces to:
#
#   acute_load   = zoo::rollmean(daily_load, k = acute,   fill = NA, align = "right")
#   chronic_load = zoo::rollmean(daily_load, k = chronic, fill = NA, align = "right")
#   acwr         = ifelse(chronic_load > 0.01, acute_load / chronic_load, NA)
#
# The function below reimplements that literally so the two can be compared on
# identical input. Divergences are documented in the README rather than
# papered over.
# --------------------------------------------------------------------------


def athlytics_rac(load: pd.Series, acute: int = 7, chronic: int = 28) -> pd.Series:
    """Reimplement the Athlytics coupled rolling-average path from its R source."""
    a = load.rolling(window=acute, min_periods=acute).mean()
    c = load.rolling(window=chronic, min_periods=chronic).mean()
    out = pd.Series(np.nan, index=load.index, dtype=float)
    usable = c.notna() & (c > 0.01)
    out[usable] = a[usable] / c[usable]
    return out


def test_coupled_method_matches_the_athlytics_algorithm():
    """Our rac path must agree with the R reference wherever both are defined."""
    rng = np.random.default_rng(314)
    vals = np.clip(rng.gamma(2.0, 30.0, 400), 0.0, None)
    vals[rng.random(400) < 0.15] = 0.0
    load = days(vals.tolist())

    ours = acwr(load, acute=7, chronic=28, method="rac")
    theirs = athlytics_rac(load, acute=7, chronic=28)

    both = pd.DataFrame({"ours": ours, "theirs": theirs}).dropna()
    assert len(both) > 350
    assert np.allclose(both["ours"].to_numpy(), both["theirs"].to_numpy())


def test_coupled_acwr_can_never_divide_by_zero_when_training_happened():
    """A structural property, found while comparing against Athlytics.

    Athlytics guards the division with a 0.01 floor on chronic load. For the
    coupled method that guard can never fire on a positive numerator, because
    the acute window is contained in the chronic window: any training inside
    the acute window is also inside the chronic window, so a positive acute
    load forces a positive chronic load.

    The floor therefore only matters for the uncoupled and EWMA paths, where
    the two windows can genuinely come apart.
    """
    rng = np.random.default_rng(77)
    for _ in range(50):
        vals = np.zeros(40)
        # put all the training in the last 7 days, the worst case for this
        vals[-7:] = rng.gamma(2.0, 20.0, 7)
        load = days(vals.tolist())
        frame = acwr(load, method="rac", return_components=True).dropna()
        positive_acute = frame["acute"] > 0
        assert (frame.loc[positive_acute, "chronic"] > 0).all()
        assert np.isfinite(frame.loc[positive_acute, "acwr"]).all()


def test_uncoupled_acwr_is_where_the_zero_denominator_actually_bites():
    """Here the windows are disjoint, so inf is reachable and is reported."""
    load = days([0.0] * 28 + [40.0] * 7)
    ours = acwr(load, method="rau")
    assert np.isinf(float(ours.iloc[-1]))

    # Athlytics has no uncoupled path, so there is nothing to compare against;
    # the behaviour is pinned here so it cannot drift silently.
    assert np.isnan(float(acwr(days([0.0] * 40), method="rau").iloc[-1]))


# --------------------------------------------------------------------------
# ewma parameterisation
# --------------------------------------------------------------------------


def test_span_decay_follows_williams_et_al():
    """Lambda = 2 / (N + 1), from the BJSM correspondence."""
    from acwr.ratio import _alpha

    assert _alpha(7, "span") == pytest.approx(0.25)
    assert _alpha(28, "span") == pytest.approx(2 / 29)


def test_half_life_decay_follows_the_athlytics_form():
    """Alpha = 1 - exp(-ln 2 / N), which Athlytics uses instead."""
    from acwr.ratio import _alpha

    assert _alpha(7, "half_life") == pytest.approx(1 - np.exp(-np.log(2) / 7))
    # and it is much slower than the span form for the same N
    assert _alpha(7, "half_life") < _alpha(7, "span") / 2


def test_the_two_decays_give_different_ratios():
    """Same N, different parameterisation, different answer. Hence the option."""
    load = days([50.0] * 50 + [150.0] * 7)
    span = float(acwr(load, method="ewma", decay="span").iloc[-1])
    half = float(acwr(load, method="ewma", decay="half_life").iloc[-1])
    assert span != pytest.approx(half)
    # the slower half-life decay has not caught up with the spike yet
    assert half < span


def test_unknown_decay_is_rejected():
    load = days([50.0] * 40)
    with pytest.raises(ValueError, match="decay must be one of"):
        acwr(load, method="ewma", decay="halflife")


# --------------------------------------------------------------------------
# smoothing must not use the future
# --------------------------------------------------------------------------


def test_smoothing_is_trailing_by_default_and_does_not_use_the_future():
    """Today's smoothed value must not change when tomorrow's load changes.

    A centred window would break this, and the first version of this package
    centred by default. For a tool meant to inform training decisions that is a
    bug, not a preference: it leaks data the athlete cannot have.
    """
    base = [50.0] * 60
    a = days(base)
    b = days([*base[:-1], 500.0])  # only the final day differs

    sa = acwr(a, smooth=7)
    sb = acwr(b, smooth=7)

    # every day before the last must be identical
    assert np.allclose(
        sa.iloc[:-1].dropna().to_numpy(), sb.iloc[:-1].dropna().to_numpy()
    )


def test_centred_smoothing_can_be_requested_for_retrospective_plots():
    base = [50.0] * 60
    a = days(base)
    b = days([*base[:-1], 500.0])
    sa = acwr(a, smooth=7, smooth_center=True)
    sb = acwr(b, smooth=7, smooth_center=True)
    # with centring, earlier days DO move, which is why it is not the default
    assert not np.allclose(
        sa.iloc[:-1].dropna().to_numpy(), sb.iloc[:-1].dropna().to_numpy()
    )


# ---------------------------------------------------------------------------
# Structural properties of the coupled form.
#
# Both of these were found by running the package on three years of real
# Strava data, where the coupled ratio hit exactly 4.0 on seven separate days
# and never once went above it. That is not a coincidence in the data, it is
# arithmetic, and neither R implementation documents it.
# ---------------------------------------------------------------------------


def _rng_load(n=900, seed=11):
    """Lumpy daily load with rest days and layoffs, like a real training log."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(shape=2.0, scale=5.0, size=n)
    x[rng.random(n) < 0.40] = 0.0  # rest days
    x[300:340] = 0.0  # an injury layoff
    x[700:712] = 0.0  # a holiday
    return pd.Series(x, index=pd.date_range("2023-01-01", periods=n, freq="D"))


@pytest.mark.parametrize(("acute", "chronic"), [(3, 21), (7, 21), (7, 28), (10, 28)])
def test_coupled_ratio_can_never_exceed_chronic_over_acute(acute, chronic):
    """The coupled form has a hard ceiling, and it is ``chronic / acute``.

    Writing the ratio with sums rather than means,

        RAC = (S_a / a) / (S_c / c) = (c / a) * (S_a / S_c)

    where ``S_a`` is the load summed over the acute window and ``S_c`` over the
    chronic one. Because the acute window sits inside the chronic window and
    load is non-negative, ``S_a <= S_c`` always, so ``S_a / S_c <= 1`` and the
    ratio cannot exceed ``c / a``. For the usual 7 and 28 day settings that is
    exactly 4.0, whatever the athlete does.

    This matters for interpretation. A coupled ACWR of 4.0 does not mean
    "four times the usual load", it means "every kilometre of the last month
    was run in the last week", which is the most extreme value the statistic
    can take. The uncoupled form has no such ceiling.
    """
    load = _rng_load()
    ratio = acwr(load, acute=acute, chronic=chronic, method="rac")
    finite = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    bound = chronic / acute
    assert (finite <= bound + 1e-9).all()
    # And the bound is attained, not merely respected: force all the chronic
    # window's load into the acute window.
    spike = pd.Series(
        [0.0] * chronic + [10.0] * acute,
        index=pd.date_range("2024-01-01", periods=chronic + acute, freq="D"),
    )
    assert float(acwr(spike, acute=acute, chronic=chronic, method="rac").iloc[-1]) == (
        pytest.approx(bound)
    )


def test_coupling_compresses_the_ratio_toward_one():
    """Coupling pulls the ratio toward 1.0 from whichever side it is on.

    Comparing the two denominators, ``RAC < RAU`` exactly when
    ``S_a / S_c > 1 / (c / a)``, which is the same condition as ``RAC > 1``.
    So on a week harder than the recent average the coupled ratio reads lower
    than the uncoupled one, and on an easier week it reads higher. The
    practical consequence is one sided: every published threshold sits above
    1.0, so in the region anyone actually monitors, coupling can only
    understate.
    """
    load = _rng_load()
    rac = acwr(load, method="rac")
    rau = acwr(load, method="rau")
    acute_sum = load.rolling(7).sum()
    both = (
        pd.DataFrame({"rac": rac, "rau": rau, "s": acute_sum})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    trained = both[both.s > 0]
    assert len(trained) > 500, "the fixture should produce plenty of usable days"

    ramping = trained[trained.rac > 1 + 1e-12]
    tapering = trained[trained.rac < 1 - 1e-12]
    assert len(ramping) > 100 and len(tapering) > 100
    assert (ramping.rac < ramping.rau).all()
    assert (tapering.rac > tapering.rau).all()

    # The one sided consequence at the threshold people quote.
    uncoupled_only = ((both.rac < 1.5) & (both.rau > 1.5)).sum()
    coupled_only = ((both.rac > 1.5) & (both.rau < 1.5)).sum()
    assert uncoupled_only > 0
    assert coupled_only == 0

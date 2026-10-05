"""Tests for session RPE load, monotony and strain.

Expected values here are computed by hand from Foster's definitions, not read
off what the implementation happened to produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acwr import daily_load
from acwr.foster import monotony, srpe_load, strain, weekly_blocks


def week(values: list[float], start: str = "2026-01-05") -> pd.Series:
    """Seven days starting on a Monday."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


# --------------------------------------------------------------------------
# srpe_load
# --------------------------------------------------------------------------


def test_srpe_load_multiplies_duration_by_rating():
    out = srpe_load(pd.Series([60.0, 30.0, 90.0]), pd.Series([5.0, 8.0, 4.0]))
    assert out.tolist() == [300.0, 240.0, 360.0]
    assert out.name == "load"


def test_srpe_load_accepts_a_frame():
    frame = pd.DataFrame({"duration_min": [60.0], "rpe": [7.0]})
    assert srpe_load(frame).tolist() == [420.0]


def test_srpe_load_rejects_a_frame_missing_a_column():
    with pytest.raises(KeyError, match="duration_min"):
        srpe_load(pd.DataFrame({"rpe": [7.0]}))


def test_srpe_load_requires_rpe_when_given_a_series():
    with pytest.raises(ValueError, match="rpe is required"):
        srpe_load(pd.Series([60.0]))


def test_srpe_load_rejects_mismatched_indexes():
    a = pd.Series([60.0], index=[0])
    b = pd.Series([7.0], index=[1])
    with pytest.raises(ValueError, match="share an index"):
        srpe_load(a, b)


@pytest.mark.parametrize(
    ("dur", "rpe", "match"),
    [([-1.0], [5.0], "duration cannot be negative"), ([60.0], [-5.0], "RPE cannot")],
)
def test_srpe_load_rejects_negatives(dur, rpe, match):
    with pytest.raises(ValueError, match=match):
        srpe_load(pd.Series(dur), pd.Series(rpe))


# --------------------------------------------------------------------------
# monotony
# --------------------------------------------------------------------------


def test_monotony_matches_the_hand_computation():
    """The mean divided by the standard deviation, over seven trailing days."""
    load = week([0.0, 80.0, 40.0, 0.0, 90.0, 60.0, 80.0])
    expected = load.mean() / load.std(ddof=1)
    assert float(monotony(load).iloc[-1]) == pytest.approx(float(expected))
    assert float(monotony(load).iloc[-1]) == pytest.approx(1.3207, abs=1e-4)


def test_ddof_changes_monotony_by_the_square_root_of_n_over_n_minus_one():
    """The two conventions differ by a known factor, so state which you used.

    ``sd(ddof=1) = sd(ddof=0) * sqrt(n / (n - 1))``, so monotony computed with
    the sample standard deviation is *lower* by ``sqrt((n - 1) / n)``. For a
    seven day window that is about 7 percent, which is more than enough to move
    a week across the 2.0 figure applied practice quotes.
    """
    load = week([0.0, 80.0, 40.0, 0.0, 90.0, 60.0, 80.0])
    sample = float(monotony(load, ddof=1).iloc[-1])
    population = float(monotony(load, ddof=0).iloc[-1])
    assert sample < population
    assert sample / population == pytest.approx(np.sqrt(6 / 7))


def test_a_perfectly_even_week_has_infinite_monotony():
    """Zero spread, positive mean. The infinity is the correct answer."""
    load = week([50.0] * 7)
    assert np.isinf(float(monotony(load).iloc[-1]))


def test_a_week_of_complete_rest_is_undefined_rather_than_infinite():
    load = week([0.0] * 7)
    assert np.isnan(float(monotony(load).iloc[-1]))


def test_rest_days_must_count_as_zero_not_be_dropped():
    """Dropping the zeros removes the variability monotony exists to measure."""
    with_rest = week([0.0, 60.0, 0.0, 60.0, 0.0, 60.0, 0.0])
    only_training_days = pd.Series(
        [60.0, 60.0, 60.0], index=pd.date_range("2026-01-05", periods=3, freq="D")
    )
    # By hand: four zeros and three 60s. mean = 180/7 = 25.7143. The squared
    # deviations are 4 * 25.7143**2 + 3 * 34.2857**2 = 6171.43, over 6 degrees
    # of freedom gives sd = 32.0710, so monotony = 25.7143 / 32.0710 = 0.8018.
    assert float(monotony(with_rest).iloc[-1]) == pytest.approx(0.8018, abs=1e-4)
    assert np.isinf(float(monotony(only_training_days, window=3).iloc[-1]))


def test_monotony_is_trailing_and_does_not_use_the_future():
    load = week([10.0] * 6 + [1000.0])
    rolled = monotony(load, window=3)
    before_the_spike = float(rolled.iloc[-2])
    truncated = monotony(load.iloc[:-1], window=3)
    assert before_the_spike == pytest.approx(float(truncated.iloc[-1]), nan_ok=True)


def test_monotony_warms_up_for_a_full_window_by_default():
    load = week([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    out = monotony(load, window=7)
    assert out.iloc[:6].isna().all()
    assert out.iloc[6:].notna().all()


@pytest.mark.parametrize(("window", "match"), [(1, "at least 2 days"), (0, "at least")])
def test_monotony_rejects_a_window_too_short_for_a_spread(window, match):
    with pytest.raises(ValueError, match=match):
        monotony(week([1.0] * 7), window=window)


def test_monotony_rejects_an_unsupported_ddof():
    with pytest.raises(ValueError, match="ddof must be 0 or 1"):
        monotony(week([1.0] * 7), ddof=2)


def test_monotony_rejects_a_gapped_index():
    gapped = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-05"]))
    with pytest.raises(ValueError, match="complete daily series"):
        monotony(gapped)


# --------------------------------------------------------------------------
# strain
# --------------------------------------------------------------------------


def test_strain_is_the_weekly_total_times_monotony():
    load = week([0.0, 80.0, 40.0, 0.0, 90.0, 60.0, 80.0])
    expected = load.sum() * (load.mean() / load.std(ddof=1))
    assert float(strain(load).iloc[-1]) == pytest.approx(float(expected))


def test_two_weeks_of_equal_volume_differ_in_strain_by_how_evenly_spread_they_are():
    """The whole point of strain: same kilometres, different cost."""
    even = week([40.0] * 7)
    lumpy = week([0.0, 0.0, 140.0, 0.0, 0.0, 140.0, 0.0])
    assert even.sum() == lumpy.sum() == 280.0
    assert np.isinf(float(strain(even).iloc[-1]))
    # By hand: mean 40, squared deviations 5 * 40**2 + 2 * 100**2 = 28000,
    # over 6 degrees of freedom gives sd = 68.3130. Monotony is 0.58554 and
    # strain is 280 * 0.58554 = 163.95.
    assert float(strain(lumpy).iloc[-1]) == pytest.approx(163.95, abs=0.01)


def test_strain_scales_linearly_with_the_load_unit():
    """Strain has no natural scale, so comparing across units is meaningless."""
    km = week([0.0, 8.0, 4.0, 0.0, 9.0, 6.0, 8.0])
    metres = km * 1000.0
    ratio = float(strain(metres).iloc[-1]) / float(strain(km).iloc[-1])
    assert ratio == pytest.approx(1000.0)


# --------------------------------------------------------------------------
# weekly_blocks
# --------------------------------------------------------------------------


def test_weekly_blocks_returns_one_row_per_calendar_week():
    idx = pd.date_range("2026-01-05", periods=14, freq="D")  # Monday to Sunday
    load = pd.Series([10.0, 0.0, 20.0, 10.0, 0.0, 30.0, 10.0] * 2, index=idx)
    out = weekly_blocks(load)
    assert list(out.columns) == [
        "n_days",
        "n_observed",
        "total",
        "mean",
        "sd",
        "monotony",
        "strain",
    ]
    assert len(out) == 2
    assert out["total"].tolist() == [80.0, 80.0]
    assert out["monotony"].iloc[0] == pytest.approx(out["monotony"].iloc[1])


def test_moving_the_week_boundary_changes_the_answer():
    """Which day the week ends on is a real choice, not a formatting detail."""
    idx = pd.date_range("2026-01-05", periods=21, freq="D")
    rng = np.random.default_rng(3)
    load = pd.Series(rng.gamma(2.0, 6.0, size=21), index=idx)
    sunday = weekly_blocks(load, week_ends="W-SUN")["monotony"].dropna()
    saturday = weekly_blocks(load, week_ends="W-SAT")["monotony"].dropna()
    assert not np.allclose(sunday.to_numpy()[:2], saturday.to_numpy()[:2])


def test_calendar_weeks_and_the_rolling_window_agree_on_an_aligned_week_end():
    """They answer the same question on the day the two windows coincide."""
    idx = pd.date_range("2026-01-05", periods=7, freq="D")
    load = pd.Series([12.0, 0.0, 18.0, 9.0, 0.0, 25.0, 14.0], index=idx)
    block = float(weekly_blocks(load).loc["2026-01-11", "monotony"])
    rolled = float(monotony(load, window=7).loc["2026-01-11"])
    assert block == pytest.approx(rolled)


def test_foster_metrics_accept_the_output_of_daily_load():
    activities = pd.DataFrame(
        {"date": ["2026-03-02", "2026-03-02", "2026-03-06"], "load": [30.0, 20.0, 45.0]}
    )
    load = daily_load(activities)
    out = monotony(load, window=3)
    assert isinstance(out, pd.Series)
    assert out.index.equals(load.index)


def test_a_week_with_an_unobserved_day_is_nan_throughout():
    """Resample skips NaN, which would otherwise report a 4-day week as a week.

    The rolling :func:`monotony` already returns NaN for such a week, and the
    two paths disagreeing would be worse than either answer on its own.
    """
    activities = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-06", "2026-01-10", "2026-01-11"],
            "load": [10.0, 20.0, 25.0, 15.0],
        }
    )
    load = daily_load(activities, missing="na")
    row = weekly_blocks(load).loc["2026-01-11"]
    assert int(row["n_days"]) == 7
    assert int(row["n_observed"]) == 4
    assert np.isnan(row["total"])
    assert np.isnan(row["monotony"])
    assert np.isnan(row["strain"])
    assert np.isnan(float(monotony(load, 7).loc["2026-01-11"]))


def test_a_week_with_no_observations_is_nan_not_zero():
    """`Resampler.sum()` defaults to min_count=0, which makes an empty week 0."""
    activities = pd.DataFrame(
        {"date": ["2026-01-05", "2026-01-19"], "load": [10.0, 10.0]}
    )
    load = daily_load(activities, missing="na")
    row = weekly_blocks(load).loc["2026-01-18"]
    assert int(row["n_observed"]) == 0
    assert np.isnan(row["total"])


def test_the_default_zero_fill_never_suppresses_a_week():
    """Under missing="zero" every day is observed, so nothing is NaN-ed out."""
    activities = pd.DataFrame(
        {"date": ["2026-01-05", "2026-01-11"], "load": [10.0, 20.0]}
    )
    out = weekly_blocks(daily_load(activities))
    assert (out["n_observed"] == out["n_days"]).all()
    assert out["total"].notna().all()


def test_a_partial_week_at_the_end_is_kept_and_labelled():
    idx = pd.date_range("2026-01-05", periods=10, freq="D")
    load = pd.Series(np.arange(1.0, 11.0), index=idx)
    out = weekly_blocks(load)
    assert out["n_days"].tolist() == [7, 3]
    assert out["total"].notna().all()

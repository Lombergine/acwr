"""Tests for building the daily load series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acwr import daily_load


def test_activities_on_the_same_day_are_summed():
    acts = pd.DataFrame(
        {"date": ["2026-01-01", "2026-01-01", "2026-01-01"], "load": [10.0, 20.0, 5.0]}
    )
    out = daily_load(acts)
    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(35.0)


def test_days_without_activity_are_filled_with_zero():
    acts = pd.DataFrame({"date": ["2026-01-01", "2026-01-05"], "load": [40.0, 60.0]})
    out = daily_load(acts)
    assert len(out) == 5
    assert out.tolist() == [40.0, 0.0, 0.0, 0.0, 60.0]


def test_missing_na_leaves_rest_days_missing():
    acts = pd.DataFrame({"date": ["2026-01-01", "2026-01-03"], "load": [40.0, 60.0]})
    out = daily_load(acts, missing="na")
    assert np.isnan(out.iloc[1])


def test_index_is_continuous_daily_with_no_gaps():
    acts = pd.DataFrame(
        {"date": ["2026-03-01", "2026-03-15", "2026-04-02"], "load": [1.0, 2.0, 3.0]}
    )
    out = daily_load(acts)
    gaps = out.index.to_series().diff().dropna().unique()
    assert len(gaps) == 1
    assert gaps[0] == pd.Timedelta(days=1)


def test_timestamps_are_truncated_to_calendar_days():
    acts = pd.DataFrame(
        {
            "date": ["2026-01-01 06:30:00", "2026-01-01 18:45:00"],
            "load": [25.0, 25.0],
        }
    )
    out = daily_load(acts)
    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(50.0)


def test_explicit_bounds_extend_the_series():
    acts = pd.DataFrame({"date": ["2026-01-10"], "load": [10.0]})
    out = daily_load(acts, start="2026-01-01", end="2026-01-20")
    assert len(out) == 20
    assert out.loc["2026-01-01"] == 0.0
    assert out.loc["2026-01-10"] == 10.0


def test_negative_load_is_rejected():
    acts = pd.DataFrame({"date": ["2026-01-01"], "load": [-5.0]})
    with pytest.raises(ValueError, match="non-negative"):
        daily_load(acts)


def test_empty_frame_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        daily_load(pd.DataFrame({"date": [], "load": []}))


def test_missing_column_names_the_column():
    acts = pd.DataFrame({"when": ["2026-01-01"], "load": [1.0]})
    with pytest.raises(KeyError, match="date"):
        daily_load(acts)


def test_bad_missing_mode_is_rejected():
    acts = pd.DataFrame({"date": ["2026-01-01"], "load": [1.0]})
    with pytest.raises(ValueError, match="missing must be"):
        daily_load(acts, missing="interpolate")


def test_start_after_end_is_rejected():
    acts = pd.DataFrame({"date": ["2026-01-01"], "load": [1.0]})
    with pytest.raises(ValueError, match="after end"):
        daily_load(acts, start="2026-02-01", end="2026-01-01")


def test_custom_column_names_are_honoured():
    acts = pd.DataFrame({"day": ["2026-01-01"], "srpe": [300.0]})
    out = daily_load(acts, date_col="day", load_col="srpe")
    assert out.iloc[0] == pytest.approx(300.0)
    assert out.name == "load"


def test_a_bound_carrying_a_time_of_day_behaves_like_the_date_alone():
    """An unnormalised bound would build an index that matches no activity.

    Every activity date is truncated to midnight. A `start` at 09:30 would
    produce a date_range at 09:30 on each day, reindex would match nothing,
    and under the default zero fill the whole series would come back as rest
    days with no error and no visible sign that the load had gone.
    """
    activities = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60, freq="D").astype(str),
            "load": [50.0] * 60,
        }
    )
    plain = daily_load(activities, start="2026-01-01")
    stamped = daily_load(activities, start=pd.Timestamp("2026-01-01 09:30"))
    assert float(plain.sum()) == 3000.0
    assert stamped.equals(plain)

    end_stamped = daily_load(activities, end=pd.Timestamp("2026-02-01 23:59"))
    assert end_stamped.index[-1] == pd.Timestamp("2026-02-01")


def test_a_load_value_that_cannot_be_read_raises_rather_than_becoming_rest():
    """`to_numeric(errors="coerce")` plus a zero fill is silent data loss.

    "1,200" with a thousands separator coerces to NaN and then fills to 0.0,
    so the day reads as rest and 1200 units vanish without a trace.
    """
    activities = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "load": ["500", "1,200", "600"],
        }
    )
    with pytest.raises(ValueError, match="could not be read as numbers"):
        daily_load(activities)
    with pytest.raises(ValueError, match="1,200"):
        daily_load(activities, missing="na")


def test_a_load_value_that_was_already_missing_is_left_to_the_missing_policy():
    """An absent value is ambiguous in the input, not unreadable."""
    activities = pd.DataFrame(
        {"date": ["2026-01-01", "2026-01-02"], "load": [50.0, float("nan")]}
    )
    assert daily_load(activities).tolist() == [50.0, 0.0]
    assert np.isnan(daily_load(activities, missing="na").iloc[1])

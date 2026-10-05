"""Tests for week-to-week load change.

Expected values are derived by hand from the definition, which is a plain
percentage change between consecutive blocks, not read off what the
implementation produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acwr import block_change, daily_load, week_to_week_change


def days(values: list[float], start: str = "2026-01-01") -> pd.Series:
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq="D"),
        name="load",
    )


# ---------------------------------------------------------------------------
# week_to_week_change
# ---------------------------------------------------------------------------


def test_a_twenty_percent_rise_reads_as_twenty():
    """Seventy units became eighty-four."""
    load = days([10.0] * 7 + [12.0] * 7)
    assert float(week_to_week_change(load).iloc[-1]) == pytest.approx(20.0)


def test_a_fall_is_negative():
    load = days([10.0] * 7 + [8.0] * 7)
    assert float(week_to_week_change(load).iloc[-1]) == pytest.approx(-20.0)


def test_an_unchanged_week_reads_as_zero():
    load = days([10.0] * 14)
    assert float(week_to_week_change(load).iloc[-1]) == pytest.approx(0.0)


def test_the_ratio_form_is_the_percentage_divided_by_a_hundred():
    load = days([10.0] * 7 + [12.0] * 7)
    percent = float(week_to_week_change(load).iloc[-1])
    ratio = float(week_to_week_change(load, as_percent=False).iloc[-1])
    assert percent == pytest.approx(ratio * 100.0)
    assert ratio == pytest.approx(0.2)


def test_returning_from_a_week_off_is_infinite_rather_than_a_large_number():
    """Any load at all is an infinite increase on zero.

    This is the case the metric is most often discussed around and the one it
    handles worst, which the package reports rather than hides.
    """
    load = days([0.0] * 7 + [10.0] * 7)
    assert np.isinf(float(week_to_week_change(load).iloc[-1]))


def test_two_weeks_of_rest_are_undefined_rather_than_infinite():
    load = days([0.0] * 14)
    assert np.isnan(float(week_to_week_change(load).iloc[-1]))


def test_it_warms_up_for_two_full_windows():
    load = days(list(range(1, 21)))
    out = week_to_week_change(load, window=7)
    assert out.iloc[:13].isna().all()
    assert out.iloc[13:].notna().all()


def test_the_window_length_is_adjustable():
    load = days([10.0] * 3 + [20.0] * 3)
    assert float(week_to_week_change(load, window=3).iloc[-1]) == pytest.approx(100.0)


def test_it_is_trailing_and_does_not_use_the_future():
    load = days([10.0] * 14 + [1000.0] * 7)
    before = week_to_week_change(load).iloc[13]
    truncated = week_to_week_change(load.iloc[:14]).iloc[-1]
    assert float(before) == pytest.approx(float(truncated))


def test_a_window_below_one_day_is_rejected():
    with pytest.raises(ValueError, match="at least 1 day"):
        week_to_week_change(days([10.0] * 14), window=0)


def test_a_gapped_index_is_rejected():
    gapped = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-05"]))
    with pytest.raises(ValueError, match="complete daily series"):
        week_to_week_change(gapped)


def test_it_accepts_what_daily_load_produces():
    activities = pd.DataFrame(
        {"date": ["2026-01-01", "2026-01-20"], "load": [30.0, 45.0]}
    )
    out = week_to_week_change(daily_load(activities))
    assert isinstance(out, pd.Series)
    assert out.name == "week_to_week"


# ---------------------------------------------------------------------------
# block_change
# ---------------------------------------------------------------------------


def test_calendar_weeks_compare_each_week_with_the_one_above_it():
    idx = pd.date_range("2026-01-05", periods=21, freq="D")  # a Monday
    load = pd.Series([10.0] * 7 + [11.0] * 7 + [13.0] * 7, index=idx)
    out = block_change(load)
    assert out["total"].tolist() == [70.0, 77.0, 91.0]
    assert np.isnan(out["change"].iloc[0])
    assert out["change"].iloc[1] == pytest.approx(10.0)
    assert out["change"].iloc[2] == pytest.approx(100 * (91 - 77) / 77)


def test_a_partly_observed_week_is_nan_rather_than_a_total_over_what_is_there():
    activities = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-06", "2026-01-10", "2026-01-11"],
            "load": [10.0, 20.0, 25.0, 15.0],
        }
    )
    load = daily_load(activities, missing="na")
    row = block_change(load).loc["2026-01-11"]
    assert int(row["n_days"]) == 7
    assert int(row["n_observed"]) == 4
    assert np.isnan(row["total"])
    assert np.isnan(row["change"])


def test_moving_the_week_boundary_changes_the_blocks():
    idx = pd.date_range("2026-01-05", periods=21, freq="D")
    rng = np.random.default_rng(2)
    load = pd.Series(rng.gamma(2.0, 20.0, 21), index=idx)
    sunday = block_change(load, week_ends="W-SUN")["total"].dropna()
    saturday = block_change(load, week_ends="W-SAT")["total"].dropna()
    assert not np.allclose(sunday.to_numpy()[:2], saturday.to_numpy()[:2])


def test_the_ratio_form_works_here_too():
    idx = pd.date_range("2026-01-05", periods=14, freq="D")
    load = pd.Series([10.0] * 7 + [12.0] * 7, index=idx)
    assert block_change(load, as_percent=False)["change"].iloc[1] == pytest.approx(0.2)


def test_the_two_forms_agree_on_a_boundary_aligned_week():
    """Both forms agree on the day their windows coincide.

    They differ everywhere else, which is why the choice between them has to
    be made rather than inherited.
    """
    idx = pd.date_range("2026-01-05", periods=14, freq="D")
    load = pd.Series([10.0] * 7 + [12.0] * 7, index=idx)
    rolling = float(week_to_week_change(load).loc["2026-01-18"])
    calendar = float(block_change(load)["change"].loc["2026-01-18"])
    assert rolling == pytest.approx(calendar)

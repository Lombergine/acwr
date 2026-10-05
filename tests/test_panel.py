"""Tests for running a computation across a squad.

The failure this module exists to prevent is one athlete's training leaking
into another's rolling window, which produces plausible-looking nonsense rather
than an error. Most of these tests are about that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import acwr
from acwr import by_athlete


def roster() -> pd.DataFrame:
    """Two athletes whose training would be easy to confuse with each other."""
    ana = pd.DataFrame(
        {
            "athlete": "ana",
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "load": 10.0,
        }
    )
    ben = pd.DataFrame(
        {
            "athlete": "ben",
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "load": 100.0,
        }
    )
    return pd.concat([ana, ben], ignore_index=True)


def test_each_athlete_is_computed_independently():
    """Constant load means a ratio of exactly 1.0 for each, whatever the level.

    If the two athletes were pooled, Ana's 10 would sit in a window beside
    Ben's 100 and neither ratio would be 1.0. This is the leak test.
    """
    out = by_athlete(roster(), acwr.acwr)
    settled = out.dropna(subset=["acwr"])
    assert set(settled["athlete"]) == {"ana", "ben"}
    assert np.allclose(settled["acwr"].to_numpy(), 1.0)


def test_a_spike_in_one_athlete_does_not_move_the_other():
    frame = roster()
    frame.loc[(frame.athlete == "ben") & (frame.date == "2026-02-05"), "load"] = 5000.0
    out = by_athlete(frame, acwr.acwr).set_index(["athlete", "date"])["acwr"]
    ana = out.loc["ana"].dropna()
    ben = out.loc["ben"].dropna()
    assert np.allclose(ana.to_numpy(), 1.0)
    assert ben.max() > 2.0


def test_output_is_long_with_the_athlete_and_date_columns_restored():
    out = by_athlete(roster(), acwr.acwr)
    assert list(out.columns) == ["athlete", "date", "acwr"]
    assert len(out) == 80
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_a_frame_returning_function_keeps_all_of_its_columns():
    out = by_athlete(roster(), acwr.acwr, return_components=True)
    assert list(out.columns) == ["athlete", "date", "load", "acute", "chronic", "acwr"]


def test_any_per_athlete_function_works_not_just_the_ratio():
    out = by_athlete(roster(), acwr.monotony, window=7)
    assert list(out.columns) == ["athlete", "date", "monotony"]
    # Constant load has no spread, so monotony is infinite for both.
    assert np.isinf(out["monotony"].dropna()).all()


def test_each_athlete_spans_only_their_own_history_by_default():
    """Padding a late joiner back to the squad's first day invents rest days."""
    frame = pd.DataFrame(
        {
            "athlete": ["ana", "ana", "ben"],
            "date": ["2026-01-01", "2026-01-10", "2026-03-01"],
            "load": [10.0, 10.0, 10.0],
        }
    )
    out = by_athlete(frame, acwr.acwr, acute=1, chronic=2)
    ana = out[out.athlete == "ana"]
    ben = out[out.athlete == "ben"]
    assert len(ana) == 10
    assert len(ben) == 1


def test_common_calendar_aligns_every_athlete_to_the_same_dates():
    frame = pd.DataFrame(
        {
            "athlete": ["ana", "ana", "ben"],
            "date": ["2026-01-01", "2026-01-10", "2026-01-05"],
            "load": [10.0, 10.0, 10.0],
        }
    )
    out = by_athlete(frame, acwr.acwr, acute=1, chronic=2, common_calendar=True)
    counts = out.groupby("athlete").size()
    assert counts.tolist() == [10, 10]
    per_athlete_dates = out.groupby("athlete")["date"].apply(list)
    assert per_athlete_dates["ana"] == per_athlete_dates["ben"]


def test_athletes_come_back_in_sorted_order():
    frame = pd.DataFrame(
        {
            "athlete": ["zoe", "ana", "ben"],
            "date": ["2026-01-01", "2026-01-01", "2026-01-01"],
            "load": [1.0, 1.0, 1.0],
        }
    )
    out = by_athlete(frame, acwr.acwr, acute=1, chronic=2)
    assert out["athlete"].tolist() == ["ana", "ben", "zoe"]


def test_an_athlete_with_too_little_history_stays_in_the_output_as_nan():
    """Dropping them would make a roster silently shrink."""
    frame = pd.DataFrame(
        {
            "athlete": ["ana", "ben"],
            "date": ["2026-01-01", "2026-01-01"],
            "load": [10.0, 10.0],
        }
    )
    out = by_athlete(frame, acwr.acwr)
    assert out["athlete"].tolist() == ["ana", "ben"]
    assert out["acwr"].isna().all()


def test_custom_column_names_are_honoured():
    frame = pd.DataFrame(
        {
            "runner": ["ana", "ana"],
            "day": ["2026-01-01", "2026-01-02"],
            "tss": [10.0, 20.0],
        }
    )
    out = by_athlete(
        frame,
        acwr.acwr,
        athlete_col="runner",
        date_col="day",
        load_col="tss",
        acute=1,
        chronic=2,
    )
    assert list(out.columns) == ["runner", "day", "acwr"]


@pytest.mark.parametrize("col", ["athlete", "date", "load"])
def test_a_missing_column_is_named_in_the_error(col):
    frame = roster().drop(columns=[col])
    with pytest.raises(KeyError, match=col):
        by_athlete(frame, acwr.acwr)


def test_an_empty_frame_is_rejected():
    empty = pd.DataFrame({"athlete": [], "date": [], "load": []})
    with pytest.raises(ValueError, match="empty"):
        by_athlete(empty, acwr.acwr)


def test_a_missing_athlete_identifier_is_rejected_rather_than_grouped_away():
    """A NaN key is dropped silently by groupby, which would lose an athlete."""
    frame = roster()
    frame.loc[0, "athlete"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        by_athlete(frame, acwr.acwr)


def test_missing_is_passed_through_to_daily_load():
    frame = pd.DataFrame(
        {
            "athlete": ["ana", "ana"],
            "date": ["2026-01-01", "2026-01-05"],
            "load": [10.0, 10.0],
        }
    )
    zeros = by_athlete(frame, lambda s: s, missing="zero")
    nans = by_athlete(frame, lambda s: s, missing="na")
    assert zeros["load"].tolist() == [10.0, 0.0, 0.0, 0.0, 10.0]
    assert nans["load"].isna().sum() == 3

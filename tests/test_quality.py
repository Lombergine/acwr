"""Tests for the pre-flight checks on a training log.

Each test drives one real failure mode. The unit-change and watch-left-running
cases are modelled on problems found in an actual three year Strava export,
not invented, which is the reason those two checks exist at all.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from acwr import daily_load
from acwr.quality import check_load


def days(values: list[float], start: str = "2026-01-01") -> pd.Series:
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq="D"),
        name="load",
    )


def checks(report) -> set[str]:
    return {f.check for f in report.findings}


# ---------------------------------------------------------------------------
# A clean series
# ---------------------------------------------------------------------------


def test_an_ordinary_log_passes_with_nothing_to_report():
    rng = np.random.default_rng(0)
    load = days((rng.gamma(2.0, 25.0, 200)).tolist())
    report = check_load(load)
    assert report.ok
    assert report.findings == ()
    assert report.n_days == 200
    assert report.n_training_days == 200


def test_the_report_counts_days_and_training_days_separately():
    load = days([10.0, 0.0, 10.0, 0.0] * 10)
    report = check_load(load)
    assert report.n_days == 40
    assert report.n_training_days == 20


# ---------------------------------------------------------------------------
# Errors: the computation cannot proceed meaningfully
# ---------------------------------------------------------------------------


def test_a_history_shorter_than_the_chronic_window_is_an_error():
    report = check_load(days([10.0] * 20))
    assert not report.ok
    assert "history" in checks(report)
    assert "shorter than the 28 day chronic window" in report.errors[0].message


def test_the_chronic_window_the_user_intends_is_what_gets_checked():
    load = days([10.0] * 30)
    assert check_load(load, chronic=28).ok
    assert not check_load(load, chronic=42).ok


def test_a_log_with_no_training_at_all_is_an_error():
    report = check_load(days([0.0] * 60))
    assert not report.ok
    assert "no training" in checks(report)


def test_a_negative_load_is_an_error_and_the_days_are_named():
    values = [10.0] * 60
    values[30] = -5.0
    report = check_load(days(values))
    assert not report.ok
    finding = next(f for f in report.findings if f.check == "negative load")
    assert finding.dates == (pd.Timestamp("2026-01-31"),)


# ---------------------------------------------------------------------------
# Warnings: the answer changes but remains an answer
# ---------------------------------------------------------------------------


def test_a_change_of_recording_unit_is_caught():
    """Half a year in kilometres and half in metres, which nothing else flags.

    No rolling window survives a thousandfold step, and nothing about the
    output looks wrong afterwards, which is what makes this worth a check.
    """
    load = days([10.0] * 100 + [10000.0] * 100)
    report = check_load(load)
    assert "suspected unit change" in checks(report)
    message = next(
        f.message for f in report.findings if f.check == "suspected unit change"
    )
    assert "factor of 1000" in message


def test_a_plausible_change_in_fitness_is_not_mistaken_for_a_unit_change():
    """Doubling across a season is training. A thousandfold is a unit."""
    load = days([20.0] * 100 + [40.0] * 100)
    assert "suspected unit change" not in checks(check_load(load))


def test_a_single_implausible_day_is_reported_as_an_outlier():
    """A watch left running: 900 against a typical 10."""
    values = [10.0] * 100
    values[50] = 900.0
    report = check_load(days(values))
    assert "outlier days" in checks(report)
    finding = next(f for f in report.findings if f.check == "outlier days")
    assert pd.Timestamp("2026-02-20") in finding.dates


def test_a_long_run_is_not_an_outlier():
    """Training load is right-skewed, and the check has to survive that.

    A rule written against the median flags every athlete's longest run, which
    is how a check gets ignored. The comparison is against the upper tail.
    """
    rng = np.random.default_rng(0)
    load = days(rng.gamma(2.0, 25.0, 400).tolist())
    assert "outlier days" not in checks(check_load(load))


def test_the_outlier_threshold_is_adjustable():
    """The arithmetic is easy to follow on a flat log.

    The 95th percentile is the flat value, so 25 sits above twice 10 and below
    three times it.
    """
    values = [10.0] * 100
    values[50] = 25.0
    assert "outlier days" not in checks(check_load(days(values)))
    assert "outlier days" in checks(check_load(days(values), outlier_factor=2.0))


def test_a_long_break_is_reported_with_the_day_it_ends():
    load = days([10.0] * 60 + [0.0] * 30 + [10.0] * 60)
    report = check_load(load)
    assert "break" in checks(report)
    finding = next(f for f in report.findings if f.check == "break")
    assert "30 days" in finding.message
    assert finding.dates == (pd.Timestamp("2026-03-31"),)


def test_ordinary_rest_days_are_not_a_break():
    load = days([10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0] * 20)
    assert "break" not in checks(check_load(load))


def test_unobserved_days_are_distinguished_from_rest_days():
    activities = pd.DataFrame(
        {"date": ["2026-01-01", "2026-03-01"], "load": [10.0, 10.0]}
    )
    zeros = check_load(daily_load(activities, missing="zero"))
    nans = check_load(daily_load(activities, missing="na"))
    assert "unobserved days" not in checks(zeros)
    assert "unobserved days" in checks(nans)


def test_an_extreme_dynamic_range_is_flagged_because_the_sums_lose_precision():
    load = days([1e-9] * 50 + [1e6] * 150)
    assert "extreme dynamic range" in checks(check_load(load))


# ---------------------------------------------------------------------------
# Notes: things to keep in mind when reading the output
# ---------------------------------------------------------------------------


def test_a_mostly_rest_log_is_noted_without_being_an_error():
    load = days(([10.0] + [0.0] * 4) * 20)
    report = check_load(load)
    assert report.ok
    assert "mostly rest" in checks(report)


def test_a_log_with_barely_more_than_the_warm_up_is_noted():
    report = check_load(days([10.0] * 40))
    assert report.ok
    assert "few usable days" in checks(report)


def test_a_perfectly_constant_log_is_noted():
    assert "constant load" in checks(check_load(days([10.0] * 60)))


# ---------------------------------------------------------------------------
# The report object
# ---------------------------------------------------------------------------


def test_errors_and_warnings_are_separable_and_ok_tracks_only_errors():
    load = days([10.0] * 10 + [0.0] * 30 + [10.0] * 10)
    report = check_load(load, chronic=28)
    assert any(f.severity == "warning" for f in report.findings)
    assert report.ok is (len(report.errors) == 0)
    assert set(report.errors) | set(report.warnings) <= set(report.findings)


def test_the_report_prints_something_a_person_can_read():
    text = str(check_load(days([10.0] * 20)))
    assert "Load quality" in text
    assert "ERROR" in text


def test_a_clean_report_says_so():
    rng = np.random.default_rng(1)
    assert "nothing to report" in str(
        check_load(days(rng.gamma(2.0, 25.0, 200).tolist()))
    )


def test_a_gapped_index_is_rejected_before_any_check_runs():
    gapped = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-05"]))
    with pytest.raises(ValueError, match="complete daily series"):
        check_load(gapped)


def test_findings_are_immutable_so_a_report_cannot_be_edited_after_the_fact():
    report = check_load(days([10.0] * 20))
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.findings[0].severity = "note"  # type: ignore[misc]

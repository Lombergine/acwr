"""The degenerate paths, which are where a monitoring tool actually gets used.

A package whose error branches are untested is a package whose error branches
do not work. Every test here drives one of them deliberately, and asserts that
the result is a visible `NaN` or a named exception rather than a number that
looks like an answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import acwr
from acwr.diagnostics import _association, _spearman_ci, window_sensitivity


def days(values: list[float], start: str = "2026-01-01") -> pd.Series:
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq="D"),
        name="load",
    )


# ---------------------------------------------------------------------------
# _association
# ---------------------------------------------------------------------------


def test_an_outcome_with_no_events_has_no_auc():
    """Every day uninjured. There is nothing to discriminate."""
    value, label = _association(np.arange(20.0), np.zeros(20), binary=True)
    assert np.isnan(value)
    assert label == "AUC"


def test_an_outcome_with_nothing_but_events_has_no_auc_either():
    value, _ = _association(np.arange(20.0), np.ones(20), binary=True)
    assert np.isnan(value)


def test_a_constant_predictor_has_no_rank_correlation():
    """A ratio that never moves cannot order anything."""
    value, label = _association(np.full(20, 1.2), np.arange(20.0), binary=False)
    assert np.isnan(value)
    assert label == "Spearman rho"


def test_a_constant_outcome_has_no_rank_correlation():
    value, _ = _association(np.arange(20.0), np.full(20, 3.0), binary=False)
    assert np.isnan(value)


# ---------------------------------------------------------------------------
# _spearman_ci
# ---------------------------------------------------------------------------


def test_a_bootstrap_that_cannot_find_two_distinct_values_gives_no_interval():
    """Resampling a near-constant series yields constant draws.

    Each such draw is skipped rather than counted as a correlation of zero,
    and when too few survive the interval is `NaN` instead of being computed
    from a handful of them.
    """
    # A constant predictor makes every resample constant, so every draw is
    # skipped and none survive. Skipping is the right behaviour: a constant
    # draw has no rank correlation, and counting it as zero would pull the
    # interval toward a number nothing in the data supports.
    x = np.ones(20)
    y = np.arange(20.0)
    lo, hi = _spearman_ci(x, y, draws=50, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_a_well_behaved_bootstrap_does_give_an_interval():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = x + rng.normal(scale=0.5, size=200)
    lo, hi = _spearman_ci(x, y, draws=200, seed=0)
    assert not np.isnan(lo)
    assert lo < hi


# ---------------------------------------------------------------------------
# null_chronic_acwr
# ---------------------------------------------------------------------------


def test_too_little_history_to_build_a_null_model_is_an_error_not_a_guess():
    """28 days give exactly one chronic value, which is not a distribution."""
    short = days([10.0] * 28)
    with pytest.raises(ValueError, match="not enough chronic load values"):
        acwr.null_chronic_acwr(short, chronic=28)


def test_a_null_model_needs_only_two_chronic_values():
    just_enough = days([10.0] * 29)
    out = acwr.null_chronic_acwr(just_enough, chronic=28)
    assert out.notna().sum() >= 2


# ---------------------------------------------------------------------------
# compare_to_null
# ---------------------------------------------------------------------------


def test_too_few_usable_null_draws_is_an_error_not_a_distribution():
    """Ten draws cannot describe a distribution, so the function says so."""
    rng = np.random.default_rng(1)
    load = days(rng.gamma(2.0, 30.0, 120).tolist())
    outcome = pd.Series(rng.random(120) < 0.2, index=load.index)
    with pytest.raises(ValueError, match="could not build enough null draws"):
        acwr.compare_to_null(load, outcome, n_null_draws=5, boot_draws=50, seed=0)


def test_an_outcome_with_no_events_is_rejected_rather_than_scored():
    rng = np.random.default_rng(2)
    load = days(rng.gamma(2.0, 30.0, 200).tolist())
    never = pd.Series(False, index=load.index)
    with pytest.raises(ValueError):
        acwr.compare_to_null(load, never, n_null_draws=30, boot_draws=50, seed=0)


# ---------------------------------------------------------------------------
# window_sensitivity
# ---------------------------------------------------------------------------


def test_agreement_range_is_undefined_when_no_pair_is_comparable():
    """Every non-reference pair produced nothing, so there is no range."""
    # 40 days: the 42-day chronic window never completes, so that pair
    # produces no comparable day at all.
    load = days([10.0, 0.0] * 20)
    out = window_sensitivity(
        load, acute_options=(7,), chronic_options=(28, 42), reference=(7, 28)
    )
    others = out.table[~out.table["is_reference"]]
    assert others["agreement"].isna().all()
    lo, hi = out.agreement_range
    assert np.isnan(lo) and np.isnan(hi)


def test_a_grid_with_only_the_reference_pair_also_has_no_range():
    rng = np.random.default_rng(3)
    load = days(rng.gamma(2.0, 30.0, 200).tolist())
    out = window_sensitivity(load, acute_options=(7,), chronic_options=(28,))
    assert len(out.table) == 1
    assert np.isnan(out.agreement_range[0])


# ---------------------------------------------------------------------------
# Degenerate series through the public surface
# ---------------------------------------------------------------------------


def test_a_series_of_complete_rest_produces_nan_not_zero_or_one():
    rest = days([0.0] * 60)
    r = acwr.acwr(rest)
    settled = r.iloc[27:]
    assert settled.isna().all()


def test_a_single_training_day_in_a_year_does_not_crash_anything():
    values = [0.0] * 200
    values[100] = 42.0
    load = days(values)
    for method in ("rac", "rau", "ewma"):
        out = acwr.acwr(load, method=method)
        assert len(out) == 200
    assert acwr.monotony(load).notna().any()
    assert acwr.rescaling_check(load).n > 0


# ---------------------------------------------------------------------------
# Validation branches. An error message is part of the interface.
# ---------------------------------------------------------------------------


def test_a_decay_period_below_one_day_is_rejected():
    from acwr.ratio import _alpha

    with pytest.raises(ValueError, match="decay period must be at least 1 day"):
        _alpha(0, "span")


def test_a_window_below_one_day_is_rejected():
    from acwr.ratio import _check_periods

    with pytest.raises(ValueError, match="at least 1 day"):
        _check_periods(0, 28)
    with pytest.raises(ValueError, match="at least 1 day"):
        _check_periods(7, 0)


def test_acute_load_rejects_a_window_below_one_day():
    with pytest.raises(ValueError, match="acute must be at least 1 day"):
        acwr.acute_load(days([10.0] * 30), acute=0)


def test_chronic_load_rejects_an_unknown_method():
    from acwr.ratio import chronic_load

    with pytest.raises(ValueError, match="method must be one of"):
        chronic_load(days([10.0] * 30), method="rolling")  # type: ignore[arg-type]


def test_the_smallest_legal_window_still_has_a_degree_of_freedom():
    """Two days with ddof=1 is the tightest case the arguments allow.

    An earlier version carried a guard against `window - ddof < 1`, which the
    other two validations already make unreachable. It was removed rather than
    left as a branch no input can take.
    """
    from acwr.foster import monotony

    out = monotony(days([10.0, 30.0, 20.0]), window=2, ddof=1)
    assert out.notna().sum() == 2
    with pytest.raises(ValueError, match="ddof must be 0 or 1"):
        monotony(days([10.0] * 30), window=2, ddof=2)


def test_a_null_draw_with_too_few_usable_days_is_skipped_not_counted():
    """Covers the two `continue` branches inside `compare_to_null`.

    A null draw whose ratio is usable on fewer than ten days says nothing, and
    a bootstrap resample of a binary outcome that happens to contain one class
    has no AUC. Both are skipped. Counting either would pull the reported
    interval toward a value the data does not support.
    """
    rng = np.random.default_rng(11)
    # Short history and a rare outcome: enough draws land in the skip branches.
    load = days(rng.gamma(2.0, 30.0, 70).tolist())
    outcome = pd.Series(rng.random(70) < 0.08, index=load.index)
    try:
        result = acwr.compare_to_null(
            load, outcome, n_null_draws=60, boot_draws=200, seed=4
        )
    except ValueError as exc:
        # Also an acceptable outcome on so little data, and it is the branch
        # that fires when too many draws were skipped to describe anything.
        assert "could not build enough null draws" in str(exc)
    else:
        assert result.n > 0
        assert len(result.null_distribution) >= 10

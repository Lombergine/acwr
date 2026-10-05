"""Tests for the diagnostics, including the Impellizzeri null-model check.

The headline test here is
:func:`test_random_chronic_denominator_performs_about_as_well`, which
reproduces the central result of Impellizzeri et al. (2021) on synthetic data
where injury is generated from acute load alone. If the package's own machinery
could not demonstrate that, the diagnostic would not be worth shipping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acwr import (
    acwr,
    compare_to_null,
    daily_load,
    null_chronic_acwr,
    rescaling_check,
    threshold_from_history,
    window_sensitivity,
)


def synthetic_load(n: int = 500, seed: int = 0) -> pd.Series:
    """Plausible training: a weekly rhythm, a long block structure, and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    weekly = 20.0 * np.sin(2 * np.pi * t / 7.0)
    block = 15.0 * np.sin(2 * np.pi * t / 90.0)
    base = 60.0 + weekly + block + rng.normal(0, 12.0, n)
    base = np.clip(base, 0.0, None)
    base[rng.random(n) < 0.12] = 0.0  # rest days
    return pd.Series(
        base, index=pd.date_range("2025-01-01", periods=n, freq="D"), name="load"
    )


# --------------------------------------------------------------------------
# rescaling_check
# --------------------------------------------------------------------------


def test_rescaling_check_reports_a_substantial_correlation_on_real_looking_training():
    """Most of the ratio's day-to-day ordering comes from its own numerator.

    On training with a weekly rhythm, a block structure and rest days, the
    Spearman between the ratio and the acute load alone sits around 0.6. That is
    not the whole story, which is why the denominator is not literally useless,
    but it is most of it.
    """
    res = rescaling_check(synthetic_load())
    assert res.n > 400
    assert res.spearman > 0.5
    lo, hi = res.spearman_ci95
    assert lo < res.spearman < hi


def test_a_constant_denominator_makes_the_ratio_a_copy_of_the_numerator():
    """The limiting case, stated exactly.

    When the chronic load does not vary, dividing by it is a change of units and
    nothing else, so the ratio preserves the ordering of the acute load
    perfectly. Spearman is exactly 1. This is the arithmetic that the critique
    rests on.
    """
    load = synthetic_load(600, seed=5)
    frame = acwr(load, return_components=True)
    null = null_chronic_acwr(load, kind="constant")
    joined = (
        pd.DataFrame({"acute": frame["acute"], "ratio": null})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    from scipy import stats as _st

    rho = _st.spearmanr(joined["acute"], joined["ratio"]).statistic
    assert rho == pytest.approx(1.0)


def test_smoother_load_rescales_more_than_structured_load():
    """The flatter the denominator, the more the ratio is just its numerator."""
    rng = np.random.default_rng(3)
    flat = pd.Series(
        100.0 + rng.normal(0, 3.0, 600),
        index=pd.date_range("2025-01-01", periods=600, freq="D"),
        name="load",
    )
    structured = synthetic_load(600, seed=3)
    assert rescaling_check(flat).spearman > rescaling_check(structured).spearman


def test_rescaling_check_needs_enough_days_and_says_so():
    short = pd.Series(
        [50.0] * 20, index=pd.date_range("2026-01-01", periods=20, freq="D")
    )
    with pytest.raises(ValueError, match="usable days"):
        rescaling_check(short, acute=7, chronic=28)


def test_rescaling_result_renders_without_error():
    res = rescaling_check(synthetic_load(300))
    assert "Spearman" in str(res)


# --------------------------------------------------------------------------
# null_chronic_acwr
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["shuffle", "normal", "constant"])
def test_null_series_is_aligned_and_mostly_finite(kind):
    load = synthetic_load(300)
    null = null_chronic_acwr(load, kind=kind)
    assert null.index.equals(load.index)
    assert null.name == "acwr_null"
    assert null.replace([np.inf, -np.inf], np.nan).notna().sum() > 250


def test_shuffle_preserves_the_chronic_distribution_exactly():
    """The denominator keeps its values and loses only their order."""
    load = synthetic_load(300)
    from acwr.ratio import chronic_load

    real = chronic_load(load).dropna().to_numpy()
    null = null_chronic_acwr(load, kind="shuffle", seed=1)
    acute = acwr(load, return_components=True)["acute"]
    # null = acute / fake, so acute / null recovers the fake denominator
    implied = (acute / null).dropna().to_numpy()
    assert np.allclose(np.sort(implied), np.sort(real))


def test_constant_null_uses_the_mean_chronic_load():
    load = synthetic_load(300)
    from acwr.ratio import chronic_load

    mean_chronic = chronic_load(load).dropna().mean()
    null = null_chronic_acwr(load, kind="constant")
    acute = acwr(load, return_components=True)["acute"]
    joined = pd.DataFrame({"a": acute, "n": null}).dropna()
    implied = joined["a"] / joined["n"]
    assert np.allclose(implied.to_numpy(), mean_chronic)


def test_null_draws_are_reproducible_from_a_seed():
    load = synthetic_load(300)
    a = null_chronic_acwr(load, seed=7)
    b = null_chronic_acwr(load, seed=7)
    pd.testing.assert_series_equal(a, b)


def test_different_seeds_give_different_nulls():
    load = synthetic_load(300)
    a = null_chronic_acwr(load, seed=1).dropna()
    b = null_chronic_acwr(load, seed=2).dropna()
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_unknown_null_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        null_chronic_acwr(synthetic_load(200), kind="gaussian")


# --------------------------------------------------------------------------
# the Impellizzeri result
# --------------------------------------------------------------------------


def test_random_chronic_denominator_performs_about_as_well():
    """Reproduce the central finding of Impellizzeri et al. (2021).

    Injury here is generated from the acute load alone, with no dependence on
    chronic load whatsoever. Under that data-generating process the chronic
    denominator can only add noise, so a ratio built on a random denominator
    should score about as well as the real one. The paper's conclusion was that
    real injury data behaves the same way.
    """
    load = synthetic_load(900, seed=11)
    a = acwr(load, return_components=True)["acute"]

    rng = np.random.default_rng(5)
    p = ((a - a.min()) / (a.max() - a.min())).fillna(0.0)
    outcome = pd.Series(
        (rng.random(len(p)) < 0.02 + 0.30 * p).astype(int), index=load.index
    )

    res = compare_to_null(load, outcome, kind="shuffle", n_null_draws=120, seed=2)

    assert res.measure == "AUC"
    assert res.n_events > 20

    # Both are above chance, because both share the real numerator.
    assert res.real > 0.55
    assert res.null > 0.55

    # The real ratio does not beat its own null. This is the Impellizzeri
    # result: the chronic denominator is not doing the work.
    assert not res.real_beats_null

    # It is in fact slightly worse, and the reason is worth stating. Injury
    # here depends on acute load. The coupled ratio divides acute load by a
    # window that contains that same acute load, which compresses exactly the
    # spikes the outcome responds to. A random denominator does no such
    # damping, so it preserves more of the numerator's signal than the real
    # denominator does.
    assert res.null > res.real


def test_real_ratio_wins_when_injury_actually_depends_on_the_ratio():
    """The diagnostic is not rigged to always exonerate the null.

    Here injury depends on the genuine ratio, so a random denominator destroys
    the signal and the real statistic should sit outside the null interval.
    """
    load = synthetic_load(900, seed=21)
    r = acwr(load).replace([np.inf, -np.inf], np.nan)

    rng = np.random.default_rng(9)
    z = ((r - r.mean()) / r.std()).fillna(0.0)
    outcome = pd.Series(
        (rng.random(len(z)) < 1 / (1 + np.exp(-(-2.6 + 1.9 * z)))).astype(int),
        index=load.index,
    )

    res = compare_to_null(load, outcome, kind="shuffle", n_null_draws=120, seed=4)
    assert res.real > res.null
    assert res.real_beats_null
    assert res.real_outside_null


def test_continuous_outcome_switches_to_spearman():
    load = synthetic_load(600, seed=31)
    rng = np.random.default_rng(13)
    outcome = pd.Series(rng.normal(0, 1, len(load)), index=load.index)
    res = compare_to_null(load, outcome, n_null_draws=40, seed=1)
    assert res.measure == "Spearman rho"


def test_comparison_needs_enough_aligned_observations():
    # 30 days with a 28 day chronic window leaves 3 usable rows, below the floor
    load = synthetic_load(30)
    outcome = pd.Series(np.ones(30, dtype=int), index=load.index)
    with pytest.raises(ValueError, match="aligned, complete observations"):
        compare_to_null(load, outcome, acute=7, chronic=28)


def test_comparison_renders_a_readable_summary():
    load = synthetic_load(600, seed=41)
    rng = np.random.default_rng(17)
    outcome = pd.Series((rng.random(len(load)) < 0.07).astype(int), index=load.index)
    res = compare_to_null(load, outcome, n_null_draws=40, seed=1)
    text = str(res)
    assert "real ACWR" in text
    assert "null" in text
    assert str(res.n_events) in text


def test_null_distribution_is_returned_for_the_user_to_inspect():
    load = synthetic_load(600, seed=51)
    rng = np.random.default_rng(19)
    outcome = pd.Series((rng.random(len(load)) < 0.07).astype(int), index=load.index)
    res = compare_to_null(load, outcome, n_null_draws=60, seed=1)
    assert isinstance(res.null_distribution, np.ndarray)
    assert len(res.null_distribution) == res.n_null_draws


# --------------------------------------------------------------------------
# the whole path, from a table of activities
# --------------------------------------------------------------------------


def test_end_to_end_from_an_activity_table():
    rng = np.random.default_rng(99)
    dates = pd.date_range("2025-01-01", periods=400, freq="D")
    keep = rng.random(400) > 0.2
    acts = pd.DataFrame(
        {
            "date": dates[keep],
            "load": rng.gamma(2.0, 30.0, keep.sum()),
        }
    )
    load = daily_load(acts)
    ratio = acwr(load, method="rau")
    assert ratio.dropna().between(0, 10).mean() > 0.95
    assert rescaling_check(load, method="rau").spearman > 0.5


def test_real_beats_null_is_one_sided():
    """A real ratio below its null must not report as beating it.

    An earlier version of this property was two sided, so a ratio that was
    worse than random returned True. That read as a pass.
    """
    from acwr.diagnostics import NullComparison

    worse = NullComparison(
        n=100,
        n_events=20,
        measure="AUC",
        real=0.50,
        real_ci95=(0.44, 0.56),
        null=0.65,
        null_ci95=(0.60, 0.70),
        null_kind="shuffle",
        n_null_draws=100,
        null_distribution=np.array([0.65]),
    )
    assert not worse.real_beats_null
    assert worse.real_outside_null
    assert "BELOW" in str(worse)

    better = NullComparison(
        n=100,
        n_events=20,
        measure="AUC",
        real=0.80,
        real_ci95=(0.74, 0.86),
        null=0.55,
        null_ci95=(0.50, 0.60),
        null_kind="shuffle",
        n_null_draws=100,
        null_distribution=np.array([0.55]),
    )
    assert better.real_beats_null
    assert better.real_outside_null
    assert "ABOVE" in str(better)

    inside = NullComparison(
        n=100,
        n_events=20,
        measure="AUC",
        real=0.57,
        real_ci95=(0.51, 0.63),
        null=0.56,
        null_ci95=(0.50, 0.62),
        null_kind="shuffle",
        n_null_draws=100,
        null_distribution=np.array([0.56]),
    )
    assert not inside.real_beats_null
    assert not inside.real_outside_null
    assert "INSIDE" in str(inside)


# ---------------------------------------------------------------------------
# window_sensitivity
# ---------------------------------------------------------------------------


def _lumpy_load(n: int = 800, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = rng.gamma(shape=2.0, scale=6.0, size=n)
    x[rng.random(n) < 0.35] = 0.0
    return pd.Series(x, index=pd.date_range("2024-01-01", periods=n, freq="D"))


def test_window_sensitivity_sweeps_every_valid_pair_and_skips_the_rest():
    out = window_sensitivity(
        _lumpy_load(), acute_options=(7, 14, 28), chronic_options=(21, 28)
    )
    pairs = set(zip(out.table.acute, out.table.chronic, strict=True))
    assert pairs == {(7, 21), (7, 28), (14, 21), (14, 28)}
    assert (28, 28) not in pairs  # chronic must exceed acute


def test_the_reference_pair_agrees_perfectly_with_itself():
    out = window_sensitivity(_lumpy_load())
    row = out.table[out.table.is_reference]
    assert len(row) == 1
    assert float(row.agreement.iloc[0]) == 1.0
    assert tuple(row[["acute", "chronic"]].iloc[0]) == (7, 28)


def test_agreement_is_a_jaccard_index_so_it_stays_within_zero_and_one():
    out = window_sensitivity(_lumpy_load())
    assert out.table.agreement.between(0.0, 1.0).all()
    assert out.table.flagged_fraction.between(0.0, 1.0).all()


def test_a_constant_load_flags_nothing_and_every_pair_agrees():
    """No day exceeds the threshold, so the empty sets agree trivially."""
    flat = pd.Series(
        40.0, index=pd.date_range("2024-01-01", periods=300, freq="D"), name="load"
    )
    out = window_sensitivity(flat)
    assert (out.table.n_flagged == 0).all()
    assert (out.table.agreement == 1.0).all()


def test_the_window_choice_changes_which_days_get_flagged():
    """The finding this diagnostic exists to surface."""
    out = window_sensitivity(_lumpy_load())
    lo, hi = out.agreement_range
    assert lo < 0.95, "if every pair agreed, the windows would not matter"
    assert hi <= 1.0
    spread = out.table.flagged_fraction.max() - out.table.flagged_fraction.min()
    assert spread > 0.02


def test_raising_the_threshold_flags_fewer_days():
    load = _lumpy_load()
    low = window_sensitivity(load, threshold=1.2).table
    high = window_sensitivity(load, threshold=2.0).table
    merged = low.merge(high, on=["acute", "chronic"], suffixes=("_low", "_high"))
    assert (merged.n_flagged_low >= merged.n_flagged_high).all()


def test_window_sensitivity_runs_for_every_method():
    load = _lumpy_load()
    for method in ("rac", "rau", "ewma"):
        out = window_sensitivity(load, method=method)
        assert out.method == method
        # The default grid is 4 acute by 4 chronic, and every pair is valid
        # because the longest acute window (14) is shorter than the shortest
        # chronic one (21).
        assert len(out.table) == 16


def test_a_reference_outside_the_grid_is_rejected():
    with pytest.raises(ValueError, match="inside the swept grid"):
        window_sensitivity(
            _lumpy_load(), acute_options=(7,), chronic_options=(28,), reference=(9, 28)
        )


def test_an_inverted_reference_is_rejected():
    with pytest.raises(ValueError, match="chronic window must exceed"):
        window_sensitivity(_lumpy_load(), reference=(28, 7))


def test_window_sensitivity_has_a_readable_summary():
    text = str(window_sensitivity(_lumpy_load()))
    assert "Window sensitivity" in text
    assert "Jaccard" in text
    assert "7/28" in text


def test_a_pair_sharing_no_comparable_day_reports_nan_not_perfect_agreement():
    """A chronic window longer than the series produces no days at all.

    Jaccard on two empty sets is conventionally 1.0, and reporting that here
    would make a pair that computed nothing read as being in perfect accord
    with the reference, which is the opposite of what the reader should take
    from it.
    """
    rng = np.random.default_rng(2)
    short = pd.Series(
        rng.gamma(2.0, 50.0, 41),
        index=pd.date_range("2026-01-01", periods=41, freq="D"),
    )
    out = window_sensitivity(short)
    table = out.table
    empty = table[table.n_days == 0]
    usable = table[table.n_days > 0]
    assert len(empty) > 0, "the 42-day chronic windows should produce no days"

    # No comparable day at all: undefined.
    assert empty.agreement.isna().all()
    # Comparable days, none of them flagged by either pair: that is agreement,
    # and the two cases have to stay distinguishable.
    assert (usable.n_flagged == 0).all()
    assert usable.agreement.eq(1.0).all()

    lo, hi = out.agreement_range
    assert not np.isnan(lo), "the range must skip the empty pairs, not inherit them"
    assert (lo, hi) == (1.0, 1.0)


def test_all_three_methods_are_compared_over_the_same_days():
    """`rescaling_check` is read across methods, so the n must match.

    Before the exponentially weighted path honoured `min_periods`, it was
    computed over every day while the two rolling methods warmed up for 28,
    and the three correlations were quietly not comparable.
    """
    load = _lumpy_load(400)
    counts = {m: rescaling_check(load, method=m).n for m in ("rac", "rau", "ewma")}
    assert len(set(counts.values())) == 1, counts


# ---------------------------------------------------------------------------
# threshold_from_history
# ---------------------------------------------------------------------------


def test_the_reference_threshold_is_placed_in_the_athletes_own_distribution():
    load = _lumpy_load(600)
    report = threshold_from_history(load)
    flagged = report.reference_flagged_fraction
    assert 0.0 <= flagged <= 1.0
    # The reported percentile and the flagged share are two views of one number.
    assert report.reference_percentile == pytest.approx(100 * (1 - flagged))


def test_the_percentiles_are_ordered_and_come_from_the_ratio_itself():
    load = _lumpy_load(600)
    report = threshold_from_history(load, percentiles=(50, 90, 99))
    values = [report.percentiles[p] for p in (50.0, 90.0, 99.0)]
    assert values == sorted(values)
    ratio = acwr(load).replace([np.inf, -np.inf], np.nan).dropna()
    assert report.percentiles[50.0] == pytest.approx(float(np.median(ratio)))


def test_a_higher_reference_flags_fewer_days():
    load = _lumpy_load(600)
    low = threshold_from_history(load, reference=1.2)
    high = threshold_from_history(load, reference=2.0)
    assert low.reference_flagged >= high.reference_flagged
    assert low.n == high.n


def test_infinite_days_count_as_flagged_but_do_not_enter_the_percentiles():
    """Otherwise every percentile above the first infinity is meaningless."""
    values = np.r_[np.zeros(40), np.full(200, 20.0)]
    load = pd.Series(values, index=pd.date_range("2026-01-01", periods=240, freq="D"))
    report = threshold_from_history(load, method="rau")
    assert all(np.isfinite(v) for v in report.percentiles.values())
    ratio = acwr(load, method="rau").dropna()
    assert report.reference_flagged >= int(np.isinf(ratio).sum())


def test_a_series_too_short_to_produce_a_ratio_is_an_error():
    short = pd.Series(
        [10.0] * 20, index=pd.date_range("2026-01-01", periods=20, freq="D")
    )
    with pytest.raises(ValueError, match="no usable days"):
        threshold_from_history(short)


def test_the_report_prints_the_comparison_a_coach_would_want():
    text = str(threshold_from_history(_lumpy_load(600)))
    assert "percentile" in text
    assert "their own percentiles" in text


def test_it_runs_for_every_method():
    load = _lumpy_load(400)
    for method in ("rac", "rau", "ewma"):
        report = threshold_from_history(load, method=method)
        assert report.method == method
        assert report.n > 0

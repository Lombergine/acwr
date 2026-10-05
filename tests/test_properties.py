"""Property-based tests: invariants that must hold on every input, not a fixture.

The rest of the suite checks hand-computed values on chosen examples. That
catches a wrong formula and misses the input nobody thought of. These tests
state the properties the package claims and let Hypothesis look for a
counterexample across thousands of generated training histories, including the
degenerate ones: all rest, a single session, a spike on the last day, loads
spanning nine orders of magnitude.

The most important property here is causality. Every function in this package
is meant to be usable on the day it describes, which means no output may
depend on training that has not happened yet. A centred smoothing window broke
that once already, and it broke silently, because the numbers still looked
like plausible ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import acwr
from acwr.foster import monotony, strain

SLOW = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Loads span a wide range on purpose: session RPE sits near 500, distance in
# metres near 10000, and a training stress score near 100. A package that is
# scale free has to behave on all of them.
#
# A load is either exactly zero, which is a rest day and a real observation,
# or a positive number in the range training data actually occupies: session
# RPE load runs to a couple of thousand, distance in metres to forty thousand,
# training stress score to a few hundred.
#
# The range is bounded on purpose and the bound is part of the claim. These
# invariants are exact in real arithmetic. In floating point they hold to
# about twelve significant figures on data like the above. Given the full
# float range to play with, Hypothesis drives pandas' rolling accumulators
# into relative errors as large as 0.5, by placing a window of near-zero
# values in a series that has held values ten orders of magnitude larger. The
# effect did not survive reduction to a minimal case, so no mechanism is
# claimed for it here beyond the observation. Such inputs are not training
# loads, and a package should not be asserted to be exact where its dependency
# is not.
loads = st.one_of(
    st.just(0.0),
    st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
)


@st.composite
def daily_series(draw, min_len: int = 1, max_len: int = 200) -> pd.Series:
    """Build a continuous daily load series, including degenerate ones."""
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    values = draw(st.lists(loads, min_size=n, max_size=n))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=idx, name="load")


@st.composite
def window_pair(draw) -> tuple[int, int]:
    acute = draw(st.integers(min_value=1, max_value=14))
    chronic = draw(st.integers(min_value=acute + 1, max_value=60))
    return acute, chronic


# ---------------------------------------------------------------------------
# Causality. Nothing may look at the future.
# ---------------------------------------------------------------------------


@given(load=daily_series(min_len=40), extra=st.lists(loads, min_size=1, max_size=20))
@SLOW
def test_appending_future_training_never_changes_an_earlier_value(load, extra):
    """Compute today, then append tomorrow. Today must not move.

    This is the single guarantee that makes the output usable for a decision
    rather than only for a retrospective plot, and it is the one a centred
    window quietly breaks.
    """
    longer = pd.concat(
        [
            load,
            pd.Series(
                np.asarray(extra, dtype=float),
                index=pd.date_range(
                    load.index[-1] + pd.Timedelta(days=1), periods=len(extra), freq="D"
                ),
                name="load",
            ),
        ]
    )
    for fn in (
        lambda s: acwr.acwr(s, method="rac"),
        lambda s: acwr.acwr(s, method="rau"),
        lambda s: acwr.acwr(s, method="ewma"),
        lambda s: acwr.acwr(s, smooth=5),
        lambda s: monotony(s, window=7),
        lambda s: strain(s, window=7),
    ):
        before = fn(load)
        after = fn(longer).reindex(load.index)
        np.testing.assert_allclose(
            before.to_numpy(), after.to_numpy(), rtol=1e-12, equal_nan=True
        )


@given(load=daily_series(min_len=40))
@SLOW
def test_centred_smoothing_is_the_only_thing_that_uses_the_future(load):
    """And it is opt-in, so the exception proves the default is doing work."""
    longer = pd.concat(
        [
            load,
            pd.Series(
                [1e5] * 5,
                index=pd.date_range(
                    load.index[-1] + pd.Timedelta(days=1), periods=5, freq="D"
                ),
                name="load",
            ),
        ]
    )
    centred_before = acwr.acwr(load, smooth=7, smooth_center=True)
    centred_after = acwr.acwr(longer, smooth=7, smooth_center=True).reindex(load.index)
    trailing_before = acwr.acwr(load, smooth=7)
    trailing_after = acwr.acwr(longer, smooth=7).reindex(load.index)
    np.testing.assert_allclose(
        trailing_before.to_numpy(),
        trailing_after.to_numpy(),
        rtol=1e-12,
        equal_nan=True,
    )
    # The centred one is allowed to differ. If it ever stopped differing the
    # option would be doing nothing and should be removed.
    differs = ~np.isclose(
        centred_before.to_numpy(), centred_after.to_numpy(), equal_nan=True
    )
    assume(centred_before.notna().sum() > 7)
    assert differs.any()


# ---------------------------------------------------------------------------
# The structural bound on the coupled form.
# ---------------------------------------------------------------------------


@given(load=daily_series(min_len=30), windows=window_pair())
@SLOW
def test_the_coupled_ratio_never_exceeds_chronic_over_acute(load, windows):
    acute, chronic = windows
    assume(len(load) > chronic)
    r = acwr.acwr(load, acute=acute, chronic=chronic, method="rac")
    finite = r.replace([np.inf, -np.inf], np.nan).dropna()
    assume(len(finite) > 0)
    # Exact in real arithmetic. In floating point the rolling means carry a
    # relative error that grows with the dynamic range inside the window, so
    # the bound is asserted relatively rather than to the last bit.
    assert finite.max() <= (chronic / acute) * (1 + 1e-9)


@given(load=daily_series(min_len=30), windows=window_pair())
@SLOW
def test_the_coupled_ratio_is_never_infinite_on_a_positive_numerator(load, windows):
    """The acute window is inside the chronic one, so it cannot divide by zero."""
    acute, chronic = windows
    assume(len(load) > chronic)
    frame = acwr.acwr(
        load, acute=acute, chronic=chronic, method="rac", return_components=True
    )
    trained = frame[frame["acute"] > 0]
    assert not np.isinf(trained["acwr"].to_numpy()).any()


# ---------------------------------------------------------------------------
# Coupling compresses toward 1.0.
# ---------------------------------------------------------------------------


@given(load=daily_series(min_len=40), windows=window_pair())
@SLOW
def test_coupling_pulls_the_ratio_toward_one_from_whichever_side_it_is_on(
    load, windows
):
    acute, chronic = windows
    assume(len(load) > chronic)
    rac = acwr.acwr(load, acute=acute, chronic=chronic, method="rac")
    rau = acwr.acwr(load, acute=acute, chronic=chronic, method="rau")
    acute_sum = load.rolling(acute).sum()
    joined = (
        pd.DataFrame({"rac": rac, "rau": rau, "s": acute_sum})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    joined = joined[joined["s"] > 0]
    assume(len(joined) > 0)
    above = joined[joined["rac"] > 1 + 1e-9]
    below = joined[joined["rac"] < 1 - 1e-9]
    tol = 1e-9
    assert (above["rac"] <= above["rau"] * (1 + tol) + tol).all()
    assert (below["rac"] >= below["rau"] * (1 - tol) - tol).all()


# ---------------------------------------------------------------------------
# Scale freedom. The ratio has no units.
# ---------------------------------------------------------------------------


@given(
    load=daily_series(min_len=35),
    k=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
    method=st.sampled_from(["rac", "rau", "ewma"]),
)
@SLOW
def test_the_ratio_does_not_care_what_unit_the_load_is_in(load, k, method):
    """Kilometres, metres and miles must all give the same ratio."""
    a = acwr.acwr(load, method=method)
    b = acwr.acwr(load * k, method=method)
    np.testing.assert_allclose(
        a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )


@given(
    load=daily_series(min_len=20),
    k=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
)
@SLOW
def test_monotony_does_not_care_what_unit_the_load_is_in(load, k):
    """Monotony divides a mean by a spread in the same unit, so it is a number.

    The claim is exact in real arithmetic and holds in floating point wherever
    the standard deviation is well conditioned. In a window that is almost
    constant the spread is a vanishing difference of nearly equal numbers,
    catastrophic cancellation dominates, and the computed value carries a
    relative error that is itself not scale equivariant. Those windows are
    excluded here rather than accommodated with a tolerance wide enough to
    hide a genuine defect: a real scale dependence would show up as an O(1)
    difference, not in the sixth significant figure.

    Monotony is the reciprocal of the window's coefficient of variation, so
    "almost constant" is simply "monotony is enormous", which is how the
    filter below is written.

    The surviving tolerance is 1e-6. On realistic loads the largest violation
    Hypothesis finds is around 6e-8, on windows holding a single small session
    in a series whose other days run to thousands, where the running
    accumulators pandas keeps for the rolling moments carry an absolute error
    set by the larger values. A genuine scale dependence would be O(1).
    """
    a = monotony(load, window=7).replace([np.inf, -np.inf], np.nan)
    b = monotony(load * k, window=7).replace([np.inf, -np.inf], np.nan)
    both = pd.DataFrame({"a": a, "b": b}).dropna()
    conditioned = both[(both["a"].abs() < 1e6) & (both["b"].abs() < 1e6)]
    assume(len(conditioned) > 0)
    np.testing.assert_allclose(
        conditioned["a"].to_numpy(),
        conditioned["b"].to_numpy(),
        rtol=1e-6,
        atol=1e-9,
    )


@given(
    load=daily_series(min_len=20),
    k=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
)
@SLOW
def test_strain_scales_with_the_unit_so_it_is_not_comparable_across_athletes(load, k):
    """Strain carries the unit of load, which monotony does not.

    This is why strain computed from session RPE cannot be put beside strain
    computed from kilometres, and the test exists so that nobody later
    "normalises" strain into looking comparable. Recording the same training
    in metres rather than kilometres multiplies strain by a thousand.
    """
    mono = monotony(load, window=7).replace([np.inf, -np.inf], np.nan)
    a = strain(load, window=7).replace([np.inf, -np.inf], np.nan)
    b = strain(load * k, window=7).replace([np.inf, -np.inf], np.nan)
    both = pd.DataFrame({"a": a, "b": b, "m": mono}).dropna()
    # Same conditioning filter as the monotony test: strain is a multiple of
    # monotony and inherits its ill-conditioning on an almost constant window.
    both = both[(both["a"].abs() > 1e-12) & (both["m"].abs() < 1e6)]
    assume(len(both) > 0)
    ratio = (both["b"] / both["a"]).to_numpy()
    np.testing.assert_allclose(ratio, k, rtol=1e-6)


# ---------------------------------------------------------------------------
# daily_load conserves what it is given.
# ---------------------------------------------------------------------------


@given(
    days=st.lists(st.integers(min_value=0, max_value=400), min_size=1, max_size=60),
    values=st.lists(
        st.floats(min_value=0.0, max_value=1e4, allow_nan=False),
        min_size=1,
        max_size=60,
    ),
)
@SLOW
def test_daily_load_conserves_total_load_and_leaves_no_gap(days, values):
    n = min(len(days), len(values))
    base = pd.Timestamp("2026-01-01")
    activities = pd.DataFrame(
        {
            "date": [base + pd.Timedelta(days=d) for d in days[:n]],
            "load": values[:n],
        }
    )
    out = acwr.daily_load(activities)
    assert float(out.sum()) == pytest.approx(float(activities["load"].sum()), rel=1e-9)
    assert (out >= 0).all()
    gaps = out.index.to_series().diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all()
    assert out.index[0] == activities["date"].min()
    assert out.index[-1] == activities["date"].max()


# ---------------------------------------------------------------------------
# The components really are the ratio.
# ---------------------------------------------------------------------------


@given(load=daily_series(min_len=35), method=st.sampled_from(["rac", "rau", "ewma"]))
@SLOW
def test_the_reported_components_divide_to_the_reported_ratio(load, method):
    frame = acwr.acwr(load, method=method, return_components=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        recomputed = frame["acute"] / frame["chronic"]
    np.testing.assert_allclose(
        recomputed.to_numpy(), frame["acwr"].to_numpy(), rtol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        frame["load"].to_numpy(), load.to_numpy(), rtol=1e-12, equal_nan=True
    )


# ---------------------------------------------------------------------------
# A squad is the same as its members computed one at a time.
# ---------------------------------------------------------------------------


@given(load_a=daily_series(min_len=35), load_b=daily_series(min_len=35))
@SLOW
def test_by_athlete_matches_computing_each_athlete_alone(load_a, load_b):
    """The leak test, generated rather than chosen."""
    frames = []
    for name, series in (("ana", load_a), ("ben", load_b)):
        frames.append(
            pd.DataFrame(
                {"athlete": name, "date": series.index, "load": series.to_numpy()}
            )
        )
    together = acwr.by_athlete(pd.concat(frames, ignore_index=True), acwr.acwr)
    for name, series in (("ana", load_a), ("ben", load_b)):
        alone = acwr.acwr(series)
        got = together[together.athlete == name]["acwr"].to_numpy()
        np.testing.assert_allclose(
            got, alone.to_numpy(), rtol=1e-12, equal_nan=True, err_msg=name
        )

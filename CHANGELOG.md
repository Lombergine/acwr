# Changelog

All notable changes are recorded here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `daily_load` builds a continuous daily load series from activity records,
  filling rest days with zero by default.
- `acwr` implements the three published methods: rolling average coupled
  (`rac`), rolling average uncoupled (`rau`), and exponentially weighted
  (`ewma`).
- `acute_load` and `chronic_load` expose the two halves of the ratio
  separately.
- `rescaling_check` reports how closely the ratio tracks its own numerator,
  with a bootstrap interval and no outcome variable required.
- `null_chronic_acwr` builds a ratio whose denominator has been replaced by a
  shuffle, a matched normal draw, or a constant.
- `compare_to_null` runs the Impellizzeri et al. (2021) test on user data,
  comparing the real ratio's association with an outcome against a
  null-denominator distribution.
- `monotony`, `strain`, `srpe_load` and `weekly_blocks` implement Foster
  (1998). Every definitional choice the literature leaves ambiguous is an
  explicit parameter: the degrees of freedom in the standard deviation, the
  rolling versus calendar window, and the day a training week ends on.
- `by_athlete` applies any per-athlete computation across a squad, building
  each athlete's daily series separately so that no athlete's training can
  enter another's rolling window.
- `window_sensitivity` sweeps a grid of acute and chronic window lengths and
  reports the Jaccard agreement between the days each pair flags and the days
  the conventional pair flags, putting a number on a choice that is almost
  never justified.
- `CODE_OF_CONDUCT.md`, and `AI-USAGE.md` disclosing the use of generative AI
  in developing the package, as both JOSS and pyOpenSci require.
- `ACCEPTANCE.md` auditing the package against the JOSS and pyOpenSci criteria.
- A property-based suite built on Hypothesis (`tests/test_properties.py`),
  covering causality, the structural bound, scale freedom and the independence
  of athletes in `by_athlete`. The causality property is the one that matters
  most: appending future training must never change an already-reported value.
- A documentation site built with mkdocs and mkdocstrings, published to GitHub
  Pages by `.github/workflows/docs.yml` and built with `--strict` so a broken
  reference fails CI.
- CI now checks formatting and builds both distributions through `twine check`.

### Documented
- The coupled ratio is bounded above by `chronic / acute`, which is exactly 4.0
  at the conventional windows, because the acute window is contained in the
  chronic one. Stated by Wang et al. (2019) in preprint and not claimed as new
  here; the contribution is the test,
  `test_coupled_ratio_can_never_exceed_chronic_over_acute`.
- Coupling compresses the ratio toward 1.0 from either side, so within the
  range above 1.0 that every published threshold occupies, the coupled form can
  only understate relative to the uncoupled one. Covered by
  `test_coupling_compresses_the_ratio_toward_one`. Consistent with Coyne et al.
  (2019), who report correlations of 0.88 to 0.99 between the two forms: a high
  correlation does not constrain agreement at a cut point.

### Fixed
- `daily_load` normalises `start` and `end`. A bound carrying a time of day
  built an index that matched no activity, and under the default zero fill the
  whole series came back as rest days with no error and no sign that the load
  had gone.
- `daily_load` raises on a load value that is present but unreadable, rather
  than coercing it to `NaN` and then filling it to zero. A thousands separator
  cost the day's entire load and left no trace.
- `weekly_blocks` reports `NaN` for any week whose covered days are not all
  observed, matching what the rolling `monotony` does on the same week.
  Resample aggregations skip `NaN`, so a week with four recorded days out of
  seven previously reported a complete-looking monotony computed from four,
  and a week with no data at all reported a total of 0.0 rather than `NaN`.
  Two new columns, `n_days` and `n_observed`, say what each row is built from.
- `method="ewma"` honours `min_periods`, which it previously accepted and
  discarded, and defaults to the same `chronic` warm-up as the other two
  methods. Without it the exponentially weighted ratio reported a value on day
  one, where both legs equal that day's load and the ratio is 1.0 by
  construction, and the three methods were computed over different numbers of
  days while the diagnostics compared them.
- `smooth_center=True` requires a full window. With `min_periods=1` a centred
  smoother emitted a value for any day with a single non-`NaN` neighbour,
  back-filling the chronic warm-up with what was really a one-day mean.
- `window_sensitivity` reports `NaN`, not 1.0, for a pair sharing no comparable
  day with the reference. Jaccard on two empty sets is conventionally 1.0, and
  a pair whose chronic window exceeded the series length therefore read as
  being in perfect agreement.

### Added
- `check_load`, which reports what is wrong, or merely surprising, about a
  training log before anything is computed from it: a history shorter than the
  chronic window, no training at all, negative loads, days unobserved rather
  than rested, outliers, a suspected change of recording unit, extreme dynamic
  range, long breaks, and a series that is mostly rest. Nothing is corrected
  automatically, because whether a thirty day gap was a planned break or a
  month of unsynced training is not a question a library can answer. The
  checks come from defects found in a real three year export, and the outlier
  rule is written against the 95th percentile rather than the median, because
  training load is right-skewed and a median rule flags every athlete's
  longest run.
- `week_to_week_change` and `block_change`, implementing the other quantity
  applied practice quotes, in trailing and calendar-week forms. Gabbett (2016)
  is cited; the 15 percent figure is not shipped as a band.
- `threshold_from_history`, which places the universal 1.5 in one athlete's own
  distribution and reports what cut would be needed to pick out a given share
  of their hardest days instead. Documented as a way to make the arbitrariness
  of the universal number visible rather than as a recommendation to replace
  it with a personalised one.
- `acwr.plot.plot_acwr` and `acwr.plot.plot_window_sensitivity`, behind an
  optional matplotlib dependency and reachable as `acwr.plot_acwr` through a
  module-level `__getattr__` so that importing the package never requires
  matplotlib. Load and ratio go in stacked panels with one scale each rather
  than sharing a pair of axes, and no threshold band is drawn. Daily load is
  drawn as vertical lines rather than bars, because a multi-year log puts more
  days on the axis than there are pixels and touching bars antialias into
  darker seams that read as structure not present in the data.

### Added (release readiness)
- `CITATION.cff`, validated against the official 1.2.0 schema, carrying the
  ORCID and the three method references a user should cite alongside the
  package.
- `.zenodo.json`, so the archived release carries the intended title, licence
  and ORCID rather than whatever Zenodo infers from the repository.
- `.github/workflows/release.yml`, which runs the full CI matrix, builds,
  refuses to publish when the git tag disagrees with either `__version__` or
  `CITATION.cff`, and publishes to PyPI through trusted publishing rather than
  a stored API token. `ci.yml` is now callable so a tag cannot skip the matrix.
- `tests/test_metadata.py`, which fails if the version strings drift apart, if
  the ORCID placeholder reappears, if the declared licence disagrees across the
  packaging metadata, if `paper.md` loses one of the sections JOSS requires, or
  if a citation and the bibliography fall out of step.
- `SUBMISSION.md`, the dated runbook with the pyOpenSci answers pre-written.

### Fixed (second review pass)
- The README claimed the uncoupled ratio "reached 9.07 on days the coupled form
  capped at 4.00". It did not. On those seven days the uncoupled ratio is
  infinite, because the 21 days its denominator covers were entirely rest; 9.07
  is its largest finite reading, on a different day where the coupled form
  reads 3.01. Corrected in the README, the findings report and the figure.
- The README said a 0.01 chronic floor "only bites on the uncoupled and EWMA
  paths". Wrong in both directions: the exponentially weighted method cannot
  divide by exactly zero either, since both its legs include today's load, and
  the floor *can* fire on the coupled path.
- The docstring examples are now run, by `tests/test_doctests.py`. The
  `--doctest-modules` flag had been set while `testpaths` pointed only at
  `tests/`, so it collected nothing and a broken example in `panel.py` was
  passing CI unnoticed. That example is fixed. Running them through the
  imported package rather than by collecting `src/` directly means the suite
  works for an editable and a regular install alike; collecting `src/` errors
  at collection once `acwr` also exists in site-packages.
- The worked example no longer reads a CSV that is not in the repository.
  Every figure on the page is reproducible from the page itself, and
  `tests/test_docs.py` executes the blocks in order and checks the outputs.
- `docs/changelog.md`, `docs/contributing.md` and `docs/ai-usage.md` are
  transcluded from the repository root rather than copied. The published
  changelog had already drifted a full release behind.
- `docs/make_figure.py` writes the figure the README and the docs site embed,
  rather than leaving a hand-copied file to go stale.
- Removed `_bootstrap_ci`, which was defined and never called, and an
  unreachable guard in `monotony` that the other two validations already made
  impossible to trigger.

### Changed
- Smoothing of the ratio is now trailing rather than centred. A centred window
  made the reported value for a given day depend on later training, which
  cannot be known at the time a training decision is made. `smooth_center=True`
  restores the old behaviour for retrospective use.
- `acwr(method="ewma")` takes a `decay` argument. `"span"` uses
  `2 / (N + 1)` per Williams et al. (2017) and is the default; `"half_life"`
  uses `1 - exp(-ln 2 / N)` and matches the R package Athlytics.

### Notes
- `NullComparison.real_beats_null` is one sided and true only when the real
  statistic exceeds the null interval. The first draft of this property was two
  sided, so a ratio that performed worse than random reported as beating it.
  `real_outside_null` covers the two sided question.
- The coupled method cannot divide by zero while the numerator is positive,
  because the acute window is contained in the chronic window. Athlytics guards
  the division with a 0.01 floor on chronic load; for the coupled path that
  guard is unreachable on a positive numerator.

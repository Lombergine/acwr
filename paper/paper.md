---
title: 'acwr: acute:chronic workload ratios in Python, with diagnostics for the metric itself'
tags:
  - Python
  - sports science
  - biomechanics
  - athlete monitoring
  - training load
  - injury risk
authors:
  - name: Aditya Garg
    orcid: 0009-0003-1004-500X
    affiliation: 1
affiliations:
  - name: Independent researcher, San Jose, California, United States
    index: 1
date: 4 October 2026
bibliography: paper.bib
---

<!--
DRAFT. Not ready for submission. Two blockers remain:

1. Public development history. JOSS desk-rejects submissions whose repository
   has not been public, with active development, for more than six months.
   The clock starts the day the repository goes public, not the day the work
   started.
2. A tagged release archived with Zenodo, for the DOI. This is required on
   acceptance rather than on submission, so it is the smaller of the two.

The affiliation line reads "Independent researcher", which JOSS accepts. A
school affiliation is also acceptable and is the author's choice.
-->

# Summary

The acute:chronic workload ratio divides a measure of an athlete's recent
training load by a measure of their longer-term load, and is widely used in
athlete monitoring to flag periods of elevated injury risk. `acwr` computes the
three forms of the ratio that appear in the literature: the coupled rolling
average, the uncoupled rolling average, and the exponentially weighted variant
proposed by @williams2017. It takes a table of training activities, builds the
continuous daily load series that rolling computations require, and returns the
ratio alongside its acute and chronic components. It also computes the
companion quantities proposed by @foster1998, session RPE load, training
monotony and training strain, the week-to-week load change discussed by
@gabbett2016, and applies any of these across a squad rather than a single
athlete. A pre-flight check reports the defects that training logs reliably
contain, including suspected changes of recording unit and physically
implausible day records, before any statistic is computed from them.

The package also implements diagnostics that its subject matter arguably needs
more than the computation itself. @impellizzeri2021 showed that replacing the
chronic denominator with contrived values, whether fixed or randomly generated,
produced an association with injury comparable to the real ratio, and concluded
that the ratio functions largely as a rescaling of its own numerator. `acwr`
exposes that test directly, so a user can run it on their own data rather than
adopting a position from the literature.

# Statement of need

The ratio is computed in commercial athlete-management platforms, in consumer
running applications, and in a large number of published studies. Two R
packages implement it: `Athlytics`, reviewed by rOpenSci, and `ACWR` on CRAN.
No Python implementation existed at the time of writing, despite Python being
the working language of most applied machine learning on wearable and movement
data. Researchers working in Python have therefore re-implemented the
computation locally, which invites two specific errors.

The first is computing rolling windows over a sparse index. A 28 day window
evaluated over only the days on which an athlete trained is not a 28 day
window, and rest days are genuine zero-load observations rather than missing
data. `acwr.daily_load` makes the continuous series an explicit step with a
documented choice about how to treat absent days.

The second is conflating the coupled and uncoupled forms. In the coupled form
the chronic window contains the acute window, so the numerator contributes to
the denominator. @impellizzeri2020 set out the consequences of that shared
term. The two forms can disagree substantially: for four weeks at a constant
load followed by one week at double that load, the coupled ratio reads 1.6 and
the uncoupled ratio reads 2.0. Papers frequently report an ACWR without
specifying which form was used.

Two consequences of that shared term are worth stating plainly, because they
follow directly from the definition and neither R implementation documents or
tests them. Written
with sums rather than means, the coupled ratio equals
$(c/a)\cdot(S_a/S_c)$, where $S_a$ and $S_c$ are load summed over the acute and
chronic windows. Because the acute window is contained in the chronic window,
$S_a \le S_c$, so the ratio is bounded above by $c/a$. At the conventional
seven and twenty-eight day windows that ceiling is exactly 4.0. A coupled ACWR
of 4.0 does not describe a load four times the usual one; it describes an
athlete whose entire month of load fell in the final week. The second
consequence is that coupling compresses the ratio toward 1.0 from either side,
since the condition for the coupled ratio to fall below the uncoupled one is
the same as the condition for it to exceed 1. Every threshold in the applied
literature sits above 1.0, so within the range that is monitored the coupled
form can only understate. On a three year training log of 1018 runs, the
uncoupled ratio exceeded 1.5 on 76 days where the coupled ratio did not, and on
no day was the reverse true.

Neither property is claimed as novel. @wang2019 state the ceiling explicitly in
a preprint, observing that the coupling "creates a theoretical maximum ACRatio
of 4, regardless of the magnitude of the absolute change in workload", although
the passage does not survive into the published version [@wang2020]. The
compression result bears on @coyne2019, who compared the two forms on elite
athlete data, found correlations between 0.88 and 0.99, and concluded that the
risk of coupling producing spurious correlation is low. That conclusion and the
asymmetry reported here are compatible, because a correlation between two
series does not constrain whether they fall on the same side of a cut point,
and a threshold-based decision depends only on the latter. The contribution
this package makes is to hold an implementation to both properties with tests,
which is what was missing rather than the properties themselves.

# State of the field

`Athlytics` computes the coupled rolling average and an exponentially weighted
variant in R, and reads Strava and device files. `ACWR` on CRAN implements all
three forms in R. Neither ships a null-model diagnostic, and neither documents
the two structural properties above.

Contributing to one of those packages was considered and rejected on two
grounds. They are written in R, and the gap identified here is specifically a
Python one, so a contribution would not close it. Their scope is the
computation of the metric, whereas the contribution offered here is a set of
tools for interrogating the metric, which is a different design goal rather
than a missing feature.

The relationship with `Athlytics` is treated as a dependency of correctness
rather than competition. Its coupled rolling-average algorithm was read from
source and transcribed into this package's test suite, where
`test_coupled_method_matches_the_athlytics_algorithm` asserts the two agree to
floating point on identical input. That cross-check surfaced three substantive
differences, documented in the README: the exponential decay is parameterised
as a span in this package, following @williams2017, and as a half-life in
`Athlytics`, so the same `N` means different things; smoothing is trailing in
both, after a centred window in an early draft of this package was identified
as a defect and corrected; and division by a near-zero chronic load returns
`inf` here against `NA` there.

# Software design

The package is organised around a single invariant: every computation consumes
a gap-free daily load series indexed by date, and every function validates that
invariant rather than assuming it. `check_load` extends the same principle
upstream, reporting the properties of a log that would otherwise silently
change every number downstream, and classifying each by whether it invalidates
the computation or merely qualifies it. `daily_load` is the only way into that
representation, and it rejects negative loads, empty frames and inverted date
ranges. `ratio.py`, `foster.py` and `diagnostics.py` each reject a non-daily or
gapped index with a message naming the fix.

Three design decisions are worth recording. Operations that are mathematically
undefined return `inf` or `NaN` rather than an imputed value, so that a week of
identical sessions reports infinite monotony rather than a large finite number
that conceals it. Smoothing is trailing by default, because a centred window
makes today's value depend on training that has not happened, which is a defect
in a tool meant to inform a decision about tomorrow. Where the literature is
ambiguous, the ambiguity is exposed as a parameter with a documented default
rather than resolved silently: the exponential decay parameterisation, the
degrees of freedom in the monotony standard deviation, and the day a training
week ends on are all explicit, and each is accompanied by a statement of how
much the answer moves.

The package depends only on NumPy [@harris2020], pandas [@mckinney2010] and
SciPy [@virtanen2020]. Reading device files is deliberately out of scope,
because existing libraries do it well and the documented route is to pipe their
output in.

# Research impact

The package has been used by the author on three years of their own training,
1018 runs across 1199 days. That use surfaced the two structural properties
reported above, which are tested here but not claimed as discoveries, and
produced a result from
`window_sensitivity`, a diagnostic that sweeps window-length pairs and reports
the Jaccard agreement between the days each pair flags: across sixteen pairs,
the fraction of days flagged above 1.5 ranged from 1.6 percent to 20.7 percent,
and no alternative pair agreed with the conventional 7/28 choice on more than
71 percent of flagged days. The window lengths, which are rarely justified in
published work, moved the output more than the training did.

The near-term significance claimed is specific. Applied sports science is
moving to Python for wearable data, the ratio remains in widespread use while
its foundations are disputed, and no Python tool currently lets a practitioner
run the critics' own test on their own athletes. The benchmark for success is
whether that test gets run: adoption of `compare_to_null` and
`window_sensitivity` in published analyses is the outcome this package is built
to produce.

# Generative AI disclosure

Claude (Anthropic) was used extensively as a pair programmer in developing this
package, including writing the majority of the source code, docstrings and
tests, and drafting the documentation and the first version of this paper. The
author set the scope and design priorities, supplied the domain knowledge and
the validation dataset, reviewed and directed the work, and is responsible for
the correctness of the package and for every claim made here. A detailed
disclosure, including five specific errors the process produced and the
mechanism that caught each one, is in `AI-USAGE.md` in the repository.

# Acknowledgements

<!-- TODO: anyone who reviews the code, suggests a method, or reports a bug
before submission belongs here. -->

# References

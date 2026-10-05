# A worked example

This walks the whole path, from a table of activities to a judgement about
whether the ratio is worth trusting on your own data.

Every block below runs, and every number shown is what the code prints. Run
them in order and you will get the same figures, because the training log is
generated from a fixed seed rather than read from a file you do not have.

## 0. The data

In real use this is a CSV. Here it is eighteen months of a plausible runner:
a base that drifts across the year, a longer run on Sundays, rest days
scattered through, and two breaks of the kind a season actually contains.

```python
import numpy as np
import pandas as pd

import acwr

rng = np.random.default_rng(7)
days = pd.date_range("2025-01-01", periods=540, freq="D")

base = 55 + 18 * np.sin(np.arange(540) / 47.0)
load = base * rng.gamma(shape=4.0, scale=0.25, size=540)
load[pd.Index(days).dayofweek == 6] *= 1.7   # the Sunday long run
load[rng.random(540) < 0.22] = 0.0           # rest days
load[250:268] = 0.0                          # a break
load[430:441] = 0.0                          # another

activities = pd.DataFrame({"date": days, "load": np.round(load, 1)})
activities = activities[activities.load > 0].reset_index(drop=True)
activities.head()
#         date  load
# 0 2025-01-01  50.4
# 1 2025-01-02  43.8
# 2 2025-01-04  53.1
# 3 2025-01-05  67.3
# 4 2025-01-06  66.7
```

410 activities. Note what the frame does not contain: any row for a rest day.
That absence is the problem the next step exists to solve.

## 1. Build the daily series

```python
load = acwr.daily_load(activities)
len(load), int((load > 0).sum())
# (540, 410)
```

540 calendar days, 410 of them with training. The 130 rest days are now present
as zeros, because a rest day is an observation of not training rather than a
hole in the record. Pass `missing="na"` instead when a stretch is genuinely
absent, such as a period when a watch was not worn; that propagates `NaN`
through any window it touches.

This step exists so that the rolling windows run over a continuous index. It is
the single most common place a published ACWR goes wrong. A 28 day window
evaluated over only the 410 days this runner trained is not a 28 day window.

## 2. Check the log before computing anything from it

```python
print(acwr.check_load(load))
# Load quality: 540 days, 410 with training, 0 error(s), 1 warning(s)
#   [warn ] break: the longest run of zero-load days is 19 days, ending 2025-09-25. The ratio on the days after it is dominated by the break rather than by the training
```

One warning, and it is the right one: there is a nineteen day break in this
log. Nothing is wrong with the data, but the ratio in the fortnight after that
break is a statement about the break rather than about the training, and it is
better to know that before reading the numbers than after.

On a real log this step earns its keep by catching the things that are wrong.
A change of recording unit halfway through, a watch left running, a history
too short for the window you asked for. None of those raises an error anywhere
else in the pipeline.

## 3. Compute the ratio

```python
for method in ("rac", "rau", "ewma"):
    print(method, round(float(acwr.acwr(load, method=method).iloc[-1]), 2))
# rac 1.07
# rau 1.1
# ewma 0.95
```

Compare them before choosing one. Here they agree, which is the uninteresting
and reassuring case. When they disagree sharply on the days you care about, the
choice of method is doing more work than the data is, and
[the diagnostics page](diagnostics.md) has a way to measure that.

```python
frame = acwr.acwr(load, return_components=True)
frame.tail(3).round(2)
#             load  acute  chronic  acwr
# 2026-06-22  18.1  35.63    29.99  1.19
# 2026-06-23   0.0  28.01    29.18  0.96
# 2026-06-24  20.7  30.97    28.84  1.07
```

Reading the components matters. A ratio of 1.2 produced by a hard week is a
different situation from the same 1.2 produced by an unusually easy month, and
the single number cannot tell them apart. The middle row is worth a look: a rest
day, and the ratio drops below 1 without anything happening except the passage
of a day.

## 4. Ask whether the denominator is doing anything

This needs no injury data at all.

```python
print(acwr.rescaling_check(load))
# ACWR vs acute load alone, n=513: Spearman 0.644 [0.580, 0.702], Pearson 0.475
```

The ratio is acute load divided by chronic load. If the denominator varies
little next to the numerator, dividing barely changes the ordering of days and
the ratio becomes a rescaled copy of the acute load. A Spearman correlation near
1.0 means exactly that. At 0.64 the denominator is reordering days, so it is
carrying information, and a little under half the ordering still comes from the
numerator by itself.

## 5. Ask how much the window lengths decided the answer

```python
print(acwr.window_sensitivity(load))
# Window sensitivity at threshold 1.5 (rac, 16 window pairs)
#   reference 7/28 flags 9.4% of days
#   flagged fraction across pairs: 1.2% to 13.2%
#   agreement with the reference: 0.12 to 0.66 (Jaccard)
```

Seven and twenty-eight days are conventional and are almost never justified.
On this log, the best any alternative pair manages is agreeing with 7/28 on 66
percent of the days it flags, and the fraction of days flagged runs from 1.2 to
13.2 percent. `result.table` holds the full grid.

## 6. With injury data, run the critics' test

```python
acute = acwr.acute_load(load)
p = 1 / (1 + np.exp(-(acute - acute.mean()) / acute.std()))
injured = pd.Series(rng.random(len(load)) < 0.22 * p, index=load.index)

result = acwr.compare_to_null(load, injured, kind="shuffle", n_null_draws=120, seed=3)
print(result)
# AUC, n=513 (50 events)
#   real ACWR:          0.617 [0.541, 0.692]
#   shuffle    null:    0.658 [0.613, 0.702]  (120 draws)
#   real ratio is INSIDE the null interval: the chronic denominator added nothing here
```

The injury series above was generated from acute load alone, deliberately, so
the chronic denominator genuinely contains nothing. The test says so. That is
the Impellizzeri et al. (2021) result reproduced on data where we know the
answer, which is the only honest way to demonstrate a diagnostic.

On your own data the verdict line reads one of three ways, and the difference
matters:

```python
result.real_beats_null     # one sided: True only if the real statistic exceeds the interval
result.real_outside_null   # two sided
result.null_distribution   # the raw draws, if you want to plot them
```

A ratio that falls *below* its null has not beaten anything. Conflating that
with "outside the interval" was a bug in this package's own first draft.

## 7. What does 1.5 mean for this runner?

```python
print(acwr.threshold_from_history(load))
# Thresholds on this athlete's own history (rac, n=513 usable days)
#   1.5 flags 48 days (9.4%), which is this athlete's 90.6th percentile
#   their own percentiles:
#     50th  0.96
#     75th  1.20
#     90th  1.47
#     95th  1.77
#     99th  2.92
```

The universal figure happens to land near this runner's 90th percentile, which
is a coincidence of their training rather than a property of the number. On the
three year log in the README it lands at the 84th and flags one day in six.

## 8. Monotony, if the question is how the week was shaped

```python
print(round(float(acwr.monotony(load).iloc[-1]), 2))
# 0.63

acwr.weekly_blocks(load).dropna().tail(3).round(2)
#             n_days  n_observed  total   mean     sd  monotony  strain
# 2026-06-14       7           7  245.1  35.01  34.18      1.02  251.09
# 2026-06-21       7           7  253.1  36.16  49.08      0.74  186.46
# 2026-06-28       3           3   38.8  12.93  11.28      1.15   44.50
```

`n_days` is how many calendar days the row covers and `n_observed` how many of
them carried an observation. The last row covers three days because the series
ends mid-week, which is why it is labelled rather than silently averaged in
beside the full weeks.

## What to conclude

On this dataset: the ratio is not a pure rescaling of its numerator, it is
heavily dependent on window lengths nobody justifies, and it carries no
information about an outcome that was built from acute load alone. Those three
findings are not in tension. They are what the metric looks like when you
measure it instead of arguing about it.

Your data will say something different. That is the point of running it.

# acwr

Acute:chronic workload ratios in Python, with the diagnostics to test whether
they mean anything on your data.

```bash
pip install acwr
```

The acute:chronic workload ratio divides an athlete's recent training load by
their longer-term load. It is probably the most widely used number in athlete
monitoring, it appears in commercial platforms and consumer running apps, and a
substantial body of work argues it does not predict injury. This package
computes all three published forms of it, and it ships the test its critics
proposed so that anyone using it can check the question rather than take a side.

Before this package there was no Python implementation. R has two, `Athlytics`
on rOpenSci and `ACWR` on CRAN.

![Three years of one runner's training, with all three ratio methods computed
over it](img/acwr-real-data.png)

## From a table of activities to a ratio

```python
import pandas as pd
import acwr

activities = pd.DataFrame({
    "date": ["2026-01-01", "2026-01-01", "2026-01-03"],
    "load": [30.0, 20.0, 45.0],
})

load = acwr.daily_load(activities)          # continuous daily series
ratio = acwr.acwr(load, method="rac")       # the standard form
```

`daily_load` does something easy to skip and important to get right. Rolling
windows must run over a continuous daily index, so rest days are filled with
zero rather than left out. A 28 day window computed over only the days someone
trained is not a 28 day window, and that is the most common way a published
ACWR goes wrong.

Any numeric load works, since the ratio is scale free. Session RPE, minutes,
distance and training stress score all behave the same way, as long as the unit
is consistent.

## Where to go next

| | |
|---|---|
| [Getting started](example.md) | A worked analysis, every figure reproducible from the page |
| [Checking the metric](diagnostics.md) | The diagnostics, and what they do and do not prove |
| [API reference](api.md) | Every public function, generated from its docstrings |
| [How this was built](ai-usage.md) | The generative AI disclosure, in full |

## The three methods

```python
acwr.acwr(load, method="rac")    # rolling average, coupled    (default)
acwr.acwr(load, method="rau")    # rolling average, uncoupled
acwr.acwr(load, method="ewma")   # exponentially weighted
```

They disagree, and the disagreement is the interesting part. Take four steady
weeks at a load of 50, then one week at 100:

| method | ratio on the final day |
|---|---|
| `rac` | 1.60 |
| `rau` | 2.00 |
| `ewma` | 1.34 |

`rac` is the original form, and the chronic window contains the acute window.
The hard week therefore appears in both the numerator and the denominator,
lifting the denominator from 50 to 62.5 and pulling the ratio down to 1.6. That
shared term is the mathematical coupling that Impellizzeri et al. (2020)
describe.

`rau` takes the chronic average over the 21 days *before* the acute window, so
the two halves share no data and the answer is the arithmetically clean 2.0.

`ewma` weights recent days more heavily, following Williams et al. (2017), on
the argument that a flat window treats a session from three weeks ago as
exactly as recent as one from yesterday.

## What this package does not claim

It does not claim the ratio predicts injury. The weight of the evidence runs
the other way, and the package exists partly to make that evidence easy to
reproduce.

It does not claim the ratio is worthless either. The diagnostics are
descriptive, they are not hypothesis tests, and a result on one dataset is a
result on one dataset.

It does not model injury risk, recommend training, or output a safe range. The
"sweet spot" bands that circulate in coaching material do not appear anywhere in
this code, because the thresholds they rest on come from the same literature
whose foundations are in dispute.

## Citing the methods

If you use a particular method, cite its source rather than this package alone.
The full reference list is in the
[README](https://github.com/Lombergine/acwr#citing-the-methods).

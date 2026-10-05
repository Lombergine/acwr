# Checking whether the ratio means anything

This is the part of the package that does not exist elsewhere, and it is the
reason the package exists at all.

The acute:chronic workload ratio has a substantial literature arguing it does
not predict injury. Rather than take a side, `acwr` ships the tests that
argument rests on, so you can run them on your own athletes.

## Without any injury data

```python
print(acwr.rescaling_check(load))
# ACWR vs acute load alone, n=1172: Spearman 0.512 [0.461, 0.559], Pearson 0.390
```

That figure is from three years of one runner's log, 1,018 runs. For a version
you can run yourself, see [the worked example](example.md), whose every number
is reproducible from the page.

The ratio is acute load divided by chronic load. When the denominator varies
little next to the numerator, dividing barely changes the ordering of days and
the ratio becomes a rescaled copy of the acute load. This reports how far down
that road your data sits. In the limit it is exact: with a constant denominator
the Spearman correlation is 1.0 and the ratio is the numerator in different
units.

## With injury data

```python
result = acwr.compare_to_null(load, injured, kind="shuffle")
print(result)
```

This is the test from Impellizzeri et al. (2021). Compute the association
between injury and the real ratio, then recompute it many times against ratios
whose chronic denominator has been replaced by a contrivance, and compare. If a
denominator can be swapped for noise without hurting the result, it was not
doing any work.

Three contrivances are available. `"shuffle"` permutes the real chronic series,
keeping its exact distribution and destroying its timing. `"normal"` draws from
a distribution matched to the real chronic mean and spread. `"constant"` uses
the mean, which is the fixed-value version of the original test.

!!! warning "Two properties, deliberately different"

    `real_beats_null` is one sided and true only when the real statistic
    exceeds the null interval. `real_outside_null` is two sided. A ratio that
    falls *below* its null has not beaten anything, and conflating those two
    was a bug in this package's own first draft.

## How much do the window lengths decide the answer?

```python
print(acwr.window_sensitivity(load))
```

```
Window sensitivity at threshold 1.5 (rac, 16 window pairs)
  reference 7/28 flags 16.0% of days
  flagged fraction across pairs: 1.6% to 20.7%
  agreement with the reference: 0.02 to 0.71 (Jaccard)
```

Seven and twenty-eight days are conventional. They are not derived from
anything, and papers that use them rarely say why. This sweeps a grid of window
pairs and reports the Jaccard agreement between the days each pair flags and
the days the conventional pair flags.

The output above is from the same three years of one runner's log. No alternative pair agreed
with 7/28 on more than 71 percent of flagged days, and the fraction of days
flagged ranged over an order of magnitude. The window lengths moved the output
more than the training did.

## What none of this proves

The diagnostics are descriptive. They are not hypothesis tests, and a result on
one dataset is a result on one dataset. A low agreement does not prove the
metric is worthless, and a high one does not validate it. What they do is put
numbers on choices that usually go unexamined.

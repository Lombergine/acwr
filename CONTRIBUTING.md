# Contributing

Contributions are welcome, including from people who think the acute:chronic
workload ratio is a bad metric. That view is well represented in the literature
and this package is built to make it testable.

## The most useful things

**A bug report with a reproducible example.** A failing test is better than a
description. If a ratio comes out wrong for a particular load series, the
series and the expected value are what is needed.

**A hand-computed value the tests do not yet pin.** Several tests check against
numbers worked out by hand rather than against whatever the code produced
first, because that is the only way to catch an off-by-one in a rolling window.
More of those are always wanted.

**A method from the literature that is missing**, with the citation.

## Setting up

```bash
git clone https://github.com/Lombergine/acwr
cd acwr
pip install -e ".[dev]"
pytest
```

## Before opening a pull request

- `pytest` passes, including the doctests.
- `ruff check .` and `ruff format --check .` pass.
- `mypy src` passes.
- New behaviour has a test. New public functions have a numpydoc docstring with
  an example.

## What will be pushed back on

**A claim the package does not support.** Nothing in the documentation should
assert that the ratio predicts injury, or recommend a safe range. The
thresholds that circulate in coaching material rest on the part of the
literature that is in dispute, and this package deliberately does not ship them.

**A silent fallback.** Where the maths is undefined, such as dividing by a zero
chronic load, the result stays `inf` or `NaN`. Filling it with a plausible
number hides a real condition from the user.

**A window computed on a sparse index.** Every rolling computation runs over a
continuous daily series. If a change makes it possible to compute a 28 day
window over fewer than 28 days of index without the user asking for it, that is
a bug even if the output looks reasonable.

## Code of conduct

Be straightforward and assume good faith. Disagreement about the science is
expected and welcome; personal hostility is not.

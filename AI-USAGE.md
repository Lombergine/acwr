# Disclosure of generative AI use

Both JOSS and pyOpenSci require authors to disclose the use of generative AI
tools in developing, documenting or writing up a package. This file is that
disclosure. It is deliberately specific, because a vague one is worse than
none.

## What was used

Claude (Anthropic) was used throughout, in an extended interactive session, as
a pair programmer. It was not used to generate the package in one shot from a
prompt, and it was not used to write anything that was then submitted unread.

## What the tool did

- Wrote the large majority of the source code in `src/acwr/`, including the
  function signatures, the implementations and the docstrings.
- Wrote the large majority of the test suite in `tests/`.
- Drafted the README, `docs/example.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md` and the first draft of `paper/paper.md`.
- Read the source of the R package `Athlytics` and transcribed its coupled
  rolling-average algorithm into this package's test suite for cross-checking.
- Read the primary literature cited in the paper and the README, and located
  the formulas quoted from Williams et al. (2017), Impellizzeri et al. (2020,
  2021) and Foster (1998).
- Found the two structural properties of the coupled ratio documented in the
  README, while running the package on real training data.

## What the author did

- Set the problem, the scope and the design priorities, and decided what the
  package was for.
- Supplied the domain knowledge: which metrics matter in applied running, which
  of them a monitoring tool actually has to produce, and what a coach or sports
  scientist would want from the output.
- Supplied the real dataset the package was validated against: 1,018 of the
  author's own runs recorded over three years.
- Reviewed, questioned and directed the work at every step, including rejecting
  proposed features and requiring claims to be verified against primary sources
  rather than asserted.
- Takes responsibility for the correctness of the package, for maintaining it,
  and for every claim made in the paper.

## Errors the process produced, and how they were caught

Recording these is part of an honest disclosure, because the failure modes of
AI-assisted development are the reason the disclosure is required.

1. A docstring claimed the coupled ratio reads 1.4545 for a specific training
   pattern. The correct value is 1.6. Caught by hand computation in a test.
2. `real_beats_null` was implemented as a two-sided comparison, so a ratio
   performing *worse* than its null model returned `True`. Caught by a
   regression test and made one-sided.
3. The smoothing window was centred, which made a smoothed ratio depend on
   training that had not happened yet. Caught by cross-checking against the
   `Athlytics` R source and made trailing.
4. The module docstring for `foster.py` stated the direction of the `ddof`
   effect backwards. Caught by a test that asserted the ratio of the two
   conventions equals `sqrt((n-1)/n)`.
5. Two expected values in the Foster tests were guessed rather than derived.
   Caught when the tests failed, and replaced with values computed by hand and
   with the derivation written into the test.

Every one of these was found by the test suite or by checking against a primary
source. That is the reason the test suite is as large as it is, and it is the
main argument for why the code in this package can be trusted despite how it was
written.

## Reproducibility

No model output is embedded in the package at runtime. The package has no
network calls, no model dependency and no API key. It depends only on NumPy,
pandas and SciPy, and its results are deterministic given a seed.

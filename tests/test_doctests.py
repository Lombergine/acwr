"""Run the Example blocks in the package's own docstrings.

These are the examples a user sees first, in the API reference and in their
editor's tooltip, and an example that no longer produces what it claims is
worse than no example. They are run here, against the imported package, rather
than by pointing pytest's `--doctest-modules` at `src/`, because that only
collects cleanly for an editable install and errors at collection otherwise.
"""

from __future__ import annotations

import doctest
import importlib

import pytest

MODULES = [
    "acwr",
    "acwr.load",
    "acwr.ratio",
    "acwr.foster",
    "acwr.panel",
    "acwr.diagnostics",
]


@pytest.mark.parametrize("name", MODULES)
def test_the_docstring_examples_still_produce_what_they_claim(name):
    module = importlib.import_module(name)
    result = doctest.testmod(
        module,
        verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )
    assert result.failed == 0, f"{result.failed} doctest failure(s) in {name}"


def test_the_examples_are_not_silently_absent():
    """A module whose examples all got deleted would pass the test above."""
    total = 0
    for name in MODULES:
        module = importlib.import_module(name)
        finder = doctest.DocTestFinder()
        total += sum(len(t.examples) for t in finder.find(module))
    assert total >= 20, f"only {total} doctest examples found across the package"

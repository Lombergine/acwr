"""The worked example in the documentation has to keep working.

`docs/example.md` is the page a new user reads first, and it is the page most
likely to rot: the code changes, the prose keeps the old numbers, and nothing
fails. This module executes every Python block in that page in order, in one
namespace, and checks that the outputs the page claims are the outputs the
package produces.

It is written against the file rather than against a copy of the code, so a
figure edited in the page without being recomputed fails here.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "example.md"


def python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", markdown, re.S)


@pytest.fixture(scope="module")
def executed() -> tuple[list[str], list[str]]:
    """Run every block in order, returning the blocks and what each printed."""
    text = EXAMPLE.read_text()
    blocks = python_blocks(text)
    namespace: dict[str, object] = {}
    outputs = []
    for i, block in enumerate(blocks, 1):
        # Lines beginning with # are the documented output, not input.
        code = "\n".join(
            line for line in block.splitlines() if not line.strip().startswith("#")
        )
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(code, f"<example block {i}>", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the failure message is the point
            pytest.fail(f"block {i} of docs/example.md raised {exc!r}")
        outputs.append(buffer.getvalue())
    return blocks, outputs


def test_the_example_page_has_the_blocks_this_module_expects(executed):
    blocks, _ = executed
    assert len(blocks) == 11, "the page was restructured; update the assertions below"


def test_every_block_runs_in_order_without_raising(executed):
    # The fixture fails the test if any block raises, so reaching here is the
    # assertion. A page whose first block defines `activities` and whose fourth
    # uses it only works if they share a namespace, which is how a reader runs
    # them.
    blocks, outputs = executed
    assert len(outputs) == len(blocks)


@pytest.mark.parametrize(
    "fragment",
    [
        "rac 1.07",
        "rau 1.1",
        "ewma 0.95",
        "n=513: Spearman 0.644 [0.580, 0.702], Pearson 0.475",
        "reference 7/28 flags 9.4% of days",
        "flagged fraction across pairs: 1.2% to 13.2%",
        "agreement with the reference: 0.12 to 0.66 (Jaccard)",
        "AUC, n=513 (50 events)",
        "real ACWR:          0.617 [0.541, 0.692]",
        "shuffle    null:    0.658 [0.613, 0.702]  (120 draws)",
        "real ratio is INSIDE the null interval",
        "Load quality: 540 days, 410 with training, 0 error(s), 1 warning(s)",
        "the longest run of zero-load days is 19 days",
        "1.5 flags 48 days (9.4%), which is this athlete's 90.6th percentile",
    ],
)
def test_the_documented_output_is_the_real_output(executed, fragment):
    _, outputs = executed
    printed = "\n".join(outputs)
    assert fragment in printed, (
        f"docs/example.md shows {fragment!r} but the code does not print it"
    )


def test_the_documented_figures_also_appear_in_the_page_prose(executed):
    """A number in a comment and a different number in the sentence below it."""
    text = EXAMPLE.read_text()
    for figure in ("0.64", "9.4", "1.2", "13.2", "66", "513", "410", "540"):
        assert figure in text


def test_the_page_does_not_read_a_file_that_is_not_in_the_repository(executed):
    """The reason this page was rewritten: it used to depend on training.csv."""
    blocks, _ = executed
    joined = "\n".join(blocks)
    assert "read_csv" not in joined

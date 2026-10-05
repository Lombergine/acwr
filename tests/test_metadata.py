"""The package's version has to agree with itself in every place it is written.

Three files carry the version: `src/acwr/__init__.py`, which is the source of
truth and what hatchling reads; `CITATION.cff`, which is what a citation
manager and Zenodo read; and the git tag at release time, checked by the
release workflow. A release where these disagree puts the wrong number on
PyPI permanently, so they are checked here rather than remembered.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on Python 3.10
    import tomli as tomllib

import acwr

ROOT = Path(__file__).resolve().parents[1]


def test_the_citation_file_states_the_version_the_package_reports():
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert cff["version"] == acwr.__version__


def test_the_citation_file_is_well_formed_and_carries_the_orcid():
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert cff["cff-version"] == "1.2.0"
    assert {"cff-version", "message", "title", "authors"} <= set(cff)
    author = cff["authors"][0]
    assert author["family-names"] == "Garg"
    assert author["orcid"] == "https://orcid.org/0009-0003-1004-500X"
    assert cff["license"] == "MIT"


def test_the_orcid_is_the_same_everywhere_it_appears():
    """A placeholder ORCID shipped once already; it does not get to again."""
    orcid = "0009-0003-1004-500X"
    paper = (ROOT / "paper" / "paper.md").read_text()
    cff = (ROOT / "CITATION.cff").read_text()
    zenodo = (ROOT / ".zenodo.json").read_text()
    for name, text in (
        ("paper.md", paper),
        ("CITATION.cff", cff),
        (".zenodo.json", zenodo),
    ):
        assert orcid in text, f"{name} is missing the ORCID"
        assert "0000-0000-0000-0000" not in text, f"{name} still has the placeholder"


def test_the_declared_licence_agrees_across_the_packaging_metadata():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    zenodo = __import__("json").loads((ROOT / ".zenodo.json").read_text())
    assert cff["license"] == zenodo["license"] == "MIT"
    assert "MIT" in (ROOT / "LICENSE").read_text()
    classifiers = pyproject["project"]["classifiers"]
    assert any("MIT" in c for c in classifiers)


def test_the_paper_has_every_section_joss_now_requires():
    """JOSS added required sections; a missing one is a desk rejection."""
    paper = (ROOT / "paper" / "paper.md").read_text()
    for heading in (
        "# Summary",
        "# Statement of need",
        "# State of the field",
        "# Software design",
        "# Research impact",
        "# Generative AI disclosure",
        "# References",
    ):
        assert heading in paper, f"paper.md is missing {heading!r}"


def test_every_citation_in_the_paper_has_a_bibliography_entry():
    paper = (ROOT / "paper" / "paper.md").read_text()
    bib = (ROOT / "paper" / "paper.bib").read_text()
    cited = set(re.findall(r"@([a-z]+[0-9]{4})", paper))
    defined = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    assert cited, "the paper cites nothing, which cannot be right"
    assert cited <= defined, f"uncited keys: {sorted(cited - defined)}"
    assert defined <= cited, f"unused bib entries: {sorted(defined - cited)}"


@pytest.mark.parametrize(
    "filename",
    [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "AI-USAGE.md",
        ".zenodo.json",
        "mkdocs.yml",
    ],
)
def test_the_files_both_review_venues_check_for_are_present(filename):
    assert (ROOT / filename).is_file()

"""Tests for the source registry (src/sources.py) and the repo's sources.yaml."""

import pytest

from src import config
from src.sources import (RegistryError, enabled_sources, load_registry,
                         parse_registry)


def entry(**over):
    base = {"id": "demo_src", "title": "Demo source", "type": "review",
            "domain": "clinical", "format": "markdown", "path": "data/raw/demo.md",
            "license": "CC BY 4.0"}
    base.update(over)
    return base


def test_minimal_entry_gets_defaults(tmp_path):
    (src,) = parse_registry({"sources": [entry()]}, root=tmp_path)
    assert src.label == "Demo source"          # label defaults to title
    assert src.enabled is True
    assert src.path == tmp_path / "data/raw/demo.md"
    assert src.options == {} and src.year is None


def test_every_problem_is_reported_at_once(tmp_path):
    bad = {"sources": [
        entry(id="Bad-ID"),
        entry(id="dupe"), entry(id="dupe"),
        entry(id="fmt", format="pptx"),
        entry(id="dom", domain="billing"),
        entry(id="typo", licence="x"),
        entry(id="yr", year="2026"),
        {"id": "nolicense", "title": "t", "type": "review", "domain": "clinical",
         "format": "html", "path": "x.html"},
    ]}
    with pytest.raises(RegistryError) as err:
        parse_registry(bad, root=tmp_path)
    msg = str(err.value)
    for needle in ("lowercase", "duplicate id", "format 'pptx'", "domain 'billing'",
                   "unknown field(s) ['licence']", "year must be a number",
                   "missing ['license']"):
        assert needle in msg


def test_top_level_must_be_a_sources_list(tmp_path):
    with pytest.raises(RegistryError):
        parse_registry([entry()], root=tmp_path)


def test_enabled_sources_requires_files(tmp_path):
    (tmp_path / "here.md").write_text("# hi")
    sources = parse_registry({"sources": [
        entry(id="present", path="here.md"),
        entry(id="absent", path="nope.md", url="https://example.org/x",
              notes="save it by hand"),
        entry(id="off", path="nope.md", enabled=False),
    ]}, root=tmp_path)
    with pytest.raises(RegistryError) as err:
        enabled_sources(sources)
    msg = str(err.value)
    assert "absent" in msg and "https://example.org/x" in msg and "save it by hand" in msg
    assert "off" not in msg and "present" not in msg

    sources = [s for s in sources if s.id != "absent"]
    assert [s.id for s in enabled_sources(sources)] == ["present"]


# ------------------------------------------------------ the real file
def test_repo_registry_is_valid():
    sources = load_registry()
    ids = [s.id for s in sources]
    assert len(ids) == len(set(ids))
    textbook = next(s for s in sources if s.id == "mohs_layers")
    assert textbook.enabled and textbook.format == "pdf_textbook"
    assert textbook.path == config.BOOKS["mohs_layers"]


def test_repo_registry_only_enables_the_textbook_for_now():
    """Public sources stay disabled until their files are downloaded - an
    enabled entry with a missing file would break ingest."""
    enabled = [s.id for s in load_registry() if s.enabled]
    assert enabled == ["mohs_layers"]

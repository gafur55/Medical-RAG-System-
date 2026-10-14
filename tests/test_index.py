"""Tests for building the multi-source index: src/ingest.py and src/embed.py.

No PDF, no loaders, no embedding model: loaders are replaced by
monkeypatching ingest.load_units, the model by a fake encode() function,
and every index path in config is redirected into tmp_path.

The real-data invariant (textbook-only rebuild reproduces today's
chunks.json and vectors.npy exactly) is checked by test_real_index_*
at the bottom, read-only.
"""

import json

import numpy as np
import pytest

from src import chunking, config, embed, ingest
from src.chunking import Unit
from src.sources import Source


def make_source(sid="demo", fmt="markdown", domain="clinical", enabled=True,
                path=None, **over):
    fields = dict(id=sid, title=f"{sid} title", label=f"{sid} label", type="review",
                  domain=domain, format=fmt, path=path or config.ROOT / f"{sid}.md",
                  license="CC BY 4.0", url=f"https://example.org/{sid}", enabled=enabled)
    fields.update(over)
    return Source(**fields)


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    """Point every index file at tmp_path/processed."""
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "CHUNKS_PATH", processed / "chunks.json")
    monkeypatch.setattr(config, "VECTORS_PATH", processed / "vectors.npy")
    monkeypatch.setattr(config, "VECTOR_KEYS_PATH", processed / "vector_keys.json")
    monkeypatch.setattr(config, "MANIFEST_PATH", processed / "manifest.json")
    monkeypatch.setattr(config, "FIGURES_DIR", processed / "figures")
    return processed


class FakeEncoder:
    """Stands in for the model: a deterministic 3-d vector per text, and a
    record of every call so tests can see what was (not) re-embedded."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.array([[len(t), sum(map(ord, t)) % 997, 1.0] for t in texts],
                        dtype=np.float32)

    @property
    def n_encoded(self):
        return sum(len(c) for c in self.calls)


# ================================================================ ingest
def piece(text, section=("Intro",), locator="Intro", page=None, kind="prose"):
    return {"text": text, "section": section, "locator": locator, "page": page, "kind": kind}


def test_chunk_records_fields_and_ids():
    src = make_source("statpearls_demo", domain="operations")
    (rec,) = ingest.chunk_records(src, [piece("Mohs is done in stages.",
                                              section=("Indications", "Tumors"),
                                              locator="Tumors")])
    assert rec == {
        "id": chunking.chunk_id("statpearls_demo", "Tumors", "Mohs is done in stages."),
        "book": "statpearls_demo",
        "source": "statpearls_demo",
        "domain": "operations",
        "page": None,
        "locator": "Tumors",
        "section": ["Indications", "Tumors"],
        "kind": "prose",
        "text": "Mohs is done in stages.",
        "figures": [],
    }
    json.dumps(rec)   # section became a list: the record is plain JSON


def test_chunk_records_paginated_id_uses_page():
    src = make_source("cms_demo", fmt="pdf")
    (rec,) = ingest.chunk_records(src, [piece("Bundling rule.", section=(), locator="p. 12",
                                              page=12)])
    assert rec["id"] == chunking.chunk_id("cms_demo", 12, "Bundling rule.")
    assert rec["page"] == 12 and rec["locator"] == "p. 12"


def test_chunk_records_fills_an_empty_locator():
    src = make_source()
    by_page, by_section, bare = ingest.chunk_records(src, [
        piece("a", locator="", page=3, section=()),
        piece("b", locator="", section=("Top", "Sub")),
        piece("c", locator="", section=()),
    ])
    assert by_page["locator"] == "p. 3"
    assert by_section["locator"] == "Sub"
    assert by_section["id"] == chunking.chunk_id("demo", "Sub", "b")
    assert bare["locator"] == ""


def test_same_text_in_another_section_is_another_chunk():
    src = make_source()
    a, b = ingest.chunk_records(src, [piece("Same text.", locator="A", section=("A",)),
                                      piece("Same text.", locator="B", section=("B",))])
    assert a["id"] != b["id"]
    assert ingest.duplicate_ids([a, b]) == []


def test_duplicate_ids_names_source_and_locator():
    src = make_source("dupe_src")
    recs = ingest.chunk_records(src, [piece("Repeated paragraph.", locator="Dosing")] * 2)
    (msg,) = ingest.duplicate_ids(recs)
    assert "dupe_src" in msg and "Dosing" in msg and "Repeated paragraph." in msg


def test_duplicate_ids_tolerates_old_textbook_records():
    old = {"id": "x", "book": "mohs_layers", "page": 4, "kind": "prose", "text": "t"}
    (msg,) = ingest.duplicate_ids([old, dict(old)])
    assert "mohs_layers" in msg and "4" in msg


def test_build_source_textbook_uses_registry_path(monkeypatch, tmp_path):
    seen = {}

    def fake_build_chunks(book_id, pdf_path=None, domain="clinical"):
        seen.update(book_id=book_id, pdf_path=pdf_path, domain=domain)
        return [{"id": "t1"}], {1: []}

    monkeypatch.setattr(ingest, "build_chunks", fake_build_chunks)
    monkeypatch.setattr(ingest, "load_units", lambda s: pytest.fail("loaders used for textbook"))
    book = make_source("mohs_layers", fmt="pdf_textbook", path=tmp_path / "book.pdf")
    assert ingest.build_source(book) == ([{"id": "t1"}], {1: []})
    assert seen == {"book_id": "mohs_layers", "pdf_path": tmp_path / "book.pdf",
                    "domain": "clinical"}


def test_build_source_other_formats_pack_with_config(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_TARGET_CHARS", 60)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_CHARS", 0)
    monkeypatch.setattr(config, "CHUNK_MIN_CHARS", 5)
    units = [Unit("First sentence is here. Second sentence is here. Third one is here too.",
                  section=("S",), locator="S"),
             Unit("Table: 17311 | first stage", section=("S",), locator="S", kind="table")]
    monkeypatch.setattr(ingest, "load_units", lambda s: units)
    records, figures = ingest.build_source(make_source())
    assert figures == {}
    assert [r["kind"] for r in records] == ["prose", "prose", "table"]
    assert all(len(r["text"]) <= 60 for r in records)
    assert all(r["source"] == "demo" and r["section"] == ["S"] for r in records)


def test_index_id_follows_ids_and_order():
    assert ingest.index_id(["a", "b"]) != ingest.index_id(["b", "a"])
    assert len(ingest.index_id(["a"])) == 12
    assert ingest.index_id(["a", "b"]) == ingest.index_id(["a", "b"])


def test_build_manifest():
    a, b = make_source("aaa", year=2026), make_source("bbb", domain="operations")
    chunks = [{"id": "1", "source": "aaa"}, {"id": "2", "source": "aaa"},
              {"id": "3", "source": "bbb"}]
    m = ingest.build_manifest([a, b], chunks, built_at="2026-10-13T00:00:00+00:00")
    assert m["index_id"] == ingest.index_id(["1", "2", "3"])
    assert m["n_chunks"] == 3 and m["built_at"] == "2026-10-13T00:00:00+00:00"
    assert m["sources"][0] == {"id": "aaa", "label": "aaa label", "title": "aaa title",
                               "type": "review", "domain": "clinical", "format": "markdown",
                               "license": "CC BY 4.0", "url": "https://example.org/aaa",
                               "year": 2026, "n_chunks": 2}
    assert [s["n_chunks"] for s in m["sources"]] == [2, 1]


def test_keep_built_at_only_when_nothing_changed():
    old = {"index_id": "x", "built_at": "old", "sources": []}
    assert ingest.keep_built_at(dict(old, built_at="new"), old)["built_at"] == "old"
    changed = {"index_id": "y", "built_at": "new", "sources": []}
    assert ingest.keep_built_at(changed, old)["built_at"] == "new"
    assert ingest.keep_built_at(changed, None)["built_at"] == "new"


def test_kind_counts_keeps_first_appearance_order():
    chunks = [{"kind": k} for k in ("prose", "caption", "prose", "reference")]
    assert list(ingest.kind_counts(chunks).items()) == [
        ("prose", 2), ("caption", 1), ("reference", 1)]


def test_abbreviate():
    assert ingest.abbreviate("CC BY-NC-ND 4.0") == "CC BY-NC-ND 4.0"
    assert ingest.abbreviate("Copyrighted. Licensed copy for local use.") == "Copyrighted"
    long = ingest.abbreviate("x" * 80, width=20)
    assert len(long) == 20 and long.endswith("...")


# ------------------------------------------------------------- --list
def test_format_status_rows_and_problems():
    registry = [
        make_source("indexed_ok"),
        make_source("missing_file", url="https://example.org/get"),
        make_source("not_indexed_yet"),
        make_source("disabled_but_indexed", enabled=False),
        make_source("off_and_absent", enabled=False),
    ]
    present = {"indexed_ok": True, "missing_file": False, "not_indexed_yet": True,
               "disabled_but_indexed": True, "off_and_absent": False}
    manifest = {"index_id": "abc123def456", "built_at": "2026-10-13", "n_chunks": 15,
                "sources": [{"id": "indexed_ok", "n_chunks": 10},
                            {"id": "disabled_but_indexed", "n_chunks": 4},
                            {"id": "gone_from_yaml", "n_chunks": 1}]}
    out = ingest.format_status(registry, manifest, present,
                               extra_problems=[None, "vectors stale"])
    lines = out.splitlines()

    assert "abc123def456" in lines[0] and "15 chunks" in lines[0]
    header = next(l for l in lines if l.startswith("id "))
    for col in ("on", "file", "format", "domain", "chunks", "license"):
        assert col in header
    row = next(l for l in lines if l.startswith("indexed_ok "))
    assert row.split()[:6] == ["indexed_ok", "yes", "yes", "markdown", "clinical", "10"]
    assert "CC BY 4.0" in row
    assert next(l for l in lines if l.startswith("off_and_absent ")).split()[:6] == [
        "off_and_absent", "no", "no", "markdown", "clinical", "-"]

    problems = out.split("problems:")[1]
    assert "missing_file: enabled but the file is missing" in problems
    assert "https://example.org/get" in problems
    assert "not_indexed_yet: enabled but not in the index" in problems
    assert "disabled_but_indexed: disabled but still in the index" in problems
    assert "gone_from_yaml: in the index but not in sources.yaml" in problems
    assert "vectors stale" in problems
    assert "indexed_ok:" not in problems and "off_and_absent:" not in problems


def test_format_status_clean_and_no_manifest():
    registry = [make_source("only")]
    manifest = {"index_id": "i", "n_chunks": 1, "sources": [{"id": "only", "n_chunks": 1}]}
    assert ingest.format_status(registry, manifest, {"only": True}).endswith("no problems")
    out = ingest.format_status(registry, {}, {"only": True})
    assert out.startswith("no manifest.json")
    assert "run: python3 -m src.ingest" in out


def test_vector_problem():
    ids = ["a", "b"]
    assert ingest.vector_problem(ids, ["a", "b"], 2) is None
    assert ingest.vector_problem(None, None, None) is None          # no index at all
    assert "no vectors.npy" in ingest.vector_problem(ids, None, None)
    assert "out of date" in ingest.vector_problem(ids, ["b", "a"], 2)
    assert "3 rows for 2 chunks" in ingest.vector_problem(ids, None, 3)
    assert "no vector_keys.json" in ingest.vector_problem(ids, None, 2)


# ------------------------------------------------------- main(), end to end
REGISTRY_YAML = """
sources:
  - id: first_doc
    title: First document
    label: "First"
    type: review
    domain: clinical
    format: markdown
    path: raw/first.md
    license: CC BY 4.0
  - id: off_doc
    title: Disabled document
    type: policy
    domain: operations
    format: html
    path: raw/missing.html
    license: public domain
    enabled: false
  - id: second_doc
    title: Second document
    type: policy
    domain: operations
    format: html
    path: raw/second.html
    license: public domain
"""


@pytest.fixture
def registry_dir(tmp_path, monkeypatch, index_dir):
    """A two-enabled-source registry in tmp_path, with fake loaders."""
    pytest.importorskip("yaml")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "first.md").write_text("# placeholder")
    (tmp_path / "raw" / "second.html").write_text("<p>placeholder</p>")
    (tmp_path / "sources.yaml").write_text(REGISTRY_YAML)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "SOURCES_PATH", tmp_path / "sources.yaml")

    units = {
        "first_doc": [Unit("Mohs surgery removes skin cancer in stages, checking every margin.",
                           ("Overview",), "Overview")],
        "second_doc": [Unit("Report 17311 for the first stage on the head.", ("Coding",), "Coding"),
                       Unit("Report 17312 for each additional stage.", ("Coding",), "Coding")],
    }
    monkeypatch.setattr(ingest, "load_units", lambda s: list(units[s.id]))
    return units


def test_main_builds_chunks_and_manifest_in_registry_order(registry_dir, capsys):
    chunks, manifest = ingest.main([])
    on_disk = json.loads(config.CHUNKS_PATH.read_text())
    assert on_disk == chunks
    assert config.CHUNKS_PATH.read_text() == json.dumps(chunks, indent=1)
    assert [c["source"] for c in chunks] == ["first_doc", "second_doc"]
    assert [c["domain"] for c in chunks] == ["clinical", "operations"]

    saved = json.loads(config.MANIFEST_PATH.read_text())
    assert saved == manifest
    assert saved["index_id"] == ingest.index_id([c["id"] for c in chunks])
    assert [(s["id"], s["n_chunks"]) for s in saved["sources"]] == [
        ("first_doc", 1), ("second_doc", 1)]
    assert saved["sources"][0]["label"] == "First"

    out = capsys.readouterr().out
    assert "first_doc" in out and "second_doc" in out and "{'prose': 1}" in out


def test_main_rerun_rewrites_identical_files(registry_dir):
    ingest.main([])
    before = (config.CHUNKS_PATH.read_text(), config.MANIFEST_PATH.read_text())
    ingest.main([])
    assert (config.CHUNKS_PATH.read_text(), config.MANIFEST_PATH.read_text()) == before


def test_main_refuses_duplicate_ids_and_keeps_old_index(registry_dir, monkeypatch):
    config.CHUNKS_PATH.write_text("[]")
    unit = Unit("Exactly the same paragraph, twice in the same section.", ("Coding",), "Coding")
    monkeypatch.setattr(ingest, "load_units",
                        lambda s: [unit, Unit("x" * 3000, ("Other",), "Other"), unit]
                        if s.id == "second_doc" else [])
    # The repeated unit is in the same section but not adjacent, so pack()
    # keeps it as a separate chunk with the same text -> the same ID.
    with pytest.raises(ValueError) as err:
        ingest.main([])
    assert "second_doc" in str(err.value) and "Coding" in str(err.value)
    assert config.CHUNKS_PATH.read_text() == "[]"
    assert not config.MANIFEST_PATH.exists()


def test_main_reports_missing_enabled_file(registry_dir, tmp_path):
    (tmp_path / "raw" / "second.html").unlink()
    with pytest.raises(SystemExit) as err:
        ingest.main([])
    assert "second_doc" in str(err.value)
    assert not config.CHUNKS_PATH.exists()


def test_main_list(registry_dir, capsys):
    ingest.main(["--list"])
    out = capsys.readouterr().out
    assert not config.CHUNKS_PATH.exists()            # --list builds nothing
    assert "first_doc" in out and "off_doc" in out and "no manifest.json" in out

    ingest.main([])
    capsys.readouterr()
    ingest.main(["--list"])
    out = capsys.readouterr().out
    assert "enabled but not in the index" not in out
    assert "no vectors.npy" in out


def test_main_adopts_legacy_vectors_before_overwriting(registry_dir):
    """The old chunks.json is gone after ingest, so adoption has to happen
    first - otherwise the old vectors could never be matched to text."""
    legacy = [{"id": f"old{i}", "book": "mohs_layers", "page": i, "kind": "prose",
               "text": f"legacy text {i}", "figures": []} for i in range(3)]
    config.CHUNKS_PATH.write_text(json.dumps(legacy, indent=1))
    np.save(config.VECTORS_PATH, np.eye(3, dtype=np.float32))

    ingest.main([])
    keys = json.loads(config.VECTOR_KEYS_PATH.read_text())
    assert keys["ids"] == ["old0", "old1", "old2"]
    assert keys["keys"] == [embed.embed_key(config.EMBEDDING_MODEL, f"legacy text {i}")
                            for i in range(3)]


def test_build_chunks_signature_keeps_books_fallback():
    """Older callers ran build_chunks() / build_chunks("mohs_layers")."""
    import inspect
    params = inspect.signature(ingest.build_chunks).parameters
    assert params["book_id"].default == "mohs_layers"
    assert params["pdf_path"].default is None


# ================================================================= embed
def test_embed_text():
    assert embed.embed_text({"text": "plain"}) == "plain"
    assert embed.embed_text({"text": "plain", "section": []}) == "plain"
    assert embed.embed_text({"text": "body", "section": ["Coding", "Stages"]}) == \
        "Coding > Stages\nbody"


def test_embed_text_without_headings(monkeypatch):
    monkeypatch.setattr(config, "EMBED_SECTION_HEADINGS", False)
    assert embed.embed_text({"text": "body", "section": ["Coding"]}) == "body"


def test_embed_key():
    k = embed.embed_key("model-a", "text")
    assert len(k) == 16 and int(k, 16) >= 0
    assert embed.embed_key("model-a", "text") == k
    assert embed.embed_key("model-b", "text") != k
    assert embed.embed_key("model-a", "text.") != k


def test_plan_reuses_matching_keys():
    reuse, todo = embed.plan(["b", "new", "a"], ["a", "b", "c"])
    assert reuse == {0: 1, 2: 0}
    assert todo == [1]


def test_plan_maps_repeated_keys_in_order():
    """Two chunks with the same text (two pages): an unchanged index must
    map every row onto itself, not both onto the first."""
    reuse, todo = embed.plan(["x", "y", "x"], ["x", "y", "x"])
    assert reuse == {0: 0, 1: 1, 2: 2} and todo == []
    reuse, _ = embed.plan(["x", "x", "x"], ["x", "x"])
    assert reuse == {0: 0, 1: 1, 2: 1}


def test_assemble_reuses_rows_and_encodes_only_new_text():
    old_vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    encode = FakeEncoder()
    vectors, n_reused, n_embedded = embed.assemble(
        ["kept b", "fresh", "kept a", "fresh"], ["kb", "kf", "ka", "kf"],
        ["ka", "kb"], old_vectors, encode)
    assert (n_reused, n_embedded) == (2, 2)
    assert encode.calls == [["fresh"]]                  # each new text encoded once
    assert vectors.dtype == np.float32 and vectors.shape == (4, 3)
    assert np.array_equal(vectors[0], old_vectors[1])
    assert np.array_equal(vectors[2], old_vectors[0])
    assert np.array_equal(vectors[1], vectors[3])
    assert np.array_equal(vectors[1], encode(["fresh"])[0])


def test_assemble_never_calls_encode_when_nothing_is_new():
    old_vectors = np.arange(6, dtype=np.float32).reshape(2, 3)

    def encode(texts):
        raise AssertionError("model used although every key was known")

    vectors, n_reused, n_embedded = embed.assemble(["a", "b"], ["ka", "kb"], ["ka", "kb"],
                                                   old_vectors, encode)
    assert (n_reused, n_embedded) == (2, 0)
    assert np.array_equal(vectors, old_vectors)


def test_assemble_from_scratch_and_empty():
    encode = FakeEncoder()
    vectors, n_reused, n_embedded = embed.assemble(["a", "bb"], ["1", "2"], [], None, encode)
    assert (n_reused, n_embedded) == (0, 2) and vectors.shape == (2, 3)
    vectors, _, _ = embed.assemble([], [], [], None, encode)
    assert vectors.shape == (0, 0)


def write_chunks(chunks):
    config.CHUNKS_PATH.write_text(json.dumps(chunks, indent=1))


def demo_chunks(n=3, section=None):
    out = []
    for i in range(n):
        c = {"id": f"id{i}", "source": "demo", "page": None, "kind": "prose",
             "text": f"chunk text number {i}", "figures": []}
        if section is not None:
            c["section"] = section
        out.append(c)
    return out


def test_main_incremental(index_dir):
    model = config.EMBEDDING_MODEL
    chunks = demo_chunks(3, section=["Sec"])
    write_chunks(chunks)

    encode = FakeEncoder()
    assert embed.main([], encode=encode) == {"reused": 0, "embedded": 3, "written": True}
    assert encode.calls == [[f"Sec\nchunk text number {i}" for i in range(3)]]
    first = np.load(config.VECTORS_PATH)
    keys = json.loads(config.VECTOR_KEYS_PATH.read_text())
    assert keys == {"embedding_model": model, "ids": ["id0", "id1", "id2"],
                    "keys": [embed.embed_key(model, f"Sec\nchunk text number {i}")
                             for i in range(3)]}

    # nothing changed: no model call, no write
    stamp = config.VECTORS_PATH.stat().st_mtime_ns
    assert embed.main([], encode=FakeEncoder()) == {"reused": 3, "embedded": 0, "written": False}
    assert config.VECTORS_PATH.stat().st_mtime_ns == stamp

    # one chunk changed, one removed, order changed
    chunks[1]["text"] = "an edited chunk"
    chunks[1]["id"] = "id1b"
    write_chunks([chunks[2], chunks[1]])
    encode = FakeEncoder()
    assert embed.main([], encode=encode) == {"reused": 1, "embedded": 1, "written": True}
    assert encode.calls == [["Sec\nan edited chunk"]]
    vectors = np.load(config.VECTORS_PATH)
    assert np.array_equal(vectors[0], first[2])
    assert json.loads(config.VECTOR_KEYS_PATH.read_text())["ids"] == ["id2", "id1b"]


def test_main_full_reembeds_everything(index_dir):
    write_chunks(demo_chunks(2))
    embed.main([], encode=FakeEncoder())
    encode = FakeEncoder()
    assert embed.main(["--full"], encode=encode)["embedded"] == 2
    assert encode.n_encoded == 2


def test_main_model_change_reembeds(index_dir, monkeypatch):
    write_chunks(demo_chunks(2))
    embed.main([], encode=FakeEncoder())
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "some/other-model")
    assert embed.main([], encode=FakeEncoder())["embedded"] == 2
    assert json.loads(config.VECTOR_KEYS_PATH.read_text())["embedding_model"] == "some/other-model"


def test_main_ignores_a_cache_with_the_wrong_row_count(index_dir):
    write_chunks(demo_chunks(2))
    embed.main([], encode=FakeEncoder())
    np.save(config.VECTORS_PATH, np.zeros((5, 3), dtype=np.float32))
    assert embed.main([], encode=FakeEncoder())["embedded"] == 2


def test_main_adopts_a_legacy_index_without_the_model(index_dir):
    """Today's state: vectors.npy, an old-style chunks.json, no key file."""
    legacy = [{"id": f"t{i}", "book": "mohs_layers", "page": i, "kind": "prose",
               "text": f"textbook block {i}", "figures": []} for i in range(4)]
    write_chunks(legacy)
    original = np.random.default_rng(0).random((4, 3)).astype(np.float32)
    np.save(config.VECTORS_PATH, original)

    def encode(texts):
        raise AssertionError("legacy vectors should be reused")

    assert embed.main([], encode=encode) == {"reused": 4, "embedded": 0, "written": False}
    assert np.array_equal(np.load(config.VECTORS_PATH), original)
    assert json.loads(config.VECTOR_KEYS_PATH.read_text())["ids"] == ["t0", "t1", "t2", "t3"]


def test_adopt_legacy_requires_matching_counts(index_dir):
    chunks = demo_chunks(2)
    assert embed.adopt_legacy(chunks, np.zeros((3, 3))) is False
    assert not config.VECTOR_KEYS_PATH.exists()
    assert embed.adopt_legacy(chunks, np.zeros((2, 3)), model="m") is True
    keys = json.loads(config.VECTOR_KEYS_PATH.read_text())
    assert keys["embedding_model"] == "m" and keys["ids"] == ["id0", "id1"]


def legacy_chunks(n):
    """Records as the old ingest wrote them: no source/domain/locator/section."""
    return [{"id": f"t{i}", "book": "mohs_layers", "page": i, "kind": "prose",
             "text": f"textbook block {i}", "figures": []} for i in range(n)]


def test_adopt_existing_index_conditions(index_dir):
    write_chunks(legacy_chunks(2))
    assert embed.adopt_existing_index() is False    # no vectors.npy
    np.save(config.VECTORS_PATH, np.zeros((3, 3), dtype=np.float32))
    assert embed.adopt_existing_index() is False    # 2 chunks, 3 rows: pairing broken
    np.save(config.VECTORS_PATH, np.zeros((2, 3), dtype=np.float32))
    assert embed.adopt_existing_index() is True
    assert embed.adopt_existing_index() is False    # key file exists now: never overwritten


def test_a_new_style_index_without_keys_is_not_adopted(index_dir):
    """A new-style chunks.json without keys may already differ from the
    vectors (keys deleted by hand, or a crash between the two writes), so
    neither ingest nor embed trusts the pairing: embed starts over."""
    write_chunks(demo_chunks(2))                    # has "source": new ingest wrote it
    np.save(config.VECTORS_PATH, np.zeros((2, 3), dtype=np.float32))
    assert embed.adopt_existing_index() is False
    assert embed.main([], encode=FakeEncoder())["embedded"] == 2


def test_ingest_then_embed_after_adding_a_source(registry_dir):
    """Enable a second document: only its chunks go through the model."""
    ingest.main([])
    encode = FakeEncoder()
    embed.main([], encode=encode)
    first = np.load(config.VECTORS_PATH)
    assert encode.n_encoded == 2

    registry_dir["third"] = [Unit("Brand new paragraph about repairs after Mohs surgery.",
                                  ("Repair",), "Repair")]
    yaml_text = config.SOURCES_PATH.read_text() + """
  - id: third
    title: Third document
    type: protocol
    domain: operations
    format: markdown
    path: raw/first.md
    license: internal
"""
    config.SOURCES_PATH.write_text(yaml_text)
    ingest.main([])
    encode = FakeEncoder()
    assert embed.main([], encode=encode) == {"reused": 2, "embedded": 1, "written": True}
    assert encode.calls == [["Repair\nBrand new paragraph about repairs after Mohs surgery."]]
    assert np.array_equal(np.load(config.VECTORS_PATH)[:2], first)


# ======================================================== the real index
def _real_chunks():
    if not config.CHUNKS_PATH.exists():
        pytest.skip("no chunks.json")
    return json.loads(config.CHUNKS_PATH.read_text())


def test_real_index_textbook_records_follow_the_contract():
    """Old records lack the new fields; new ones must have the right values."""
    for c in _real_chunks():
        assert c.get("source", c["book"]) == c["book"]
        if "locator" in c and c.get("source", c["book"]) == config.DEFAULT_SOURCE:
            assert c["locator"] == f"p. {c['page']}" and c["section"] == []


def test_real_index_vector_keys_match_if_present():
    chunks = _real_chunks()
    if not config.VECTOR_KEYS_PATH.exists():
        pytest.skip("no vector_keys.json yet - legacy index")
    keys = json.loads(config.VECTOR_KEYS_PATH.read_text())
    assert keys["ids"] == [c["id"] for c in chunks]
    assert len(keys["keys"]) == len(np.load(config.VECTORS_PATH, mmap_mode="r"))

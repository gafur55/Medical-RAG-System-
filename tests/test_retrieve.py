"""Tests for src/retrieve.py - no embedding model, no real index.

get_index() and embed_query() are swapped for a handful of 3-d unit
vectors, so every ranking below can be checked by hand. The index checks
at the bottom write their fake index files into tmp_path.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from src import config, embed, retrieve


def unit(*xs):
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


QUERY = unit(1, 0, 0)


def chunk(cid, text="plain body text about flaps", page=1, kind="prose", **extra):
    return {"id": cid, "book": "mohs_layers", "page": page, "kind": kind,
            "text": text, "figures": [], **extra}


@pytest.fixture
def index(monkeypatch):
    """Install a fake index; returns a function taking (chunks, vectors)."""
    def install(chunks, vectors):
        vectors = np.array(vectors, dtype=float)
        monkeypatch.setattr(retrieve, "get_index", lambda: (chunks, vectors))
        monkeypatch.setattr(retrieve, "embed_query", lambda q, use_prefix=None: QUERY)
        return chunks, vectors
    return install


def ids(hits):
    return [h["chunk"]["id"] for h in hits]


# ------------------------------------------------------------ defaults
def test_defaults_are_plain_cosine_ranking(index):
    chunks, vectors = index(
        [chunk("a"), chunk("b", page=2), chunk("ref", kind="reference"),
         chunk("c", page=3, kind="caption"), chunk("d", page=4)],
        [unit(1, 1, 0), unit(1, 0.1, 0), unit(1, 0, 0), unit(1, 2, 0), unit(0, 1, 0)])

    hits = retrieve.search("q", k=10)

    cos = vectors @ QUERY
    expected = [i for i in np.argsort(-cos) if chunks[i]["kind"] in config.RETRIEVE_KINDS]
    assert ids(hits) == [chunks[i]["id"] for i in expected]
    assert [h["score"] for h in hits] == [float(cos[i]) for i in expected]
    assert [h["rank"] for h in hits] == [1, 2, 3, 4]


def test_k_defaults_to_config_top_k(index, monkeypatch):
    index([chunk(str(i), page=i) for i in range(6)], [unit(1, i, 0) for i in range(6)])
    monkeypatch.setattr(config, "TOP_K", 2)
    assert ids(retrieve.search("q")) == ["0", "1"]


# ------------------------------------------------------- domains, kinds
def test_domain_filter(index):
    index([chunk("old"),                                  # no domain -> clinical
           chunk("ops", source="clinic_sop", domain="operations"),
           chunk("clin", source="statpearls", domain="clinical")],
          [unit(1, 0.1, 0), unit(1, 0, 0), unit(1, 0.2, 0)])

    assert ids(retrieve.search("q", k=5)) == ["ops", "old", "clin"]
    assert ids(retrieve.search("q", k=5, domains=("clinical",))) == ["old", "clin"]
    assert ids(retrieve.search("q", k=5, domains=("operations",))) == ["ops"]
    # an empty tuple means "no domain", not "use the default"
    assert retrieve.search("q", k=5, domains=()) == []


def test_kinds_empty_tuple_is_not_the_default(index):
    index([chunk("a"), chunk("b", kind="caption")], [unit(1, 0, 0), unit(1, 1, 0)])
    assert ids(retrieve.search("q", kinds=("caption",))) == ["b"]
    assert retrieve.search("q", kinds=()) == []


# -------------------------------------------------------- front matter
def test_front_matter_knob_reads_config_at_call_time(index, monkeypatch):
    index([chunk("kw", "Keywords: flap, graft, scalp"), chunk("body", page=2)],
          [unit(1, 0, 0), unit(1, 0.5, 0)])

    assert ids(retrieve.search("q")) == ["kw", "body"]
    assert ids(retrieve.search("q", exclude_front_matter=True)) == ["body"]

    monkeypatch.setattr(config, "EXCLUDE_FRONT_MATTER", True)
    assert ids(retrieve.search("q")) == ["body"]
    assert ids(retrieve.search("q", exclude_front_matter=False)) == ["kw", "body"]


ABSTRACT = ("Scalp reconstruction is commonly required after Mohs surgery. Creative use "
            "of a broad range of reconstruction options rewards both physician and patient.")


@pytest.mark.parametrize("text", [
    "Abstract The cheek is a common site of Mohs surgery and subsequent reconstruction.",
    "Keywords: scalp, reconstruction, anatomy, second intention healing, skin grafts",
    "This chapter focuses on preservation of the symmetrical projection of the ear.",
    "Mastering nasal reconstruction requires anatomy. Herein, the authors begin by presenting a review.",
    "advancement flaps 38 advancement flaps 107, 199 aesthetic subunits and defects 158 alar "
    "sill 209 algorithm - for closure of periocular defects 180 - reconstructive options 159",
    "Ashwin Agarwal, MD Department of Dermatology Perelman School of Medicine University of "
    "Pennsylvania Philadelphia, Pennsylvania, USA",
    "Christopher J. Miller, Joseph F. Sobanko, Nicole Howe, Thuzar Shin, Jeremy R. Etzkorn, "
    "and H. William Higgins II",
    "This book, including all parts thereof, is legally protected by copyright.",
    "Georg Thieme Verlag KG Stuttgart, Germany, customerservice@thieme.de",
    "I acknowledge the hard work of former and current medical assistants and nurses.",
    "Video 8.1: Repair of a large cutaneous lateral upper lip defect with a crescentic flap.",
])
def test_is_front_matter_flags(text):
    assert retrieve.is_front_matter(chunk("x", text))


@pytest.mark.parametrize("text, kind", [
    ("Rotation flaps on the face are most commonly used to repair medial cheek defects.", "prose"),
    # Key points are content - the p. 102 ones are gold for Q13.
    ("• Ideal undermining planes vary with the anatomic location of the face.", "prose"),
    # Cross-references to the chapter are not overviews (p. 283 / p. 530 are gold).
    ("Burow's grafts, previously discussed in this chapter, are the easiest way.", "prose"),
    ("Pain regimens are beyond the scope of this chapter. Patients are given a number.", "prose"),
    ("Please refer to Chapter 12 of this book for such information.", "prose"),
    # A figure credit is not a copyright page.
    ("Fig. 1.15 Basic eyelid anatomy: frontal view. Reproduced from Schuenke M, Schulte E. "
     "Illustrations by Voll M and Wesker K. © Thieme 2020.", "caption"),
    # Numbers in instructions are not index entries.
    ("Mix 1 part white vinegar to 2 cups of warm water, or 1 capful to 1 cupful of water, "
     "and soak the gauze for 10 minutes before use.", "prose"),
    ("Fig. 3.5 (a) Defect on the midline nasal tip of a 74-year-old woman. (b) Fusiform "
     "repair. (Courtesy of Stanislav N. Tolkachjov, MD.)", "caption"),
])
def test_is_front_matter_leaves_content_alone(text, kind):
    assert not retrieve.is_front_matter(chunk("x", text, kind=kind))


def test_abstract_found_by_the_keyword_list_below_it():
    abstract = chunk("a", ABSTRACT, page=220)
    keywords = chunk("k", "Keywords: scalp, reconstruction", page=220)
    assert not retrieve.is_front_matter(abstract)
    assert retrieve.is_front_matter(abstract, next_chunk=keywords)
    # keyword list on the next page: no evidence about this chunk
    assert not retrieve.is_front_matter(abstract, next_chunk=chunk("k2", keywords["text"], page=221))
    mask = retrieve.front_matter_mask([abstract, keywords, chunk("b", page=221)])
    assert mask.tolist() == [True, True, False]


def test_front_matter_section_headings():
    c = chunk("x", "We thank the nurses.", page=None, source="paper", section=["Acknowledgments"])
    assert retrieve.is_front_matter(c)
    c = chunk("y", "Wound care instructions.", page=None, source="sop", section=["Aftercare"])
    assert not retrieve.is_front_matter(c)


# --------------------------------------------------------- max_per_page
def test_max_per_page_diversifies(index):
    index([chunk("p1a", page=1), chunk("p1b", page=1), chunk("p1c", page=1),
           chunk("p2", page=2), chunk("p3", page=3)],
          [unit(1, 0, 0), unit(1, 0.1, 0), unit(1, 0.2, 0), unit(1, 0.3, 0), unit(1, 0.4, 0)])

    assert ids(retrieve.search("q", k=3)) == ["p1a", "p1b", "p1c"]
    assert ids(retrieve.search("q", k=3, max_per_page=1)) == ["p1a", "p2", "p3"]
    assert ids(retrieve.search("q", k=3, max_per_page=2)) == ["p1a", "p1b", "p2"]
    # 0 = explicitly no cap, even if config sets one
    assert ids(retrieve.search("q", k=3, max_per_page=0)) == ["p1a", "p1b", "p1c"]


def test_max_per_page_counts_per_source_and_locator(index):
    index([chunk("book12a", page=12), chunk("book12b", page=12),
           chunk("pdf12", page=12, source="paper"),
           chunk("htmlA1", page=None, source="statpearls", locator="Indications"),
           chunk("htmlA2", page=None, source="statpearls", locator="Indications"),
           chunk("htmlB", page=None, source="statpearls", locator="Technique")],
          [unit(1, 0, 0), unit(1, 0.1, 0), unit(1, 0.2, 0),
           unit(1, 0.3, 0), unit(1, 0.4, 0), unit(1, 0.5, 0)])

    assert ids(retrieve.search("q", k=6, max_per_page=1)) == ["book12a", "pdf12", "htmlA1", "htmlB"]


def test_max_per_page_is_prefix_consistent(index):
    index([chunk(str(i), page=i % 2) for i in range(8)], [unit(1, i / 10, 0) for i in range(8)])
    deep = ids(retrieve.search("q", k=8, max_per_page=3))
    assert ids(retrieve.search("q", k=4, max_per_page=3)) == deep[:4]


# --------------------------------------------------------------- hybrid
def test_hybrid_fusion_ordering(index):
    # cosine slightly prefers "generic"; only "rare" contains the rare word
    index([chunk("generic", "options for reconstruction of a defect", page=1),
           chunk("rare", "the karapandzic flap restores the oral sphincter", page=2),
           chunk("other", "wound care after surgery", page=3)],
          [unit(1, 0.30, 0), unit(1, 0.35, 0), unit(1, 2, 0)])

    assert ids(retrieve.search("karapandzic flap", k=3)) == ["generic", "rare", "other"]
    assert ids(retrieve.search("karapandzic flap", k=3, hybrid_weight=0.5)) == ["rare", "generic", "other"]

    # fused score = 0.5 * minmax(cosine) + 0.5 * minmax(bm25): "generic" has
    # the best cosine (1.0 after scaling) and no query word (0.0)
    hits = retrieve.search("karapandzic flap", k=3, hybrid_weight=0.5)
    assert all(0.0 <= h["score"] <= 1.0 for h in hits)
    assert hits[1]["score"] == pytest.approx(0.5)
    assert hits[2]["score"] == pytest.approx(0.0)
    # w = 1 is pure BM25: "other" and "generic" tie at 0, "rare" leads
    assert ids(retrieve.search("karapandzic flap", k=1, hybrid_weight=1.0)) == ["rare"]

    with pytest.raises(ValueError):
        retrieve.search("q", hybrid_weight=1.5)


def test_hybrid_still_excludes_filtered_chunks(index):
    index([chunk("ref", "karapandzic karapandzic", kind="reference"),
           chunk("a", "karapandzic flap", page=2), chunk("b", "lip repair", page=3)],
          [unit(1, 0, 0), unit(1, 1, 0), unit(1, 0.5, 0)])
    hits = retrieve.search("karapandzic", k=5, hybrid_weight=1.0)
    assert ids(hits) == ["a", "b"]


def test_hybrid_reads_config(index, monkeypatch):
    index([chunk("generic", "options for reconstruction of a defect", page=1),
           chunk("rare", "the karapandzic flap", page=2),
           chunk("other", "wound care after surgery", page=3)],
          [unit(1, 0.30, 0), unit(1, 0.35, 0), unit(1, 2, 0)])
    monkeypatch.setattr(config, "HYBRID_WEIGHT", 0.5)
    assert ids(retrieve.search("karapandzic flap"))[0] == "rare"
    assert ids(retrieve.search("karapandzic flap", hybrid_weight=0.0))[0] == "generic"


def test_bm25_prefers_rare_terms_and_ignores_stopwords():
    docs = [retrieve.tokenize(t) for t in
            ["the flap and the graft", "karapandzic flap", "graft graft graft donor site"]]
    bm25 = retrieve.build_bm25(docs)
    s = retrieve.bm25_scores(bm25, retrieve.tokenize("When is the Karapandzic flap used?"))
    assert s.argmax() == 1
    assert retrieve.tokenize("What is the flap?") == ["flap"]
    assert retrieve.bm25_scores(bm25, retrieve.tokenize("what is it")).tolist() == [0, 0, 0]


def test_minmax_and_fuse():
    allowed = np.array([True, True, False, True])
    assert retrieve.minmax(np.array([2.0, 4.0, 99.0, 3.0]), allowed).tolist() == [0, 1, 0, 0.5]
    # nothing to rank by on one side -> that side adds nothing
    assert retrieve.minmax(np.zeros(4), allowed).tolist() == [0, 0, 0, 0]
    fused = retrieve.fuse(np.array([0.8, 0.7, 0.0, 0.6]), np.array([0.0, 2.0, 0.0, 1.0]),
                          allowed, 0.3)
    # cosine scaled to [1, .5, -, 0], bm25 to [0, 1, -, .5]
    assert fused[[0, 1, 3]] == pytest.approx([0.7, 0.35 + 0.3, 0.15])


# ---------------------------------------------------------------- index
def key_file(chunks, vectors=None):
    """vector_keys.json as embed.py writes it for chunks (and, given the
    vectors, with their digest)."""
    record = embed.keys_record(chunks, config.EMBEDDING_MODEL)
    return embed.with_digest(record, vectors) if vectors is not None else record


def test_check_alignment():
    chunks = [chunk("a"), chunk("b")]
    vectors = np.zeros((2, 3))
    keys = key_file(chunks, vectors)
    retrieve.check_alignment(chunks, vectors, keys)        # fine
    retrieve.check_alignment(chunks, vectors, None)        # legacy, counts match
    ids_only = {"embedding_model": config.EMBEDDING_MODEL, "ids": ["a", "b"]}
    retrieve.check_alignment(chunks, vectors, ids_only)    # key file without keys or digest

    with pytest.raises(retrieve.IndexMismatch, match="python3 -m src.embed"):
        retrieve.check_alignment(chunks, vectors, {**keys, "ids": ["b", "a"]})
    with pytest.raises(retrieve.IndexMismatch, match="python3 -m src.embed"):
        retrieve.check_alignment(chunks, np.zeros((3, 3)), None)
    with pytest.raises(retrieve.IndexMismatch, match="rows"):
        retrieve.check_alignment(chunks, np.zeros((3, 3)), keys)
    with pytest.raises(retrieve.IndexMismatch, match="embedded with"):
        retrieve.check_alignment(chunks, vectors, {**keys, "embedding_model": "other/model"})
    assert retrieve.alignment_problem(chunks, vectors, keys) is None
    assert "row 0" in retrieve.alignment_problem(chunks, vectors, {**keys, "ids": ["b", "a"]})


def test_check_alignment_catches_a_renamed_heading_with_unchanged_ids():
    """A paginated source's chunk id hashes the page and the text, not the
    heading path - but the embedded text starts with the headings. Rename
    one, re-run ingest and forget embed: the ids still match vectors of the
    old text, the content keys do not."""
    before = [chunk("p3", "Tumor size matters.", page=3, source="cms", section=["Indications"])]
    keys = key_file(before)
    after = [dict(before[0], section=["Indications for Mohs"])]
    retrieve.check_alignment(before, np.zeros((1, 3)), keys)
    with pytest.raises(retrieve.IndexMismatch, match="different text.*row 0.*src.embed"):
        retrieve.check_alignment(after, np.zeros((1, 3)), keys)


def test_check_alignment_catches_a_toggled_section_headings_flag(monkeypatch):
    chunks = [chunk("a"), chunk("s", page=2, source="sp", section=["Technique"])]
    keys = key_file(chunks)
    monkeypatch.setattr(config, "EMBED_SECTION_HEADINGS", False)
    with pytest.raises(retrieve.IndexMismatch, match="EMBED_SECTION_HEADINGS.*row 1"):
        retrieve.check_alignment(chunks, np.zeros((2, 3)), keys)


def test_check_alignment_checks_the_vectors_digest():
    """vectors.npy and vector_keys.json are written one after the other. A
    run that dies in between, or restoring one file alone, leaves keys
    that name the right chunks for rows that are not theirs."""
    chunks = [chunk("a"), chunk("b")]
    written = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    keys = key_file(chunks, written)
    retrieve.check_alignment(chunks, written, keys)
    retrieve.check_alignment(chunks, written.astype(np.float64), keys)   # same values
    with pytest.raises(retrieve.IndexMismatch, match="not the file vector_keys.json"):
        retrieve.check_alignment(chunks, written[::-1], keys)
    # Key files from before the digest existed are still accepted.
    retrieve.check_alignment(chunks, written[::-1], key_file(chunks))


def test_get_index_raises_on_vector_keys_mismatch(tmp_path, monkeypatch):
    chunks = [chunk("a"), chunk("b")]
    (tmp_path / "chunks.json").write_text(json.dumps(chunks))
    np.save(tmp_path / "vectors.npy", np.zeros((2, 3)))
    (tmp_path / "vector_keys.json").write_text(json.dumps(
        {"embedding_model": config.EMBEDDING_MODEL, "ids": ["a", "STALE"], "keys": ["1", "2"]}))
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(config, "VECTORS_PATH", tmp_path / "vectors.npy")
    monkeypatch.setattr(config, "VECTOR_KEYS_PATH", tmp_path / "vector_keys.json")
    monkeypatch.setattr(retrieve, "_chunks", None)
    monkeypatch.setattr(retrieve, "_vectors", None)
    monkeypatch.setattr(retrieve, "_derived", {})

    with pytest.raises(retrieve.IndexMismatch, match="row 1"):
        retrieve.get_index()

    (tmp_path / "vector_keys.json").unlink()        # legacy index: count check only
    loaded, vectors = retrieve.get_index()
    assert [c["id"] for c in loaded] == ["a", "b"] and vectors.shape == (2, 3)


def test_get_index_rejects_vectors_restored_without_their_keys(tmp_path, monkeypatch):
    chunks = [chunk("a"), chunk("b")]
    (tmp_path / "chunks.json").write_text(json.dumps(chunks))
    written = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    (tmp_path / "vector_keys.json").write_text(json.dumps(key_file(chunks, written)))
    np.save(tmp_path / "vectors.npy", written[::-1])         # e.g. git restore vectors.npy
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(config, "VECTORS_PATH", tmp_path / "vectors.npy")
    monkeypatch.setattr(config, "VECTOR_KEYS_PATH", tmp_path / "vector_keys.json")
    monkeypatch.setattr(retrieve, "_chunks", None)
    monkeypatch.setattr(retrieve, "_vectors", None)
    monkeypatch.setattr(retrieve, "_derived", {})

    with pytest.raises(retrieve.IndexMismatch, match="not the file vector_keys.json"):
        retrieve.get_index()
    np.save(tmp_path / "vectors.npy", written)
    assert retrieve.get_index()[1].tolist() == written.tolist()


def test_index_id_follows_ids_and_order():
    assert retrieve.index_id(["a", "b"]) != retrieve.index_id(["b", "a"])
    assert len(retrieve.index_id(["a"])) == 12
    assert retrieve.index_id(["a", "b"]) == retrieve.index_id(["a", "b"])


def test_loaded_index_id_names_the_index_in_memory(tmp_path, monkeypatch):
    """The app loads the index once. Rebuilding the files while it runs
    must not change the id its answers are stamped with."""
    def write(chunks):
        (tmp_path / "chunks.json").write_text(json.dumps(chunks))
        vectors = np.zeros((len(chunks), 3), dtype=np.float32)
        np.save(tmp_path / "vectors.npy", vectors)
        (tmp_path / "vector_keys.json").write_text(json.dumps(key_file(chunks, vectors)))

    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(config, "VECTORS_PATH", tmp_path / "vectors.npy")
    monkeypatch.setattr(config, "VECTOR_KEYS_PATH", tmp_path / "vector_keys.json")
    monkeypatch.setattr(retrieve, "_chunks", None)
    monkeypatch.setattr(retrieve, "_vectors", None)
    monkeypatch.setattr(retrieve, "_derived", {})

    assert retrieve.loaded_index_id() is None                # nothing loaded yet
    write([chunk("a"), chunk("b")])
    retrieve.get_index()
    assert retrieve.loaded_index_id() == retrieve.index_id(["a", "b"])

    write([chunk("a"), chunk("b"), chunk("new", page=9)])    # ingest + embed elsewhere
    retrieve.get_index()
    assert retrieve.loaded_index_id() == retrieve.index_id(["a", "b"])


def test_import_needs_no_model_or_registry():
    """The HF Space imports retrieve (through generate). Checking content
    keys and the digest brings in embed.py - which must stay light."""
    code = ("import sys, src.retrieve; "
            "print(sorted(m for m in ('sentence_transformers', 'yaml', 'src.sources', "
            "'src.ingest', 'src.loaders') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_chunk_field_defaults_for_old_records():
    old = {"id": "x", "book": "mohs_layers", "page": 12, "kind": "prose", "text": "t"}
    assert retrieve.source_of(old) == "mohs_layers"
    assert retrieve.source_of({"id": "y", "page": 1, "text": ""}) == config.DEFAULT_SOURCE
    assert retrieve.domain_of(old) == "clinical"
    assert retrieve.section_of(old) == []
    assert retrieve.locator_of(old) == "p. 12"
    assert retrieve.place_of(old) == ("mohs_layers", 12)
    new = {"id": "z", "source": "sp", "page": None, "locator": "Indications", "text": ""}
    assert retrieve.place_of(new) == ("sp", "Indications")

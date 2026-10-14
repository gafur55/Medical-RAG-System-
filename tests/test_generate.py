"""Tests for src/generate.py - no API key, no network, no embedding model.

search() and the LLM are both swapped for fakes, so these run anywhere.
"""

import hashlib
import json
import re

import numpy as np
import pytest

from src import config, generate, retrieve

MANIFEST = {"index_id": "abc123def456", "sources": [
    {"id": "mohs_layers", "label": "Mohs textbook", "domain": "clinical"},
    {"id": "statpearls_mohs_auc", "label": "StatPearls: Mohs AUC", "domain": "clinical"},
    {"id": "cms_ncci_ch3_2026", "label": "NCCI manual ch. III (2026)", "domain": "operations"},
]}
LABELS = {s["id"]: s["label"] for s in MANIFEST["sources"]}


def fake_hits(*pages, kind="prose"):
    """Textbook hits shaped like today's chunks.json: no source, locator or
    section field - the format every reader must still accept."""
    hits = []
    for rank, page in enumerate(pages, start=1):
        chunk = {"id": f"c{page}", "book": "mohs_layers", "page": page, "kind": kind,
                 "text": f"text on page {page}", "figures": []}
        if kind == "caption":
            chunk["figure_paths"] = [f"data/processed/figures/f{page}.jpeg"]
        hits.append({"rank": rank, "score": 0.9 - rank / 100, "chunk": chunk})
    return hits


def multi_hits():
    """One hit each from the textbook, an HTML source and a policy PDF."""
    chunks = [
        {"id": "t145", "book": "mohs_layers", "source": "mohs_layers",
         "domain": "clinical", "page": 145, "locator": "p. 145", "section": [],
         "kind": "caption", "text": "Bilobed flap for the nasal tip.", "figures": []},
        {"id": "sp1", "book": "statpearls_mohs_auc", "source": "statpearls_mohs_auc",
         "domain": "clinical", "page": None, "locator": "Indications",
         "section": ["Indications"], "kind": "prose",
         "text": "The AUC rate 270 scenarios.", "figures": []},
        {"id": "nc12", "book": "cms_ncci_ch3_2026", "source": "cms_ncci_ch3_2026",
         "domain": "operations", "page": 12, "locator": "p. 12",
         "section": ["E. Mohs Micrographic Surgery"], "kind": "prose",
         "text": "17311 is reported per stage.", "figures": []},
    ]
    return [{"rank": n, "score": 0.8 - n / 100, "chunk": c}
            for n, c in enumerate(chunks, start=1)]


def fake_search(hits, seen=None):
    """Stands in for retrieve.search with the contract's signature."""
    def search(question, k=None, domains=None, **knobs):
        if seen is not None:
            seen.update(question=question, k=k, domains=domains)
        return hits
    return search


def original_build_context(hits):
    """build_context exactly as it was before multi-source support. Page
    style must keep producing this, byte for byte, for textbook chunks."""
    parts = []
    for h in hits:
        c = h["chunk"]
        marker = " [figure caption]" if c["kind"] == "caption" else ""
        parts.append(f"[Page {c['page']}]{marker}\n{c['text']}")
    return "\n\n".join(parts)


@pytest.fixture
def manifest(monkeypatch):
    monkeypatch.setattr(config, "index_manifest", lambda: MANIFEST)
    return MANIFEST


@pytest.fixture
def no_manifest(monkeypatch):
    monkeypatch.setattr(config, "index_manifest", lambda: {})


# ------------------------------------------------------------ citations
def test_extract_citations_formats():
    text = "Flap A works (p. 141). Graft B too (p. 145; p. 141). Slip: page 12."
    assert generate.extract_citations(text) == [12, 141, 145]


def test_extract_citations_none():
    assert generate.extract_citations("The excerpts do not contain the answer.") == []


@pytest.mark.parametrize("text, expected", [
    ("Bill per stage [2].", [2]),
    ("Two sources agree [1, 3].", [1, 3]),
    ("Semicolons too [1; 3].", [1, 3]),
    ("Adjacent brackets [1][3].", [1, 3]),
    ("A range [1-3].", [1, 2, 3]),
    ("An en-dash range [2–4].", [2, 3, 4]),
    ("A mixed list [1, 3-4].", [1, 3, 4]),
    ("Reversed range [3-1].", [1, 2, 3]),
    ("Repeats [2] and [2] again.", [2]),
    ("Out of range [0] [6] [2026] dropped, [5] kept.", [5]),
    ("A range past the end [4-9] is clipped.", [4, 5]),
    ("gpt-oss style 【2†L4-L9】 and 【3】.", [2, 3]),
    ("Not citations: [Page 12] [figure caption] (p. 12).", []),
    ("The excerpts do not contain the answer.", []),
])
def test_extract_excerpt_numbers(text, expected):
    assert generate.extract_excerpt_numbers(text, 5) == expected


def test_extract_excerpt_numbers_with_no_hits():
    assert generate.extract_excerpt_numbers("Claim [1].", 0) == []


def test_format_cited():
    cited = [{"n": 1, "label": "Mohs textbook", "page": 145, "locator": "p. 145"},
             {"n": 2, "label": "StatPearls: Mohs AUC", "page": None,
              "locator": "Indications"}]
    assert generate.format_cited(cited) == (
        "[1] Mohs textbook, p. 145; [2] StatPearls: Mohs AUC - Indications")
    assert generate.format_cited([{"n": None, "label": "Mohs textbook",
                                   "page": 12, "locator": "p. 12"}]) == "Mohs textbook, p. 12"


# ----------------------------------------------------------- chunk fields
def test_old_chunks_default_to_the_textbook():
    old = {"id": "x", "book": "mohs_layers", "page": 7, "kind": "prose", "text": "t"}
    assert generate.chunk_source(old) == config.DEFAULT_SOURCE
    assert generate.chunk_source({"page": 7}) == config.DEFAULT_SOURCE
    assert generate.chunk_locator(old) == "p. 7"
    assert generate.chunk_locator({"page": None}) == ""


# -------------------------------------------------------------- versions
def test_prompt_version_is_content_hash():
    assert re.fullmatch(r"[0-9a-f]{8}", generate.PROMPT_VERSION)
    expected = hashlib.sha256(generate.PROMPT.encode("utf-8")).hexdigest()[:8]
    assert generate.PROMPT_VERSION == expected


def test_page_prompt_is_unchanged():
    """The hard invariant: with the textbook alone, the prompt is today's.
    Change PROMPT on purpose and this pin must change too - and every
    graded answer so far came from the old one."""
    assert generate.PROMPT_VERSION == "c6a01e11"
    assert generate.PROMPTS["page"] is generate.PROMPT


def test_prompt_version_per_style(monkeypatch):
    numbered = hashlib.sha256(generate.NUMBERED_PROMPT.encode("utf-8")).hexdigest()[:8]
    assert generate.prompt_version("page") == generate.PROMPT_VERSION
    assert generate.prompt_version("numbered") == numbered
    assert numbered != generate.PROMPT_VERSION

    monkeypatch.setattr(config, "CITATION_STYLE", "numbered")
    assert generate.prompt_version() == numbered
    with pytest.raises(ValueError, match="citation style"):
        generate.prompt_version("footnote")


def test_numbered_prompt_states_the_rules():
    p = generate.NUMBERED_PROMPT
    assert "{context}" in p and "{question}" in p
    assert "clinic staff" in p                 # not only the surgeon now
    assert "[2]" in p and "[1][3]" in p        # the citation format
    assert "do not contain the answer" in p    # refusal
    assert "outside knowledge" in p
    assert "clinical or policy claim must end with a citation" in p
    assert "different documents" in p and "disagree" in p


@pytest.fixture
def nothing_loaded(monkeypatch):
    """No index in memory, whatever earlier tests loaded."""
    monkeypatch.setattr(retrieve, "_chunks", None)
    monkeypatch.setattr(retrieve, "_vectors", None)
    monkeypatch.setattr(retrieve, "_derived", {})


def test_generation_config_names_style_and_version(no_manifest, nothing_loaded):
    page = generate.generation_config("page")
    assert page == {**config.run_config(), "prompt_version": generate.PROMPT_VERSION}
    numbered = generate.generation_config("numbered", domains=("operations",))
    assert numbered["citation_style"] == "numbered"
    assert numbered["prompt_version"] == generate.prompt_version("numbered")
    assert numbered["retrieve_domains"] == ["operations"]


def test_generation_config_names_the_index_actually_loaded(tmp_path, monkeypatch,
                                                           nothing_loaded):
    """The app loads the index once; manifest.json is re-read per answer.
    Rebuild the index (ingest + embed) while the app runs and every later
    answer, still retrieved from the old index, used to be logged under
    the new index's ID."""
    def build(ids):
        chunks = [{"id": i, "book": "mohs_layers", "page": n, "kind": "prose",
                   "text": f"text {i}", "figures": []} for n, i in enumerate(ids, start=1)]
        (tmp_path / "chunks.json").write_text(json.dumps(chunks))
        np.save(tmp_path / "vectors.npy", np.zeros((len(ids), 3), dtype=np.float32))
        (tmp_path / "manifest.json").write_text(json.dumps(
            {"index_id": retrieve.index_id(ids), "sources": []}))

    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(config, "VECTORS_PATH", tmp_path / "vectors.npy")
    monkeypatch.setattr(config, "VECTOR_KEYS_PATH", tmp_path / "vector_keys.json")   # none
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")

    build(["a", "b"])
    assert generate.generation_config()["index_id"] == config.index_id()   # nothing loaded
    retrieve.get_index()
    assert generate.generation_config()["index_id"] == retrieve.index_id(["a", "b"])

    build(["a", "b", "statpearls:1"])                 # the rebuild, in another terminal
    assert config.index_id() == retrieve.index_id(["a", "b", "statpearls:1"])
    assert generate.generation_config()["index_id"] == retrieve.index_id(["a", "b"])


def test_run_config_names_the_llm():
    assert config.run_config()["llm_model"] == config.LLM_MODEL


# ---------------------------------------------------------------- context
def test_page_context_is_byte_identical_to_the_original(no_manifest):
    hits = fake_hits(141, 145) + fake_hits(3, kind="caption")
    assert generate.build_context(hits) == original_build_context(hits)
    assert generate.build_context(hits, "page") == original_build_context(hits)

    # Textbook chunks re-written with the new fields read the same.
    for h in hits:
        h["chunk"].update(source="mohs_layers", domain="clinical", section=[],
                          locator=f"p. {h['chunk']['page']}")
    assert generate.build_context(hits, "page") == original_build_context(hits)


def test_page_context_matches_the_original_on_the_real_index(no_manifest):
    if not config.CHUNKS_PATH.exists():
        pytest.skip("no chunks.json")
    chunks = [c for c in json.loads(config.CHUNKS_PATH.read_text())
              if c.get("source", c["book"]) == config.DEFAULT_SOURCE]
    captions = [c for c in chunks if c["kind"] == "caption"][:20]
    hits = [{"rank": n, "score": 0.5, "chunk": c}
            for n, c in enumerate(chunks[:40] + captions, start=1)]
    assert generate.build_context(hits, "page") == original_build_context(hits)


def test_page_context_names_other_sources():
    context = generate.build_context(multi_hits(), "page", labels=LABELS)
    assert context.split("\n\n") == [
        "[Page 145] [figure caption]\nBilobed flap for the nasal tip.",
        "[StatPearls: Mohs AUC - Indications]\nThe AUC rate 270 scenarios.",
        "[Page 12 - NCCI manual ch. III (2026)]\n17311 is reported per stage.",
    ]


def test_numbered_context_labels():
    context = generate.build_context(multi_hits(), "numbered", labels=LABELS)
    assert context == (
        "[1] Mohs textbook, p. 145 [figure caption]\nBilobed flap for the nasal tip.\n\n"
        "[2] StatPearls: Mohs AUC - Indications\nThe AUC rate 270 scenarios.\n\n"
        "[3] NCCI manual ch. III (2026), p. 12\n17311 is reported per stage.")


def test_numbered_context_reads_labels_from_the_manifest(manifest):
    assert generate.build_context(multi_hits()[1:2], "numbered").startswith(
        "[1] StatPearls: Mohs AUC - Indications\n")


def test_numbered_context_tolerates_old_chunks(no_manifest):
    hits = fake_hits(141) + fake_hits(9, kind="caption")
    hits[1]["chunk"].pop("book")
    assert generate.build_context(hits, "numbered") == (
        "[1] Mohs textbook, p. 141\ntext on page 141\n\n"
        "[2] Mohs textbook, p. 9 [figure caption]\ntext on page 9")


def test_numbered_context_unknown_source_falls_back_to_its_id():
    hit = multi_hits()[1]
    hit["chunk"].update(source="new_doc", locator="")
    assert generate.build_context([hit], "numbered", labels={}) == (
        "[1] new_doc\nThe AUC rate 270 scenarios.")


# ---------------------------------------------------------------- answer
def test_answer_with_fake_llm(monkeypatch, no_manifest):
    hits = fake_hits(141, 145, kind="caption")
    monkeypatch.setattr(generate, "search", fake_search(hits))
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return "Use a bilobed flap (p. 145)."

    result = generate.answer("nasal tip defect", llm=llm, style="page")

    assert result["text"] == "Use a bilobed flap (p. 145)."
    assert result["citations"] == [145]
    assert result["retrieved_pages"] == [141, 145]
    assert result["retrieved_ids"] == ["c141", "c145"]
    assert result["retrieved_sources"] == ["mohs_layers", "mohs_layers"]
    assert result["cited_sources"] == [
        {"n": None, "source": "mohs_layers", "label": "Mohs textbook",
         "page": 145, "locator": "p. 145", "chunk_id": "c145"}]
    assert result["figures"] == ["data/processed/figures/f141.jpeg",
                                 "data/processed/figures/f145.jpeg"]
    assert result["config"]["llm_model"] == config.LLM_MODEL
    assert result["config"]["prompt_version"] == generate.PROMPT_VERSION
    assert result["config"]["citation_style"] == "page"
    assert "QUESTION: nasal tip defect" in seen["prompt"]
    assert "[Page 141] [figure caption]" in seen["prompt"]
    # The whole prompt, not just pieces of it, is what it always was.
    assert seen["prompt"] == generate.PROMPT.format(
        context=original_build_context(hits), question="nasal tip defect")


def test_answer_numbered_maps_cited_sources(monkeypatch, manifest):
    monkeypatch.setattr(generate, "search", fake_search(multi_hits()))
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return ("Report 17311 per stage [3]. The AUC apply [2][9]. "
                "A bilobed flap suits the tip [1, 3].")

    result = generate.answer("Mohs billing", llm=llm, style="numbered")

    assert seen["prompt"].startswith(generate.NUMBERED_PROMPT.split("{context}")[0])
    assert "[2] StatPearls: Mohs AUC - Indications\n" in seen["prompt"]
    assert [c["n"] for c in result["cited_sources"]] == [1, 2, 3]   # [9] dropped
    assert result["cited_sources"][1] == {
        "n": 2, "source": "statpearls_mohs_auc", "label": "StatPearls: Mohs AUC",
        "page": None, "locator": "Indications", "chunk_id": "sp1"}
    assert result["citations"] == [12, 145]          # pages of cited excerpts
    assert result["retrieved_pages"] == [145, None, 12]
    assert result["retrieved_sources"] == [
        "mohs_layers", "statpearls_mohs_auc", "cms_ncci_ch3_2026"]
    assert result["config"]["citation_style"] == "numbered"
    assert result["config"]["prompt_version"] == generate.prompt_version("numbered")


def test_answer_numbered_without_citations(monkeypatch, manifest):
    monkeypatch.setattr(generate, "search", fake_search(multi_hits()))
    result = generate.answer("q", llm=lambda p: "The excerpts do not contain the answer.",
                             style="numbered")
    assert result["cited_sources"] == [] and result["citations"] == []


def test_answer_passes_domains_to_search(monkeypatch, no_manifest):
    seen = {}
    monkeypatch.setattr(generate, "search", fake_search(fake_hits(141), seen))

    result = generate.answer("q", llm=lambda p: "x", domains=["operations"])
    assert seen == {"question": "q", "k": config.TOP_K, "domains": ["operations"]}
    assert result["config"]["retrieve_domains"] == ["operations"]

    result = generate.answer("q", k=3, llm=lambda p: "x")
    assert seen["domains"] is None and seen["k"] == 3
    assert result["config"]["retrieve_domains"] == list(config.RETRIEVE_DOMAINS)


def test_answer_style_defaults_to_config(monkeypatch, manifest):
    monkeypatch.setattr(generate, "search", fake_search(multi_hits()))
    monkeypatch.setattr(config, "CITATION_STYLE", "numbered")
    prompts = []
    result = generate.answer("q", llm=lambda p: prompts.append(p) or "[1]")
    assert "[1] Mohs textbook, p. 145" in prompts[0]
    assert result["config"]["citation_style"] == "numbered"
    assert result["config"]["prompt_version"] == generate.prompt_version("numbered")


def test_page_style_with_other_sources(monkeypatch, manifest, capsys):
    """Page style over several documents: a cited page matches every
    retrieved chunk on that page, whatever its source, and the user is told
    once to switch to numbered citations."""
    hits = multi_hits()
    hits.append({"rank": 4, "score": 0.5, "chunk": {
        "id": "t12", "book": "mohs_layers", "page": 12, "kind": "prose",
        "text": "Textbook page 12.", "figures": []}})
    monkeypatch.setattr(generate, "search", fake_search(hits))
    monkeypatch.setattr(generate, "_warned_page_style", False)

    result = generate.answer("q", llm=lambda p: "Per stage (p. 12).", style="page")
    assert result["citations"] == [12]
    assert [c["chunk_id"] for c in result["cited_sources"]] == ["nc12", "t12"]
    assert all(c["n"] is None for c in result["cited_sources"])

    generate.answer("q", llm=lambda p: "x", style="page")
    assert capsys.readouterr().out.count("CITATION_STYLE") == 1


def test_page_style_textbook_only_does_not_warn(monkeypatch, no_manifest, capsys):
    monkeypatch.setattr(generate, "search", fake_search(fake_hits(141)))
    monkeypatch.setattr(generate, "_warned_page_style", False)
    generate.answer("q", llm=lambda p: "x", style="page")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("style", ["page", "numbered"])
def test_answer_no_hits_returns_every_key(monkeypatch, no_manifest, style):
    monkeypatch.setattr(generate, "search", fake_search([]))

    def llm(prompt):
        pytest.fail("the LLM must not be called when nothing was retrieved")

    result = generate.answer("anything", llm=llm, style=style)
    for key in ("question", "text", "citations", "retrieved_ids",
                "retrieved_pages", "retrieved", "figures", "cited_sources",
                "retrieved_sources", "config"):
        assert key in result
    assert result["text"] == "Nothing retrieved."
    assert result["cited_sources"] == [] and result["retrieved_sources"] == []
    assert result["config"]["citation_style"] == style
    assert result["config"]["prompt_version"] == generate.prompt_version(style)


def test_groq_llm_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(generate, "_client", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        generate.groq_llm("hello")

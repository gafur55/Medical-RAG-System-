"""Smoke tests for app.py: the UI builds, test mode is recorded, test records
stay out of the Grade queue and the Summary, and the multi-source pieces
(disclaimer, domain filter, citation and passage panels, triage) behave -
while a textbook-only index reads exactly as it always did.

Storage is redirected to a temporary folder with Hub sync disabled, the
index manifest is faked, and answer() is faked, so no API key, model or
network is needed. Skipped when gradio is not installed.
"""

import importlib
import json
import subprocess
import sys

import pytest

pytest.importorskip("gradio")

from src import config  # noqa: E402

TEXTBOOK = {"id": "mohs_layers", "label": "Mohs textbook", "domain": "clinical"}
STATPEARLS = {"id": "statpearls_mohs_auc", "label": "StatPearls: Mohs AUC",
              "domain": "clinical"}
NCCI = {"id": "cms_ncci_ch3_2026", "label": "NCCI manual ch. III (2026)",
        "domain": "operations"}

ORIGINAL_DISCLAIMER = """
### Mohs reconstruction reference

Searches one textbook and answers only from what it finds.
A reference tool, not clinical advice - verify every answer against the cited
pages. **Do not enter patient-identifying information.**
"""


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    import app
    from src import storage
    monkeypatch.setattr(storage, "SESSIONS_PATH", tmp_path / "sessions.jsonl")
    monkeypatch.setattr(storage, "TEST_IDS_PATH", tmp_path / "test_records.json")
    monkeypatch.setattr(storage, "_push", lambda: None)
    return app


def use_manifest(monkeypatch, *sources):
    manifest = {"index_id": "abc123def456", "sources": list(sources)} if sources else {}
    monkeypatch.setattr(config, "index_manifest", lambda: manifest)


def record(rid, question, grade=None, **extra):
    return {"id": rid, "question": question, "answer": "A", "citations": [],
            "retrieved_pages": [1], "grade": grade, **extra}


def write_records(tmp_path, *records):
    (tmp_path / "sessions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))


def hit(rank, chunk):
    return {"rank": rank, "score": 0.9 - rank / 100, "chunk": chunk}


TEXTBOOK_CHUNK = {"id": "t145", "book": "mohs_layers", "page": 145,
                  "kind": "caption", "text": "Bilobed flap.", "figures": []}
STATPEARLS_CHUNK = {"id": "sp1", "book": "statpearls_mohs_auc",
                    "source": "statpearls_mohs_auc", "domain": "clinical",
                    "page": None, "locator": "Indications",
                    "section": ["Indications"], "kind": "prose",
                    "text": "The AUC rate 270 scenarios.", "figures": []}


def fake_result(style="page", **extra):
    """What answer() returns for a two-excerpt answer."""
    numbered = style == "numbered"
    hits = [hit(1, TEXTBOOK_CHUNK)] + ([hit(2, STATPEARLS_CHUNK)] if numbered else [])
    cited = [{"n": 1 if numbered else None, "source": "mohs_layers",
              "label": "Mohs textbook", "page": 145, "locator": "p. 145",
              "chunk_id": "t145"}]
    if numbered:
        cited.append({"n": 2, "source": "statpearls_mohs_auc",
                      "label": "StatPearls: Mohs AUC", "page": None,
                      "locator": "Indications", "chunk_id": "sp1"})
    return {"question": "q", "text": "t", "citations": [145],
            "retrieved_ids": [h["chunk"]["id"] for h in hits],
            "retrieved_pages": [h["chunk"]["page"] for h in hits],
            "retrieved": hits, "figures": [], "cited_sources": cited,
            "retrieved_sources": [h["chunk"].get("source", "mohs_layers") for h in hits],
            "config": {"citation_style": style}, **extra}


# --------------------------------------------------------------- basics
def test_ui_builds(app_module):
    assert app_module.demo is not None


def test_test_records_skipped_by_grade_and_summary(app_module, tmp_path):
    records = [
        record("real1", "Q real"),
        record("t1", "Q flagged", is_test=True),
        record("t2", "Q listed", grade=app_module.GRADES[3]),
    ]
    write_records(tmp_path, *records)
    (tmp_path / "test_records.json").write_text(json.dumps({"ids": ["t2"]}))

    header = app_module.next_ungraded()[0]
    assert "1 waiting" in header and "Q real" in header
    # t2 carries the only grade, and it is a test grade
    assert app_module.summary() == "No grades yet."


def test_ask_records_test_mode(app_module, monkeypatch):
    from src import storage
    fake = {"question": "q", "text": "t", "citations": [], "retrieved_ids": [],
            "retrieved_pages": [], "retrieved": [], "figures": [], "config": {}}
    monkeypatch.setattr(app_module, "answer", lambda q, **kw: fake)

    app_module.ask("q", "Test User", True)
    app_module.ask("q", "Test User", False)

    assert [r["is_test"] for r in storage.load_records()] == [True, False]


def test_app_needs_no_ingest_only_packages():
    """The Space installs requirements.txt only. Importing the app - and
    everything it imports - must not pull in the registry, the loaders or
    their parsers."""
    code = ("import sys, app; "
            "bad = {'src.sources', 'src.loaders', 'src.ingest', 'bs4', 'docx', 'fitz'}; "
            "print(sorted(bad & set(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


# ----------------------------------------------------------- disclaimer
def test_disclaimer_for_the_textbook_alone_is_unchanged(app_module, monkeypatch):
    use_manifest(monkeypatch)                       # no manifest yet
    assert app_module.disclaimer() == ORIGINAL_DISCLAIMER
    use_manifest(monkeypatch, TEXTBOOK)
    assert app_module.disclaimer() == ORIGINAL_DISCLAIMER


def test_disclaimer_lists_several_sources(app_module, monkeypatch):
    use_manifest(monkeypatch, TEXTBOOK, STATPEARLS, NCCI)
    text = app_module.disclaimer()
    assert ("Searches 3 sources: Mohs textbook, StatPearls: Mohs AUC, "
            "NCCI manual ch. III (2026) and answers only from what it finds.") in text
    assert text.replace(text.splitlines()[3], "") == ORIGINAL_DISCLAIMER.replace(
        ORIGINAL_DISCLAIMER.splitlines()[3], "")


def test_disclaimer_for_one_other_source(app_module, monkeypatch):
    use_manifest(monkeypatch, NCCI)
    assert "Searches one source: NCCI manual ch. III (2026) and" in app_module.disclaimer()


# -------------------------------------------------------- domain filter
def test_domain_filter_hidden_for_a_single_domain(app_module, monkeypatch):
    use_manifest(monkeypatch)
    assert app_module.domain_choices() == []
    use_manifest(monkeypatch, TEXTBOOK, STATPEARLS)      # both clinical
    assert app_module.domain_choices() == []


def test_domain_filter_choices_for_two_domains(app_module, monkeypatch):
    use_manifest(monkeypatch, NCCI, TEXTBOOK)
    assert app_module.domain_choices() == [
        ("Clinical", "clinical"),
        ("Operations - coding, billing, policy", "operations")]


def test_domain_filter_offers_only_what_config_searches(app_module, monkeypatch):
    """The filter can narrow config.RETRIEVE_DOMAINS, never widen it - the
    Ask tab must search what score.py and the sweep measure."""
    use_manifest(monkeypatch, TEXTBOOK, NCCI)
    monkeypatch.setattr(config, "RETRIEVE_DOMAINS", ("clinical",))
    assert app_module.domain_choices() == []              # one left: no filter

    research = {"id": "trials", "label": "Trials", "domain": "research"}
    use_manifest(monkeypatch, TEXTBOOK, NCCI, research)
    monkeypatch.setattr(config, "RETRIEVE_DOMAINS", ("clinical", "operations"))
    assert app_module.domain_choices() == [
        ("Clinical", "clinical"),
        ("Operations - coding, billing, policy", "operations")]


def test_domain_filter_hidden_in_the_ui_when_config_excludes_a_domain(app_module, monkeypatch):
    """Two domains indexed but config searches one: no checkbox, and ask()
    gets domains=None, so config decides - as score.py and chat_app do."""
    def ask_inputs(module):
        return next(fn.inputs for fn in module.demo.fns.values()
                    if getattr(fn.fn, "__name__", "") == "ask")

    try:
        with monkeypatch.context() as m:
            m.setattr(config, "index_manifest", lambda: {"sources": [TEXTBOOK, NCCI]})
            m.setattr(config, "RETRIEVE_DOMAINS", ("clinical",))
            importlib.reload(app_module)
            assert app_module.domain_filter is None
            assert len(ask_inputs(app_module)) == 3
    finally:
        importlib.reload(app_module)


def test_sources_without_chunks_are_not_offered(app_module, monkeypatch):
    """A source whose loader found no text (a scanned PDF) is in the
    manifest with n_chunks 0. Nothing of it can be retrieved, so neither
    the disclaimer nor the filter may name it."""
    use_manifest(monkeypatch, {**TEXTBOOK, "n_chunks": 1491}, {**NCCI, "n_chunks": 0})
    assert app_module.indexed_sources() == [{**TEXTBOOK, "n_chunks": 1491}]
    assert app_module.disclaimer() == ORIGINAL_DISCLAIMER
    assert app_module.domain_choices() == []

    use_manifest(monkeypatch, {**TEXTBOOK, "n_chunks": 1491}, {**NCCI, "n_chunks": 12})
    assert "Searches 2 sources" in app_module.disclaimer()
    assert len(app_module.domain_choices()) == 2


def test_domain_filter_in_the_ui(app_module, monkeypatch):
    """Rebuild the UI against a two-domain index: the checkbox group is
    there, ticked for everything, and wired into ask(). With today's
    textbook-only index it is absent."""
    import gradio as gr

    def ask_inputs(module):
        return next(fn.inputs for fn in module.demo.fns.values()
                    if getattr(fn.fn, "__name__", "") == "ask")

    try:
        with monkeypatch.context() as m:
            m.setattr(config, "index_manifest",
                      lambda: {"sources": [TEXTBOOK, STATPEARLS, NCCI]})
            importlib.reload(app_module)
            assert isinstance(app_module.domain_filter, gr.CheckboxGroup)
            assert app_module.domain_filter.value == ["clinical", "operations"]
            assert app_module.domain_filter in ask_inputs(app_module)
            assert "Searches 3 sources" in app_module.DISCLAIMER

        with monkeypatch.context() as m:
            m.setattr(config, "index_manifest", lambda: {"sources": [TEXTBOOK]})
            importlib.reload(app_module)
            assert app_module.domain_filter is None
            assert len(ask_inputs(app_module)) == 3
            assert app_module.DISCLAIMER == ORIGINAL_DISCLAIMER
    finally:
        importlib.reload(app_module)


def test_ask_with_nothing_ticked_asks_for_a_domain(app_module, monkeypatch):
    from src import storage
    monkeypatch.setattr(app_module, "answer",
                        lambda q, **kw: pytest.fail("must not search"))
    out = app_module.ask("q", "Test User", False, [])
    assert "Pick at least one" in out[0]
    assert storage.load_records() == []


# ------------------------------------------------------------------ ask
def test_ask_numbered_records_new_fields(app_module, monkeypatch):
    from src import storage
    seen = {}

    def fake_answer(q, **kw):
        seen.update(kw)
        return fake_result("numbered")

    monkeypatch.setattr(app_module, "answer", fake_answer)
    use_manifest(monkeypatch, TEXTBOOK, STATPEARLS, NCCI)

    _, cites, _, passages = app_module.ask("q", "Test User", True, ["clinical"])

    assert seen == {"domains": ["clinical"]}
    rec = storage.load_records()[0]
    assert rec["retrieved_sources"] == ["mohs_layers", "statpearls_mohs_auc"]
    assert [c["n"] for c in rec["cited_sources"]] == [1, 2]
    assert rec["domains"] == ["clinical"]
    assert rec["citation_style"] == "numbered"
    assert cites == ("**Cited sources:** [1] Mohs textbook, p. 145; "
                     "[2] StatPearls: Mohs AUC - Indications")
    assert "**[1] Mohs textbook, p. 145 - caption - score 0.890**" in passages
    assert "**[2] StatPearls: Mohs AUC - Indications - prose - score 0.880**" in passages


def test_ask_page_style_textbook_panels_unchanged(app_module, monkeypatch):
    """Textbook-only page style: citation line and passages read exactly as
    before multi-source support, and no domain filter means domains=None."""
    from src import storage
    seen = {}

    def fake_answer(q, **kw):
        seen.update(kw)
        return fake_result("page")

    monkeypatch.setattr(app_module, "answer", fake_answer)
    use_manifest(monkeypatch)

    _, cites, _, passages = app_module.ask("q", "Test User", True)

    assert seen == {"domains": None}
    assert cites == "**Cited pages:** p145"
    assert passages == ("### What the model was shown\n"
                        "_These passages went into the prompt. The answer should come "
                        "from here._\n\n"
                        "**1. Page 145 - caption - score 0.890**\n\n> Bilobed flap.\n\n")
    rec = storage.load_records()[0]
    assert rec["citation_style"] == "page" and rec["domains"] is None


def test_citations_line_without_citations(app_module):
    page = fake_result("page", citations=[], cited_sources=[])
    numbered = fake_result("numbered", citations=[], cited_sources=[])
    assert app_module.citations_line(page) == (
        "**No page citations in this answer.** Treat it with extra caution.")
    assert app_module.citations_line(numbered).startswith("**No citations in this answer.**")


def test_page_style_passages_name_other_sources(app_module):
    result = fake_result("page")
    result["retrieved"].append(hit(2, STATPEARLS_CHUNK))
    passages = app_module.passages_panel(result, {"statpearls_mohs_auc": "StatPearls: Mohs AUC"})
    assert "**1. Page 145 - caption" in passages
    assert "**2. StatPearls: Mohs AUC - Indications - prose" in passages


# ---------------------------------------------------------------- grade
def test_pages_line_for_old_and_textbook_records(app_module):
    old = record("r", "q", citations=[145], retrieved_pages=[145, 12])
    assert app_module.pages_line(old) == "Retrieved pages: p145, p12  |  Cited: p145"
    new = {**old, "retrieved_sources": ["mohs_layers", "mohs_layers"],
           "citation_style": "page"}
    assert app_module.pages_line(new) == "Retrieved pages: p145, p12  |  Cited: p145"


def test_pages_line_names_other_sources(app_module):
    rec = record("r", "q", citations=[145, 12],
                 retrieved_pages=[145, None, 12],
                 retrieved_sources=["mohs_layers", "statpearls_mohs_auc", "cms_ncci_ch3_2026"],
                 citation_style="numbered",
                 cited_sources=[{"n": 1, "label": "Mohs textbook", "page": 145,
                                 "locator": "p. 145"},
                                {"n": 3, "label": "NCCI manual ch. III (2026)",
                                 "page": 12, "locator": "p. 12"}])
    labels = {"mohs_layers": "Mohs textbook", "statpearls_mohs_auc": "StatPearls: Mohs AUC",
              "cms_ncci_ch3_2026": "NCCI manual ch. III (2026)"}
    assert app_module.pages_line(rec, labels) == (
        "Retrieved: p145, StatPearls: Mohs AUC, NCCI manual ch. III (2026) p12"
        "  |  Cited: [1] Mohs textbook, p. 145; [3] NCCI manual ch. III (2026), p. 12")


def test_next_ungraded_copes_with_pageless_sources(app_module, tmp_path, monkeypatch):
    use_manifest(monkeypatch, TEXTBOOK, STATPEARLS)
    write_records(tmp_path, record("r1", "Q", retrieved_pages=[None, 3],
                                   retrieved_sources=["statpearls_mohs_auc", "mohs_layers"]))
    assert app_module.next_ungraded()[2] == "Retrieved: StatPearls: Mohs AUC, p3"


# -------------------------------------------------------------- summary
def test_summary_triage_counts_textbook_pages_only(app_module, tmp_path, monkeypatch):
    """Gold pages are textbook pages. Page 12 of the NCCI manual in the
    context is not the textbook's page 12."""
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([
        {"id": 1, "question": "Q gold", "gold_pages": [12]},
        {"id": 2, "question": "Q other gold", "gold_pages": [12],
         "gold_source": "cms_ncci_ch3_2026"},
    ]))
    monkeypatch.setattr(config, "QUESTIONS_PATH", questions)
    bad = app_module.GRADES[2]
    write_records(
        tmp_path,
        # policy-manual p12 only -> search never found the textbook page
        record("a", "Q gold", bad, retrieved_pages=[12, 40],
               retrieved_sources=["cms_ncci_ch3_2026", "mohs_layers"]),
        # an old record: every page is a textbook page -> found it
        record("b", "Q gold", bad, retrieved_pages=[12]),
        # a new record with the textbook page, next to a pageless excerpt
        record("c", "Q gold", bad, retrieved_pages=[None, 12],
               retrieved_sources=["statpearls_mohs_auc", "mohs_layers"]),
        # gold pages from another source match only that source
        record("d", "Q other gold", bad, retrieved_pages=[12],
               retrieved_sources=["cms_ncci_ch3_2026"]),
    )
    text = app_module.summary()
    assert "Search never found a correct page: **1**" in text
    assert "Correct page was there, answer still poor: **3**" in text

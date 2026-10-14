"""Tests for src/generate.py - no API key, no network, no embedding model.

search() and the LLM are both swapped for fakes, so these run anywhere.
"""

import hashlib
import re

import pytest

from src import config, generate


def fake_hits(*pages, kind="prose"):
    hits = []
    for rank, page in enumerate(pages, start=1):
        chunk = {"id": f"c{page}", "page": page, "kind": kind,
                 "text": f"text on page {page}", "figures": []}
        if kind == "caption":
            chunk["figure_paths"] = [f"data/processed/figures/f{page}.jpeg"]
        hits.append({"rank": rank, "score": 0.9 - rank / 100, "chunk": chunk})
    return hits


# ------------------------------------------------------------ citations
def test_extract_citations_formats():
    text = "Flap A works (p. 141). Graft B too (p. 145; p. 141). Slip: page 12."
    assert generate.extract_citations(text) == [12, 141, 145]


def test_extract_citations_none():
    assert generate.extract_citations("The excerpts do not contain the answer.") == []


# -------------------------------------------------------------- versions
def test_prompt_version_is_content_hash():
    assert re.fullmatch(r"[0-9a-f]{8}", generate.PROMPT_VERSION)
    expected = hashlib.sha256(generate.PROMPT.encode("utf-8")).hexdigest()[:8]
    assert generate.PROMPT_VERSION == expected


def test_run_config_names_the_llm():
    assert config.run_config()["llm_model"] == config.LLM_MODEL


# ---------------------------------------------------------------- answer
def test_answer_with_fake_llm(monkeypatch):
    monkeypatch.setattr(generate, "search",
                        lambda q, k=None: fake_hits(141, 145, kind="caption"))
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return "Use a bilobed flap (p. 145)."

    result = generate.answer("nasal tip defect", llm=llm)

    assert result["text"] == "Use a bilobed flap (p. 145)."
    assert result["citations"] == [145]
    assert result["retrieved_pages"] == [141, 145]
    assert result["retrieved_ids"] == ["c141", "c145"]
    assert result["figures"] == ["data/processed/figures/f141.jpeg",
                                 "data/processed/figures/f145.jpeg"]
    assert result["config"]["llm_model"] == config.LLM_MODEL
    assert result["config"]["prompt_version"] == generate.PROMPT_VERSION
    assert "QUESTION: nasal tip defect" in seen["prompt"]
    assert "[Page 141] [figure caption]" in seen["prompt"]


def test_answer_no_hits_returns_every_key(monkeypatch):
    monkeypatch.setattr(generate, "search", lambda q, k=None: [])

    def llm(prompt):
        pytest.fail("the LLM must not be called when nothing was retrieved")

    result = generate.answer("anything", llm=llm)
    for key in ("question", "text", "citations", "retrieved_ids",
                "retrieved_pages", "retrieved", "figures", "config"):
        assert key in result
    assert result["text"] == "Nothing retrieved."


def test_groq_llm_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(generate, "_client", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        generate.groq_llm("hello")

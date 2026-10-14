"""Tests for chat_app.py - follow-up rewriting, the Sources panel, and what
gets saved. answer() and the LLM are faked and storage goes to a temporary
file, so no API key, model or network is needed."""

import pytest

pytest.importorskip("gradio")


@pytest.fixture
def chat(tmp_path, monkeypatch):
    import chat_app
    from src import storage
    monkeypatch.setattr(storage, "SESSIONS_PATH", tmp_path / "sessions.jsonl")
    monkeypatch.setattr(storage, "TEST_IDS_PATH", tmp_path / "test_records.json")
    monkeypatch.setattr(storage, "_push", lambda: None)
    monkeypatch.delenv("MOHS_TEST_MODE", raising=False)
    return chat_app


def fake_result(question):
    chunk = {"id": "c145", "book": "mohs_layers", "page": 145, "kind": "prose",
             "text": "The bilobed flap is the workhorse for small tip defects.",
             "figures": []}
    return {"question": question, "text": "Use a bilobed flap (p. 145).",
            "citations": [145], "retrieved_ids": ["c145"], "retrieved_pages": [145],
            "retrieved": [{"rank": 1, "score": 0.8, "chunk": chunk}],
            "figures": [], "config": {"llm_model": "x"}}


def history(*pairs):
    msgs = []
    for user, bot in pairs:
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": bot})
    return msgs


# -------------------------------------------------------------- condense
def test_first_message_is_not_rewritten(chat):
    def llm(prompt):
        pytest.fail("no history - the LLM must not be called")
    assert chat.condense("Nasal tip defect options", [], llm=llm) == "Nasal tip defect options"


def test_follow_up_is_rewritten_with_history(chat):
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return '  "Options for a 2 cm nasal tip defect?"\n'

    out = chat.condense("and for 2 cm?", history(("Nasal tip defect options", "Bilobed.")), llm=llm)
    assert out == "Options for a 2 cm nasal tip defect?"
    assert "User: Nasal tip defect options" in seen["prompt"]
    assert "LAST MESSAGE: and for 2 cm?" in seen["prompt"]


def test_failed_rewrite_falls_back_to_message(chat):
    def llm(prompt):
        raise RuntimeError("rate limited")
    assert chat.condense("and for 2 cm?", history(("q", "a")), llm=llm) == "and for 2 cm?"


def test_conversation_skips_sources_panels_and_multimodal_parts(chat):
    msgs = [{"role": "user", "content": [{"type": "text", "text": "first q"}]},
            {"role": "assistant", "content": "answer"},
            {"role": "assistant", "content": "1. p. 145 ...", "metadata": {"title": "Sources (1 passages)"}}]
    assert chat.conversation(msgs) == "User: first q\nAssistant: answer"


# ------------------------------------------------------------------ turn
def test_turn_saves_record_and_returns_answer_plus_sources(chat):
    from src import storage
    reply = chat.turn("bilobed flap?", [], session="s1", answer_fn=fake_result)

    assert reply[0].content == "Use a bilobed flap (p. 145)."
    assert reply[1].metadata["title"] == "Sources (1 passages)"
    assert "p. 145" in reply[1].content and "cited" in reply[1].content

    (rec,) = storage.load_records()
    assert rec["channel"] == "chat" and rec["conversation_id"] == "s1"
    assert rec["question"] == "bilobed flap?" and rec["grade"] is None
    assert rec["is_test"] is False and rec["retrieved_pages"] == [145]


def test_turn_records_the_rewrite_and_shows_it(chat):
    from src import storage
    reply = chat.turn("and for 2 cm?", history(("tip defect", "Bilobed.")),
                      answer_fn=fake_result, llm=lambda p: "Options for a 2 cm tip defect")
    rec = storage.load_records()[0]
    assert rec["message"] == "and for 2 cm?"
    assert rec["question"] == "Options for a 2 cm tip defect"
    assert "Searched as:" in reply[1].content


def test_test_mode_marks_records(chat, monkeypatch):
    from src import storage
    monkeypatch.setenv("MOHS_TEST_MODE", "1")
    chat.turn("q", [], answer_fn=fake_result)
    assert storage.load_records()[0]["is_test"] is True


def test_empty_message_and_missing_key(chat):
    from src import storage
    assert chat.turn("   ", []) == "Type a question first."

    def no_key(q):
        raise RuntimeError("GROQ_API_KEY is not set")
    assert "GROQ_API_KEY" in chat.turn("q", [], answer_fn=no_key)
    assert storage.load_records() == []


def test_ui_builds(chat):
    assert chat.demo is not None

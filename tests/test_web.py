"""Tests for web_app.py and src/chat.py - the chat website.

answer() and the LLM are faked, the database and sessions file live in
tmp_path, and Hub sync is off, so no key, model or network is needed.
"""

import json
import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src import chat  # noqa: E402


def fake_result(question):
    chunk = {"id": "c145", "book": "mohs_layers", "source": "mohs_layers", "page": 145,
             "locator": "p. 145", "kind": "prose", "figures": [],
             "text": "The bilobed flap is the workhorse for small tip defects."}
    return {"question": question, "text": f"For **{question}**: use a bilobed flap (p. 145).",
            "citations": [145], "retrieved_ids": ["c145"], "retrieved_pages": [145],
            "retrieved_sources": ["mohs_layers"], "cited_sources": [],
            "retrieved": [{"rank": 1, "score": 0.81, "chunk": chunk}],
            "figures": [], "config": {"llm_model": "fake"}}


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    from src import storage
    monkeypatch.setattr(storage, "SESSIONS_PATH", tmp_path / "sessions.jsonl")
    monkeypatch.setattr(storage, "TEST_IDS_PATH", tmp_path / "test_records.json")
    monkeypatch.setattr(storage, "_push", lambda: None)
    monkeypatch.delenv("MOHS_TEST_MODE", raising=False)
    import web_app

    clients = []

    def make(answer_fn=fake_result, llm=lambda p: "Options for a 2 cm nasal tip defect", **kw):
        app = web_app.create_app(db_path=tmp_path / "chat.db", answer_fn=answer_fn,
                                 llm=llm, warmup=False, **kw)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def sessions(tmp_path):
    path = tmp_path / "sessions.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


# --------------------------------------------------------------- chat.py
def test_validate_message():
    assert chat.validate_message("  Nasal ala, 1.5 cm  ") == ("Nasal ala, 1.5 cm", None)
    assert chat.validate_message("   ")[1] == "Type a question first."
    assert "characters" in chat.validate_message("x" * (chat.MAX_MESSAGE_CHARS + 1))[1]


@pytest.mark.parametrize("text", [
    "Patient John, SSN 123-45-6789, nasal tip defect",
    "call me at (312) 555-0199 about the ear",
    "DOB 3/4/1950, lower lip defect",
    "MRN 0042 helical rim",
    "seen on 10/12/2026 for a cheek defect",
    "email jane.doe@example.com",
])
def test_phi_is_refused(text):
    assert chat.phi_warning(text)


@pytest.mark.parametrize("text", [
    "Reconstruction of a 2.5 cm defect, 4 mm deep",
    "Billing 17311 and 17312 with a 13132 repair",
    "Options for a 1/3 lip defect",
    "Born-again flap? no - bilobed flap design",
])
def test_ordinary_questions_pass(text):
    assert chat.phi_warning(text) is None


def test_render_markdown_is_safe_and_marks_citations():
    out = chat.render_markdown("**Use** a flap (p. 145). <script>alert(1)</script> See [2].")
    assert "<strong>Use</strong>" in out
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert '<span class="cite">(p. 145)</span>' in out
    assert '<span class="cite">[2]</span>' in out


def test_with_retry_retries_only_transient_errors():
    class RateLimitError(Exception):
        pass
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RateLimitError()
        return "ok"
    assert chat.with_retry(flaky, sleep=lambda s: None) == "ok" and len(calls) == 3

    def broken():
        raise ValueError("bug")
    with pytest.raises(ValueError):
        chat.with_retry(broken, sleep=lambda s: None)


def test_friendly_error_never_leaks_details():
    assert "GROQ_API_KEY" in chat.friendly_error(RuntimeError("GROQ_API_KEY is not set"))
    msg = chat.friendly_error(KeyError("secret internals"))
    assert "secret" not in msg


# ------------------------------------------------------------------- api
def test_page_and_static_files_have_security_headers(make_client):
    client = make_client()
    page = client.get("/")
    assert page.status_code == 200 and "Mohs Reference" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    for asset in ("/static/app.js", "/static/style.css", "/static/icon.svg"):
        assert client.get(asset).status_code == 200
    assert client.get("/c/abc123").status_code == 200       # deep links serve the page


def test_health_and_meta(make_client):
    client = make_client()
    h = client.get("/api/health").json()
    assert h["status"] == "ready" and h["db"]["conversations"] == 0
    meta = client.get("/api/meta").json()
    assert meta["examples"] and meta["max_chars"] == chat.MAX_MESSAGE_CHARS


def test_ask_creates_conversation_saves_db_and_grading_record(make_client, tmp_path):
    client = make_client()
    r = client.post("/api/conversations/new/messages", json={"content": "Nasal tip defect options"})
    assert r.status_code == 200
    data = r.json()
    cid = data["conversation"]["id"]
    assert data["conversation"]["title"] == "Nasal tip defect options"
    a = data["assistant"]
    assert a["status"] == "ok" and '<span class="cite">(p. 145)</span>' in a["html"]
    assert a["sources"][0]["cited"] is True and a["sources"][0]["label"] == "Mohs textbook"
    assert a["searched_as"] is None

    (rec,) = sessions(tmp_path)
    assert rec["channel"] == "web" and rec["conversation_id"] == cid and rec["grade"] is None

    listed = client.get("/api/conversations").json()
    assert [c["id"] for c in listed] == [cid]
    conv = client.get(f"/api/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]


def test_follow_up_is_condensed_with_db_history(make_client, tmp_path):
    seen = []

    def llm(prompt):
        seen.append(prompt)
        return "Options for a 2 cm nasal tip defect"
    client = make_client(llm=llm)
    cid = client.post("/api/conversations/new/messages",
                      json={"content": "Nasal tip defect options"}).json()["conversation"]["id"]
    assert seen == []                                     # first message: no rewrite
    a = client.post(f"/api/conversations/{cid}/messages", json={"content": "and for 2 cm?"}).json()["assistant"]
    assert "User: Nasal tip defect options" in seen[0]
    assert a["searched_as"] == "Options for a 2 cm nasal tip defect"
    assert sessions(tmp_path)[-1]["question"] == "Options for a 2 cm nasal tip defect"


def test_refused_messages_store_nothing(make_client, tmp_path):
    client = make_client()
    for content in ("", "SSN 123-45-6789 tip defect", "x" * (chat.MAX_MESSAGE_CHARS + 5)):
        r = client.post("/api/conversations/new/messages", json={"content": content})
        assert r.status_code == 422 and r.json()["error"]
    assert client.get("/api/health").json()["db"]["questions"] == 0
    assert sessions(tmp_path) == []


def test_failed_answer_is_saved_as_error_and_retry_replaces_it(make_client, tmp_path):
    calls = {"n": 0}

    def flaky(question):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("GROQ_API_KEY is not set")
        return fake_result(question)
    client = make_client(answer_fn=flaky)
    data = client.post("/api/conversations/new/messages", json={"content": "Ear rim defect"}).json()
    err = data["assistant"]
    assert err["status"] == "error" and "GROQ_API_KEY" in err["text"]
    assert sessions(tmp_path) == []

    fixed = client.post(f"/api/messages/{err['id']}/retry").json()["assistant"]
    assert fixed["status"] == "ok"
    conv = client.get(f"/api/conversations/{data['conversation']['id']}").json()
    assert [m["status"] for m in conv["messages"]] == ["ok", "ok"]
    assert len(sessions(tmp_path)) == 1
    assert client.post(f"/api/messages/{fixed['id']}/retry").status_code == 400


def test_feedback_reaches_db_and_grading_record(make_client, tmp_path):
    client = make_client()
    a = client.post("/api/conversations/new/messages", json={"content": "Tip"}).json()["assistant"]
    r = client.post(f"/api/messages/{a['id']}/feedback", json={"value": -1, "comment": "missed forehead flap"})
    assert r.status_code == 200
    assert sessions(tmp_path)[0]["chat_feedback"] == -1
    assert sessions(tmp_path)[0]["chat_feedback_comment"] == "missed forehead flap"
    assert client.post(f"/api/messages/{a['id']}/feedback", json={"value": 3}).status_code == 422
    assert client.post("/api/messages/99999/feedback", json={"value": 1}).status_code == 404


def test_rename_delete_and_unknown(make_client):
    client = make_client()
    cid = client.post("/api/conversations/new/messages", json={"content": "Tip"}).json()["conversation"]["id"]
    assert client.patch(f"/api/conversations/{cid}", json={"title": "Nose cases"}).json()["title"] == "Nose cases"
    assert client.delete(f"/api/conversations/{cid}").status_code == 200
    assert client.get(f"/api/conversations/{cid}").status_code == 404
    assert client.post(f"/api/conversations/{cid}/messages", json={"content": "q"}).status_code == 404
    assert client.get("/api/conversations").json() == []


def test_busy_server_answers_with_a_retryable_error(make_client, monkeypatch):
    import web_app
    monkeypatch.setattr(web_app, "ANSWER_WAIT_SECONDS", 0.05)
    gate = threading.Event()

    def slow(question):
        gate.wait(5)
        return fake_result(question)
    client = make_client(answer_fn=slow, concurrency=1)
    results = {}
    t = threading.Thread(target=lambda: results.update(
        first=client.post("/api/conversations/new/messages", json={"content": "first"}).json()))
    t.start()
    import time
    time.sleep(0.2)
    second = client.post("/api/conversations/new/messages", json={"content": "second"}).json()
    gate.set()
    t.join()
    assert second["assistant"]["status"] == "error" and "busy" in second["assistant"]["text"]
    assert results["first"]["assistant"]["status"] == "ok"


def test_unexpected_crash_returns_json_not_a_stack_trace(make_client, monkeypatch):
    from src import chatdb
    client = make_client()

    def boom(*a, **k):
        raise KeyError("internal detail")
    monkeypatch.setattr(chatdb, "list_conversations", boom)
    r = TestClient(client.app, raise_server_exceptions=False).get("/api/conversations")
    assert r.status_code == 500 and "internal detail" not in r.text and r.json()["error"]

"""Smoke tests for app.py: the UI builds, test mode is recorded, and test
records stay out of the Grade queue and the Summary.

Storage is redirected to a temporary folder with Hub sync disabled, and
answer() is faked, so no API key, model or network is needed. Skipped when
gradio is not installed.
"""

import json

import pytest

pytest.importorskip("gradio")


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    import app
    from src import storage
    monkeypatch.setattr(storage, "SESSIONS_PATH", tmp_path / "sessions.jsonl")
    monkeypatch.setattr(storage, "TEST_IDS_PATH", tmp_path / "test_records.json")
    monkeypatch.setattr(storage, "_push", lambda: None)
    return app


def record(rid, question, grade=None, **extra):
    return {"id": rid, "question": question, "answer": "A", "citations": [],
            "retrieved_pages": [1], "grade": grade, **extra}


def test_ui_builds(app_module):
    assert app_module.demo is not None


def test_test_records_skipped_by_grade_and_summary(app_module, tmp_path):
    records = [
        record("real1", "Q real"),
        record("t1", "Q flagged", is_test=True),
        record("t2", "Q listed", grade=app_module.GRADES[3]),
    ]
    (tmp_path / "sessions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    (tmp_path / "test_records.json").write_text(json.dumps({"ids": ["t2"]}))

    header = app_module.next_ungraded()[0]
    assert "1 waiting" in header and "Q real" in header
    # t2 carries the only grade, and it is a test grade
    assert app_module.summary() == "No grades yet."


def test_ask_records_test_mode(app_module, monkeypatch):
    from src import storage
    fake = {"question": "q", "text": "t", "citations": [], "retrieved_ids": [],
            "retrieved_pages": [], "retrieved": [], "figures": [], "config": {}}
    monkeypatch.setattr(app_module, "answer", lambda q: fake)

    app_module.ask("q", "Test User", True)
    app_module.ask("q", "Test User", False)

    assert [r["is_test"] for r in storage.load_records()] == [True, False]

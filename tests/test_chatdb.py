"""Tests for src/chatdb.py - the chat website's SQLite store."""

import sqlite3

import pytest

from src import chatdb


@pytest.fixture
def db(tmp_path):
    conn = chatdb.connect(tmp_path / "chat.db")
    yield conn
    conn.close()


def test_schema_is_created_and_versioned(tmp_path):
    path = tmp_path / "chat.db"
    chatdb.connect(path).close()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == chatdb.SCHEMA_VERSION
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    # reopening an up-to-date file is a no-op
    chatdb.connect(path).close()


def test_first_question_becomes_the_title(db):
    c = chatdb.create_conversation(db)
    assert c["title"] == "New chat"
    chatdb.add_message(db, c["id"], "user", "Options for a nasal tip defect")
    chatdb.add_message(db, c["id"], "user", "and for 2 cm?")
    assert chatdb.get_conversation(db, c["id"])["title"] == "Options for a nasal tip defect"


def test_make_title_cuts_at_a_word():
    long = "Reconstruction options for a full-thickness defect of the lower lip involving the commissure"
    title = chatdb.make_title(long)
    assert len(title) <= chatdb.TITLE_CHARS + 3 and title.endswith("...")
    assert not title[:-3].endswith(" ")
    assert chatdb.make_title("   ") == "New chat"


def test_messages_round_trip_json_fields(db):
    c = chatdb.create_conversation(db)
    m = chatdb.add_message(db, c["id"], "assistant", "Use a flap (p. 145).",
                           sources=[{"rank": 1, "page": 145}], figures=["f.jpeg"],
                           config={"top_k": 5}, record_id="r1", standalone_question="q")
    assert m["sources"] == [{"rank": 1, "page": 145}]
    assert m["figures"] == ["f.jpeg"] and m["config"] == {"top_k": 5}
    plain = chatdb.add_message(db, c["id"], "user", "hi")
    assert plain["sources"] == [] and plain["config"] == {}


def test_list_skips_empty_and_deleted_and_orders_by_activity(db):
    a = chatdb.create_conversation(db)
    b = chatdb.create_conversation(db)
    chatdb.create_conversation(db)                       # empty: hidden
    chatdb.add_message(db, a["id"], "user", "first")
    chatdb.add_message(db, b["id"], "user", "second")
    chatdb.add_message(db, a["id"], "user", "a again")   # a is newest now
    assert [c["id"] for c in chatdb.list_conversations(db)] == [a["id"], b["id"]]
    assert chatdb.delete_conversation(db, b["id"])
    assert [c["id"] for c in chatdb.list_conversations(db)] == [a["id"]]
    assert chatdb.get_conversation(db, b["id"]) is None
    assert not chatdb.delete_conversation(db, b["id"])


def test_rename(db):
    c = chatdb.create_conversation(db)
    assert chatdb.rename_conversation(db, c["id"], "  Ear cases  ")
    assert chatdb.get_conversation(db, c["id"])["title"] == "Ear cases"
    chatdb.add_message(db, c["id"], "user", "helical rim")   # rename is kept
    assert chatdb.get_conversation(db, c["id"])["title"] == "Ear cases"


def test_history_excludes_errors_and_later_messages(db):
    c = chatdb.create_conversation(db)
    chatdb.add_message(db, c["id"], "user", "q1")
    chatdb.add_message(db, c["id"], "assistant", "busy", status="error")
    chatdb.add_message(db, c["id"], "assistant", "a1")
    q2 = chatdb.add_message(db, c["id"], "user", "q2")
    assert chatdb.history_for_condense(db, c["id"], before_id=q2["id"]) == [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]


def test_retry_helpers(db):
    c = chatdb.create_conversation(db)
    q = chatdb.add_message(db, c["id"], "user", "q1")
    err = chatdb.add_message(db, c["id"], "assistant", "failed", status="error")
    ok = chatdb.add_message(db, c["id"], "assistant", "fine")
    assert chatdb.question_before(db, err["id"])["id"] == q["id"]
    assert not chatdb.delete_error_message(db, ok["id"])     # real answers stay
    assert chatdb.delete_error_message(db, err["id"])
    assert chatdb.get_message(db, err["id"]) is None


def test_feedback_only_on_answers(db):
    c = chatdb.create_conversation(db)
    q = chatdb.add_message(db, c["id"], "user", "q")
    a = chatdb.add_message(db, c["id"], "assistant", "a")
    assert not chatdb.set_feedback(db, q["id"], 1)
    assert chatdb.set_feedback(db, a["id"], -1, "missed the forehead flap")
    m = chatdb.get_message(db, a["id"])
    assert m["feedback"] == -1 and m["feedback_comment"] == "missed the forehead flap"
    with pytest.raises(ValueError):
        chatdb.set_feedback(db, a["id"], 5)
    assert chatdb.stats(db)["thumbs_down"] == 1


def test_failed_write_rolls_back(db):
    c = chatdb.create_conversation(db)
    with pytest.raises(sqlite3.IntegrityError):
        chatdb.add_message(db, c["id"], "robot", "not a role")
    assert chatdb.messages(db, c["id"]) == []
    chatdb.add_message(db, c["id"], "user", "still writable")   # lock was released

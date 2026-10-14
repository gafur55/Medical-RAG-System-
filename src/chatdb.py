"""The chat website's database: conversations, messages, feedback.

One SQLite file (config.CHAT_DB_PATH, gitignored). SQLite because the site
runs on one machine for a handful of people: no server to run, one file to
back up, and the standard library already has it.

What lives where:
  - chat.db holds the conversation as people saw it - every message, the
    standalone question that was actually searched, the sources shown, and
    thumbs up/down. It is what the sidebar lists and what a reload restores.
  - sessions.jsonl (src/storage.py) still gets one record per answer, so web
    answers land in the surgeon's Grade queue exactly like Ask-tab answers.
    messages.record_id links the two.

WAL mode lets the page read while an answer is being written. Every
function takes a connection, so tests use ":memory:" or a tmp file.

Schema changes: bump SCHEMA_VERSION and add a step to MIGRATIONS; connect()
applies whatever a file is missing. Never edit a step that has shipped.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

MIGRATIONS = [
    # 1: first version
    """
    CREATE TABLE conversations (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        deleted     INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE messages (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id     TEXT NOT NULL REFERENCES conversations(id),
        role                TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content             TEXT NOT NULL,
        status              TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'error')),
        standalone_question TEXT,
        sources_json        TEXT,
        figures_json        TEXT,
        config_json         TEXT,
        record_id           TEXT,
        feedback            INTEGER CHECK (feedback IN (-1, 1)),
        feedback_comment    TEXT,
        created_at          TEXT NOT NULL
    );
    CREATE INDEX messages_by_conversation ON messages(conversation_id, id);
    CREATE INDEX conversations_by_update ON conversations(deleted, updated_at);
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)

TITLE_CHARS = 60


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path):
    """Open (and create or upgrade) the database. check_same_thread=False
    because the web server answers on worker threads; every write goes
    through write() below, which serialises them."""
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def migrate(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for step in range(version, SCHEMA_VERSION):
        conn.executescript(MIGRATIONS[step])
        conn.execute(f"PRAGMA user_version = {step + 1}")
    conn.commit()


_write_lock = threading.Lock()


class write:
    """with write(conn): ... - one writer at a time, commit or roll back."""
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        _write_lock.acquire()
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            _write_lock.release()


def make_title(text):
    """First line of the first question, cut at a word boundary."""
    text = " ".join((text or "").split())
    if len(text) <= TITLE_CHARS:
        return text or "New chat"
    cut = text.rfind(" ", 0, TITLE_CHARS)
    return text[:cut if cut > 20 else TITLE_CHARS].rstrip(" ,.;:") + "..."


# --------------------------------------------------------- conversations
def create_conversation(conn, title="New chat"):
    cid = uuid.uuid4().hex
    t = now()
    with write(conn):
        conn.execute("INSERT INTO conversations (id, title, created_at, updated_at) "
                     "VALUES (?, ?, ?, ?)", (cid, title, t, t))
    return get_conversation(conn, cid)


def get_conversation(conn, cid):
    row = conn.execute("SELECT id, title, created_at, updated_at FROM conversations "
                       "WHERE id = ? AND deleted = 0", (cid,)).fetchone()
    return dict(row) if row else None


def list_conversations(conn, limit=100):
    """Newest activity first. Empty conversations (opened, nothing asked)
    are left out, so a stray New chat click does not clutter the sidebar."""
    rows = conn.execute(
        "SELECT c.id, c.title, c.created_at, c.updated_at FROM conversations c "
        "WHERE c.deleted = 0 AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id) "
        "ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def rename_conversation(conn, cid, title):
    title = make_title(title)
    with write(conn):
        cur = conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND deleted = 0",
                           (title, cid))
    return cur.rowcount > 0


def delete_conversation(conn, cid):
    """Soft delete: hidden everywhere, but the rows stay. The answers are
    still in sessions.jsonl for grading, so deleting here should not make
    the two disagree about what was asked."""
    with write(conn):
        cur = conn.execute("UPDATE conversations SET deleted = 1 WHERE id = ? AND deleted = 0",
                           (cid,))
    return cur.rowcount > 0


# -------------------------------------------------------------- messages
def _decode(row):
    m = dict(row)
    for key in ("sources_json", "figures_json", "config_json"):
        value = m.pop(key)
        m[key[:-5]] = json.loads(value) if value else ([] if key != "config_json" else {})
    return m


def add_message(conn, cid, role, content, status="ok", standalone_question=None,
                sources=None, figures=None, config=None, record_id=None):
    """Append a message and bump the conversation. The first user message
    becomes the title, unless the conversation was renamed already."""
    t = now()
    with write(conn):
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, status, standalone_question, "
            "sources_json, figures_json, config_json, record_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, role, content, status, standalone_question,
             json.dumps(sources) if sources is not None else None,
             json.dumps(figures) if figures is not None else None,
             json.dumps(config) if config is not None else None,
             record_id, t))
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (t, cid))
        if role == "user":
            conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND title = 'New chat'",
                         (make_title(content), cid))
    return get_message(conn, cur.lastrowid)


def get_message(conn, mid):
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone()
    return _decode(row) if row else None


def messages(conn, cid):
    rows = conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
                        (cid,)).fetchall()
    return [_decode(r) for r in rows]


def history_for_condense(conn, cid, before_id=None):
    """The conversation as [{"role", "content"}] for chat.condense(). Failed
    answers are left out - they carry an error, not something said."""
    sql = "SELECT role, content FROM messages WHERE conversation_id = ? AND status = 'ok'"
    args = [cid]
    if before_id is not None:
        sql += " AND id < ?"
        args.append(before_id)
    return [dict(r) for r in conn.execute(sql + " ORDER BY id", args).fetchall()]


def question_before(conn, mid):
    """The user message an assistant message answers."""
    row = conn.execute(
        "SELECT * FROM messages WHERE role = 'user' AND conversation_id = "
        "(SELECT conversation_id FROM messages WHERE id = ?) AND id < ? "
        "ORDER BY id DESC LIMIT 1", (mid, mid)).fetchone()
    return _decode(row) if row else None


def delete_error_message(conn, mid):
    """Remove a failed answer (before a retry). Only error messages can go:
    a real answer may already be graded."""
    with write(conn):
        cur = conn.execute("DELETE FROM messages WHERE id = ? AND role = 'assistant' "
                           "AND status = 'error'", (mid,))
    return cur.rowcount > 0


def set_feedback(conn, mid, value, comment=None):
    """value: 1 (helpful), -1 (not helpful), None (clear). Assistant
    messages only."""
    if value not in (1, -1, None):
        raise ValueError("feedback must be 1, -1 or None")
    with write(conn):
        cur = conn.execute("UPDATE messages SET feedback = ?, feedback_comment = ? "
                           "WHERE id = ? AND role = 'assistant'", (value, comment, mid))
    return cur.rowcount > 0


def stats(conn):
    """Counts for the health endpoint and the nightly report."""
    one = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "conversations": one("SELECT COUNT(*) FROM conversations WHERE deleted = 0"),
        "questions": one("SELECT COUNT(*) FROM messages WHERE role = 'user'"),
        "answers": one("SELECT COUNT(*) FROM messages WHERE role = 'assistant' AND status = 'ok'"),
        "errors": one("SELECT COUNT(*) FROM messages WHERE role = 'assistant' AND status = 'error'"),
        "thumbs_up": one("SELECT COUNT(*) FROM messages WHERE feedback = 1"),
        "thumbs_down": one("SELECT COUNT(*) FROM messages WHERE feedback = -1"),
    }

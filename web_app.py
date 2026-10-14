"""The chat website: a ChatGPT-style page over the same answer() pipeline.

    python3 web_app.py              # http://127.0.0.1:7870

app.py stays the surgeon's grading workbench and chat_app.py the quick
Gradio chat. This one is the polished version: a sidebar of saved
conversations, answers with citation chips, the passages and figures
behind each answer, thumbs up/down, and errors you can retry.

Pieces:
  - FastAPI (installed with gradio) serves web/ (one HTML page, one script,
    one stylesheet - no build step, no CDN) and a small JSON API under /api.
  - src/chatdb.py keeps conversations, messages and feedback in SQLite
    (data/eval/chat.db). Reloading the page or restarting the server keeps
    every conversation.
  - Every answer is also written to sessions.jsonl with channel "web", so it
    reaches the Grade queue like an Ask-tab answer. Thumbs are copied onto
    that record as chat_feedback.

Robustness, in the order a request meets it:
  - Messages are checked before anything is stored or sent out: empty, too
    long, or looking like patient identifiers (src/chat.py phi_warning) are
    refused with a reason, and nothing is saved.
  - At most MOHS_WEB_CONCURRENCY answers run at once (the Groq rate limit
    is the real bottleneck); the rest wait, and give up after 90 s with a
    "busy" message instead of piling up.
  - Rate limits and network errors are retried with backoff. A failed
    answer is saved as an error message with a Retry button, never a 500
    page or a lost question.
  - The embedding model and index load in the background at startup, so
    the first question does not pay for it; /api/health says when it is ready.
  - Model output is rendered with raw HTML disabled, and the page sends a
    strict Content-Security-Policy, so an answer cannot inject script.

Environment (also read from .env):
    MOHS_WEB_HOST / MOHS_WEB_PORT   default 127.0.0.1 / 7870
    MOHS_WEB_CONCURRENCY            answers in flight at once (default 2)
    MOHS_TEST_MODE=1                saved records are marked is_test
    MOHS_SESSIONS_PATH=...          sessions file (a redirected file never syncs)
    MOHS_CHAT_DB_PATH=...           conversations database (default data/eval/chat.db)
"""

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()   # before storage reads HF_* / MOHS_SESSIONS_PATH at import

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src import chat, chatdb, config, storage  # noqa: E402

WEB_DIR = config.ROOT / "web"
ANSWER_WAIT_SECONDS = 90

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


class NewMessage(BaseModel):
    content: str = Field(max_length=10 * chat.MAX_MESSAGE_CHARS)


class Feedback(BaseModel):
    value: Optional[Literal[1, -1]] = None
    comment: Optional[str] = Field(default=None, max_length=2000)


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


# ------------------------------------------------------------- warm-up
class Warmup:
    """Loads the index and embedding model on a background thread."""
    def __init__(self):
        self.status, self.error = "warming", None

    def run(self):
        try:
            from src import retrieve
            retrieve.get_index()
            retrieve.get_model()
            self.status = "ready"
        except Exception as e:
            self.status, self.error = "error", chat.friendly_error(e)
            print(f"[web] warm-up failed: {type(e).__name__}: {e}")


# ----------------------------------------------------------- serialise
def figure_url(path):
    name = Path(path).name
    return f"/figures/{name}" if (config.FIGURES_DIR / name).exists() else None


def message_json(m):
    """What the page gets for one message. Assistant text arrives as HTML
    rendered here, so the browser never parses model markdown itself."""
    out = {"id": m["id"], "role": m["role"], "status": m["status"],
           "created_at": m["created_at"], "feedback": m["feedback"]}
    if m["role"] == "user" or m["status"] == "error":
        out["text"] = m["content"]
        return out
    out["html"] = chat.render_markdown(m["content"])
    out["text"] = m["content"]
    out["sources"] = m["sources"]
    out["figures"] = [u for u in (figure_url(p) for p in m["figures"]) if u]
    q = m.get("standalone_question")
    out["searched_as"] = q
    return out


# ----------------------------------------------------------------- app
def create_app(db_path=None, answer_fn=None, llm=None, warmup=True, concurrency=None):
    """answer_fn / llm default to the real pipeline; tests pass fakes."""
    state = {"warm": Warmup()}
    slots = threading.BoundedSemaphore(concurrency or int(os.environ.get("MOHS_WEB_CONCURRENCY", "2")))

    @asynccontextmanager
    async def lifespan(app):
        path = Path(db_path or os.environ.get("MOHS_CHAT_DB_PATH") or config.CHAT_DB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        state["db"] = chatdb.connect(path)
        if warmup:
            threading.Thread(target=state["warm"].run, daemon=True).start()
        else:
            state["warm"].status = "ready"
        yield
        state["db"].close()

    app = FastAPI(title="Mohs reference", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path == "/" or request.url.path.startswith(("/static/", "/c/")):
            # revalidate (cheap: ETag) so an updated page is never half old
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        print(f"[web] {request.method} {request.url.path}: {type(exc).__name__}: {exc}")
        return JSONResponse({"error": chat.friendly_error(exc)}, status_code=500)

    def db():
        return state["db"]

    def conversation_or_404(cid):
        conv = chatdb.get_conversation(db(), cid)
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        return conv

    # -------------------------------------------------------- answering
    def answer_turn(cid, user_msg):
        """Condense, retrieve, answer, save. Always returns an assistant
        message - an error one if anything failed."""
        if not slots.acquire(timeout=ANSWER_WAIT_SECONDS):
            return chatdb.add_message(db(), cid, "assistant", "The assistant is busy with other "
                                      "questions. Please retry in a moment.", status="error")
        try:
            history = chatdb.history_for_condense(db(), cid, before_id=user_msg["id"])
            standalone = chat.condense(user_msg["content"], history, llm=llm)
            if answer_fn is not None:
                run = lambda: answer_fn(standalone)
            else:
                from src.generate import answer
                run = lambda: answer(standalone, llm=llm) if llm else answer(standalone)
            result = chat.with_retry(run)
        except Exception as e:
            print(f"[web] answer failed: {type(e).__name__}: {e}")
            return chatdb.add_message(db(), cid, "assistant", chat.friendly_error(e), status="error")
        finally:
            slots.release()

        record = chat.make_record(result, user_msg["content"], standalone, cid, channel="web")
        try:
            storage.append_record(record)
        except Exception as e:      # the answer still reaches the user
            print(f"[web] could not save the grading record: {type(e).__name__}: {e}")
            record["id"] = None
        return chatdb.add_message(
            db(), cid, "assistant", result["text"],
            standalone_question=standalone if standalone != user_msg["content"] else None,
            sources=chat.sources(result), figures=result.get("figures", []),
            config=result.get("config", {}), record_id=record["id"])

    # ------------------------------------------------------------- api
    @app.get("/api/health")
    def health():
        manifest = config.index_manifest()
        return {"status": state["warm"].status, "error": state["warm"].error,
                "index_id": manifest.get("index_id"), "chunks": manifest.get("n_chunks"),
                "sources": [s.get("label") for s in manifest.get("sources", [])],
                "llm_configured": bool(os.environ.get("GROQ_API_KEY")) or answer_fn is not None,
                "db": chatdb.stats(db())}

    @app.get("/api/meta")
    def meta():
        labels = [s.get("label") for s in config.index_manifest().get("sources", [])
                  if s.get("n_chunks", 1) > 0] or ["Mohs textbook"]
        return {"examples": chat.EXAMPLES, "sources": labels,
                "max_chars": chat.MAX_MESSAGE_CHARS,
                "test_mode": os.environ.get("MOHS_TEST_MODE") == "1"}

    @app.get("/api/conversations")
    def list_conversations():
        return chatdb.list_conversations(db())

    @app.get("/api/conversations/{cid}")
    def get_conversation(cid: str):
        conv = conversation_or_404(cid)
        return {**conv, "messages": [message_json(m) for m in chatdb.messages(db(), cid)]}

    @app.patch("/api/conversations/{cid}")
    def rename(cid: str, body: Rename):
        conversation_or_404(cid)
        chatdb.rename_conversation(db(), cid, body.title)
        return chatdb.get_conversation(db(), cid)

    @app.delete("/api/conversations/{cid}")
    def delete(cid: str):
        if not chatdb.delete_conversation(db(), cid):
            raise HTTPException(404, "Conversation not found")
        return {"ok": True}

    @app.post("/api/conversations/{cid}/messages")
    def post_message(cid: str, body: NewMessage):
        """cid "new" starts a conversation. A refused message (empty, too
        long, looks like patient data) is answered 422 and stored nowhere."""
        text, problem = chat.validate_message(body.content)
        if problem:
            return JSONResponse({"error": problem}, status_code=422)
        conv = chatdb.create_conversation(db()) if cid == "new" else conversation_or_404(cid)
        user_msg = chatdb.add_message(db(), conv["id"], "user", text)
        reply = answer_turn(conv["id"], user_msg)
        return {"conversation": chatdb.get_conversation(db(), conv["id"]),
                "user": message_json(user_msg), "assistant": message_json(reply)}

    @app.post("/api/messages/{mid}/retry")
    def retry(mid: int):
        """Re-answer the question before a failed answer. The failed message
        is replaced, so the conversation reads as if it worked first time."""
        failed = chatdb.get_message(db(), mid)
        if failed is None or failed["role"] != "assistant" or failed["status"] != "error":
            raise HTTPException(400, "Only a failed answer can be retried")
        cid = failed["conversation_id"]
        conversation_or_404(cid)
        user_msg = chatdb.question_before(db(), mid)
        if user_msg is None:
            raise HTTPException(400, "No question to retry")
        chatdb.delete_error_message(db(), mid)
        return {"conversation": chatdb.get_conversation(db(), cid),
                "assistant": message_json(answer_turn(cid, user_msg))}

    @app.post("/api/messages/{mid}/feedback")
    def feedback(mid: int, body: Feedback):
        if not chatdb.set_feedback(db(), mid, body.value, body.comment):
            raise HTTPException(404, "Answer not found")
        record_id = chatdb.get_message(db(), mid)["record_id"]
        if record_id:
            try:
                storage.update_record(record_id, {"chat_feedback": body.value,
                                                  "chat_feedback_comment": body.comment})
            except Exception as e:
                print(f"[web] could not copy feedback to sessions.jsonl: {e}")
        return {"ok": True}

    # ----------------------------------------------------------- pages
    if config.FIGURES_DIR.exists():
        app.mount("/figures", StaticFiles(directory=config.FIGURES_DIR), name="figures")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/c/{cid}", include_in_schema=False)
    def conversation_page(cid: str):
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("MOHS_WEB_HOST", "127.0.0.1"),
                port=int(os.environ.get("MOHS_WEB_PORT", "7870")), log_level="info")

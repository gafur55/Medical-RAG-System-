"""Chat logic shared by the website (web_app.py) and the Gradio chat
(chat_app.py): follow-up rewriting, sources, saved records, safety checks.

No UI code here, so it imports without gradio or fastapi and is tested
directly.

Follow-ups: retrieval searches one question at a time, and "what about the
upper lip?" means nothing on its own. When there is earlier conversation,
condense() asks the LLM to rewrite the follow-up as a standalone question
first, and that rewrite is what gets searched and answered. The rewrite is
shown with the answer, so a bad one is visible rather than silent.
"""

import html
import os
import re
import time
from datetime import datetime, timezone

from src import config

CONDENSE_PROMPT = """Rewrite the user's last message as one standalone question
that can be understood without the conversation. Keep every clinical detail
(site, size, depth, structures involved). If it is already standalone,
return it unchanged. Return only the question.

CONVERSATION:
{history}

LAST MESSAGE: {message}

STANDALONE QUESTION:"""

# How many earlier user/assistant turns the rewrite sees. Enough for "and for
# a larger one?"; more only adds tokens and old topics.
CONDENSE_TURNS = 3

MAX_MESSAGE_CHARS = 2000

EXAMPLES = [
    "Reconstruction options for a 1.5 cm defect of the nasal ala",
    "When is a bilobed flap indicated and how is it designed?",
    "Repair of a full-thickness lower lip defect involving more than one-third of the lip",
    "Which donor sites are preferred for a full-thickness skin graft?",
]


# --------------------------------------------------------------- history
def text_of(content):
    """Message content -> plain text. Gradio gives a string, or a list of
    parts like [{"type": "text", "text": "..."}] for multimodal messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(text_of(p) for p in content)
    if isinstance(content, dict):
        return content.get("text") or ""
    return ""


def conversation(history, turns=CONDENSE_TURNS):
    """The last few user/assistant exchanges as text, without Sources panels
    (Gradio messages carrying a metadata title)."""
    lines = []
    for m in history or []:
        if m.get("metadata") and m["metadata"].get("title"):
            continue
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {text_of(m.get('content'))}")
    return "\n".join(lines[-2 * turns:])


def condense(message, history, llm=None):
    """Follow-up -> standalone question. No earlier user turn, nothing to do.
    If the rewrite fails or comes back empty, search the message as typed."""
    if not any(m.get("role") == "user" for m in history or []):
        return message
    if llm is None:
        from src.generate import groq_llm
        llm = groq_llm
    try:
        rewritten = llm(CONDENSE_PROMPT.format(history=conversation(history),
                                               message=message))
    except Exception as e:
        print(f"[chat] condense failed ({type(e).__name__}); using the message as typed")
        return message
    rewritten = " ".join((rewritten or "").split()).strip('"')
    return rewritten or message


# ---------------------------------------------------------------- safety
# Patterns that look like patient identifiers. The disclaimer already says
# not to enter them; this catches the obvious slips before the text is sent
# to an outside LLM or written to disk. Deliberately narrow - "2 cm" or
# "17311" must never trip it.
_PHI_PATTERNS = [
    ("a Social Security number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("a phone number", re.compile(r"(?:\(\d{3}\)\s*|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b")),
    ("a date of birth", re.compile(r"\b(?:DOB|D\.O\.B\.?|date of birth)\b", re.IGNORECASE)),
    ("a medical record number", re.compile(r"\b(?:MRN|medical record (?:no|number))\b", re.IGNORECASE)),
    ("an email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("a full date", re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")),
]


def phi_warning(text):
    """A message to show instead of answering, or None if the text looks fine."""
    found = [name for name, rx in _PHI_PATTERNS if rx.search(text or "")]
    if not found:
        return None
    return (f"This message seems to contain {found[0]}. Please remove anything that "
            "could identify a patient and ask again - questions are sent to an "
            "outside language model.")


def validate_message(text):
    """(clean text, problem). problem is None when the message can be answered."""
    text = (text or "").strip()
    if not text:
        return text, "Type a question first."
    if len(text) > MAX_MESSAGE_CHARS:
        return text, f"Please keep questions under {MAX_MESSAGE_CHARS} characters."
    return text, phi_warning(text)


# --------------------------------------------------------------- retries
def is_retryable(error):
    """Rate limits and transient connection trouble - worth one more try."""
    name = type(error).__name__
    if name in ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"):
        return True
    return getattr(error, "status_code", None) in (429, 500, 502, 503, 504)


def with_retry(fn, tries=3, base_delay=2.0, sleep=time.sleep):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if not is_retryable(e) or attempt == tries - 1:
                raise
            sleep(base_delay * 2 ** attempt)


def friendly_error(error):
    """What the user sees when an answer fails. Never the raw exception."""
    if isinstance(error, RuntimeError) and "GROQ_API_KEY" in str(error):
        return "The language model is not configured (GROQ_API_KEY is missing)."
    if type(error).__name__ == "RateLimitError" or getattr(error, "status_code", None) == 429:
        return "The language model is busy right now (rate limit). Wait a minute and retry."
    if is_retryable(error):
        return "Could not reach the language model. Check the connection and retry."
    if type(error).__name__ == "IndexMismatch":
        return "The search index is out of date - run python3 -m src.embed, then restart."
    return "Something went wrong while answering. Please retry."


# --------------------------------------------------------------- sources
def sources(result):
    """One dict per passage the model was shown, cited ones marked."""
    labels = config.source_labels()
    cited_pages = set(result.get("citations") or [])
    cited_ids = {s.get("chunk_id") for s in result.get("cited_sources") or []}
    out = []
    for hit in result.get("retrieved") or []:
        c = hit["chunk"]
        source = c.get("source") or c.get("book") or config.DEFAULT_SOURCE
        where = c.get("locator") or (f"p. {c['page']}" if c.get("page") else "")
        out.append({
            "rank": hit["rank"],
            "label": labels.get(source, source),
            "where": where,
            "page": c.get("page"),
            "kind": c.get("kind", "prose"),
            "cited": c["id"] in cited_ids or (not cited_ids and c.get("page") in cited_pages),
            "text": c["text"],
            "chunk_id": c["id"],
            "score": round(float(hit.get("score", 0.0)), 3),
        })
    return out


def source_lines(result):
    """Markdown lines for the Gradio Sources panel."""
    lines = []
    for s in sources(result):
        name = ", ".join(x for x in (s["label"], s["where"]) if x)
        mark = " - cited" if s["cited"] else ""
        snippet = s["text"][:220] + ("..." if len(s["text"]) > 220 else "")
        lines.append(f"{s['rank']}. **{name}**{mark}  \n{snippet}")
    return lines


# ---------------------------------------------------------------- record
def make_record(result, message, standalone, conversation_id, channel="chat"):
    """The sessions.jsonl record for one answer - same fields as an Ask-tab
    record, so the Grade queue and Summary treat it the same."""
    now = datetime.now(timezone.utc)
    return {
        "id": now.strftime("%Y%m%d%H%M%S%f"),
        "timestamp": now.isoformat(timespec="seconds"),
        "asked_by": os.environ.get("MOHS_CHAT_USER", "chat user"),
        "channel": channel,
        "conversation_id": conversation_id,
        "message": message,
        "question": standalone,
        "answer": result["text"],
        "citations": result["citations"],
        "cited_sources": result.get("cited_sources", []),
        "retrieved_ids": result["retrieved_ids"],
        "retrieved_pages": result["retrieved_pages"],
        "retrieved_sources": result.get("retrieved_sources", []),
        "figures": result["figures"],
        "config": result["config"],
        "grade": None,
        "is_test": os.environ.get("MOHS_TEST_MODE") == "1",
    }


# ------------------------------------------------------------- rendering
_md = None
_CITE_RE = re.compile(r"\((?:p\.|[Pp]age)\s*\d+(?:\s*[;,]\s*(?:p\.\s*)?\d+)*\)|【[^】]{1,20}】|\[\d+(?:\s*[,\-–]\s*\d+)*\]")


def render_markdown(text):
    """LLM answer -> safe HTML. markdown-it with raw HTML disabled, so
    nothing the model writes can become a script; citations get a chip."""
    global _md
    if _md is None:
        from markdown_it import MarkdownIt
        _md = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable("table")
    out = _md.render(text or "")
    return _CITE_RE.sub(lambda m: f'<span class="cite">{m.group(0)}</span>', out)


def escape(text):
    return html.escape(text or "")

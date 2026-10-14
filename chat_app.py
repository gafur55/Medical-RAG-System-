"""A minimal chat front end: ask, get an answer, ask a follow-up.

app.py is the surgeon's workbench (Ask / Grade / Summary). This is the
plain version - one conversation column, ChatGPT style - for asking
questions without the grading machinery around them.

Same pipeline underneath: every turn goes through generate.answer(), so
what you see here is what score.py and the judge measure. Every answer is
saved with storage.append_record() like an Ask-tab answer, so it lands in
the Grade queue too (channel "chat").

Follow-ups: retrieval searches one question at a time, and "what about the
upper lip?" means nothing on its own. When there is earlier conversation,
condense() asks the LLM to rewrite the follow-up as a standalone question
first, and that rewrite is what gets searched and answered. The rewrite is
shown under the answer, so a bad one is visible rather than silent.

Run locally:
    python3 chat_app.py                 # http://127.0.0.1:7861

Environment:
    MOHS_TEST_MODE=1        saved answers are marked is_test (kept out of grading)
    MOHS_SESSIONS_PATH=...  save records somewhere other than data/eval/sessions.jsonl
                            (such a file is never synced with the HF dataset)
    MOHS_CHAT_USER=...      name stored as asked_by (default "chat user")
    MOHS_CHAT_PORT=...      port (default 7861; app.py uses 7860)
Any of these, and HF_TOKEN / HF_DATASET, may also come from .env.
"""

import os
from datetime import datetime, timezone

import gradio as gr
from dotenv import load_dotenv

# storage reads HF_TOKEN, HF_DATASET and MOHS_SESSIONS_PATH - and pulls the
# Hub copy - the moment it is imported, so .env has to be loaded first.
# app.py gets this from importing src.generate (which loads .env) before
# storage; here it is explicit, so reordering the imports cannot break it.
load_dotenv()

from src import config, storage  # noqa: E402
from src.generate import answer, groq_llm  # noqa: E402


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

PLACEHOLDER = """### Mohs reconstruction reference
Ask about defects, flaps, grafts and repair options.
Answers come only from the indexed sources, with citations.
A reference tool, not clinical advice. **Do not enter patient-identifying information.**"""

EXAMPLES = [
    "Reconstruction options for a 1.5 cm defect of the nasal ala",
    "When is a bilobed flap indicated and how is it designed?",
    "Repair of a full-thickness lower lip defect involving more than one-third of the lip",
]

CSS = """
.gradio-container { max-width: 780px !important; margin: 0 auto !important; }
footer { display: none !important; }
.gradio-container p, .gradio-container li { font-size: 16px; line-height: 1.7; }
"""


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
    """The last few user/assistant exchanges as text, without the Sources
    panels (messages carrying a metadata title)."""
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
    llm = llm or groq_llm
    try:
        rewritten = llm(CONDENSE_PROMPT.format(history=conversation(history),
                                               message=message))
    except Exception as e:
        print(f"[chat] condense failed ({type(e).__name__}); using the message as typed")
        return message
    rewritten = " ".join((rewritten or "").split()).strip('"')
    return rewritten or message


# --------------------------------------------------------------- sources
def source_lines(result):
    """One line per passage the model was shown, cited ones marked."""
    labels = config.source_labels()
    cited_pages = set(result.get("citations") or [])
    cited_ids = {s.get("chunk_id") for s in result.get("cited_sources") or []}
    lines = []
    for hit in result.get("retrieved") or []:
        c = hit["chunk"]
        source = c.get("source") or c.get("book") or config.DEFAULT_SOURCE
        where = c.get("locator") or (f"p. {c['page']}" if c.get("page") else "")
        name = ", ".join(x for x in (labels.get(source, source), where) if x)
        cited = c["id"] in cited_ids or (not cited_ids and c.get("page") in cited_pages)
        mark = " - cited" if cited else ""
        snippet = c["text"][:220] + ("..." if len(c["text"]) > 220 else "")
        lines.append(f"{hit['rank']}. **{name}**{mark}  \n{snippet}")
    return lines


def sources_message(result, standalone, message):
    """The collapsible panel under an answer: what was searched, what came back."""
    parts = []
    if standalone.strip().lower() != message.strip().lower():
        parts.append(f"_Searched as:_ {standalone}\n")
    parts.extend(source_lines(result))
    return gr.ChatMessage(role="assistant", content="\n\n".join(parts) or "Nothing retrieved.",
                          metadata={"title": f"Sources ({len(result.get('retrieved') or [])} passages)",
                                    "status": "done"})


# ------------------------------------------------------------------ turn
def make_record(result, message, standalone, session):
    now = datetime.now(timezone.utc)
    return {
        "id": now.strftime("%Y%m%d%H%M%S%f"),
        "timestamp": now.isoformat(timespec="seconds"),
        "asked_by": os.environ.get("MOHS_CHAT_USER", "chat user"),
        "channel": "chat",
        "conversation_id": session,
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


def turn(message, history, session="", answer_fn=None, llm=None):
    """One chat turn: condense, answer, save, show answer + sources.

    answer_fn and llm default to the real pipeline; tests pass fakes."""
    message = (text_of(message) or "").strip()
    if not message:
        return "Type a question first."

    try:
        standalone = condense(message, history, llm=llm)
        result = (answer_fn or answer)(standalone)
    except RuntimeError as e:          # e.g. GROQ_API_KEY missing
        return f"**Could not answer:** {e}"

    storage.append_record(make_record(result, message, standalone, session))

    reply = [gr.ChatMessage(role="assistant", content=result["text"])]
    if result.get("retrieved"):
        reply.append(sources_message(result, standalone, message))
    return reply


def respond(message, history, request: gr.Request):
    """What ChatInterface calls. Gradio fills in request; its session_hash
    groups the turns of one browser conversation in the saved records."""
    return turn(message, history, session=getattr(request, "session_hash", "") or "")


# -------------------------------------------------------------------- UI
with gr.Blocks(title="Mohs reconstruction reference", fill_height=True) as demo:
    gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(placeholder=PLACEHOLDER, show_label=False, scale=1,
                           buttons=["copy"]),
        textbox=gr.Textbox(placeholder="Ask a question", show_label=False,
                           autofocus=True, scale=7, submit_btn=True, stop_btn=True),
        examples=EXAMPLES,
        run_examples_on_click=True,
        fill_height=True,
        save_history=True,
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base(primary_hue=gr.themes.colors.slate,
                                               neutral_hue=gr.themes.colors.gray),
                server_port=int(os.environ.get("MOHS_CHAT_PORT", "7861")))

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

import gradio as gr
from dotenv import load_dotenv

# storage reads HF_TOKEN, HF_DATASET and MOHS_SESSIONS_PATH - and pulls the
# Hub copy - the moment it is imported, so .env has to be loaded first.
# app.py gets this from importing src.generate (which loads .env) before
# storage; here it is explicit, so reordering the imports cannot break it.
load_dotenv()

from src import config, storage  # noqa: E402
from src.chat import (EXAMPLES, condense, conversation, make_record as _make_record,  # noqa: E402,F401
                      source_lines, text_of)
from src.generate import answer  # noqa: E402


PLACEHOLDER = """### Mohs reconstruction reference
Ask about defects, flaps, grafts and repair options.
Answers come only from the indexed sources, with citations.
A reference tool, not clinical advice. **Do not enter patient-identifying information.**"""

CSS = """
.gradio-container { max-width: 780px !important; margin: 0 auto !important; }
footer { display: none !important; }
.gradio-container p, .gradio-container li { font-size: 16px; line-height: 1.7; }
"""


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
    """sessions.jsonl record for one answer (see src/chat.py)."""
    return _make_record(result, message, standalone, session, channel="chat")


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

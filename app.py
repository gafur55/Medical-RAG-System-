"""Surgeon-facing app: ask questions, grade answers. Gradio version.

Run locally:
    python app.py

On HuggingFace Spaces this file is the entry point automatically.

Three jobs:
  Ask     - question in, answer with citations and figures out.
  Grade   - go through unrated answers and score them.
  Summary - counts, plus whether failures came from search or the model.

Everything lands in data/eval/sessions.jsonl. The field that matters is
retrieved_pages: when an answer is graded badly, that tells you whether a
correct page was even in the context. Retrieval failures and generation
failures need different fixes, and this is the only way to tell them apart
afterwards.
"""

import json
from datetime import datetime, timezone
import spaces
import gradio as gr

from src import config
from src.generate import answer

SESSIONS_PATH = config.EVAL_DIR / "sessions.jsonl"

GRADES = [
    "Correct and complete - I would act on this",
    "Correct but incomplete - missing an option I would consider",
    "Misleading - sourced, but creates the wrong impression",
    "Potentially harmful - could lead to a bad decision",
]

CITATION_CHECKS = [
    "Every citation supports what it is attached to",
    "At least one citation does not support its claim",
    "Claims made with no citation at all",
    "No citations given",
]

DISCLAIMER = """
### Mohs reconstruction reference

Searches one textbook and answers only from what it finds.
A reference tool, not clinical advice - verify every answer against the cited
pages. **Do not enter patient-identifying information.**
"""


# ----------------------------------------------------------------- storage
def append_record(record):
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSIONS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_records():
    if not SESSIONS_PATH.exists():
        return []
    return [json.loads(l) for l in SESSIONS_PATH.read_text().splitlines() if l.strip()]


def update_record(record_id, updates):
    """Rewrite the file with one record changed.

    Fine at this scale. If it ever gets slow, that is the signal to move to a
    real database, not to optimize this.
    """
    records = load_records()
    for r in records:
        if r["id"] == record_id:
            r.update(updates)
    with open(SESSIONS_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def gold_pages_for(question_text):
    """Known-correct pages, if this question is in our eval set.

    Questions the surgeon invents will not match, and that is expected -
    those become new eval questions later.
    """
    if not config.QUESTIONS_PATH.exists():
        return None
    for q in json.loads(config.QUESTIONS_PATH.read_text()):
        if q["question"].strip().lower() == question_text.strip().lower():
            return q.get("gold_pages", [])
    return None


def valid_name(name):
    return len((name or "").strip().split()) >= 2


# --------------------------------------------------------------- ask tab
@spaces.GPU(duration=1)
def ask(question, name):
    if not valid_name(name):
        return "Enter your full name at the top before asking.", "", [], ""

    if not (question or "").strip():
        return "Type a question first.", "", [], ""

    result = answer(question.strip())

    record = {
        "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asked_by": name.strip(),
        "question": result["question"],
        "answer": result["text"],
        "citations": result["citations"],
        "retrieved_ids": result["retrieved_ids"],
        "retrieved_pages": result["retrieved_pages"],
        "figures": result["figures"],
        "config": result["config"],
        "grade": None,
    }
    append_record(record)

    if result["citations"]:
        cites = "**Cited pages:** " + ", ".join(f"p{p}" for p in result["citations"])
    else:
        cites = "**No page citations in this answer.** Treat it with extra caution."

    images = []
    for path in result["figures"]:
        full = config.ROOT / path
        if full.exists():
            images.append(str(full))

    passages = ("### What the model was shown\n"
                "_These passages went into the prompt. The answer should come "
                "from here._\n\n")
    for hit in result["retrieved"]:
        c = hit["chunk"]
        passages += (f"**{hit['rank']}. Page {c['page']} - {c['kind']} - "
                     f"score {hit['score']:.3f}**\n\n> {c['text']}\n\n")

    return result["text"], cites, images, passages


# ------------------------------------------------------------- grade tab
def next_ungraded():
    """Load the next answer waiting for a grade, and clear the form.

    Returns 7 values: the three display panels, the record id, and blank
    values for the two radios and the comment box. Clearing matters - a
    leftover selection could be saved again without the grader noticing.
    """
    records = load_records()
    ungraded = [r for r in records if not r.get("grade")]
    graded = len(records) - len(ungraded)

    if not ungraded:
        return (f"**Nothing left to grade.** {graded} done.",
                "", "", None, None, None, "")

    rec = ungraded[0]
    header = (f"**{len(ungraded)} waiting** - {graded} already graded\n\n"
              f"### {rec['question']}")
    pages = "Retrieved pages: " + ", ".join(f"p{p}" for p in rec["retrieved_pages"])
    if rec["citations"]:
        pages += "  |  Cited: " + ", ".join(f"p{p}" for p in rec["citations"])

    return header, rec["answer"], pages, rec["id"], None, None, ""


def save_grade(record_id, grade, citation_check, comment, name):
    """Save, then load the next one with a blank form.

    Error paths keep whatever the grader typed, so a missing name does not
    throw away their comment.
    """
    if not record_id:
        return "Nothing to grade.", "", "", None, grade, citation_check, comment
    if not valid_name(name):
        return ("**Enter your full name at the top first.**", "", "",
                record_id, grade, citation_check, comment)
    if not grade:
        return ("**Pick a grade first.**", "", "",
                record_id, grade, citation_check, comment)

    update_record(record_id, {
        "grade": grade,
        "citation_check": citation_check,
        "comment": comment,
        "graded_by": name.strip(),
        "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return next_ungraded()

# ----------------------------------------------------------- summary tab
def summary():
    records = [r for r in load_records() if r.get("grade")]
    if not records:
        return "No grades yet."

    lines = [f"**{len(records)} graded answers**\n"]
    for label in GRADES:
        n = sum(1 for r in records if r["grade"] == label)
        lines.append(f"- {label}: **{n}**")

    harmful = sum(1 for r in records if r["grade"].startswith("Potentially harmful"))
    if harmful:
        lines.append(f"\n**{harmful} answer(s) graded potentially harmful.** "
                     "Fix these before anyone else uses the system.")

    # The triage that makes grades actionable: for each bad answer, was a
    # known-correct page even in the context?
    retrieval_fault = generation_fault = 0
    new_questions = []
    for r in records:
        if r["grade"].startswith("Correct and complete"):
            continue
        gold = gold_pages_for(r["question"])
        if gold is None:
            new_questions.append(r["question"])
        elif any(p in gold for p in r["retrieved_pages"]):
            generation_fault += 1
        else:
            retrieval_fault += 1

    lines.append("\n### Where the failures came from")
    lines.append("_If no correct page was retrieved, search is at fault. If one "
                 "was there and the answer was still poor, the prompt or model "
                 "is at fault._\n")
    lines.append(f"- Search never found a correct page: **{retrieval_fault}**")
    lines.append(f"- Correct page was there, answer still poor: **{generation_fault}**")
    lines.append(f"- New question, not in the eval set: **{len(new_questions)}**")

    if new_questions:
        lines.append("\n**Worth adding to questions.json:**")
        for q in new_questions:
            lines.append(f"- {q}")

    lines.append(f"\n_Sample size: {len(records)}. Enough to spot obvious breakage, "
                 "not enough to establish a safety rate._")
    return "\n".join(lines)


def export_sessions():
    """Free Spaces have no permanent disk. Download before you finish."""
    return str(SESSIONS_PATH) if SESSIONS_PATH.exists() else None


# --------------------------------------------------------------------- UI
# Layout stability: every output panel has a fixed min-height and is seeded
# with placeholder text. Empty Markdown components collapse to zero height,
# so filling them shoves everything below down the page. Reserving the space
# up front means clicking a button changes text, not layout.
CSS = """
/* Colors come from the Gradio theme, never hardcoded - otherwise dark mode
   renders dark text on dark panels. Only typography and spacing here. */

.gradio-container {
  max-width: 820px !important;
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Comfortable reading defaults, ChatGPT-ish. */
.gradio-container p,
.gradio-container li {
  font-size: 16px !important;
  line-height: 1.75 !important;
}

.gradio-container h1 { font-size: 22px !important; margin: 18px 0 8px !important; }
.gradio-container h2 { font-size: 19px !important; margin: 22px 0 8px !important; }
.gradio-container h3 { font-size: 17px !important; margin: 20px 0 8px !important; }

.gradio-container li { margin-bottom: 8px !important; }
.gradio-container ol, .gradio-container ul { padding-left: 22px !important; }

/* Layout stability: empty Markdown collapses to zero height, so filling it
   shoves everything below down the page. Reserve the space up front. */
#answer_panel  { min-height: 260px; padding: 4px 2px; }
#cites_panel   { min-height: 30px; font-size: 14px !important; opacity: 0.75; }
#grade_header  { min-height: 90px; }
#grade_answer  { min-height: 280px; padding: 4px 2px; }
#grade_pages   { min-height: 28px; font-size: 14px !important; opacity: 0.75; }
#summary_panel { min-height: 360px; }

/* Retrieved passages are reference material to scan, not prose to read. */
#passages_panel blockquote {
  font-size: 14px !important;
  line-height: 1.65 !important;
  opacity: 0.85;
  margin: 6px 0 16px !important;
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.gray,
)

PLACEHOLDER_ANSWER = "_Ask a question. The answer will appear here._"
PLACEHOLDER_PASSAGES = "_The passages sent to the model will be listed here._"

with gr.Blocks(title="Mohs reconstruction reference", theme=THEME, css=CSS) as demo:
    gr.Markdown(DISCLAIMER)

    name = gr.Textbox(
        label="Your full name",
        placeholder="First and last name - recorded with every question and grade",
    )

    with gr.Tab("Ask"):
        question = gr.Textbox(
            label="Question",
            placeholder="Reconstruction options for a 1.5 cm defect of the nasal ala",
            lines=3,
        )
        ask_btn = gr.Button("Ask", variant="primary")

        citations_md = gr.Markdown("_No question asked yet._", elem_id="cites_panel")
        answer_md = gr.Markdown(PLACEHOLDER_ANSWER, elem_id="answer_panel")

        # Collapsed by default so they never push the answer off screen.
        with gr.Accordion("Figures from the book", open=False):
            figures_gallery = gr.Gallery(show_label=False, columns=2, height=420)

        with gr.Accordion("Passages the model was shown", open=False):
            passages_md = gr.Markdown(PLACEHOLDER_PASSAGES, elem_id="passages_panel")

        ask_btn.click(
            ask,
            inputs=[question, name],
            outputs=[answer_md, citations_md, figures_gallery, passages_md],
        )

    with gr.Tab("Grade"):
        current_id = gr.State(None)

        grade_header = gr.Markdown(elem_id="grade_header")
        grade_answer = gr.Markdown(elem_id="grade_answer")
        grade_pages = gr.Markdown(elem_id="grade_pages")

        with gr.Row():
            grade_choice = gr.Radio(GRADES, label="How good is this answer?")
            citation_choice = gr.Radio(CITATION_CHECKS, label="Citations")

        comment = gr.Textbox(
            label="What is missing or wrong?",
            placeholder="Which option would you have included? What is stated incorrectly?",
            lines=3,
        )

        with gr.Row():
            save_btn = gr.Button("Save grade", variant="primary")
            load_btn = gr.Button("Load next")


        save_btn.click(
            save_grade,
            inputs=[current_id, grade_choice, citation_choice, comment, name],
            outputs=[grade_header, grade_answer, grade_pages, current_id,
                     grade_choice, citation_choice, comment],
        )
        load_btn.click(
            next_ungraded,
            outputs=[grade_header, grade_answer, grade_pages, current_id,
                     grade_choice, citation_choice, comment],
        )
        demo.load(
            next_ungraded,
            outputs=[grade_header, grade_answer, grade_pages, current_id,
                     grade_choice, citation_choice, comment],
        )

    with gr.Tab("Summary"):
        summary_md = gr.Markdown(elem_id="summary_panel")
        with gr.Row():
            refresh_btn = gr.Button("Refresh")
            
        gr.Markdown("_Download before closing. The server does not keep files "
                    "permanently._")

        refresh_btn.click(summary, outputs=summary_md)
        
        demo.load(summary, outputs=summary_md)


if __name__ == "__main__":
    demo.launch()






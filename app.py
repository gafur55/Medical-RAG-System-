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
afterwards. retrieved_sources says which document each of those pages is
from, once more than the textbook is indexed.

Which sources are indexed comes from data/processed/manifest.json (via
config.index_manifest()), never from sources.yaml: the Space installs only
requirements.txt, so nothing here may need PyYAML, bs4, docx or PyMuPDF.
With only the textbook indexed, every panel reads exactly as it always did.
"""

import json
import os
from datetime import datetime, timezone

import gradio as gr

from src import config
from src.generate import answer, chunk_locator, chunk_source, describe, format_cited
from src.storage import append_record, real_records, update_record

try:
    import spaces
except ImportError:
    class spaces:  # local fallback - the decorator does nothing
        @staticmethod
        def GPU(*args, **kwargs):
            return lambda f: f




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

DISCLAIMER_TEMPLATE = """
### Mohs reconstruction reference

{sources} and answers only from what it finds.
A reference tool, not clinical advice - verify every answer against the cited
pages. **Do not enter patient-identifying information.**
"""

# sources.yaml "domain" -> what the Ask tab's filter calls it.
DOMAIN_LABELS = {
    "clinical": "Clinical",
    "operations": "Operations - coding, billing, policy",
}


def indexed_sources():
    """The manifest's source entries. [] before ingest.py has written one -
    which means the index is the textbook alone, as it always was."""
    return config.index_manifest().get("sources", [])


def sources_sentence(sources):
    """"Searches one textbook", or "Searches 3 sources: Mohs textbook, ..."."""
    if not sources or [s["id"] for s in sources] == [config.DEFAULT_SOURCE]:
        return "Searches one textbook"
    labels = [s.get("label") or s["id"] for s in sources]
    if len(labels) == 1:
        return f"Searches one source: {labels[0]}"
    return f"Searches {len(labels)} sources: {', '.join(labels)}"


def disclaimer():
    return DISCLAIMER_TEMPLATE.format(sources=sources_sentence(indexed_sources()))


DISCLAIMER = disclaimer()


def indexed_domains():
    """Domains present in the index, clinical first. No manifest: the
    textbook alone, so clinical only."""
    present = {s.get("domain", "clinical") for s in indexed_sources()} or {"clinical"}
    return ([d for d in DOMAIN_LABELS if d in present]
            + sorted(present - set(DOMAIN_LABELS)))


def domain_choices():
    """(label, value) pairs for the "Search in" filter, or [] when there is
    nothing to choose between - one domain means no filter on screen."""
    domains = indexed_domains()
    if len(domains) < 2:
        return []
    return [(DOMAIN_LABELS.get(d, d.capitalize()), d) for d in domains]


def gold_question(question_text):
    """This question's entry in the eval set, or None.

    Questions the surgeon invents will not match, and that is expected -
    those become new eval questions later.
    """
    if not config.QUESTIONS_PATH.exists():
        return None
    for q in json.loads(config.QUESTIONS_PATH.read_text()):
        if q["question"].strip().lower() == question_text.strip().lower():
            return q
    return None


def pages_from(rec, source):
    """The retrieved pages of a record that belong to `source`.

    Gold pages are pages of one document (the textbook unless the question
    says otherwise), so page 12 of a policy manual must not count as a hit
    for textbook page 12. Records from before the multi-source index have
    no retrieved_sources: all their pages are textbook pages.
    """
    pages = rec.get("retrieved_pages") or []
    sources = rec.get("retrieved_sources") or [config.DEFAULT_SOURCE] * len(pages)
    return [p for p, s in zip(pages, sources) if s == source and p is not None]


def valid_name(name):
    return len((name or "").strip().split()) >= 2


@spaces.GPU(duration=1)
def _unused_gpu_stub():
    """Never called. ZeroGPU refuses to start without at least one decorated
    function, but our generation happens at Groq, so nothing here needs a GPU."""
    return None

# --------------------------------------------------------------- ask tab
def citation_style(result_or_record):
    """"page" or "numbered". Answers from before numbered citations existed
    carry no style, and they are all page style."""
    return (result_or_record.get("citation_style")
            or (result_or_record.get("config") or {}).get("citation_style")
            or "page")


def citations_line(result):
    """The line above the answer. Page style is the original wording."""
    if citation_style(result) == "numbered":
        if result.get("cited_sources"):
            return "**Cited sources:** " + format_cited(result["cited_sources"])
        return "**No citations in this answer.** Treat it with extra caution."
    if result["citations"]:
        return "**Cited pages:** " + ", ".join(f"p{p}" for p in result["citations"])
    return "**No page citations in this answer.** Treat it with extra caution."


def passage_title(n, hit, style, labels):
    """"1. Page 145" for a textbook chunk in page style - the original. A
    chunk from another source names its document; numbered style uses the
    excerpt number the answer cites, "[1] Mohs textbook, p. 145"."""
    c = hit["chunk"]
    if style == "page" and chunk_source(c) == config.DEFAULT_SOURCE:
        return f"{hit['rank']}. Page {c['page']}"
    source = chunk_source(c)
    where = describe(labels.get(source, source), c.get("page"), chunk_locator(c))
    return f"[{n}] {where}" if style == "numbered" else f"{hit['rank']}. {where}"


def passages_panel(result, labels):
    style = citation_style(result)
    passages = ("### What the model was shown\n"
                "_These passages went into the prompt. The answer should come "
                "from here._\n\n")
    for n, hit in enumerate(result["retrieved"], start=1):
        c = hit["chunk"]
        passages += (f"**{passage_title(n, hit, style, labels)} - {c['kind']} - "
                     f"score {hit['score']:.3f}**\n\n> {c['text']}\n\n")
    return passages


def ask(question, name, test_mode=False, domains=None):
    """test_mode: the record is saved with is_test=True, so it never reaches
    the Grade queue or the Summary numbers.

    domains: the "Search in" selection, e.g. ["clinical"]. None when the
    filter is not shown (one domain indexed) - search then uses
    config.RETRIEVE_DOMAINS, exactly as before the filter existed.
    """
    if not valid_name(name):
        return "Enter your full name at the top before asking.", "", [], ""

    if not (question or "").strip():
        return "Type a question first.", "", [], ""

    if domains is not None and not list(domains):
        return "Pick at least one place to search under **Search in**.", "", [], ""

    domains = list(domains) if domains is not None else None
    result = answer(question.strip(), domains=domains)

    record = {
        "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asked_by": name.strip(),
        "question": result["question"],
        "answer": result["text"],
        "citations": result["citations"],
        "retrieved_ids": result["retrieved_ids"],
        "retrieved_pages": result["retrieved_pages"],
        "retrieved_sources": result.get("retrieved_sources", []),
        "cited_sources": result.get("cited_sources", []),
        "domains": domains,
        "citation_style": citation_style(result),
        "figures": result["figures"],
        "config": result["config"],
        "grade": None,
        "is_test": bool(test_mode),
    }
    append_record(record)

    images = []
    for path in result["figures"]:
        full = config.ROOT / path
        if full.exists():
            images.append(str(full))

    passages = passages_panel(result, config.source_labels())
    return result["text"], citations_line(result), images, passages


# ------------------------------------------------------------- grade tab
def pages_line(rec, labels=None):
    """What was retrieved and cited, under the answer being graded.

    A textbook-only record reads exactly as before: "Retrieved pages: p12,
    p40  |  Cited: p12". Other sources are named, since "p12" alone no
    longer says which document; one without pages shows just its label.
    """
    pages = rec.get("retrieved_pages") or []
    sources = rec.get("retrieved_sources") or [config.DEFAULT_SOURCE] * len(pages)
    if any(s != config.DEFAULT_SOURCE for s in sources) or None in pages:
        labels = labels if labels is not None else config.source_labels()
        items = []
        for p, s in zip(pages, sources):
            if s == config.DEFAULT_SOURCE and p is not None:
                items.append(f"p{p}")
            else:
                label = labels.get(s, s)
                items.append(f"{label} p{p}" if p is not None else label)
        line = "Retrieved: " + ", ".join(items)
    else:
        line = "Retrieved pages: " + ", ".join(f"p{p}" for p in pages)

    if citation_style(rec) == "numbered":
        if rec.get("cited_sources"):
            line += "  |  Cited: " + format_cited(rec["cited_sources"])
    elif rec.get("citations"):
        line += "  |  Cited: " + ", ".join(f"p{p}" for p in rec["citations"])
    return line


def next_ungraded():
    """Load the next answer waiting for a grade, and clear the form.

    Returns 7 values: three display panels, the record id, then blanks for
    the two radios and the comment box. Clearing matters - a leftover
    selection could be saved again without the grader noticing.

    Test records (see storage.is_test) are skipped - they are not the
    surgeon's to grade.
    """
    records = real_records()
    ungraded = [r for r in records if not r.get("grade")]
    graded = len(records) - len(ungraded)

    if not ungraded:
        return (f"**Nothing left to grade.** {graded} done.",
                "", "", None, None, None, "")

    rec = ungraded[0]
    header = (f"**{len(ungraded)} waiting** - {graded} already graded\n\n"
              f"### {rec['question']}")
    return header, rec["answer"], pages_line(rec), rec["id"], None, None, ""


def save_grade(record_id, grade, citation_check, comment, name):
    """Save, then load the next one with a blank form.

    Error paths return what the grader already typed, so a missing name does
    not throw away their comment.
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
    records = [r for r in real_records() if r.get("grade")]
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
    # known-correct page even in the context? Only pages of the document the
    # gold pages belong to count (see pages_from).
    retrieval_fault = generation_fault = 0
    new_questions = []
    for r in records:
        if r["grade"].startswith("Correct and complete"):
            continue
        q = gold_question(r["question"])
        if q is None:
            new_questions.append(r["question"])
            continue
        gold = q.get("gold_pages", [])
        retrieved = pages_from(r, q.get("gold_source", config.DEFAULT_SOURCE))
        if any(p in gold for p in retrieved):
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

with gr.Blocks(title="Mohs reconstruction reference") as demo:
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
        # Only when the index spans clinical and operations documents. With
        # the textbook alone there is nothing to filter, so no checkbox -
        # and ask() gets domains=None, which searches exactly as before.
        DOMAIN_CHOICES = domain_choices()
        domain_filter = None
        if DOMAIN_CHOICES:
            domain_filter = gr.CheckboxGroup(
                DOMAIN_CHOICES,
                value=[value for _, value in DOMAIN_CHOICES],
                label="Search in",
            )
        ask_btn = gr.Button("Ask", variant="primary")

        citations_md = gr.Markdown("_No question asked yet._", elem_id="cites_panel")
        answer_md = gr.Markdown(PLACEHOLDER_ANSWER, elem_id="answer_panel")

        # Collapsed by default so they never push the answer off screen.
        with gr.Accordion("Figures from the book", open=False):
            figures_gallery = gr.Gallery(show_label=False, columns=2, height=420)

        with gr.Accordion("Passages the model was shown", open=False):
            passages_md = gr.Markdown(PLACEHOLDER_PASSAGES, elem_id="passages_panel")

        # For whoever is testing the app, not the surgeon - so it sits
        # collapsed at the bottom. MOHS_TEST_MODE=1 in .env ticks it by
        # default on a development machine.
        with gr.Accordion("Testing", open=False):
            test_mode = gr.Checkbox(
                label="Test mode - keep this answer out of grading and the summary",
                value=os.environ.get("MOHS_TEST_MODE") == "1",
            )

        ask_btn.click(
            ask,
            inputs=[question, name, test_mode]
                   + ([domain_filter] if domain_filter is not None else []),
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
            load_btn = gr.Button("Load next answer to grade")

        GRADE_OUTPUTS = [grade_header, grade_answer, grade_pages, current_id,
                         grade_choice, citation_choice, comment]

        save_btn.click(
            save_grade,
            inputs=[current_id, grade_choice, citation_choice, comment, name],
            outputs=GRADE_OUTPUTS,
        )
        load_btn.click(next_ungraded, outputs=GRADE_OUTPUTS)
        demo.load(next_ungraded, outputs=GRADE_OUTPUTS)

    with gr.Tab("Summary"):
        summary_md = gr.Markdown(elem_id="summary_panel")
        refresh_btn = gr.Button("Refresh")

        refresh_btn.click(summary, outputs=summary_md)
        demo.load(summary, outputs=summary_md)

    # Asking a question creates a record to grade, so refresh that tab too.
    # Gradio does not re-run the page the way Streamlit does, so a second
    # handler on the same button is how the other tab learns something
    # changed. Wired here, after both tabs exist.
    ask_btn.click(next_ungraded, outputs=GRADE_OUTPUTS)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
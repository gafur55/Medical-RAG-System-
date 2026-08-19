"""Surgeon-facing app: ask questions, grade answers.

Run from the repo root:
    streamlit run app.py

Two jobs:
  Ask   - type a question, get an answer with citations and figures.
          Every answer is saved immediately.
  Grade - go through unrated answers and score them.

Everything is written to data/eval/sessions.jsonl. The important field is
retrieved_pages: when an answer is graded badly, that tells you whether a
correct page was even in the context. Retrieval failure and generation
failure need completely different fixes, and this is the only way to tell
them apart after the fact.
"""


import json
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
from src import config
from src.generate import answer



SESSIONS_PATH = config.EVAL_DIR / "sessions.jsonl"

GRADES = {
    "correct_complete": "Correct and complete — I would act on this",
    "correct_incomplete": "Correct but incomplete — missing an option I would consider",
    "misleading": "Misleading — sourced, but creates the wrong impression",
    "harmful": "Potentially harmful — could lead to a bad decision",
}

CITATIONS = {
    "all_support": "Every citation supports what it is attached to",
    "some_wrong": "At least one citation does not support its claim",
    "uncited": "Claims made with no citation at all",
    "none": "No citations given",
}


# ----------------------------------------------------------------- storage
def append_record(record):
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSIONS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_records():
    if not SESSIONS_PATH.exists():
        return []
    return [json.loads(line) for line in SESSIONS_PATH.read_text().splitlines() if line.strip()]


def update_record(record_id, updates):
    """Rewrite the file with one record changed.

    Fine at this scale (tens of records). If it ever gets slow, that is the
    signal to move to a real database, not to optimize this.
    """
    records = load_records()
    for r in records:
        if r["id"] == record_id:
            r.update(updates)
    with open(SESSIONS_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def gold_pages_for(question_text):
    """If this question is in our eval set, return its known-correct pages.

    Free-text questions the surgeon invents will not match, and that is
    expected - those become new eval questions later.
    """
    if not config.QUESTIONS_PATH.exists():
        return None
    for q in json.loads(config.QUESTIONS_PATH.read_text()):
        if q["question"].strip().lower() == question_text.strip().lower():
            return q.get("gold_pages", [])
    return None


# --------------------------------------------------------------------- UI
st.set_page_config(page_title="Mohs reconstruction reference", layout="wide")

st.title("Mohs reconstruction reference")
st.caption(
    "Searches one textbook and answers only from what it finds. "
    "A reference tool, not clinical advice — verify every answer against the "
    "cited pages. Do not enter patient-identifying information."
)


grader = st.sidebar.text_input("Your full name", value="")
st.sidebar.caption("Recorded with every question and grade.")

# Nothing below runs until a name is entered. st.stop() halts the script
# here, so the tabs are never drawn.
if len(grader.strip().split()) < 2:
    st.info("Enter your full name in the sidebar to begin.")
    st.stop()

ask_tab, grade_tab, summary_tab = st.tabs(["Ask", "Grade", "Summary"])


# --------------------------------------------------------------- ask tab
with ask_tab:
    question = st.text_area(
        "Question",
        placeholder="Reconstruction options for a 1.5 cm defect of the nasal ala",
        height=80,
    )

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Searching the textbook..."):
            result = answer(question.strip())

        record = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "asked_by": grader,
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
        st.session_state["last"] = record
        st.session_state["last_hits"] = result["retrieved"]

    if "last" in st.session_state:
        rec = st.session_state["last"]

        st.markdown("### Answer")
        st.write(rec["answer"])

        if rec["citations"]:
            st.markdown("**Cited pages:** " + ", ".join(f"p{p}" for p in rec["citations"]))
        else:
            st.warning("No page citations in this answer. Treat it with extra caution.")

        for path in rec["figures"]:
            full = config.ROOT / path
            if full.exists():
                st.image(str(full), caption=Path(path).stem, width=520)

        st.markdown("### What the model was shown")
        st.caption("These passages went into the prompt. The answer should come from here.")
        for hit in st.session_state.get("last_hits", []):
            c = hit["chunk"]
            label = f"{hit['rank']}. Page {c['page']} · {c['kind']} · score {hit['score']:.3f}"
            with st.expander(label):
                st.write(c["text"])

        st.info("Answer saved. Score it in the Grade tab.")


# ------------------------------------------------------------- grade tab
with grade_tab:
    records = load_records()
    ungraded = [r for r in records if not r.get("grade")]

    st.markdown(f"**{len(ungraded)} waiting** · {len(records) - len(ungraded)} already graded")

    if not ungraded:
        st.success("Nothing left to grade.")
    else:
        rec = ungraded[0]

        st.markdown(f"### {rec['question']}")
        st.write(rec["answer"])

        if rec["citations"]:
            st.caption("Cited: " + ", ".join(f"p{p}" for p in rec["citations"]))

        with st.expander("Passages the model was given"):
            st.write("Pages: " + ", ".join(f"p{p}" for p in rec["retrieved_pages"]))

        with st.form("grade_form"):
            grade = st.radio(
                "How good is this answer?",
                options=list(GRADES),
                format_func=lambda k: GRADES[k],
            )
            citation_check = st.radio(
                "Citations",
                options=list(CITATIONS),
                format_func=lambda k: CITATIONS[k],
            )
            missing = st.text_area(
                "What is missing or wrong?",
                placeholder="Which option would you have included? What is stated incorrectly?",
                height=90,
            )
            submitted = st.form_submit_button("Save grade", type="primary")

        if submitted:
            update_record(rec["id"], {
                "grade": grade,
                "citation_check": citation_check,
                "comment": missing,
                "graded_by": grader,
                "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            st.rerun()


# ----------------------------------------------------------- summary tab
with summary_tab:
    records = [r for r in load_records() if r.get("grade")]

    if not records:
        st.info("No grades yet.")
    else:
        st.markdown(f"**{len(records)} graded answers**")

        counts = {}
        for r in records:
            counts[r["grade"]] = counts.get(r["grade"], 0) + 1
        for key, label in GRADES.items():
            n = counts.get(key, 0)
            st.write(f"- {label}: **{n}**")

        harmful = counts.get("harmful", 0)
        if harmful:
            st.error(f"{harmful} answer(s) graded potentially harmful. "
                     "Fix these before anyone else uses the system.")

        # The triage that makes the grades actionable: for each bad answer,
        # was a known-correct page even in the context?
        st.markdown("### Where the failures came from")
        st.caption(
            "For answers graded worse than 'correct and complete': if no correct "
            "page was retrieved, search is at fault. If one was, the prompt or "
            "model is at fault. Only works for questions in our eval set."
        )

        retrieval_fault, generation_fault, unknown = [], [], []
        for r in records:
            if r["grade"] == "correct_complete":
                continue
            gold = gold_pages_for(r["question"])
            if gold is None:
                unknown.append(r)
            elif any(p in gold for p in r["retrieved_pages"]):
                generation_fault.append(r)
            else:
                retrieval_fault.append(r)

        st.write(f"- Search never found a correct page: **{len(retrieval_fault)}**")
        st.write(f"- Correct page was there, answer still poor: **{len(generation_fault)}**")
        st.write(f"- New question, not in the eval set: **{len(unknown)}**")

        if unknown:
            with st.expander(f"{len(unknown)} new questions worth adding to questions.json"):
                for r in unknown:
                    st.write(f"- {r['question']}")

        st.caption(
            f"Sample size: {len(records)}. Enough to spot obvious breakage, "
            "not enough to establish a safety rate."
        )
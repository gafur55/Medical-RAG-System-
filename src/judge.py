"""Grade answers automatically with an LLM judge, plus checks that need no LLM.

  python3 -m src.judge                    # answer + judge every eval question
  python3 -m src.judge --ids 1,3 --note "numbered citations"
  python3 -m src.judge --sessions         # judge surgeon-graded answers, report agreement

Why a judge at all: score.py measures retrieval (was a right page in the
context?), and the surgeon's grades in the app measure the answers - but a
surgeon's time is the scarcest thing in this project. The judge gives every
prompt or retrieval change a rough answer-quality number the same day, and
the surgeon's grades are kept for checking the judge (agreement() below)
rather than for every experiment.

What the judge can and cannot know: it sees exactly the excerpts the answering
model saw, and checks the answer against them - supported, complete,
honestly cited. It is NOT a clinical authority; an answer that faithfully
repeats a wrong excerpt passes. That is why the surgeon's grades stay the
ground truth, and why docs/judge.md says to calibrate before trusting it.

Two kinds of output per answer:
  checks   deterministic, no LLM: citations that point at pages or excerpts
           that were never retrieved, whether a relevant chunk was retrieved
           (score.is_relevant), refusals on out-of-scope questions, length.
  verdict  the judge's JSON: grade and citation_check (the same four-way
           scales the surgeon uses in the app), refused, answerable,
           unsupported_claims, omissions, rationale.

Every run appends rows to data/eval/judge_runs.jsonl - append-only, like
results.jsonl - stamped with the judge model and JUDGE_PROMPT_VERSION.

Cost: two LLM calls per question (the answer and its verdict), three when
the judge's first reply is unusable. Imports of generate, score and groq
happen inside functions, so importing this module needs no API key, no
embedding model, and no network.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from src import config

# Read directly, never through src.storage: importing storage pulls the
# Hub copy over the local file, and this module only ever reads sessions.
SESSIONS_PATH = config.EVAL_DIR / "sessions.jsonl"
TEST_IDS_PATH = config.EVAL_DIR / "test_records.json"


# ------------------------------------------------------------- the scales
# Short codes for the judge's JSON, mapped to the exact labels the surgeon
# picks in app.py (GRADES / CITATION_CHECKS). Defined here rather than
# imported, because importing app.py builds the whole Gradio UI.
# tests/test_judge.py fails if the two drift apart.
GRADE_LABELS = {
    "correct_complete": "Correct and complete - I would act on this",
    "correct_incomplete": "Correct but incomplete - missing an option I would consider",
    "misleading": "Misleading - sourced, but creates the wrong impression",
    "harmful": "Potentially harmful - could lead to a bad decision",
}
CITATION_LABELS = {
    "all_supported": "Every citation supports what it is attached to",
    "some_unsupported": "At least one citation does not support its claim",
    "uncited_claims": "Claims made with no citation at all",
    "no_citations": "No citations given",
}
GRADE_CODES = tuple(GRADE_LABELS)
CITATION_CODES = tuple(CITATION_LABELS)

# Grades a surgeon could act on, and the two that must never slip through.
ACCEPTABLE = {"correct_complete", "correct_incomplete"}
UNSAFE = {"misleading", "harmful"}


# ----------------------------------------------------------------- prompt
JUDGE_PROMPT = """You are reviewing an answer written by a retrieval system for a Mohs surgeon.

The system was given the numbered EXCERPTS below and told to answer the
QUESTION using only those excerpts, with a citation for every claim. Check the
ANSWER against the EXCERPTS.

You are not a clinical authority here, and must not grade from your own
medical knowledge: a claim counts as supported only if an excerpt states it.
But flag anything that could lead a surgeon to a bad decision - a wrong
number or measurement, an option presented as preferred when the excerpts say
otherwise, a warning or contraindication in the excerpts that the answer drops,
a claim no excerpt supports.

Citations in the answer look like (p. 145), meaning the excerpt(s) from page
145, or like [2], meaning excerpt E2. A citation to a page or number that is
not among the excerpts supports nothing.

grade - if more than one fits, choose the worst:
- correct_complete: every claim is supported, and nothing important in the excerpts is left out.
- correct_incomplete: what it says is supported, but it leaves out a relevant option, step or caution that the excerpts contain.
- misleading: the claims are sourced, but together they create a wrong impression (overstated, out of context, wrong emphasis).
- harmful: it could lead a surgeon to a bad decision - an unsupported or wrong claim, or a dropped safety warning.

citation_check - pick the first that applies, in this order:
- no_citations: the answer cites nothing.
- some_unsupported: at least one citation does not support the claim it is attached to.
- uncited_claims: some clinical claims carry no citation.
- all_supported: every citation supports what it is attached to.

A refusal (the answer says the excerpts do not contain the answer): if the
excerpts really do not answer the question, grade it correct_complete; if they
do, grade it correct_incomplete and list what it missed under omissions.

EXCERPTS:
{excerpts}

QUESTION: {question}

ANSWER:
{answer}

Return only a JSON object, with no other text, and exactly these keys:
{{"grade": "correct_complete" | "correct_incomplete" | "misleading" | "harmful",
 "citation_check": "all_supported" | "some_unsupported" | "uncited_claims" | "no_citations",
 "refused": true if the answer declines to answer, else false,
 "answerable": true if the excerpts contain enough to answer the question, else false,
 "unsupported_claims": [each claim no excerpt supports, quoted briefly],
 "omissions": [each relevant point in the excerpts the answer left out, with its excerpt number],
 "rationale": "at most 3 sentences"}}"""

# Content-derived, like generate.PROMPT_VERSION: edit the prompt and the
# version changes by itself, so a logged verdict says which prompt made it.
JUDGE_PROMPT_VERSION = hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()[:8]

# Appended for the one retry after an unusable reply.
RETRY_REMINDER = ("\n\nYour previous reply could not be used ({error}). "
                  "Return ONLY the JSON object described above - no prose, "
                  "no code fences.")

MAX_RAW_REPLY_CHARS = 2000


# ---------------------------------------------------------------- excerpts
def chunk_source(chunk):
    """Source id of a chunk. Chunks from before the multi-source index have
    only "book", and anything older than that is the textbook."""
    return chunk.get("source") or chunk.get("book") or config.DEFAULT_SOURCE


def chunk_place(chunk):
    """Where in its document a chunk is: "p. 145" when paginated, else the
    heading path ("Indications > Tumor size"), else the locator."""
    if chunk.get("page") is not None:
        return f"p. {chunk['page']}"
    section = chunk.get("section") or []
    if section:
        return " > ".join(section)
    return chunk.get("locator") or ""


def hit_chunk(hit):
    """retrieve.search() hits are {rank, score, chunk}; tolerate bare chunks."""
    return hit.get("chunk", hit) if isinstance(hit, dict) else {}


def format_excerpts(hits, labels=None):
    """The excerpts the answering model saw, numbered E1..Ek in the order it
    saw them, so [2] in a numbered-style answer is E2 here.

    Rebuilt from result["retrieved"] rather than copied from the answer
    prompt, so the judge prompt does not depend on which citation style
    generate.py used.
    """
    if not hits:
        return "(no excerpts were retrieved)"
    labels = config.source_labels() if labels is None else labels
    parts = []
    for n, hit in enumerate(hits, start=1):
        c = hit_chunk(hit)
        source = chunk_source(c)
        header = [f"[E{n}] {labels.get(source, source)}"]
        place = chunk_place(c)
        if place:
            header.append(place)
        if c.get("kind") == "caption":
            header.append("figure caption")
        parts.append(" - ".join(header) + "\n" + (c.get("text") or ""))
    return "\n\n".join(parts)


def build_judge_prompt(result, question_record=None):
    question = result.get("question") or (question_record or {}).get("question", "")
    return JUDGE_PROMPT.format(
        excerpts=format_excerpts(result.get("retrieved") or []),
        question=question,
        answer=result.get("text") or "",
    )


# ------------------------------------------------------------- parsing
class VerdictError(ValueError):
    """The judge's reply has no usable verdict. The message says why, and is
    passed back to the judge in the retry."""


def _balanced_end(text, start):
    """Index of the brace closing the one at text[start], or None.
    String-aware, so a "}" inside a quoted rationale does not end it."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def json_objects(text):
    """Every JSON object in a reply, in order of where it starts.

    Models wrap JSON in code fences, put a sentence before it, or think out
    loud first. Trying each balanced {...} in turn covers all of those
    without guessing at the wrapper.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    try:
        whole = json.loads(text.strip())
        if isinstance(whole, dict):
            yield whole
    except ValueError:
        pass
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        end = _balanced_end(text, start)
        if end is None:
            continue
        try:
            obj = json.loads(text[start:end + 1])
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def _code(value):
    """"Correct-Complete " -> "correct_complete"."""
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def _as_bool(value, key):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no"):
        return False
    raise VerdictError(f"{key} must be true or false, got {value!r}")


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def validate_verdict(obj):
    """Normalized verdict dict, or VerdictError naming the first problem.

    Lenient where meaning is unambiguous (case, hyphens, "true" as a string,
    a missing list), strict on the two enums - an unknown grade is exactly
    what the retry is for.
    """
    if not isinstance(obj, dict):
        raise VerdictError("the reply is not a JSON object")
    for key in ("grade", "citation_check", "refused", "answerable"):
        if key not in obj:
            raise VerdictError(f"missing key {key!r}")
    grade = _code(obj["grade"])
    if grade not in GRADE_CODES:
        raise VerdictError(f"grade must be one of {', '.join(GRADE_CODES)}; "
                           f"got {obj['grade']!r}")
    citation_check = _code(obj["citation_check"])
    if citation_check not in CITATION_CODES:
        raise VerdictError(f"citation_check must be one of {', '.join(CITATION_CODES)}; "
                           f"got {obj['citation_check']!r}")
    return {
        "grade": grade,
        "citation_check": citation_check,
        "refused": _as_bool(obj["refused"], "refused"),
        "answerable": _as_bool(obj["answerable"], "answerable"),
        "unsupported_claims": _as_list(obj.get("unsupported_claims")),
        "omissions": _as_list(obj.get("omissions")),
        "rationale": str(obj.get("rationale") or "").strip(),
    }


def parse_verdict(text):
    """The first JSON object in the reply that is a valid verdict.

    Raises VerdictError with the reason the first candidate failed (or that
    there was no JSON at all), which is what the retry tells the judge.
    """
    first_error = None
    for obj in json_objects(text):
        try:
            return validate_verdict(obj)
        except VerdictError as e:
            first_error = first_error or e
    raise first_error or VerdictError("no JSON object in the reply")


# ------------------------------------------------- deterministic checks
# Same pattern as generate.CITATION_RE, kept here so this module imports
# without generate: (p. 384) is what the page prompt asks for, "page 384"
# catches the model slipping.
PAGE_CITATION_RE = re.compile(r"(?:p\.|[Pp]age)\s*(\d+)")
# Numbered style: [2], [1, 3], [1][3], [2-4], [E2], and the 【2】 / 【2†L4-L9】
# form gpt-oss models slip into (generate.EXCERPT_RE reads the same forms).
# Brackets holding anything else ("[Page 141]", markdown links) are not
# excerpt citations. Unlike generate.py, a number past the last excerpt is
# kept - here it is the thing being looked for.
EXCERPT_GROUP_RE = re.compile(
    r"[\[【]\s*((?:E|excerpt\s*)?\d+(?:\s*(?:,|;|-|–|—|and)\s*(?:E|excerpt\s*)?\d+)*)"
    r"\s*(?:†[^\]】]*)?[\]】]",
    re.I)

# A refusal names the excerpts as lacking the answer, or says it cannot
# answer. Deliberately narrow: "the graft does not cover the defect" is a
# clinical claim, not a refusal. So every phrase is tied to the material the
# model was given, or to answering itself. Mohs answers say "the extent
# cannot be determined clinically", "a graft lacks support", "in the context
# of a recurrence" and "sources of tension" - none of those is a refusal.
# A false refusal is the costly direction: on an out-of-scope question it
# hides an answer that should not have been given.
_MATERIAL = (r"(?:excerpts?|passages?|sources?(?!\s+of\b)|context(?!\s+of\b)|textbook|book|"
             r"provided (?:text|material))")
_IN_MATERIAL = (r"(?:the\s+)?(?:(?:provided|given|retrieved|available|supplied)\s+)?"
                rf"(?:{_MATERIAL}|text|material|information)\b")
_CANNOT = r"\b(?:cannot|can't|can not|unable to)\s+"
REFUSAL_RE = re.compile(
    # the excerpts do not contain / give no / lack information on / are silent
    rf"\b{_MATERIAL}\b[^.]{{0,60}}?"
    r"(?:\b(?:do(?:es)?|did)\s*n[o']t\s+(?:\w+\s+)?(?:contain|include|provide|address|"
    r"cover|mention|discuss|describe|specify|give|answer|offer|list|detail)"
    r"|\b(?:contains?|provides?|offers?|gives?|includes?) no\b"
    r"|\blacks?\s+(?:any\s+|specific\s+|the\s+)?(?:information|details?|data|answer|"
    r"mention|discussion|guidance)\b"
    r"|\b(?:is|are) silent\b)"
    # I cannot answer; cannot be determined - but only from the excerpts
    rf"|{_CANNOT}(?:answer|be answered|provide an answer)\b"
    rf"|{_CANNOT}(?:determine|be determined)\b[^.]{{0,40}}?"
    rf"\b(?:from|based on|using|in)\s+{_IN_MATERIAL}"
    # no information in the excerpts / not enough information to answer
    r"|\b(?:no|insufficient|not enough) (?:relevant |specific )?information\b[^.]{0,40}?"
    rf"(?:\b(?:in|from|within)\s+{_IN_MATERIAL}|\bto answer\b)"
    rf"|\bnot (?:addressed|covered|discussed|mentioned|described) in\s+{_IN_MATERIAL}"
    r"|\bnothing retrieved\b",
    re.I)
REFUSAL_OPENING_CHARS = 250


def cited_pages(text):
    return sorted({int(p) for p in PAGE_CITATION_RE.findall(text or "")})


MAX_CITED_RANGE = 20


def cited_excerpts(text):
    """Excerpt numbers cited in numbered style, ranges expanded. A "range"
    wider than MAX_CITED_RANGE is no real citation; only its two ends are
    kept, which is enough to show it points past the last excerpt."""
    numbers = set()
    for group in EXCERPT_GROUP_RE.findall(text or ""):
        for part in re.split(r"\s*(?:,|;|\band\b)\s*", group, flags=re.I):
            ends = [int(n) for n in re.findall(r"\d+", part)]
            if len(ends) == 2 and 0 <= ends[1] - ends[0] <= MAX_CITED_RANGE:
                numbers.update(range(ends[0], ends[1] + 1))
            else:
                numbers.update(ends)
    return sorted(numbers)


def is_refusal(text, n_citations=0):
    """Heuristic: the answer declines. A refusal phrase in the opening
    counts; further in, only when nothing is cited - "the excerpts do not
    give a size limit (p. 12)" midway through a real answer is not one."""
    text = text or ""
    if REFUSAL_RE.search(text[:REFUSAL_OPENING_CHARS]):
        return True
    return n_citations == 0 and bool(REFUSAL_RE.search(text))


def _relevance_fn():
    from src.score import is_relevant
    return is_relevant


def deterministic_checks(result, question_record=None, is_relevant=None):
    """Everything that can be checked without asking a model.

    fabricated_*: citations that point at a page or excerpt number the model
    was never shown. Always wrong, whatever the judge thinks.
    retrieval_hit: a relevant chunk (score.is_relevant) was in the context.
    None when the question has no gold labels - out-of-scope questions and
    surgeons' own questions.
    refusal_correct: only for out_of_scope questions, where refusing is the
    right answer. None otherwise.
    refused_despite_relevant: an in-scope question was refused although a
    relevant chunk was in the context. The judge, seeing only excerpts, can
    agree the refusal was fair - the gold labels say a right passage was
    there, so a human should look.
    """
    text = result.get("text") or ""
    hits = result.get("retrieved") or []
    chunks = [hit_chunk(h) for h in hits]

    retrieved_pages = {c.get("page") for c in chunks} if chunks else \
        set(result.get("retrieved_pages") or [])
    retrieved_pages.discard(None)
    k = len(chunks) if chunks else len(result.get("retrieved_ids") or [])

    pages = set(cited_pages(text)) | set(result.get("citations") or [])
    excerpts = set(cited_excerpts(text)) | {
        s["n"] for s in (result.get("cited_sources") or [])
        if isinstance(s, dict) and isinstance(s.get("n"), int)}
    fabricated_pages = sorted(p for p in pages if p not in retrieved_pages)
    fabricated_excerpts = sorted(n for n in excerpts if not 1 <= n <= k)

    q = question_record or {}
    retrieval_hit = relevant_ranks = None
    if chunks and (q.get("gold_pages") or q.get("gold_anchors")):
        is_relevant = is_relevant or _relevance_fn()
        relevant_ranks = [rank for rank, c in enumerate(chunks, start=1)
                          if is_relevant(c, q)]
        retrieval_hit = bool(relevant_ranks)
    elif q.get("gold_pages") or q.get("gold_anchors"):
        retrieval_hit, relevant_ranks = False, []

    refused = is_refusal(text, n_citations=len(pages) + len(excerpts))
    expected_refusal = (q.get("type") == "out_of_scope") if q else None

    return {
        "cited_pages": sorted(pages),
        "cited_excerpts": sorted(excerpts),
        "fabricated_pages": fabricated_pages,
        "fabricated_excerpts": fabricated_excerpts,
        "fabricated": bool(fabricated_pages or fabricated_excerpts),
        "n_retrieved": k,
        "retrieval_hit": retrieval_hit,
        "relevant_ranks": relevant_ranks,
        "refused": refused,
        "expected_refusal": expected_refusal,
        "refusal_correct": refused if expected_refusal else None,
        "refused_despite_relevant": bool(refused and retrieval_hit and not expected_refusal),
        "answer_chars": len(text),
        "answer_words": len(text.split()),
    }


# -------------------------------------------------------------- the judge
_client = None
_json_mode = True   # switched off for the session if the API rejects it


def _status(e):
    return getattr(e, "status_code", None) or getattr(
        getattr(e, "response", None), "status_code", None)


def is_retryable(e):
    """Rate limited (429) or over capacity (503): worth waiting for."""
    return _status(e) in (429, 503) or type(e).__name__ == "RateLimitError"


def _retry_after(e):
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def with_backoff(fn, tries=5, base_delay=2.0, max_delay=60.0, sleep=None):
    """fn(), retried on rate limits with exponential backoff (2, 4, 8, 16 s,
    or the server's retry-after if longer). Any other error is raised at
    once - a bad key does not get better by waiting."""
    sleep = sleep or time.sleep
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if not is_retryable(e) or attempt == tries - 1:
                raise
            delay = max(base_delay * 2 ** attempt, _retry_after(e) or 0)
            sleep(min(delay, max_delay))


def _get_client():
    """Created on first use, like generate.groq_llm, so importing this module
    needs no key. Its own client with the SDK's retries off, so with_backoff
    is the only retry policy and the two do not multiply."""
    global _client
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set - add it to .env")
        from groq import Groq
        _client = Groq(api_key=key, max_retries=0)
    return _client


def groq_judge_llm(prompt):
    """Default judge: one Groq chat completion with config.JUDGE_MODEL at
    temperature 0, in JSON mode.

    JSON mode is a convenience, not a dependency: if the API rejects it (the
    model does not support it, or its own JSON validation fails) the call is
    repeated without it and parse_verdict() digs the JSON out of plain text.
    """
    client = _get_client()

    def call():
        global _json_mode
        kwargs = {"model": config.JUDGE_MODEL, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]}
        if _json_mode:
            try:
                return client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs)
            except Exception as e:
                if _status(e) != 400:
                    raise
                if "response_format" in str(e):
                    _json_mode = False
        return client.chat.completions.create(**kwargs)

    return with_backoff(call).choices[0].message.content


def judge_model_name(llm=None):
    """What to log as the judge. A passed-in llm may say via .model."""
    return config.JUDGE_MODEL if llm is None else getattr(llm, "model", "custom")


def judge_answer(result, question_record=None, llm=None):
    """Judge one generate.answer() result. Never raises.

    Returns {"verdict", "checks", "error", "attempts", "raw_reply",
    "judge_model", "judge_prompt_version"}. verdict is None when the judge
    failed; error says why, and raw_reply keeps its last reply for a look.
    The deterministic checks are filled in either way.

    question_record (a questions.json entry) is used only by the checks.
    The judge is not told the question type or gold pages, so an eval
    question and a surgeon's own question are judged the same way.

    llm: function prompt -> text. None means groq_judge_llm.
    """
    out = {"verdict": None, "checks": {}, "error": None, "attempts": 0,
           "raw_reply": None, "judge_model": judge_model_name(llm),
           "judge_prompt_version": JUDGE_PROMPT_VERSION}
    try:
        out["checks"] = deterministic_checks(result, question_record)
        prompt = build_judge_prompt(result, question_record)
    except Exception as e:
        out["error"] = f"could not prepare the judge: {type(e).__name__}: {e}"
        return out

    call = llm or groq_judge_llm
    reply = None
    for attempt in range(2):
        out["attempts"] = attempt + 1
        try:
            reply = call(prompt)
        except Exception as e:
            # Rate limits were already retried inside the call.
            out["error"] = f"judge call failed: {type(e).__name__}: {e}"
            break
        try:
            out["verdict"] = parse_verdict(reply)
            out["error"] = None
            break
        except VerdictError as e:
            out["error"] = f"unusable judge reply: {e}"
            prompt = prompt + RETRY_REMINDER.format(error=e)

    if out["verdict"] is None and reply is not None:
        out["raw_reply"] = str(reply)[:MAX_RAW_REPLY_CHARS]
    return out


# ---------------------------------------------------------------- eval run
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(timestamp):
    """"2026-10-13T21:15:00+00:00" -> "20261013211500". Every row of one run
    shares it, so a run can be pulled out of judge_runs.jsonl."""
    return re.sub(r"\D", "", timestamp)[:14]


def _sources_of(result):
    if result.get("retrieved_sources") is not None:
        return list(result["retrieved_sources"])
    return [chunk_source(hit_chunk(h)) for h in result.get("retrieved") or []]


def make_row(result, judged, question_record=None, run_id=None, timestamp=None,
             note="", mode="eval", error=None):
    """One judge_runs.jsonl line. Chunk text is left out - retrieved_ids
    point into chunks.json, which keeps the log small."""
    q = question_record or {}
    result = result or {}
    judged = judged or {}
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "note": note,
        "mode": mode,
        "id": q.get("id"),
        "type": q.get("type"),
        "question": result.get("question") or q.get("question"),
        "answer": result.get("text"),
        "citations": result.get("citations") or [],
        "cited_sources": result.get("cited_sources") or [],
        "retrieved_ids": result.get("retrieved_ids") or [],
        "retrieved_pages": result.get("retrieved_pages") or [],
        "retrieved_sources": _sources_of(result),
        "checks": judged.get("checks") or {},
        "verdict": judged.get("verdict"),
        "error": error or judged.get("error"),
        "judge_attempts": judged.get("attempts", 0),
        "judge_raw_reply": judged.get("raw_reply"),
        "answer_config": result.get("config") or {},
        "judge_model": judged.get("judge_model") or config.JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }


def run_eval(questions, answer_fn=None, judge_llm=None, ids=None, run_id=None,
             timestamp=None, note="", progress=None):
    """Answer and judge each question; one row per question.

    answer_fn: question text -> generate.answer() result. None means
    generate.answer, imported here so this module loads without it.
    ids: only these question ids. run_id: derived from timestamp when not
    given (the CLI stamps both). progress: called with each row as it is
    finished, so a long run shows it is alive.

    A question that fails - in retrieval, in the answer, in the judge - gets
    an error row and the run moves on. The answer call gets the same
    rate-limit backoff as the judge: a full run sends both models' calls to
    one Groq account, and the answers hit its tokens-per-minute cap too.
    """
    timestamp = timestamp or now_iso()
    run_id = run_id or make_run_id(timestamp)
    if answer_fn is None:
        from src.generate import answer as answer_fn
    wanted = None if ids is None else set(ids)

    rows = []
    for q in questions:
        if wanted is not None and q.get("id") not in wanted:
            continue
        try:
            result = with_backoff(lambda: answer_fn(q["question"]))
        except Exception as e:
            row = make_row({}, {"judge_model": judge_model_name(judge_llm)}, q,
                           run_id, timestamp, note,
                           error=f"answer failed: {type(e).__name__}: {e}")
        else:
            judged = judge_answer(result, q, llm=judge_llm)
            row = make_row(result, judged, q, run_id, timestamp, note)
        rows.append(row)
        if progress:
            progress(row)
    return rows


def append_rows(rows, path=None):
    """Append, never overwrite. Like results.jsonl, this is an experiment log."""
    path = path or config.JUDGE_RUNS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------- summary
def row_key(row):
    """Session record id for --sessions rows (two records can share an eval
    question), question id for eval rows."""
    return row.get("record_id") if row.get("record_id") is not None else row.get("id")


def _sorted_keys(keys):
    return sorted(keys, key=lambda k: (isinstance(k, str), k))


def flag_reasons(row):
    """Why a row needs a human look: unsafe grade, fabricated citation, an
    out-of-scope question answered instead of refused, or an in-scope one
    refused although a relevant chunk was in the context."""
    verdict = row.get("verdict") or {}
    checks = row.get("checks") or {}
    reasons = []
    if verdict.get("grade") in UNSAFE:
        reasons.append(verdict["grade"])
    if checks.get("fabricated"):
        reasons.append("fabricated_citation")
    if checks.get("refusal_correct") is False:
        reasons.append("failed_refusal")
    if checks.get("refused_despite_relevant"):
        reasons.append("refused_despite_relevant")
    return reasons


def summarize(rows):
    """Counts for one run. Deterministic checks count even when the judge
    failed on a row; grades count only where there is a verdict.

    triage splits the not-fully-correct answers the way the app's Summary
    does: no relevant chunk retrieved -> search is at fault; one was there
    -> the prompt or model is. A refusal with a relevant chunk in the context
    counts as falling short even when the judge accepted it.
    """
    grades = dict.fromkeys(GRADE_CODES, 0)
    citation_checks = dict.fromkeys(CITATION_CODES, 0)
    error_ids, flags, refusal_mismatch = [], {}, []
    oos_n = oos_refused = fabricated = labeled = hits = judge_refused = 0
    triage = {"retrieval_fault": 0, "generation_fault": 0, "unlabeled": 0}

    for row in rows:
        key = row_key(row)
        verdict = row.get("verdict")
        checks = row.get("checks") or {}
        if row.get("error") or not verdict:
            error_ids.append(key)

        if checks.get("expected_refusal"):
            oos_n += 1
            oos_refused += bool(checks.get("refused"))
        fabricated += bool(checks.get("fabricated"))
        if checks.get("retrieval_hit") is not None:
            labeled += 1
            hits += checks["retrieval_hit"]

        if verdict:
            grades[verdict["grade"]] += 1
            citation_checks[verdict["citation_check"]] += 1
            judge_refused += verdict["refused"]
            if "refused" in checks and verdict["refused"] != checks["refused"]:
                refusal_mismatch.append(key)
            short = (verdict["grade"] != "correct_complete"
                     or checks.get("refused_despite_relevant"))
            if short and not checks.get("expected_refusal"):
                hit = checks.get("retrieval_hit")
                triage["unlabeled" if hit is None else
                       "generation_fault" if hit else "retrieval_fault"] += 1

        reasons = flag_reasons(row)
        if reasons:
            flags[key] = reasons

    return {
        "n": len(rows),
        "judged": sum(grades.values()),
        "errors": len(error_ids),
        "error_ids": error_ids,
        "grades": grades,
        "citation_checks": citation_checks,
        "out_of_scope": {"n": oos_n, "refused": oos_refused,
                         "accuracy": oos_refused / oos_n if oos_n else None},
        "fabricated_citations": fabricated,
        "retrieval": {"labeled": labeled, "hits": hits,
                      "hit_rate": hits / labeled if labeled else None},
        "judge_refused": judge_refused,
        "refusal_detector_disagrees": refusal_mismatch,
        "triage": triage,
        "flagged": _sorted_keys(flags),
        "flag_reasons": flags,
        "judge_models": sorted({r.get("judge_model") for r in rows if r.get("judge_model")}),
        "judge_prompt_versions": sorted({r.get("judge_prompt_version") for r in rows
                                         if r.get("judge_prompt_version")}),
    }


def _label(key):
    """Q3 for an eval question, the record id for a session record."""
    return f"Q{key}" if isinstance(key, int) else str(key)


def _labels(keys):
    return ", ".join(_label(k) for k in keys)


MAX_ERROR_DISPLAY = 160


def _ratio(num, den):
    return f"{num}/{den} ({num / den:.2f})" if den else "n/a"


def format_row(row):
    """One line per answer for the CLI: id, verdict, checks, flags."""
    label = _label(row_key(row))
    if row.get("record_id") is not None and row.get("id") is not None:
        label += f" (Q{row['id']})"
    verdict = row.get("verdict")
    checks = row.get("checks") or {}
    if verdict:
        judged = f"{verdict['grade']:<19} {verdict['citation_check']:<16}"
    else:
        judged = f"{'ERROR':<19} {'':<16}"
    hit = {True: "hit ", False: "MISS", None: "-   "}[checks.get("retrieval_hit")]
    extra = flag_reasons(row)
    if row.get("error"):
        extra.append(row["error"][:MAX_ERROR_DISPLAY])
    return f"  {label:<6} {judged} {hit} {'  '.join(extra)}".rstrip()


def format_summary(s):
    lines = [f"judged {s['judged']} of {s['n']} answers ({s['errors']} errors)"
             f" | judge {', '.join(s['judge_models']) or '-'}"
             f" | prompt {', '.join(s['judge_prompt_versions']) or '-'}", "", "  grade"]
    lines += [f"    {code:<20} {n}" for code, n in s["grades"].items()]
    lines.append("  citations")
    lines += [f"    {code:<20} {n}" for code, n in s["citation_checks"].items()]
    r, oos, t = s["retrieval"], s["out_of_scope"], s["triage"]
    lines += [
        "",
        f"  relevant chunk in context:  {_ratio(r['hits'], r['labeled'])}",
        f"  out-of-scope refused:       {_ratio(oos['refused'], oos['n'])}",
        f"  fabricated citations:       {s['fabricated_citations']}",
        f"  not fully correct:          {t['retrieval_fault']} search never found a "
        f"relevant chunk, {t['generation_fault']} had one and still fell short, "
        f"{t['unlabeled']} unlabeled",
    ]
    if s["refusal_detector_disagrees"]:
        lines.append(f"  judge and refusal detector disagree on: "
                     f"{_labels(s['refusal_detector_disagrees'])}")
    if s["error_ids"]:
        lines.append(f"  failed (no verdict): {_labels(s['error_ids'])}")
    if s["flagged"]:
        lines.append("\n  flagged for a human look:")
        lines += [f"    {_label(k)}: {', '.join(s['flag_reasons'][k])}" for k in s["flagged"]]
    else:
        lines.append("\n  nothing flagged")
    return "\n".join(lines)


# --------------------------------------------------- calibration vs surgeon
def _label_code(value, labels):
    """App label or short code -> short code; None if unrecognized. A label
    matches on its part before " - " too, so a later rewording of the
    explanation after the dash does not orphan old grades."""
    if not isinstance(value, str) or not value.strip():
        return None
    if value in labels:
        return value
    head = value.split(" - ")[0].strip()
    for code, label in labels.items():
        if value == label or label.split(" - ")[0] == head:
            return code
    return None


def grade_code(value):
    return _label_code(value, GRADE_LABELS)


def citation_code(value):
    return _label_code(value, CITATION_LABELS)


def _verdict_of(item):
    """agreement() takes run rows, judge_answer() outputs or bare verdicts."""
    if not isinstance(item, dict):
        return None
    return item["verdict"] if "verdict" in item else item


def _pair(judged, doctor_records):
    """Rows that carry record_id are matched to records by id; anything else
    must be a list aligned with doctor_records."""
    if judged and all(isinstance(j, dict) and j.get("record_id") is not None
                      for j in judged):
        by_id = {d.get("id"): d for d in doctor_records}
        pairs = [(j, by_id[j["record_id"]]) for j in judged if j["record_id"] in by_id]
        return pairs, len(judged) - len(pairs)
    if len(judged) != len(doctor_records):
        raise ValueError(f"{len(judged)} verdicts for {len(doctor_records)} records - "
                         "pass aligned lists, or rows with record_id")
    return list(zip(judged, doctor_records)), 0


def agreement(judged, doctor_records):
    """How well the judge agrees with the surgeon. Pure - no files, no LLM.

    exact_match: same grade out of four. binary_match: both say "could act
    on it" (correct_*) or both say "could not" (misleading/harmful) - the
    distinction that matters most.

    judge_fine_doctor_bad is the list to read first: answers the surgeon
    called misleading or harmful that the judge passed. Every one is a
    reason not to trust the judge alone yet. judge_bad_doctor_fine is the
    opposite, cheaper failure - the judge being stricter than the surgeon.

    confusion[doctor_grade][judge_grade] = count.
    """
    pairs, unmatched = _pair(judged, doctor_records)
    confusion = {d: dict.fromkeys(GRADE_CODES, 0) for d in GRADE_CODES}
    n = matches = binary = cite_n = cite_matches = skipped = 0
    fine_bad, bad_fine = [], []

    for item, record in pairs:
        verdict = _verdict_of(item) or {}
        judge_grade = grade_code(verdict.get("grade"))
        doctor_grade = grade_code(record.get("grade"))
        if judge_grade is None or doctor_grade is None:
            skipped += 1
            continue
        n += 1
        matches += judge_grade == doctor_grade
        binary += (judge_grade in ACCEPTABLE) == (doctor_grade in ACCEPTABLE)
        confusion[doctor_grade][judge_grade] += 1
        entry = {"record_id": record.get("id"), "question": record.get("question"),
                 "doctor": doctor_grade, "judge": judge_grade,
                 "doctor_comment": record.get("comment"),
                 "judge_rationale": verdict.get("rationale")}
        if judge_grade in ACCEPTABLE and doctor_grade in UNSAFE:
            fine_bad.append(entry)
        elif judge_grade in UNSAFE and doctor_grade in ACCEPTABLE:
            bad_fine.append(entry)

        judge_cite = citation_code(verdict.get("citation_check"))
        doctor_cite = citation_code(record.get("citation_check"))
        if judge_cite and doctor_cite:
            cite_n += 1
            cite_matches += judge_cite == doctor_cite

    return {
        "n": n,
        "exact_matches": matches,
        "exact_match": matches / n if n else None,
        "binary_match": binary / n if n else None,
        "confusion": confusion,
        "judge_fine_doctor_bad": fine_bad,
        "judge_bad_doctor_fine": bad_fine,
        "citation_n": cite_n,
        "citation_exact_match": cite_matches / cite_n if cite_n else None,
        "skipped": skipped + unmatched,
    }


def format_agreement(a):
    if not a["n"]:
        return "no surgeon-graded answers to compare with"
    lines = [f"agreement with the surgeon on {a['n']} answers",
             f"  same grade:            {a['exact_matches']}/{a['n']} ({a['exact_match']:.2f})",
             f"  same act / don't act:  {a['binary_match']:.2f}"]
    if a["citation_n"]:
        lines.append(f"  same citation check:   {a['citation_exact_match']:.2f} "
                     f"(of {a['citation_n']})")
    lines.append("\n  confusion (rows: surgeon, columns: judge)")
    lines.append("    " + " " * 20 + " ".join(f"{c[:10]:>10}" for c in GRADE_CODES))
    for d in GRADE_CODES:
        lines.append(f"    {d:<20}" + " ".join(f"{a['confusion'][d][j]:>10}"
                                            for j in GRADE_CODES))
    if a["judge_fine_doctor_bad"]:
        lines.append("\n  judge passed, surgeon did not:")
        lines += [f"    {e['record_id']}: surgeon {e['doctor']}, judge {e['judge']} - "
                  f"{e['question']}" for e in a["judge_fine_doctor_bad"]]
    return "\n".join(lines)


# ------------------------------------------------------ surgeon's sessions
def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_test_ids(path=None):
    """Same list storage.test_record_ids() reads, read the same way."""
    path = path or TEST_IDS_PATH
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).get("ids", []))


def is_test_record(record, test_ids):
    """storage.is_test, restated so this module never imports storage."""
    return bool(record.get("is_test")) or record.get("id") in test_ids


def graded_real_records(records, test_ids):
    """What the app's Summary counts: graded, and not a test record."""
    return [r for r in records if r.get("grade") and not is_test_record(r, test_ids)]


def rehydrate(record, chunks_by_id):
    """A session record as a generate.answer() result, with the retrieved
    chunks looked up in chunks.json by id. None if any chunk is gone (the
    index was rebuilt since) - judging against different excerpts than the
    model saw would be meaningless."""
    ids = record.get("retrieved_ids") or []
    if not ids or any(i not in chunks_by_id for i in ids):
        return None
    hits = [{"rank": rank, "score": None, "chunk": chunks_by_id[i]}
            for rank, i in enumerate(ids, start=1)]
    return {
        "question": record.get("question"),
        "text": record.get("answer") or "",
        "citations": record.get("citations") or [],
        "cited_sources": record.get("cited_sources") or [],
        "retrieved_ids": ids,
        "retrieved_pages": record.get("retrieved_pages") or [],
        "retrieved_sources": record.get("retrieved_sources"),
        "retrieved": hits,
        "config": record.get("config") or {},
    }


def question_lookup(questions):
    """{normalized question text: questions.json entry}, so a surgeon asking
    an eval question verbatim gets its retrieval check (as app.py does)."""
    return {q["question"].strip().lower(): q for q in questions}


def judge_sessions(records, chunks_by_id, questions=(), judge_llm=None, run_id=None,
                   timestamp=None, note="", progress=None):
    """Judge surgeon-graded records. Returns (rows, skipped_record_ids)."""
    timestamp = timestamp or now_iso()
    run_id = run_id or make_run_id(timestamp)
    lookup = question_lookup(questions)
    rows, skipped = [], []
    for record in records:
        result = rehydrate(record, chunks_by_id)
        if result is None:
            skipped.append(record.get("id"))
            continue
        q = lookup.get((record.get("question") or "").strip().lower())
        judged = judge_answer(result, q, llm=judge_llm)
        row = make_row(result, judged, q, run_id, timestamp, note, mode="sessions")
        row.update({"record_id": record.get("id"),
                    "doctor_grade": grade_code(record.get("grade")),
                    "doctor_citation_check": citation_code(record.get("citation_check"))})
        rows.append(row)
        if progress:
            progress(row)
    return rows, skipped


# --------------------------------------------------------------------- CLI
def parse_ids(text):
    """"1,3" -> [1, 3]."""
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--ids takes comma-separated numbers, got {text!r}")


def _main_eval(args, run_id, timestamp):
    questions = json.loads(config.QUESTIONS_PATH.read_text())
    if args.ids:
        missing = sorted(set(args.ids) - {q["id"] for q in questions})
        if missing:
            print(f"no such question id: {missing}")
    print(f"run {run_id}: answering and judging "
          f"{len(args.ids) if args.ids else len(questions)} questions "
          f"(2 LLM calls each)\n")
    rows = run_eval(questions, ids=args.ids, run_id=run_id, timestamp=timestamp,
                    note=args.note, progress=lambda row: print(format_row(row)))
    if not rows:
        print("nothing to judge")
        return
    print("\n" + format_summary(summarize(rows)))
    print(f"\nlogged -> {append_rows(rows)}")


def _main_sessions(args, run_id, timestamp):
    records = load_jsonl(SESSIONS_PATH)
    graded = graded_real_records(records, load_test_ids())
    print(f"{len(records)} session records, {len(graded)} graded by the surgeon "
          "and not test records")
    if not graded:
        print("nothing to compare yet - grade some answers in the app first "
              "(test records are left out, see data/eval/test_records.json)")
        return
    chunks_by_id = {c["id"]: c for c in json.loads(config.CHUNKS_PATH.read_text())}
    questions = (json.loads(config.QUESTIONS_PATH.read_text())
                 if config.QUESTIONS_PATH.exists() else [])
    rows, skipped = judge_sessions(graded, chunks_by_id, questions, run_id=run_id,
                                   timestamp=timestamp, note=args.note,
                                   progress=lambda row: print(format_row(row)))
    if skipped:
        print(f"skipped {len(skipped)} records whose chunks are no longer in the index")
    if not rows:
        return
    print("\n" + format_summary(summarize(rows)))
    print("\n" + format_agreement(agreement(rows, graded)))
    print(f"\nlogged -> {append_rows(rows)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ids", type=parse_ids, default=None,
                        help="comma-separated question ids, e.g. 1,3")
    parser.add_argument("--note", default="", help="what you changed this run")
    parser.add_argument("--sessions", action="store_true",
                        help="judge the surgeon-graded answers in sessions.jsonl "
                             "and report agreement with the surgeon")
    args = parser.parse_args(argv)

    timestamp = now_iso()
    run_id = make_run_id(timestamp)
    if args.sessions:
        _main_sessions(args, run_id, timestamp)
    else:
        _main_eval(args, run_id, timestamp)


if __name__ == "__main__":
    main(sys.argv[1:])

"""The overnight report: one markdown page that says whether the system is
still healthy, and what needs a human tomorrow.

  python3 -m src.report               # everything, including the LLM judge
  python3 -m src.report --skip-llm    # no Groq calls: index, search, grades
  python3 -m src.report --log         # also log tonight's search scores
  python3 -m src.report --out PATH    # write somewhere else

Writes data/eval/reports/<YYYY-MM-DD>.md (UTC date; a second run on the same
day gets -2, -3, ...) and prints its path. A scheduler runs it every night -
see docs/report.md.

Why a dated markdown file and not a dashboard: the readers are a student
and a surgeon, once a day. A file needs no server, stays in a folder as a
history, and reads fine in any editor. Next to each report sits a .json with
the same numbers, so tomorrow's report can say what changed since tonight
without parsing markdown.

Why every section is wrapped: the night the Groq key expires should still
produce the index check and the surgeon's grades. A section that raises
shows its error in its own place, and the rest of the report carries on.

What it never does: write sessions.jsonl, import src/storage.py (that pulls
from the Hub when imported), or append to results.jsonl without --log -
results.jsonl is the experiment log, and a nightly line in it should be a
choice. The judge's rows do go to judge_runs.jsonl, because that file
exists to collect them.
"""

import argparse
import json
import re
import subprocess
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src import config

# What score.py measures. recall@10 is logged but not shown: it is maxed out
# at 1.0 and no longer tells experiments apart (docs/experiments.md).
METRICS = ("recall@1", "recall@5", "recall@10", "precision@5", "mrr")
SHOWN = ("recall@1", "recall@5", "precision@5", "mrr")
METRIC_LABELS = {
    "recall@1": "right passage ranked first (recall@1)",
    "recall@5": "right passage in the top 5 (recall@5)",
    "precision@5": "share of the top 5 that is relevant (precision@5)",
    "mrr": "how high the first right passage ranks (MRR)",
}
# A drop at least this big in any metric is flagged. With 13 questions one
# question moves recall by 0.077, so a lost hit always crosses it, while
# precision@5 and MRR can drift a little without raising an alarm.
REGRESSION_DROP = 0.05

LOG_NOTE = "nightly report"
HARMFUL = "Potentially harmful"
MISLEADING = "Misleading"
# Answers quote a copyrighted textbook. The report shows a sentence or so,
# enough to recognise the answer, never the passage itself.
EXCERPT_CHARS = 160
COMMENT_CHARS = 300
MAX_LISTED = 15


@dataclass
class Section:
    """One part of the report. facts are the machine-readable findings that
    the action items and tomorrow's comparison read; body is what people read."""
    key: str
    title: str
    body: str
    facts: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def now_utc():
    return datetime.now(timezone.utc)


def iso(t):
    return t.isoformat(timespec="seconds")


def parse_time(value):
    """ISO timestamp -> timezone-aware datetime, or None if unreadable.
    A timestamp without a zone is taken as UTC, which is what the app writes."""
    if not value:
        return None
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def when(t):
    """Human-readable UTC time: 2026-10-13 03:00 UTC."""
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if t else "unknown"


def clip(text, limit):
    """One line, at most limit characters. Free text in a table cell must not
    break the layout, and long answer text must not end up in the report."""
    text = re.sub(r"\s+", " ", "" if text is None else str(text)).strip()
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "-"
    return str(value)


def table(headers, rows):
    """A markdown table. Pipes and newlines in cells are neutralised, since
    questions and grader comments are free text."""
    def cell(value):
        return clip(fmt(value), 400).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    """(records, n_unreadable). A bad line is counted, not fatal - one
    half-written line must not hide every other record."""
    path = Path(path)
    if not path.exists():
        return [], 0
    records, bad = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            bad += 1
    return records, bad


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def load_questions():
    return json.loads(config.QUESTIONS_PATH.read_text())


def normalize_question(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def config_changes(old, new):
    """Settings present in both configs whose values differ, as "key a -> b".
    Old results.jsonl lines lack the newer keys; a missing key is not a change."""
    old, new = old or {}, new or {}
    return [f"{k} {fmt(old[k])} -> {fmt(new[k])}"
            for k in new if k in old and old[k] != new[k]]


# ------------------------------------------------------- file locations
def report_path(reports_dir, now):
    """reports_dir/<YYYY-MM-DD>.md, or -2, -3, ... if that day has one already.
    Never overwrites: a second run the same night is a second report."""
    reports_dir = Path(reports_dir)
    day = now.strftime("%Y-%m-%d")
    n = 1
    while True:
        path = reports_dir / (f"{day}.md" if n == 1 else f"{day}-{n}.md")
        if not path.exists() and not path.with_suffix(".json").exists():
            return path
        n += 1


def previous_report(reports_dir, before):
    """The metrics .json of the latest report generated before `before`, or None.

    Ordered by the generated_at stored inside, not by file name - "-2"
    sorts before "." and "-10" before "-2", so names would lie.
    """
    best = None
    for path in Path(reports_dir).glob("*.json"):
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        t = parse_time(data.get("generated_at"))
        if t and t < before and (best is None or t > best[0]):
            best = (t, data)
    return best[1] if best else None


def sessions_path():
    # Paths are computed at call time so tests can point config.EVAL_DIR
    # at a temporary folder.
    return config.EVAL_DIR / "sessions.jsonl"


def listed_test_ids_path():
    return config.EVAL_DIR / "test_records.json"


# ---------------------------------------------------------- 1. the run
def git_info(root=None):
    """{"commit", "branch", "dirty"} for the code that ran, or None without git."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root or config.ROOT, check=True,
                               capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return {"commit": git("rev-parse", "--short", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return None


def run_settings():
    """config.run_config() plus the version of the prompt the configured
    citation style actually uses, when generate imports."""
    settings = config.run_config()
    try:
        from src.generate import prompt_version
        settings["prompt_version"] = prompt_version(config.CITATION_STYLE)
    except Exception:
        pass
    return settings


def section_header(ctx):
    git = git_info()
    if git:
        code = f"`{git['commit']}` on `{git['branch']}`"
        code += " (with uncommitted changes)" if git["dirty"] else " (clean)"
    else:
        code = "unknown (git not available)"
    settings = run_settings()
    index = settings.get("index_id")
    lines = [
        f"- **Generated:** {when(ctx['now'])}",
        f"- **Code:** {code}",
        f"- **Index:** " + (f"`{index}`" if index else
                            "no index ID yet (no manifest.json - run `python3 -m src.ingest`)"),
        f"- **LLM checks:** " + ("skipped (`--skip-llm`)" if ctx["skip_llm"]
                                 else f"answers by `{config.LLM_MODEL}`, judged by `{config.JUDGE_MODEL}`"),
    ]
    previous = ctx.get("previous")
    if previous:
        lines.append(f"- **Previous report:** {previous.get('report', '?')} "
                     f"({when(parse_time(previous.get('generated_at')))})")
    else:
        lines.append("- **Previous report:** none - this is the first")
    lines += ["", "Settings this run used:", "",
              table(["setting", "value"], [(k, v) for k, v in settings.items()])]
    return Section("header", "This run", "\n".join(lines),
                   {"git": git, "config": settings, "index_id": index})


# ------------------------------------------------- 2. index and sources
REGISTRY_STATUS = {
    "ok": "enabled, file present",
    "missing": "ENABLED BUT FILE MISSING",
    "ready": "file downloaded, not enabled yet",
    "off": "disabled, not downloaded",
}


def registry_status(source):
    """One of REGISTRY_STATUS's keys for a sources.Source."""
    if source.enabled:
        return "ok" if source.available else "missing"
    return "ready" if source.available else "off"


def source_of(chunk):
    """Source id of a chunk. Records from before the multi-source index
    have only "book"; anything with neither is the textbook."""
    return chunk.get("source") or chunk.get("book") or config.DEFAULT_SOURCE


def index_counts(chunks):
    """{source_id: {"n": chunks, "domains": sorted domains}}, in index order."""
    counts = {}
    for c in chunks:
        entry = counts.setdefault(source_of(c), {"n": 0, "domains": set()})
        entry["n"] += 1
        entry["domains"].add(c.get("domain") or "clinical")
    return {sid: {"n": e["n"], "domains": sorted(e["domains"])} for sid, e in counts.items()}


def alignment_problems(chunk_ids, n_vectors, vector_keys=None, model=None):
    """Why search results could be quietly wrong, as sentences. [] = aligned.

    Row i of vectors.npy must describe chunk i. A count check catches a
    forgotten embed after re-ingesting; vector_keys.json's ids catch the
    subtler case of the same count but different chunks.
    """
    model = model or config.EMBEDDING_MODEL
    problems = []
    if n_vectors != len(chunk_ids):
        problems.append(f"{len(chunk_ids)} chunks but {n_vectors} vectors - "
                        "run `python3 -m src.embed`")
    if vector_keys is not None:
        ids = list(vector_keys.get("ids") or [])
        if ids != list(chunk_ids):
            first = next((i for i, (a, b) in enumerate(zip(ids, chunk_ids)) if a != b),
                         min(len(ids), len(chunk_ids)))
            problems.append(f"vector_keys.json lists different chunks than chunks.json "
                            f"(first difference at row {first}) - run `python3 -m src.embed`")
        built_with = vector_keys.get("embedding_model")
        if built_with and built_with != model:
            problems.append(f"vectors were built with `{built_with}` but config says "
                            f"`{model}` - run `python3 -m src.embed`")
    return problems


def count_vectors(path):
    import numpy as np
    return int(np.load(path, mmap_mode="r").shape[0])


def section_index(ctx):
    chunks = json.loads(config.CHUNKS_PATH.read_text())
    counts = index_counts(chunks)
    manifest = config.index_manifest()
    labels = config.source_labels()
    lines, facts = [], {}

    if manifest:
        lines.append(f"Index `{manifest.get('index_id', '?')}`, built "
                     f"{when(parse_time(manifest.get('built_at')))}: **{len(chunks)} chunks**.")
        listed = {s["id"]: s.get("n_chunks") for s in manifest.get("sources", [])}
        drift = [f"{sid} (manifest {listed.get(sid)}, chunks.json {counts.get(sid, {}).get('n', 0)})"
                 for sid in sorted(set(listed) | set(counts))
                 if listed.get(sid) != counts.get(sid, {}).get("n")]
        if drift:
            facts["manifest_drift"] = drift
            lines.append("**Problem:** manifest.json and chunks.json disagree on chunk counts: "
                         + "; ".join(drift) + ". Re-run `python3 -m src.ingest`.")
    else:
        lines.append(f"**{len(chunks)} chunks.** _No manifest.json yet: this index was built "
                     "before multi-source ingest. `python3 -m src.ingest` writes one._")

    lines += ["", table(["source", "label", "domain", "chunks"],
                        [(sid, labels.get(sid, sid), e["domains"], e["n"])
                         for sid, e in counts.items()])]

    # Alignment: the one failure that makes every answer quietly wrong.
    vector_keys = read_json(config.VECTOR_KEYS_PATH) if config.VECTOR_KEYS_PATH.exists() else None
    n_vectors = count_vectors(config.VECTORS_PATH)
    problems = alignment_problems([c["id"] for c in chunks], n_vectors, vector_keys)
    facts["alignment"] = problems
    lines.append("")
    if problems:
        lines += ["**Problem: the search files do not line up.**"] + [f"- {p}" for p in problems]
    else:
        detail = ("vector_keys.json lists the same chunks in the same order"
                  if vector_keys is not None else "no vector_keys.json yet, so only counts were checked")
        lines.append(f"Search files line up: {len(chunks)} chunks, {n_vectors} vectors ({detail}).")

    lines += ["", "**Registry (sources.yaml)**", ""]
    lines += registry_lines(set(counts), facts)
    facts["counts"] = {sid: e["n"] for sid, e in counts.items()}
    return Section("index", "Index and sources", "\n".join(lines), facts)


def registry_lines(indexed, facts):
    """Registry table + the facts the action items need. Separate from the
    index part so a broken sources.yaml still leaves the chunk counts above."""
    try:
        from src.sources import load_registry
        sources = load_registry()
    except Exception as e:
        facts["registry_error"] = f"{type(e).__name__}: {e}"
        return [f"**Problem:** sources.yaml could not be read: `{clip(e, 300)}`"]

    by_status = {}
    for s in sources:
        by_status.setdefault(registry_status(s), []).append(s)
    facts["missing_enabled"] = [s.id for s in by_status.get("missing", [])]
    facts["ready_to_enable"] = [s.id for s in by_status.get("ready", [])]
    # Cross-check with the index: the registry says what SHOULD be there.
    facts["not_indexed"] = [s.id for s in by_status.get("ok", []) if s.id not in indexed]
    enabled = {s.id for s in sources if s.enabled}
    facts["stale_in_index"] = sorted(sid for sid in indexed if sid not in enabled)

    shown = [s for s in sources if registry_status(s) != "off"]
    rows = []
    for s in shown:
        status = REGISTRY_STATUS[registry_status(s)]
        if s.id in facts["not_indexed"]:
            status += " - not in the index yet"
        rows.append((s.id, s.label, s.domain, status))
    lines = [table(["source", "label", "domain", "status"], rows)] if rows else []
    off = by_status.get("off", [])
    if off:
        lines += ["", f"{len(off)} more listed but not downloaded (disabled): "
                  + ", ".join(s.id for s in off) + "."]
    if facts["stale_in_index"]:
        lines += ["", "**In the index but not enabled in sources.yaml:** "
                  + ", ".join(facts["stale_in_index"])
                  + ". Re-run `python3 -m src.ingest && python3 -m src.embed` to drop them."]
    for s in by_status.get("missing", []):
        hint = f" Get it from {s.url}." if s.url else ""
        where = s.path.relative_to(config.ROOT) if s.path.is_relative_to(config.ROOT) else s.path
        lines += ["", f"**Problem:** `{s.id}` is enabled but its file is missing "
                      f"(`{where}`).{hint}"]
    return lines


# ------------------------------------------------- 3. retrieval quality
def labeled_questions(questions):
    """Questions search can be scored on: labeled, and not out of scope
    (those test refusal - no right page exists)."""
    return [q for q in questions if q.get("type") != "out_of_scope"
            and (q.get("gold_pages") or q.get("gold_anchors"))]


def aggregate(rows):
    """score.py's summary numbers, from per-question rows."""
    out = {"n_questions": len(rows)}
    for key in METRICS:
        field_ = "rr" if key == "mrr" else key
        values = [r[field_] for r in rows if r.get(field_) is not None]
        out[key] = sum(values) / len(values) if values else None
    return out


def score_questions(questions):
    """(summary, per_question rows) from src/score.py.

    score.score_all() is the shared entry point once it exists; until then
    score_one() per question gives the same rows. Tests replace this whole
    function, so no model is loaded.
    """
    from src import score
    if hasattr(score, "score_all"):
        return score.score_all(questions)
    rows = [score.score_one(q) for q in questions]
    return aggregate(rows), rows


def per_question(rows):
    """{question id (str): the parts tomorrow needs to spot a lost hit}."""
    return {str(r["id"]): {"recall@5": r.get("recall@5"), "rr": r.get("rr"),
                           "first_relevant_rank": r.get("first_relevant_rank")}
            for r in rows}


def metric_regressions(current, previous, drop=REGRESSION_DROP):
    """Metrics that fell by at least `drop`: [(metric, before, after)]."""
    out = []
    for key in SHOWN:
        before, after = (previous or {}).get(key), current.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) \
                and before - after >= drop - 1e-9:
            out.append((key, before, after))
    return out


def hit_changes(current, previous):
    """(lost, gained) question ids: had a right passage in the top 5 last
    time and not now, and the reverse. Questions new to either side are skipped."""
    lost = [q for q, r in current.items()
            if r.get("recall@5") == 0 and (previous.get(q) or {}).get("recall@5") == 1]
    gained = [q for q, r in current.items()
              if r.get("recall@5") == 1 and (previous.get(q) or {}).get("recall@5") == 0]
    return lost, gained


def last_logged_run(path):
    """The newest results.jsonl line that has scores, or None."""
    records, _ = read_jsonl(path)
    scored = [r for r in records if isinstance(r.get("mrr"), (int, float))]
    return scored[-1] if scored else None


def describe_regressions(metric_drops, lost, baseline, settings_changed):
    why = f" since {baseline}"
    caveat = " (settings changed, so this may be expected)" if settings_changed else ""
    lines = [f"{METRIC_LABELS.get(k, k)} fell from {b:.3f} to {a:.3f}{why}{caveat}"
             for k, b, a in metric_drops]
    lines += [f"Q{q} no longer has a right passage in the top 5{why}{caveat}" for q in lost]
    return lines


def section_retrieval(ctx):
    questions = load_questions()
    scorable = labeled_questions(questions)
    if not scorable:
        return Section("retrieval", "Search quality",
                       "_No labeled questions in questions.json - nothing to score._", {})
    summary, rows = score_questions(scorable)
    summary = summary or {}
    metrics = aggregate(rows)
    metrics.update({k: summary[k] for k in ("n_questions",) + METRICS
                    if isinstance(summary.get(k), (int, float))})
    pq = per_question(rows)
    # score_all() reports the config the scores were measured under.
    settings = summary.get("config") or config.run_config()
    text = {str(q["id"]): q["question"] for q in questions}

    previous = ctx.get("previous") or {}
    prev_retrieval = previous.get("retrieval") or {}
    prev_metrics = prev_retrieval.get("metrics")
    logged = last_logged_run(config.RESULTS_PATH)
    # With --log the previous report also wrote the last results.jsonl line;
    # comparing against it twice would report every change twice.
    if logged and previous and logged.get("timestamp") == previous.get("generated_at"):
        logged = None

    regressions, lost, gained = [], [], []
    if prev_metrics:
        changed = bool(config_changes(previous.get("config"), settings))
        lost, gained = hit_changes(pq, prev_retrieval.get("per_question") or {})
        regressions += describe_regressions(metric_regressions(metrics, prev_metrics), lost,
                                            "the previous report", changed)
    if logged:
        changed = bool(config_changes(logged.get("config"), settings))
        regressions += describe_regressions(metric_regressions(metrics, logged), [],
                                            "the last logged run", changed)

    n = metrics["n_questions"]
    headers = ["measure", "tonight"]
    if prev_metrics:
        headers.append("previous report")
    if logged:
        headers.append(f"last logged run ({clip(logged.get('note') or 'no note', 40)})")
    rows_ = []
    for key in SHOWN:
        row = [METRIC_LABELS[key], metrics.get(key)]
        if prev_metrics:
            row.append(prev_metrics.get(key))
        if logged:
            row.append(logged.get(key))
        rows_.append(row)

    skipped = len(questions) - len(scorable)
    lines = [f"{n} labeled questions scored ({skipped} not scored: out of scope or "
             "unlabeled). Higher is better for every measure.", "", table(headers, rows_)]
    # Notes go after a blank line: a text line right under a table would
    # be read as one more table row.
    notes = []
    if logged:
        notes.append(f"Last logged run: {when(parse_time(logged.get('timestamp')))}.")
        diff = config_changes(logged.get("config"), settings)
        if diff:
            notes.append("Settings differ from that run: " + "; ".join(diff) + ".")
    if prev_metrics:
        diff = config_changes(previous.get("config"), settings)
        if diff:
            notes.append("Settings differ from the previous report: " + "; ".join(diff) + ".")
    for note in notes:
        lines += ["", f"_{note}_"]

    if regressions:
        lines += ["", "**Got worse:**"] + [f"- {r}" for r in regressions]
    elif prev_metrics or logged:
        lines += ["", "No regressions."]
    if gained:
        lines += ["", "Newly found since the previous report: "
                  + ", ".join(f"Q{q}" for q in gained) + "."]

    misses = [r for r in rows if not r.get("recall@5")]
    if misses:
        lines += ["", "**Questions with no right passage in the top 5:**", "",
                  table(["question", "first right passage at rank", "top 5 pages"],
                        [(f"Q{r['id']}: {clip(text.get(str(r['id']), ''), 90)}",
                          r.get("first_relevant_rank") or "not in top 10",
                          r.get("top_pages") or "-") for r in misses])]
    else:
        lines += ["", "Every question had a right passage in the top 5."]

    if ctx.get("log"):
        lines.append(f"\n_Logged to results.jsonl as \"{LOG_NOTE}\"._")
    # The same line score.py would log (score.log_record), stamped with the
    # report's own time so tomorrow can tell it apart from experiment runs.
    extra = {k: v for k, v in summary.items()
             if k not in ("unlabeled", "out_of_scope", "config", "timestamp", "note")}
    log_entry = {"timestamp": iso(ctx["now"]), "note": LOG_NOTE, **extra, **metrics,
                 "config": settings}
    return Section("retrieval", "Search quality", "\n".join(lines),
                   {"metrics": metrics, "per_question": pq, "regressions": regressions,
                    "misses": [str(r["id"]) for r in misses], "log_entry": log_entry})


# --------------------------------------------------- 4. answer quality
# judge.py's short codes, in plain words. Kept here as well as in judge.py
# so a judge module that fails to import still leaves the report readable.
JUDGE_GRADES = {"correct_complete": "Correct and complete",
                "correct_incomplete": "Correct but incomplete",
                "misleading": "Misleading", "harmful": "Potentially harmful"}
JUDGE_CITATIONS = {"all_supported": "every citation supports its claim",
                   "some_unsupported": "a citation does not support its claim",
                   "uncited_claims": "claims with no citation",
                   "no_citations": "no citations"}
FLAG_REASONS = {"misleading": "graded misleading", "harmful": "graded potentially harmful",
                "fabricated_citation": "cites a page it was never shown",
                "failed_refusal": "answered a question it should have refused",
                "refused_despite_relevant": "refused although a right passage was in the context"}


def import_judge():
    from src import judge
    return judge


def supported_kwargs(fn, **kwargs):
    """The kwargs fn accepts. judge.run_eval() takes note/timestamp today,
    but the contract only promises (questions, answer_fn, judge_llm, ids)."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def ratio(num, den):
    return f"{num} of {den}" if den else "-"


def judge_measures(summary):
    """(label, value) rows for judge.summarize()'s headline numbers."""
    oos = summary.get("out_of_scope") or {}
    hits = summary.get("retrieval") or {}
    return [
        ("answers checked", ratio(summary.get("judged", 0), summary.get("n", 0))),
        ("could not be answered or judged", summary.get("errors", 0)),
        ("out-of-scope questions correctly refused", ratio(oos.get("refused", 0), oos.get("n", 0))),
        ("answers citing a page the model was never shown", summary.get("fabricated_citations", 0)),
        ("a right passage was in the context", ratio(hits.get("hits", 0), hits.get("labeled", 0))),
        ("answers flagged for a human look", len(summary.get("flagged") or [])),
    ]


def judge_body(summary, rows, previous=None):
    """Markdown for one judge run: headline numbers (with the previous
    report's alongside), grade and citation counts, then each flagged answer."""
    lines = []
    measures = judge_measures(summary)
    if previous:
        before = dict(judge_measures(previous))
        lines.append(table(["measure", "tonight", "previous report"],
                           [(k, v, before.get(k)) for k, v in measures]))
    else:
        lines.append(table(["measure", "tonight"], measures))
    grades = summary.get("grades") or {}
    if grades:
        lines += ["", table(["judge's grade", "answers"],
                            [(JUDGE_GRADES.get(k, k), n) for k, n in grades.items()])]
    checks = summary.get("citation_checks") or {}
    if checks:
        lines += ["", table(["citations", "answers"],
                            [(JUDGE_CITATIONS.get(k, k), n) for k, n in checks.items()])]
    errors = [r for r in rows if r.get("error")]
    if errors:
        lines += ["", f"First error: `{clip(errors[0]['error'], 200)}`"]

    flagged = [str(i) for i in summary.get("flagged") or []]
    if flagged:
        reasons = summary.get("flag_reasons") or {}
        lines += ["", "**Flagged answers** (the judge is not a clinician - "
                      "these need the surgeon's eye, not automatic trust):"]
        lines += flagged_lines(rows, flagged, {str(k): v for k, v in reasons.items()})
    else:
        lines += ["", "No answers flagged."]
    return "\n".join(lines)


def row_id(row):
    return str(row.get("id") if row.get("id") is not None else row.get("record_id", "?"))


def flagged_lines(rows, flagged_ids, reasons=None):
    """Question, grade, why, unsupported claims - and only the first
    sentence or so of the answer, since answers quote the textbook."""
    lines, reasons = [], reasons or {}
    wanted = set(flagged_ids)
    for row in rows:
        if row_id(row) not in wanted:
            continue
        verdict = row.get("verdict") or {}
        grade = JUDGE_GRADES.get(verdict.get("grade"), verdict.get("grade") or "not judged")
        why = ", ".join(FLAG_REASONS.get(r, r) for r in reasons.get(row_id(row), []))
        lines.append(f"- **Q{row_id(row)}: {clip(row.get('question'), 120)}** - "
                     f"judge's grade: {grade}" + (f" ({why})" if why else ""))
        if verdict.get("rationale"):
            lines.append(f"  - Judge's reason: {clip(verdict['rationale'], COMMENT_CHARS)}")
        claims = verdict.get("unsupported_claims") or []
        for claim in (claims if isinstance(claims, list) else [claims])[:3]:
            lines.append(f"  - Unsupported: \"{clip(claim, EXCERPT_CHARS)}\"")
        if row.get("answer"):
            lines.append(f"  - Answer began: \"{clip(row['answer'], EXCERPT_CHARS)}\"")
    return lines


def section_answers(ctx):
    if ctx["skip_llm"]:
        return Section("answers", "Answer quality",
                       "_Skipped (`--skip-llm`): no LLM calls were made tonight._",
                       {"skipped": True})
    try:
        judge = import_judge()
    except ImportError as e:
        return Section("answers", "Answer quality",
                       "_The answer checker (src/judge.py) is not available, so answers "
                       f"were not checked tonight: `{clip(e, 200)}`._",
                       {"unavailable": f"{type(e).__name__}: {e}"})

    questions = load_questions()
    rows = judge.run_eval(questions, **supported_kwargs(
        judge.run_eval, note=LOG_NOTE, timestamp=iso(ctx["now"])))
    # Logged before anything else can fail: these rows cost LLM calls.
    append_jsonl(config.JUDGE_RUNS_PATH,
                 [{**row, "note": row.get("note") or LOG_NOTE} for row in rows])
    summary = judge.summarize(rows) or {}

    previous = (ctx.get("previous") or {}).get("answers")
    lines = [f"{len(rows)} questions answered by `{config.LLM_MODEL}` and checked by "
             f"`{config.JUDGE_MODEL}` (rows appended to `{config.JUDGE_RUNS_PATH.name}`).",
             "", judge_body(summary, rows, previous)]
    errors = [r for r in rows if r.get("error")]
    facts = {"summary": summary, "n": len(rows),
             "flagged": [str(i) for i in summary.get("flagged") or []],
             "fabricated": summary.get("fabricated_citations", 0),
             # summarize() also counts rows the judge left without a verdict
             "errors": summary.get("errors", len(errors)),
             "first_error": clip(errors[0]["error"], 150) if errors else None}
    return Section("answers", "Answer quality", "\n".join(lines), facts)


# --------------------------------------------------- 5. surgeon activity
def read_test_ids(path):
    data = read_json(path, {})
    return set(data.get("ids", [])) if isinstance(data, dict) else set()


def is_test(record, test_ids):
    """storage.is_test's rule, copied rather than imported: src.storage
    pulls sessions from the Hub as soon as it is imported."""
    return bool(record.get("is_test")) or record.get("id") in test_ids


def after(value, since):
    """True if timestamp `value` is later than `since`; everything counts
    when there is no earlier report. An unreadable time is not new."""
    if since is None:
        return True
    t = parse_time(value)
    return t is not None and t > since


def short_grade(grade):
    """"Misleading - sourced, but ..." -> "Misleading"."""
    return str(grade).split(" - ")[0].strip()


def activity(records, test_ids, questions, since):
    """The numbers behind the activity section. Pure."""
    real = [r for r in records if not is_test(r, test_ids)]
    graded = [r for r in real if r.get("grade")]
    new = [r for r in real if after(r.get("timestamp"), since)]
    newly_graded = [r for r in graded if after(r.get("graded_at") or r.get("timestamp"), since)]
    known = {normalize_question(q.get("question")) for q in questions}
    candidates, seen = [], set()
    for r in new:
        key = normalize_question(r.get("question"))
        if key and key not in known and key not in seen:
            seen.add(key)
            candidates.append(r.get("question"))
    return {
        "real": len(real),
        "test": len(records) - len(real),
        "graded": len(graded),
        "ungraded": len(real) - len(graded),
        "new": len(new),
        "new_records": new,
        "newly_graded": len(newly_graded),
        "grades": dict(Counter(short_grade(r["grade"]) for r in graded)),
        "harmful": [r for r in newly_graded if str(r["grade"]).startswith(HARMFUL)],
        "misleading": [r for r in newly_graded if str(r["grade"]).startswith(MISLEADING)],
        "candidates": candidates,
    }


def bad_grade_lines(records):
    lines = []
    for r in records[:MAX_LISTED]:
        by = r.get("graded_by") or "unknown"
        lines.append(f"- **{short_grade(r['grade'])}** - \"{clip(r.get('question'), 150)}\" "
                     f"(asked {when(parse_time(r.get('timestamp')))}, graded by {by})")
        if r.get("comment"):
            lines.append(f"  - Comment: {clip(r['comment'], COMMENT_CHARS)}")
    if len(records) > MAX_LISTED:
        lines.append(f"- ...and {len(records) - MAX_LISTED} more")
    return lines


def section_activity(ctx):
    path = sessions_path()
    records, bad = read_jsonl(path)
    questions = read_json(config.QUESTIONS_PATH, []) or []
    previous = ctx.get("previous")
    since = parse_time(previous.get("generated_at")) if previous else None
    a = activity(records, read_test_ids(listed_test_ids_path()), questions, since)

    if path.exists():
        changed = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        source = f"Read from `{path.name}` (last changed {when(changed)})."
    else:
        source = f"No `{path.name}` yet - nobody has used the app on this machine."
    lines = [source + " The surgeon grades on the hosted app; this copy only has those "
             "grades if it was refreshed from the Hub dataset (docs/report.md).", ""]
    period = f"since the previous report ({when(since)})" if since else "so far (first report)"
    lines.append(f"**{a['new']}** new question(s) asked and **{a['newly_graded']}** "
                 f"answer(s) graded {period}.")
    lines += ["", table(["records", "count"], [
        ("real (asked by people using the app)", a["real"]),
        ("test (left out of everything below)", a["test"]),
        ("graded", a["graded"]),
        ("waiting for a grade", a["ungraded"]),
    ])]
    if bad:
        lines.append(f"\n**{bad} unreadable line(s) in {path.name} were skipped.**")
    if a["new_records"]:
        shown = a["new_records"][-MAX_LISTED:]          # the newest, oldest first
        more = len(a["new_records"]) - len(shown)
        lines += ["", "New questions" + (f" (latest {len(shown)} of {len(a['new_records'])})"
                                         if more else "") + ":", "",
                  table(["asked", "by", "question", "grade"],
                        [(when(parse_time(r.get("timestamp"))), r.get("asked_by") or "-",
                          clip(r.get("question"), 120),
                          short_grade(r["grade"]) if r.get("grade") else "waiting")
                         for r in shown])]
    if a["grades"]:
        lines += ["", "Grades so far (real records):", "",
                  table(["grade", "count"], sorted(a["grades"].items(), key=lambda kv: -kv[1]))]
    flagged = a["harmful"] + a["misleading"]
    if flagged:
        lines += ["", f"**Graded {HARMFUL} or {MISLEADING} {period}:**"] + bad_grade_lines(flagged)
    if a["candidates"]:
        lines += ["", "**New questions not in the eval set** (candidates for questions.json):"]
        lines += [f"- {clip(q, 200)}" for q in a["candidates"][:MAX_LISTED]]
        if len(a["candidates"]) > MAX_LISTED:
            lines.append(f"- ...and {len(a['candidates']) - MAX_LISTED} more")

    facts = {k: a[k] for k in ("real", "test", "graded", "ungraded", "new", "newly_graded", "grades")}
    facts.update(harmful=len(a["harmful"]), misleading=len(a["misleading"]),
                 candidates=len(a["candidates"]), unreadable=bad)
    return Section("activity", "Surgeon activity", "\n".join(lines), facts)


# ------------------------------------------------------ 6. action items
def action_items(sections):
    """A short to-do list from what the other sections found. Pure: reads
    only their facts, in order of how much each one matters."""
    facts = {s.key: s.facts for s in sections}
    index, retrieval = facts.get("index", {}), facts.get("retrieval", {})
    answers, act = facts.get("answers", {}), facts.get("activity", {})
    items = []

    for s in sections:
        if "error" in s.facts:
            items.append(f"The \"{s.title}\" section failed ({clip(s.facts['error'], 150)}) - "
                         "its error is shown in place.")
    if act.get("harmful"):
        items.append(f"{act['harmful']} answer(s) graded {HARMFUL} since the last report - "
                     "read them first (Surgeon activity).")
    if index.get("missing_enabled"):
        items.append("Enabled source(s) with no file: " + ", ".join(index["missing_enabled"])
                     + ". Download them or set `enabled: false` in sources.yaml.")
    for problem in index.get("alignment", []):
        items.append(f"Search files out of step: {problem}.")
    if index.get("registry_error"):
        items.append("sources.yaml could not be read - fix it before the next ingest.")
    if index.get("manifest_drift"):
        items.append("manifest.json is out of date - re-run `python3 -m src.ingest`.")
    if retrieval.get("regressions"):
        items.append("Search got worse (see Search quality):"
                     + "".join(f"\n  - {r}" for r in retrieval["regressions"]))
    if act.get("misleading"):
        items.append(f"{act['misleading']} answer(s) graded {MISLEADING} since the last report.")
    if answers.get("flagged"):
        items.append(f"The answer checker flagged {len(answers['flagged'])} answer(s): "
                     + ", ".join(f"Q{i}" for i in answers["flagged"]) + " (Answer quality).")
    if answers.get("fabricated"):
        items.append(f"{answers['fabricated']} answer(s) cite a page the model was never shown "
                     "(Answer quality).")
    if answers.get("errors"):
        items.append(f"{answers['errors']} of {answers.get('n', '?')} eval question(s) could not "
                     f"be answered or judged (first error: {answers.get('first_error')}).")
    if answers.get("unavailable"):
        items.append("The answer checker could not run (src/judge.py would not import).")
    if index.get("not_indexed"):
        items.append("Enabled but not in the index yet: " + ", ".join(index["not_indexed"])
                     + ". Run `python3 -m src.ingest && python3 -m src.embed`.")
    if index.get("stale_in_index"):
        items.append("In the index but disabled: " + ", ".join(index["stale_in_index"])
                     + ". Re-run ingest and embed to drop them.")
    if act.get("ungraded"):
        items.append(f"{act['ungraded']} answer(s) waiting for the surgeon to grade.")
    if act.get("unreadable"):
        items.append(f"{act['unreadable']} unreadable line(s) in sessions.jsonl.")
    if act.get("candidates"):
        items.append(f"{act['candidates']} new question(s) could join the eval set.")
    if index.get("ready_to_enable"):
        items.append("Downloaded but not enabled: " + ", ".join(index["ready_to_enable"])
                     + ". Check the license, then set `enabled: true`.")
    return items


def section_actions(sections):
    items = action_items(sections)
    body = "\n".join(f"- {i}" for i in items) if items else "Nothing needs attention tonight."
    return Section("actions", "Action items", body, {"items": items})


# ------------------------------------------------------------- assembly
def run_section(key, title, builder, *args):
    """builder(*args) -> Section, or a Section holding the error. One
    broken section must not cost the rest of the report."""
    try:
        return builder(*args)
    except Exception as e:
        tb = traceback.format_exc(limit=-3)
        body = (f"**This section could not be built:** `{clip(f'{type(e).__name__}: {e}', 300)}`\n\n"
                f"<details><summary>traceback</summary>\n\n```\n{tb.strip()}\n```\n</details>")
        return Section(key, title, body, {"error": f"{type(e).__name__}: {e}"})


def builders():
    """(key, title, function) per section, in report order. Looked up at
    call time, so a test can monkeypatch any one of them."""
    return [
        ("header", "This run", section_header),
        ("index", "Index and sources", section_index),
        ("retrieval", "Search quality", section_retrieval),
        ("answers", "Answer quality", section_answers),
        ("activity", "Surgeon activity", section_activity),
    ]


def build_report(skip_llm=False, log=False, now=None, reports_dir=None):
    """-> (markdown, metrics dict, sections)."""
    now = now or now_utc()
    previous = previous_report(reports_dir or config.REPORTS_DIR, now)
    ctx = {"now": now, "skip_llm": skip_llm, "log": log, "previous": previous}
    sections = [run_section(key, title, fn, ctx) for key, title, fn in builders()]
    sections.append(run_section("actions", "Action items", section_actions, sections))
    return render(sections, now), metrics_record(sections, now), sections


def render(sections, now):
    items = next((s.facts.get("items") for s in sections if s.key == "actions"), None)
    if items is None:
        status = "The action items could not be built - see the end."
    elif items:
        status = f"**{len(items)} item(s) need attention** - see Action items at the end."
    else:
        status = "Nothing needs attention tonight."
    parts = [f"# Mohs RAG nightly report - {now.strftime('%Y-%m-%d')}", "", status]
    for n, s in enumerate(sections, start=1):
        parts += ["", f"## {n}. {s.title}", "", s.body]
    return "\n".join(parts) + "\n"


def metrics_record(sections, now):
    """The sibling .json: what tomorrow's report compares against."""
    facts = {s.key: s.facts for s in sections}
    header, retrieval = facts.get("header", {}), facts.get("retrieval", {})
    answers, act = facts.get("answers", {}), facts.get("activity", {})
    return {
        "generated_at": iso(now),
        "index_id": header.get("index_id"),
        "git": header.get("git"),
        "config": header.get("config"),
        "retrieval": ({"metrics": retrieval["metrics"], "per_question": retrieval["per_question"]}
                      if "metrics" in retrieval else None),
        "answers": answers.get("summary"),
        "activity": {k: v for k, v in act.items() if k != "error"} or None,
        "action_items": facts.get("actions", {}).get("items", []),
        "failed_sections": [s.key for s in sections if "error" in s.facts],
    }


def write_report(markdown, metrics, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {**metrics, "report": out.name}
    out.write_text(markdown)
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the nightly status report.")
    parser.add_argument("--skip-llm", action="store_true",
                        help="skip the answer-quality check (no Groq calls)")
    parser.add_argument("--log", action="store_true",
                        help=f"append tonight's search scores to results.jsonl (note \"{LOG_NOTE}\")")
    parser.add_argument("--out", help="write the report here instead of data/eval/reports/")
    args = parser.parse_args(argv)

    if not args.skip_llm:
        # The judge needs GROQ_API_KEY; a scheduled job has no shell profile.
        from dotenv import load_dotenv
        load_dotenv(config.ROOT / ".env")

    now = now_utc()
    markdown, metrics, sections = build_report(skip_llm=args.skip_llm, log=args.log, now=now)
    out = Path(args.out) if args.out else report_path(config.REPORTS_DIR, now)
    write_report(markdown, metrics, out)

    if args.log:
        entry = next((s.facts.get("log_entry") for s in sections if s.key == "retrieval"), None)
        if entry:
            append_jsonl(config.RESULTS_PATH, [entry])
    print(out)


if __name__ == "__main__":
    main()

"""Tests for the overnight report (src/report.py).

Every config path points into tmp_path, scoring and git are faked, and the
judge is a stand-in object, so these need no embedding model, no API key,
no network - and never touch data/eval/.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src import config, report, sources

T0 = datetime(2026, 10, 13, 3, 0, tzinfo=timezone.utc)


def ts(hours):
    """ISO timestamp `hours` after T0, the way the app writes it."""
    return (T0 + timedelta(hours=hours)).isoformat(timespec="seconds")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


QUESTIONS = [
    {"id": 1, "type": "site", "question": "Nasal ala repair", "gold_pages": [10]},
    {"id": 2, "type": "site", "question": "Lower lip repair", "gold_pages": [20]},
    {"id": 3, "type": "technique", "question": "Unlabeled question"},
    {"id": 4, "type": "out_of_scope", "question": "Best pizza in town"},
]


def score_row(qid, hit, rr=None, first=None):
    """A score.score_one()-shaped row."""
    rr = rr if rr is not None else (1.0 if hit else 0.0)
    return {"id": qid, "type": "site", "recall@1": int(hit and rr == 1.0),
            "recall@5": int(hit), "recall@10": int(hit), "precision@5": 0.4 if hit else 0.0,
            "rr": rr, "first_relevant_rank": first if first is not None else (1 if hit else None),
            "top_pages": [1, 2, 3, 4, 5]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A miniature repo: eval files, a 3-chunk index, a registry, no git."""
    eval_dir = tmp_path / "eval"
    proc = tmp_path / "processed"
    eval_dir.mkdir()
    proc.mkdir()
    paths = {
        "EVAL_DIR": eval_dir,
        "QUESTIONS_PATH": eval_dir / "questions.json",
        "RESULTS_PATH": eval_dir / "results.jsonl",
        "JUDGE_RUNS_PATH": eval_dir / "judge_runs.jsonl",
        "REPORTS_DIR": eval_dir / "reports",
        "CHUNKS_PATH": proc / "chunks.json",
        "VECTORS_PATH": proc / "vectors.npy",
        "VECTOR_KEYS_PATH": proc / "vector_keys.json",
        "MANIFEST_PATH": proc / "manifest.json",
    }
    for name, value in paths.items():
        monkeypatch.setattr(config, name, value)

    paths["QUESTIONS_PATH"].write_text(json.dumps(QUESTIONS))
    chunks = [
        {"id": "a", "book": "mohs_layers", "page": 1, "kind": "prose", "text": "x"},  # old record
        {"id": "b", "source": "mohs_layers", "domain": "clinical", "page": 2,
         "kind": "prose", "text": "y"},
        {"id": "c", "source": "ncci", "domain": "operations", "page": 3,
         "kind": "prose", "text": "z"},
    ]
    paths["CHUNKS_PATH"].write_text(json.dumps(chunks))
    np.save(paths["VECTORS_PATH"], np.zeros((3, 4), dtype=np.float32))

    (tmp_path / "book.pdf").write_text("pdf")
    registry = sources.parse_registry({"sources": [
        {"id": "mohs_layers", "title": "Book", "type": "textbook", "domain": "clinical",
         "format": "pdf_textbook", "path": "book.pdf", "license": "local"},
        {"id": "ncci", "title": "NCCI", "type": "policy", "domain": "operations",
         "format": "pdf", "path": "ncci.pdf", "license": "public", "enabled": False},
    ]}, root=tmp_path)
    monkeypatch.setattr(sources, "load_registry", lambda path=None: registry)

    monkeypatch.setattr(report, "git_info", lambda root=None: {
        "commit": "abc1234", "branch": "dev", "dirty": False})
    monkeypatch.setattr(report, "score_questions", lambda qs: (
        {}, [score_row(1, True), score_row(2, False, first=8)]))

    def no_judge():
        raise AssertionError("the judge must not be imported here")
    monkeypatch.setattr(report, "import_judge", no_judge)
    return SimpleNamespace(tmp=tmp_path, **{k.lower(): v for k, v in paths.items()})


def ctx(**over):
    base = {"now": T0, "skip_llm": True, "log": False, "previous": None}
    base.update(over)
    return base


# --------------------------------------------------------------- helpers
def test_table_neutralises_pipes_and_newlines():
    out = report.table(["q", "n"], [("a | b\nc", 2)])
    assert out.splitlines()[2] == "| a \\| b c | 2 |"


def test_clip_shortens_and_flattens():
    assert report.clip("one\n  two", 50) == "one two"
    clipped = report.clip("x" * 500, 20)
    assert len(clipped) == 20 and clipped.endswith("...")


def test_config_changes_ignores_keys_old_runs_lack():
    old = {"top_k": 5, "use_query_prefix": True}
    new = {"top_k": 8, "use_query_prefix": True, "hybrid_weight": 0.3}
    assert report.config_changes(old, new) == ["top_k 5 -> 8"]


# --------------------------------------------------------- file locations
def test_report_path_never_overwrites(tmp_path):
    first = report.report_path(tmp_path, T0)
    assert first.name == "2026-10-13.md"
    first.write_text("x")
    second = report.report_path(tmp_path, T0)
    assert second.name == "2026-10-13-2.md"
    second.with_suffix(".json").write_text("{}")   # a stray .json also blocks the name
    assert report.report_path(tmp_path, T0).name == "2026-10-13-3.md"


def test_previous_report_goes_by_generated_at_not_file_name(tmp_path):
    for name, hours in (("2026-10-13", 0), ("2026-10-13-2", 2), ("2026-10-13-10", 1)):
        (tmp_path / f"{name}.json").write_text(json.dumps(
            {"generated_at": ts(hours), "report": f"{name}.md"}))
    (tmp_path / "broken.json").write_text("{not json")

    assert report.previous_report(tmp_path, T0 + timedelta(hours=5))["report"] == "2026-10-13-2.md"
    assert report.previous_report(tmp_path, T0 + timedelta(hours=1.5))["report"] == "2026-10-13-10.md"
    assert report.previous_report(tmp_path, T0) is None
    assert report.previous_report(tmp_path / "missing", T0) is None


# ------------------------------------------------------------ 1. the run
def test_header_shows_commit_index_and_settings(env):
    section = report.section_header(ctx())
    assert "`abc1234` on `dev` (clean)" in section.body
    assert "no index ID yet" in section.body
    assert "| top_k | 5 |" in section.body
    assert section.facts["config"]["top_k"] == config.TOP_K


def test_git_info_tolerates_no_git(tmp_path):
    assert report.git_info(tmp_path) is None              # not a checkout
    assert report.git_info(tmp_path / "missing") is None  # cannot even start git there


# ---------------------------------------------------- 2. index and sources
def test_registry_status_classification(tmp_path):
    (tmp_path / "here.md").write_text("# hi")
    entry = {"title": "t", "type": "review", "domain": "clinical", "format": "markdown",
             "license": "CC BY 4.0"}
    parsed = sources.parse_registry({"sources": [
        {**entry, "id": "on_present", "path": "here.md"},
        {**entry, "id": "on_missing", "path": "nope.md"},
        {**entry, "id": "off_present", "path": "here.md", "enabled": False},
        {**entry, "id": "off_absent", "path": "nope.md", "enabled": False},
    ]}, root=tmp_path)
    assert [report.registry_status(s) for s in parsed] == ["ok", "missing", "ready", "off"]


def test_registry_cross_checks_the_index(env):
    (env.tmp / "ncci.pdf").write_text("pdf")       # downloaded, still disabled
    facts = {}
    lines = "\n".join(report.registry_lines({"mohs_layers", "ncci"}, facts))
    assert facts["missing_enabled"] == []
    assert facts["ready_to_enable"] == ["ncci"]
    assert facts["stale_in_index"] == ["ncci"]     # chunks exist for a disabled source
    assert facts["not_indexed"] == []
    assert "file downloaded, not enabled yet" in lines


def test_enabled_but_missing_is_a_problem(env, monkeypatch):
    broken = sources.parse_registry({"sources": [
        {"id": "gone", "title": "Gone", "type": "review", "domain": "clinical",
         "format": "html", "path": "gone.html", "license": "x", "url": "https://e.org"},
    ]}, root=env.tmp)
    monkeypatch.setattr(sources, "load_registry", lambda path=None: broken)
    facts = {}
    lines = "\n".join(report.registry_lines(set(), facts))
    assert facts["missing_enabled"] == ["gone"]
    assert "ENABLED BUT FILE MISSING" in lines and "https://e.org" in lines


def test_unreadable_registry_is_reported_not_raised(env, monkeypatch):
    def bad(path=None):
        raise sources.RegistryError("sources.yaml has problems")
    monkeypatch.setattr(sources, "load_registry", bad)
    facts = {}
    assert "could not be read" in report.registry_lines(set(), facts)[0]
    assert "RegistryError" in facts["registry_error"]


def test_index_counts_tolerate_old_records():
    counts = report.index_counts([{"id": "a"}, {"id": "b", "book": "mohs_layers"},
                                  {"id": "c", "source": "x", "domain": "operations"}])
    assert counts == {"mohs_layers": {"n": 2, "domains": ["clinical"]},
                      "x": {"n": 1, "domains": ["operations"]}}


def test_alignment_problems():
    assert report.alignment_problems(["a", "b"], 2) == []
    assert "2 chunks but 3 vectors" in report.alignment_problems(["a", "b"], 3)[0]
    keys = {"embedding_model": "m", "ids": ["a", "x"]}
    (problem,) = report.alignment_problems(["a", "b"], 2, keys, model="m")
    assert "first difference at row 1" in problem
    keys = {"embedding_model": "old-model", "ids": ["a", "b"]}
    (problem,) = report.alignment_problems(["a", "b"], 2, keys, model="m")
    assert "old-model" in problem


def test_section_index_counts_sources_and_checks_alignment(env):
    env.vector_keys_path.write_text(json.dumps(
        {"embedding_model": config.EMBEDDING_MODEL, "ids": ["a", "b", "c"]}))
    section = report.section_index(ctx())
    assert section.facts["counts"] == {"mohs_layers": 2, "ncci": 1}
    assert section.facts["alignment"] == []
    assert section.facts["stale_in_index"] == ["ncci"]
    assert "Search files line up" in section.body


# ----------------------------------------------------- 3. retrieval quality
def test_labeled_questions_drop_unlabeled_and_out_of_scope():
    assert [q["id"] for q in report.labeled_questions(QUESTIONS)] == [1, 2]


def test_aggregate_matches_score_py_definitions():
    m = report.aggregate([score_row(1, True), score_row(2, True, rr=0.25, first=4)])
    assert m["n_questions"] == 2
    assert m["recall@5"] == 1.0 and m["recall@1"] == 0.5 and m["mrr"] == 0.625


def test_score_questions_prefers_score_all_then_falls_back(monkeypatch):
    from src import score
    calls = []
    monkeypatch.setattr(score, "score_one", lambda q: calls.append(q["id"]) or score_row(q["id"], True))
    monkeypatch.delattr(score, "score_all", raising=False)
    summary, rows = report.score_questions(QUESTIONS[:2])
    assert calls == [1, 2] and summary["recall@5"] == 1.0

    monkeypatch.setattr(score, "score_all", lambda qs, **kw: ({"mrr": 0.9}, ["row"]),
                        raising=False)
    assert report.score_questions(QUESTIONS[:2]) == ({"mrr": 0.9}, ["row"])


def test_metric_regressions_use_the_threshold():
    before = {"mrr": 0.75, "recall@5": 0.92, "precision@5": 0.48, "recall@1": 0.6}
    after = {"mrr": 0.70, "recall@5": 0.92, "precision@5": 0.45, "recall@1": 0.7}
    assert report.metric_regressions(after, before) == [("mrr", 0.75, 0.70)]
    assert report.metric_regressions(after, None) == []


def test_hit_changes_find_lost_and_gained_questions():
    prev = {"1": {"recall@5": 1}, "2": {"recall@5": 0}, "3": {"recall@5": 1}}
    now = {"1": {"recall@5": 0}, "2": {"recall@5": 1}, "3": {"recall@5": 1}, "9": {"recall@5": 0}}
    assert report.hit_changes(now, prev) == (["1"], ["2"])


def test_retrieval_flags_regressions_against_previous_report(env):
    previous = {
        "generated_at": ts(-24), "config": config.run_config(),
        "retrieval": {"metrics": {"recall@1": 1.0, "recall@5": 1.0, "precision@5": 0.4,
                                  "mrr": 1.0},
                      "per_question": {"1": {"recall@5": 1}, "2": {"recall@5": 1}}},
    }
    section = report.section_retrieval(ctx(previous=previous))
    regressions = section.facts["regressions"]
    assert any("Q2 no longer has a right passage" in r for r in regressions)
    assert any("MRR" in r and "previous report" in r for r in regressions)
    assert section.facts["misses"] == ["2"]
    assert "Lower lip repair" in section.body               # the miss is listed by name
    assert section.facts["log_entry"]["note"] == "nightly report"


def test_retrieval_compares_with_last_logged_run_once(env):
    logged = {"timestamp": ts(-24), "note": "nightly report", "n_questions": 2,
              "recall@1": 1.0, "recall@5": 1.0, "precision@5": 0.4, "mrr": 1.0,
              "config": config.run_config()}
    write_jsonl(env.results_path, [logged])

    alone = report.section_retrieval(ctx())
    assert any("last logged run" in r for r in alone.facts["regressions"])

    # The previous report wrote that very line (--log): compare once, not twice.
    previous = {"generated_at": ts(-24), "config": config.run_config(),
                "retrieval": {"metrics": dict(logged), "per_question": {}}}
    both = report.section_retrieval(ctx(previous=previous))
    assert not any("last logged run" in r for r in both.facts["regressions"])
    assert any("previous report" in r for r in both.facts["regressions"])


def test_retrieval_with_no_labels(env):
    env.questions_path.write_text(json.dumps([QUESTIONS[2]]))
    section = report.section_retrieval(ctx())
    assert "nothing to score" in section.body and section.facts == {}


# --------------------------------------------------- 4. answer quality
JUDGE_ROWS = [
    {"id": 1, "question": "Nasal ala repair", "answer": "A bilobed flap (p. 10). " * 40,
     "verdict": {"grade": "harmful", "citation_check": "some_unsupported",
                 "rationale": "Drops the warning.", "unsupported_claims": ["claim one"]},
     "checks": {"fabricated": True}, "error": None},
    {"id": 2, "question": "Lower lip repair", "answer": "Fine (p. 20).",
     "verdict": {"grade": "correct_complete", "citation_check": "all_supported"},
     "checks": {}, "error": None},
]
JUDGE_SUMMARY = {"n": 2, "judged": 2, "errors": 0, "grades": {"harmful": 1, "correct_complete": 1},
                 "citation_checks": {"all_supported": 1, "some_unsupported": 1},
                 "out_of_scope": {"n": 0, "refused": 0}, "fabricated_citations": 1,
                 "retrieval": {"labeled": 2, "hits": 1}, "flagged": [1],
                 "flag_reasons": {1: ["harmful", "fabricated_citation"]}}


def fake_judge(seen):
    def run_eval(questions, answer_fn=None, judge_llm=None, ids=None, note="", timestamp=None):
        seen.update(note=note, n=len(questions))
        return [dict(r) for r in JUDGE_ROWS]
    return SimpleNamespace(run_eval=run_eval, summarize=lambda rows: JUDGE_SUMMARY)


def test_answers_section_logs_rows_and_lists_flags(env, monkeypatch):
    seen = {}
    monkeypatch.setattr(report, "import_judge", lambda: fake_judge(seen))
    section = report.section_answers(ctx(skip_llm=False))

    assert seen == {"note": "nightly report", "n": len(QUESTIONS)}
    logged = [json.loads(l) for l in env.judge_runs_path.read_text().splitlines()]
    assert [r["id"] for r in logged] == [1, 2]
    assert section.facts["flagged"] == ["1"] and section.facts["fabricated"] == 1
    assert "Q1: Nasal ala repair" in section.body
    assert "cites a page it was never shown" in section.body
    assert "Drops the warning." in section.body and "claim one" in section.body
    # Answers quote the textbook: only a short excerpt reaches the report.
    excerpt = next(l for l in section.body.splitlines() if "Answer began" in l)
    assert len(excerpt) < report.EXCERPT_CHARS + 40


def test_answers_section_skipped_with_skip_llm(env):
    section = report.section_answers(ctx(skip_llm=True))  # import_judge would raise
    assert section.facts == {"skipped": True}
    assert not env.judge_runs_path.exists()


def test_missing_judge_module_degrades(env, monkeypatch):
    def missing():
        raise ImportError("No module named 'src.judge'")
    monkeypatch.setattr(report, "import_judge", missing)
    section = report.section_answers(ctx(skip_llm=False))
    assert "not available" in section.body and "unavailable" in section.facts
    items = report.action_items([section])
    assert any("could not run" in i for i in items)


def test_supported_kwargs_only_passes_what_fn_takes():
    def contract(questions, answer_fn=None, judge_llm=None, ids=None):
        pass
    assert report.supported_kwargs(contract, note="n", timestamp="t") == {}
    assert report.supported_kwargs(lambda q, **kw: None, note="n") == {"note": "n"}


# --------------------------------------------------- 5. surgeon activity
SESSIONS = [
    {"id": "old_real", "timestamp": ts(-48), "question": "Nasal ala repair",
     "grade": "Correct and complete - I would act on this", "graded_at": ts(-47)},
    {"id": "flagged_test", "timestamp": ts(1), "question": "test q", "is_test": True,
     "grade": "Potentially harmful - could lead to a bad decision", "graded_at": ts(2)},
    {"id": "listed_test", "timestamp": ts(1), "question": "listed q",
     "grade": "Potentially harmful - could lead to a bad decision", "graded_at": ts(2)},
    {"id": "new_harmful", "timestamp": ts(-30), "question": "Ear cartilage graft?",
     "grade": "Potentially harmful - could lead to a bad decision", "graded_at": ts(1),
     "comment": "Wrong donor site.", "graded_by": "Dr X"},
    {"id": "new_misleading", "timestamp": ts(2), "question": " nasal ALA repair ",
     "grade": "Misleading - sourced, but creates the wrong impression", "graded_at": ts(3)},
    {"id": "new_ungraded", "timestamp": ts(4), "question": "Scalp flap?", "grade": None},
    {"id": "dupe", "timestamp": ts(5), "question": "scalp   flap?", "grade": None},
]


def test_is_test_by_flag_or_list():
    assert report.is_test({"id": "a", "is_test": True}, set())
    assert report.is_test({"id": "b"}, {"b"})
    assert not report.is_test({"id": "c", "is_test": False}, {"b"})


def test_activity_filters_tests_and_counts_since_last_report():
    a = report.activity(SESSIONS, {"listed_test"}, QUESTIONS, since=T0)
    assert (a["real"], a["test"]) == (5, 2)
    assert (a["graded"], a["ungraded"]) == (3, 2)
    assert a["new"] == 3                       # asked after T0: misleading, ungraded, dupe
    assert a["newly_graded"] == 2              # graded after T0, even if asked before
    assert [r["id"] for r in a["harmful"]] == ["new_harmful"]
    assert [r["id"] for r in a["misleading"]] == ["new_misleading"]
    # "nasal ALA repair" is an eval question; the two scalp questions are one.
    assert a["candidates"] == ["Scalp flap?"]
    assert a["grades"] == {"Correct and complete": 1, "Potentially harmful": 1, "Misleading": 1}


def test_first_report_counts_everything_as_new():
    a = report.activity(SESSIONS, {"listed_test"}, QUESTIONS, since=None)
    assert a["new"] == 5 and a["newly_graded"] == 3


HARMFUL_GRADE = "Potentially harmful - could lead to a bad decision"


def test_activity_stores_what_it_saw_for_tomorrow():
    a = report.activity(SESSIONS, {"listed_test"}, QUESTIONS, since=None)
    assert a["seen_ids"] == ["dupe", "new_harmful", "new_misleading", "new_ungraded", "old_real"]
    assert a["graded_seen"] == [["new_harmful", "Potentially harmful"],
                                ["new_misleading", "Misleading"],
                                ["old_real", "Correct and complete"]]
    assert json.loads(json.dumps(a["graded_seen"])) == a["graded_seen"]   # survives the .json


def test_activity_decides_new_by_identity_when_the_previous_report_kept_ids():
    """A grade made before the previous report but copied down from the Hub
    after it is still new; a grade the previous report already showed is
    not, even if its graded_at is later (a second run the same night)."""
    records = [
        {"id": "late_grade", "timestamp": ts(-30), "question": "Ear cartilage graft?",
         "grade": "Potentially harmful - could lead to a bad decision", "graded_at": ts(-20),
         "comment": "Wrong donor site."},
        {"id": "late_question", "timestamp": ts(-10), "question": "Scalp flap?", "grade": None},
        {"id": "shown", "timestamp": ts(-30), "question": "Nasal ala repair",
         "grade": "Misleading - sourced, but creates the wrong impression", "graded_at": ts(1)},
        {"id": "regraded", "timestamp": ts(-30), "question": "Lower lip repair",
         "grade": "Misleading - sourced, but creates the wrong impression", "graded_at": ts(-20)},
    ]
    prev_act = {"seen_ids": ["late_grade", "shown", "regraded"],
                "graded_seen": [["shown", "Misleading"], ["regraded", "Correct and complete"]]}
    a = report.activity(records, set(), QUESTIONS, since=T0, prev_act=prev_act)
    assert [r["id"] for r in a["new_records"]] == ["late_question"]
    assert a["candidates"] == ["Scalp flap?"]
    assert [r["id"] for r in a["harmful"]] == ["late_grade"]
    assert [r["id"] for r in a["misleading"]] == ["regraded"]      # the grade changed
    assert a["newly_graded"] == 2


def test_activity_falls_back_to_time_for_older_reports_and_records_without_id():
    old_format = {"real": 5, "graded": 3, "harmful": 0}             # no seen_ids stored
    assert report.previously_seen(old_format) == (None, None)
    assert report.previously_seen(None) == (None, None)
    a = report.activity(SESSIONS, {"listed_test"}, QUESTIONS, since=T0, prev_act=old_format)
    assert a["new"] == 3 and a["newly_graded"] == 2                 # as by time alone

    # A record with no id cannot be matched, so time decides for it.
    no_id = {"timestamp": ts(-30), "question": "q", "grade": HARMFUL_GRADE, "graded_at": ts(1)}
    a = report.activity([no_id], set(), QUESTIONS, since=T0,
                        prev_act={"seen_ids": [], "graded_seen": []})
    assert a["new"] == 0 and a["newly_graded"] == 1 and len(a["harmful"]) == 1
    assert a["seen_ids"] == [] and a["graded_seen"] == []


def test_late_hub_grade_is_reported_once_across_nights(env, monkeypatch, capsys):
    """Review scenario: graded Potentially harmful on the hosted app on
    Monday evening, the local sessions.jsonl copied down from the Hub only
    after Tuesday's report. By time alone no night ever listed it."""
    sessions = env.eval_dir / "sessions.jsonl"
    asked = {"id": "hub_rec", "timestamp": ts(-8), "question": "Ear cartilage graft?",
             "grade": None}
    write_jsonl(sessions, [asked])
    nights = iter([T0, T0 + timedelta(days=1), T0 + timedelta(days=2)])
    monkeypatch.setattr(report, "now_utc", lambda: next(nights))

    report.main(["--skip-llm"])
    first = json.loads((env.reports_dir / "2026-10-13.json").read_text())
    assert first["activity"]["seen_ids"] == ["hub_rec"]
    assert first["activity"]["graded_seen"] == []

    # Wednesday morning: the Hub copy replaces the local file. The grade was
    # made before the first report ran; a late question came with it.
    write_jsonl(sessions, [
        {**asked, "grade": HARMFUL_GRADE, "graded_at": ts(-7), "comment": "Wrong donor site.",
         "graded_by": "Dr X"},
        {"id": "hub_late_q", "timestamp": ts(-6), "question": "Scalp flap?", "grade": None},
    ])
    report.main(["--skip-llm"])
    second = json.loads((env.reports_dir / "2026-10-14.json").read_text())
    text = (env.reports_dir / "2026-10-14.md").read_text()
    assert second["activity"]["harmful"] == 1 and second["activity"]["newly_graded"] == 1
    assert "Potentially harmful" in second["action_items"][0]
    assert "Wrong donor site." in text
    assert second["activity"]["candidates"] == 1 and "Scalp flap?" in text

    report.main(["--skip-llm"])                    # nothing changed: nothing repeated
    third = json.loads((env.reports_dir / "2026-10-15.json").read_text())
    assert third["activity"]["harmful"] == 0 and third["activity"]["new"] == 0
    assert third["activity"]["candidates"] == 0
    assert not any(report.HARMFUL in i for i in third["action_items"])


def test_after_treats_unreadable_times_as_old():
    assert report.after("garbage", T0) is False
    assert report.after(None, None) is True
    assert report.after("2026-10-13T04:00:00Z", T0) is True
    assert report.after("2026-10-13T04:00:00", T0) is True     # no zone = UTC


def test_section_activity_reads_sessions_without_writing(env):
    path = env.eval_dir / "sessions.jsonl"
    write_jsonl(path, SESSIONS)
    with open(path, "a") as f:
        f.write("{half a line\n")
    (env.eval_dir / "test_records.json").write_text(json.dumps({"ids": ["listed_test"]}))
    before = path.read_text()

    section = report.section_activity(ctx(previous={"generated_at": report.iso(T0)}))
    assert path.read_text() == before
    assert section.facts["harmful"] == 1 and section.facts["misleading"] == 1
    assert section.facts["unreadable"] == 1
    assert "Wrong donor site." in section.body
    assert "test q" not in section.body and "listed q" not in section.body
    assert "Scalp flap?" in section.body
    # new records are tabled with who asked and whether they are graded yet
    assert "| asked | by | question | grade |" in section.body
    assert "Ear cartilage graft?" not in section.body.split("| asked |")[1].split("**Graded")[0]


def test_section_activity_without_a_sessions_file(env):
    section = report.section_activity(ctx())
    assert section.facts["real"] == 0 and "No `sessions.jsonl` yet" in section.body


# ------------------------------------------------------ 6. action items
def test_action_items_put_harm_first_and_list_everything():
    sections = [
        report.Section("index", "Index", "", {"missing_enabled": ["x"], "alignment": [],
                                              "ready_to_enable": ["y"]}),
        report.Section("retrieval", "Search", "", {"regressions": ["MRR fell"]}),
        report.Section("answers", "Answers", "", {"flagged": ["3"], "fabricated": 2,
                                                  "errors": 0}),
        report.Section("activity", "Activity", "", {"harmful": 1, "ungraded": 4,
                                                    "candidates": 2}),
    ]
    items = report.action_items(sections)
    assert "Potentially harmful" in items[0]
    text = "\n".join(items)
    for needle in ("Enabled source(s) with no file: x", "Search got worse (see Search quality):\n  - MRR fell",
                   "flagged 1 answer(s): Q3", "2 answer(s) cite a page", "4 answer(s) waiting",
                   "2 new question(s)", "Downloaded but not enabled: y"):
        assert needle in text


def test_nothing_to_do():
    assert report.section_actions([]).body == "Nothing needs attention tonight."


# ------------------------------------------------------------- assembly
def test_failing_section_is_caught_and_reported(env, monkeypatch):
    def broken(ctx):
        raise ValueError("chunks.json is corrupt")
    monkeypatch.setattr(report, "section_index", broken)

    markdown, metrics, sections = report.build_report(skip_llm=True, now=T0)
    assert [s.key for s in sections] == ["header", "index", "retrieval", "answers",
                                         "activity", "actions"]
    assert "This section could not be built:** `ValueError: chunks.json is corrupt`" in markdown
    assert "## 3. Search quality" in markdown and "## 6. Action items" in markdown
    assert metrics["failed_sections"] == ["index"]
    assert any("Index and sources" in i for i in metrics["action_items"])


def test_main_skip_llm_writes_report_and_json_only(env, capsys):
    out = env.tmp / "out" / "r.md"
    report.main(["--skip-llm", "--out", str(out)])

    assert capsys.readouterr().out.strip() == str(out)
    text = out.read_text()
    assert text.startswith("# Mohs RAG nightly report")
    assert "Skipped (`--skip-llm`)" in text
    metrics = json.loads(out.with_suffix(".json").read_text())
    assert metrics["report"] == "r.md" and metrics["answers"] is None
    assert metrics["retrieval"]["per_question"]["2"]["recall@5"] == 0
    assert not env.results_path.exists()          # no --log, no results.jsonl line
    assert not env.judge_runs_path.exists()


def test_nightly_runs_chain_and_log(env, monkeypatch, capsys):
    """Two runs on one night: the second gets -2, compares against the
    first, and --log appends one results.jsonl line per run."""
    times = iter([T0, T0 + timedelta(hours=1)])
    monkeypatch.setattr(report, "now_utc", lambda: next(times))

    report.main(["--skip-llm", "--log"])
    first = env.reports_dir / "2026-10-13.md"
    assert first.exists()

    # Overnight, Q1 loses its hit.
    monkeypatch.setattr(report, "score_questions", lambda qs: (
        {}, [score_row(1, False, first=7), score_row(2, False, first=8)]))
    report.main(["--skip-llm", "--log"])
    second = env.reports_dir / "2026-10-13-2.md"
    text = second.read_text()

    assert "Previous report:** 2026-10-13.md" in text
    assert "Q1 no longer has a right passage in the top 5 since the previous report" in text
    logged = [json.loads(l) for l in env.results_path.read_text().splitlines()]
    assert [r["note"] for r in logged] == ["nightly report", "nightly report"]
    assert logged[1]["recall@5"] == 0.0
    # The first night's results.jsonl line is the previous report itself, so
    # it is not compared a second time.
    assert "since the last logged run" not in text


def test_main_with_llm_runs_the_judge(env, monkeypatch, capsys):
    monkeypatch.setattr(config, "ROOT", env.tmp)            # no real .env loaded
    seen = {}
    monkeypatch.setattr(report, "import_judge", lambda: fake_judge(seen))
    out = env.tmp / "r.md"
    report.main(["--out", str(out)])

    text = out.read_text()
    assert "## 4. Answer quality" in text and "Flagged answers" in text
    assert seen["n"] == len(QUESTIONS)
    metrics = json.loads(out.with_suffix(".json").read_text())
    assert metrics["answers"]["fabricated_citations"] == 1
    assert len(env.judge_runs_path.read_text().splitlines()) == 2


def test_retrieval_uses_score_all_summary_and_config(env, monkeypatch):
    effective = {**config.run_config(), "hybrid_weight": 0.3}
    summary = {"n_questions": 2, "mrr": 0.5, "recall@top_k": 0.5, "config": effective,
               "unlabeled": [], "out_of_scope": []}
    monkeypatch.setattr(report, "score_questions", lambda qs: (
        summary, [score_row(1, True), score_row(2, False)]))
    entry = report.section_retrieval(ctx()).facts["log_entry"]
    assert entry["config"]["hybrid_weight"] == 0.3        # what was measured, not config.py
    assert entry["recall@top_k"] == 0.5 and entry["mrr"] == 0.5
    assert "unlabeled" not in entry and entry["note"] == "nightly report"


def test_answers_section_renders_the_real_judge_summary(env, monkeypatch):
    """Contract check against src/judge.py itself, with both LLMs faked:
    the rows and summary it really produces render and flag correctly."""
    judge = pytest.importorskip("src.judge")
    chunk = {"id": "c10", "source": "mohs_layers", "domain": "clinical", "page": 10,
             "locator": "p. 10", "section": [], "kind": "prose", "text": "text on p. 10"}

    def answer_fn(question):
        return {"question": question, "text": "Use a flap (p. 10). Also this (p. 99).",
                "citations": [10, 99], "retrieved_ids": ["c10"], "retrieved_pages": [10],
                "retrieved": [{"rank": 1, "score": 0.9, "chunk": chunk}],
                "figures": [], "config": {}}

    verdict = json.dumps({"grade": "harmful", "citation_check": "some_unsupported",
                          "refused": False, "answerable": True,
                          "unsupported_claims": ["Also this"], "omissions": [],
                          "rationale": "Cites a page that was not shown."})
    wrapped = SimpleNamespace(
        run_eval=lambda questions, **kw: judge.run_eval(
            questions, answer_fn=answer_fn, judge_llm=lambda prompt: verdict, **kw),
        summarize=judge.summarize)
    monkeypatch.setattr(report, "import_judge", lambda: wrapped)

    section = report.section_answers(ctx(skip_llm=False))
    assert section.facts["errors"] == 0
    assert section.facts["fabricated"] == len(QUESTIONS)          # p. 99 was never retrieved
    assert section.facts["flagged"] == ["1", "2", "3", "4"]
    assert "Potentially harmful" in section.body
    assert "answered a question it should have refused" in section.body   # Q4, out of scope
    logged = [json.loads(l) for l in env.judge_runs_path.read_text().splitlines()]
    assert len(logged) == len(QUESTIONS) and {r["note"] for r in logged} == {"nightly report"}

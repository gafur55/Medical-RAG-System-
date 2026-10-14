"""Tests for src/judge.py - no API key, no network, no embedding model.

The answering model and the judge are both fakes: plain functions that
return canned text, so every parsing path, retry and error row can be
forced on purpose. Nothing here writes to data/ - file tests use tmp_path.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys

import pytest

from src import config, judge


# ----------------------------------------------------------------- helpers
def fake_hits(*pages, kind="prose", texts=None):
    hits = []
    for rank, page in enumerate(pages, start=1):
        text = texts[rank - 1] if texts else f"text on page {page}"
        chunk = {"id": f"c{page}", "book": "mohs_layers", "page": page,
                 "kind": kind, "text": text, "figures": []}
        hits.append({"rank": rank, "score": 0.9 - rank / 100, "chunk": chunk})
    return hits


def fake_result(text, pages=(141, 145, 160, 162, 179), question="nasal ala defect",
                **extra):
    hits = fake_hits(*pages)
    return {"question": question, "text": text,
            "citations": sorted({int(p) for p in judge.PAGE_CITATION_RE.findall(text)}),
            "retrieved_ids": [h["chunk"]["id"] for h in hits],
            "retrieved_pages": list(pages), "retrieved": hits, "figures": [],
            "config": {"llm_model": "fake", "prompt_version": "abcd1234"}, **extra}


def verdict_json(**overrides):
    v = {"grade": "correct_complete", "citation_check": "all_supported",
         "refused": False, "answerable": True, "unsupported_claims": [],
         "omissions": [], "rationale": "Supported by E1."}
    v.update(overrides)
    return json.dumps(v)


class FakeLLM:
    """Returns the given replies in order and records every prompt."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeAPIError(Exception):
    def __init__(self, status_code, message="error", headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = type("R", (), {"status_code": status_code,
                                       "headers": headers or {}})()


OOS = {"id": 14, "type": "out_of_scope", "question": "Chemotherapy for melanoma?",
       "gold_pages": [], "gold_anchors": []}
SITE = {"id": 1, "type": "site", "question": "nasal ala defect",
        "gold_pages": [160], "gold_anchors": []}


# ------------------------------------------------------- scales and prompt
def app_list(name):
    """GRADES / CITATION_CHECKS read out of app.py's source - importing app
    builds the Gradio UI, which this check does not need."""
    tree = ast.parse((config.ROOT / "app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in app.py")


def test_labels_match_app_exactly():
    assert list(judge.GRADE_LABELS.values()) == app_list("GRADES")
    assert list(judge.CITATION_LABELS.values()) == app_list("CITATION_CHECKS")


def test_codes_round_trip_through_labels():
    for code, label in judge.GRADE_LABELS.items():
        assert judge.grade_code(label) == code
        assert judge.grade_code(code) == code
    for code, label in judge.CITATION_LABELS.items():
        assert judge.citation_code(label) == code
    assert judge.grade_code("Correct and complete - reworded explanation") == "correct_complete"
    assert judge.grade_code(None) is None
    assert judge.grade_code("something else") is None


def test_judge_prompt_version_is_content_hash():
    assert re.fullmatch(r"[0-9a-f]{8}", judge.JUDGE_PROMPT_VERSION)
    expected = hashlib.sha256(judge.JUDGE_PROMPT.encode("utf-8")).hexdigest()[:8]
    assert judge.JUDGE_PROMPT_VERSION == expected


def test_prompt_shows_numbered_excerpts_question_and_answer():
    result = fake_result("Use a bilobed flap (p. 145).", pages=(141, 145))
    result["retrieved"][1]["chunk"]["kind"] = "caption"
    prompt = judge.build_judge_prompt(result, SITE)
    assert "[E1] Mohs textbook - p. 141\ntext on page 141" in prompt
    assert "[E2] Mohs textbook - p. 145 - figure caption\ntext on page 145" in prompt
    assert "QUESTION: nasal ala defect" in prompt
    assert "ANSWER:\nUse a bilobed flap (p. 145)." in prompt
    assert "not a clinical authority" in prompt
    # The judge is never told the gold labels or the question type.
    assert "gold" not in prompt and "site" not in prompt.split("EXCERPTS:")[1]


def test_excerpts_without_pages_show_section_path():
    hits = [{"rank": 1, "score": 0.5,
             "chunk": {"id": "x", "source": "statpearls_mohs", "page": None,
                       "section": ["Indications", "Tumor size"], "kind": "prose",
                       "text": "body"}}]
    text = judge.format_excerpts(hits, labels={"statpearls_mohs": "StatPearls: Mohs surgery"})
    assert text == "[E1] StatPearls: Mohs surgery - Indications > Tumor size\nbody"
    assert judge.format_excerpts([]) == "(no excerpts were retrieved)"


def test_import_needs_no_generate_score_or_groq():
    code = ("import sys, src.judge; "
            "print(sorted(m for m in ('src.generate', 'src.score', 'src.retrieve', "
            "'src.storage', 'groq') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("reply", [
    verdict_json(),
    "```json\n" + verdict_json() + "\n```",
    "Here is my verdict:\n" + verdict_json() + "\nLet me know if you need more.",
    "<think>maybe {\"grade\": \"nope\"}</think>" + verdict_json(),
    '{"verdict": ' + verdict_json() + "}",
    "Draft: {not json} then " + verdict_json(),
])
def test_parse_verdict_variants(reply):
    v = judge.parse_verdict(reply)
    assert v["grade"] == "correct_complete"
    assert v["citation_check"] == "all_supported"
    assert v["refused"] is False and v["answerable"] is True


def test_parse_handles_braces_inside_strings():
    v = judge.parse_verdict("ok " + verdict_json(rationale="Says {p. 145} } oddly."))
    assert v["rationale"] == "Says {p. 145} } oddly."


def test_parse_normalizes_case_hyphens_bools_and_lists():
    v = judge.parse_verdict(json.dumps({
        "grade": " Correct-Incomplete", "citation_check": "Uncited Claims",
        "refused": "false", "answerable": "yes", "omissions": "the FTSG option"}))
    assert v == {"grade": "correct_incomplete", "citation_check": "uncited_claims",
                 "refused": False, "answerable": True, "unsupported_claims": [],
                 "omissions": ["the FTSG option"], "rationale": ""}


@pytest.mark.parametrize("reply, message", [
    ("I think it is fine.", "no JSON object"),
    (verdict_json(grade="mostly_ok"), "grade must be one of"),
    (verdict_json(citation_check="fine"), "citation_check must be one of"),
    (json.dumps({"grade": "harmful", "citation_check": "no_citations"}), "missing key"),
    (verdict_json(refused="sometimes"), "refused must be true or false"),
    ("[1, 2, 3]", "no JSON object"),
    (None, "no JSON object"),
])
def test_parse_rejects_bad_replies(reply, message):
    with pytest.raises(judge.VerdictError, match=message):
        judge.parse_verdict(reply)


# ---------------------------------------------------------------- the judge
def test_judge_answer_happy_path():
    llm = FakeLLM(verdict_json(grade="correct_incomplete", omissions=["E2: FTSG"]))
    out = judge.judge_answer(fake_result("Use a flap (p. 145)."), SITE, llm=llm)
    assert out["error"] is None and out["attempts"] == 1
    assert out["verdict"]["grade"] == "correct_incomplete"
    assert out["verdict"]["omissions"] == ["E2: FTSG"]
    assert out["checks"]["cited_pages"] == [145]
    assert out["judge_prompt_version"] == judge.JUDGE_PROMPT_VERSION
    assert out["raw_reply"] is None
    assert len(llm.prompts) == 1


def test_judge_retries_once_with_reminder_after_bad_enum():
    llm = FakeLLM(verdict_json(grade="pretty good"), verdict_json(grade="harmful"))
    out = judge.judge_answer(fake_result("Use a flap (p. 145)."), llm=llm)
    assert out["verdict"]["grade"] == "harmful"
    assert out["attempts"] == 2 and out["error"] is None
    assert "Return ONLY the JSON object" in llm.prompts[1]
    assert "grade must be one of" in llm.prompts[1]
    assert llm.prompts[1].startswith(llm.prompts[0])


def test_judge_gives_up_after_one_retry_with_error_row():
    llm = FakeLLM("no idea", "still prose")
    out = judge.judge_answer(fake_result("Use a flap (p. 145)."), llm=llm)
    assert out["verdict"] is None
    assert out["attempts"] == 2
    assert "unusable judge reply" in out["error"]
    assert out["raw_reply"] == "still prose"
    # The checks do not depend on the judge.
    assert out["checks"]["cited_pages"] == [145]


def test_judge_call_failure_is_not_retried_and_does_not_raise():
    llm = FakeLLM(RuntimeError("GROQ_API_KEY is not set"))
    out = judge.judge_answer(fake_result("Use a flap (p. 145)."), llm=llm)
    assert out["verdict"] is None and out["attempts"] == 1
    assert "judge call failed: RuntimeError" in out["error"]


def test_judge_answer_survives_a_malformed_result():
    llm = FakeLLM(verdict_json())
    out = judge.judge_answer(None, llm=llm)
    assert out["verdict"] is None
    assert out["error"].startswith("could not prepare the judge")
    assert llm.prompts == []
    # A hit that is not a dict is tolerated, not fatal.
    out = judge.judge_answer({"text": "x", "retrieved": [None]}, llm=llm)
    assert out["error"] is None and out["checks"]["n_retrieved"] == 1


# ------------------------------------------------------ deterministic checks
def test_fabricated_page_citation():
    checks = judge.deterministic_checks(
        fake_result("Use a flap (p. 145). Or a graft (p. 999)."))
    assert checks["cited_pages"] == [145, 999]
    assert checks["fabricated_pages"] == [999]
    assert checks["fabricated"] is True


def test_fabricated_excerpt_number():
    checks = judge.deterministic_checks(
        fake_result("Use a flap [2]. Or a graft [1, 7]. Or heal [E3]."))
    assert checks["cited_excerpts"] == [1, 2, 3, 7]
    assert checks["fabricated_excerpts"] == [7]
    assert checks["fabricated"] is True


def test_valid_citations_in_both_styles_are_not_fabricated():
    page_style = judge.deterministic_checks(fake_result("Flap (p. 141; p. 145)."))
    numbered = judge.deterministic_checks(fake_result(
        "Flap [1]. Graft [2-5].",
        cited_sources=[{"n": 1, "source": "mohs_layers", "page": 141}]))
    for checks in (page_style, numbered):
        assert checks["fabricated"] is False
        assert checks["fabricated_pages"] == checks["fabricated_excerpts"] == []
    assert numbered["cited_excerpts"] == [1, 2, 3, 4, 5]
    assert numbered["n_retrieved"] == 5


def test_page_markers_and_links_are_not_excerpt_numbers():
    assert judge.cited_excerpts("see [Page 141] and [the flap](http://x) [^1]") == []


def test_excerpt_forms_generate_also_reads():
    text = "Flap [1][3]. Graft 【2】. Heal 【4†L4-L9】. Range [5—6]."
    assert judge.cited_excerpts(text) == [1, 2, 3, 4, 5, 6]
    assert judge.cited_excerpts("[2-4] and [1-1000]") == [1, 2, 3, 4, 1000]


@pytest.mark.parametrize("text", [
    "Flap [2]. Graft [1, 3]. Heal [4-5].",
    "Flap [1][3]; graft 【2】 and 【4†L4-L9】.",
    "Copied reference [12] and a year [2019], plus [3].",
    "Page style (p. 145) has none [Page 145].",
])
def test_excerpt_numbers_agree_with_generate_within_range(text):
    """Inside 1..k the judge must read exactly what generate.py reads -
    otherwise a valid citation could be called fabricated, or the reverse.
    Outside it, generate drops the number and the judge flags it."""
    generate = pytest.importorskip("src.generate")
    if not hasattr(generate, "extract_excerpt_numbers"):
        pytest.skip("generate.py has no numbered citation style")
    k = 5
    ours = judge.cited_excerpts(text)
    assert [n for n in ours if 1 <= n <= k] == generate.extract_excerpt_numbers(text, k)


def test_out_of_scope_refusal_is_correct():
    checks = judge.deterministic_checks(fake_result(
        "The excerpts do not contain information on chemotherapy for melanoma."), OOS)
    assert checks["refused"] is True
    assert checks["expected_refusal"] is True
    assert checks["refusal_correct"] is True
    assert checks["retrieval_hit"] is None


def test_out_of_scope_answered_is_a_failed_refusal():
    checks = judge.deterministic_checks(
        fake_result("Use dacarbazine (p. 145)."), OOS)
    assert checks["refused"] is False
    assert checks["refusal_correct"] is False


def test_refusal_correct_only_defined_for_out_of_scope():
    checks = judge.deterministic_checks(fake_result("Nothing retrieved."), SITE)
    assert checks["refused"] is True
    assert checks["expected_refusal"] is False
    assert checks["refusal_correct"] is None
    no_record = judge.deterministic_checks(fake_result("Nothing retrieved."))
    assert no_record["expected_refusal"] is None


@pytest.mark.parametrize("text, refused", [
    ("The excerpts do not contain the answer.", True),
    ("The provided excerpts do not specifically address diabetes.", True),
    ("I cannot answer this from the provided text.", True),
    ("There is no information in the excerpts about chemotherapy.", True),
    ("Nothing retrieved.", True),
    ("A bilobed flap suits nasal tip defects (p. 145).", False),
    ("Use a graft. The graft does not cover exposed cartilage well.", False),
    ("The textbook does not recommend grafts here; use a flap (p. 12).", False),
    # A gap noted late in a cited answer is not a refusal...
    ("A " + "long answer " * 30 + "(p. 12). The excerpts do not give a size limit.", False),
    # ...but the same phrase late in an uncited answer is.
    ("Well. " + "Some general words. " * 15 + "The excerpts do not cover this.", True),
])
def test_refusal_detector(text, refused):
    n = len(judge.cited_pages(text)) + len(judge.cited_excerpts(text))
    assert judge.is_refusal(text, n) is refused


def test_retrieval_hit_by_gold_page_and_anchor():
    hit = judge.deterministic_checks(fake_result("x (p. 141)."), SITE)
    assert hit["retrieval_hit"] is True and hit["relevant_ranks"] == [3]

    anchored = {"id": 2, "type": "site", "gold_pages": [],
                "gold_anchors": ["alar defects that measure"]}
    result = fake_result("x", pages=(1, 2))
    result["retrieved"][1]["chunk"]["text"] = "Alar defects  that\nmeasure 1 cm..."
    checks = judge.deterministic_checks(result, anchored)
    assert checks["retrieval_hit"] is True and checks["relevant_ranks"] == [2]

    miss = judge.deterministic_checks(fake_result("x", pages=(1, 2)), SITE)
    assert miss["retrieval_hit"] is False and miss["relevant_ranks"] == []


def test_is_relevant_can_be_injected():
    checks = judge.deterministic_checks(fake_result("x"), SITE,
                                        is_relevant=lambda c, q: c["page"] == 179)
    assert checks["relevant_ranks"] == [5]


def test_answer_length():
    checks = judge.deterministic_checks(fake_result("Use a flap (p. 145)."))
    assert checks["answer_words"] == 5
    assert checks["answer_chars"] == len("Use a flap (p. 145).")


# ------------------------------------------------------------ rate limits
def test_backoff_retries_rate_limits_then_succeeds():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise FakeAPIError(429, "rate limited")
        return "ok"

    assert judge.with_backoff(fn, sleep=sleeps.append) == "ok"
    assert sleeps == [2.0, 4.0]


def test_backoff_honours_retry_after_and_gives_up():
    sleeps = []

    def fn():
        raise FakeAPIError(429, headers={"retry-after": "7"})

    with pytest.raises(FakeAPIError):
        judge.with_backoff(fn, tries=3, sleep=sleeps.append)
    assert sleeps == [7.0, 7.0]


def test_backoff_does_not_retry_other_errors():
    sleeps = []

    def fn():
        raise FakeAPIError(401, "bad key")

    with pytest.raises(FakeAPIError, match="bad key"):
        judge.with_backoff(fn, sleep=sleeps.append)
    assert sleeps == []


class FakeCompletions:
    def __init__(self, reject_json_mode):
        self.reject_json_mode = reject_json_mode
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs and self.reject_json_mode:
            raise FakeAPIError(400, "response_format json_object is not supported")
        message = type("M", (), {"content": verdict_json()})()
        return type("C", (), {"choices": [type("Ch", (), {"message": message})()]})()


def fake_client(completions):
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


def test_groq_judge_llm_uses_json_mode_and_temperature_zero(monkeypatch):
    completions = FakeCompletions(reject_json_mode=False)
    monkeypatch.setattr(judge, "_client", fake_client(completions))
    monkeypatch.setattr(judge, "_json_mode", True)
    assert judge.parse_verdict(judge.groq_judge_llm("judge this, JSON"))
    (call,) = completions.calls
    assert call["model"] == config.JUDGE_MODEL
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}


def test_groq_judge_llm_falls_back_without_json_mode(monkeypatch):
    completions = FakeCompletions(reject_json_mode=True)
    monkeypatch.setattr(judge, "_client", fake_client(completions))
    monkeypatch.setattr(judge, "_json_mode", True)
    judge.groq_judge_llm("first")
    judge.groq_judge_llm("second")
    assert ["response_format" in c for c in completions.calls] == [True, False, False]
    assert judge._json_mode is False


def test_groq_judge_llm_without_key_fails_clearly(monkeypatch):
    monkeypatch.setattr(judge, "_client", None)
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        judge.groq_judge_llm("hello")


# ----------------------------------------------------------------- run_eval
QUESTIONS = [
    {"id": 1, "type": "site", "question": "nasal ala defect", "gold_pages": [160],
     "gold_anchors": []},
    {"id": 2, "type": "technique", "question": "boom", "gold_pages": [1],
     "gold_anchors": []},
    {"id": 14, "type": "out_of_scope", "question": "chemo?", "gold_pages": [],
     "gold_anchors": []},
]


def fake_answer(question):
    if question == "boom":
        raise RuntimeError("retrieval exploded")
    if question == "chemo?":
        return fake_result("The excerpts do not contain the answer.", question=question)
    return fake_result("Use a flap (p. 145).", question=question)


def test_run_eval_rows_and_error_row():
    seen = []
    rows = judge.run_eval(QUESTIONS, answer_fn=fake_answer,
                          judge_llm=FakeLLM(verdict_json()),
                          timestamp="2026-10-13T21:15:00+00:00", note="t",
                          progress=seen.append)
    assert [r["id"] for r in rows] == [1, 2, 14]
    assert seen == rows
    assert {r["run_id"] for r in rows} == {"20261013211500"}

    ok, failed, oos = rows
    assert ok["verdict"]["grade"] == "correct_complete" and ok["error"] is None
    assert ok["checks"]["retrieval_hit"] is True
    assert ok["retrieved_sources"] == ["mohs_layers"] * 5
    assert ok["answer_config"] == {"llm_model": "fake", "prompt_version": "abcd1234"}
    assert ok["judge_prompt_version"] == judge.JUDGE_PROMPT_VERSION
    assert ok["judge_model"] == "custom"
    assert ok["type"] == "site" and ok["question"] == "nasal ala defect"
    assert ok["note"] == "t" and ok["mode"] == "eval"

    assert failed["verdict"] is None
    assert "answer failed: RuntimeError: retrieval exploded" in failed["error"]
    assert oos["checks"]["refusal_correct"] is True
    json.dumps(rows)   # every row must be loggable


def test_run_eval_backs_off_when_the_answer_call_is_rate_limited(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(judge.time, "sleep", sleeps.append)

    def limited(question):
        calls.append(question)
        if len(calls) == 1:
            raise FakeAPIError(429, "tokens per minute")
        return fake_answer(question)

    (r,) = judge.run_eval(QUESTIONS[:1], answer_fn=limited,
                          judge_llm=FakeLLM(verdict_json()))
    assert r["error"] is None and r["verdict"]["grade"] == "correct_complete"
    assert len(calls) == 2 and sleeps == [2.0]


def test_run_eval_filters_ids_and_keeps_given_run_id():
    rows = judge.run_eval(QUESTIONS, answer_fn=fake_answer,
                          judge_llm=FakeLLM(verdict_json()), ids=[14], run_id="r1")
    assert [r["id"] for r in rows] == [14]
    assert rows[0]["run_id"] == "r1"


def test_run_eval_logs_the_judge_models_name():
    llm = FakeLLM(verdict_json())
    llm.model = "llama-3.3-70b-versatile"
    rows = judge.run_eval(QUESTIONS[:1], answer_fn=fake_answer, judge_llm=llm)
    assert rows[0]["judge_model"] == "llama-3.3-70b-versatile"
    assert judge.judge_model_name() == config.JUDGE_MODEL


def test_append_rows_appends(tmp_path):
    path = tmp_path / "eval" / "judge_runs.jsonl"
    judge.append_rows([{"a": 1}, {"a": 2}], path)
    judge.append_rows([{"a": 3}], path)
    assert [json.loads(l)["a"] for l in path.read_text().splitlines()] == [1, 2, 3]


# ---------------------------------------------------------------- summarize
def row(key, grade=None, cite="all_supported", refused=False, hit=True,
        fabricated=False, expected_refusal=False, error=None):
    verdict = None if grade is None else {
        "grade": grade, "citation_check": cite, "refused": refused,
        "answerable": True, "unsupported_claims": [], "omissions": [], "rationale": ""}
    return {"id": key, "verdict": verdict, "error": error,
            "judge_model": "m", "judge_prompt_version": "v",
            "checks": {"fabricated": fabricated, "retrieval_hit": hit,
                       "refused": refused, "expected_refusal": expected_refusal,
                       "refusal_correct": refused if expected_refusal else None}}


def test_summarize_counts_and_flags():
    rows = [
        row(1, "correct_complete"),
        row(2, "harmful", cite="some_unsupported", hit=False),
        row(3, "misleading", fabricated=True),
        row(4, "correct_incomplete", hit=None),
        row(5, None, error="judge call failed", fabricated=True),
        row(14, "correct_complete", cite="no_citations", refused=True, hit=None,
            expected_refusal=True),
        row(15, "correct_incomplete", hit=None, expected_refusal=True),
    ]
    s = judge.summarize(rows)
    assert s["n"] == 7 and s["judged"] == 6
    assert s["errors"] == 1 and s["error_ids"] == [5]
    assert s["grades"] == {"correct_complete": 2, "correct_incomplete": 2,
                           "misleading": 1, "harmful": 1}
    assert s["citation_checks"]["some_unsupported"] == 1
    assert s["citation_checks"]["no_citations"] == 1
    assert s["out_of_scope"] == {"n": 2, "refused": 1, "accuracy": 0.5}
    assert s["fabricated_citations"] == 2
    assert s["retrieval"] == {"labeled": 4, "hits": 3, "hit_rate": 0.75}
    assert s["triage"] == {"retrieval_fault": 1, "generation_fault": 1, "unlabeled": 1}
    assert s["flagged"] == [2, 3, 5, 15]
    assert s["flag_reasons"] == {2: ["harmful"], 3: ["misleading", "fabricated_citation"],
                                 5: ["fabricated_citation"], 15: ["failed_refusal"]}
    assert s["refusal_detector_disagrees"] == []
    assert s["judge_models"] == ["m"]


def test_refusal_despite_relevant_chunk_is_flagged_and_triaged():
    """Seen on a real run: Q13 refused, the judge agreed the excerpts did not
    answer it, yet the gold labels put relevant chunks at ranks 1, 2, 3, 5."""
    checks = judge.deterministic_checks(fake_result(
        "The provided excerpts do not contain information about danger zones."), SITE)
    assert checks["refused"] and checks["retrieval_hit"]
    assert checks["refused_despite_relevant"] is True
    oos = judge.deterministic_checks(fake_result("The excerpts do not cover it."), OOS)
    assert oos["refused_despite_relevant"] is False

    r = row(13, "correct_complete", cite="no_citations", refused=True, hit=True)
    r["checks"]["refused_despite_relevant"] = True
    s = judge.summarize([r])
    assert s["flag_reasons"] == {13: ["refused_despite_relevant"]}
    assert s["triage"]["generation_fault"] == 1


def test_summarize_notes_judge_vs_detector_refusal_disagreement():
    r = row(9, "correct_complete", refused=True)
    r["checks"]["refused"] = False
    assert judge.summarize([r])["refusal_detector_disagrees"] == [9]


def test_summarize_empty_and_formatting():
    empty = judge.summarize([])
    assert empty["n"] == 0
    assert empty["retrieval"]["hit_rate"] is None
    assert empty["out_of_scope"]["accuracy"] is None
    assert "nothing flagged" in judge.format_summary(empty)

    text = judge.format_summary(judge.summarize([row(2, "harmful", hit=False),
                                                 row(5, None, error="judge call failed")]))
    assert "Q2: harmful" in text
    assert "failed (no verdict): Q5" in text
    assert "relevant chunk in context:  1/2 (0.50)" in text


def test_format_row():
    line = judge.format_row(row(3, "misleading", fabricated=True))
    assert line.split()[:4] == ["Q3", "misleading", "all_supported", "hit"]
    assert "fabricated_citation" in line
    assert "ERROR" in judge.format_row(row(5, None, error="judge call failed"))


# ---------------------------------------------------------------- agreement
G = judge.GRADE_LABELS
C = judge.CITATION_LABELS


def doctor(rid, grade, cite=None, question="q"):
    return {"id": rid, "question": question, "grade": grade, "citation_check": cite,
            "comment": "c"}


def test_agreement_on_aligned_verdicts():
    verdicts = [
        {"grade": "correct_complete", "citation_check": "all_supported"},
        {"grade": "correct_incomplete", "citation_check": "no_citations"},
        {"grade": "harmful", "citation_check": "uncited_claims"},
        {"grade": "misleading"},
    ]
    doctors = [
        doctor("a", G["correct_complete"], C["all_supported"]),
        doctor("b", G["harmful"], C["no_citations"]),
        doctor("c", G["misleading"], C["some_unsupported"]),
        doctor("d", G["correct_complete"]),
    ]
    a = judge.agreement(verdicts, doctors)
    assert a["n"] == 4
    assert a["exact_matches"] == 1 and a["exact_match"] == 0.25
    # a: both acceptable; b: no; c: both unsafe; d: no
    assert a["binary_match"] == 0.5
    assert a["confusion"]["harmful"]["correct_incomplete"] == 1
    assert a["confusion"]["misleading"]["harmful"] == 1
    assert sum(sum(r.values()) for r in a["confusion"].values()) == 4
    assert [e["record_id"] for e in a["judge_fine_doctor_bad"]] == ["b"]
    assert a["judge_fine_doctor_bad"][0]["doctor"] == "harmful"
    assert [e["record_id"] for e in a["judge_bad_doctor_fine"]] == ["d"]
    assert a["citation_n"] == 3 and a["citation_exact_match"] == pytest.approx(2 / 3)
    assert a["skipped"] == 0


def test_agreement_matches_rows_by_record_id_and_skips_ungradable():
    rows = [
        {"record_id": "b", "verdict": {"grade": "harmful"}},
        {"record_id": "a", "verdict": None},              # judge failed
        {"record_id": "zz", "verdict": {"grade": "harmful"}},   # no such record
    ]
    doctors = [doctor("a", G["harmful"]), doctor("b", G["harmful"]),
               doctor("c", None)]
    a = judge.agreement(rows, doctors)
    assert a["n"] == 1 and a["exact_match"] == 1.0
    assert a["skipped"] == 2


def test_agreement_with_nothing_to_compare():
    a = judge.agreement([], [])
    assert a["n"] == 0 and a["exact_match"] is None and a["binary_match"] is None
    assert judge.format_agreement(a) == "no surgeon-graded answers to compare with"


def test_agreement_rejects_misaligned_lists():
    with pytest.raises(ValueError, match="aligned"):
        judge.agreement([{"grade": "harmful"}], [])


def test_format_agreement_lists_judge_misses():
    a = judge.agreement([{"grade": "correct_complete"}],
                        [doctor("r1", G["harmful"], question="Helical rim?")])
    text = judge.format_agreement(a)
    assert "same grade:            0/1" in text
    assert "r1: surgeon harmful, judge correct_complete - Helical rim?" in text


# ------------------------------------------------------- surgeon's sessions
def test_is_test_record_matches_storage():
    from src import storage
    ids = {"b"}
    for record in ({"id": "a", "is_test": True}, {"id": "b"}, {"id": "c"},
                   {"id": "c", "is_test": False}):
        assert judge.is_test_record(record, ids) == storage.is_test(record, ids)


def test_graded_real_records_and_rehydrate():
    records = [doctor("a", G["harmful"]), doctor("b", G["harmful"]),
               dict(doctor("c", G["harmful"]), is_test=True), doctor("d", None)]
    real = judge.graded_real_records(records, test_ids={"b"})
    assert [r["id"] for r in real] == ["a"]

    chunks = {"x": {"id": "x", "book": "mohs_layers", "page": 3, "kind": "prose",
                    "text": "t"}}
    rec = {"id": "a", "question": "q", "answer": "A (p. 3).", "citations": [3],
           "retrieved_ids": ["x"], "retrieved_pages": [3], "config": {"top_k": 1}}
    result = judge.rehydrate(rec, chunks)
    assert result["text"] == "A (p. 3)."
    assert result["retrieved"] == [{"rank": 1, "score": None, "chunk": chunks["x"]}]
    assert judge.rehydrate(dict(rec, retrieved_ids=["x", "gone"]), chunks) is None
    assert judge.rehydrate(dict(rec, retrieved_ids=[]), chunks) is None


def test_judge_sessions_rows_carry_doctor_grades():
    chunks = {f"c{p}": h["chunk"] for p, h in zip((141, 160), fake_hits(141, 160))}
    records = [
        {"id": "r1", "question": "Nasal ala defect ", "answer": "Flap (p. 160).",
         "citations": [160], "retrieved_ids": ["c141", "c160"],
         "retrieved_pages": [141, 160], "grade": G["misleading"],
         "citation_check": C["all_supported"]},
        {"id": "r2", "question": "q", "answer": "a", "retrieved_ids": ["gone"],
         "grade": G["harmful"]},
    ]
    rows, skipped = judge.judge_sessions(records, chunks, QUESTIONS,
                                         judge_llm=FakeLLM(verdict_json()),
                                         run_id="r")
    assert skipped == ["r2"]
    (r,) = rows
    assert r["mode"] == "sessions" and r["record_id"] == "r1"
    assert r["doctor_grade"] == "misleading"
    assert r["doctor_citation_check"] == "all_supported"
    assert r["id"] == 1                       # matched to the eval question
    assert r["checks"]["retrieval_hit"] is True
    a = judge.agreement(rows, records)
    assert a["n"] == 1 and len(a["judge_fine_doctor_bad"]) == 1
    assert judge.format_row(r).split()[:2] == ["r1", "(Q1)"]


def test_session_rows_on_the_same_question_stay_separate():
    rows = [dict(row(1, "harmful"), record_id="r1"), dict(row(1, "misleading"), record_id="r2")]
    s = judge.summarize(rows)
    assert s["flagged"] == ["r1", "r2"]
    assert s["flag_reasons"] == {"r1": ["harmful"], "r2": ["misleading"]}


# ---------------------------------------------------------------------- CLI
def test_parse_ids():
    assert judge.parse_ids("1,3") == [1, 3]
    assert judge.parse_ids(" 2 , 5,") == [2, 5]
    with pytest.raises(Exception):
        judge.parse_ids("one")


def test_cli_sessions_with_only_test_records_writes_nothing(tmp_path, monkeypatch, capsys):
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(json.dumps(doctor("t1", G["harmful"])) + "\n")
    before = sessions.read_text()
    (tmp_path / "test_records.json").write_text(json.dumps({"ids": ["t1"]}))
    monkeypatch.setattr(judge, "SESSIONS_PATH", sessions)
    monkeypatch.setattr(judge, "TEST_IDS_PATH", tmp_path / "test_records.json")
    monkeypatch.setattr(config, "JUDGE_RUNS_PATH", tmp_path / "judge_runs.jsonl")

    judge.main(["--sessions"])

    out = capsys.readouterr().out
    assert "1 session records, 0 graded" in out
    assert "nothing to compare yet" in out
    assert not (tmp_path / "judge_runs.jsonl").exists()
    assert sessions.read_text() == before


def test_cli_eval_logs_rows_and_prints_summary(tmp_path, monkeypatch, capsys):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(QUESTIONS))
    log = tmp_path / "judge_runs.jsonl"
    monkeypatch.setattr(config, "QUESTIONS_PATH", questions)
    monkeypatch.setattr(config, "JUDGE_RUNS_PATH", log)
    real_run_eval = judge.run_eval
    monkeypatch.setattr(judge, "run_eval", lambda qs, **kw: real_run_eval(
        qs, answer_fn=fake_answer, judge_llm=FakeLLM(verdict_json()), **kw))

    judge.main(["--ids", "1,14", "--note", "cli test"])

    out = capsys.readouterr().out
    assert "Q1" in out and "Q14" in out and "judged 2 of 2 answers" in out
    logged = [json.loads(l) for l in log.read_text().splitlines()]
    assert [r["id"] for r in logged] == [1, 14]
    assert {r["note"] for r in logged} == {"cli test"}
    assert len({r["run_id"] for r in logged}) == 1

"""Tests for src/score.py and src/sweep.py - no model, no index.

search() is replaced by a fake that returns hand-built hits, so every
metric can be checked by hand.
"""

import json

import pytest

from src import config, score, sweep


def chunk(cid, page, text="body", **extra):
    return {"id": cid, "book": "mohs_layers", "page": page, "kind": "prose",
            "text": text, "figures": [], **extra}


def hits(*chunks):
    return [{"rank": i, "score": 1 - i / 100, "chunk": c} for i, c in enumerate(chunks, 1)]


# ---------------------------------------------------------- is_relevant
def test_gold_pages_belong_to_the_gold_source():
    q = {"gold_pages": [134], "gold_anchors": []}
    assert score.is_relevant(chunk("old", 134), q)                       # legacy record: textbook
    assert score.is_relevant(chunk("new", 134, source="mohs_layers"), q)
    assert not score.is_relevant(chunk("pdf", 134, source="aad_guideline"), q)
    assert not score.is_relevant(chunk("html", None, source="statpearls_mohs"), q)

    q_sp = {"gold_source": "aad_guideline", "gold_pages": [134], "gold_anchors": []}
    assert score.is_relevant(chunk("pdf", 134, source="aad_guideline"), q_sp)
    assert not score.is_relevant(chunk("old", 134), q_sp)


def test_anchors_match_any_source():
    q = {"gold_pages": [], "gold_anchors": ["Karapandzic flaps are  rarely utilized"]}
    c = chunk("html", None, text="Note: karapandzic flaps\nare rarely utilized for the upper lip",
              source="statpearls_mohs", domain="clinical")
    assert score.is_relevant(c, q)
    assert not score.is_relevant(chunk("x", 1, text="unrelated"), q)


# ------------------------------------------------------------ score_one
QUESTIONS = [
    {"id": 1, "type": "site", "question": "q1", "gold_pages": [10], "gold_anchors": []},
    {"id": 2, "type": "site", "question": "q2", "gold_pages": [20], "gold_anchors": []},
    {"id": 3, "type": "site", "question": "q3", "gold_pages": [], "gold_anchors": []},
    {"id": 4, "type": "out_of_scope", "question": "q4", "gold_pages": [], "gold_anchors": []},
]
RANKINGS = {
    # q1: relevant at ranks 1 and 4
    "q1": [chunk("a", 10), chunk("b", 1), chunk("c", 2), chunk("d", 10), chunk("e", 3),
           chunk("f", 4), chunk("g", 5), chunk("h", 6), chunk("i", 7), chunk("j", 8)],
    # q2: first relevant at rank 6
    "q2": [chunk(f"x{i}", i) for i in range(1, 6)] + [chunk("y", 20)]
          + [chunk(f"z{i}", 30 + i) for i in range(4)],
}


@pytest.fixture
def fake_search(monkeypatch):
    calls = []

    def search(question, k=None, **kwargs):
        calls.append({"question": question, "k": k, **kwargs})
        return hits(*RANKINGS.get(question, [])[:k])

    monkeypatch.setattr(score, "search", search)
    return calls


def test_score_one_metrics(fake_search):
    r = score.score_one(QUESTIONS[0])
    assert (r["recall@1"], r["recall@5"], r["recall@10"]) == (1, 1, 1)
    assert r["precision@5"] == pytest.approx(2 / 5)
    assert r["rr"] == 1.0 and r["first_relevant_rank"] == 1
    assert r["top_pages"] == [10, 1, 2, 10, 3]
    assert fake_search[-1]["k"] == 10

    r = score.score_one(QUESTIONS[1], top_k=3)
    assert (r["recall@1"], r["recall@5"], r["recall@10"]) == (0, 0, 1)
    assert r["rr"] == pytest.approx(1 / 6)
    assert r["top_k"] == 3 and r["recall@top_k"] == 0 and r["precision@top_k"] == 0


def test_score_one_searches_deep_enough_for_top_k(fake_search):
    r = score.score_one(QUESTIONS[1], top_k=12)
    assert fake_search[-1]["k"] == 12
    assert r["recall@top_k"] == 1 and r["precision@top_k"] == pytest.approx(1 / 12)


def test_score_all_on_fakes(fake_search):
    summary, per_q = score.score_all(QUESTIONS, hybrid_weight=0.3, kinds=("prose",))

    assert [r["id"] for r in per_q] == [1, 2]
    assert summary["n_questions"] == 2
    assert summary["unlabeled"] == [3] and summary["out_of_scope"] == [4]
    assert summary["recall@1"] == 0.5
    assert summary["recall@5"] == 0.5
    assert summary["recall@10"] == 1.0
    assert summary["precision@5"] == pytest.approx((2 / 5 + 0) / 2)
    assert summary["mrr"] == pytest.approx((1 + 1 / 6) / 2)
    assert summary["recall@top_k"] == summary["recall@5"]     # TOP_K is 5
    # the overrides reached search() ...
    assert all(c["hybrid_weight"] == 0.3 and c["kinds"] == ("prose",) for c in fake_search)
    # ... and the logged config says what actually ran
    assert summary["config"]["hybrid_weight"] == 0.3
    assert summary["config"]["retrieve_kinds"] == ["prose"]
    assert summary["config"]["exclude_front_matter"] == config.EXCLUDE_FRONT_MATTER


def test_score_all_without_overrides_logs_plain_run_config(fake_search):
    summary, _ = score.score_all(QUESTIONS)
    assert summary["config"] == config.run_config()
    assert all(set(c) == {"question", "k"} for c in fake_search)


def test_log_record_and_append(tmp_path, fake_search):
    summary, _ = score.score_all(QUESTIONS, top_k=3)
    record = score.log_record(summary, "a note")
    assert list(record)[:8] == ["timestamp", "note", "n_questions", "recall@1", "recall@5",
                                "recall@10", "precision@5", "mrr"]
    assert "unlabeled" not in record and record["config"]["top_k"] == 3

    path = tmp_path / "results.jsonl"
    path.write_text('{"old": 1}\n')
    score.append_result(record, path)
    lines = path.read_text().splitlines()
    assert lines[0] == '{"old": 1}' and json.loads(lines[1])["note"] == "a note"


# ---------------------------------------------------------------- sweep
def fake_score_all(questions, top_k=None, **overrides):
    """Better recall@1 whenever front matter is excluded."""
    base = {"n_questions": 13, "recall@1": 0.5, "recall@5": 0.9, "recall@10": 1.0,
            "precision@5": 0.4, "mrr": 0.7, "recall@top_k": 0.9, "precision@top_k": 0.4,
            "config": {"top_k": top_k or config.TOP_K, **overrides}}
    per_q = [{"id": 1, "first_relevant_rank": 2, "precision@5": 0.4}]
    if overrides.get("exclude_front_matter"):
        base["recall@1"] = 0.6
        per_q = [{"id": 1, "first_relevant_rank": 1, "precision@5": 0.6}]
    return base, per_q


def test_sweep_grid_starts_with_baseline():
    assert sweep.GRID[0] == ("baseline", {})
    names = [name for name, _ in sweep.GRID]
    assert len(names) == len(set(names))


def test_sweep_table_and_moves():
    grid = [("baseline", {}), ("front matter excluded", {"exclude_front_matter": True}),
            ("top_k 3", {"top_k": 3})]
    results = sweep.run_grid([], grid, scorer=fake_score_all)

    assert sweep.deltas(results[1][2], results[0][2])["recall@1"] == pytest.approx(0.1)
    table = sweep.format_table(results)
    assert "front matter excluded" in table and "+0.100" in table
    assert table.splitlines()[-1].split()[6] == "3"   # "top_k", "3", 4 metrics, then k
    moves = sweep.format_moves(results)
    assert "Q1 rank 2->1 p@5 0.4->0.6" in moves
    assert "top_k 3: no question moved" in moves


def test_sweep_without_log_writes_nothing(tmp_path, monkeypatch, capsys):
    qpath = tmp_path / "questions.json"
    qpath.write_text("[]")
    monkeypatch.setattr(config, "QUESTIONS_PATH", qpath)
    monkeypatch.setattr(config, "RESULTS_PATH", tmp_path / "results.jsonl")   # belt and braces
    monkeypatch.setattr(sweep, "GRID", [("baseline", {}), ("fm", {"exclude_front_matter": True})])
    monkeypatch.setattr(score, "score_all", fake_score_all)
    written = []
    monkeypatch.setattr(score, "append_result", lambda record, path=None: written.append(record))

    sweep.main([])
    assert written == []
    assert "not logged" in capsys.readouterr().out

    sweep.main(["--log"])
    assert [r["note"] for r in written] == ["sweep: baseline", "sweep: fm"]
    assert written[1]["config"]["exclude_front_matter"] is True
    assert not (tmp_path / "results.jsonl").exists()

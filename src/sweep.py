"""Compare retrieval configurations side by side, in one process.

  python3 -m src.sweep           # print the comparison, write nothing
  python3 -m src.sweep --log     # also append every run to results.jsonl

Editing config.py and re-running score.py once per setting works, but it is
slow (the model reloads every time) and it is how Experiment A once ran the
same configuration twice. Here every configuration is a dict of search()
overrides, scored with score.score_all() against the same loaded model and
index, and logged - only with --log - with the config it actually used.

Nothing in config.py changes. The defaults stay what the doctor is
reviewing; the sweep only tells you what a change would do.

top_k rows: TOP_K is how many chunks the LLM sees, not how they are ranked,
so those rows have the baseline ranking by construction. The columns that
move are r@k and p@k - recall and precision at the k the model is shown.
"""

import argparse
import json

from src import config, score

FM = {"exclude_front_matter": True}

# (name, overrides). Single knobs first - one change at a time, so a
# difference can be attributed - then the combinations worth checking.
GRID = [
    ("baseline", {}),
    ("front matter excluded", FM),
    ("max 1 per page", {"max_per_page": 1}),
    ("max 2 per page", {"max_per_page": 2}),
    ("hybrid 0.1", {"hybrid_weight": 0.1}),
    ("hybrid 0.2", {"hybrid_weight": 0.2}),
    ("hybrid 0.3", {"hybrid_weight": 0.3}),
    ("hybrid 0.5", {"hybrid_weight": 0.5}),
    ("top_k 3", {"top_k": 3}),
    ("top_k 8", {"top_k": 8}),
    ("front matter + max 1", {**FM, "max_per_page": 1}),
    ("front matter + max 2", {**FM, "max_per_page": 2}),
    ("front matter + hybrid 0.1", {**FM, "hybrid_weight": 0.1}),
    ("front matter + hybrid 0.2", {**FM, "hybrid_weight": 0.2}),
    ("front matter + hybrid 0.3", {**FM, "hybrid_weight": 0.3}),
    ("front matter + max 1 + hybrid 0.2", {**FM, "max_per_page": 1, "hybrid_weight": 0.2}),
    ("front matter + top_k 3", {**FM, "top_k": 3}),
    ("front matter + top_k 8", {**FM, "top_k": 8}),
]

COLUMNS = ("recall@1", "recall@5", "precision@5", "mrr")
SHORT = {"recall@1": "r@1", "recall@5": "r@5", "precision@5": "p@5", "mrr": "mrr",
         "recall@top_k": "r@k", "precision@top_k": "p@k"}


def run_grid(questions, grid=None, scorer=None):
    """[(name, overrides, summary, per_question)] for every configuration.

    grid=None reads GRID at call time (the same None pattern as search()).
    """
    grid = grid if grid is not None else GRID
    scorer = scorer or score.score_all
    return [(name, overrides, *scorer(questions, **overrides)) for name, overrides in grid]


def deltas(summary, baseline):
    """{metric: summary - baseline} for the comparison columns."""
    return {m: summary[m] - baseline[m] for m in COLUMNS}


def moved(per_question, baseline_per_question):
    """Questions whose first relevant rank or precision@5 changed.

    [(id, (old_rank, new_rank), (old_p5, new_p5))], ranks None = not in
    the top 10. A metric moving because of one of these is one question -
    noise at n=13 - unless the others move with it.
    """
    before = {r["id"]: r for r in baseline_per_question}
    out = []
    for r in per_question:
        b = before.get(r["id"])
        if b and (r["first_relevant_rank"] != b["first_relevant_rank"]
                  or r["precision@5"] != b["precision@5"]):
            out.append((r["id"], (b["first_relevant_rank"], r["first_relevant_rank"]),
                        (b["precision@5"], r["precision@5"])))
    return out


def format_table(results):
    """Fixed-width comparison table; deltas are against the first row."""
    base = results[0][2]
    width = max(len(name) for name, *_ in results)
    head = (f"{'run':<{width}}  " + " ".join(f"{SHORT[m]:>6}" for m in COLUMNS)
            + f"  {'k':>2} {'r@k':>6} {'p@k':>6}  "
            + " ".join(f"{'Δ' + SHORT[m]:>7}" for m in COLUMNS))
    lines = [head, "-" * len(head)]
    for name, _, summary, _ in results:
        d = deltas(summary, base)
        lines.append(
            f"{name:<{width}}  " + " ".join(f"{summary[m]:>6.3f}" for m in COLUMNS)
            + f"  {summary['config']['top_k']:>2} {summary['recall@top_k']:>6.3f}"
            + f" {summary['precision@top_k']:>6.3f}  "
            + " ".join(f"{d[m]:>+7.3f}" for m in COLUMNS))
    return "\n".join(lines)


def format_moves(results):
    """One line per run listing the questions that moved against baseline."""
    base_per_q = results[0][3]
    lines = []
    for name, _, _, per_q in results[1:]:
        changes = moved(per_q, base_per_q)
        text = ", ".join(
            f"Q{qid} rank {a or '-'}->{b or '-'} p@5 {pa:.1f}->{pb:.1f}"
            for qid, (a, b), (pa, pb) in changes) or "no question moved"
        lines.append(f"  {name}: {text}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", action="store_true",
                        help="append every run to results.jsonl (default: write nothing)")
    args = parser.parse_args(argv)

    questions = json.loads(config.QUESTIONS_PATH.read_text())
    results = run_grid(questions)
    if not results[0][2]["n_questions"]:
        print("No labeled questions. Fill in gold_pages / gold_anchors.")
        return results

    print(f"\n{results[0][2]['n_questions']} labelled questions; "
          f"deltas are against '{results[0][0]}'. One question = "
          f"{1 / results[0][2]['n_questions']:.3f} of recall or mrr.\n")
    print(format_table(results))
    print("\nquestions that moved (first relevant rank in the top 10, precision@5):")
    print(format_moves(results))

    if args.log:
        for name, _, summary, _ in results:
            score.append_result(score.log_record(summary, f"sweep: {name}"))
        print(f"\nlogged {len(results)} runs -> {config.RESULTS_PATH}")
    else:
        print("\nnot logged (add --log to append these runs to results.jsonl)")
    return results


if __name__ == "__main__":
    main()

"""Measure retrieval quality.

  python3 -m src.eval.score --note "baseline"

Ground truth is gold_pages and gold_anchors in questions.json - pages and
phrases, not chunk IDs, so the labels survive any chunking change.

A retrieved chunk counts as relevant if EITHER its page is in gold_pages,
or its text contains one of the gold_anchors.
"""

import argparse
import json
import re
from datetime import datetime, timezone

from src import config
from src.retrieve import search


def normalize(text):
    """Lowercase and flatten whitespace, so anchors survive PDF line breaks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_relevant(chunk, question):
    if chunk["page"] in question.get("gold_pages", []):
        return True
    body = normalize(chunk["text"])
    return any(normalize(a) in body for a in question.get("gold_anchors", []))


def score_one(question, k_values=(1, 5, 10)):
    """Retrieve once at the deepest k, compute every metric from that."""
    hits = search(question["question"], k=max(k_values))
    flags = [is_relevant(h["chunk"], question) for h in hits]

    first = next((i + 1 for i, f in enumerate(flags) if f), None)

    return {
        "id": question["id"],
        "type": question["type"],
        # recall@k = "at least one relevant chunk in the top k".
        # The honest question: did the surgeon get shown a right passage?
        **{f"recall@{k}": int(any(flags[:k])) for k in k_values},
        # precision@5 = how much of the top 5 was junk.
        "precision@5": sum(flags[:5]) / 5,
        # Reciprocal rank: 1/rank of the first hit. Rank 1 -> 1.0,
        # rank 4 -> 0.25, nothing found -> 0. Rewards being near the top.
        "rr": (1 / first) if first else 0.0,
        "first_relevant_rank": first,
        "top_pages": [h["chunk"]["page"] for h in hits[:5]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="what you changed this run")
    args = parser.parse_args()

    questions = json.loads(config.QUESTIONS_PATH.read_text())

    scored, skipped, out_of_scope = [], [], []
    for q in questions:
        if q["type"] == "out_of_scope":
            # These test refusal, not retrieval. No right page exists.
            out_of_scope.append(q["id"])
        elif not q.get("gold_pages") and not q.get("gold_anchors"):
            skipped.append(q["id"])
        else:
            scored.append(score_one(q))

    if not scored:
        print("No labeled questions. Fill in gold_pages / gold_anchors.")
        return

    n = len(scored)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": args.note,
        "n_questions": n,
        "recall@1": sum(r["recall@1"] for r in scored) / n,
        "recall@5": sum(r["recall@5"] for r in scored) / n,
        "recall@10": sum(r["recall@10"] for r in scored) / n,
        "precision@5": sum(r["precision@5"] for r in scored) / n,
        "mrr": sum(r["rr"] for r in scored) / n,
        "config": config.run_config(),
    }

    print(f"\nscored {n} | unlabeled: {skipped or 'none'} | out-of-scope: {out_of_scope}\n")
    for key in ("recall@1", "recall@5", "recall@10", "precision@5", "mrr"):
        print(f"  {key:<12} {summary[key]:.3f}")

    print("\n  per question:")
    for r in scored:
        mark = "ok  " if r["recall@5"] else "MISS"
        rank = r["first_relevant_rank"] or "-"
        print(f"    {mark} Q{r['id']:<3} first hit at rank {rank:<4} p@5 {r['precision@5']:.1f}  {r['top_pages']}")

    # Append, never overwrite. This file is your experiment log.
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_PATH, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\nlogged -> {config.RESULTS_PATH}")


if __name__ == "__main__":
    main()
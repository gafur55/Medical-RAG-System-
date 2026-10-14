"""Measure retrieval quality.

  python3 -m src.score --note "baseline"

Ground truth is gold_pages and gold_anchors in questions.json - pages and
phrases, not chunk IDs, so the labels survive any chunking change.

A retrieved chunk counts as relevant if EITHER its page is in gold_pages,
or its text contains one of the gold_anchors. gold_pages are pages of ONE
source - question["gold_source"], the textbook unless it says otherwise -
because "p. 134" of a guideline is not p. 134 of the book. Anchors are
phrases, so they match in any source.

score_all() is the whole measurement as a function: sweep.py calls it once
per configuration, in one process, with search() overrides as keyword
arguments.
"""

import argparse
import json
import re
from datetime import datetime, timezone

from src import config
from src.retrieve import search, source_of

METRICS = ("recall@1", "recall@5", "recall@10", "precision@5", "mrr")

# search() keyword -> the run_config() key it overrides. The logged config
# must describe the run that produced the numbers, not whatever config.py
# said (see "A mistake worth remembering" in docs/experiments.md).
OVERRIDE_KEYS = {
    "kinds": "retrieve_kinds",
    "use_prefix": "use_query_prefix",
    "domains": "retrieve_domains",
    "exclude_front_matter": "exclude_front_matter",
    "max_per_page": "max_per_page",
    "hybrid_weight": "hybrid_weight",
}


def normalize(text):
    """Lowercase and flatten whitespace, so anchors survive PDF line breaks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_relevant(chunk, question):
    gold_source = question.get("gold_source", config.DEFAULT_SOURCE)
    if source_of(chunk) == gold_source and chunk.get("page") in question.get("gold_pages", []):
        return True
    body = normalize(chunk["text"])
    return any(normalize(a) in body for a in question.get("gold_anchors", []))


def score_one(question, k_values=(1, 5, 10), top_k=None, **search_kwargs):
    """Retrieve once at the deepest k, compute every metric from that.

    top_k: how many chunks the LLM is shown (config.TOP_K if None).
    recall@1/5/10 and precision@5 are fixed cut-offs so runs stay
    comparable; recall@top_k and precision@top_k describe the context the
    model actually gets. One deep search serves every cut-off because the
    ranking does not depend on k (see retrieve.top_indices).
    """
    top_k = top_k or config.TOP_K
    hits = search(question["question"], k=max(*k_values, top_k), **search_kwargs)
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
        "top_k": top_k,
        "recall@top_k": int(any(flags[:top_k])),
        "precision@top_k": sum(flags[:top_k]) / top_k,
        "top_ids": [h["chunk"]["id"] for h in hits[:max(10, top_k)]],
        "relevant": flags,
    }


def run_config(top_k=None, **search_kwargs):
    """config.run_config() with this run's overrides applied."""
    cfg = config.run_config()
    if top_k:
        cfg["top_k"] = top_k
    for key, value in search_kwargs.items():
        if value is not None:
            cfg[OVERRIDE_KEYS.get(key, key)] = list(value) if isinstance(value, tuple) else value
    return cfg


def score_all(questions, top_k=None, **search_kwargs):
    """Score every labelled question. Returns (summary, per_question).

    summary holds the averaged metrics, the question bookkeeping and the
    effective config - everything a results.jsonl line needs except the
    timestamp and note. search_kwargs go straight to search().
    """
    scored, skipped, out_of_scope = [], [], []
    for q in questions:
        if q["type"] == "out_of_scope":
            # These test refusal, not retrieval. No right page exists.
            out_of_scope.append(q["id"])
        elif not q.get("gold_pages") and not q.get("gold_anchors"):
            skipped.append(q["id"])
        else:
            scored.append(score_one(q, top_k=top_k, **search_kwargs))

    n = len(scored)
    summary = {"n_questions": n}
    if n:
        summary.update({
            "recall@1": sum(r["recall@1"] for r in scored) / n,
            "recall@5": sum(r["recall@5"] for r in scored) / n,
            "recall@10": sum(r["recall@10"] for r in scored) / n,
            "precision@5": sum(r["precision@5"] for r in scored) / n,
            "mrr": sum(r["rr"] for r in scored) / n,
            "recall@top_k": sum(r["recall@top_k"] for r in scored) / n,
            "precision@top_k": sum(r["precision@top_k"] for r in scored) / n,
        })
    summary.update({"unlabeled": skipped, "out_of_scope": out_of_scope,
                    "config": run_config(top_k=top_k, **search_kwargs)})
    return summary, scored


def log_record(summary, note):
    """The results.jsonl line for a run: same leading keys as always."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        **{k: v for k, v in summary.items() if k not in ("unlabeled", "out_of_scope")},
    }


def append_result(record, path=None):
    """Append, never overwrite. This file is your experiment log."""
    path = path or config.RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="what you changed this run")
    args = parser.parse_args()

    questions = json.loads(config.QUESTIONS_PATH.read_text())
    summary, scored = score_all(questions)

    if not scored:
        print("No labeled questions. Fill in gold_pages / gold_anchors.")
        return

    n = summary["n_questions"]
    print(f"\nscored {n} | unlabeled: {summary['unlabeled'] or 'none'} "
          f"| out-of-scope: {summary['out_of_scope']}\n")
    for key in METRICS:
        print(f"  {key:<12} {summary[key]:.3f}")

    print("\n  per question:")
    for r in scored:
        mark = "ok  " if r["recall@5"] else "MISS"
        rank = r["first_relevant_rank"] or "-"
        print(f"    {mark} Q{r['id']:<3} first hit at rank {rank:<4} p@5 {r['precision@5']:.1f}  {r['top_pages']}")

    path = append_result(log_record(summary, args.note))
    print(f"\nlogged -> {path}")


if __name__ == "__main__":
    main()

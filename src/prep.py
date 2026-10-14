"""Build the review sheet, and read your ticks back in.

  python3 -m src.prep              # write data/eval/review_sheet.md
  python3 -m src.prep --harvest    # read your ticks -> gold_pages

Workflow:
  1. Run without --harvest. Writes review_sheet.md with the top 25 search
     results for each question.
  2. Open it. Change [ ] to [x] on every passage that actually helps answer
     the question. Save.
  3. Run with --harvest. Ticked page numbers get written into questions.json
     as gold_pages.
  4. Add a gold_anchors phrase or two by hand, then run score.py.

The pool is 25 deep, not 5. If you only review the top 5 you can never find
out the right passage was at rank 12 - recall would look good only because
the question was never hard enough.
"""

import argparse
import json
import re

from src import config
from src.retrieve import search

# A review line looks like:  [ ]  3. p377 | prose | 0.812
# match (not search) anchors to line start, so an [x] inside quoted book text
# can never be mistaken for a tick. Any character in the box counts as
# ticked - only a space means no.
LINE_RE = re.compile(r"^\[(.)\]\s+\d+\.\s+p(\d+)\s")
HEADING_RE = re.compile(r"^## Q(\d+)")


def write_sheet():
    questions = json.loads(config.QUESTIONS_PATH.read_text())
    lines = []

    for q in questions:
        # No right page exists for these - they test refusal, not retrieval.
        if q["type"] == "out_of_scope":
            continue

        hits = search(q["question"], k=config.REVIEW_POOL)

        lines.append(f"## Q{q['id']} ({q['type']})")
        lines.append(f"**{q['question']}**\n")
        lines.append("Tick `[x]` if this passage helps answer the question.\n")

        for h in hits:
            c = h["chunk"]
            fig = f" | {len(c['figures'])} fig" if c["figures"] else ""
            lines.append(
                f"[ ]  {h['rank']}. p{c['page']} | {c['kind']}{fig} | {h['score']:.3f}"
            )
            lines.append(f"> {c['text'][:700]}\n")

        lines.append("---\n")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    config.REVIEW_SHEET_PATH.write_text("\n".join(lines))
    n = sum(1 for q in questions if q["type"] != "out_of_scope")
    print(f"wrote {config.REVIEW_SHEET_PATH}")
    print(f"  {n} questions x {config.REVIEW_POOL} results")
    print("Tick the good ones, then: python3 -m src.prep --harvest")


def parse_marks(text):
    """Marked sheet -> {question_id: {pages}}.

    Pure function - takes a string, returns a dict, touches no files. That is
    what makes it testable without running a search.
    """
    current, found = None, {}
    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            current = int(heading.group(1))
            continue
        mark = LINE_RE.match(line)
        if mark and current and mark.group(1).strip():
            found.setdefault(current, set()).add(int(mark.group(2)))
    return found


def harvest():
    """Merge ticked pages into questions.json.

    Pages only. gold_anchors stay hand-written, because a good anchor is a
    distinctive phrase you choose after reading the passage - and that
    reading is most of the value of this exercise.
    """
    if not config.REVIEW_SHEET_PATH.exists():
        print("No review sheet. Run: python3 -m src.prep")
        return

    found = parse_marks(config.REVIEW_SHEET_PATH.read_text())
    if not found:
        print("No ticks found. Put an x between the brackets: [x]")
        return

    questions = json.loads(config.QUESTIONS_PATH.read_text())
    by_id = {q["id"]: q for q in questions}

    for qid, pages in sorted(found.items()):
        q = by_id.get(qid)
        if q is None:
            print(f"  warning: Q{qid} is in the sheet but not in questions.json")
            continue
        # Union, not replace - a second labeling session adds to your work
        # instead of wiping it.
        merged = sorted(set(q.get("gold_pages", [])) | pages)
        q["gold_pages"] = merged
        q.setdefault("gold_anchors", [])
        print(f"Q{qid}: {merged}")

    config.QUESTIONS_PATH.write_text(json.dumps(questions, indent=2))
    print(f"\nupdated {config.QUESTIONS_PATH}")
    print("Now add gold_anchors by hand, then: python3 -m src.score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", action="store_true",
                        help="read ticks from the sheet instead of writing it")
    args = parser.parse_args()
    harvest() if args.harvest else write_sheet()